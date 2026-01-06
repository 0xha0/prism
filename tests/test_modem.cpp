/**
 * @file test_modem.cpp
 * @ingroup tests
 * @brief 调制解调算子单元测试
 *
 * 本文件涵盖了 QAM 和 PSK 两大类调制方式的映射与解映射算子测试，包括：
 * - **QAM 映射 (qamMap)**：验证不同阶数 (4, 16, 64) 下星座图的正确性
 * - **PSK 映射 (pskMap)**：验证相位偏移和幅度归一化特性
 * - **往返测试 (Roundtrip)**：验证 Map -> Demap 链路的符号恢复能力
 * - **数字混频器 (Mixer)**：验证复数旋转和频率生成的正确性
 *
 * 验证确保了 Complex 数据 (c, x) 布局与数值精度符合预期
 */

#include <cassert>
#include <cmath>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "prism/dsl/modem.h"
#include "prism/dsl/signal.h"
#include "prism/runtime/executor.h"
#include "prism/types.h"
#include "test_utils.h"

using prism::complex32_t;
using prism::complex64_t;
using prism::real32_t;
using prism::real64_t;
using namespace prism::dsl;
using namespace prism::runtime;
using namespace prism::tests;

/// @addtogroup tests
/// @{

// ============================================================================
// 测试数据基准 (Ground Truth)
// ============================================================================

/**
 * @brief QAM16 标准化星座图 I 分量预期值
 *
 * 对应 16 个符号索引 (0-15) 的 I 路坐标
 * 归一化后的电平：$-1, -1/3, 1/3, 1$
 */
template <typename T>
std::vector<T> getQam16ExpectedI() {
  return {static_cast<T>(-1.0),      static_cast<T>(-0.333333), static_cast<T>(0.333333),
          static_cast<T>(1.0),       static_cast<T>(-1.0),      static_cast<T>(-0.333333),
          static_cast<T>(0.333333),  static_cast<T>(1.0),       static_cast<T>(-1.0),
          static_cast<T>(-0.333333), static_cast<T>(0.333333),  static_cast<T>(1.0),
          static_cast<T>(-1.0),      static_cast<T>(-0.333333), static_cast<T>(0.333333),
          static_cast<T>(1.0)};
}

/**
 * @brief QAM16 标准化星座图 Q 分量预期值
 *
 * 对应 16 个符号索引 (0-15) 的 Q 路坐标
 */
template <typename T>
std::vector<T> getQam16ExpectedQ() {
  return {static_cast<T>(-1.0),      static_cast<T>(-1.0),      static_cast<T>(-1.0),
          static_cast<T>(-1.0),      static_cast<T>(-0.333333), static_cast<T>(-0.333333),
          static_cast<T>(-0.333333), static_cast<T>(-0.333333), static_cast<T>(0.333333),
          static_cast<T>(0.333333),  static_cast<T>(0.333333),  static_cast<T>(0.333333),
          static_cast<T>(1.0),       static_cast<T>(1.0),       static_cast<T>(1.0),
          static_cast<T>(1.0)};
}

template <typename T>
std::pair<T, T> qamPoint(int order, int symbol) {
  int const sqrtOrder = static_cast<int>(std::sqrt(order));
  int const iIdx = symbol % sqrtOrder;
  int const qIdx = symbol / sqrtOrder;
  T const norm = static_cast<T>(sqrtOrder - 1);
  T const iVal = (static_cast<T>(2 * iIdx) - norm) / norm;
  T const qVal = (static_cast<T>(2 * qIdx) - norm) / norm;
  return {iVal, qVal};
}

template <typename T>
std::pair<T, T> pskPoint(int order, int symbol) {
  T const twoPi = static_cast<T>(2.0 * M_PI_VAL);
  T const phaseOffset = static_cast<T>(M_PI_VAL / order);
  T const theta = (twoPi * static_cast<T>(symbol) / static_cast<T>(order)) + phaseOffset;
  return {static_cast<T>(std::cos(theta)), static_cast<T>(std::sin(theta))};
}

// ============================================================================
// QAM 映射测试组
// ============================================================================

/**
 * @brief [Test] QAM16 映射 I 分量验证
 */
template <typename T>
void testQam16MapIComponent() {
  using Traits = TestTraits<T>;
  std::string const name = "QAM16 I Component (" + TypeName<T>::get() + ")";
  int const nSyms = 16;

  // 输入是 Real 符号索引，使用一维 Buffer
  auto inputBuf = Traits::makeBuffer(nSyms);
  std::vector<T> data(nSyms);
  for (int i = 0; i < nSyms; ++i) {
    data[i] = static_cast<T>(i);
  }
  Traits::fillBuffer(inputBuf, data);

  Signal const x = Signal::input(nSyms, Traits::scalarType());
  Signal const y = modem::qamMap(x, 16);

  // 输出是 Complex 信号
  using ResultTraits =
      TestTraits<std::conditional_t<std::is_same_v<T, float>, complex32_t, complex64_t>>;

  auto result = Executor::run<typename ResultTraits::ElementType>(y, inputBuf);

  bool pass = true;
  auto expectedI = getQam16ExpectedI<T>();

  for (int i = 0; i < nSyms; ++i) {
    auto val = ResultTraits::getElement(result, i);
    if (!approxEqual(val.real(), expectedI[i])) pass = false;
  }

  TestPrinter::printTestResult(name, pass);
  assert(pass);
}

/**
 * @brief [Test] QAM16 映射 Q 分量验证
 */
template <typename T>
void testQam16MapQComponent() {
  using Traits = TestTraits<T>;
  // 输出是 Complex 信号
  using ResultTraits =
      TestTraits<std::conditional_t<std::is_same_v<T, float>, complex32_t, complex64_t>>;

  std::string const name = "QAM16 Q Component (" + TypeName<T>::get() + ")";
  int const nSyms = 16;

  // 输入是 Real 符号索引，使用一维 Buffer
  auto inputBuf = Traits::makeBuffer(nSyms);
  std::vector<T> data(nSyms);
  for (int i = 0; i < nSyms; ++i) {
    data[i] = static_cast<T>(i);
  }
  Traits::fillBuffer(inputBuf, data);

  Signal const x = Signal::input(nSyms, Traits::scalarType());
  Signal const y = modem::qamMap(x, 16);

  auto result = Executor::run<typename ResultTraits::ElementType>(y, inputBuf);

  bool pass = true;
  auto expectedQ = getQam16ExpectedQ<T>();

  for (int i = 0; i < nSyms; ++i) {
    auto val = ResultTraits::getElement(result, i);
    if (!approxEqual(val.imag(), expectedQ[i])) pass = false;
  }

  TestPrinter::printTestResult(name, pass);
  assert(pass);
}

/**
 * @brief [Test] QAM4 (QPSK) 映射验证
 */
template <typename T>
void testQam4Map() {
  using Traits = TestTraits<T>;
  using ResultTraits =
      TestTraits<std::conditional_t<std::is_same_v<T, float>, complex32_t, complex64_t>>;

  std::string const name = "QAM4 (QPSK) (" + TypeName<T>::get() + ")";
  int const nSyms = 4;

  // 输入是 Real 符号索引，使用一维 Buffer
  auto inputBuf = Traits::makeBuffer(nSyms);
  std::vector<T> data(nSyms);
  for (int i = 0; i < nSyms; ++i) {
    data[i] = static_cast<T>(i);
  }
  Traits::fillBuffer(inputBuf, data);

  Signal const x = Signal::input(nSyms, Traits::scalarType());
  Signal const y = modem::qamMap(x, 4);

  auto result = Executor::run<typename ResultTraits::ElementType>(y, inputBuf);

  // 验证四个顶点的坐标
  bool pass = true;
  auto v0 = ResultTraits::getElement(result, 0);
  auto v1 = ResultTraits::getElement(result, 1);

  if (!approxEqual(v0.real(), static_cast<T>(-1.0))) pass = false;
  if (!approxEqual(v0.imag(), static_cast<T>(-1.0))) pass = false;

  if (!approxEqual(v1.real(), static_cast<T>(1.0))) pass = false;
  if (!approxEqual(v1.imag(), static_cast<T>(-1.0))) pass = false;

  TestPrinter::printTestResult(name, pass);
  assert(pass);
}

// ============================================================================
// QAM 解映射测试组
// ============================================================================

/**
 * @brief [Test] QAM 映射-解映射往返测试 (Roundtrip)
 */
template <typename T>
void testQamRoundtrip(int order) {
  using Traits = TestTraits<T>;
  std::string const name =
      "QAM" + std::to_string(order) + " Roundtrip (" + TypeName<T>::get() + ")";
  int const nSyms = order;
  // Roundtrip output is 1D (Real Indices). Pipeline is 1D-like.
  // Should use 1D Buffer (Traits::makeBuffer).
  auto inputBuf = Traits::makeBuffer(nSyms);
  std::vector<T> data(nSyms);
  for (int i = 0; i < nSyms; ++i) {
    data[i] = static_cast<T>(i);
  }
  Traits::fillBuffer(inputBuf, data);

  Signal const x = Signal::input(nSyms, Traits::scalarType());
  Signal const mapped = modem::qamMap(x, order);
  Signal const demapped = modem::qamDemap(mapped, order);

  auto result = Executor::run<T>(demapped, inputBuf);

  bool pass = true;
  for (int i = 0; i < nSyms; ++i) {
    auto expected = static_cast<T>(i);
    auto val = Traits::getElement(result, i);
    if (!approxEqual(val, expected)) pass = false;
  }

  TestPrinter::printTestResult(name, pass);
  assert(pass);
}

/**
 * @brief [Test] QAM 解映射 (I/Q 分离输入)
 */
template <typename T>
void testQamDemapIq(int order) {
  using Traits = TestTraits<T>;
  std::string const name = "QAM" + std::to_string(order) + " Demap IQ (" + TypeName<T>::get() + ")";

  int const nSyms = order;
  std::vector<T> iData(nSyms);
  std::vector<T> qData(nSyms);
  for (int s = 0; s < nSyms; ++s) {
    auto const [iVal, qVal] = qamPoint<T>(order, s);
    iData[s] = iVal;
    qData[s] = qVal;
  }

  auto iBuf = Traits::makeBuffer(nSyms);
  auto qBuf = Traits::makeBuffer(nSyms);
  Traits::fillBuffer(iBuf, iData);
  Traits::fillBuffer(qBuf, qData);

  Signal const iSig = Signal::input(nSyms, Traits::scalarType());
  Signal const qSig = Signal::input(nSyms, Traits::scalarType());
  Signal const demapped = modem::qamDemap(iSig, qSig, order);

  auto result = Executor::run<T>(demapped, {iBuf, qBuf});

  bool pass = true;
  for (int i = 0; i < nSyms; ++i) {
    auto val = Traits::getElement(result, i);
    if (!approxEqual(val, static_cast<T>(i))) pass = false;
  }

  TestPrinter::printTestResult(name, pass);
  assert(pass);
}

// ============================================================================
// PSK 映射测试组
// ============================================================================

/**
 * @brief [Test] BPSK 映射验证
 */
template <typename T>
void testBpskMap() {
  using Traits = TestTraits<T>;
  using ResultTraits =
      TestTraits<std::conditional_t<std::is_same_v<T, float>, complex32_t, complex64_t>>;

  std::string const name = "BPSK Map (" + TypeName<T>::get() + ")";
  int const nSyms = 2;

  // 输入是 Real 符号索引，使用一维 Buffer
  auto inputBuf = Traits::makeBuffer(nSyms);
  std::vector<T> const data = {static_cast<T>(0.0), static_cast<T>(1.0)};
  Traits::fillBuffer(inputBuf, data);

  Signal const x = Signal::input(nSyms, Traits::scalarType());
  Signal const y = modem::pskMap(x, 2);

  auto result = Executor::run<typename ResultTraits::ElementType>(y, inputBuf);

  bool pass = true;
  auto v0 = ResultTraits::getElement(result, 0);
  auto v1 = ResultTraits::getElement(result, 1);

  if (!approxEqual(v0.real(), static_cast<T>(0.0), 1e-3)) pass = false;
  if (!approxEqual(v0.imag(), static_cast<T>(1.0), 1e-3)) pass = false;
  if (!approxEqual(v1.real(), static_cast<T>(0.0), 1e-3)) pass = false;
  if (!approxEqual(v1.imag(), static_cast<T>(-1.0), 1e-3)) pass = false;

  TestPrinter::printTestResult(name, pass);
  assert(pass);
}

/**
 * @brief [Test] QPSK (PSK4) 映射单位圆验证
 */
template <typename T>
void testQpskMap() {
  using Traits = TestTraits<T>;
  using ResultTraits =
      TestTraits<std::conditional_t<std::is_same_v<T, float>, complex32_t, complex64_t>>;

  std::string const name = "QPSK Map (" + TypeName<T>::get() + ")";
  int const nSyms = 4;

  // 输入是 Real 符号索引，使用一维 Buffer
  auto inputBuf = Traits::makeBuffer(nSyms);
  std::vector<T> data(nSyms);
  for (int i = 0; i < nSyms; ++i) {
    data[i] = static_cast<T>(i);
  }
  Traits::fillBuffer(inputBuf, data);

  Signal const x = Signal::input(nSyms, Traits::scalarType());
  Signal const y = modem::pskMap(x, 4);

  auto result = Executor::run<typename ResultTraits::ElementType>(y, inputBuf);

  bool pass = true;
  for (int i = 0; i < nSyms; ++i) {
    auto val = ResultTraits::getElement(result, i);
    T const mag = std::abs(val);
    if (!approxEqual(mag, static_cast<T>(1.0), 1e-3)) pass = false;
  }

  TestPrinter::printTestResult(name, pass);
  assert(pass);
}

// ============================================================================
// PSK 往返测试组
// ============================================================================

/**
 * @brief [Test] PSK 映射-解映射往返测试 (Roundtrip)
 */
template <typename T>
void testPskRoundtrip(int order) {
  using Traits = TestTraits<T>;
  std::string phaseName;
  if (order == 2) {
    phaseName = "BPSK";
  } else if (order == 4) {
    phaseName = "QPSK";
  } else {
    phaseName = std::to_string(order) + "PSK";
  }

  std::string const name = phaseName + " Roundtrip (" + TypeName<T>::get() + ")";
  int const nSyms = order;
  // Output Real -> 1D Input
  auto inputBuf = Traits::makeBuffer(nSyms);
  std::vector<T> data(nSyms);
  for (int i = 0; i < nSyms; ++i) {
    data[i] = static_cast<T>(i);
  }
  Traits::fillBuffer(inputBuf, data);

  Signal const x = Signal::input(nSyms, Traits::scalarType());
  Signal const mapped = modem::pskMap(x, order);
  Signal const demapped = modem::pskDemap(mapped, order);

  auto result = Executor::run<T>(demapped, inputBuf);

  bool pass = true;
  for (int i = 0; i < nSyms; ++i) {
    auto expected = static_cast<T>(i);
    auto val = Traits::getElement(result, i);
    if (!approxEqual(val, expected)) pass = false;
  }

  TestPrinter::printTestResult(name, pass);
  assert(pass);
}

/**
 * @brief [Test] PSK 解映射 (I/Q 分离输入)
 */
template <typename T>
void testPskDemapIq(int order) {
  using Traits = TestTraits<T>;
  std::string const name = std::to_string(order) + "PSK Demap IQ (" + TypeName<T>::get() + ")";

  int const nSyms = order;
  std::vector<T> iData(nSyms);
  std::vector<T> qData(nSyms);
  for (int s = 0; s < nSyms; ++s) {
    auto const [iVal, qVal] = pskPoint<T>(order, s);
    iData[s] = iVal;
    qData[s] = qVal;
  }

  auto iBuf = Traits::makeBuffer(nSyms);
  auto qBuf = Traits::makeBuffer(nSyms);
  Traits::fillBuffer(iBuf, iData);
  Traits::fillBuffer(qBuf, qData);

  Signal const iSig = Signal::input(nSyms, Traits::scalarType());
  Signal const qSig = Signal::input(nSyms, Traits::scalarType());
  Signal const demapped = modem::pskDemap(iSig, qSig, order);

  auto result = Executor::run<T>(demapped, {iBuf, qBuf});

  bool pass = true;
  for (int i = 0; i < nSyms; ++i) {
    auto val = Traits::getElement(result, i);
    if (!approxEqual(val, static_cast<T>(i))) pass = false;
  }

  TestPrinter::printTestResult(name, pass);
  assert(pass);
}

// ============================================================================
// 数字混频器测试组
// ============================================================================

/**
 * @brief [Test] 混频器功能测试 (Mixer)
 */
template <typename ComplexT>
void testMixer() {
  using Traits = TestTraits<ComplexT>;
  using RealT = typename ComplexT::value_type;
  std::string const name = "Mixer (" + TypeName<ComplexT>::get() + ")";

  bool pass = true;

  // Input: Constant 1 (Real, so 1+0j)
  int const len = 4;

  std::vector<ComplexT> expected = {{static_cast<RealT>(1.0), static_cast<RealT>(0.0)},
                                    {static_cast<RealT>(0.0), static_cast<RealT>(1.0)},
                                    {static_cast<RealT>(-1.0), static_cast<RealT>(0.0)},
                                    {static_cast<RealT>(0.0), static_cast<RealT>(-1.0)}};

  auto inputBuf = Traits::makeBuffer(len);
  // Fill complex buffer with 1.0 (Real) -> (1,0)
  std::vector<ComplexT> data(len);
  for (int i = 0; i < len; ++i) data[i] = {1.0, 0.0};
  Traits::fillBuffer(inputBuf, data);

  auto s = Signal::input(len, Traits::scalarType());
  auto mixed = modem::mixer(s, 1.0, 4.0);

  auto out = Executor::run<ComplexT>(mixed, inputBuf);

  for (int i = 0; i < len; ++i) {
    auto val = Traits::getElement(out, i);
    double const eps = (sizeof(RealT) == 8) ? 1e-10 : 1e-5;
    if (std::abs(val - expected[i]) > eps) {
      pass = false;
      TestPrinter::printTestResult(name + " mismatch", false, "idx=" + std::to_string(i));
    }
  }

  TestPrinter::printTestResult(name, pass);
  assert(pass);
}

// ============================================================================
// Main Entry
// ============================================================================

int main() {
  TestPrinter::printSuiteHeader("Modem Tests");

  TestPrinter::printSection("QAM Map");
  runTest([] { testQam16MapIComponent<real32_t>(); }, "QAM16 I Component (Real32)");
  runTest([] { testQam16MapIComponent<real64_t>(); }, "QAM16 I Component (Real64)");
  runTest([] { testQam16MapQComponent<real32_t>(); }, "QAM16 Q Component (Real32)");
  runTest([] { testQam16MapQComponent<real64_t>(); }, "QAM16 Q Component (Real64)");
  runTest([] { testQam4Map<real32_t>(); }, "QAM4 Map (Real32)");
  runTest([] { testQam4Map<real64_t>(); }, "QAM4 Map (Real64)");
  runTest([] { testQamRoundtrip<real32_t>(4); }, "QAM4 Roundtrip (Real32)");
  runTest([] { testQamRoundtrip<real64_t>(4); }, "QAM4 Roundtrip (Real64)");
  runTest([] { testQamRoundtrip<real32_t>(16); }, "QAM16 Roundtrip (Real32)");
  runTest([] { testQamRoundtrip<real64_t>(16); }, "QAM16 Roundtrip (Real64)");
  runTest([] { testQamRoundtrip<real32_t>(64); }, "QAM64 Roundtrip (Real32)");
  runTest([] { testQamRoundtrip<real64_t>(64); }, "QAM64 Roundtrip (Real64)");

  TestPrinter::printSection("QAM Demap (I/Q)");
  runTest([] { testQamDemapIq<real32_t>(4); }, "QAM4 Demap IQ (Real32)");
  runTest([] { testQamDemapIq<real64_t>(4); }, "QAM4 Demap IQ (Real64)");
  runTest([] { testQamDemapIq<real32_t>(16); }, "QAM16 Demap IQ (Real32)");
  runTest([] { testQamDemapIq<real64_t>(16); }, "QAM16 Demap IQ (Real64)");

  TestPrinter::printSection("PSK Map");
  runTest([] { testBpskMap<real32_t>(); }, "BPSK Map (Real32)");
  runTest([] { testBpskMap<real64_t>(); }, "BPSK Map (Real64)");
  runTest([] { testQpskMap<real32_t>(); }, "QPSK Map (Real32)");
  runTest([] { testQpskMap<real64_t>(); }, "QPSK Map (Real64)");
  runTest([] { testPskRoundtrip<real32_t>(2); }, "BPSK Roundtrip (Real32)");
  runTest([] { testPskRoundtrip<real64_t>(2); }, "BPSK Roundtrip (Real64)");
  runTest([] { testPskRoundtrip<real32_t>(4); }, "QPSK Roundtrip (Real32)");
  runTest([] { testPskRoundtrip<real64_t>(4); }, "QPSK Roundtrip (Real64)");
  runTest([] { testPskRoundtrip<real32_t>(8); }, "8PSK Roundtrip (Real32)");
  runTest([] { testPskRoundtrip<real64_t>(8); }, "8PSK Roundtrip (Real64)");

  TestPrinter::printSection("PSK Demap (I/Q)");
  runTest([] { testPskDemapIq<real32_t>(4); }, "QPSK Demap IQ (Real32)");
  runTest([] { testPskDemapIq<real64_t>(4); }, "QPSK Demap IQ (Real64)");
  runTest([] { testPskDemapIq<real32_t>(8); }, "8PSK Demap IQ (Real32)");
  runTest([] { testPskDemapIq<real64_t>(8); }, "8PSK Demap IQ (Real64)");

  TestPrinter::printSection("Mixer");
  runTest([] { testMixer<complex32_t>(); }, "Mixer (Complex32)");
  runTest([] { testMixer<complex64_t>(); }, "Mixer (Complex64)");

  TestPrinter::printSummary();
  return 0;
}
