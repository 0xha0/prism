/**
 * @file test_modem.cpp
 * @ingroup tests
 * @brief 调制解调算子单元测试
 *
 * 覆盖 QAM 映射/解映射、PSK 映射/解映射、混频器等算子，模板化验证不同精度。
 */

#include <Halide.h>

#include <cassert>
#include <cmath>
#include <string>
#include <type_traits>
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
// 测试数据
// ============================================================================

/** @brief QAM16 期望 I 分量（归一化后） */
template <typename T>
std::vector<T> getQam16ExpectedI() {
  return {static_cast<T>(-1.0),     static_cast<T>(-0.333333),
          static_cast<T>(0.333333), static_cast<T>(1.0),
          static_cast<T>(-1.0),     static_cast<T>(-0.333333),
          static_cast<T>(0.333333), static_cast<T>(1.0),
          static_cast<T>(-1.0),     static_cast<T>(-0.333333),
          static_cast<T>(0.333333), static_cast<T>(1.0),
          static_cast<T>(-1.0),     static_cast<T>(-0.333333),
          static_cast<T>(0.333333), static_cast<T>(1.0)};
}

template <typename T>
std::vector<T> getQam16ExpectedQ() {
  return {static_cast<T>(-1.0),      static_cast<T>(-1.0),
          static_cast<T>(-1.0),      static_cast<T>(-1.0),
          static_cast<T>(-0.333333), static_cast<T>(-0.333333),
          static_cast<T>(-0.333333), static_cast<T>(-0.333333),
          static_cast<T>(0.333333),  static_cast<T>(0.333333),
          static_cast<T>(0.333333),  static_cast<T>(0.333333),
          static_cast<T>(1.0),       static_cast<T>(1.0),
          static_cast<T>(1.0),       static_cast<T>(1.0)};
}

// ============================================================================
// QAM 映射测试
// ============================================================================

/** @brief QAM16 映射：验证 I 分量 */
template <typename T>
void testQam16MapIComponent() {
  std::string const name = "QAM16 I Component (" + TypeName<T>::get() + ")";
  int const nSyms = 16;
  Halide::Buffer<T> inputBuf(nSyms);
  for (int i = 0; i < nSyms; ++i) {
    inputBuf(i) = static_cast<T>(i);
  }

  Signal const x = Signal::input(nSyms);
  Signal const y = modem::qamMap(x, 16);

  auto result = Executor::run<T>(y, inputBuf);

  bool pass = true;
  // Verify output length (2N for I/Q interleaved)
  if (result.width() != nSyms * 2) pass = false;

  auto expectedI = getQam16ExpectedI<T>();

  // Verify I component
  for (int i = 0; i < nSyms; ++i) {
    T const gotI = result(2 * i);
    if (!approxEqual(gotI, expectedI[i])) pass = false;
  }

  TestPrinter::printTestResult(name, pass);
  assert(pass);
}

/** @brief QAM16 映射：验证 Q 分量 */
template <typename T>
void testQam16MapQComponent() {
  std::string const name = "QAM16 Q Component (" + TypeName<T>::get() + ")";
  int const nSyms = 16;
  Halide::Buffer<T> inputBuf(nSyms);
  for (int i = 0; i < nSyms; ++i) {
    inputBuf(i) = static_cast<T>(i);
  }

  Signal const x = Signal::input(nSyms);
  Signal const y = modem::qamMap(x, 16);

  auto result = Executor::run<T>(y, inputBuf);

  bool pass = true;
  auto expectedQ = getQam16ExpectedQ<T>();

  // Verify Q component
  for (int i = 0; i < nSyms; ++i) {
    T const gotQ = result((2 * i) + 1);
    // Use slightly relaxed epsilon for 64-bit strictness if needed, but 1e-4 is
    // fine
    if (!approxEqual(gotQ, expectedQ[i])) pass = false;
  }

  TestPrinter::printTestResult(name, pass);
  assert(pass);
}

/** @brief QAM4/QPSK 映射 */
template <typename T>
void testQam4Map() {
  std::string const name = "QAM4 (QPSK) (" + TypeName<T>::get() + ")";
  int const nSyms = 4;
  Halide::Buffer<T> inputBuf(nSyms);
  for (int i = 0; i < nSyms; ++i) {
    inputBuf(i) = static_cast<T>(i);
  }

  Signal const x = Signal::input(nSyms);
  Signal const y = modem::qamMap(x, 4);

  auto result = Executor::run<T>(y, inputBuf);

  // QAM4: sym 0 -> I=-1, Q=-1
  //       sym 1 -> I=1, Q=-1
  //       sym 2 -> I=-1, Q=1
  //       sym 3 -> I=1, Q=1
  bool pass = true;
  if (result.width() != 8) pass = false;
  if (!approxEqual(result(0), static_cast<T>(-1.0))) pass = false;  // sym0 I
  if (!approxEqual(result(1), static_cast<T>(-1.0))) pass = false;  // sym0 Q
  if (!approxEqual(result(2), static_cast<T>(1.0))) pass = false;   // sym1 I
  if (!approxEqual(result(3), static_cast<T>(-1.0))) pass = false;  // sym1 Q

  TestPrinter::printTestResult(name, pass);
  assert(pass);
}

// ============================================================================
// QAM 解映射测试
// ============================================================================

/** @brief QAM 映射-解映射往返校验 */
template <typename T>
void testQamRoundtrip(int order) {
  std::string const name =
      "QAM" + std::to_string(order) + " Roundtrip (" + TypeName<T>::get() + ")";
  int const nSyms = order;
  Halide::Buffer<T> inputBuf(nSyms);
  for (int i = 0; i < nSyms; ++i) {
    inputBuf(i) = static_cast<T>(i);
  }

  Signal const x = Signal::input(nSyms);
  Signal const mapped = modem::qamMap(x, order);
  Signal const demapped = modem::qamDemap(mapped, order);

  auto result = Executor::run<T>(demapped, inputBuf);

  bool pass = true;
  for (int i = 0; i < nSyms; ++i) {
    auto const expected = static_cast<T>(i);
    // Demapping results in indices, which should be integers
    if (!approxEqual(result(i), expected)) pass = false;
  }

  TestPrinter::printTestResult(name, pass);
  assert(pass);
}

// ============================================================================
// PSK 映射测试
// ============================================================================

/** @brief BPSK 映射：验证相位与 I/Q 输出 */
template <typename T>
void testBpskMap() {
  std::string const name = "BPSK Map (" + TypeName<T>::get() + ")";
  int const nSyms = 2;
  Halide::Buffer<T> inputBuf(nSyms);
  inputBuf(0) = static_cast<T>(0.0);
  inputBuf(1) = static_cast<T>(1.0);

  Signal const x = Signal::input(nSyms);
  Signal const y = modem::pskMap(x, 2);

  auto result = Executor::run<T>(y, inputBuf);

  // BPSK: θ = 2π*k/2 + π/2 = kπ + π/2
  // k=0: θ=π/2 -> I=0, Q=1
  // k=1: θ=3π/2 -> I=0, Q=-1
  bool pass = true;
  if (result.width() != 4) pass = false;
  if (!approxEqual(result(0), static_cast<T>(0.0), 1e-3)) {
    pass = false;  // sym0 I
  }
  if (!approxEqual(result(1), static_cast<T>(1.0), 1e-3)) {
    pass = false;  // sym0 Q
  }
  if (!approxEqual(result(2), static_cast<T>(0.0), 1e-3)) {
    pass = false;  // sym1 I
  }
  if (!approxEqual(result(3), static_cast<T>(-1.0), 1e-3)) {
    pass = false;  // sym1 Q
  }

  TestPrinter::printTestResult(name, pass);
  assert(pass);
}

/** @brief QPSK 映射：验证幅度在单位圆上 */
template <typename T>
void testQpskMap() {
  std::string const name = "QPSK Map (" + TypeName<T>::get() + ")";
  int const nSyms = 4;
  Halide::Buffer<T> inputBuf(nSyms);
  for (int i = 0; i < nSyms; ++i) {
    inputBuf(i) = static_cast<T>(i);
  }

  Signal const x = Signal::input(nSyms);
  Signal const y = modem::pskMap(x, 4);

  auto result = Executor::run<T>(y, inputBuf);

  // QPSK: θ = 2π*k/4 + π/4 = kπ/2 + π/4
  // All points on unit circle
  bool pass = true;
  if (result.width() != 8) pass = false;
  for (int i = 0; i < nSyms; ++i) {
    T const sigI = result(2 * i);
    T const sigQ = result((2 * i) + 1);
    T const mag = std::sqrt((sigI * sigI) + (sigQ * sigQ));
    if (!approxEqual(mag, static_cast<T>(1.0), 1e-3)) pass = false;
  }

  TestPrinter::printTestResult(name, pass);
  assert(pass);
}

// ============================================================================
// PSK 往返测试
// ============================================================================

/** @brief PSK 映射/解映射往返校验 */
template <typename T>
void testPskRoundtrip(int order) {
  std::string phaseName;
  if (order == 2) {
    phaseName = "BPSK";
  } else if (order == 4) {
    phaseName = "QPSK";
  } else {
    phaseName = std::to_string(order) + "PSK";
  }
  std::string const name =
      phaseName + " Roundtrip (" + TypeName<T>::get() + ")";
  int const nSyms = order;
  Halide::Buffer<T> inputBuf(nSyms);
  for (int i = 0; i < nSyms; ++i) {
    inputBuf(i) = static_cast<T>(i);
  }

  Signal const x = Signal::input(nSyms);
  Signal const mapped = modem::pskMap(x, order);
  Signal const demapped = modem::pskDemap(mapped, order);

  auto result = Executor::run<T>(demapped, inputBuf);

  bool pass = true;
  for (int i = 0; i < nSyms; ++i) {
    auto const expected = static_cast<T>(i);
    if (!approxEqual(result(i), expected)) pass = false;
  }

  TestPrinter::printTestResult(name, pass);
  assert(pass);
}

// ============================================================================
// Mixer 测试
// ============================================================================

/** @brief 混频器：验证旋转相位序列 */
template <typename ComplexT>
void testMixer() {
  using RealT = typename ComplexT::value_type;
  std::string const name = "Mixer (" + TypeName<ComplexT>::get() + ")";

  bool pass = true;

  // Input: Constant 1 (Real)
  // Mixer freq=1, fs=4 -> exp(j*pi/2*n) -> 1, j, -1, -j
  int const len = 4;

  std::vector<ComplexT> expected = {
      {static_cast<RealT>(1.0), static_cast<RealT>(0.0)},
      {static_cast<RealT>(0.0), static_cast<RealT>(1.0)},
      {static_cast<RealT>(-1.0), static_cast<RealT>(0.0)},
      {static_cast<RealT>(0.0), static_cast<RealT>(-1.0)}};

  {
    auto s = Signal::constant(1.0, len);
    auto mixed = modem::mixer(s, 1.0, 4.0);

    auto out = Executor::run<ComplexT>(mixed);

    for (int i = 0; i < len; ++i) {
      ComplexT val;
      if constexpr (std::is_same_v<RealT, double>) {
        val = getComplex64(out, i);
      } else {
        val = getComplex32(out, i);
      }
      double const eps = (sizeof(RealT) == 8) ? 1e-10 : 1e-5;
      if (std::abs(val - expected[i]) > eps) {
        pass = false;
        TestPrinter::printTestResult(name + " mismatch", false,
                                     "idx=" + std::to_string(i));
      }
    }
  }

  TestPrinter::printTestResult(name, pass);
  assert(pass);
}

// ============================================================================
// Main
// ============================================================================

int main() {
  TestPrinter::printSuiteHeader("Modem Tests (Refactored)");

  TestPrinter::printSection("QAM Map (Real32)");
  runTest([] { testQam16MapIComponent<real32_t>(); },
          "QAM16 I Component (Real32)");
  runTest([] { testQam16MapQComponent<real32_t>(); },
          "QAM16 Q Component (Real32)");
  runTest([] { testQam4Map<real32_t>(); }, "QAM4 Map (Real32)");

  TestPrinter::printSection("QAM Map (Real64)");
  runTest([] { testQam16MapIComponent<real64_t>(); },
          "QAM16 I Component (Real64)");
  runTest([] { testQam16MapQComponent<real64_t>(); },
          "QAM16 Q Component (Real64)");
  runTest([] { testQam4Map<real64_t>(); }, "QAM4 Map (Real64)");

  TestPrinter::printSection("QAM Roundtrip (Real32)");
  runTest([] { testQamRoundtrip<real32_t>(16); }, "QAM16 Roundtrip (Real32)");
  runTest([] { testQamRoundtrip<real32_t>(4); }, "QAM4 Roundtrip (Real32)");
  runTest([] { testQamRoundtrip<real32_t>(64); }, "QAM64 Roundtrip (Real32)");

  TestPrinter::printSection("QAM Roundtrip (Real64)");
  runTest([] { testQamRoundtrip<real64_t>(16); }, "QAM16 Roundtrip (Real64)");
  runTest([] { testQamRoundtrip<real64_t>(4); }, "QAM4 Roundtrip (Real64)");
  runTest([] { testQamRoundtrip<real64_t>(64); }, "QAM64 Roundtrip (Real64)");

  TestPrinter::printSection("PSK Map (Real32)");
  runTest([] { testBpskMap<real32_t>(); }, "BPSK Map (Real32)");
  runTest([] { testQpskMap<real32_t>(); }, "QPSK Map (Real32)");

  TestPrinter::printSection("PSK Map (Real64)");
  runTest([] { testBpskMap<real64_t>(); }, "BPSK Map (Real64)");
  runTest([] { testQpskMap<real64_t>(); }, "QPSK Map (Real64)");

  TestPrinter::printSection("PSK Roundtrip (Real32)");
  runTest([] { testPskRoundtrip<real32_t>(2); }, "BPSK Roundtrip (Real32)");
  runTest([] { testPskRoundtrip<real32_t>(4); }, "QPSK Roundtrip (Real32)");
  runTest([] { testPskRoundtrip<real32_t>(8); }, "8PSK Roundtrip (Real32)");

  TestPrinter::printSection("PSK Roundtrip (Real64)");
  runTest([] { testPskRoundtrip<real64_t>(2); }, "BPSK Roundtrip (Real64)");
  runTest([] { testPskRoundtrip<real64_t>(4); }, "QPSK Roundtrip (Real64)");
  runTest([] { testPskRoundtrip<real64_t>(8); }, "8PSK Roundtrip (Real64)");

  TestPrinter::printSection("Mixer (Complex32)");
  runTest([] { testMixer<complex32_t>(); }, "Mixer (Complex32)");

  TestPrinter::printSection("Mixer (Complex64)");
  runTest([] { testMixer<complex64_t>(); }, "Mixer (Complex64)");

  TestPrinter::printSummary();
  return 0;
}

/// @}
