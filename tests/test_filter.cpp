/**
 * @file test_filter.cpp
 * @ingroup tests
 * @brief 滤波算子单元测试
 *
 * 本文件覆盖了 PRISM DSL 提供的三种主要滤波器算子的测试：
 * - **FIR (Finite Impulse Response)**：涵盖实数/复数卷积、冲激响应验证
 * - **Median Filter (中值滤波器)**：验证非线性滤波效果，用于去除椒盐噪声
 * - **Moving Average (滑动平均滤波器)**：验证平滑效果
 *
 * 测试涵盖了不同的窗口长度、数据类型 (Real/Complex) 和精度 (32/64位)
 */

#include <Halide.h>

#include <cassert>
#include <cmath>
#include <complex>
#include <cstddef>
#include <string>
#include <type_traits>
#include <vector>

#include "prism/dsl/filter.h"
#include "prism/dsl/signal.h"
#include "prism/runtime/executor.h"
#include "prism/types.h"
#include "test_utils.h"

using prism::complex32_t;
using prism::complex64_t;
using prism::IS_COMPLEX_V;
using prism::real32_t;
using prism::real64_t;
using namespace prism::dsl;
using namespace prism::runtime;
using namespace prism::tests;

/// @addtogroup tests
/// @{

// ============================================================================
// 辅助数据生成 (Data Generation)
// ============================================================================

/**
 * @brief 获取 RRC (Root Raised Cosine) 滤波器 Taps
 *
 * 一个典型的通信系统成型滤波器系数，用于 FIR 测试
 * 包含负值和正值，能较好地检验卷积实现的正确性
 */
template <typename T>
std::vector<typename TestTraits<T>::BufferElemType> getRRCTaps() {
  using RealT = typename TestTraits<T>::BufferElemType;
  return {
      static_cast<RealT>(0.014675),  static_cast<RealT>(-0.005671), static_cast<RealT>(-0.034726),
      static_cast<RealT>(-0.048463), static_cast<RealT>(-0.021758), static_cast<RealT>(0.053149),
      static_cast<RealT>(0.156148),  static_cast<RealT>(0.245903),  static_cast<RealT>(0.281488),
      static_cast<RealT>(0.245903),  static_cast<RealT>(0.156148),  static_cast<RealT>(0.053149),
      static_cast<RealT>(-0.021758), static_cast<RealT>(-0.048463), static_cast<RealT>(-0.034726),
      static_cast<RealT>(-0.005671), static_cast<RealT>(0.014675)};
}

/**
 * @brief 中值滤波测试输入信号
 *
 * 包含孤立的极大值 (8.0)，用于验证中值滤波的椒盐噪声去除能力
 */
template <typename T>
std::vector<T> getMedianIn() {
  return {static_cast<T>(1.0), static_cast<T>(5.0), static_cast<T>(2.0),
          static_cast<T>(8.0), static_cast<T>(3.0), static_cast<T>(4.0)};
}

/**
 * @brief 中值滤波 (Window=3) 期望输出
 */
template <typename T>
std::vector<T> getMedianOutWindow3() {
  return {static_cast<T>(1.0), static_cast<T>(2.0), static_cast<T>(5.0),
          static_cast<T>(3.0), static_cast<T>(4.0), static_cast<T>(3.0)};
}

/**
 * @brief 中值滤波 (Window=5) 期望输出
 */
template <typename T>
std::vector<T> getMedianOutWindow5() {
  return {static_cast<T>(1.0), static_cast<T>(2.0), static_cast<T>(3.0),
          static_cast<T>(4.0), static_cast<T>(3.0), static_cast<T>(3.0)};
}

// ============================================================================
// FIR 期望值辅助 (Reference Helper)
// ============================================================================

template <typename T>
bool checkBufferMatches(const Halide::Buffer<typename TestTraits<T>::BufferElemType>& result,
                        const std::vector<T>& expected, double tol = 1e-4) {
  for (size_t i = 0; i < expected.size(); ++i) {
    T const actual = TestTraits<T>::getElement(result, static_cast<int>(i));
    auto const diff = std::abs(actual - expected[i]);
    if (diff > static_cast<decltype(diff)>(tol)) {
      return false;
    }
  }
  return true;
}

template <typename RealT>
std::vector<std::complex<RealT>> toComplexVector(const std::vector<RealT>& data) {
  std::vector<std::complex<RealT>> out;
  out.reserve(data.size());
  for (auto v : data) {
    out.emplace_back(v, static_cast<RealT>(0));
  }
  return out;
}

template <typename RealT>
std::vector<std::complex<RealT>> toComplexVector(const std::vector<std::complex<RealT>>& data) {
  return data;
}

/** @brief FIR 朴素参考实现 $O(N \cdot M)$ */
template <typename RealT>
std::vector<std::complex<RealT>> naiveFir(const std::vector<std::complex<RealT>>& input,
                                          const std::vector<std::complex<RealT>>& taps) {
  int const len = static_cast<int>(input.size());
  int const tapsLen = static_cast<int>(taps.size());
  std::vector<std::complex<RealT>> out(len, std::complex<RealT>{});
  for (int x = 0; x < len; ++x) {
    std::complex<RealT> acc = {};
    for (int k = 0; k < tapsLen; ++k) {
      int const idx = x - k;
      if (idx >= 0 && idx < len) {
        acc += input[idx] * taps[k];
      }
    }
    out[x] = acc;
  }
  return out;
}

template <typename InT, typename TapT>
std::string firMixedName() {
  return "FIR Mixed (" + TypeName<InT>::get() + ", " + TypeName<TapT>::get() + ")";
}

template <typename RealT>
struct FirMixedData {
  using ComplexT = std::complex<RealT>;
  std::vector<RealT> realIn;
  std::vector<ComplexT> complexIn;
  std::vector<RealT> realTaps;
  std::vector<ComplexT> complexTaps;
};

template <typename RealT>
FirMixedData<RealT> makeFirMixedData() {
  using ComplexT = std::complex<RealT>;
  FirMixedData<RealT> data;
  data.realIn = {static_cast<RealT>(1.0), static_cast<RealT>(2.0), static_cast<RealT>(-1.0),
                 static_cast<RealT>(0.5)};
  data.complexIn = {ComplexT{static_cast<RealT>(1.0), static_cast<RealT>(1.0)},
                    ComplexT{static_cast<RealT>(2.0), static_cast<RealT>(-0.5)},
                    ComplexT{static_cast<RealT>(-1.0), static_cast<RealT>(0.25)},
                    ComplexT{static_cast<RealT>(0.5), static_cast<RealT>(-1.0)}};
  data.realTaps = {static_cast<RealT>(0.25), static_cast<RealT>(0.5), static_cast<RealT>(0.25)};
  data.complexTaps = {ComplexT{static_cast<RealT>(0.5), static_cast<RealT>(0.25)},
                      ComplexT{static_cast<RealT>(-0.25), static_cast<RealT>(0.5)},
                      ComplexT{static_cast<RealT>(0.25), static_cast<RealT>(-0.25)}};
  return data;
}

template <typename T, typename RealT>
decltype(auto) selectFirInput(const FirMixedData<RealT>& data) {
  if constexpr (TestTraits<T>::IS_COMPLEX) {
    return (data.complexIn);
  } else {
    return (data.realIn);
  }
}

template <typename T, typename RealT>
decltype(auto) selectFirTaps(const FirMixedData<RealT>& data) {
  if constexpr (TestTraits<T>::IS_COMPLEX) {
    return (data.complexTaps);
  } else {
    return (data.realTaps);
  }
}

template <typename RealT>
std::string firSymmetryName(const std::string& label) {
  return "FIR Symmetry " + label + " (" + TypeName<RealT>::get() + ")";
}

template <typename RealT>
std::vector<std::complex<RealT>> getFirSymmetryInput() {
  using ComplexT = std::complex<RealT>;
  return {ComplexT{static_cast<RealT>(1.0), static_cast<RealT>(0.5)},
          ComplexT{static_cast<RealT>(-0.5), static_cast<RealT>(1.5)},
          ComplexT{static_cast<RealT>(2.0), static_cast<RealT>(-1.0)},
          ComplexT{static_cast<RealT>(0.0), static_cast<RealT>(0.25)},
          ComplexT{static_cast<RealT>(-1.5), static_cast<RealT>(-0.5)},
          ComplexT{static_cast<RealT>(0.75), static_cast<RealT>(1.0)}};
}

// ============================================================================
// FIR 滤波器测试组
// ============================================================================

/**
 * @brief [Test] FIR 冲激响应测试 (Impulse Response)
 */
template <typename T>
void testFirImpulseResponse() {
  using Traits = TestTraits<T>;
  using RealT = typename Traits::BufferElemType;
  std::string const name = "FIR Impulse (" + TypeName<T>::get() + ")";
  int const totalLen = 100;
  std::vector<T> impulseData(totalLen, static_cast<T>(0.0));
  impulseData[totalLen / 2] = static_cast<T>(1.0);

  auto inputBuf = Traits::makeBuffer(totalLen);
  Traits::fillBuffer(inputBuf, impulseData);

  // Taps always Real
  std::vector<RealT> rrcTaps = getRRCTaps<T>();

  auto x = Signal::input(totalLen, Traits::scalarType());
  auto y = filter::fir(x, rrcTaps);

  auto result = Executor::run<T>(y, inputBuf);

  // 1. 寻找输出中的峰值 (模值)
  RealT maxVal = 0.0;
  int maxIdx = -1;
  int const width = (result.dimensions() > 1) ? result.dim(1).extent() : result.width();
  for (int i = 0; i < width; ++i) {
    auto val = Traits::getElement(result, i);
    auto mag = std::abs(val);
    if (mag > maxVal) {
      maxVal = mag;
      maxIdx = i;
    }
  }

  // 验证是否有有效输出
  bool pass = true;
  if (maxVal < 1e-6) {
    pass = false;
    TestPrinter::printTestResult(name + " [No Output]", false, "Output is all zeros");
    return;
  }

  // 2. 逐点匹配 Taps (Alignment check)
  size_t const tapsLen = rrcTaps.size();
  size_t const startCheck = maxIdx - (tapsLen / 2);

  for (size_t i = 0; i < tapsLen; ++i) {
    size_t const resIdx = startCheck + i;
    if (resIdx >= 0 && resIdx < static_cast<size_t>(width)) {
      auto actual = Traits::getElement(result, (int)resIdx);
      RealT const expected = rrcTaps[i];
      if (std::abs(actual - static_cast<T>(expected)) > 1e-3) {
        pass = false;
      }
    }
  }

  TestPrinter::printTestResult(name, pass);
  assert(pass);
}

template <typename RealT, typename InT, typename TapT>
void testFirMixedCase() {
  using ComplexT = std::complex<RealT>;
  using OutT = std::conditional_t<TestTraits<InT>::IS_COMPLEX || TestTraits<TapT>::IS_COMPLEX,
                                  ComplexT, RealT>;
  static_assert(std::is_same_v<RealT, typename TestTraits<InT>::BufferElemType>,
                "Input precision must match RealT");
  static_assert(std::is_same_v<RealT, typename TestTraits<TapT>::BufferElemType>,
                "Tap precision must match RealT");

  std::string const name = firMixedName<InT, TapT>();
  auto const data = makeFirMixedData<RealT>();
  auto const& input = selectFirInput<InT>(data);
  auto const& taps = selectFirTaps<TapT>(data);

  auto buf = TestTraits<InT>::makeBuffer(input.size());
  TestTraits<InT>::fillBuffer(buf, input);

  auto x = Signal::input(input.size(), TestTraits<InT>::scalarType());
  auto y = filter::fir(x, taps);
  auto out = Executor::run<OutT>(y, buf);

  auto expectedC = naiveFir(toComplexVector(input), toComplexVector(taps));
  bool pass = true;
  if constexpr (TestTraits<OutT>::IS_COMPLEX) {
    pass = checkBufferMatches<OutT>(out, expectedC);
  } else {
    std::vector<RealT> expected(expectedC.size());
    for (size_t i = 0; i < expectedC.size(); ++i) {
      expected[i] = expectedC[i].real();
    }
    pass = checkBufferMatches<RealT>(out, expected);
  }

  TestPrinter::printTestResult(name, pass);
  assert(pass);
}

/**
 * @brief [Test] FIR 对称/反对称 taps
 */
template <typename RealT, typename TapT>
void testFirSymmetryCase(const std::string& label, const std::vector<TapT>& taps) {
  using ComplexT = std::complex<RealT>;
  using Traits = TestTraits<ComplexT>;
  std::string const name = firSymmetryName<RealT>(label);
  auto const input = getFirSymmetryInput<RealT>();

  auto buf = Traits::makeBuffer(input.size());
  Traits::fillBuffer(buf, input);

  auto x = Signal::input(input.size(), Traits::scalarType());
  auto y = filter::fir(x, taps);
  auto out = Executor::run<ComplexT>(y, buf);
  auto expected = naiveFir(input, toComplexVector(taps));
  bool const pass = checkBufferMatches<ComplexT>(out, expected);

  TestPrinter::printTestResult(name, pass);
  assert(pass);
}

template <typename RealT>
void testFirSymmetryEven() {
  std::vector<RealT> const taps = {static_cast<RealT>(1.0), static_cast<RealT>(2.0),
                                   static_cast<RealT>(3.0), static_cast<RealT>(2.0),
                                   static_cast<RealT>(1.0)};
  testFirSymmetryCase<RealT, RealT>("Even", taps);
}

template <typename RealT>
void testFirSymmetryOdd() {
  std::vector<RealT> const taps = {static_cast<RealT>(1.0), static_cast<RealT>(2.0),
                                   static_cast<RealT>(0.0), static_cast<RealT>(-2.0),
                                   static_cast<RealT>(-1.0)};
  testFirSymmetryCase<RealT, RealT>("Odd", taps);
}

template <typename RealT>
void testFirSymmetryConjEven() {
  using ComplexT = std::complex<RealT>;
  std::vector<ComplexT> const taps = {ComplexT(static_cast<RealT>(1.0), static_cast<RealT>(2.0)),
                                      ComplexT(static_cast<RealT>(0.5), static_cast<RealT>(0.0)),
                                      ComplexT(static_cast<RealT>(1.0), static_cast<RealT>(-2.0))};
  testFirSymmetryCase<RealT, ComplexT>("ConjEven", taps);
}

template <typename RealT>
void testFirSymmetryConjOdd() {
  using ComplexT = std::complex<RealT>;
  std::vector<ComplexT> const taps = {ComplexT(static_cast<RealT>(1.0), static_cast<RealT>(2.0)),
                                      ComplexT(static_cast<RealT>(0.0), static_cast<RealT>(1.0)),
                                      ComplexT(static_cast<RealT>(-1.0), static_cast<RealT>(2.0))};
  testFirSymmetryCase<RealT, ComplexT>("ConjOdd", taps);
}

/**
 * @brief [Test] FIR 零输入测试 (Zero Input)
 */
template <typename T>
void testFirZeroInput() {
  using Traits = TestTraits<T>;
  using RealT = typename Traits::BufferElemType;
  std::string const name = "FIR Zero Input (" + TypeName<T>::get() + ")";
  int const len = 50;
  auto inputBuf = Traits::makeBuffer(len);
  std::vector<T> const zeros(len, static_cast<T>(0.0));
  Traits::fillBuffer(inputBuf, zeros);

  std::vector<RealT> const rrcTaps = getRRCTaps<T>();
  Signal const x = Signal::input(len, Traits::scalarType());
  Signal const y = filter::fir(x, rrcTaps);

  auto result = Executor::run<T>(y, inputBuf);

  bool pass = true;
  for (int i = 0; i < len; ++i) {
    auto val = Traits::getElement(result, i);
    if (std::abs(val) > 1e-4) pass = false;
  }

  TestPrinter::printTestResult(name, pass);
  assert(pass);
}

// ============================================================================
// 非线性滤波器测试组 (Median / Moving Average)
// ============================================================================

/**
 * @brief [Test] 中值滤波 (W=3)
 */
template <typename T>
void testMedianWindow3() {
  using Traits = TestTraits<T>;
  std::string const name = "Median Filter Window 3 (" + TypeName<T>::get() + ")";
  auto medianIn = getMedianIn<T>();
  auto medianOut = getMedianOutWindow3<T>();
  int const len = medianIn.size();

  auto inputBuf = Traits::makeBuffer(len);
  Traits::fillBuffer(inputBuf, medianIn);

  Signal const x = Signal::input(len, Traits::scalarType());
  Signal const y = filter::median(x, 3);

  auto result = Executor::run<T>(y, inputBuf);

  bool pass = true;
  for (int i = 0; i < len; ++i) {
    auto val = Traits::getElement(result, i);
    if (!approxEqual(val, medianOut[i])) pass = false;
  }

  TestPrinter::printTestResult(name, pass);
  assert(pass);
}

/**
 * @brief [Test] 中值滤波 (W=5)
 */
template <typename T>
void testMedianWindow5() {
  using Traits = TestTraits<T>;
  std::string const name = "Median Filter Window 5 (" + TypeName<T>::get() + ")";
  auto medianIn = getMedianIn<T>();
  auto medianOut = getMedianOutWindow5<T>();
  int const len = medianIn.size();

  auto inputBuf = Traits::makeBuffer(len);
  Traits::fillBuffer(inputBuf, medianIn);

  Signal const x = Signal::input(len, Traits::scalarType());
  Signal const y = filter::median(x, 5);

  auto result = Executor::run<T>(y, inputBuf);

  bool pass = true;
  for (int i = 0; i < len; ++i) {
    auto val = Traits::getElement(result, i);
    if (!approxEqual(val, medianOut[i])) pass = false;
  }

  TestPrinter::printTestResult(name, pass);
  assert(pass);
}

/**
 * @brief [Test] 复数中值滤波 (W=3, magnitude sort)
 */
template <typename T>
void testMedianComplexWindow3() {
  static_assert(IS_COMPLEX_V<T>, "Complex type required");
  using Traits = TestTraits<T>;
  using RealT = typename T::value_type;
  std::string const name = "Median Filter Window 3 (Complex Mag) (" + TypeName<T>::get() + ")";

  std::vector<T> const medianIn = {T(static_cast<RealT>(1.0), static_cast<RealT>(2.0)),
                                   T(static_cast<RealT>(-2.0), static_cast<RealT>(0.0)),
                                   T(static_cast<RealT>(0.0), static_cast<RealT>(3.0)),
                                   T(static_cast<RealT>(4.0), static_cast<RealT>(0.0)),
                                   T(static_cast<RealT>(-1.0), static_cast<RealT>(-1.0))};
  std::vector<T> const medianOut = {medianIn[1], medianIn[0], medianIn[2], medianIn[2],
                                    medianIn[4]};
  int const len = static_cast<int>(medianIn.size());

  auto inputBuf = Traits::makeBuffer(len);
  Traits::fillBuffer(inputBuf, medianIn);

  Signal const x = Signal::input(len, Traits::scalarType());
  Signal const y = filter::median(x, 3);

  auto result = Executor::run<T>(y, inputBuf);

  bool pass = true;
  for (int i = 0; i < len; ++i) {
    auto val = Traits::getElement(result, i);
    if (!approxEqual(val, medianOut[i])) pass = false;
  }

  TestPrinter::printTestResult(name, pass);
  assert(pass);
}

/**
 * @brief [Test] 滑动平均 (Moving Average, W=4)
 */
template <typename T>
void testMovingAverage() {
  using Traits = TestTraits<T>;
  std::string const name = "Moving Average (" + TypeName<T>::get() + ")";
  int const len = 8;
  auto inputBuf = Traits::makeBuffer(len);
  std::vector<T> data(len);
  for (int i = 0; i < len; ++i) {
    data[i] = static_cast<T>(i + 1);  // 1, 2, 3, ...
  }
  Traits::fillBuffer(inputBuf, data);

  Signal const x = Signal::input(len, Traits::scalarType());
  Signal const y = filter::movingAverage(x, 4);

  auto result = Executor::run<T>(y, inputBuf);

  bool pass = true;
  auto v3 = Traits::getElement(result, 3);
  auto v4 = Traits::getElement(result, 4);
  auto v5 = Traits::getElement(result, 5);

  if (std::abs(v3 - static_cast<T>(2.5)) > 1e-3) pass = false;
  if (std::abs(v4 - static_cast<T>(3.5)) > 1e-3) pass = false;
  if (std::abs(v5 - static_cast<T>(4.5)) > 1e-3) pass = false;

  TestPrinter::printTestResult(name, pass);
  assert(pass);
}

// ============================================================================
// 大规模压力测试
// ============================================================================

/**
 * @brief [Test] 大规模 FIR 滤波 (256 taps)
 */
template <typename T>
void testLargeFir() {
  using Traits = TestTraits<T>;
  using RealT = typename Traits::BufferElemType;
  std::string const name = "Large FIR (256 taps) (" + TypeName<T>::get() + ")";

  // 构造直通滤波器 Taps (Real)
  int const tapsLen = 256;
  std::vector<RealT> taps(tapsLen, static_cast<RealT>(0.0));
  taps[0] = static_cast<RealT>(1.0);

  int const sigLen = 1000;
  auto inputBuf = Traits::makeBuffer(sigLen);
  std::vector<T> data(sigLen);
  for (int i = 0; i < sigLen; ++i) {
    data[i] = static_cast<T>(i % 10);
  }
  Traits::fillBuffer(inputBuf, data);

  Signal const x = Signal::input(sigLen, Traits::scalarType());
  Signal const y = filter::fir(x, taps);

  auto result = Executor::run<T>(y, inputBuf);

  bool pass = true;
  for (int i = 0; i < sigLen; ++i) {
    auto val = Traits::getElement(result, i);
    T expected = data[i];
    if (std::abs(val - expected) > 1e-4) pass = false;
  }

  TestPrinter::printTestResult(name, pass);
  assert(pass);
}

// ============================================================================
// Main Entry
// ============================================================================

int main() {
  TestPrinter::printSuiteHeader("Filter Tests");

  TestPrinter::printSection("FIR Basic Tests");
  runTest([]() { testFirImpulseResponse<real32_t>(); }, "FIR Impulse (Real32)");
  runTest([]() { testFirImpulseResponse<real64_t>(); }, "FIR Impulse (Real64)");
  runTest([]() { testFirImpulseResponse<complex32_t>(); }, "FIR Impulse (Complex32)");
  runTest([]() { testFirImpulseResponse<complex64_t>(); }, "FIR Impulse (Complex64)");
  runTest([]() { testFirZeroInput<real32_t>(); }, "FIR Zero (Real32)");
  runTest([]() { testFirZeroInput<real64_t>(); }, "FIR Zero (Real64)");
  runTest([]() { testFirZeroInput<complex32_t>(); }, "FIR Zero (Complex32)");
  runTest([]() { testFirZeroInput<complex64_t>(); }, "FIR Zero (Complex64)");
  runTest([]() { testLargeFir<real32_t>(); }, "Large FIR (Real32)");
  runTest([]() { testLargeFir<real64_t>(); }, "Large FIR (Real64)");
  runTest([]() { testLargeFir<complex32_t>(); }, "Large FIR (Complex32)");
  runTest([]() { testLargeFir<complex64_t>(); }, "Large FIR (Complex64)");

  TestPrinter::printSection("FIR Mixed Types Tests");
  runTest([]() { testFirMixedCase<real32_t, real32_t, real32_t>(); },
          firMixedName<real32_t, real32_t>());
  runTest([]() { testFirMixedCase<real32_t, real32_t, complex32_t>(); },
          firMixedName<real32_t, complex32_t>());
  runTest([]() { testFirMixedCase<real32_t, complex32_t, real32_t>(); },
          firMixedName<complex32_t, real32_t>());
  runTest([]() { testFirMixedCase<real32_t, complex32_t, complex32_t>(); },
          firMixedName<complex32_t, complex32_t>());

  runTest([]() { testFirMixedCase<real64_t, real64_t, real64_t>(); },
          firMixedName<real64_t, real64_t>());
  runTest([]() { testFirMixedCase<real64_t, real64_t, complex64_t>(); },
          firMixedName<real64_t, complex64_t>());
  runTest([]() { testFirMixedCase<real64_t, complex64_t, real64_t>(); },
          firMixedName<complex64_t, real64_t>());
  runTest([]() { testFirMixedCase<real64_t, complex64_t, complex64_t>(); },
          firMixedName<complex64_t, complex64_t>());

  TestPrinter::printSection("FIR Symmetry Tests");
  runTest([]() { testFirSymmetryEven<real32_t>(); }, firSymmetryName<real32_t>("Even"));
  runTest([]() { testFirSymmetryOdd<real32_t>(); }, firSymmetryName<real32_t>("Odd"));
  runTest([]() { testFirSymmetryConjEven<real32_t>(); }, firSymmetryName<real32_t>("ConjEven"));
  runTest([]() { testFirSymmetryConjOdd<real32_t>(); }, firSymmetryName<real32_t>("ConjOdd"));
  runTest([]() { testFirSymmetryEven<real64_t>(); }, firSymmetryName<real64_t>("Even"));
  runTest([]() { testFirSymmetryOdd<real64_t>(); }, firSymmetryName<real64_t>("Odd"));
  runTest([]() { testFirSymmetryConjEven<real64_t>(); }, firSymmetryName<real64_t>("ConjEven"));
  runTest([]() { testFirSymmetryConjOdd<real64_t>(); }, firSymmetryName<real64_t>("ConjOdd"));

  TestPrinter::printSection("MovingAverage Filter Tests");
  runTest([]() { testMovingAverage<real32_t>(); }, "Moving Avg (Real32)");
  runTest([]() { testMovingAverage<real64_t>(); }, "Moving Avg (Real64)");
  runTest([]() { testMovingAverage<complex32_t>(); }, "Moving Avg (Complex32)");
  runTest([]() { testMovingAverage<complex64_t>(); }, "Moving Avg (Complex64)");

  TestPrinter::printSection("Median Filter Tests");
  runTest([]() { testMedianWindow3<real32_t>(); }, "Median (Window=3) (Real32)");
  runTest([]() { testMedianWindow3<real64_t>(); }, "Median (Window=3) (Real64)");
  runTest([]() { testMedianWindow5<real32_t>(); }, "Median (Window=5) (Real32)");
  runTest([]() { testMedianWindow5<real64_t>(); }, "Median (Window=5) (Real64)");
  runTest([]() { testMedianComplexWindow3<complex32_t>(); }, "Median (Window=3) (Complex32)");
  runTest([]() { testMedianComplexWindow3<complex64_t>(); }, "Median (Window=3) (Complex64)");

  TestPrinter::printSummary();
  return 0;
}
