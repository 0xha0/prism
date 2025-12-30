/**
 * @file test_filter.cpp
 * @ingroup tests
 * @brief 滤波算子单元测试
 *
 * 覆盖 FIR/IIR/移动平均/中值滤波等算子，针对实/复数与不同精度进行验证。
 */

#include <Halide.h>

#include <cassert>
#include <cmath>
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
using prism::real32_t;
using prism::real64_t;
using prism::ScalarType;
using namespace prism::dsl;
using namespace prism::runtime;
using namespace prism::tests;

/// @addtogroup tests
/// @{

// ============================================================================
// 数据生成与辅助
// ============================================================================

/** @brief RRC 滤波器 taps（以 T 精度返回） */
template <typename T>
std::vector<T> getRRCTaps() {
  // Original float taps cast to T
  return {static_cast<T>(0.014675),  static_cast<T>(-0.005671),
          static_cast<T>(-0.034726), static_cast<T>(-0.048463),
          static_cast<T>(-0.021758), static_cast<T>(0.053149),
          static_cast<T>(0.156148),  static_cast<T>(0.245903),
          static_cast<T>(0.281488),  static_cast<T>(0.245903),
          static_cast<T>(0.156148),  static_cast<T>(0.053149),
          static_cast<T>(-0.021758), static_cast<T>(-0.048463),
          static_cast<T>(-0.034726), static_cast<T>(-0.005671),
          static_cast<T>(0.014675)};
}

/** @brief 低通滤波器 b 系数 */
template <typename T>
std::vector<T> getLPFB() {
  return {static_cast<T>(0.004824), static_cast<T>(0.019297),
          static_cast<T>(0.028946), static_cast<T>(0.019297),
          static_cast<T>(0.004824)};
}

/** @brief 低通滤波器 a 系数 */
template <typename T>
std::vector<T> getLPFA() {
  return {static_cast<T>(1.000000), static_cast<T>(-2.369513),
          static_cast<T>(2.313988), static_cast<T>(-1.054665),
          static_cast<T>(0.187379)};
}

/** @brief 低通滤波器预期脉冲响应 */
template <typename T>
std::vector<T> getLPFImpulseResp() {
  return {static_cast<T>(0.004824),  static_cast<T>(0.030728),
          static_cast<T>(0.090593),  static_cast<T>(0.167942),
          static_cast<T>(0.224638),  static_cast<T>(0.233454),
          static_cast<T>(0.193510),  static_cast<T>(0.123763),
          static_cast<T>(0.049603),  static_cast<T>(-0.008508),
          static_cast<T>(-0.040672), static_cast<T>(-0.047561),
          static_cast<T>(-0.036850), static_cast<T>(-0.018562),
          static_cast<T>(-0.001252), static_cast<T>(0.010033),
          static_cast<T>(0.013999),  static_cast<T>(0.012111),
          static_cast<T>(0.007121),  static_cast<T>(0.001733)};
}

/** @brief 中值滤波测试输入 */
template <typename T>
std::vector<T> getMedianIn() {
  return {static_cast<T>(1.0), static_cast<T>(5.0), static_cast<T>(2.0),
          static_cast<T>(8.0), static_cast<T>(3.0), static_cast<T>(4.0)};
}

/** @brief 中值滤波预期输出 */
template <typename T>
std::vector<T> getMedianOut() {
  return {static_cast<T>(1.0), static_cast<T>(2.0), static_cast<T>(5.0),
          static_cast<T>(3.0), static_cast<T>(4.0), static_cast<T>(3.0)};
}

/** @brief 中值滤波（窗口=5）预期输出 */
template <typename T>
std::vector<T> getMedianOutWindow5() {
  return {static_cast<T>(1.0), static_cast<T>(2.0), static_cast<T>(3.0),
          static_cast<T>(4.0), static_cast<T>(3.0), static_cast<T>(3.0)};
}

// ============================================================================
// FIR 滤波测试
// ============================================================================

/** @brief FIR 冲激响应测试 */
template <typename T>
void testFirImpulseResponse() {
  std::string const name = "FIR Impulse Response (" + TypeName<T>::get() + ")";
  int const totalLen = 100;
  std::vector<T> impulseData(totalLen, static_cast<T>(0.0));
  impulseData[totalLen / 2] = static_cast<T>(1.0);

  Halide::Buffer<T> inputBuf(totalLen);
  for (int i = 0; i < totalLen; ++i) {
    inputBuf(i) = impulseData[i];
  }

  std::vector<T> rrcTaps = getRRCTaps<T>();

  Signal const x = Signal::input(totalLen);

  Signal const y = filter::fir(x, rrcTaps);

  auto result = Executor::run<T>(y, inputBuf);

  // 查找峰值位置和幅度
  T maxVal = 0.0;
  int maxIdx = -1;
  for (int i = 0; i < result.width(); ++i) {
    if (result(i) > maxVal) {
      maxVal = result(i);
      maxIdx = i;
    }
  }

  // Verify
  size_t const tapsLen = rrcTaps.size();
  size_t const startCheck = maxIdx - (tapsLen / 2);

  bool pass = true;
  for (size_t i = 0; i < tapsLen; ++i) {
    size_t const resIdx = startCheck + i;
    if (resIdx >= 0 && resIdx < (size_t)result.width()) {
      // Use slightly looser epsilon for R32 accumulated error
      if (!approxEqual(result((int)resIdx), rrcTaps[i], 1e-3)) pass = false;
    }
  }

  TestPrinter::printTestResult(name, pass);
  assert(pass);
}

/** @brief FIR 零输入测试（输出应全 0） */
template <typename T>
void testFirZeroInput() {
  std::string const name = "FIR Zero Input (" + TypeName<T>::get() + ")";
  int const len = 50;
  Halide::Buffer<T> inputBuf(len);
  for (int i = 0; i < len; ++i) {
    inputBuf(i) = static_cast<T>(0.0);
  }

  std::vector<T> const rrcTaps = getRRCTaps<T>();
  Signal const x = Signal::input(len);
  Signal const y = filter::fir(x, rrcTaps);

  auto result = Executor::run<T>(y, inputBuf);

  bool pass = true;
  for (int i = 0; i < len; ++i) {
    if (!approxEqual(result(i), static_cast<T>(0.0))) pass = false;
  }

  TestPrinter::printTestResult(name, pass);
  assert(pass);
}

// ============================================================================
// IIR 滤波测试
// ============================================================================

/** @brief IIR 冲激响应测试（验证系数与数值稳定性） */
template <typename T>
void testIirImpulseResponse() {
  std::string const name = "IIR Impulse Response (" + TypeName<T>::get() + ")";
  int const len = 20;
  Halide::Buffer<T> inputBuf(len);
  for (int i = 0; i < len; ++i) {
    inputBuf(i) = (i == 0) ? static_cast<T>(1.0) : static_cast<T>(0.0);
  }

  std::vector<T> const lpfB = getLPFB<T>();
  std::vector<T> const lpfA = getLPFA<T>();
  std::vector<T> lpfImpulse = getLPFImpulseResp<T>();

  Signal const x = Signal::input(len);
  Signal const y = filter::iir(x, lpfB, lpfA);

  auto result = Executor::run<T>(y, inputBuf);

  bool pass = true;
  for (int i = 0; i < len; ++i) {
    if (!approxEqual(result(i), lpfImpulse[i], 1e-3)) pass = false;
  }

  TestPrinter::printTestResult(name, pass);
  assert(pass);
}

/** @brief IIR 退化为 FIR 的情况（仅保留 b 系数） */
template <typename T>
void testIirAsFir() {
  std::string const name = "IIR as Pure FIR (" + TypeName<T>::get() + ")";
  std::vector<T> const b = {static_cast<T>(0.25), static_cast<T>(0.5),
                            static_cast<T>(0.25)};
  std::vector<T> const a = {static_cast<T>(1.0)};

  int const len = 10;
  Halide::Buffer<T> inputBuf(len);
  for (int i = 0; i < len; ++i) {
    inputBuf(i) = (i == 0) ? static_cast<T>(1.0) : static_cast<T>(0.0);
  }

  Signal const x = Signal::input(len);
  Signal const y = filter::iir(x, b, a);

  auto result = Executor::run<T>(y, inputBuf);

  bool pass = true;
  if (!approxEqual(result(0), static_cast<T>(0.25), 1e-3)) pass = false;
  if (!approxEqual(result(1), static_cast<T>(0.5), 1e-3)) pass = false;
  if (!approxEqual(result(2), static_cast<T>(0.25), 1e-3)) pass = false;
  for (int i = 3; i < len; ++i) {
    if (!approxEqual(result(i), static_cast<T>(0.0), 1e-3)) pass = false;
  }

  TestPrinter::printTestResult(name, pass);
  assert(pass);
}

// ============================================================================
// FIR 滤波测试（复数）
// ============================================================================

/** @brief 复数输入的 FIR 测试，验证双通道卷积正确性 */
template <typename ComplexT>
void testComplexFir() {
  // Infer scalar type from ComplexT
  using RealT = typename ComplexT::value_type;
  std::string const name = "Complex FIR (" + TypeName<ComplexT>::get() + ")";

  bool pass = true;

  std::vector<ComplexT> const sigData = {{1, 1}, {2, 2}, {3, 3}};
  std::vector<RealT> const taps = {static_cast<RealT>(0.5),
                                   static_cast<RealT>(0.5)};
  int const len = 3;

  ScalarType const st =
      (sizeof(RealT) == 8) ? ScalarType::C64 : ScalarType::C32;
  auto s = Signal::input(len, st);
  auto fir = filter::fir(s, taps);

  // Buffer
  Halide::Buffer<RealT> inBuf(2, len);
  fillComplexBuffer(inBuf, sigData);

  Executor const exec;
  auto out = exec.run<ComplexT>(fir, inBuf);

  // y[0] = 0.5+0.5i
  // y[1] = 1.5+1.5i
  // y[2] = 2.5+2.5i
  std::vector<ComplexT> expected = {{0.5, 0.5}, {1.5, 1.5}, {2.5, 2.5}};

  for (int i = 0; i < len; ++i) {
    ComplexT act;
    if constexpr (std::is_same_v<RealT, double>) {
      act = getComplex64(out, i);
    } else {
      act = getComplex32(out, i);
    }
    ComplexT exp = expected[i];
    if (std::abs(act - exp) > 1e-5) {
      pass = false;
      TestPrinter::printTestResult(name + " mismatch", false,
                                   "idx=" + std::to_string(i));
    }
  }

  TestPrinter::printTestResult(name, pass);
  assert(pass);
}

// ============================================================================
// 中值滤波与滑动平均测试
// ============================================================================

/** @brief 中值滤波窗口=3 的输出验证 */
template <typename T>
void testMedianWindow3() {
  std::string const name =
      "Median Filter Window 3 (" + TypeName<T>::get() + ")";
  auto medianIn = getMedianIn<T>();
  auto medianOut = getMedianOut<T>();
  int const len = medianIn.size();

  Halide::Buffer<T> inputBuf(len);
  for (int i = 0; i < len; ++i) {
    inputBuf(i) = medianIn[i];
  }

  Signal const x = Signal::input(len);
  Signal const y = filter::median(x, 3);

  auto result = Executor::run<T>(y, inputBuf);

  bool pass = true;
  for (int i = 0; i < len; ++i) {
    if (!approxEqual(result(i), medianOut[i])) pass = false;
  }

  TestPrinter::printTestResult(name, pass);
  assert(pass);
}

/** @brief 中值滤波窗口=5 的输出验证 */
template <typename T>
void testMedianWindow5() {
  std::string const name =
      "Median Filter Window 5 (" + TypeName<T>::get() + ")";
  auto medianIn = getMedianIn<T>();
  auto medianOut = getMedianOutWindow5<T>();
  int const len = medianIn.size();

  Halide::Buffer<T> inputBuf(len);
  for (int i = 0; i < len; ++i) {
    inputBuf(i) = medianIn[i];
  }

  Signal const x = Signal::input(len);
  Signal const y = filter::median(x, 5);

  auto result = Executor::run<T>(y, inputBuf);

  bool pass = true;
  for (int i = 0; i < len; ++i) {
    if (!approxEqual(result(i), medianOut[i])) pass = false;
  }

  TestPrinter::printTestResult(name, pass);
  assert(pass);
}
/** @brief 滑动平均窗口=4 的输出验证 */
template <typename T>
void testMovingAverage() {
  std::string const name = "Moving Average (" + TypeName<T>::get() + ")";
  int const len = 8;
  Halide::Buffer<T> inputBuf(len);
  for (int i = 0; i < len; ++i) {
    inputBuf(i) = static_cast<T>(i + 1);  // 1, 2, 3, ...
  }

  Signal const x = Signal::input(len);
  Signal const y = filter::movingAverage(x, 4);

  auto result = Executor::run<T>(y, inputBuf);

  // result[3] = (4+3+2+1)/4 = 2.5
  // result[4] = (5+4+3+2)/4 = 3.5
  bool pass = true;
  if (!approxEqual(result(3), static_cast<T>(2.5), 1e-3)) pass = false;
  if (!approxEqual(result(4), static_cast<T>(3.5), 1e-3)) pass = false;
  if (!approxEqual(result(5), static_cast<T>(4.5), 1e-3)) pass = false;

  TestPrinter::printTestResult(name, pass);
  assert(pass);
}

// ============================================================================
// 大规模测试
// ============================================================================

/** @brief 大规模 FIR（256 taps）正确性冒烟 */
template <typename T>
void testLargeFir() {
  std::string const name = "Large FIR (256 taps) (" + TypeName<T>::get() + ")";

  // Create 256 taps: [1, 0, ..., 0] -> Identity
  int const tapsLen = 256;
  std::vector<T> taps(tapsLen, static_cast<T>(0.0));
  taps[0] = static_cast<T>(1.0);  // Identity filter

  int const sigLen = 1000;
  Halide::Buffer<T> inputBuf(sigLen);
  for (int i = 0; i < sigLen; ++i) {
    inputBuf(i) = static_cast<T>(i % 10);
  }

  Signal const x = Signal::input(sigLen);
  Signal const y = filter::fir(x, taps);

  auto result = Executor::run<T>(y, inputBuf);

  bool pass = true;
  for (int i = 0; i < sigLen; ++i) {
    // With identity taps, output should equal input (valid range starts from 0
    // effectively with padding) Implementation pads, so convolution with
    // [1,0..] works like delay 0.
    if (!approxEqual(result(i), inputBuf(i))) pass = false;
  }

  TestPrinter::printTestResult(name, pass);
  assert(pass);
}

// ============================================================================
// Main
// ============================================================================

int main() {
  TestPrinter::printSuiteHeader("Filter Tests (Refactored)");

  auto run = [](auto func, const std::string& name) {
    TestPrinter::printSection(name + " [Running on CPU & GPU]");
    runTest(func, name);
  };

  run([]() { testFirImpulseResponse<real32_t>(); }, "FIR Impulse (Real32)");
  run([]() { testFirZeroInput<real32_t>(); }, "FIR Zero (Real32)");
  run([]() { testLargeFir<real32_t>(); }, "Large FIR (Real32)");

  run([]() { testFirImpulseResponse<real64_t>(); }, "FIR Impulse (Real64)");
  run([]() { testFirZeroInput<real64_t>(); }, "FIR Zero (Real64)");
  run([]() { testLargeFir<real64_t>(); }, "Large FIR (Real64)");

  run([]() { testComplexFir<complex32_t>(); }, "FIR Complex (Complex32)");
  run([]() { testComplexFir<complex64_t>(); }, "FIR Complex (Complex64)");

  run([]() { testIirImpulseResponse<real32_t>(); }, "IIR Impulse (Real32)");
  run([]() { testIirAsFir<real32_t>(); }, "IIR as FIR (Real32)");

  run([]() { testIirImpulseResponse<real64_t>(); }, "IIR Impulse (Real64)");
  run([]() { testIirAsFir<real64_t>(); }, "IIR as FIR (Real64)");

  run([]() { testMedianWindow3<real32_t>(); }, "Median (Window=3) (Real32)");
  run([]() { testMedianWindow5<real32_t>(); }, "Median (Window=5) (Real32)");
  run([]() { testMedianWindow3<real64_t>(); }, "Median (Window=3) (Real64)");
  run([]() { testMedianWindow5<real64_t>(); }, "Median (Window=5) (Real64)");

  run([]() { testMovingAverage<real32_t>(); }, "Moving Avg (Real32)");
  run([]() { testMovingAverage<real64_t>(); }, "Moving Avg (Real64)");

  TestPrinter::printSummary();
  return 0;
}

/// @}
