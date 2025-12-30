/**
 * @file test_fft.cpp
 * @ingroup tests
 * @brief FFT Anchor 接口单元测试
 *
 * 覆盖 C2C/R2C/C2R/批量接口，验证单/双精度下的正确性。
 */

#include <cassert>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "prism/backend/fft_backend.h"
#include "prism/runtime/fft.h"
#include "prism/types.h"
#include "test_utils.h"

using namespace prism;
using namespace prism::runtime;
using namespace prism::tests;

/// @addtogroup tests
/// @{

// ==============================================================================
// 模板化测试用例
// ==============================================================================

/**
 * @brief 冲激信号 FFT：验证 C2C 正变换
 * @tparam T 精度类型
 */
template <typename T>
void testImpulseFft() {
  std::string const name = "Impulse FFT (8pt) (" + TypeName<T>::get() + ")";
  std::vector<std::complex<T>> data(8, {0, 0});
  data[0] = {1, 0};  // Impulse

  FFT::forward(data.data(), 8);

  // Impulse FFT -> All ones
  std::vector<std::complex<T>> const expected(8, {1, 0});
  T err = maxError(data, expected);

  bool const pass = err < 1e-5;
  std::ostringstream oss;
  oss << "err=" << std::scientific << err;
  TestPrinter::printTestResult(name, pass, oss.str());
  assert(pass);
}

/**
 * @brief 正弦波频谱定位测试
 * @tparam T 精度类型
 */
template <typename T>
void testSineFft() {
  std::string const name =
      "Sine FFT (64pt, freq=4) (" + TypeName<T>::get() + ")";
  const int n = 64;
  const int freq = 4;
  std::vector<std::complex<T>> data(n);
  const T piVal = M_PI_VAL;

  for (int i = 0; i < n; ++i) {
    T t = static_cast<T>(i) / static_cast<T>(n);
    data[i] = {std::sin(2 * piVal * freq * t), 0};
  }

  FFT::forward(data.data(), n);

  T peakMag = std::abs(data[freq]);
  T expectedPeak = static_cast<T>(n) / 2.0;

  bool const pass = std::abs(peakMag - expectedPeak) < expectedPeak * 0.1;
  std::ostringstream oss;
  oss << "peak=" << std::fixed << std::setprecision(1) << peakMag;
  TestPrinter::printTestResult(name, pass, oss.str());
  assert(pass);
}

/**
 * @brief C2C 正反变换往返误差测试
 * @tparam T 精度类型
 */
template <typename T>
void testRoundtripC2C() {
  std::string const name = "Roundtrip C2C (128pt) (" + TypeName<T>::get() + ")";
  const int n = 128;
  std::vector<std::complex<T>> original(n);
  std::vector<std::complex<T>> data(n);

  for (int i = 0; i < n; ++i) {
    original[i] = {static_cast<T>(i), static_cast<T>(i * 0.5)};
    data[i] = original[i];
  }

  FFT::forward(data.data(), n);
  FFT::inverse(data.data(), n);  // normalize=true

  T err = maxError(data, original);
  T tol = (sizeof(T) == 8) ? 1e-10 : 1e-3;

  bool const pass = err < tol;
  std::ostringstream oss;
  oss << "err=" << std::scientific << err;
  TestPrinter::printTestResult(name, pass, oss.str());
  assert(pass);
}

/**
 * @brief R2C/C2R 往返误差测试
 * @tparam T 精度类型
 */
template <typename T>
void testR2cC2r() {
  std::string const name =
      "R2C -> C2R Roundtrip (128pt) (" + TypeName<T>::get() + ")";
  const int n = 128;  // must be even
  std::vector<T> input(n);
  for (int i = 0; i < n; ++i) input[i] = static_cast<T>(std::cos(0.5 * i));

  std::vector<std::complex<T>> freq((n / 2) + 1);
  std::vector<T> output(n);

  // R2C
  FFT::forward(input.data(), freq.data(), n);

  // C2R
  FFT::inverse(freq.data(), output.data(), n);

  T err = maxError(input, output);
  T tol = (sizeof(T) == 8) ? 1e-10 : 1e-3;

  bool const pass = err < tol;
  std::ostringstream oss;
  oss << "err=" << std::scientific << err;
  TestPrinter::printTestResult(name, pass, oss.str());
  assert(pass);
}

/**
 * @brief 批量 C2C 变换正确性测试
 * @tparam T 精度类型
 */
template <typename T>
void testBatch() {
  std::string const name = "Batch C2C (16x64) (" + TypeName<T>::get() + ")";
  int64_t const n = 64;
  int64_t const batch = 16;
  std::vector<std::complex<T>> data(n * batch);
  std::vector<std::complex<T>> original(n * batch);

  for (int i = 0; i < n * batch; ++i) {
    data[i] = {static_cast<T>(i % n), 0};
    original[i] = data[i];
  }

  FFT::batch(data.data(), n, batch, -1);  // Forward
  FFT::batch(data.data(), n, batch, 1);   // Inverse

  T err = maxError(data, original);
  T tol = (sizeof(T) == 8) ? 1e-10 : 1e-3;

  bool const pass = err < tol;
  std::ostringstream oss;
  oss << "err=" << std::scientific << err;
  TestPrinter::printTestResult(name, pass, oss.str());
  assert(pass);
}

// ==============================================================================
// 主测试运行器
// ==============================================================================

int main() {
  TestPrinter::printSuiteHeader("FFT Tests (Refactored)");
  TestPrinter::printInfo("Backend", backend::getFftBackend().name());

  // Real32 Tests
  TestPrinter::printSection("FFT Operations (Real32)");
  testImpulseFft<real32_t>();
  testSineFft<real32_t>();
  testRoundtripC2C<real32_t>();
  testR2cC2r<real32_t>();
  testBatch<real32_t>();

  // Real64 Tests
  TestPrinter::printSection("FFT Operations (Real64)");
  if (FFT::supports(ScalarType::F64, FftTransType::C2C)) {
    try {
      testImpulseFft<real64_t>();
      testSineFft<real64_t>();
      testRoundtripC2C<real64_t>();
      testR2cC2r<real64_t>();
      testBatch<real64_t>();
    } catch (const std::exception& e) {
      // We can iterate individually if we want cleaner output but this wraps
      // the block
      TestPrinter::printTestResult("FFT (Real64) Block", false, e.what());
    }
  } else {
    TestPrinter::printSkip("FFT (Real64)", "Backend reports no support");
  }

  TestPrinter::printSummary();
  return 0;
}

/// @}
