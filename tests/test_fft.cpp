/**
 * @file test_fft.cpp
 * @ingroup tests
 * @brief FFT 变换单元测试
 *
 * 验证 PRISM FFT 抽象层的正确性，覆盖以下场景：
 * - 复数到复数 (C2C)：正变换与反变换
 * - 实数到复数/复数到实数 (R2C/C2R)：利用共轭对称性优化
 * - 批量变换 (Batch FFT)：多通道同时处理
 * - 精度支持：Real32 与 Real64 (若后端支持)
 *
 * 测试方法包括冲激响应验证、特定频率信号幅值验证以及正反变换对的还原度测试
 * (Round-trip)
 */

#include <cassert>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
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

namespace {
/** @brief 获取当前 FFT 后端名称 (用于日志显示) */
const std::string& fftBackendLabel() {
  static std::string const label = []() {
    std::string name = prism::backend::getFftBackend().name();
    auto pos = name.find(" (");
    if (pos != std::string::npos) {
      name = name.substr(0, pos);
    }
    return name;
  }();
  return label;
}

inline void fftSync() { FFT::deviceSyncGlobal(); }
}  // namespace

// ==============================================================================
// 模板化测试用例 (Templated Test Cases)
// ==============================================================================

/**
 * @brief [Test] 冲激信号 FFT 测试 (Impulse Response)
 *
 * 验证 C2C 正变换的基础性质
 * - **输入**：离散冲激信号 $\delta[n] = \{1, 0, \dots, 0\}$
 * - **期望输出**：全频带恒定幅值 Spectrum[k] = $\{1, 1, \dots, 1\}$
 *
 * 物理意义：时域的单位冲激对应频域的常数谱（全通特性）
 */
template <typename T>
void testImpulseFft() {
  std::string const name = "Impulse FFT (8pt)";
  std::vector<std::complex<T>> data(8, {0, 0});
  data[0] = {1, 0};  // Impulse

  FFT::forward(data.data(), 8);
  fftSync();

  // Impulse FFT -> All ones
  std::vector<std::complex<T>> const expected(8, {1, 0});
  T err = maxError(data, expected);

  bool const pass = err < 1e-5;
  std::ostringstream oss;
  oss << "err=" << std::scientific << err;
  TestPrinter::printTestResult(name, pass, oss.str(), fftBackendLabel());
  assert(pass);
}

/**
 * @brief [Test] 单频正弦波频谱定位测试 (Tone Detection)
 *
 * 验证 FFT 能正确识别信号频率和幅值
 * - **输入**：$x[n] = \sin(2\pi k n / N)$，其中 $k=4, N=64$
 * - **期望输出**：在 index=4 处出现峰值，理论幅值为 $N/2$ (因为 Sine
 * 包含正负两个频率分量)
 *
 * 注意：$\sin(\omega t) = (e^{j\omega t} - e^{-j\omega t}) /
 * 2j$，故正频率分量幅值为 $N/2$
 */
template <typename T>
void testSineFft() {
  std::string const name = "Sine (64pt, freq=4)";
  const int n = 64;
  const int freq = 4;
  std::vector<std::complex<T>> data(n);
  const T piVal = M_PI_VAL;

  for (int i = 0; i < n; ++i) {
    T t = static_cast<T>(i) / static_cast<T>(n);
    data[i] = {std::sin(2 * piVal * freq * t), 0};
  }

  FFT::forward(data.data(), n);
  fftSync();

  T peakMag = std::abs(data[freq]);
  // sin 的幅度为1，FFT 后单边峰值幅度为 N/2 = 32
  T expectedPeak = static_cast<T>(n) / 2.0;

  bool const pass = std::abs(peakMag - expectedPeak) < expectedPeak * 0.1;
  std::ostringstream oss;
  oss << "peak=" << std::fixed << std::setprecision(1) << peakMag;
  TestPrinter::printTestResult(name, pass, oss.str(), fftBackendLabel());
  assert(pass);
}

/**
 * @brief [Test] C2C 正反变换往返测试 (Round-trip C2C)
 *
 * 验证 FFT 与 IFFT 的互逆性
 * 流程：Data -> Forward FFT -> Inverse FFT (normalized) -> Data'
 * 验证 Data 与 Data' 的误差是否在容限范围内
 */
template <typename T>
void testRoundtripC2C() {
  std::string const name = "C2C-Roundtrip (128pt)";
  const int n = 128;
  std::vector<std::complex<T>> original(n);
  std::vector<std::complex<T>> data(n);

  for (int i = 0; i < n; ++i) {
    original[i] = {static_cast<T>(i), static_cast<T>(i * 0.5)};
    data[i] = original[i];
  }

  FFT::forward(data.data(), n);
  fftSync();
  FFT::inverse(data.data(), n);  // library should normalize by 1/N
  fftSync();

  T err = maxError(data, original);
  // Double 精度更高，Single 精度较低
  T tol = (sizeof(T) == 8) ? 1e-10 : 1e-3;

  bool const pass = err < tol;
  std::ostringstream oss;
  oss << "err=" << std::scientific << err;
  TestPrinter::printTestResult(name, pass, oss.str(), fftBackendLabel());
  assert(pass);
}

/**
 * @brief [Test] R2C/C2R 往返测试 (Round-trip R2C-C2R)
 *
 * 验证实数 FFT 接口的正确性
 * 流程：Real Data -> R2C Forward -> Complex Spectrum (N/2+1) -> C2R Inverse ->
 * Real Data'
 *
 * R2C 变换利用实信号频谱的共轭对称性，只计算前 $N/2+1$ 个点，提升效率
 */
template <typename T>
void testR2cC2r() {
  std::string const name = "R2C-Roundtrip (128pt)";
  const int n = 128;  // must be even for standard R2C
  std::vector<T> input(n);
  for (int i = 0; i < n; ++i) input[i] = static_cast<T>(std::cos(0.5 * i));

  std::vector<std::complex<T>> freq((n / 2) + 1);
  std::vector<T> output(n);

  // R2C: Input (N) -> Freq (N/2 + 1)
  FFT::forward(input.data(), freq.data(), n);
  fftSync();

  // C2R: Freq (N/2 + 1) -> Output (N)
  FFT::inverse(freq.data(), output.data(), n);
  fftSync();

  T err = maxError(input, output);
  T tol = (sizeof(T) == 8) ? 1e-10 : 1e-3;

  bool const pass = err < tol;
  std::ostringstream oss;
  oss << "err=" << std::scientific << err;
  TestPrinter::printTestResult(name, pass, oss.str(), fftBackendLabel());
  assert(pass);
}

/**
 * @brief [Test] R2C/C2R 非原位操作约束测试
 *
 * 验证当前后端在 Out-of-Place 调用模式下检测指针重叠的能力 (如果支持)
 */
template <typename T>
void testR2cC2rOutOfPlace() {
  std::string const name = "R2C/C2R Out-of-Place";
  const int n = 8;
  std::vector<T> raw(n + 2);
  auto* realPtr = raw.data();
  auto* complexPtr = reinterpret_cast<std::complex<T>*>(raw.data());

  bool forwardThrows = false;
  try {
    FFT::forward(realPtr, complexPtr, n);
  } catch (const std::exception& e) {
    forwardThrows = std::string(e.what()).find("out-of-place") != std::string::npos;
  }

  bool inverseThrows = false;
  try {
    FFT::inverse(complexPtr, realPtr, n);
  } catch (const std::exception& e) {
    inverseThrows = std::string(e.what()).find("out-of-place") != std::string::npos;
  }

  bool const pass = forwardThrows && inverseThrows;
  TestPrinter::printTestResult(name, pass, "alias check", fftBackendLabel());
  assert(pass);
}

/**
 * @brief [Test] 批量 FFT 测试 (Batch Transformation)
 *
 * 验证对多帧数据同时进行变换
 * - BatchSize=16, N=64
 */
template <typename T>
void testBatch() {
  std::string const name = "Batch C2C (16x64)";
  int64_t const n = 64;
  int64_t const batch = 16;
  std::vector<std::complex<T>> data(n * batch);
  std::vector<std::complex<T>> original(n * batch);

  for (int i = 0; i < n * batch; ++i) {
    data[i] = {static_cast<T>(i % n), 0};
    original[i] = data[i];
  }

  // sign = -1 for Forward
  FFT::batch(data.data(), n, batch, -1);
  fftSync();
  // sign = 1 for Inverse
  FFT::batch(data.data(), n, batch, 1);
  fftSync();

  T err = maxError(data, original);
  T tol = (sizeof(T) == 8) ? 1e-10 : 1e-3;

  bool const pass = err < tol;
  std::ostringstream oss;
  oss << "err=" << std::scientific << err;
  TestPrinter::printTestResult(name, pass, oss.str(), fftBackendLabel());
  assert(pass);
}

/**
 * @brief [Test] 反复 C2C 正反变换精度漂移测试
 *
 * 对同一缓冲区反复执行 FFT + IFFT，观察误差随迭代次数的增长
 * 该测试默认跳过，可通过设置 PRISM_FFT_DRIFT_TEST=1 启用
 */
template <typename T>
void testIterativeC2CDrift() {
  std::string const name = "C2C Drift";
  if (std::getenv("PRISM_FFT_DRIFT_TEST") == nullptr) {
    TestPrinter::printSkip(name, "set PRISM_FFT_DRIFT_TEST=1");
    return;
  }

  int const n = 1024;
  std::vector<std::complex<T>> data(n);
  std::vector<std::complex<T>> original(n);
  constexpr T piVal = static_cast<T>(M_PI_VAL);

  for (int i = 0; i < n; ++i) {
    T t = static_cast<T>(i) / static_cast<T>(n);
    T re = std::sin(2 * piVal * 5 * t) + (static_cast<T>(0.1) * std::cos(2 * piVal * 13 * t));
    T im = std::cos(2 * piVal * 3 * t) * static_cast<T>(0.25);
    original[i] = {re, im};
    data[i] = original[i];
  }

  std::vector<int> const checkpoints = {1, 10, 100, 500, 1000, 2000, 5000};
  int const maxIters = checkpoints.back();
  size_t checkpointIdx = 0;
  int firstBad = -1;
  T const threshold = (sizeof(T) == 8) ? static_cast<T>(1e-8) : static_cast<T>(1e-3);

  std::ostringstream oss;
  for (int iter = 1; iter <= maxIters; ++iter) {
    FFT::forward(data.data(), n);
    fftSync();
    FFT::inverse(data.data(), n);
    fftSync();
    if (iter == checkpoints[checkpointIdx]) {
      T const err = maxError(data, original);
      if (checkpointIdx > 0) oss << ", ";
      oss << iter << ":" << std::scientific << err;
      if (firstBad < 0 && err > threshold) {
        firstBad = iter;
      }
      ++checkpointIdx;
      if (checkpointIdx >= checkpoints.size()) break;
    }
  }

  if (firstBad >= 0) {
    oss << " | first>th=" << firstBad;
  } else {
    oss << " | first>th=none";
  }
  TestPrinter::printTestResult(name, true, oss.str(), fftBackendLabel());
}

/**
 * @brief [Test] DeviceBuffer dirty flags contract
 *
 * 验证 host/device dirty 标志的互斥与 copy API 的行为
 * 仅在 backend 支持 device buffer 时运行
 */
void testDeviceBufferFlags() {
  std::string const name = "DeviceBuffer Dirty Flags";
  if (!FFT::supportsDeviceBuffer(ScalarType::C32, FftTransType::C2C)) {
    TestPrinter::printSkip(name, "Backend reports no device buffer support");
    return;
  }

  bool pass = true;
  std::string detail;
  try {
    auto buf = FFT::acquireDeviceBuffer(ScalarType::C32, FftTransType::C2C, 8);
    pass &= !buf.host_dirty();
    pass &= !buf.device_dirty();

    buf.set_host_dirty();
    pass &= buf.host_dirty();

    bool threw = false;
    try {
      buf.set_device_dirty();
    } catch (const std::exception&) {
      threw = true;
    }
    pass &= threw;

    buf.set_host_dirty(false);
    buf.set_device_dirty();
    pass &= buf.device_dirty();

    threw = false;
    try {
      buf.set_host_dirty();
    } catch (const std::exception&) {
      threw = true;
    }
    pass &= threw;

    buf.set_device_dirty(false);
    buf.set_host_dirty();
    buf.copy_to_device();
    pass &= !buf.host_dirty();

    buf.set_device_dirty();
    buf.copy_to_host();
    pass &= !buf.device_dirty();
  } catch (const std::exception& e) {
    pass = false;
    detail = e.what();
  }

  TestPrinter::printTestResult(name, pass, detail, fftBackendLabel());
  assert(pass);
}

// ==============================================================================
// 主测试运行器
// ==============================================================================

int main() {
  TestPrinter::printSuiteHeader("FFT Tests");
  TestPrinter::printInfo("Backend", backend::getFftBackend().name());

  // Real32 Tests
  TestPrinter::printSection("FFT Operations (Real32)");
  testImpulseFft<real32_t>();
  testSineFft<real32_t>();
  testRoundtripC2C<real32_t>();
  testR2cC2r<real32_t>();
  testR2cC2rOutOfPlace<real32_t>();
  testBatch<real32_t>();
  testIterativeC2CDrift<real32_t>();
  testDeviceBufferFlags();

  // Real64 Tests (Dynamic check for support)
  TestPrinter::printSection("FFT Operations (Real64)");
  bool const hasC2c64 = FFT::supports(ScalarType::C64, FftTransType::C2C);
  bool const hasR2c64 = FFT::supports(ScalarType::F64, FftTransType::R2C);
  if (hasC2c64 || hasR2c64) {
    try {
      testImpulseFft<real64_t>();
      testSineFft<real64_t>();
      testRoundtripC2C<real64_t>();
      testR2cC2r<real64_t>();
      testR2cC2rOutOfPlace<real64_t>();
      testBatch<real64_t>();
      testIterativeC2CDrift<real64_t>();
    } catch (const std::exception& e) {
      TestPrinter::printTestResult("FFT (Real64) Block", false, e.what(), fftBackendLabel());
    }
  } else {
    TestPrinter::printSkip("FFT (Real64)", "Backend reports no support");
  }

  TestPrinter::printSummary();
  return 0;
}

/// @}
