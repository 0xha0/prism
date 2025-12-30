/**
 * @file bench_fft.cpp
 * @ingroup benchmarks
 * @brief FFT 性能测试
 *
 * 测试当前 FFT 后端的极限性能
 */

#include <iomanip>
#include <iostream>
#include <vector>

#include "bench_util.h"
#include "prism/backend/fft_backend.h"
#include "prism/types.h"

using namespace prism;
using namespace prism::benchmark;

/// @addtogroup benchmarks
/// @{

/** @brief 输出表头 */
static void printHeader() {
  std::cout << std::setw(12) << "Size" << std::setw(15) << "C2C (ms)"
            << std::setw(15) << "R2C (ms)" << std::setw(15) << "Throughput"
            << "\n";
  BenchPrinter::printSeparator(55);
}

/** @brief 单个大小的 FFT 基准 */
void benchFft(int size) {
  std::vector<complex32_t> cData(size);
  std::vector<real32_t> rIn(size);
  std::vector<complex32_t> r2cOut((size / 2) + 1);

  for (int i = 0; i < size; ++i) {
    cData[i] = {static_cast<real32_t>(i), 0.0F};
    rIn[i] = static_cast<real32_t>(i);
  }

  auto& fft = backend::getFftBackend();

  real64_t const c2cMs = BenchmarkRunner::run(
      [&]() { fft.forwardC2c<real32_t>(cData.data(), size); });

  real64_t const r2cMs = BenchmarkRunner::run(
      [&]() { fft.forwardR2c<real32_t>(rIn.data(), r2cOut.data(), size); });

  real64_t const throughput = (size * 1e-6) / (c2cMs * 1e-3);  // M samples/s

  std::cout << std::setw(12) << size << std::setw(13) << std::fixed
            << std::setprecision(3) << c2cMs << "ms" << std::setw(13) << r2cMs
            << "ms" << std::setw(12) << std::setprecision(1) << throughput
            << " MS/s"
            << "\n";
}

int main() {
  BenchPrinter::printSuiteHeader("FFT");
  BenchPrinter::printBackendInfo();

  printHeader();

  std::vector<int> const sizes = {256,   1024,   4096,   16384,
                                  65536, 262144, 1048576};
  for (int const size : sizes) {
    benchFft(size);
  }

  BenchPrinter::printSummary();
  return 0;
}

/// @}
