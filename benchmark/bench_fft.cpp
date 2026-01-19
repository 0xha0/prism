/**
 * @file bench_fft.cpp
 * @ingroup benchmarks
 * @brief FFT 变换性能基准测试
 *
 * 直接调用 Runtime 的 FFT 后端接口进行测试
 * 涵盖：
 * - Complex-to-Complex (C2C)
 * - Real-to-Complex (R2C)
 * - Batch C2C
 * - 不同长度 (256 到 1M 点)
 *
 * 用于验证底层 FFT 库 (如 vkFFT, vDSP, cuFFT) 的集成性能
 */

#include <array>
#include <complex>
#include <iomanip>
#include <iostream>
#include <optional>
#include <string>
#include <type_traits>
#include <vector>

#include "bench_util.h"
#include "prism/backend/fft_backend.h"
#include "prism/runtime/fft.h"
#include "prism/types.h"

using namespace prism;
using namespace prism::benchmark;

/// @addtogroup benchmarks
/// @{

namespace {
constexpr int kSizeWidth = 12;
constexpr int kOpWidth = 10;
constexpr int kTimeWidth = 15;
constexpr int kThroughputWidth = 15;
constexpr int kSeparatorWidth = kSizeWidth + kOpWidth + kTimeWidth + kThroughputWidth;
constexpr int kTimeCellWidth = 13;
constexpr int kThroughputCellWidth = 12;
constexpr int kDefaultIterations = 256;
constexpr std::array<int, 6> kDefaultSizes = {256, 1024, 4096, 16384, 65536, 262144};
constexpr std::array<int, 4> kDefaultBatchSizes = {8, 16, 32, 64};
constexpr const char* kOpC2C = "C2C";
constexpr const char* kOpR2C = "R2C";
constexpr const char* kOpC2R = "C2R";

std::string makeBatchOpLabel(const char* base, int batch) {
  return std::string(base) + " x" + std::to_string(batch);
}
}  // namespace

/**
 * @brief 打印 FFT 基准测试的表头
 *
 * 包含 Size, C2C 耗时, R2C 耗时及折算的吞吐率 (Mpt/s)
 */
static void printHeader() {
  std::cout << std::setw(kSizeWidth) << "Size" << std::setw(kOpWidth) << "Op"
            << std::setw(kTimeWidth) << "Time (ms)" << std::setw(kThroughputWidth)
            << "Throughput"
            << "\n";
  BenchPrinter::printSeparator(kSeparatorWidth);
}

static void printCellMs(const std::optional<real64_t>& value) {
  if (value) {
    std::cout << std::setw(kTimeCellWidth) << std::fixed << std::setprecision(3) << *value
              << "ms";
  } else {
    std::cout << std::setw(kTimeCellWidth) << "n/a"
              << "  ";
  }
}

static std::optional<real64_t> toThroughput(int64_t size, const std::optional<real64_t>& ms) {
  if (!ms || *ms <= 0 || size <= 0) return std::nullopt;
  return (size * 1e-6) / (*ms * 1e-3);
}

static void printRow(int size, const std::string& op, const std::optional<real64_t>& timeMs,
                     int64_t throughputSize) {
  std::cout << std::setw(kSizeWidth) << size << std::setw(kOpWidth) << op;
  printCellMs(timeMs);
  auto const throughput = toThroughput(throughputSize, timeMs);
  if (throughput) {
    std::cout << std::setw(kThroughputCellWidth) << std::setprecision(1) << *throughput
              << " Mpt/s";
  } else {
    std::cout << std::setw(kThroughputCellWidth) << "n/a"
              << "  ";
  }
  std::cout << "\n" << std::flush;
}

static void printRow(int size, const std::string& op, const std::optional<real64_t>& timeMs) {
  printRow(size, op, timeMs, size);
}

template <typename RealT>
void benchFftPrecision(const std::vector<int>& sizes, const std::vector<int>& batchSizes,
                       int iterations) {
  using ComplexT = std::complex<RealT>;
  std::string const precision = precisionName<RealT>();
  auto& fft = backend::getFftBackend();

  ScalarType const realType = std::is_same_v<RealT, real64_t> ? ScalarType::F64 : ScalarType::F32;
  ScalarType const complexType =
      std::is_same_v<RealT, real64_t> ? ScalarType::C64 : ScalarType::C32;
  bool const hasC2c = fft.supports(complexType, runtime::FftTransType::C2C);
  bool const hasR2c = fft.supports(realType, runtime::FftTransType::R2C);
  bool const hasC2r = fft.supports(realType, runtime::FftTransType::C2R);
  if (!hasC2c && !hasR2c && !hasC2r) {
    BenchPrinter::printSection("FFT " + precision + " (unsupported)");
    return;
  }

  BenchPrinter::printSection("FFT " + precision);
  printHeader();

  for (int const size : sizes) {
    std::vector<ComplexT> cData(size);
    std::vector<RealT> rIn(size);
    std::vector<ComplexT> r2cOut((size / 2) + 1);
    std::vector<RealT> r2cBack(size);

    for (int i = 0; i < size; ++i) {
      cData[i] = {static_cast<RealT>(i), static_cast<RealT>(0)};
      rIn[i] = static_cast<RealT>(i);
    }

    std::optional<real64_t> c2cMs;
    std::optional<real64_t> r2cMs;
    std::optional<real64_t> c2rMs;
    if (hasC2c) {
      if (runtime::FFT::supportsDeviceBuffer(complexType, runtime::FftTransType::C2C)) {
        auto buffer =
            runtime::FFT::acquireDeviceBuffer(complexType, runtime::FftTransType::C2C, size);
        c2cMs = BenchmarkRunner::runSafe(
            [&]() -> runtime::FFT::DeviceBuffer& {
              runtime::FFT::executeDeviceBuffer(buffer, -1);
              return buffer;
            },
            iterations);
      } else {
        c2cMs = BenchmarkRunner::runSafe([&]() { runtime::FFT::forward(cData.data(), size); },
                                         iterations);
      }
    }
    printRow(size, kOpC2C, c2cMs);
    for (int const batch : batchSizes) {
      if (batch <= 0) continue;
      std::optional<real64_t> batchMs;
      if (hasC2c) {
        if (runtime::FFT::supportsDeviceBuffer(complexType, runtime::FftTransType::C2C)) {
          auto buffer = runtime::FFT::acquireDeviceBuffer(complexType, runtime::FftTransType::C2C,
                                                          size, batch);
          batchMs = BenchmarkRunner::runSafe(
              [&]() -> runtime::FFT::DeviceBuffer& {
                runtime::FFT::executeDeviceBuffer(buffer, -1);
                return buffer;
              },
              iterations);
        } else {
          std::vector<ComplexT> batchData(static_cast<size_t>(size) * batch);
          for (int i = 0; i < size * batch; ++i) {
            batchData[i] = {static_cast<RealT>(i), static_cast<RealT>(0)};
          }
          batchMs = BenchmarkRunner::runSafe(
              [&]() { runtime::FFT::batch(batchData.data(), size, batch, -1); }, iterations);
        }
      }
      printRow(size, makeBatchOpLabel(kOpC2C, batch), batchMs,
               static_cast<int64_t>(size) * batch);
    }
    if (hasR2c) {
      if (runtime::FFT::supportsDeviceBuffer(realType, runtime::FftTransType::R2C)) {
        auto buffer = runtime::FFT::acquireDeviceBuffer(realType, runtime::FftTransType::R2C, size);
        r2cMs = BenchmarkRunner::runSafe(
            [&]() -> runtime::FFT::DeviceBuffer& {
              runtime::FFT::executeDeviceBuffer(buffer, -1);
              return buffer;
            },
            iterations);
      } else {
        r2cMs = BenchmarkRunner::runSafe(
            [&]() { runtime::FFT::forward(rIn.data(), r2cOut.data(), size); }, iterations);
      }
    }
    if (!hasR2c) {
      for (int i = 0; i < static_cast<int>(r2cOut.size()); ++i) {
        r2cOut[i] = {static_cast<RealT>(i), static_cast<RealT>(-i)};
      }
    }
    printRow(size, kOpR2C, r2cMs);
    if (hasC2r) {
      if (runtime::FFT::supportsDeviceBuffer(realType, runtime::FftTransType::C2R)) {
        auto buffer = runtime::FFT::acquireDeviceBuffer(realType, runtime::FftTransType::C2R, size);
        c2rMs = BenchmarkRunner::runSafe(
            [&]() -> runtime::FFT::DeviceBuffer& {
              runtime::FFT::executeDeviceBuffer(buffer, 1);
              return buffer;
            },
            iterations);
      } else {
        c2rMs = BenchmarkRunner::runSafe(
            [&]() { runtime::FFT::inverse(r2cOut.data(), r2cBack.data(), size); }, iterations);
      }
    }
    printRow(size, kOpC2R, c2rMs);
  }
}

int main() {
  BenchPrinter::printSuiteHeader("FFT");
  BenchPrinter::printBackendInfo();

  std::vector<int> sizes(kDefaultSizes.begin(), kDefaultSizes.end());
  sizes = scaledSizes(std::move(sizes));
  std::vector<int> batchSizes(kDefaultBatchSizes.begin(), kDefaultBatchSizes.end());
  batchSizes = scaledSizes(std::move(batchSizes));
  int const iterations = benchIterations(kDefaultIterations);

  forEachPrecision<real32_t, real64_t>([&](auto tag) {
    using T = typename decltype(tag)::type;
    benchFftPrecision<T>(sizes, batchSizes, iterations);
  });

  BenchPrinter::printSummary();
  return 0;
}

/// @}
