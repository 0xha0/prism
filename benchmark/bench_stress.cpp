/**
 * @file bench_stress.cpp
 * @ingroup benchmarks
 * @brief GPU 压力测试
 *
 * 包含 DSL、零拷贝 FFT 和 memcpy FFT 的完整测试
 */

#include <Halide.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <vector>

#include "bench_util.h"
#include "prism/backend/fft_backend.h"
#include "prism/dsl/filter.h"
#include "prism/dsl/ops.h"
#include "prism/dsl/signal.h"
#include "prism/runtime/executor.h"
#include "prism/types.h"

using namespace prism;
using namespace prism::dsl;
using namespace prism::runtime;
using namespace prism::benchmark;
using namespace std::chrono;

/// @addtogroup benchmarks
/// @{

int main() {
  BenchPrinter::printSuiteHeader("GPU Stress");
  BenchPrinter::printBackendInfo();

  auto& fft = backend::getFftBackend();

  // ========== 参数 ==========
  constexpr int dslSize = 8 * 1024 * 1024;
  constexpr int dslIterations = 1024;

  constexpr int fftSize = 16 * 1024;
  constexpr int fftBatch = 8192;
  constexpr int fftIters = 1024;

  std::cout << "DSL: " << dslSize / 1024 / 1024 << "M samples × "
            << dslIterations << " iters\n";
  std::cout << "FFT: " << fftSize << " × " << fftBatch << " batch × "
            << fftIters << " iters\n\n";

  // ========== DSL 压力测试 ==========
  BenchPrinter::printSection("DSL GPU Stress Test");

  Halide::Buffer<real32_t> buf(dslSize);
  for (int i = 0; i < dslSize; ++i) {
    buf(i) = static_cast<real32_t>(i % 1000) / 1000.0F;
  }

  auto x = Signal::input(dslSize);
  std::vector<real32_t> const taps(64, 1.0F / 64.0F);
  auto y = abs(scale(filter::fir(x, taps), 2.0));

  std::cout << "  编译 GPU Pipeline...\n";
  auto pipeline = Executor::compile<real32_t>(y, ExecMode::GPU);

  std::cout << "  运行 " << dslIterations << " 次迭代...\n";
  auto dslStart = high_resolution_clock::now();

  for (int i = 0; i < dslIterations; ++i) {
    auto result = pipeline.run(buf);
    if (i % 10 == 0) {
      std::cout << "    进度: " << i << "/" << dslIterations << "\r"
                << std::flush;
    }
  }

  auto dslEnd = high_resolution_clock::now();
  real64_t const dslElapsed = duration<real64_t>(dslEnd - dslStart).count();
  real64_t const dslThroughput =
      (static_cast<real64_t>(dslSize) * dslIterations) / dslElapsed / 1e9;

  std::cout << "\n  DSL 完成: " << std::fixed << std::setprecision(2)
            << dslElapsed << "s, " << dslThroughput << " Gsamples/s\n\n";

  // ========== 零拷贝 FFT 压力测试 ==========
  BenchPrinter::printSection("Zero-Copy FFT GPU Stress Test");

  complex32_t* gpuBuffer = fft.getGpuBuffer(fftSize, fftBatch);
  if (!gpuBuffer) {
    std::cout << "  零拷贝不支持，跳过\n\n";
  } else {
    for (size_t i = 0; i < static_cast<size_t>(fftSize) * fftBatch; ++i) {
      gpuBuffer[i] = {static_cast<real32_t>(i % 1000) / 1000.0F, 0.0F};
    }

    std::cout << "  使用零拷贝 GPU Buffer\n";
    std::cout << "  运行 " << fftIters << " 次 FFT...\n";

    auto fftStart = high_resolution_clock::now();

    for (int i = 0; i < fftIters; ++i) {
      fft.executeOnBuffer(fftSize, fftBatch, -1);
      if (i % 10 == 0) {
        std::cout << "    进度: " << i << "/" << fftIters << "\r" << std::flush;
      }
    }

    auto fftEnd = high_resolution_clock::now();
    real64_t const fftElapsed = duration<real64_t>(fftEnd - fftStart).count();
    int64_t const totalSamples =
        static_cast<int64_t>(fftSize) * fftBatch * fftIters;
    real64_t const fftThroughput = totalSamples / fftElapsed / 1e9;

    std::cout << "\n  零拷贝 FFT 完成: " << std::fixed << std::setprecision(2)
              << fftElapsed << "s, " << fftThroughput << " Gsamples/s\n\n";
  }

  // ========== 对比：带 memcpy 的 FFT ==========
  BenchPrinter::printSection("Memcpy FFT Benchmark");

  std::vector<complex32_t> cpuData(static_cast<size_t>(fftSize) * fftBatch);
  for (size_t i = 0; i < cpuData.size(); ++i) {
    cpuData[i] = {static_cast<real32_t>(i % 1000) / 1000.0F, 0.0F};
  }

  auto memcpyStart = high_resolution_clock::now();

  for (int i = 0; i < fftIters; ++i) {
    fft.batchC2c<real32_t>(cpuData.data(), fftSize, fftBatch, -1);
    if (i % 10 == 0) {
      std::cout << "    进度: " << i << "/" << fftIters << "\r" << std::flush;
    }
  }

  auto memcpyEnd = high_resolution_clock::now();
  real64_t const memcpyElapsed =
      duration<real64_t>(memcpyEnd - memcpyStart).count();
  real64_t const memcpyThroughput =
      (static_cast<real64_t>(fftSize) * fftBatch * fftIters) / memcpyElapsed /
      1e9;

  std::cout << "\n  带 memcpy: " << std::fixed << std::setprecision(2)
            << memcpyElapsed << "s, " << memcpyThroughput << " Gsamples/s\n\n";

  // ========== 汇总 ==========
  BenchPrinter::printSummary();
  std::cout << "DSL 吞吐:      " << dslThroughput << " Gsamples/s\n";
  std::cout << "FFT 零拷贝:    高吞吐（GPU 持续计算）\n";
  std::cout << "FFT 带 memcpy: " << memcpyThroughput << " Gsamples/s\n";

  return 0;
}

/// @}
