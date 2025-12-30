#pragma once

/**
 * @file bench_util.h
 * @ingroup benchmarks
 * @brief Benchmark 辅助工具：计时、打印、后端信息
 */

#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "prism/backend/fft_backend.h"
#include "prism/types.h"

namespace prism::benchmark {

/// @addtogroup benchmarks
/// @{

using real64_t = double;

// ============================================================================
// // 计时工具
// ============================================================================
// //

/**
 * @brief 简单的 benchmark 运行器
 *
 */
class BenchmarkRunner {
 public:
  /**
   * @brief 运行测试函数并返回平均耗时 (ms)
   *
   * @tparam Func 可调用对象
   * @param func 测试函数
   * @param iterations 迭代次数
   * @return real64_t 平均耗时 (ms)
   */
  template <typename Func>
  static real64_t run(Func&& func, int iterations = 100) {
    // Warmup
    func();

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
      func();
    }
    auto end = std::chrono::high_resolution_clock::now();

    return std::chrono::duration<real64_t, std::milli>(end - start).count() /
           iterations;
  }
};

// ============================================================================
// // 打印工具
// ============================================================================
// //

/**
 * @brief 统一的 Benchmark 输出工具
 *
 */
class BenchPrinter {
 public:
  static void printSuiteHeader(const std::string& name) {
    std::cout << "=== PRISM " << name << " Benchmark ===\n\n";
  }

  static void printBackendInfo() {
    auto& fft = backend::getFftBackend();
    std::cout << "FFT Backend: " << fft.name() << "\n";

#ifdef PRISM_GPU_BACKEND_Metal
    std::cout << "GPU Backend: Metal\n";
#elif defined(PRISM_GPU_BACKEND_CUDA)
    std::cout << "GPU Backend: CUDA\n";
#elif defined(PRISM_GPU_BACKEND_OpenCL)
    std::cout << "GPU Backend: OpenCL\n";
#else
    std::cout << "GPU Backend: None\n";
#endif
    std::cout << "\n";
  }

  static void printSection(const std::string& name) {
    std::cout << ">>> " << name << "...\n";
  }

  static void printSeparator(int width = 70) {
    std::cout << std::string(width, '-') << "\n";
  }

  static void printSummary() { std::cout << "\n=== Benchmark Complete ===\n"; }
};

/// @}

}  // namespace prism::benchmark
