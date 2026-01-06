#pragma once

/**
 * @file bench_util.h
 * @ingroup benchmarks
 * @brief Benchmark 辅助工具库
 *
 * 提供标准化的基准测试基础设施，包括：
 * - 计时器 (BenchmarkRunner)：支持自动检测 GPU 同步需求 (device_sync)
 * - 打印器 (BenchPrinter)：统一的格式化输出、后端信息展示
 * - 类型别名与辅助 SFINAE 模板
 */

#include <Halide.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <optional>
#include <string>
#include <type_traits>
#include <vector>

#include "prism/backend/fft_backend.h"
#include "prism/runtime/executor.h"
#include "prism/types.h"

namespace prism::benchmark {

/// @addtogroup benchmarks
/// @{

using real64_t = double;
using runtime::ExecMode;
using runtime::Executor;

// ============================================================================
// Precision Helpers
// ============================================================================

template <typename T>
struct TypeTag {
  using type = T;
};

template <typename... Ts, typename Func>
void forEachPrecision(Func&& func) {
  (func(TypeTag<Ts>{}), ...);
}

template <typename T>
constexpr const char* precisionName() {
  if constexpr (std::is_same_v<T, real32_t>) return "F32";
  if constexpr (std::is_same_v<T, real64_t>) return "F64";
  if constexpr (std::is_same_v<T, complex32_t>) return "C32";
  if constexpr (std::is_same_v<T, complex64_t>) return "C64";
  return "Unknown";
}

template <typename T>
std::string withPrecision(const std::string& base) {
  return base + " (" + precisionName<T>() + ")";
}

// ============================================================================
// Benchmark Config
// ============================================================================

inline int getEnvInt(const char* name, int defaultValue, bool* overridden = nullptr) {
  if (overridden) *overridden = false;
  if (const char* env = std::getenv(name)) {
    char* end = nullptr;
    long const value = std::strtol(env, &end, 10);
    if (end != env) {
      if (overridden) *overridden = true;
      return static_cast<int>(value);
    }
  }
  return defaultValue;
}

inline int benchScale() {
  int const scale = getEnvInt("PRISM_BENCH_SCALE", 1);
  return std::max(1, scale);
}

inline int scaleValue(int value) {
  int const scale = benchScale();
  if (scale <= 1) return value;
  return std::max(1, value / scale);
}

inline int benchValue(const char* envName, int defaultValue) {
  bool overridden = false;
  int value = getEnvInt(envName, defaultValue, &overridden);
  if (!overridden) value = scaleValue(value);
  return value;
}

inline int benchSize(int defaultSize) { return benchValue("PRISM_BENCH_SIZE", defaultSize); }

inline int benchIterations(int defaultIters) {
  return benchValue("PRISM_BENCH_ITERS", defaultIters);
}

inline bool benchSkipGpu() { return getEnvInt("PRISM_BENCH_SKIP_GPU", 0) != 0; }

inline std::vector<int> scaledSizes(std::vector<int> sizes) {
  int const scale = benchScale();
  if (scale > 1) {
    for (auto& size : sizes) {
      size = std::max(1, size / scale);
    }
  }
  std::sort(sizes.begin(), sizes.end());
  sizes.erase(std::unique(sizes.begin(), sizes.end()), sizes.end());
  return sizes;
}

// ============================================================================
// Buffer Helpers
// ============================================================================

template <typename T>
struct BenchTraits {
  using BufferElemType = typename ToHalideType<T>::Type;
  static constexpr bool isComplex = IS_COMPLEX_V<T>;

  static ScalarType scalarType() { return getScalarType<T>(); }

  static Halide::Buffer<BufferElemType> makeBuffer(size_t len) {
    if constexpr (isComplex) {
      return Halide::Buffer<BufferElemType>(2, static_cast<int>(len));
    } else {
      return Halide::Buffer<BufferElemType>(static_cast<int>(len));
    }
  }

  static void fillLinear(Halide::Buffer<BufferElemType>& buf) {
    constexpr BufferElemType kScale = static_cast<BufferElemType>(0.001);
    if constexpr (isComplex) {
      int const len = buf.dim(1).extent();
      for (int i = 0; i < len; ++i) {
        auto val = static_cast<BufferElemType>(i % 1024) * kScale;
        buf(0, i) = val;
        buf(1, i) = -val;
      }
    } else {
      int const len = buf.width();
      for (int i = 0; i < len; ++i) {
        buf(i) = static_cast<BufferElemType>(i % 1024) * kScale;
      }
    }
  }
};

// ============================================================================
// Benchmark Results
// ============================================================================

struct BenchTimes {
  std::optional<real64_t> jit;
  std::optional<real64_t> aotCpu;
  std::optional<real64_t> aotGpu;
};

// ============================================================================
// 计时工具 (Timing Utils)
// ============================================================================

// SFINAE helper to detect .device_sync()
template <typename T, typename = void>
struct has_device_sync : std::false_type {};

template <typename T>
struct has_device_sync<T, std::void_t<decltype(std::declval<T>().device_sync())>> : std::true_type {
};

/**
 * @brief 简单的 benchmark 运行器
 *
 * 封装了循环执行、预热 (Warmup) 和平均耗时计算逻辑
 */
class BenchmarkRunner {
 public:
  /**
   * @brief 运行测试函数并返回平均耗时 (ms)
   *
   * 自动处理 GPU buffer 的同步问题：如果 Func 返回值具有 device_sync() 方法，
   * 则会在计时结束前调用它，确保 GPU 任务完成
   *
   * @tparam Func 可调用对象，通常为 lambda
   * @param func 待测目标函数
   * @param iterations 迭代次数 (默认100)
   * @return real64_t 平均单次执行耗时 (毫秒)
   */
  template <typename Func>
  static real64_t run(Func&& func, int iterations = 100) {
    using ResultType = std::invoke_result_t<Func>;
    constexpr bool canSync = has_device_sync<ResultType>::value;

    // Warmup (预热以消除首次加载开销)
    if constexpr (canSync) {
      func().device_sync();
    } else {
      func();
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
      if constexpr (canSync) {
        // 仅在最后一次迭代强制同步，或者每次都同步？
        // 为了测量吞吐率，通常允许 GPU 队列堆积，仅最后同步
        // 但为了更精确的单次延迟测量，可能需要每次同步
        // 此处策略：在最后一次迭代强制同步，将总时间除以次数
        if (i == iterations - 1) {
          func().device_sync();
        } else {
          func();
        }
      } else {
        func();
      }
    }
    auto end = std::chrono::high_resolution_clock::now();

    return std::chrono::duration<real64_t, std::milli>(end - start).count() / iterations;
  }

  template <typename Func>
  static std::optional<real64_t> runSafe(Func&& func, int iterations = 100) {
    int const iters = std::max(1, iterations);
    try {
      return run(std::forward<Func>(func), iters);
    } catch (const Halide::CompileError&) {
      return std::nullopt;
    } catch (const Halide::RuntimeError&) {
      return std::nullopt;
    } catch (const std::exception&) {
      return std::nullopt;
    } catch (...) {
      return std::nullopt;
    }
  }

  template <typename T, typename SignalT, typename BufferT>
  static BenchTimes runSignalBench(const SignalT& signal, BufferT& input, int iterations = 100) {
    BenchTimes times;
    bool const skipGpu = benchSkipGpu();

    Executor::setMode(ExecMode::CPU);
    times.jit = runSafe([&]() { return Executor::run<T>(signal, input); }, iterations);
    Executor::setMode(ExecMode::AUTO);

    try {
      auto pipeline = Executor::compile<T>(signal, ExecMode::CPU);
      times.aotCpu = runSafe([&]() { return pipeline.run(input); }, iterations);
    } catch (const Halide::CompileError&) {
      times.aotCpu = std::nullopt;
    } catch (const Halide::RuntimeError&) {
      times.aotCpu = std::nullopt;
    } catch (const std::exception&) {
      times.aotCpu = std::nullopt;
    }

    if (!skipGpu) {
      try {
        auto pipeline = Executor::compile<T>(signal, ExecMode::GPU);
        times.aotGpu = runSafe([&]() { return pipeline.run(input); }, iterations);
      } catch (const Halide::CompileError&) {
        times.aotGpu = std::nullopt;
      } catch (const Halide::RuntimeError&) {
        times.aotGpu = std::nullopt;
      } catch (const std::exception&) {
        times.aotGpu = std::nullopt;
      }
    }

    return times;
  }

  template <typename T, typename SignalT, typename BufferT>
  static BenchTimes runSignalBenchMulti(const SignalT& signal, const std::vector<BufferT>& inputs,
                                        int iterations = 100) {
    BenchTimes times;
    bool const skipGpu = benchSkipGpu();

    Executor::setMode(ExecMode::CPU);
    times.jit = runSafe([&]() { return Executor::run<T>(signal, inputs); }, iterations);

    try {
      auto pipeline = Executor::compile<T>(signal, ExecMode::CPU);
      times.aotCpu = runSafe([&]() { return pipeline.run(inputs); }, iterations);
    } catch (const Halide::CompileError&) {
      times.aotCpu = std::nullopt;
    } catch (const Halide::RuntimeError&) {
      times.aotCpu = std::nullopt;
    } catch (const std::exception&) {
      times.aotCpu = std::nullopt;
    }

    if (!skipGpu) {
      try {
        auto pipeline = Executor::compile<T>(signal, ExecMode::GPU);
        times.aotGpu = runSafe([&]() { return pipeline.run(inputs); }, iterations);
      } catch (const Halide::CompileError&) {
        times.aotGpu = std::nullopt;
      } catch (const Halide::RuntimeError&) {
        times.aotGpu = std::nullopt;
      } catch (const std::exception&) {
        times.aotGpu = std::nullopt;
      }
    }

    Executor::setMode(ExecMode::AUTO);
    return times;
  }
};

// ============================================================================
// 打印工具 (Printing Utils)
// ============================================================================

/**
 * @brief 统一的 Benchmark 输出格式化工具
 *
 * 负责打印表头、分节符、后端配置信息等，保持所有 benchmark 输出风格一致
 */
class BenchPrinter {
 public:
  /** @brief 打印套件标题 */
  static void printSuiteHeader(const std::string& name) {
    std::cout << "=== PRISM " << name << " Benchmark ===\n\n";
  }

  /** @brief 打印当前侦测到的计算后端信息 (FFT/GPU) */
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

  /** @brief 打印章节标题 */
  static void printSection(const std::string& name) { std::cout << ">>> " << name << "...\n"; }

  /** @brief 打印分隔线 */
  static void printSeparator(int width = 70) { std::cout << std::string(width, '-') << "\n"; }

  /** @brief 打印结束摘要 */
  static void printSummary() { std::cout << "\n=== Benchmark Complete ===\n"; }

  static void printBenchHeader(const std::string& label, int labelWidth = 18) {
    std::cout << std::setw(labelWidth) << label << std::setw(11) << "JIT CPU" << std::setw(11)
              << "AOT CPU" << std::setw(11) << "AOT GPU"
              << "\n";
    printSeparator(labelWidth + 33);
  }

  static void printBenchResult(const std::string& name, const BenchTimes& times,
                               int labelWidth = 18) {
    std::cout << std::setw(labelWidth) << name << std::fixed << std::setprecision(3);
    printTimeCell(times.jit);
    printTimeCell(times.aotCpu);
    printTimeCell(times.aotGpu);
    std::cout << "\n";
  }

 private:
  static void printTimeCell(const std::optional<real64_t>& value) {
    if (value) {
      std::cout << std::setw(9) << *value << "ms";
    } else {
      std::cout << std::setw(9) << "n/a" << "  ";
    }
  }
};

/// @}

}  // namespace prism::benchmark
