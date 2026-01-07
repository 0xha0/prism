/**
 * @file bench_filter.cpp
 * @ingroup benchmarks
 * @brief Filter 算子性能基准测试
 *
 * 专门针对 FIR、Moving Average、Median 等滤波器算子进行性能评估
 * 对比维度包括：
 * - 运行模式：JIT (即时编译) vs AOT (预编译)
 * - 硬件后端：CPU (Host) vs GPU (Compute Shader/Kernel)
 * - 滤波器规格：不同 Taps 长度、窗口大小
 */

#include <string>
#include <vector>

#include "bench_util.h"
#include "prism/dsl/filter.h"
#include "prism/dsl/signal.h"
#include "prism/runtime/executor.h"
#include "prism/types.h"

using prism::real32_t;
using prism::real64_t;
using prism::ScalarType;
using namespace prism::dsl;
using namespace prism::runtime;
using namespace prism::benchmark;

/// @addtogroup benchmarks
/// @{

constexpr int LABEL_WIDTH = 26;

/**
 * @brief 执行单个滤波器的三项基准测试
 * @tparam BuildFunc 用于构建 DSL Signal 的 Lambda 类型
 * @param baseName 测试项目名称
 * @param size 数据规模 (样本数)
 * @param iterations 迭代次数
 * @param buildFn 构建 DSL 的回调函数
 *
 * 该函数会依次执行以下步骤：
 * 1. 准备输入数据
 * 2. 构建 DSL 计算图
 * 3. 运行 JIT-CPU 并在 BenchmarkRunner 中统计耗时
 * 4. 运行 AOT-CPU/AOT-GPU 并在 BenchmarkRunner 中统计耗时
 * 5. 打印汇总结果
 */
template <typename T, typename BuildFunc>
static void benchFilter(const std::string& baseName, int size, int iterations, BuildFunc buildFn) {
  using Traits = BenchTraits<T>;
  auto buf = Traits::makeBuffer(size);
  Traits::fillLinear(buf);

  auto sig = buildFn(size, Traits::scalarType());
  auto times = BenchmarkRunner::runSignalBench<T>(sig, buf, iterations);

  BenchPrinter::printBenchResult(withPrecision<T>(baseName), times, LABEL_WIDTH);
}

template <typename T>
static void benchFiltersForType(int size, int iterations) {
  std::vector<T> taps16(16, static_cast<T>(1.0) / static_cast<T>(16.0));
  std::vector<T> taps64(64, static_cast<T>(1.0) / static_cast<T>(64.0));

  benchFilter<T>("FIR (16 taps)", size, iterations, [&](int s, ScalarType type) {
    return filter::fir(Signal::input(s, type), taps16);
  });

  benchFilter<T>("FIR (64 taps)", size, iterations, [&](int s, ScalarType type) {
    return filter::fir(Signal::input(s, type), taps64);
  });

  benchFilter<T>("MovingAvg (16)", size, iterations, [](int s, ScalarType type) {
    return filter::movingAverage(Signal::input(s, type), 16);
  });

  benchFilter<T>("MovingAvg (64)", size, iterations, [](int s, ScalarType type) {
    return filter::movingAverage(Signal::input(s, type), 64);
  });

  benchFilter<T>("Median (5)", size, iterations,
                 [](int s, ScalarType type) { return filter::median(Signal::input(s, type), 5); });

  benchFilter<T>("Median (7)", size, iterations,
                 [](int s, ScalarType type) { return filter::median(Signal::input(s, type), 7); });
}

int main() {
  BenchPrinter::printSuiteHeader("Filter Benchmark");
  BenchPrinter::printBackendInfo();

  int const benchSizeValue = benchSize(102400);
  int const iterations = benchIterations(100);
  BenchPrinter::printBenchHeader("Filter", LABEL_WIDTH);

  forEachPrecision<real32_t, real64_t>([&](auto tag) {
    using T = typename decltype(tag)::type;
    benchFiltersForType<T>(benchSizeValue, iterations);
  });

  BenchPrinter::printSummary();
  Executor::setMode(ExecMode::AUTO);
  return 0;
}

/// @}

/// @}
