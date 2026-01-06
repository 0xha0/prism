/**
 * @file bench_ops.cpp
 * @ingroup benchmarks
 * @brief 基础算子性能基准测试
 *
 * 评估基础数学算子 (Add, Sub, Mul, Div, Scale, Abs) 的性能
 * 这些算子通常是 Memory-Bound (带宽受限) 的，
 * 测试旨在观察不同后端在处理简单向量化操作时的吞吐能力
 */

#include <string>

#include "bench_util.h"
#include "prism/dsl/ops.h"
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

template <typename T, typename BuildFunc>
static void benchOp(const std::string& baseName, int size, int iterations, BuildFunc buildFn) {
  using Traits = BenchTraits<T>;
  auto buf = Traits::makeBuffer(size);
  Traits::fillLinear(buf);

  auto sig = buildFn(size, Traits::scalarType());
  auto times = BenchmarkRunner::runSignalBench<T>(sig, buf, iterations);

  BenchPrinter::printBenchResult(withPrecision<T>(baseName), times);
}

template <typename T>
static void benchOpsForType(int size, int iterations) {
  benchOp<T>("Add", size, iterations, [](int s, ScalarType type) {
    auto a = Signal::input(s, type);
    return add(a, a);
  });

  benchOp<T>("Sub", size, iterations, [](int s, ScalarType type) {
    auto a = Signal::input(s, type);
    return sub(a, a);
  });

  benchOp<T>("Mul", size, iterations, [](int s, ScalarType type) {
    auto a = Signal::input(s, type);
    return mul(a, a);
  });

  benchOp<T>("Div", size, iterations, [](int s, ScalarType type) {
    auto a = Signal::input(s, type);
    return div(a, a);
  });

  benchOp<T>("Scale", size, iterations, [](int s, ScalarType type) {
    return scale(Signal::input(s, type), static_cast<T>(2.5));
  });

  benchOp<T>("Abs", size, iterations,
             [](int s, ScalarType type) { return abs(Signal::input(s, type)); });

  benchOp<T>("Scale+Abs", size, iterations, [](int s, ScalarType type) {
    return abs(scale(Signal::input(s, type), static_cast<T>(2.5)));
  });
}

int main() {
  BenchPrinter::printSuiteHeader("Ops");
  BenchPrinter::printBackendInfo();

  int const testSize = benchSize(102400);
  int const iterations = benchIterations(100);
  BenchPrinter::printBenchHeader("Operation");

  forEachPrecision<real32_t, real64_t>([&](auto tag) {
    using T = typename decltype(tag)::type;
    benchOpsForType<T>(testSize, iterations);
  });

  BenchPrinter::printSummary();
  Executor::setMode(ExecMode::AUTO);
  return 0;
}

/// @}
