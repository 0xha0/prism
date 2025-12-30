/**
 * @file bench_ops.cpp
 * @ingroup benchmarks
 * @brief DSL 全部算子性能测试
 *
 * 比较 CPU vs GPU，JIT vs AOT
 */

#include <Halide.h>

#include <iomanip>
#include <iostream>
#include <string>

#include "bench_util.h"
#include "prism/dsl/ops.h"
#include "prism/dsl/signal.h"
#include "prism/runtime/executor.h"
#include "prism/types.h"

using prism::real32_t;
using prism::real64_t;
using namespace prism::dsl;
using namespace prism::runtime;
using namespace prism::benchmark;

/// @addtogroup benchmarks
/// @{

/** @brief 打印操作基准表头 */
static void printHeader() {
  std::cout << std::setw(18) << "Operation" << std::setw(11) << "JIT CPU"
            << std::setw(11) << "JIT GPU" << std::setw(11) << "AOT CPU"
            << std::setw(11) << "AOT GPU"
            << "\n";
  BenchPrinter::printSeparator(62);
}

static void printResult(const std::string& name, real64_t jc, real64_t jg,
                        real64_t ac, real64_t ag) {
  std::cout << std::setw(18) << name << std::fixed << std::setprecision(3);
  std::cout << std::setw(9) << jc << "ms";
  std::cout << std::setw(9) << (jg > 0 ? jg : 0) << (jg > 0 ? "ms" : "  ");
  std::cout << std::setw(9) << ac << "ms";
  std::cout << std::setw(9) << (ag > 0 ? ag : 0) << (ag > 0 ? "ms" : "  ");
  std::cout << "\n";
}

template <typename BuildFunc>
static void benchOp(const std::string& name, int size, BuildFunc buildFn) {
  Halide::Buffer<real32_t> buf(size);
  for (int i = 0; i < size; ++i) buf(i) = static_cast<real32_t>(i);

  auto sig = buildFn(size);

  Executor::setMode(ExecMode::CPU);
  real64_t const jc =
      BenchmarkRunner::run([&]() { Executor::run<real32_t>(sig, buf); });

  Executor::setMode(ExecMode::GPU);
  real64_t const jg =
      BenchmarkRunner::run([&]() { Executor::run<real32_t>(sig, buf); });

  auto pc = Executor::compile<real32_t>(sig, ExecMode::CPU);
  real64_t const ac = BenchmarkRunner::run([&]() { pc.run(buf); });

  auto pg = Executor::compile<real32_t>(sig, ExecMode::GPU);
  real64_t const ag = BenchmarkRunner::run([&]() { pg.run(buf); });

  printResult(name, jc, jg, ac, ag);
}

int main() {
  BenchPrinter::printSuiteHeader("Ops");
  BenchPrinter::printBackendInfo();

  constexpr int testSize = 102400;
  printHeader();

  benchOp("Add", testSize, [](int s) {
    auto a = Signal::input(s);
    auto b = Signal::input(s);
    return add(a, b);
  });

  benchOp("Sub", testSize, [](int s) {
    auto a = Signal::input(s);
    auto b = Signal::input(s);
    return sub(a, b);
  });

  benchOp("Mul", testSize, [](int s) {
    auto a = Signal::input(s);
    auto b = Signal::input(s);
    return mul(a, b);
  });

  benchOp("Div", testSize, [](int s) {
    auto a = Signal::input(s);
    auto b = Signal::input(s);
    return div(a, b);
  });

  benchOp("Scale", testSize,
          [](int s) { return scale(Signal::input(s), 2.5); });

  benchOp("Abs", testSize, [](int s) { return abs(Signal::input(s)); });

  benchOp("Scale+Abs", testSize,
          [](int s) { return abs(scale(Signal::input(s), 2.5)); });

  BenchPrinter::printSummary();
  Executor::setMode(ExecMode::AUTO);
  return 0;
}

/// @}
