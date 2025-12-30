/**
 * @file bench_filter.cpp
 * @ingroup benchmarks
 * @brief Filter 性能测试
 *
 * 比较 FIR/IIR/MovingAverage 在 CPU/GPU，JIT/AOT 上的性能
 */

#include <Halide.h>

#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "bench_util.h"
#include "prism/dsl/filter.h"
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

/** @brief 打印滤波基准表头 */
static void printHeader() {
  std::cout << std::setw(18) << "Filter" << std::setw(11) << "JIT CPU"
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
static void benchFilter(const std::string& name, int size, BuildFunc buildFn) {
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
  BenchPrinter::printSuiteHeader("Filter");
  BenchPrinter::printBackendInfo();

  constexpr int benchSize = 102400;
  printHeader();

  // FIR 滤波器测试（不同长度的系数）
  std::vector<real32_t> taps16(16, 1.0F / 16.0F);
  std::vector<real32_t> taps64(64, 1.0F / 64.0F);

  benchFilter("FIR (16 taps)", benchSize,
              [&](int s) { return filter::fir(Signal::input(s), taps16); });

  benchFilter("FIR (64 taps)", benchSize,
              [&](int s) { return filter::fir(Signal::input(s), taps64); });

  // IIR 滤波器测试（Butterworth LPF 系数）
  std::vector<real32_t> iirB = {0.004824F, 0.019297F, 0.028946F, 0.019297F,
                                0.004824F};
  std::vector<real32_t> iirA = {1.0F, -2.369513F, 2.313988F, -1.054665F,
                                0.187379F};

  benchFilter("IIR (4th order)", benchSize,
              [&](int s) { return filter::iir(Signal::input(s), iirB, iirA); });

  // 移动平均测试
  benchFilter("MovingAvg (16)", benchSize, [](int s) {
    return filter::movingAverage(Signal::input(s), 16);
  });

  benchFilter("MovingAvg (64)", benchSize, [](int s) {
    return filter::movingAverage(Signal::input(s), 64);
  });

  // 中值滤波测试
  benchFilter("Median (3)", benchSize,
              [](int s) { return filter::median(Signal::input(s), 3); });

  BenchPrinter::printSummary();
  Executor::setMode(ExecMode::AUTO);
  return 0;
}

/// @}
