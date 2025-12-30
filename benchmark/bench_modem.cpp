/**
 * @file bench_modem.cpp
 * @ingroup benchmarks
 * @brief Modem 性能测试
 *
 * 比较 CPU vs GPU，JIT vs AOT
 */

#include <Halide.h>

#include <iomanip>
#include <iostream>
#include <string>

#include "bench_util.h"
#include "prism/dsl/modem.h"
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

/** @brief 打印调制解调基准表头 */
static void printHeader() {
  std::cout << std::setw(18) << "Modem Op" << std::setw(11) << "JIT CPU"
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
static void benchModem(const std::string& name, int size, BuildFunc buildFn) {
  Halide::Buffer<real32_t> buf(size);
  for (int i = 0; i < size; ++i) buf(i) = static_cast<real32_t>(i % 4);

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
  BenchPrinter::printSuiteHeader("Modem");
  BenchPrinter::printBackendInfo();

  constexpr int testSize = 102400;
  printHeader();

  benchModem("Mixer", testSize, [](int s) {
    return modem::mixer(Signal::input(s), 1000.0, 48000.0);
  });

  benchModem("QAMMap (4)", testSize,
             [](int s) { return modem::qamMap(Signal::input(s), 4); });

  benchModem("QAMMap (16)", testSize,
             [](int s) { return modem::qamMap(Signal::input(s), 16); });

  benchModem("QAMDemap (4)", testSize,
             [](int s) { return modem::qamDemap(Signal::input(s), 4); });

  BenchPrinter::printSummary();
  Executor::setMode(ExecMode::AUTO);
  return 0;
}

/// @}
