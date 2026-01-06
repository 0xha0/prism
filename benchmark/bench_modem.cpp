/**
 * @file bench_modem.cpp
 * @ingroup benchmarks
 * @brief Modem 算子性能基准测试
 *
 * 评估调制解调相关算子的性能，包括:
 * - Mixer (数字混频/变频)
 * - QAM Mapper/Demapper (正交幅度调制映射/解映射)
 * - PSK Mapper/Demapper (相移键控映射/解映射)
 *
 * 对比 JIT-CPU、AOT-CPU 与 AOT-GPU 的性能差异
 */

#include <complex>
#include <string>

#include "bench_util.h"
#include "prism/dsl/modem.h"
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

constexpr int kLabelWidth = 26;

template <typename OutT, typename InT, typename BuildFunc>
static void benchModem(const std::string& baseName, int size, int iterations, BuildFunc buildFn) {
  using InTraits = BenchTraits<InT>;
  auto buf = InTraits::makeBuffer(size);
  InTraits::fillLinear(buf);

  auto sig = buildFn(size, InTraits::scalarType());
  auto times = BenchmarkRunner::runSignalBench<OutT>(sig, buf, iterations);

  BenchPrinter::printBenchResult(withPrecision<OutT>(baseName), times, kLabelWidth);
}

template <typename RealT>
static void benchModemForType(int size, int iterations) {
  using ComplexT = std::complex<RealT>;

  benchModem<RealT, RealT>("Mixer", size, iterations, [](int s, ScalarType type) {
    return modem::mixer(Signal::input(s, type), 1000.0, 48000.0);
  });

  benchModem<ComplexT, RealT>("BPSKMap", size, iterations, [](int s, ScalarType type) {
    return modem::pskMap(Signal::input(s, type), 2);
  });

  benchModem<ComplexT, RealT>("QPSKMap", size, iterations, [](int s, ScalarType type) {
    return modem::qamMap(Signal::input(s, type), 4);
  });

  benchModem<ComplexT, RealT>("8PSKMap", size, iterations, [](int s, ScalarType type) {
    return modem::pskMap(Signal::input(s, type), 8);
  });

  benchModem<RealT, ComplexT>("BPSKDemap", size, iterations, [](int s, ScalarType type) {
    return modem::pskDemap(Signal::input(s, type), 2);
  });

  benchModem<RealT, ComplexT>("QSKDemap", size, iterations, [](int s, ScalarType type) {
    return modem::pskDemap(Signal::input(s, type), 4);
  });

  benchModem<RealT, ComplexT>("8PSKDemap", size, iterations, [](int s, ScalarType type) {
    return modem::pskDemap(Signal::input(s, type), 8);
  });

  benchModem<ComplexT, RealT>("16QAMMap", size, iterations, [](int s, ScalarType type) {
    return modem::qamMap(Signal::input(s, type), 16);
  });

  benchModem<ComplexT, RealT>("64QAMMap", size, iterations, [](int s, ScalarType type) {
    return modem::qamMap(Signal::input(s, type), 64);
  });

  benchModem<ComplexT, RealT>("128QAMMap", size, iterations, [](int s, ScalarType type) {
    return modem::qamMap(Signal::input(s, type), 128);
  });

  benchModem<RealT, ComplexT>("16QAMDemap", size, iterations, [](int s, ScalarType type) {
    return modem::qamDemap(Signal::input(s, type), 16);
  });

  benchModem<RealT, ComplexT>("64QAMDemap", size, iterations, [](int s, ScalarType type) {
    return modem::qamDemap(Signal::input(s, type), 64);
  });

  benchModem<RealT, ComplexT>("128QAMDemap", size, iterations, [](int s, ScalarType type) {
    return modem::qamDemap(Signal::input(s, type), 128);
  });
}

int main() {
  BenchPrinter::printSuiteHeader("Modem");
  BenchPrinter::printBackendInfo();

  int const testSize = benchSize(102400);
  int const iterations = benchIterations(100);
  BenchPrinter::printBenchHeader("Modem Op", kLabelWidth);

  forEachPrecision<real32_t, real64_t>([&](auto tag) {
    using T = typename decltype(tag)::type;
    benchModemForType<T>(testSize, iterations);
  });

  BenchPrinter::printSummary();
  Executor::setMode(ExecMode::AUTO);
  return 0;
}

/// @}
