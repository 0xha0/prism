/**
 * @file bench_adv_ops.cpp
 * @ingroup benchmarks
 * @brief Advanced ops benchmark (convolve/kron/resample/pack)
 */

#include <Halide.h>

#include <algorithm>
#include <complex>
#include <string>
#include <vector>

#include "bench_util.h"
#include "prism/dsl/ops.h"
#include "prism/dsl/signal.h"
#include "prism/runtime/executor.h"
#include "prism/types.h"

using prism::real32_t;
using prism::real64_t;
using prism::ScalarType;
using prism::ToHalideType;
using namespace prism::dsl;
using namespace prism::runtime;
using namespace prism::benchmark;

/// @addtogroup benchmarks
/// @{

constexpr int LABEL_WIDTH = 32;  // NOLINT

template <typename OutT, typename InT, typename BuildFunc>
static void benchUnary(const std::string& baseName, int size, int iterations, BuildFunc buildFn) {
  using InTraits = BenchTraits<InT>;
  auto buf = InTraits::makeBuffer(size);
  InTraits::fillLinear(buf);

  auto sig = buildFn(size, InTraits::scalarType());
  auto times = BenchmarkRunner::runSignalBench<OutT>(sig, buf, iterations);

  BenchPrinter::printBenchResult(withPrecision<OutT>(baseName), times, LABEL_WIDTH);
}

template <typename OutT, typename InA, typename InB, typename BuildFunc>
static void benchBinary(const std::string& baseName, int aLen, int bLen, int iterations,
                        BuildFunc buildFn) {
  using BufferElemT = typename ToHalideType<OutT>::Type;
  using TraitsA = BenchTraits<InA>;
  using TraitsB = BenchTraits<InB>;

  auto bufA = TraitsA::makeBuffer(aLen);
  auto bufB = TraitsB::makeBuffer(bLen);
  TraitsA::fillLinear(bufA);
  TraitsB::fillLinear(bufB);

  auto sig = buildFn(aLen, bLen, TraitsA::scalarType(), TraitsB::scalarType());
  std::vector<Halide::Buffer<BufferElemT>> const inputs = {bufA, bufB};
  auto times = BenchmarkRunner::runSignalBenchMulti<OutT>(sig, inputs, iterations);

  BenchPrinter::printBenchResult(withPrecision<OutT>(baseName), times, LABEL_WIDTH);
}

template <typename RealT>
static void benchAdvOpsForType(int baseSize, int iterations) {
  using ComplexT = std::complex<RealT>;

  int const convALen = std::max(256, baseSize / 8);
  int const convBLen = std::max(64, baseSize / 32);
  int const kronALen = std::max(128, baseSize / 64);
  int const kronBLen = std::max(64, baseSize / 128);

  benchBinary<RealT, RealT, RealT>("Convolve (R,R)", convALen, convBLen, iterations,
                                   [](int aLen, int bLen, ScalarType aType, ScalarType bType) {
                                     auto a = Signal::input(aLen, aType);
                                     auto b = Signal::input(bLen, bType);
                                     return convolve(a, b);
                                   });

  benchBinary<ComplexT, ComplexT, ComplexT>(
      "Convolve (C,C)", convALen, convBLen, iterations,
      [](int aLen, int bLen, ScalarType aType, ScalarType bType) {
        auto a = Signal::input(aLen, aType);
        auto b = Signal::input(bLen, bType);
        return convolve(a, b);
      });

  benchBinary<RealT, RealT, RealT>("Kron (R,R)", kronALen, kronBLen, iterations,
                                   [](int aLen, int bLen, ScalarType aType, ScalarType bType) {
                                     auto a = Signal::input(aLen, aType);
                                     auto b = Signal::input(bLen, bType);
                                     return kron(a, b);
                                   });

  benchBinary<ComplexT, ComplexT, ComplexT>(
      "Kron (C,C)", kronALen, kronBLen, iterations,
      [](int aLen, int bLen, ScalarType aType, ScalarType bType) {
        auto a = Signal::input(aLen, aType);
        auto b = Signal::input(bLen, bType);
        return kron(a, b);
      });

  benchUnary<RealT, RealT>("Upsample x4", baseSize, iterations, [](int len, ScalarType type) {
    auto x = Signal::input(len, type);
    return upsample(x, 4, 1);
  });

  benchUnary<ComplexT, ComplexT>("Upsample x4", baseSize, iterations, [](int len, ScalarType type) {
    auto x = Signal::input(len, type);
    return upsample(x, 4, 1);
  });

  benchUnary<RealT, RealT>("Downsample /3", baseSize, iterations, [](int len, ScalarType type) {
    auto x = Signal::input(len, type);
    return downsample(x, 3, 1);
  });

  benchUnary<ComplexT, ComplexT>("Downsample /3", baseSize, iterations,
                                 [](int len, ScalarType type) {
                                   auto x = Signal::input(len, type);
                                   return downsample(x, 3, 1);
                                 });

  benchBinary<ComplexT, RealT, RealT>("ComplexPack (I,Q)", baseSize, baseSize, iterations,
                                      [](int lenI, int lenQ, ScalarType typeI, ScalarType typeQ) {
                                        auto i = Signal::input(lenI, typeI);
                                        auto q = Signal::input(lenQ, typeQ);
                                        return complexPack(i, q);
                                      });

  benchUnary<RealT, ComplexT>("Real (DePack)", baseSize, iterations, [](int len, ScalarType type) {
    auto x = Signal::input(len, type);
    return real(x);
  });

  benchUnary<RealT, ComplexT>("Imag (DePack)", baseSize, iterations, [](int len, ScalarType type) {
    auto x = Signal::input(len, type);
    return imag(x);
  });
}

int main() {
  BenchPrinter::printSuiteHeader("Advanced Ops");
  BenchPrinter::printBackendInfo();
  BenchPrinter::printBenchHeader("Operation", LABEL_WIDTH);

  int const baseSize = benchSize(16384);
  int const iterations = benchIterations(50);

  forEachPrecision<real32_t, real64_t>([&](auto tag) {
    using T = typename decltype(tag)::type;
    benchAdvOpsForType<T>(baseSize, iterations);
  });

  BenchPrinter::printSummary();
  Executor::setMode(ExecMode::AUTO);
  return 0;
}

/// @}
