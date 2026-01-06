/**
 * @file bench_stress.cpp
 * @ingroup benchmarks
 * @brief AOT 压力测试 (DSSS 链路 + FFT)
 */

#include <Halide.h>

#include <complex>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

#include "bench_util.h"
#include "prism/dsl/filter.h"
#include "prism/dsl/modem.h"
#include "prism/dsl/ops.h"
#include "prism/dsl/signal.h"
#include "prism/runtime/executor.h"
#include "prism/runtime/fft.h"
#include "prism/types.h"

using prism::getScalarType;
using prism::real32_t;
using prism::real64_t;
using prism::ScalarType;
using namespace prism::dsl;
using namespace prism::runtime;
using namespace prism::benchmark;

/// @addtogroup benchmarks
/// @{

constexpr int kLabelWidth = 40;  // NOLINT
constexpr int kQamOrder = 64;    // NOLINT

struct AotRates {
  std::optional<real64_t> cpu;
  std::optional<real64_t> gpu;
};

struct SingleRates {
  std::optional<real64_t> value;
};

static void printAotHeader(const std::string& label, int width = kLabelWidth) {
  std::cout << std::setw(width) << label << std::setw(13) << "AOT CPU" << std::setw(13) << "AOT GPU"
            << "\n";
  BenchPrinter::printSeparator(width + 26);
}

static void printRateCell(const std::optional<real64_t>& value) {
  if (value) {
    std::cout << std::setw(9) << std::fixed << std::setprecision(3) << *value << "MS/s";
  } else {
    std::cout << std::setw(9) << "n/a"
              << "  ";
  }
}

static void printAotResult(const std::string& name, const AotRates& rates,
                           int width = kLabelWidth) {
  std::cout << std::setw(width) << name;
  printRateCell(rates.cpu);
  printRateCell(rates.gpu);
  std::cout << "\n";
}

static void printSingleHeader(const std::string& label, const std::string& col,
                              int width = kLabelWidth) {
  std::cout << std::setw(width) << label << std::setw(13) << col << "\n";
  BenchPrinter::printSeparator(width + 13);
}

static void printSingleResult(const std::string& name, const SingleRates& rates,
                              int width = kLabelWidth) {
  std::cout << std::setw(width) << name;
  printRateCell(rates.value);
  std::cout << "\n";
}

static std::optional<real64_t> toThroughput(const std::optional<real64_t>& ms, int64_t samples) {
  if (!ms || *ms <= 0 || samples <= 0) return std::nullopt;
  return (static_cast<real64_t>(samples) * 1e-6) / (*ms * 1e-3);
}

template <typename OutT, typename BufferT>
AotRates runAot(const Signal& signal, const BufferT& input, int64_t samples, int iterations) {
  AotRates rates;
  try {
    auto pipeline = Executor::compile<OutT>(signal, ExecMode::CPU);
    auto const ms = BenchmarkRunner::runSafe([&]() { return pipeline.run(input); }, iterations);
    rates.cpu = toThroughput(ms, samples);
  } catch (const Halide::CompileError&) {
    rates.cpu = std::nullopt;
  } catch (const Halide::RuntimeError&) {
    rates.cpu = std::nullopt;
  } catch (const std::exception&) {
    rates.cpu = std::nullopt;
  }

  if (!benchSkipGpu()) {
    try {
      auto pipeline = Executor::compile<OutT>(signal, ExecMode::GPU);
      auto gpuInput = input;
      gpuInput.copy_to_device();
      gpuInput.set_host_dirty(false);
      auto const ms =
          BenchmarkRunner::runSafe([&]() { return pipeline.run(gpuInput); }, iterations);
      rates.gpu = toThroughput(ms, samples);
    } catch (const Halide::CompileError&) {
      rates.gpu = std::nullopt;
    } catch (const Halide::RuntimeError&) {
      rates.gpu = std::nullopt;
    } catch (const std::exception&) {
      rates.gpu = std::nullopt;
    }
  }
  Executor::setMode(ExecMode::AUTO);
  return rates;
}

template <typename RealT>
std::vector<RealT> makeUnitTaps(int taps) {
  std::vector<RealT> coeffs(static_cast<size_t>(taps));
  RealT const scale = static_cast<RealT>(1.0) / static_cast<RealT>(taps);
  for (int i = 0; i < taps; ++i) {
    coeffs[static_cast<size_t>(i)] = scale;
  }
  return coeffs;
}

template <typename RealT>
std::vector<RealT> makeSpreadingCode(int len) {
  std::vector<RealT> code(static_cast<size_t>(len));
  for (int i = 0; i < len; ++i) {
    code[static_cast<size_t>(i)] = (i % 2 == 0) ? static_cast<RealT>(1) : static_cast<RealT>(-1);
  }
  return code;
}

template <typename RealT>
Halide::Buffer<RealT> makeSymbolBuffer(int len, int order) {
  Halide::Buffer<RealT> buf(len);
  for (int i = 0; i < len; ++i) {
    buf(i) = static_cast<RealT>(i % order);
  }
  return buf;
}

template <typename RealT>
void benchDsssModChain(int symbolCount, int spreadLen, int upFactor, int iterations) {
  using ComplexT = std::complex<RealT>;
  auto const taps = makeUnitTaps<RealT>(64);
  auto const code = makeSpreadingCode<RealT>(spreadLen);

  auto bits = Signal::input(symbolCount, getScalarType<RealT>());
  auto mod = modem::qamMap(bits, kQamOrder);
  auto codeSig = Signal::constant(code, code.size());
  auto spread = kron(mod, codeSig);
  auto up = upsample(spread, upFactor, 0);
  auto shaped = filter::fir(up, taps);

  auto input = makeSymbolBuffer<RealT>(symbolCount, kQamOrder);
  int64_t const samples = static_cast<int64_t>(symbolCount) * spreadLen * upFactor;
  auto rates = runAot<ComplexT>(shaped, input, samples, iterations);
  printAotResult(withPrecision<ComplexT>("DSSS Mod Chain (QAM64)"), rates);
}

template <typename RealT>
void benchDsssDemodChain(int symbolCount, int spreadLen, int upFactor, int iterations) {
  using ComplexT = std::complex<RealT>;
  auto const taps = makeUnitTaps<RealT>(64);
  auto const code = makeSpreadingCode<RealT>(spreadLen);

  int const rxLen = symbolCount * spreadLen * upFactor;
  auto rx = Signal::input(rxLen, getScalarType<ComplexT>());
  auto matched = filter::fir(rx, taps);
  auto decim = downsample(matched, upFactor, 0);
  auto despread = filter::fir(decim, code);
  auto sym = downsample(despread, spreadLen, 0);
  auto demod = modem::qamDemap(sym, kQamOrder);

  auto input = BenchTraits<ComplexT>::makeBuffer(rxLen);
  BenchTraits<ComplexT>::fillLinear(input);
  auto const samples = static_cast<int64_t>(rxLen);
  auto rates = runAot<RealT>(demod, input, samples, iterations);
  printAotResult(withPrecision<RealT>("DSSS Demod Chain (QAM64)"), rates);
}

template <typename RealT>
void benchFftOps(int size, int batch, int iterations) {
  using ComplexT = std::complex<RealT>;
  ScalarType const realType = getScalarType<RealT>();
  ScalarType const complexType = getScalarType<ComplexT>();

  bool const hasC2c = FFT::supports(complexType, FftTransType::C2C);
  bool const hasR2c = FFT::supports(realType, FftTransType::R2C);
  bool const hasC2r = FFT::supports(realType, FftTransType::C2R);

  int64_t const samples = static_cast<int64_t>(size) * batch;

  std::optional<real64_t> c2cMs;
  std::optional<real64_t> r2cMs;
  std::optional<real64_t> c2rMs;

  if (hasC2c) {
    if (FFT::supportsDeviceBuffer(complexType, FftTransType::C2C)) {
      auto buffer = FFT::acquireDeviceBuffer(complexType, FftTransType::C2C, size, batch);
      c2cMs = BenchmarkRunner::runSafe(
          [&]() -> FFT::DeviceBuffer& {
            FFT::executeDeviceBuffer(buffer, -1);
            return buffer;
          },
          iterations);
    } else {
      std::vector<ComplexT> cData(static_cast<size_t>(size) * batch);
      c2cMs = BenchmarkRunner::runSafe([&]() { FFT::batch(cData.data(), size, batch, -1); },
                                       iterations);
    }
  }
  printSingleResult(withPrecision<RealT>("FFT C2C"), {toThroughput(c2cMs, samples)});

  if (hasR2c) {
    if (FFT::supportsDeviceBuffer(realType, FftTransType::R2C)) {
      auto buffer = FFT::acquireDeviceBuffer(realType, FftTransType::R2C, size, batch);
      r2cMs = BenchmarkRunner::runSafe(
          [&]() -> FFT::DeviceBuffer& {
            FFT::executeDeviceBuffer(buffer, -1);
            return buffer;
          },
          iterations);
    } else {
      std::vector<RealT> rIn(static_cast<size_t>(size) * batch);
      std::vector<ComplexT> rOut(static_cast<size_t>((size / 2) + 1) * batch);
      int const outStride = (size / 2) + 1;
      r2cMs = BenchmarkRunner::runSafe(
          [&]() {
            for (int b = 0; b < batch; ++b) {
              FFT::forward(rIn.data() + b * size, rOut.data() + b * outStride, size);
            }
          },
          iterations);
    }
  }
  printSingleResult(withPrecision<RealT>("FFT R2C"), {toThroughput(r2cMs, samples)});

  if (hasC2r) {
    if (FFT::supportsDeviceBuffer(realType, FftTransType::C2R)) {
      auto buffer = FFT::acquireDeviceBuffer(realType, FftTransType::C2R, size, batch);
      c2rMs = BenchmarkRunner::runSafe(
          [&]() -> FFT::DeviceBuffer& {
            FFT::executeDeviceBuffer(buffer, 1);
            return buffer;
          },
          iterations);
    } else {
      std::vector<ComplexT> rOut(static_cast<size_t>((size / 2) + 1) * batch);
      std::vector<RealT> rBack(static_cast<size_t>(size) * batch);
      int const inStride = (size / 2) + 1;
      c2rMs = BenchmarkRunner::runSafe(
          [&]() {
            for (int b = 0; b < batch; ++b) {
              FFT::inverse(rOut.data() + b * inStride, rBack.data() + b * size, size, true);
            }
          },
          iterations);
    }
  }
  printSingleResult(withPrecision<RealT>("FFT C2R"), {toThroughput(c2rMs, samples)});
}

int main() {
  BenchPrinter::printSuiteHeader("Stress");
  BenchPrinter::printBackendInfo();

  int const symbolCount = benchSize(8192);
  int const spreadLen = 64;
  int const upFactor = 8;
  int const iterations = benchIterations(6);

  int const fftSize = benchSize(8 * 4096);
  int const fftBatch = benchSize(512);
  int const fftIterations = benchIterations(1024);

  BenchPrinter::printSection("DSSS Mod Chain (AOT)");
  printAotHeader("Operation");
  forEachPrecision<real32_t, real64_t>([&](auto tag) {
    using T = typename decltype(tag)::type;
    benchDsssModChain<T>(symbolCount, spreadLen, upFactor, iterations);
  });

  BenchPrinter::printSection("DSSS Demod Chain (AOT)");
  printAotHeader("Operation");
  forEachPrecision<real32_t, real64_t>([&](auto tag) {
    using T = typename decltype(tag)::type;
    benchDsssDemodChain<T>(symbolCount, spreadLen, upFactor, iterations);
  });

  BenchPrinter::printSection("FFT (C2C/C2R/R2C)");
  printSingleHeader("Operation", "MS/s");
  forEachPrecision<real32_t, real64_t>([&](auto tag) {
    using T = typename decltype(tag)::type;
    benchFftOps<T>(fftSize, fftBatch, fftIterations);
  });

  BenchPrinter::printSummary();
  return 0;
}

/// @}
