/**
 * @file test_simulation.cpp
 * @ingroup tests
 * @brief 仿真模块单元测试
 *
 * 覆盖随机源与信道模型，验证单/双精度下的数值正确性。
 */

#include <cassert>
#include <cmath>
#include <complex>
#include <string>
#include <vector>

#include "prism/simulation/channel.h"
#include "prism/simulation/rng.h"
#include "prism/simulation/source.h"
#include "prism/types.h"
#include "test_utils.h"

using prism::complex32_t;
using prism::complex64_t;
using prism::real32_t;
using prism::real64_t;
using namespace prism::simulation;
using namespace prism::tests;

/// @addtogroup tests
/// @{

/** @brief 计算复信号平均功率 */
template <typename T>
T calculateMeanPower(const std::vector<std::complex<T>>& signal) {
  if (signal.empty()) return 0.0;
  T power = 0.0;
  for (const auto& s : signal) {
    power += std::norm(s);
  }
  return power / static_cast<T>(signal.size());
}

template <typename T>
T calculateMean(const std::vector<T>& signal) {
  if (signal.empty()) return 0.0;
  T sum = 0.0;
  for (const auto& s : signal) {
    sum += s;
  }
  return sum / static_cast<T>(signal.size());
}

template <typename T>
std::complex<T> calculateMean(const std::vector<std::complex<T>>& signal) {
  if (signal.empty()) return {0.0, 0.0};
  std::complex<T> sum = {0.0, 0.0};
  for (const auto& s : signal) {
    sum += s;
  }
  return sum / static_cast<T>(signal.size());
}

// ==============================================================================
// 信号源与信道测试
// ==============================================================================

/** @brief 随机符号/QAM/PSK 生成器统计校验 */
template <typename RealType>
void testSourceGenerators() {
  std::string const suffix = "(" + TypeName<RealType>::get() + ")";
  RNG::setSeed(42);

  // 1. randomSymbols
  {
    auto syms = randomSymbols<RealType>(1000);
    bool pass = (syms.size() == 1000);
    for (const auto& s : syms) {
      if (std::abs(s.real()) > 1.0 || std::abs(s.imag()) > 1.0) pass = false;
    }
    TestPrinter::printTestResult("randomSymbols " + suffix, pass);
    assert(pass);
  }

  // 2. randomQam (QPSK)
  {
    auto qam4 = randomQam<RealType>(1000, 4);
    RealType pwr4 = calculateMeanPower(qam4);
    bool const pass = std::abs(pwr4 - 1.0) < 0.1;
    TestPrinter::printTestResult("randomQam (QPSK) " + suffix, pass,
                                 "pwr=" + std::to_string(pwr4));
    assert(pass);
  }

  // 3. randomQam (16QAM)
  {
    auto qam16 = randomQam<RealType>(1000, 16);
    RealType pwr16 = calculateMeanPower(qam16);
    bool const pass = std::abs(pwr16 - 1.0) < 0.15;
    TestPrinter::printTestResult("randomQam (16QAM) " + suffix, pass,
                                 "pwr=" + std::to_string(pwr16));
    assert(pass);
  }

  // 4. randomPsk (QPSK)
  {
    auto psk4 = randomPsk<RealType>(1000, 4);
    bool pass = true;
    for (const auto& s : psk4) {
      if (std::abs(std::abs(s) - 1.0) >= 1e-4) pass = false;
    }
    RealType pwrPsk = calculateMeanPower(psk4);
    if (std::abs(pwrPsk - 1.0) >= 1e-4) pass = false;
    TestPrinter::printTestResult("randomPsk (QPSK) " + suffix, pass,
                                 "pwr=" + std::to_string(pwrPsk));
    assert(pass);
  }
}

// ==============================================================================
// Channel Tests
// ==============================================================================

/** @brief AWGN 信道均值与噪声功率检查 */
template <typename RealType>
void testChannelAwgn() {
  std::string const suffix = "(" + TypeName<RealType>::get() + ")";
  RNG::setSeed(42);
  const int n = 50000;

  // 1. Real AWGN
  {
    std::vector<RealType> const sigReal(n, static_cast<RealType>(1.0));
    auto noisyReal = awgn(sigReal, static_cast<RealType>(20.0));  // 20dB
    RealType meanReal = calculateMean(noisyReal);
    bool const pass = std::abs(meanReal - 1.0) < 0.05;
    TestPrinter::printTestResult("Real AWGN (20dB) " + suffix, pass,
                                 "mean=" + std::to_string(meanReal));
    assert(pass);
  }

  // 2. Complex AWGN
  {
    std::vector<std::complex<RealType>> const sigComplex(n, {1.0, 0.0});
    auto noisyComplex = awgn(sigComplex, static_cast<RealType>(10.0));  // 10dB
    auto meanComplex = calculateMean(noisyComplex);
    bool const passMean = std::abs(meanComplex.real() - 1.0) < 0.05;

    RealType noiseVar = 0.0;
    for (const auto& s : noisyComplex) {
      noiseVar += std::norm(s - std::complex<RealType>(1.0, 0.0));
    }
    noiseVar /= n;
    bool const passVar = std::abs(noiseVar - 0.1) <
                         0.02;  // 0.1 noise power for 10dB (if signal is 1)

    bool const pass = passMean && passVar;
    TestPrinter::printTestResult("Complex AWGN (10dB) " + suffix, pass,
                                 "var=" + std::to_string(noiseVar));
    assert(pass);
  }
}

/** @brief 衰落、多普勒、相位噪声组合测试 */
template <typename RealType>
void testChannelEffects() {
  std::string const suffix = "(" + TypeName<RealType>::get() + ")";

  using ComplexT = std::complex<RealType>;
  std::vector<ComplexT> const sig(100, {1.0, 0.0});

  // 1. Doppler
  {
    auto shifted = doppler(sig, static_cast<RealType>(100.0),
                           static_cast<RealType>(1000.0));
    auto twoPi = static_cast<RealType>(2.0 * M_PI_VAL);
    auto expectedPhase = static_cast<RealType>(twoPi * 0.1 * 10.0);

    ComplexT const expected =
        std::polar(static_cast<RealType>(1.0), expectedPhase);

    bool const pass = std::abs(shifted[10] - expected) < 1e-4;
    TestPrinter::printTestResult("Doppler Shift " + suffix, pass);
    assert(pass);
  }

  // 2. Fading
  {
    std::vector<ComplexT> const taps = {{0.5, 0.0}, {0.5, 0.0}};
    std::vector<int> const delays = {0, 1};
    auto faded = fading(sig, taps, delays);
    bool pass = true;
    if (std::abs(faded[0] - ComplexT(0.5, 0.0)) >= 1e-4) pass = false;
    if (std::abs(faded[1] - ComplexT(1.0, 0.0)) >= 1e-4) pass = false;
    TestPrinter::printTestResult("Freq Selective Fading " + suffix, pass);
    assert(pass);
  }
}

// ==============================================================================
// Shared Tests
// ==============================================================================

/** @brief 随机比特均匀性测试 */
void testRandomBits() {
  RNG::setSeed(42);
  auto bits = randomBits(1000);
  bool pass = (bits.size() == 1000);
  int sum = 0;
  for (auto b : bits) {
    if (b != 0 && b != 1) pass = false;
    sum += b;
  }
  if (sum <= 400 || sum >= 600) pass = false;
  TestPrinter::printTestResult("Random Bits Uniformity", pass,
                               "sum=" + std::to_string(sum));
  assert(pass);
}

// ==============================================================================
// Main
// ==============================================================================

int main() {
  TestPrinter::printSuiteHeader("Simulation Module Tests (Refactored)");

  TestPrinter::printSection("Shared Utils");
  testRandomBits();

  TestPrinter::printSection("Source Generators (Real32)");
  testSourceGenerators<real32_t>();

  TestPrinter::printSection("Source Generators (Real64)");
  testSourceGenerators<real64_t>();

  TestPrinter::printSection("Channel (Real32)");
  testChannelAwgn<real32_t>();
  testChannelEffects<real32_t>();

  TestPrinter::printSection("Channel (Real64)");
  testChannelAwgn<real64_t>();
  testChannelEffects<real64_t>();

  TestPrinter::printSummary();
  return 0;
}

/// @}
