/**
 * @file test_simulation.cpp
 * @ingroup tests
 * @brief 仿真模块单元测试
 *
 * 验证 PRISM 内置的信号源 (Source) 和信道 (Channel) 模型的正确性与统计特性
 * - **信号源**：验证 Random Bits/Syms/QAM/PSK 的分布均匀性和功率归一化
 * - **信道模型**：验证 AWGN (加性高斯白噪声)、Multipath Fading
 * (多径衰落)、Doppler (多普勒频移) 等效应
 *
 * 采用统计方法 (如均值、方差验证) 确保随机过程符合预期分布
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

// ==============================================================================
// 辅助统计函数 (Statistical Helpers)
// ==============================================================================

/**
 * @brief 计算复信号的平均功率 (Mean Power)
 * $P = \frac{1}{N} \sum |x[n]|^2$
 */
template <typename T>
T calculateMeanPower(const std::vector<std::complex<T>>& signal) {
  if (signal.empty()) return 0.0;
  T power = 0.0;
  for (const auto& s : signal) {
    power += std::norm(s);  // norm returns real^2 + imag^2
  }
  return power / static_cast<T>(signal.size());
}

/**
 * @brief 计算实信号的算术平均值 (Mean)
 */
template <typename T>
T calculateMean(const std::vector<T>& signal) {
  if (signal.empty()) return 0.0;
  T sum = 0.0;
  for (const auto& s : signal) {
    sum += s;
  }
  return sum / static_cast<T>(signal.size());
}

/**
 * @brief 计算复信号的算术平均值 (Mean)
 */
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
// 信号源生成器测试 (Source Generator Tests)
// ==============================================================================

/**
 * @brief [Test] 随机信号源测试
 *
 * 验证以下生成器的基本特性：
 * 1. **randomSymbols**: 是否生成了有效范围内的符号，且长度正确
 * 2. **randomQam/randomPsk**: 验证生成信号的平均功率是否归一化为 1.0 (Unit
 * Power)
 */
template <typename RealType>
void testSourceGenerators() {
  std::string const suffix = "(" + TypeName<RealType>::get() + ")";
  RNG::setSeed(42);

  // 1. randomSymbols
  {
    auto syms = randomSymbols<RealType>(1000, 4);
    bool pass = (syms.size() == 1000);
    // 符号索引应为整数 (浮点表示)
    for (const auto& s : syms) {
      // 符号索引应在 [0, M-1] 范围内
      if (s < static_cast<RealType>(0.0) || s > static_cast<RealType>(3.0)) {
        pass = false;
      }
    }
    TestPrinter::printTestResult("randomSymbols " + suffix, pass);
    assert(pass);
  }

  // 2. randomQam (QPSK, M=4)
  {
    auto qam4 = randomQam<RealType>(1000, 4);
    RealType pwr4 = calculateMeanPower(qam4);
    // 验证功率归一化 (允许一定统计误差)
    bool const pass = std::abs(pwr4 - 1.0) < 0.1;
    TestPrinter::printTestResult("randomQam (QPSK) " + suffix, pass, "pwr=" + std::to_string(pwr4));
    assert(pass);
  }

  // 3. randomQam (16QAM, M=16)
  {
    auto qam16 = randomQam<RealType>(1000, 16);
    RealType pwr16 = calculateMeanPower(qam16);
    bool const pass = std::abs(pwr16 - 1.0) < 0.15;
    TestPrinter::printTestResult("randomQam (16QAM) " + suffix, pass,
                                 "pwr=" + std::to_string(pwr16));
    assert(pass);
  }

  // 4. randomPsk (QPSK, M=4)
  {
    auto psk4 = randomPsk<RealType>(1000, 4);
    bool pass = true;
    for (const auto& s : psk4) {
      // PSK 所有点都在单位圆上，模应严格为 1
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
// 信道模型测试
// ==============================================================================

/**
 * @brief [Test] AWGN 信道测试
 *
 * 验证加高斯白噪声后的信号统计特性：
 * 1. 均值：噪声均值为 0，加上信号后均值应接近信号本身
 * 2. 方差 (功率)：验证添加的噪声功率是否符合指定的 SNR
 */
template <typename RealType>
void testChannelAwgn() {
  std::string const suffix = "(" + TypeName<RealType>::get() + ")";
  RNG::setSeed(42);
  const int n = 50000;  // 足够大的样本量以收敛

  // 1. 实数 AWGN
  {
    std::vector<RealType> const sigReal(n, static_cast<RealType>(1.0));
    // SNR = 20dB -> Noise Power = 0.01 (Signal Power=1)
    auto noisyReal = awgn(sigReal, static_cast<RealType>(20.0));
    RealType meanReal = calculateMean(noisyReal);
    // 均值应保持为 1.0
    bool const pass = std::abs(meanReal - 1.0) < 0.05;
    TestPrinter::printTestResult("Real AWGN (20dB) " + suffix, pass,
                                 "mean=" + std::to_string(meanReal));
    assert(pass);
  }

  // 2. 复数 AWGN
  {
    std::vector<std::complex<RealType>> const sigComplex(n, {1.0, 0.0});
    // SNR = 10dB -> Noise Power = 0.1 (Signal Power=1)
    auto noisyComplex = awgn(sigComplex, static_cast<RealType>(10.0));
    auto meanComplex = calculateMean(noisyComplex);
    bool const passMean = std::abs(meanComplex.real() - 1.0) < 0.05;

    // 计算实际噪声功率 (方差)
    RealType noiseVar = 0.0;
    for (const auto& s : noisyComplex) {
      noiseVar += std::norm(s - std::complex<RealType>(1.0, 0.0));
    }
    noiseVar /= n;

    // 验证噪声方差接近 0.1
    bool const passVar = std::abs(noiseVar - 0.1) < 0.02;

    bool const pass = passMean && passVar;
    TestPrinter::printTestResult("Complex AWGN (10dB) " + suffix, pass,
                                 "var=" + std::to_string(noiseVar));
    assert(pass);
  }
}

/**
 * @brief [Test] 衰落、多普勒等信道效应测试
 */
template <typename RealType>
void testChannelEffects() {
  std::string const suffix = "(" + TypeName<RealType>::get() + ")";

  using ComplexT = std::complex<RealType>;
  std::vector<ComplexT> const sig(100, {1.0, 0.0});

  // 1. Doppler Shift
  // 验证固定频偏造成的相位旋转
  {
    // dopplerHz=100, fs=1000 -> 归一化频率 0.1 -> 每个采样旋转 0.1 * 2pi
    auto shifted = doppler(sig, static_cast<RealType>(100.0), static_cast<RealType>(1000.0));
    auto twoPi = static_cast<RealType>(2.0 * M_PI_VAL);
    // 第 10 个点的相位应旋转 10 * 0.1 * 2pi
    auto expectedPhase = static_cast<RealType>(twoPi * 0.1 * 10.0);

    ComplexT const expected = std::polar(static_cast<RealType>(1.0), expectedPhase);

    // 考虑 2pi 周期性，直接比较复数值
    bool const pass = std::abs(shifted[10] - expected) < 1e-4;
    TestPrinter::printTestResult("Doppler Shift " + suffix, pass);
    assert(pass);
  }

  // 2. Frequency Selective Fading (多径)
  // Taps = [0.5, 0.5], Delays = [0, 1]
  // y[n] = 0.5*x[n] + 0.5*x[n-1]
  {
    std::vector<ComplexT> const taps = {{0.5, 0.0}, {0.5, 0.0}};
    std::vector<int> const delays = {0, 1};
    auto faded = fading(sig, taps, delays);

    bool pass = true;
    // x[0]=1, x[-1]=0 -> y[0] = 0.5*1 + 0 = 0.5
    if (std::abs(faded[0] - ComplexT(0.5, 0.0)) >= 1e-4) pass = false;
    // x[1]=1, x[0]=1 -> y[1] = 0.5*1 + 0.5*1 = 1.0
    if (std::abs(faded[1] - ComplexT(1.0, 0.0)) >= 1e-4) pass = false;

    TestPrinter::printTestResult("Freq Selective Fading " + suffix, pass);
    assert(pass);
  }
}

// ==============================================================================
// 通用工具测试
// ==============================================================================

/**
 * @brief [Test] 随机比特生成器均匀性测试
 *
 * 生成大量比特，验证 0 和 1 的数量大致相等 (Sum ≈ N/2)
 */
void testRandomBits() {
  RNG::setSeed(42);
  auto bits = randomBits(1000);
  bool pass = (bits.size() == 1000);
  int sum = 0;
  for (auto b : bits) {
    if (b != 0 && b != 1) pass = false;
    sum += b;
  }
  // 1000 bits, sum should be around 500. Range [400, 600] is 500 +/- 100 (loose
  // check)
  if (sum <= 400 || sum >= 600) pass = false;
  TestPrinter::printTestResult("Random Bits Uniformity", pass, "sum=" + std::to_string(sum));
  assert(pass);
}

// ==============================================================================
// Main Entry
// ==============================================================================

int main() {
  TestPrinter::printSuiteHeader("Simulation Module Tests");

  TestPrinter::printSection("Shared Utils");
  testRandomBits();

  TestPrinter::printSection("Source Generators");
  testSourceGenerators<real32_t>();
  testSourceGenerators<real64_t>();

  TestPrinter::printSection("Channel");
  testChannelAwgn<real32_t>();
  testChannelAwgn<real64_t>();
  testChannelEffects<real32_t>();
  testChannelEffects<real64_t>();

  TestPrinter::printSummary();
  return 0;
}

/// @}
