/**
 * @file channel.h
 * @ingroup simulation
 * @brief 信道模型函数库
 *
 * 提供一系列独立的信道仿真函数，包括：
 * - 噪声：AWGN, 相位噪声
 * - 衰落：多径衰落 (Multipath Fading)
 * - 频偏/多普勒：Constant Doppler, Linear Ramp Doppler, CFO
 *
 * 设计为函数式接口，方便在仿真链路中组合使用
 * 支持实数与复数输入（根据函数特性），统一支持 fp32/fp64
 */

#ifndef PRISM_SIMULATION_CHANNEL_H
#define PRISM_SIMULATION_CHANNEL_H

#include <algorithm>
#include <cmath>
#include <complex>
#include <type_traits>
#include <vector>

#include "prism/simulation/rng.h"
#include "prism/types.h"

namespace prism::simulation {

/// @addtogroup simulation
/// @{

// ============================================================================
// AWGN 信道
// ============================================================================

namespace detail {
template <typename T>
std::vector<T> generateNoise(int64_t length, real64_t power, RNG& gen) {
  std::vector<T> noise(length);
  real64_t const sigma = std::sqrt(power);

  if constexpr (IS_COMPLEX_V<T>) {
    // 复数噪声：实部虚部各分担一半功率 (sigma/sqrt(2))
    // RNG::gaussianComplex 内部已处理 sigma = stddev
    // 若 power 为 total power，则 stddev = sqrt(power).
    using ValueType = typename T::value_type;
    for (int64_t i = 0; i < length; ++i) {
      auto c = gen.gaussianComplex(sigma);
      noise[i] = T(static_cast<ValueType>(c.real()), static_cast<ValueType>(c.imag()));
    }
  } else {
    // 实数噪声
    for (int64_t i = 0; i < length; ++i) {
      noise[i] = static_cast<T>(gen.gaussian(0.0, sigma));
    }
  }
  return noise;
}
}  // namespace detail

/**
 * @brief AWGN (加性高斯白噪声) 信道
 *
 * 向输入信号添加零均值的高斯噪声
 * 噪声功率根据输入信号的测量功率与目标信噪比 (SNR) 计算得出：
 * $P_{noise} = P_{signal} / 10^{SNR_{dB}/10}$
 *
 * @tparam T 信号类型 (支持实数 real32/64 或复数 complex32/64)
 * @param signal 输入信号序列
 * @param snrDb 目标信噪比 (dB)
 * @param rng 随机数生成器指针 (可选，传 nullptr 则使用全局单例 RNG)
 * @return 叠加噪声后的信号序列 (Signal + Noise)
 */
template <typename T>
std::vector<T> awgn(const std::vector<T>& signal, real64_t snrDb, RNG* rng = nullptr) {
  if (signal.empty()) return {};
  RNG& gen = rng ? *rng : RNG::global();

  real64_t signalPower = 0.0;
  for (const auto& s : signal) {
    if constexpr (IS_COMPLEX_V<T>) {
      signalPower += std::norm(s);
    } else {
      signalPower += s * s;
    }
  }
  signalPower /= static_cast<real64_t>(signal.size());

  real64_t const noisePower = signalPower / std::pow(10.0, snrDb / 10.0);
  std::vector<T> noise =
      detail::generateNoise<T>(static_cast<int64_t>(signal.size()), noisePower, gen);

  std::vector<T> output(signal.size());
  for (size_t i = 0; i < signal.size(); ++i) {
    output[i] = signal[i] + noise[i];
  }
  return output;
}

// ============================================================================
// 多径衰落
// ============================================================================

/**
 * @brief 多径衰落信道 (Multipath Fading)
 *
 * 模拟信号通过多条路径传播产生的时散效应 (Frequency Selective Fading)
 * 数学模型为线性时不变 (LTI) 离散卷积：
 * $y[n] = \sum_{k} h[k] \cdot x[n - \tau_k]$
 *
 * @tparam T 信号类型
 * @param signal 输入信号
 * @param taps 多径衰落系数 $h[k]$ (复数或实数增益)
 * @param delays 各路径的采样点延迟 $\tau_k$ (非负整数)
 * @return 叠加衰落后的输出 (长度调整为与输入一致，拖尾部分被截断)
 */
template <typename T>
std::vector<T> fading(const std::vector<T>& signal, const std::vector<T>& taps,
                      const std::vector<int>& delays) {
  if (signal.empty() || taps.empty()) return signal;

  int maxDelay = 0;
  for (int const d : delays) {
    maxDelay = std::max(d, maxDelay);
  }

  // 初始化为 0
  std::vector<T> output(signal.size() + maxDelay);
  // vector 默认初始化已经为 0 (对于 arithmetic types 和 std::complex)

  for (size_t p = 0; p < taps.size(); ++p) {
    int const delay = (p < delays.size()) ? delays[p] : 0;
    T const tap = taps[p];
    for (size_t i = 0; i < signal.size(); ++i) {
      output[i + delay] += signal[i] * tap;
    }
  }

  output.resize(signal.size());
  return output;
}

// ============================================================================
// 多普勒 / 频偏 (仅限复数)
// ============================================================================

/**
 * @brief 恒定多普勒频移 (Constant Doppler Shift)
 *
 * 模拟发射机与接收机之间的相对运动导致的频率偏移
 * 操作：将信号乘以复旋转矢量：
 * $y[n] = x[n] \cdot e^{j 2\pi f_d n / f_s}$
 *
 * @tparam T 必须是复数类型 (complex32_t / complex64_t)
 * @param signal 输入信号
 * @param dopplerHz 频移量 $f_d$ (Hz)
 * @param sampleRate 采样率 $f_s$ (Hz)
 * @return 旋转后的信号
 */
template <typename T>
std::enable_if_t<IS_COMPLEX_V<T>, std::vector<T>> doppler(const std::vector<T>& signal,
                                                          real64_t dopplerHz, real64_t sampleRate) {
  if (signal.empty()) return {};

  using ValueType = typename T::value_type;
  std::vector<T> output(signal.size());
  real64_t const phaseIncrement = 2.0 * M_PI_VAL * dopplerHz / sampleRate;

  for (size_t i = 0; i < signal.size(); ++i) {
    real64_t const phase = phaseIncrement * static_cast<real64_t>(i);
    T const phasor(static_cast<ValueType>(std::cos(phase)),
                   static_cast<ValueType>(std::sin(phase)));
    output[i] = signal[i] * phasor;
  }
  return output;
}

/**
 * @brief 时变多普勒 (Linear Ramp Doppler)
 *
 * 模拟加速运动（如卫星过顶），频偏随时间线性变化
 * 频偏从 `dopplerStart` 线性变化到 `dopplerEnd`
 *
 * @param signal 输入信号
 * @param dopplerStart 起始频移 (Hz)
 * @param dopplerEnd 结束频移 (Hz)
 * @param sampleRate 采样率 (Hz)
 * @return 频移线性扫描后的信号
 */
template <typename T>
std::enable_if_t<IS_COMPLEX_V<T>, std::vector<T>> dopplerRamp(const std::vector<T>& signal,
                                                              real64_t dopplerStart,
                                                              real64_t dopplerEnd,
                                                              real64_t sampleRate) {
  if (signal.empty()) return {};

  using ValueType = typename T::value_type;
  std::vector<T> output(signal.size());
  auto const n = static_cast<real64_t>(signal.size());

  real64_t phase = 0.0;
  for (size_t i = 0; i < signal.size(); ++i) {
    real64_t const t = static_cast<real64_t>(i) / n;
    real64_t const currentDoppler = dopplerStart + ((dopplerEnd - dopplerStart) * t);
    real64_t const phaseIncrement = 2.0 * M_PI_VAL * currentDoppler / sampleRate;
    phase += phaseIncrement;

    T const phasor(static_cast<ValueType>(std::cos(phase)),
                   static_cast<ValueType>(std::sin(phase)));
    output[i] = signal[i] * phasor;
  }
  return output;
}

/**
 * @brief 载波频率偏移 (CFO - Carrier Frequency Offset)
 *
 * 别名 `doppler`，通常用于模拟晶振误差 (PPM)
 *
 * @param signal 输入信号
 * @param cfoHz 频偏 (Hz)
 * @param sampleRate 采样率 (Hz)
 * @return 叠加频偏后的信号
 */
template <typename T>
auto frequencyOffset(const std::vector<T>& signal, real64_t cfoHz, real64_t sampleRate) {
  return doppler(signal, cfoHz, sampleRate);
}

// ============================================================================
// 相位噪声 (仅限复数)
// ============================================================================

/**
 * @brief 相位噪声 (Phase Noise)
 *
 * 模拟本振源的不稳定性，采用随机游走 (Random Walk / Wiener Process) 模型：
 * $\phi[n] = \phi[n-1] + \mathcal{N}(0, \sigma^2)$
 * $y[n] = x[n] \cdot e^{j \phi[n]}$
 *
 * @param signal 输入信号
 * @param phaseNoiseStd 步进高斯噪声的标准差 (Radians/Sample)
 * @param rng 随机数生成器
 * @return 叠加相噪后的信号
 */
template <typename T>
std::enable_if_t<IS_COMPLEX_V<T>, std::vector<T>> phaseNoise(const std::vector<T>& signal,
                                                             real64_t phaseNoiseStd,
                                                             RNG* rng = nullptr) {
  if (signal.empty()) return {};

  RNG& gen = rng ? *rng : RNG::global();
  using ValueType = typename T::value_type;

  std::vector<T> output(signal.size());
  real64_t accumulatedPhase = 0.0;

  for (size_t i = 0; i < signal.size(); ++i) {
    accumulatedPhase += gen.gaussian(0.0, phaseNoiseStd);
    T const phasor(static_cast<ValueType>(std::cos(accumulatedPhase)),
                   static_cast<ValueType>(std::sin(accumulatedPhase)));
    output[i] = signal[i] * phasor;
  }
  return output;
}

/// @}

}  // namespace prism::simulation

#endif  // PRISM_SIMULATION_CHANNEL_H
