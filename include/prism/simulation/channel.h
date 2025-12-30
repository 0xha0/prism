/**
 * @file channel.h
 * @ingroup simulation
 * @brief 信道模型
 *
 * 仿真常见信道效应：噪声、衰落、多普勒/频偏、相位噪声。
 * 支持 complex 和 real 输入，便于链路级性能评估。
 * 统一支持 fp32 和 fp64 精度。
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
// awgn 信道
// ============================================================================

namespace detail {
template <typename T>
std::vector<T> generateNoise(int64_t length, real64_t power, RNG& gen) {
  std::vector<T> noise(length);
  real64_t const sigma = std::sqrt(power);

  if constexpr (IS_COMPLEX_V<T>) {
    // 复数噪声
    using ValueType = typename T::value_type;
    for (int64_t i = 0; i < length; ++i) {
      auto c = gen.gaussianComplex(sigma);
      noise[i] =
          T(static_cast<ValueType>(c.real()), static_cast<ValueType>(c.imag()));
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
 * @brief awgn 信道（加性高斯白噪声）
 * @tparam T 信号类型 (real32/64 或 complex32/64)
 * @param signal 输入信号
 * @param snrDb 目标信噪比（dB，信号功率基于输入估计）
 * @param rng 随机数生成器（nullptr 使用全局 RNG）
 * @return 添加噪声后的信号
 */
template <typename T>
std::vector<T> awgn(const std::vector<T>& signal, real64_t snrDb,
                    RNG* rng = nullptr) {
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
  std::vector<T> noise = detail::generateNoise<T>(
      static_cast<int64_t>(signal.size()), noisePower, gen);

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
 * @brief 多径衰落信道
 * @tparam T 信号类型 (real32/64 或 complex32/64)
 * @param signal 输入信号
 * @param taps 多径系数
 * @param delays 对应的样本延迟
 * @return 叠加衰落后的输出
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
 * @brief 多普勒频移
 * @tparam T 必须是复数类型 (complex32_t / complex64_t)
 * @param signal 输入信号
 * @param dopplerHz 频移（Hz）
 * @param sampleRate 采样率（Hz）
 * @return 加权后的信号
 */
template <typename T>
std::enable_if_t<IS_COMPLEX_V<T>, std::vector<T>> doppler(
    const std::vector<T>& signal, real64_t dopplerHz, real64_t sampleRate) {
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
 * @brief 时变多普勒（线性变化）
 * @param signal 输入信号
 * @param dopplerStart 起始频移（Hz）
 * @param dopplerEnd 结束频移（Hz）
 * @param sampleRate 采样率（Hz）
 * @return 频移按线性变化叠加后的信号
 */
template <typename T>
std::enable_if_t<IS_COMPLEX_V<T>, std::vector<T>> dopplerRamp(
    const std::vector<T>& signal, real64_t dopplerStart, real64_t dopplerEnd,
    real64_t sampleRate) {
  if (signal.empty()) return {};

  using ValueType = typename T::value_type;
  std::vector<T> output(signal.size());
  auto const n = static_cast<real64_t>(signal.size());

  real64_t phase = 0.0;
  for (size_t i = 0; i < signal.size(); ++i) {
    real64_t const t = static_cast<real64_t>(i) / n;
    real64_t const currentDoppler =
        dopplerStart + ((dopplerEnd - dopplerStart) * t);
    real64_t const phaseIncrement =
        2.0 * M_PI_VAL * currentDoppler / sampleRate;
    phase += phaseIncrement;

    T const phasor(static_cast<ValueType>(std::cos(phase)),
                   static_cast<ValueType>(std::sin(phase)));
    output[i] = signal[i] * phasor;
  }
  return output;
}

/**
 * @brief 频率偏移（CFO）
 * @param signal 输入信号
 * @param cfoHz 频偏（Hz）
 * @param sampleRate 采样率（Hz）
 * @return 叠加频偏后的信号
 */
template <typename T>
auto frequencyOffset(const std::vector<T>& signal, real64_t cfoHz,
                     real64_t sampleRate) {
  return doppler(signal, cfoHz, sampleRate);
}

// ============================================================================
// 相位噪声 (仅限复数)
// ============================================================================

/**
 * @brief 相位噪声
 * @param signal 输入信号
 * @param phaseNoiseStd 相位噪声标准差（弧度）
 * @param rng 随机数生成器（nullptr 使用全局 RNG）
 * @return 叠加随机游走相位后的信号
 */
template <typename T>
std::enable_if_t<IS_COMPLEX_V<T>, std::vector<T>> phaseNoise(
    const std::vector<T>& signal, real64_t phaseNoiseStd, RNG* rng = nullptr) {
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
