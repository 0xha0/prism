/**
 * @file source.h
 * @ingroup simulation
 * @brief 仿真信号源
 *
 * 提供算法验证所需的随机比特、符号与 QAM/PSK 序列生成器。
 * 支持 fp32 (real32_t) 和 fp64 (real64_t) 精度。
 */

#ifndef PRISM_SIMULATION_SOURCE_H
#define PRISM_SIMULATION_SOURCE_H

#include <cmath>
#include <complex>
#include <cstdint>
#include <type_traits>
#include <vector>

#include "prism/simulation/rng.h"
#include "prism/types.h"

namespace prism::simulation {

/// @addtogroup simulation
/// @{

/**
 * @brief 生成随机比特序列
 * @param length 比特数量
 * @param rng 随机数生成器（nullptr 使用全局 RNG）
 * @return 长度为 length 的 0/1 序列
 */
inline std::vector<uint8_t> randomBits(int64_t length, RNG* rng = nullptr) {
  RNG& gen = rng ? *rng : RNG::global();
  std::vector<uint8_t> bits(length);
  for (int64_t i = 0; i < length; ++i) {
    bits[i] = gen.bit();
  }
  return bits;
}

/**
 * @brief 生成随机复数符号
 * @tparam T 基础实数类型 (real32_t / real64_t)
 * @param length 符号数量
 * @param rng 随机数生成器（nullptr 使用全局 RNG）
 * @return 均匀分布在单位方形的复数符号
 */
template <typename T>
std::enable_if_t<IS_REAL_TYPE_V<T>, std::vector<std::complex<T>>> randomSymbols(
    int64_t length, RNG* rng = nullptr) {
  RNG& gen = rng ? *rng : RNG::global();
  std::vector<std::complex<T>> symbols(length);
  for (int64_t i = 0; i < length; ++i) {
    auto const real = static_cast<T>(gen.uniform(-1.0, 1.0));
    auto const imag = static_cast<T>(gen.uniform(-1.0, 1.0));
    symbols[i] = std::complex<T>(real, imag);
  }
  return symbols;
}

/**
 * @brief 生成随机 QAM 符号
 * @tparam T 基础实数类型 (real32_t / real64_t)
 * @param length 符号数量
 * @param order QAM 阶数 (4, 16, 64, 256)
 * @param rng 随机数生成器（nullptr 使用全局 RNG）
 * @return 归一化功率的 QAM 点集合
 */
template <typename T>
std::enable_if_t<IS_REAL_TYPE_V<T>, std::vector<std::complex<T>>> randomQam(
    int64_t length, int order = 4, RNG* rng = nullptr) {
  RNG& gen = rng ? *rng : RNG::global();

  int const m = static_cast<int>(std::sqrt(order));
  auto const scale = static_cast<T>(1.0 / std::sqrt(2.0 * (order - 1.0) / 3.0));

  std::uniform_int_distribution<int> dist(0, m - 1);

  std::vector<std::complex<T>> symbols(length);
  for (int64_t i = 0; i < length; ++i) {
    auto const valR = static_cast<T>((2 * dist(gen.engine())) - m + 1);
    auto const valI = static_cast<T>((2 * dist(gen.engine())) - m + 1);
    symbols[i] = std::complex<T>(valR * scale, valI * scale);
  }
  return symbols;
}

/**
 * @brief 生成随机 PSK 符号
 * @tparam T 基础实数类型 (real32_t / real64_t)
 * @param length 符号数量
 * @param order PSK 阶数 (e.g. 2, 4, 8)
 * @param rng 随机数生成器（nullptr 使用全局 RNG）
 * @return 归一化功率(模为1)的 PSK 点集合
 */
template <typename T>
std::enable_if_t<IS_REAL_TYPE_V<T>, std::vector<std::complex<T>>> randomPsk(
    int64_t length, int order = 2, RNG* rng = nullptr) {
  RNG& gen = rng ? *rng : RNG::global();

  std::uniform_int_distribution<int> dist(0, order - 1);

  std::vector<std::complex<T>> symbols(length);
  T const phaseStep = static_cast<T>(2.0 * M_PI_VAL / order);

  for (int64_t i = 0; i < length; ++i) {
    int const k = dist(gen.engine());
    T const phase = k * phaseStep;
    symbols[i] = std::complex<T>(std::cos(phase), std::sin(phase));
  }
  return symbols;
}

/// @}

}  // namespace prism::simulation

#endif  // PRISM_SIMULATION_SOURCE_H
