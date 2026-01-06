/**
 * @file source.h
 * @ingroup simulation
 * @brief 仿真信号源函数库
 *
 * 提供常用的仿真信号源生成函数，支持：
 * - 随机比特流 (Random Bits)
 * - 随机整数符号 (Random Symbols)
 * - 随机 QAM 星座点 (Random QAM)
 * - 随机 PSK 星座点 (Random PSK)
 *
 * 统一支持 fp32/fp64 精度和自定义 RNG
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
 * @param length 序列长度 (Bits count)
 * @param rng 随机数生成器 (可选)
 * @return 包含 0 或 1 的向量
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
 * @brief 生成均匀分布的随机符号 (整数索引)
 *
 * 生成 [0, order) 范围内的随机整数，存储为浮点类型
 *
 * @tparam T 基础实数类型 (real32_t / real64_t)
 * @param length 符号数量
 * @param order 符号阶数 M (e.g. 2, 4, 8)
 * @param rng 随机数生成器
 * @return 符号序列 (值为 0, 1, ..., M-1)
 */
template <typename T>
std::enable_if_t<IS_REAL_TYPE_V<T>, std::vector<T>> randomSymbols(int64_t length, int order,
                                                                  RNG* rng = nullptr) {
  RNG& gen = rng ? *rng : RNG::global();
  std::vector<T> symbols(length);
  for (int64_t i = 0; i < length; ++i) {
    symbols[i] = static_cast<T>(std::floor(gen.uniform(0, order)));
  }
  return symbols;
}

/**
 * @brief 生成随机 QAM 星座点序列
 *
 * 均匀选取 M-QAM 星座图上的点，并进行功率归一化
 *
 * @par 功率归一化
 * 为了使平均符号能量 $E[|s|^2] = 1$，对星座点坐标进行缩放：
 * Scale = $1 / \sqrt{2(M-1)/3}$
 *
 * @tparam T 基础实数类型
 * @param length 序列长度
 * @param order 调制阶数 M (必须是平方数, e.g. 4, 16, 64)
 * @param rng 随机数生成器
 * @return 归一化后的复数符号序列
 */
template <typename T>
std::enable_if_t<IS_REAL_TYPE_V<T>, std::vector<std::complex<T>>> randomQam(int64_t length,
                                                                            int order = 4,
                                                                            RNG* rng = nullptr) {
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
 * @brief 生成随机 PSK 星座点序列
 *
 * 均匀选取 M-PSK 星座图上的点（位于单位圆上）
 * 模值恒为 1，平均功率为 1
 * 星座点：$s_k = e^{j 2\pi k / M}$
 *
 * @tparam T 基础实数类型
 * @param length 序列长度
 * @param order 调制阶数 M (e.g. 2, 4, 8)
 * @param rng 随机数生成器
 * @return 复数符号序列
 */
template <typename T>
std::enable_if_t<IS_REAL_TYPE_V<T>, std::vector<std::complex<T>>> randomPsk(int64_t length,
                                                                            int order = 2,
                                                                            RNG* rng = nullptr) {
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
