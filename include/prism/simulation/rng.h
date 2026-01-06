/**
 * @file rng.h
 * @ingroup simulation
 * @brief 统一随机数生成器 (Random Number Generator)
 *
 * 提供基于 Mersenne Twister 的伪随机数生成功能，支持：
 * - **全局单例模式**：通过 `RNG::global()` 获取，适合简单脚本
 * - **独立实例模式**：通过构造 `RNG` 对象获取，适合多线程并行仿真
 *
 * 底层使用 `std::mt19937_64` 引擎，确保跨平台结果一致性
 */

#ifndef PRISM_SIMULATION_RNG_H
#define PRISM_SIMULATION_RNG_H

#include <cstdint>
#include <random>

#include "prism/types.h"

namespace prism::simulation {

/// @addtogroup simulation
/// @{

/**
 * @brief 统一伪随机数生成器 (PRNG) 类
 *
 * 封装了 `std::mt19937_64` (64-bit Mersenne Twister 19937)，
 * 用于生成高质量的伪随机数
 *
 * @par 线程安全性说明
 * - **独立实例**：完全线程安全，建议在并行仿真中为每个线程创建一个 `RNG` 实例，
 *   并使用不同的种子（如 `base_seed + thread_id`）
 * - **全局实例**：`RNG::global()` 返回的是函数内静态变量
 *   *警告*：若实现未使用 `thread_local`，则在多线程竞争时不仅不安全，
 *   还会破坏内部状态导致生成的序列质量下降，当前的实现是简单的 `static` 单例，
 *   因此 **非线程安全**，多线程环境下请务必使用独立实例
 *
 * @par 使用示例
 * @code
 * // 1. 全局模式（适用于单线程简单脚本）
 * RNG::setSeed(12345);
 * auto val = RNG::global().gaussian();
 *
 * // 2. 独立模式（适用于多线程并行仿真）
 * RNG my_rng(999 + thread_id);
 * auto val2 = my_rng.uniform(0.0, 1.0);
 * @endcode
 */
class RNG {
 public:
  /**
   * @brief 默认种子值 (42)
   */
  static constexpr uint64_t DEFAULT_SEED = 42;

  /**
   * @brief 构造独立 RNG 实例
   * @param seed 初始种子，若为 0，则使用 `std::random_device` 获取随机种子
   */
  explicit RNG(uint64_t seed = DEFAULT_SEED);

  /**
   * @brief 获取全局 RNG 单例
   * @return 全局 RNG 对象的引用
   * @warning 非线程安全，多线程环境请使用局部实例
   */
  static RNG& global();

  /**
   * @brief 设置全局 RNG 的种子
   * @param seed 新的种子值
   *
   * 调用此函数会重置全局 RNG 的状态
   */
  static void setSeed(uint64_t seed);

  /**
   * @brief 获取当前使用的种子
   * @return 最近一次 `reset` 或构造时使用的种子值
   */
  [[nodiscard]] uint64_t seed() const { return seed_; }

  /**
   * @brief 重置 RNG 到初始种子状态
   *
   * 用于重新开始生成相同的随机序列（复现实验）
   */
  void reset();

  /**
   * @brief 使用新种子重置 RNG
   * @param newSeed 新种子，若为 0，则使用随机设备生成种子
   */
  void reset(uint64_t newSeed);

  // ========== 生成方法 ==========

  /**
   * @brief 生成 [0, 1) 区间的均匀分布随机数
   * @return $U \sim \text{Uniform}[0, 1)$
   */
  real64_t uniform();

  /**
   * @brief 生成 [a, b) 区间的均匀分布随机数
   * @param a 下界（包含）
   * @param b 上界（不包含）
   * @return $X \sim \text{Uniform}[a, b)$
   */
  real64_t uniform(real64_t a, real64_t b);

  /**
   * @brief 生成标准正态（高斯）分布随机数
   * @param mean 均值 $\mu$ (默认 0.0)
   * @param stddev 标准差 $\sigma$ (默认 1.0)
   * @return $X \sim \mathcal{N}(\mu, \sigma^2)$
   */
  real64_t gaussian(real64_t mean = 0.0, real64_t stddev = 1.0);

  /**
   * @brief 生成复数高斯噪声 (AWGN)
   * @param stddev 复信号的总标准差 $\sigma$ (默认 1.0)
   * @return $Z = X + jY$，其中 $X, Y \sim \mathcal{N}(0, \sigma^2/2)$
   *
   * 模值的期望为 $\sigma$，功率期望为 $E[|Z|^2] = \sigma^2$
   */
  complex64_t gaussianComplex(real64_t stddev = 1.0);

  /**
   * @brief 生成随机比特
   * @return 0 或 1 (概率各 0.5)
   */
  uint8_t bit();

  /**
   * @brief 获取底层的 Mersenne Twister 引擎
   * @return `std::mt19937_64` 引擎引用
   */
  std::mt19937_64& engine() { return engine_; }

 private:
  uint64_t seed_;
  std::mt19937_64 engine_;
};

/// @}

}  // namespace prism::simulation

#endif  // PRISM_SIMULATION_RNG_H
