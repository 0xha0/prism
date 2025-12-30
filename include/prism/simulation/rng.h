/**
 * @file rng.h
 * @ingroup simulation
 * @brief 统一随机数生成器
 *
 * 提供全局和独立两种使用模式：
 * - **全局模式**：RNG::global() 获取全局实例，RNG::set_seed() 设置种子
 * - **独立模式**：创建独立 RNG 实例，互不干扰
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
 * @brief 统一伪随机数生成器
 *
 * 封装 Mersenne Twister 引擎，支持多种分布。
 *
 * @par 使用示例
 * @code
 * // 全局模式（推荐用于可复现仿真）
 * RNG::set_seed(12345);
 * auto val = RNG::global().gaussian();
 *
 * // 独立模式
 * RNG my_rng(999);
 * auto val2 = my_rng.uniform();
 * @endcode
 */
class RNG {
 public:
  /**
   * @brief 默认种子值
   */
  static constexpr uint64_t DEFAULT_SEED = 42;

  /**
   * @brief 构造独立实例
   * @param seed 种子（0 = 使用随机种子）
   */
  explicit RNG(uint64_t seed = DEFAULT_SEED);

  /**
   * @brief 获取全局实例
   * @return 线程安全的全局 RNG
   */
  static RNG& global();

  /**
   * @brief 设置全局种子（重置全局 RNG）
   */
  static void setSeed(uint64_t seed);

  /**
   * @brief 获取当前种子
   * @return 最近一次使用的种子
   */
  [[nodiscard]] uint64_t seed() const { return seed_; }

  /**
   * @brief 重置到初始状态
   */
  void reset();

  /**
   * @brief 使用新种子重置
   */
  void reset(uint64_t newSeed);

  // ========== 生成方法 ==========

  /**
   * @brief 生成 [0, 1) 均匀分布
   * @return 抽样值
   */
  real64_t uniform();

  /**
   * @brief 生成 [a, b) 均匀分布
   * @return 抽样值
   */
  real64_t uniform(real64_t a, real64_t b);

  /**
   * @brief 生成高斯分布
   * @return 抽样值
   */
  real64_t gaussian(real64_t mean = 0.0, real64_t stddev = 1.0);

  /**
   * @brief 生成复数高斯噪声
   * @return 均值 0、方差 stddev^2 的复数
   */
  complex64_t gaussianComplex(real64_t stddev = 1.0);

  /**
   * @brief 生成随机比特
   * @return 0 或 1
   */
  uint8_t bit();

  /**
   * @brief 获取底层引擎
   */
  std::mt19937_64& engine() { return engine_; }

 private:
  uint64_t seed_;
  std::mt19937_64 engine_;
};

/// @}

}  // namespace prism::simulation

#endif  // PRISM_SIMULATION_RNG_H
