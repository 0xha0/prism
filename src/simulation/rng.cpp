/**
 * @file rng.cpp
 * @ingroup simulation
 * @brief 随机数生成器实现
 */

#include "prism/simulation/rng.h"

#include <cmath>
#include <cstdint>
#include <random>

#include "prism/types.h"

namespace prism::simulation {

/// @addtogroup simulation
/// @{

/**
 * @brief 构造函数：初始化 RNG
 *
 * @param seed 初始种子，默认 42
 */
RNG::RNG(uint64_t seed) : seed_(seed) { reset(seed); }

/**
 * @brief 获取全局 RNG 实例
 *
 * 这里使用局部静态变量实现单例模式（Lazy Initialization）
 * @note 非线程安全
 */
RNG& RNG::global() {
  static RNG instance(DEFAULT_SEED);
  return instance;
}

/**
 * @brief 设置全局 RNG 种子
 */
void RNG::setSeed(uint64_t seed) { global().reset(seed); }

/**
 * @brief 重置到初始种子状态
 */
void RNG::reset() { engine_.seed(seed_); }

/**
 * @brief 使用新种子重置
 *
 * @param newSeed 新种子，如果传入 0，则使用 `std::random_device`
 * 获取物理/系统随机熵源
 */
void RNG::reset(uint64_t newSeed) {
  if (newSeed == 0) {
    std::random_device rd;
    seed_ = rd();
  } else {
    seed_ = newSeed;
  }
  engine_.seed(seed_);
}

real64_t RNG::uniform() {
  std::uniform_real_distribution<real64_t> dist(0.0, 1.0);
  return dist(engine_);
}

real64_t RNG::uniform(real64_t a, real64_t b) {
  std::uniform_real_distribution<real64_t> dist(a, b);
  return dist(engine_);
}

real64_t RNG::gaussian(real64_t mean, real64_t stddev) {
  std::normal_distribution<real64_t> dist(mean, stddev);
  return dist(engine_);
}

complex64_t RNG::gaussianComplex(real64_t stddev) {
  // 实部虚部各分担一半功率：stddev_component = stddev / sqrt(2)
  // E[|Z|^2] = E[X^2] + E[Y^2] = sigma^2/2 + sigma^2/2 = sigma^2
  real64_t const sigma = stddev / std::sqrt(2.0);
  std::normal_distribution<real64_t> dist(0.0, sigma);
  return {dist(engine_), dist(engine_)};
}

uint8_t RNG::bit() {
  std::uniform_int_distribution<uint8_t> dist(0, 1);
  return dist(engine_);
}

/// @}

}  // namespace prism::simulation
