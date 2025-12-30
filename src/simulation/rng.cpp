/**
 * @file rng.cpp
 * @ingroup simulation
 * @brief 全局随机数生成器实现
 */

#include "prism/simulation/rng.h"

#include <cmath>
#include <cstdint>
#include <random>

#include "prism/types.h"

namespace prism::simulation {

/// @addtogroup simulation
/// @{

/** @brief 使用给定种子构造 RNG，0 则保持默认种子逻辑 */
RNG::RNG(uint64_t seed) : seed_(seed) { reset(seed); }

/** @brief 获取线程安全的全局实例（懒初始化） */
RNG& RNG::global() {
  static RNG instance(DEFAULT_SEED);
  return instance;
}

/** @brief 设置全局种子并重置全局 RNG */
void RNG::setSeed(uint64_t seed) { global().reset(seed); }

/** @brief 重置到当前 seed 状态 */
void RNG::reset() { engine_.seed(seed_); }

/** @brief 使用新的种子重置（0 表示使用随机设备） */
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
  // 实部虚部各 stddev/sqrt(2)，总功率为 stddev^2
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
