/**
 * @file schedule_config.h
 * @ingroup runtime
 * @brief Halide 调度策略配置
 *
 * 提供自适应和自定义两种模式配置 GPU tile 大小、CPU 向量化宽度等调度参数。
 * 默认自动模式会根据数据规模选择最优参数，用户也可显式指定以针对特定硬件优化。
 *
 * @par 使用示例 - 自动模式（默认）
 * @code
 * // 无需配置，系统自动根据数据长度选择最优参数
 * auto result = Executor::run<real32_t>(signal);
 * @endcode
 *
 * @par 使用示例 - 自定义模式
 * @code
 * ScheduleConfig config;
 * config.gpuTileSize = 512;       // 自定义 GPU tile 大小
 * config.cpuVectorWidth = 8;      // 自定义 CPU 向量化宽度
 * config.cpuParallel = true;      // 启用 CPU 并行
 *
 * Executor::setScheduleConfig(config);
 * auto result = Executor::run<real32_t>(signal);
 *
 * // 恢复自动模式
 * Executor::setScheduleConfig(ScheduleConfig{});
 * @endcode
 */

#ifndef PRISM_RUNTIME_SCHEDULECONFIG_H
#define PRISM_RUNTIME_SCHEDULECONFIG_H

#include <Halide.h>

#include <cstdint>

namespace prism::runtime {

/// @addtogroup runtime
/// @{

/**
 * @brief Halide 调度策略配置结构体
 *
 * 用于控制 GPU tiling、CPU 向量化与并行等调度行为。
 * 所有值为 0 时表示自动模式，系统将根据数据规模自适应选择。
 *
 * @note 公共成员设计便于用户直接配置，符合 POD 配置结构体惯例
 */
// NOLINTBEGIN(misc-non-private-member-variables-in-classes)
struct ScheduleConfig {
  // NOLINTNEXTLINE(readability-identifier-naming)
  static constexpr int kDefaultGpuMinExtent = 64;
  // NOLINTNEXTLINE(readability-identifier-naming)
  static constexpr int kDefaultCpuParallelThreshold = 1024;

  // ==========================================================================
  // GPU 调度参数
  // ==========================================================================

  /**
   * @brief GPU tile 大小
   *
   * - 0 = 自动模式，根据数据长度选择（64/128/256/512）
   * - 非 0 = 使用指定值（建议为 2 的幂次：32/64/128/256/512/1024）
   *
   * @note 数据长度小于 gpuMinExtent 时不会应用 tiling
   */
  int gpuTileSize = 0;

  /**
   * @brief 使用 GPU tiling 的最小数据长度
   *
   * 小于此值的数据将跳过 tiling，避免调度开销超过计算收益。
   * 默认 64，可根据实际 GPU 调整。
   */
  int gpuMinExtent = kDefaultGpuMinExtent;

  // ==========================================================================
  // CPU 调度参数
  // ==========================================================================

  /**
   * @brief CPU 向量化宽度（SIMD lane 数量）
   *
   * - 0 = 自动模式（float 默认 8，double 默认 4）
   * - 非 0 = 使用指定值（建议：4/8/16，需匹配硬件 SIMD 宽度）
   *
   * @note SSE/NEON = 4, AVX/AVX2 = 8, AVX-512 = 16
   */
  int cpuVectorWidth = 0;

  /**
   * @brief 是否启用 CPU 多线程并行
   *
   * 启用后会在外层循环应用 parallel() 调度，利用多核。
   * 仅当数据长度 >= cpuParallelThreshold 时生效。
   */
  bool cpuParallel = true;

  /**
   * @brief CPU 并行的最小数据长度阈值
   *
   * 小于此值的数据不启用并行，避免线程调度开销。
   * 默认 1024。
   */
  int cpuParallelThreshold = kDefaultCpuParallelThreshold;

  // ==========================================================================
  // 自动调度策略方法
  // ==========================================================================

  /**
   * @brief 获取实际 GPU tile 大小
   *
   * 当 gpuTileSize = 0 时根据 extent 自动选择，否则返回用户指定值。
   * 若 extent < gpuMinExtent，返回 0 表示不使用 tiling。
   *
   * @param extent 数据长度
   * @return 应使用的 tile 大小，0 = 不使用 tiling
   */
  [[nodiscard]] int getEffectiveGpuTileSize(int64_t extent) const;

  /**
   * @brief 获取实际 CPU 向量化宽度
   *
   * 当 cpuVectorWidth = 0 时根据 extent 自动选择，否则返回用户指定值。
   *
   * @param extent 数据长度
   * @return 应使用的向量化宽度，1 = 不向量化
   */
  [[nodiscard]] int getEffectiveCpuVectorWidth(int64_t extent) const;

  /**
   * @brief 判断是否应启用 CPU 并行
   *
   * @param extent 数据长度
   * @return true 如果应启用并行
   */
  [[nodiscard]] bool shouldEnableCpuParallel(int64_t extent) const;

  // ==========================================================================
  // 调度应用方法
  // ==========================================================================

  /**
   * @brief 对 Halide Func 应用 GPU 调度
   *
   * 根据配置对 result Func 应用合适的 gpu_tile 调度。
   * 若 extent < gpuMinExtent 或 tile = 0，不应用任何调度。
   *
   * @param result 要调度的 Halide Func
   * @param extent 数据长度
   */
  void applyGpuSchedule(Halide::Func& result, int64_t extent) const;

  /**
   * @brief 对 Halide Func 应用 CPU 调度
   *
   * 根据配置应用向量化和/或并行调度。
   *
   * @param result 要调度的 Halide Func
   * @param extent 数据长度
   */
  void applyCpuSchedule(Halide::Func& result, int64_t extent) const;
};
// NOLINTEND(misc-non-private-member-variables-in-classes)

/// @}

}  // namespace prism::runtime

#endif  // PRISM_RUNTIME_SCHEDULECONFIG_H
