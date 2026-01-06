/**
 * @file schedule_config.h
 * @ingroup runtime
 * @brief Halide 自动调度配置
 *
 * 定义了 Halide Autoscheduler 的配置结构与策略枚举
 * 目前主要支持 Halide 官方 Autoscheduler 的参数透传，暂未开放手动调度接口
 * 在 JIT 默认路径下，通常使用 Mullapudi2016 作为基线调度策略
 */

#ifndef PRISM_RUNTIME_SCHEDULECONFIG_H
#define PRISM_RUNTIME_SCHEDULECONFIG_H

#include <cstdint>
#include <map>
#include <string>

namespace prism::runtime {

/// @addtogroup runtime
/// @{

/**
 * @brief 调度器类型
 *
 * 定义了计算图编译时的调度策略
 */
enum class SchedulerKind : std::uint8_t {
  NONE,   ///< 不应用任何调度（Halide 默认行主序计算）
  AUTO,   ///< 启用 Halide Autoscheduler（自动搜索最优调度）
  MANUAL  ///< 手动调度（预留接口，暂未实现）
};

/**
 * @brief 调度配置参数
 *
 * 用于控制 Halide Autoscheduler 的行为
 */
// NOLINTBEGIN(misc-non-private-member-variables-in-classes)
struct SchedulerConfig {
  /// 调度策略类型
  SchedulerKind kind = SchedulerKind::NONE;

  /// Autoscheduler 名称
  /// @note 若为空，Executor 将根据硬件自动选择默认值：
  /// - CPU: "Adams2019"
  /// - GPU: "Anderson2021"
  std::string name;

  /// 传递给 Autoscheduler 的额外参数
  /// 例如：{{"gpu", "1"}} 用于强制 GPU 调度
  std::map<std::string, std::string> extra;
};
// NOLINTEND(misc-non-private-member-variables-in-classes)

using ScheduleConfig = SchedulerConfig;

/// @}

}  // namespace prism::runtime

#endif  // PRISM_RUNTIME_SCHEDULECONFIG_H
