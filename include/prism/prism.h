/**
 * @file prism.h
 * @ingroup core
 * @brief PRISM 主入口与生命周期管理
 *
 * 暴露初始化、关闭与后端查询接口，建议所有用户代码包含本头文件。
 */

#ifndef PRISM_PRISM_H
#define PRISM_PRISM_H

#include "prism/backend/backend_registry.h"

namespace prism {

/// @addtogroup core
/// @{

/**
 * @brief 版本信息常量
 *
 * 可在运行时查询以实现兼容性检查或日志输出。
 */
struct Version {
  static constexpr int MAJOR = 0;  ///< 主版本号（破坏性更新时递增）
  static constexpr int MINOR = 1;  ///< 次版本号（新增特性）
  static constexpr int PATCH = 0;  ///< 补丁号（兼容性修复）
  static constexpr const char* STRING = "0.1.0";  ///< 语义化版本字符串
};

/**
 * @brief 初始化 PRISM 库
 *
 * 当前逻辑主要用于选择默认后端，预留后续资源初始化的扩展点。
 * 建议在应用启动时调用，搭配 @ref shutdown 使用。
 *
 * @return 初始化成功返回 true
 */
inline bool initialize() {
  setBackendPreference(BackendType::AUTO);
  return true;
}

/**
 * @brief 释放 PRISM 资源
 *
 * 目前无需显式释放，但保留接口便于未来添加后端资源回收。
 */
inline void shutdown() {
  // 当前无需释放资源
}

/**
 * @brief 获取当前后端类型
 * @return 后端枚举值，详见 @ref prism::BackendType
 */
inline BackendType getBackend() { return backendInUse(); }

/**
 * @brief 获取后端名称（便于日志输出与调试）
 * @return 后端名称字符串
 */
inline const char* getBackendName() { return backendName(getBackend()); }

/// @}

}  // namespace prism

#endif  // PRISM_PRISM_H
