/**
 * @file prism.h
 * @ingroup core
 * @brief PRISM 主入口与生命周期管理
 *
 * 暴露初始化、关闭与后端查询接口，建议所有用户代码显式包含本头文件
 * PRISM 的生命周期管理对于 GPU 后端资源的分配与释放尤为重要
 */

#ifndef PRISM_PRISM_H
#define PRISM_PRISM_H

#include "prism/backend/fft_backend.h"
#include "prism/runtime/executor.h"

namespace prism {

/// @addtogroup core
/// @{

/**
 * @brief 版本信息常量
 *
 * 可在运行时查询以实现兼容性检查或日志输出
 */
struct Version {
  static constexpr int MAJOR = 0;                 ///< 主版本号（破坏性更新时递增）
  static constexpr int MINOR = 1;                 ///< 次版本号（新增特性）
  static constexpr int PATCH = 0;                 ///< 补丁号（兼容性修复）
  static constexpr const char* STRING = "0.1.0";  ///< 语义化版本字符串
};

/**
 * @brief 初始化 PRISM 库
 *
 * 初始化库的全局状态，包括：
 * 1. 自动检测系统硬件能力（CPU/GPU）
 * 2. 设置默认的计算后端（偏好顺序：CUDA > HIP > OpenCL > vDSP > PocketFFT）
 * 3. 预分配必要的全局资源（若有）
 *
 * 建议在应程序主线程启动时尽早调用
 * 该函数应当与 @ref shutdown 配对使用，以确保资源的正确释放
 *
 * @note 目前实现非线程安全，请在主线程串行调用
 *
 * @return 初始化成功返回 true；若发生不可恢复错误（如无法加载后端动态库）返回
 * false
 */
inline bool initialize() {
  prism::backend::setFftBackendPreference(prism::backend::FftBackendType::AUTO);
  return true;
}

/**
 * @brief 释放 PRISM 资源
 *
 * 清理 initialize 阶段分配的全局资源
 * 虽然当前版本可能无需显式释放内存（依赖 RAII），但为了代码的前向兼容性，
 * 强烈建议在程序退出前显式调用此函数
 */
inline void shutdown() {
  // 当前无需释放资源
}

/**
 * @brief 获取当前正在使用的 FFT 后端类型
 *
 * 该类型可能由 initialize() 自动选择，也可能由用户通过
 * setFftBackendPreference() 显式指定
 *
 * @return 后端枚举值，详见 @ref prism::backend::FftBackendType
 *         如果尚未初始化，可能返回默认值或 UNKNOWN
 */
inline prism::backend::FftBackendType getFftBackendType() {
  return prism::backend::getFftBackendInUseType();
}

/**
 * @brief 获取后端名称（便于日志输出与调试）
 * @return 后端名称字符串
 */
inline const char* getFftBackendName() { return prism::backend::getFftBackendInUseName(); }

/**
 * @brief 获取当前 Halide 编译器后端名称
 * @return 后端名称字符串 (e.g. "Auto", "GPU (Metal)")
 */
inline std::string getHalideBackendName() {
  return prism::runtime::Executor::getCurrentTargetName();
}

/// @}

}  // namespace prism

#endif  // PRISM_PRISM_H
