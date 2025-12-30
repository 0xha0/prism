/**
 * @file backend_registry.h
 * @ingroup backend
 * @brief 后端选择与注册
 *
 * 负责记录编译期可用的后端并提供运行期的选择接口，当前通过编译宏检测。
 */

#ifndef PRISM_BACKEND_REGISTRY_H
#define PRISM_BACKEND_REGISTRY_H

#include <cstdint>

namespace prism {

/// @addtogroup backend
/// @{

/** @brief 可用后端类型 */
enum class BackendType : std::uint8_t {
  AUTO,   ///< 自动检测（默认）
  CPU,    ///< 纯 CPU 回退
  VDSP,   ///< macOS Accelerate/vDSP
  CUDA,   ///< NVIDIA cuFFT
  HIP,    ///< AMD hipFFT
  VK_FFT  ///< vkFFT（多平台）
};

/**
 * @brief 设置后端偏好（Auto 为自动检测）
 * @param type 目标后端枚举
 */
void setBackendPreference(BackendType type);

/**
 * @brief 获取当前后端类型
 * @return 已选中的后端
 */
BackendType backendInUse();

/**
 * @brief 获取后端名称
 * @param type 后端枚举
 * @return 可读名称
 */
const char* backendName(BackendType type);

/// @}

}  // namespace prism

#endif  // PRISM_BACKEND_REGISTRY_H
