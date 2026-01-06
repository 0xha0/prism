/**
 * @file backend_registry.h
 * @ingroup backend
 * @brief FFT 后端注册与选择
 *
 * 负责管理运行时可用的 FFT 后端，并提供偏好设置接口
 * 系统会根据编译选项（是否启用 CUDA/ROCm/VkFFT）和运行时硬件检测结果，
 * 动态选择最适合的后端
 */

#ifndef PRISM_BACKEND_REGISTRY_H
#define PRISM_BACKEND_REGISTRY_H

#include <cstdint>

namespace prism {

/// @addtogroup backend
/// @{

/**
 * @brief 可用的 FFT 后端类型
 *
 * 定义了系统支持的各种 FFT 实现库
 */
enum class FftBackendType : std::uint8_t {
  AUTO,   ///< 自动选择最优后端（尝试顺序：CUDA/HIP -> VkFFT/vDSP -> CPU）
  VDSP,   ///< Apple Accelerate vDSP（仅 macOS/iOS 可用，性能优异）
  CUDA,   ///< NVIDIA cuFFT（仅在 NVIDIA GPU 上可用）
  HIP,    ///< AMD hipFFT/rocFFT（仅在 AMD GPU 上可用）
  VK_FFT  ///< VkFFT（通用于 Vulkan/OpenCL/Metal 的高性能FFT）
};

/**
 * @brief 设置 FFT 后端偏好
 *
 * 提示 Runtime 在初始化或首次调用 FFT 时优先尝试指定的后端
 *
 * @param type 用户偏好的后端类型
 * @note 此设置只是"提示"，若硬件不支持或未编译对应后端，
 *       系统仍会自动回退到其他可用后端（例如 CPU）
 * @note 建议在 `prism::initialize()` 之前调用此函数
 */
void setFftBackendPreference(FftBackendType type);

/**
 * @brief 获取当前实际生效的 FFT 后端
 *
 * @return 当前系统决定使用的后端类型（例如 CUDA 或 VDSP）
 *         若尚未初始化，返回默认值
 */
FftBackendType getFftBackendInUseType();

/**
 * @brief 获取后端的字符串名称
 *
 * 用于日志记录与调试，例如 "CUDA (cuFFT)", "vDSP", "Cpu (PocketFFT)" 等
 *
 * @param type 后端枚举值
 * @return 对应的可读 C 风格字符串
 */
const char* getFftBackendName(FftBackendType type);

/// @}

}  // namespace prism

#endif  // PRISM_BACKEND_REGISTRY_H
