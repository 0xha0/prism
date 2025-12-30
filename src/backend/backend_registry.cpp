/**
 * @file backend_registry.cpp
 * @ingroup backend
 * @brief 后端选择与名称查询实现
 */

#include "prism/backend/backend_registry.h"

namespace prism {

/// @addtogroup backend
/// @{

namespace {
// NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables)
BackendType gPreference = BackendType::AUTO;
BackendType gSelected = BackendType::CPU;
// NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables)

/** @brief 根据编译宏自动选择后端（无硬件探测，仅宏层级） */
BackendType autoSelectBackend() {
#ifdef PRISM_HAS_VDSP
  return BackendType::VDSP;
#elif defined(PRISM_HAS_CUFFT)
  return BackendType::Cuda;
#elif defined(PRISM_HAS_HIPFFT)
  return BackendType::Hip;
#elif defined(PRISM_HAS_VKFFT)
  return BackendType::VK_FFT;
#else
  return BackendType::CPU;
#endif
}
}  // namespace

/**
 * @brief 设置后端偏好并更新当前选择
 *
 * 传入 AUTO 时按编译宏顺序自动选择，否则直接使用用户指定的后端。
 */
void setBackendPreference(BackendType type) {
  gPreference = type;
  if (gPreference == BackendType::AUTO) {
    gSelected = autoSelectBackend();
  } else {
    gSelected = gPreference;
  }
}

/** @brief 返回当前在用的后端枚举 */
BackendType backendInUse() { return gSelected; }

/**
 * @brief 获取后端名称字符串
 * @param type 目标后端
 */
const char* backendName(BackendType type) {
  switch (type) {
    case BackendType::AUTO:
      return "Auto";
    case BackendType::CPU:
      return "CPU";
    case BackendType::VDSP:
      return "vDSP";
    case BackendType::CUDA:
      return "CUDA";
    case BackendType::HIP:
      return "HIP";
    case BackendType::VK_FFT:
      return "vkFFT";
    default:
      return "Unknown";
  }
}

/// @}

}  // namespace prism
