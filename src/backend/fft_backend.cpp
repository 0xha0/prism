/**
 * @file fft_backend.cpp
 * @ingroup backend
 * @brief FFT 后端选择器实现
 *
 * 根据编译时宏定义与运行时偏好设置，动态选择并返回单例的 FFT 后端实例
 * 优先级顺序：
 * - PRISM_HAS_VDSP:   macOS vDSP (优先)
 * - PRISM_HAS_CUFFT:  NVIDIA cuFFT
 * - PRISM_HAS_HIPFFT: AMD hipFFT
 * - PRISM_HAS_VKFFT:  vkFFT (Metal/CUDA/HIP fallback)
 * - PRISM_HAS_STUB:   Stub (Fallback)
 */

#include "prism/backend/fft_backend.h"

namespace prism::backend {

/// @addtogroup backend
/// @{

namespace {
// NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables)
/// 全局偏好设置
FftBackendType gFftPreference = FftBackendType::AUTO;
/// 当前实际生效的后端（缓存值）
FftBackendType gFftSelected = FftBackendType::AUTO;
// NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables)

/**
 * @brief 根据编译宏自动选择最优后端
 *
 * 此函数仅基于编译期宏定义进行判断，不进行运行时硬件探测
 */
FftBackendType autoSelectFftBackend() {
#ifdef PRISM_HAS_VDSP
  return FftBackendType::VDSP;
#elif defined(PRISM_HAS_CUFFT)
  return FftBackendType::CUDA;
#elif defined(PRISM_HAS_HIPFFT)
  return FftBackendType::HIP;
#elif defined(PRISM_HAS_VKFFT)
  return FftBackendType::VK_FFT;
#else
  return FftBackendType::STUB;
#endif
}
}  // namespace

/**
 * @brief 设置后端偏好并更新当前选择
 *
 * 若设置为 AUTO，则重新执行自动选择逻辑；否则尝试使用指定的后端
 */
void setFftBackendPreference(FftBackendType type) {
  gFftPreference = type;
  if (gFftPreference == FftBackendType::AUTO) {
    gFftSelected = autoSelectFftBackend();
  } else {
    gFftSelected = gFftPreference;
  }
}

/** @brief 返回当前已激活的后端枚举 */
FftBackendType getFftBackendInUseType() {
  if (gFftSelected == FftBackendType::AUTO) {
    gFftSelected = autoSelectFftBackend();
  }
  return gFftSelected;
}

/**
 * @brief 获取后端名称字符串
 */
const char* getFftBackendInUseName() { return getFftBackend().name(); }

// 各后端实例获取函数的声明
#ifdef PRISM_HAS_VDSP
extern FFTBackend& getVdspBackend();
#endif

#ifdef PRISM_HAS_CUFFT
extern FFTBackend& getCufftBackend();
#endif

#ifdef PRISM_HAS_HIPFFT
extern FFTBackend& getHipfftBackend();
#endif

#ifdef PRISM_HAS_VKFFT
extern FFTBackend& getVkfftBackend();
#endif

/**
 * @brief 获取唯一的全局 FFT 后端实例
 *
 * 根据编译选项分发到具体的单例获取函数，若无任何后端可用，返回 Stub 后端
 */
FFTBackend& getFftBackend() {
#ifdef PRISM_HAS_VDSP
  return getVdspBackend();
#elif defined(PRISM_HAS_CUFFT)
  return getCufftBackend();
#elif defined(PRISM_HAS_HIPFFT)
  return getHipfftBackend();
#elif defined(PRISM_HAS_VKFFT)
  return getVkfftBackend();
#else
  // Stub 后端：用于无 FFT 实现的环境，抛出运行时错误
  static class StubFFTBackend : public FFTBackend {
   public:
    [[nodiscard]] bool isAvailable() const override { return false; }
    [[nodiscard]] const char* name() const override { return "Stub (no backend)"; }
    [[nodiscard]] bool supports(ScalarType precision, FftTransType /*type*/) const override {
      if (!isFloatType(precision)) return false;
      return false;
    }
    void forwardC2cImpl(complex32_t*, int64_t) override {
      throw std::runtime_error("No FFT backend available");
    }
    void forwardC2cImpl(complex64_t*, int64_t) override {
      throw std::runtime_error("No FFT backend available");
    }
    void inverseC2cImpl(complex32_t*, int64_t, bool) override {
      throw std::runtime_error("No FFT backend available");
    }
    void inverseC2cImpl(complex64_t*, int64_t, bool) override {
      throw std::runtime_error("No FFT backend available");
    }
    void forwardR2cImpl(const real32_t*, complex32_t*, int64_t) override {
      throw std::runtime_error("No FFT backend available");
    }
    void forwardR2cImpl(const real64_t*, complex64_t*, int64_t) override {
      throw std::runtime_error("No FFT backend available");
    }
    void inverseC2rImpl(const complex32_t*, real32_t*, int64_t, bool) override {
      throw std::runtime_error("No FFT backend available");
    }
    void inverseC2rImpl(const complex64_t*, real64_t*, int64_t, bool) override {
      throw std::runtime_error("No FFT backend available");
    }
  } stub;
  return stub;
#endif
}

/// @}

}  // namespace prism::backend
