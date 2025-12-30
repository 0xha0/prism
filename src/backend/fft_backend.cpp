/**
 * @file fft_backend.cpp
 * @ingroup backend
 * @brief FFT 后端选择器
 *
 * 根据编译时宏选择可用的 FFT 后端：
 * - PRISM_HAS_VDSP:   macOS vDSP (优先)
 * - PRISM_HAS_CUFFT:  NVIDIA cuFFT
 * - PRISM_HAS_HIPFFT: AMD hipFFT
 * - PRISM_HAS_VKFFT:  vkFFT (Metal/CUDA/HIP fallback)
 */

#include "prism/backend/fft_backend.h"

namespace prism::backend {

/// @addtogroup backend
/// @{

// 各后端导出的获取函数（由对应源文件实现）
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
 * @brief 获取当前可用的 FFT 后端
 *
 * 优先级：vDSP > cuFFT > hipFFT > vkFFT > Stub
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
  // Stub 后端
  static class StubFFTBackend : public FFTBackend {
   public:
    /// @brief 不可用的占位后端，用于在无依赖环境下提供一致接口
    bool is_available() const override { return false; }
    const char* name() const override { return "Stub (no backend)"; }
    void forward_c2c_32(complex32_t*, int64_t) override {
      throw std::runtime_error("No FFT backend");
    }
    void forward_c2c_64(complex64_t*, int64_t) override {
      throw std::runtime_error("No FFT backend");
    }
    void inverse_c2c_32(complex32_t*, int64_t, bool) override {
      throw std::runtime_error("No FFT backend");
    }
    void inverse_c2c_64(complex64_t*, int64_t, bool) override {
      throw std::runtime_error("No FFT backend");
    }
    void forward_r2c_32(const real32_t*, complex32_t*, int64_t) override {
      throw std::runtime_error("No FFT backend");
    }
    void forward_r2c_64(const real64_t*, complex64_t*, int64_t) override {
      throw std::runtime_error("No FFT backend");
    }
    void inverse_c2r_32(const complex32_t*, real32_t*, int64_t, bool) override {
      throw std::runtime_error("No FFT backend");
    }
    void inverse_c2r_64(const complex64_t*, real64_t*, int64_t, bool) override {
      throw std::runtime_error("No FFT backend");
    }
  } stub;
  return stub;
#endif
}

/// @}

}  // namespace prism::backend
