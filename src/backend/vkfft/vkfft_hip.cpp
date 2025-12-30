/**
 * @file vkfft_hip.cpp
 * @ingroup backend
 * @brief vkFFT HIP 后端实现
 *
 * 此后端在 VKFFT_BACKEND=2 时编译
 * TODO: 实现 vkFFT HIP 后端
 */

#include "prism/backend/fft_backend.h"

#if defined(PRISM_HAS_VKFFT) && (VKFFT_BACKEND == 2)

#include <hip/hip_runtime.h>

#include <stdexcept>
#include <vector>

#include "vkFFT.h"

namespace prism::backend {

/// @addtogroup backend
/// @{

/// @brief vkFFT HIP 后端占位实现，待对接 ROCm 平台
class VkFFTHipBackend : public FFTBackend {
 public:
  bool is_available() const override { return true; }
  const char* name() const override { return "vkFFT (HIP)"; }

  void forwardC2cImpl(complex32_t* data, int64_t n) override {
    // TODO: 实现 vkFFT HIP C2C
    throw std::runtime_error("vkFFT HIP backend not implemented yet");
  }

  void forwardC2cImpl(complex64_t* data, int64_t n) override {
    throw std::runtime_error("vkFFT HIP backend not implemented yet");
  }

  void inverse_c2c_impl(complex32_t* data, int64_t n, bool normalize) override {
    throw std::runtime_error("vkFFT HIP backend not implemented yet");
  }

  void inverse_c2c_impl(complex64_t* data, int64_t n, bool normalize) override {
    throw std::runtime_error("vkFFT HIP backend not implemented yet");
  }

  void forwardR2cImpl(const real32_t* in, complex32_t* out,
                      int64_t n) override {
    throw std::runtime_error("vkFFT HIP backend not implemented yet");
  }

  void forwardR2cImpl(const real64_t* in, complex64_t* out,
                      int64_t n) override {
    throw std::runtime_error("vkFFT HIP backend not implemented yet");
  }

  void inverseC2rImpl(const complex32_t* in, real32_t* out, int64_t n,
                      bool normalize) override {
    throw std::runtime_error("vkFFT HIP backend not implemented yet");
  }

  void inverseC2rImpl(const complex64_t* in, real64_t* out, int64_t n,
                      bool normalize) override {
    throw std::runtime_error("vkFFT HIP backend not implemented yet");
  }
};

static VkFFTHipBackend g_vkfft_hip_backend;

FFTBackend& get_vkfft_backend() { return g_vkfft_hip_backend; }

/// @}

}  // namespace prism::backend

#endif  // PRISM_HAS_VKFFT && VKFFT_BACKEND == 2
