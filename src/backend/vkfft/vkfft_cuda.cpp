/**
 * @file vkfft_cuda.cpp
 * @ingroup backend
 * @brief vkFFT CUDA 后端实现
 *
 * 此后端在 VKFFT_BACKEND=1 时编译
 * TODO: 实现 vkFFT CUDA 后端
 */

#include "prism/backend/fft_backend.h"

#if defined(PRISM_HAS_VKFFT) && (VKFFT_BACKEND == 1)

#include <cuda_runtime.h>

#include <stdexcept>
#include <vector>

#include "vkFFT.h"

namespace prism::backend {

/// @addtogroup backend
/// @{

/// @brief vkFFT CUDA 后端占位实现，接口预留以便后续接入
class VkFFTCudaBackend : public FFTBackend {
 public:
  [[nodiscard]] bool isAvailable() const override { return false; }
  [[nodiscard]] const char* name() const override { return "vkFFT (CUDA)"; }
  [[nodiscard]] bool supports(ScalarType precision, FftTransType /*type*/) const override {
    if (!isFloatType(precision)) return false;
    return false;
  }

  void forwardC2cImpl(complex32_t* data, int64_t n) override {
    // TODO: 实现 vkFFT CUDA C2C
    throw std::runtime_error("vkFFT CUDA backend not implemented yet");
  }

  void forwardC2cImpl(complex64_t* data, int64_t n) override {
    throw std::runtime_error("vkFFT CUDA backend not implemented yet");
  }

  void inverseC2cImpl(complex32_t* data, int64_t n, bool normalize) override {
    throw std::runtime_error("vkFFT CUDA backend not implemented yet");
  }

  void inverseC2cImpl(complex64_t* data, int64_t n, bool normalize) override {
    throw std::runtime_error("vkFFT CUDA backend not implemented yet");
  }

  void forwardR2cImpl(const real32_t* in, complex32_t* out, int64_t n) override {
    throw std::runtime_error("vkFFT CUDA backend not implemented yet");
  }

  void forwardR2cImpl(const real64_t* in, complex64_t* out, int64_t n) override {
    throw std::runtime_error("vkFFT CUDA backend not implemented yet");
  }

  void inverseC2rImpl(const complex32_t* in, real32_t* out, int64_t n, bool normalize) override {
    throw std::runtime_error("vkFFT CUDA backend not implemented yet");
  }

  void inverseC2rImpl(const complex64_t* in, real64_t* out, int64_t n, bool normalize) override {
    throw std::runtime_error("vkFFT CUDA backend not implemented yet");
  }
};

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
static VkFFTCudaBackend gVkfftCudaBackend;

FFTBackend& getVkfftBackend() { return gVkfftCudaBackend; }

/// @}

}  // namespace prism::backend

#endif  // PRISM_HAS_VKFFT && VKFFT_BACKEND == 1
