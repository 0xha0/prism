/**
 * @file fft_cufft.cpp
 * @ingroup backend
 * @brief cuFFT 后端实现 (NVIDIA CUDA)
 *
 * 提供了基于 NVIDIA cuFFT 库的 FFT 实现
 * @note 当前仅为占位符实现，尚未绑定实际 cuFFT API
 */

#if defined(PRISM_HAS_CUFFT)

#include <cufft.h>

#include <stdexcept>

#include "prism/backend/fft_backend.h"

namespace prism::backend {

/// @addtogroup backend
/// @{

/// @brief cuFFT 后端占位类
class CuFFTBackend : public FFTBackend {
 public:
  [[nodiscard]] bool isAvailable() const override { return false; }
  [[nodiscard]] const char* name() const override { return "cuFFT"; }
  [[nodiscard]] bool supports(ScalarType precision, FftTransType /*type*/) const override {
    if (!isFloatType(precision)) return false;
    return false;
  }

  void forwardC2cImpl(complex32_t* data, int64_t n) override {
    // TODO: 调用 cufftExecC2C
    throw std::runtime_error("cuFFT backend not implemented yet");
  }

  void forwardC2cImpl(complex64_t* data, int64_t n) override {
    throw std::runtime_error("cuFFT backend not implemented yet");
  }

  void inverseC2cImpl(complex32_t* data, int64_t n, bool normalize) override {
    throw std::runtime_error("cuFFT backend not implemented yet");
  }

  void inverseC2cImpl(complex64_t* data, int64_t n, bool normalize) override {
    throw std::runtime_error("cuFFT backend not implemented yet");
  }

  void forwardR2cImpl(const real32_t* in, complex32_t* out, int64_t n) override {
    throw std::runtime_error("cuFFT backend not implemented yet");
  }

  void forwardR2cImpl(const real64_t* in, complex64_t* out, int64_t n) override {
    throw std::runtime_error("cuFFT backend not implemented yet");
  }

  void inverseC2rImpl(const complex32_t* in, real32_t* out, int64_t n, bool normalize) override {
    throw std::runtime_error("cuFFT backend not implemented yet");
  }

  void inverseC2rImpl(const complex64_t* in, real64_t* out, int64_t n, bool normalize) override {
    throw std::runtime_error("cuFFT backend not implemented yet");
  }
};

static CuFFTBackend g_cufft_backend;

FFTBackend& getCufftBackend() { return g_cufft_backend; }

/// @}

}  // namespace prism::backend

#endif  // PRISM_HAS_CUFFT
