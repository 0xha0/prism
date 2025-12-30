/**
 * @file fft_cufft.cpp
 * @ingroup backend
 * @brief cuFFT FFT 后端实现 (NVIDIA CUDA)
 *
 * TODO: 使用 cuFFT API 实现 FFT
 */

#include "prism/backend/fft_backend.h"

#if defined(PRISM_HAS_CUFFT)

#include <cufft.h>

#include <stdexcept>

namespace prism::backend {

/// @addtogroup backend
/// @{

/// @brief cuFFT 后端占位实现，当前仅抛出未实现异常
class CuFFTBackend : public FFTBackend {
 public:
  bool is_available() const override { return true; }
  const char* name() const override { return "cuFFT"; }

  void forwardC2cImpl(complex32_t* data, int64_t n) override {
    // TODO: 实现 cufftExecC2C
    throw std::runtime_error("cuFFT backend not implemented yet");
  }

  void forwardC2cImpl(complex64_t* data, int64_t n) override {
    throw std::runtime_error("cuFFT backend not implemented yet");
  }

  void inverse_c2c_impl(complex32_t* data, int64_t n, bool normalize) override {
    throw std::runtime_error("cuFFT backend not implemented yet");
  }

  void inverse_c2c_impl(complex64_t* data, int64_t n, bool normalize) override {
    throw std::runtime_error("cuFFT backend not implemented yet");
  }

  void forwardR2cImpl(const real32_t* in, complex32_t* out,
                      int64_t n) override {
    throw std::runtime_error("cuFFT backend not implemented yet");
  }

  void forwardR2cImpl(const real64_t* in, complex64_t* out,
                      int64_t n) override {
    throw std::runtime_error("cuFFT backend not implemented yet");
  }

  void inverseC2rImpl(const complex32_t* in, real32_t* out, int64_t n,
                      bool normalize) override {
    throw std::runtime_error("cuFFT backend not implemented yet");
  }

  void inverseC2rImpl(const complex64_t* in, real64_t* out, int64_t n,
                      bool normalize) override {
    throw std::runtime_error("cuFFT backend not implemented yet");
  }
};

static CuFFTBackend g_cufft_backend;

FFTBackend& getCufftBackend() { return g_cufft_backend; }

/// @}

}  // namespace prism::backend

#endif  // PRISM_HAS_CUFFT
