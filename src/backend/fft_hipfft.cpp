/**
 * @file fft_hipfft.cpp
 * @ingroup backend
 * @brief hipFFT FFT 后端实现 (AMD ROCm)
 *
 * TODO: 使用 hipFFT API 实现 FFT
 */

#include "prism/backend/fft_backend.h"

#if defined(PRISM_HAS_HIPFFT)

#include <hipfft/hipfft.h>

#include <stdexcept>

namespace prism::backend {

/// @addtogroup backend
/// @{

/// @brief hipFFT 后端占位实现，接口预留以便后续接入 ROCm
class HipFFTBackend : public FFTBackend {
 public:
  bool is_available() const override { return true; }
  const char* name() const override { return "hipFFT"; }

  void forwardC2cImpl(complex32_t* data, int64_t n) override {
    // TODO: 实现 hipfftExecC2C
    throw std::runtime_error("hipFFT backend not implemented yet");
  }

  void forwardC2cImpl(complex64_t* data, int64_t n) override {
    throw std::runtime_error("hipFFT backend not implemented yet");
  }

  void inverse_c2c_impl(complex32_t* data, int64_t n, bool normalize) override {
    throw std::runtime_error("hipFFT backend not implemented yet");
  }

  void inverse_c2c_impl(complex64_t* data, int64_t n, bool normalize) override {
    throw std::runtime_error("hipFFT backend not implemented yet");
  }

  void forwardR2cImpl(const real32_t* in, complex32_t* out,
                      int64_t n) override {
    throw std::runtime_error("hipFFT backend not implemented yet");
  }

  void forwardR2cImpl(const real64_t* in, complex64_t* out,
                      int64_t n) override {
    throw std::runtime_error("hipFFT backend not implemented yet");
  }

  void inverseC2rImpl(const complex32_t* in, real32_t* out, int64_t n,
                      bool normalize) override {
    throw std::runtime_error("hipFFT backend not implemented yet");
  }

  void inverseC2rImpl(const complex64_t* in, real64_t* out, int64_t n,
                      bool normalize) override {
    throw std::runtime_error("hipFFT backend not implemented yet");
  }
};

static HipFFTBackend g_hipfft_backend;

FFTBackend& getHipfftBackend() { return g_hipfft_backend; }

/// @}

}  // namespace prism::backend

#endif  // PRISM_HAS_HIPFFT
