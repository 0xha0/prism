/**
 * @file fft_hipfft.cpp
 * @ingroup backend
 * @brief hipFFT 后端实现 (AMD ROCm)
 *
 * 提供了基于 AMD hipFFT 库的 FFT 实现
 * @note 当前仅为占位符实现，尚未绑定实际 hipFFT API
 */

#ifdef PRISM_HAS_HIPFFT

#include <hipfft/hipfft.h>

#include <stdexcept>

#include "prism/backend/fft_backend.h"

namespace prism::backend {

/// @addtogroup backend
/// @{

/// @brief hipFFT 后端占位类
class HipFFTBackend : public FFTBackend {
 public:
  [[nodiscard]] bool isAvailable() const override { return false; }
  [[nodiscard]] const char* name() const override { return "hipFFT"; }
  [[nodiscard]] bool supports(ScalarType precision, FftTransType /*type*/) const override {
    if (!isFloatType(precision)) return false;
    return false;
  }

  void forwardC2cImpl(complex32_t* data, int64_t n) override {
    // TODO: 调用 hipfftExecC2C
    throw std::runtime_error("hipFFT backend not implemented yet");
  }

  void forwardC2cImpl(complex64_t* data, int64_t n) override {
    throw std::runtime_error("hipFFT backend not implemented yet");
  }

  void inverseC2cImpl(complex32_t* data, int64_t n, bool normalize) override {
    throw std::runtime_error("hipFFT backend not implemented yet");
  }

  void inverseC2cImpl(complex64_t* data, int64_t n, bool normalize) override {
    throw std::runtime_error("hipFFT backend not implemented yet");
  }

  void forwardR2cImpl(const real32_t* in, complex32_t* out, int64_t n) override {
    throw std::runtime_error("hipFFT backend not implemented yet");
  }

  void forwardR2cImpl(const real64_t* in, complex64_t* out, int64_t n) override {
    throw std::runtime_error("hipFFT backend not implemented yet");
  }

  void inverseC2rImpl(const complex32_t* in, real32_t* out, int64_t n, bool normalize) override {
    throw std::runtime_error("hipFFT backend not implemented yet");
  }

  void inverseC2rImpl(const complex64_t* in, real64_t* out, int64_t n, bool normalize) override {
    throw std::runtime_error("hipFFT backend not implemented yet");
  }
};

static HipFFTBackend g_hipfft_backend;

FFTBackend& getHipfftBackend() { return g_hipfft_backend; }

/// @}

}  // namespace prism::backend

#endif  // PRISM_HAS_HIPFFT
