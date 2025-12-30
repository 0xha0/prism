/**
 * @file fft_vdsp.cpp
 * @ingroup backend
 * @brief vDSP FFT 后端实现 (macOS Accelerate Framework)
 */

#include <vecLib/vDSP.h>

#include <cstdint>

#include "prism/backend/fft_backend.h"
#include "prism/types.h"

#if defined(PRISM_HAS_VDSP)

#include <cmath>
#include <stdexcept>
#include <vector>

namespace prism::backend {

/// @addtogroup backend
/// @{

class VDSPFFTBackend : public FFTBackend {
 public:
  [[nodiscard]] bool isAvailable() const override { return true; }
  [[nodiscard]] const char* name() const override { return "vDSP"; }

  [[nodiscard]] bool supports(ScalarType precision,
                              FftTransType /*type*/) const override {
    // vDSP supports both single and double precision
    return precision == ScalarType::F32 || precision == ScalarType::F64;
  }

  void forwardC2cImpl(complex32_t* data, int64_t n) override {
    if (n <= 0 || (n & (n - 1)) != 0) {
      throw std::invalid_argument("FFT length must be power of 2");
    }
    int const log2n = static_cast<int>(std::log2(n));
    FFTSetup setup = vDSP_create_fftsetup(log2n, FFT_RADIX2);
    if (!setup) throw std::runtime_error("Failed to create vDSP FFT setup");

    DSPSplitComplex split;
    std::vector<real32_t> realPart(n);
    std::vector<real32_t> imagPart(n);
    for (int64_t i = 0; i < n; ++i) {
      realPart[i] = data[i].real();
      imagPart[i] = data[i].imag();
    }
    split.realp = realPart.data();
    split.imagp = imagPart.data();
    vDSP_fft_zip(setup, &split, 1, log2n, FFT_FORWARD);
    for (int64_t i = 0; i < n; ++i) {
      data[i] = complex32_t(realPart[i], imagPart[i]);
    }
    vDSP_destroy_fftsetup(setup);
  }

  void forwardC2cImpl(complex64_t* data, int64_t n) override {
    if (n <= 0 || (n & (n - 1)) != 0) {
      throw std::invalid_argument("FFT length must be power of 2");
    }
    int const log2n = static_cast<int>(std::log2(n));
    FFTSetupD setup = vDSP_create_fftsetupD(log2n, FFT_RADIX2);
    if (!setup) throw std::runtime_error("Failed to create vDSP FFT setup");

    DSPDoubleSplitComplex split;
    std::vector<real64_t> realPart(n);
    std::vector<real64_t> imagPart(n);
    for (int64_t i = 0; i < n; ++i) {
      realPart[i] = data[i].real();
      imagPart[i] = data[i].imag();
    }
    split.realp = realPart.data();
    split.imagp = imagPart.data();
    vDSP_fft_zipD(setup, &split, 1, log2n, FFT_FORWARD);
    for (int64_t i = 0; i < n; ++i) {
      data[i] = complex64_t(realPart[i], imagPart[i]);
    }
    vDSP_destroy_fftsetupD(setup);
  }

  void inverseC2cImpl(complex32_t* data, int64_t n, bool normalize) override {
    if (n <= 0 || (n & (n - 1)) != 0) {
      throw std::invalid_argument("FFT length must be power of 2");
    }
    int const log2n = static_cast<int>(std::log2(n));
    FFTSetup setup = vDSP_create_fftsetup(log2n, FFT_RADIX2);
    if (!setup) throw std::runtime_error("Failed to create vDSP FFT setup");

    DSPSplitComplex split;
    std::vector<real32_t> realPart(n);
    std::vector<real32_t> imagPart(n);
    for (int64_t i = 0; i < n; ++i) {
      realPart[i] = data[i].real();
      imagPart[i] = data[i].imag();
    }
    split.realp = realPart.data();
    split.imagp = imagPart.data();
    vDSP_fft_zip(setup, &split, 1, log2n, FFT_INVERSE);
    if (normalize) {
      real32_t const scale = 1.0F / static_cast<real32_t>(n);
      vDSP_vsmul(realPart.data(), 1, &scale, realPart.data(), 1, n);
      vDSP_vsmul(imagPart.data(), 1, &scale, imagPart.data(), 1, n);
    }
    for (int64_t i = 0; i < n; ++i) {
      data[i] = complex32_t(realPart[i], imagPart[i]);
    }
    vDSP_destroy_fftsetup(setup);
  }

  void inverseC2cImpl(complex64_t* data, int64_t n, bool normalize) override {
    if (n <= 0 || (n & (n - 1)) != 0) {
      throw std::invalid_argument("FFT length must be power of 2");
    }
    int const log2n = static_cast<int>(std::log2(n));
    FFTSetupD setup = vDSP_create_fftsetupD(log2n, FFT_RADIX2);
    if (!setup) throw std::runtime_error("Failed to create vDSP FFT setup");

    DSPDoubleSplitComplex split;
    std::vector<real64_t> realPart(n);
    std::vector<real64_t> imagPart(n);
    for (int64_t i = 0; i < n; ++i) {
      realPart[i] = data[i].real();
      imagPart[i] = data[i].imag();
    }
    split.realp = realPart.data();
    split.imagp = imagPart.data();
    vDSP_fft_zipD(setup, &split, 1, log2n, FFT_INVERSE);
    if (normalize) {
      real64_t const scale = 1.0 / static_cast<real64_t>(n);
      vDSP_vsmulD(realPart.data(), 1, &scale, realPart.data(), 1, n);
      vDSP_vsmulD(imagPart.data(), 1, &scale, imagPart.data(), 1, n);
    }
    for (int64_t i = 0; i < n; ++i) {
      data[i] = complex64_t(realPart[i], imagPart[i]);
    }
    vDSP_destroy_fftsetupD(setup);
  }

  void forwardR2cImpl(const real32_t* in, complex32_t* out,
                      int64_t n) override {
    if (n <= 0 || (n & (n - 1)) != 0) {
      throw std::invalid_argument("FFT length must be power of 2");
    }
    int const log2n = static_cast<int>(std::log2(n));
    FFTSetup setup = vDSP_create_fftsetup(log2n, FFT_RADIX2);
    if (!setup) throw std::runtime_error("Failed to create vDSP FFT setup");

    std::vector<real32_t> realPart(n / 2);
    std::vector<real32_t> imagPart(n / 2);
    DSPSplitComplex const split = {realPart.data(), imagPart.data()};
    vDSP_ctoz(reinterpret_cast<const DSPComplex*>(in), 2, &split, 1, n / 2);
    vDSP_fft_zrip(setup, &split, 1, log2n, FFT_FORWARD);
    out[0] = complex32_t(split.realp[0], 0.0F);
    for (int64_t i = 1; i < n / 2; ++i) {
      out[i] = complex32_t(split.realp[i], split.imagp[i]);
    }
    out[n / 2] = complex32_t(split.imagp[0], 0.0F);
    vDSP_destroy_fftsetup(setup);
  }

  void forwardR2cImpl(const real64_t* in, complex64_t* out,
                      int64_t n) override {
    if (n <= 0 || (n & (n - 1)) != 0) {
      throw std::invalid_argument("FFT length must be power of 2");
    }
    int const log2n = static_cast<int>(std::log2(n));
    FFTSetupD setup = vDSP_create_fftsetupD(log2n, FFT_RADIX2);
    if (!setup) throw std::runtime_error("Failed to create vDSP FFT setup");

    std::vector<real64_t> realPart(n / 2);
    std::vector<real64_t> imagPart(n / 2);
    DSPDoubleSplitComplex const split = {realPart.data(), imagPart.data()};
    vDSP_ctozD(reinterpret_cast<const DSPDoubleComplex*>(in), 2, &split, 1,
               n / 2);
    vDSP_fft_zripD(setup, &split, 1, log2n, FFT_FORWARD);
    out[0] = complex64_t(split.realp[0], 0.0);
    for (int64_t i = 1; i < n / 2; ++i) {
      out[i] = complex64_t(split.realp[i], split.imagp[i]);
    }
    out[n / 2] = complex64_t(split.imagp[0], 0.0);
    vDSP_destroy_fftsetupD(setup);
  }

  void inverseC2rImpl(const complex32_t* in, real32_t* out, int64_t n,
                      bool normalize) override {
    if (n <= 0 || (n & (n - 1)) != 0) {
      throw std::invalid_argument("FFT length must be power of 2");
    }
    int const log2n = static_cast<int>(std::log2(n));
    FFTSetup setup = vDSP_create_fftsetup(log2n, FFT_RADIX2);
    if (!setup) throw std::runtime_error("Failed to create vDSP FFT setup");

    std::vector<real32_t> realPart(n / 2);
    std::vector<real32_t> imagPart(n / 2);
    DSPSplitComplex const split = {realPart.data(), imagPart.data()};
    split.realp[0] = in[0].real();
    split.imagp[0] = in[n / 2].real();
    for (int64_t i = 1; i < n / 2; ++i) {
      split.realp[i] = in[i].real();
      split.imagp[i] = in[i].imag();
    }
    vDSP_fft_zrip(setup, &split, 1, log2n, FFT_INVERSE);
    vDSP_ztoc(&split, 1, reinterpret_cast<DSPComplex*>(out), 2, n / 2);
    if (normalize) {
      real32_t const scale = 0.5F / static_cast<real32_t>(n);
      vDSP_vsmul(out, 1, &scale, out, 1, n);
    }
    vDSP_destroy_fftsetup(setup);
  }

  void inverseC2rImpl(const complex64_t* in, real64_t* out, int64_t n,
                      bool normalize) override {
    if (n <= 0 || (n & (n - 1)) != 0) {
      throw std::invalid_argument("FFT length must be power of 2");
    }
    int const log2n = static_cast<int>(std::log2(n));
    FFTSetupD setup = vDSP_create_fftsetupD(log2n, FFT_RADIX2);
    if (!setup) throw std::runtime_error("Failed to create vDSP FFT setup");

    std::vector<real64_t> realPart(n / 2);
    std::vector<real64_t> imagPart(n / 2);
    DSPDoubleSplitComplex const split = {realPart.data(), imagPart.data()};
    split.realp[0] = in[0].real();
    split.imagp[0] = in[n / 2].real();
    for (int64_t i = 1; i < n / 2; ++i) {
      split.realp[i] = in[i].real();
      split.imagp[i] = in[i].imag();
    }
    vDSP_fft_zripD(setup, &split, 1, log2n, FFT_INVERSE);
    vDSP_ztocD(&split, 1, reinterpret_cast<DSPDoubleComplex*>(out), 2, n / 2);
    if (normalize) {
      real64_t const scale = 0.5 / static_cast<real64_t>(n);
      vDSP_vsmulD(out, 1, &scale, out, 1, n);
    }
    vDSP_destroy_fftsetupD(setup);
  }
};

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
static VDSPFFTBackend gVdspBackend;

FFTBackend& getVdspBackend() { return gVdspBackend; }

/// @}

}  // namespace prism::backend

#endif  // PRISM_HAS_VDSP
