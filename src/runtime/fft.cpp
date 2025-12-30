/**
 * @file fft.cpp
 * @ingroup runtime
 * @brief FFT Anchor 实现
 *
 * 调用 fft_backend 模板接口执行 FFT。
 */

#include "prism/runtime/fft.h"

#include <complex>
#include <cstdint>
#include <stdexcept>

#include "prism/backend/fft_backend.h"
#include "prism/types.h"

namespace prism::runtime {

/// @addtogroup runtime
/// @{

// ============================================================================
// C2C 正变换
// ============================================================================

void FFT::forwardImpl(std::complex<real32_t>* data, int64_t n,
                      FFTOptions opts) {
  auto& backend = backend::getFftBackend();
  if (!backend.isAvailable()) {
    throw std::runtime_error("FFT backend not available");
  }

  if (opts.async) {
    backend.executeOnBufferAsync(n, 1, -1, false);
  } else {
    backend.forwardC2c(data, n);
  }
}

void FFT::forwardImpl(std::complex<real64_t>* data, int64_t n,
                      FFTOptions /*opts*/) {
  auto& backend = backend::getFftBackend();
  if (!backend.isAvailable()) {
    throw std::runtime_error("FFT backend not available");
  }

  backend.forwardC2c(data, n);
}

// ============================================================================
// C2C 逆变换
// ============================================================================

void FFT::inverseImpl(std::complex<real32_t>* data, int64_t n,
                      FFTOptions opts) {
  auto& backend = backend::getFftBackend();
  if (!backend.isAvailable()) {
    throw std::runtime_error("FFT backend not available");
  }

  backend.inverseC2c(data, n, opts.normalize);
}

void FFT::inverseImpl(std::complex<real64_t>* data, int64_t n,
                      FFTOptions opts) {
  auto& backend = backend::getFftBackend();
  if (!backend.isAvailable()) {
    throw std::runtime_error("FFT backend not available");
  }

  backend.inverseC2c(data, n, opts.normalize);
}

// ============================================================================
// R2C 正变换
// ============================================================================

void FFT::forwardR2cImpl(const real32_t* in, std::complex<real32_t>* out,
                         int64_t n) {
  auto& backend = backend::getFftBackend();
  if (!backend.isAvailable()) {
    throw std::runtime_error("FFT backend not available");
  }

  backend.forwardR2c(in, out, n);
}

void FFT::forwardR2cImpl(const real64_t* in, std::complex<real64_t>* out,
                         int64_t n) {
  auto& backend = backend::getFftBackend();
  if (!backend.isAvailable()) {
    throw std::runtime_error("FFT backend not available");
  }

  backend.forwardR2c(in, out, n);
}

// ============================================================================
// C2R 逆变换
// ============================================================================

void FFT::inverseC2rImpl(const std::complex<real32_t>* in, real32_t* out,
                         int64_t n, FFTOptions opts) {
  auto& backend = backend::getFftBackend();
  if (!backend.isAvailable()) {
    throw std::runtime_error("FFT backend not available");
  }

  backend.inverseC2r(in, out, n, opts.normalize);
}

void FFT::inverseC2rImpl(const std::complex<real64_t>* in, real64_t* out,
                         int64_t n, FFTOptions opts) {
  auto& backend = backend::getFftBackend();
  if (!backend.isAvailable()) {
    throw std::runtime_error("FFT backend not available");
  }

  backend.inverseC2r(in, out, n, opts.normalize);
}

// ============================================================================
// 批量变换
// ============================================================================

void FFT::batchImpl(std::complex<real32_t>* data, int64_t n, int64_t batch,
                    int direction, FFTOptions opts) {
  auto& backend = backend::getFftBackend();
  if (!backend.isAvailable()) {
    throw std::runtime_error("FFT backend not available");
  }

  if (opts.async) {
    backend.executeOnBufferAsync(n, batch, direction, false);
  } else {
    backend.batchC2c(data, n, batch, direction);

    if (direction == 1 && opts.normalize) {
      real32_t const scale = 1.0F / static_cast<real32_t>(n);
      for (int64_t i = 0; i < n * batch; ++i) {
        data[i] *= scale;
      }
    }
  }
}

void FFT::batchImpl(std::complex<real64_t>* data, int64_t n, int64_t batch,
                    int direction, FFTOptions opts) {
  auto& backend = backend::getFftBackend();
  if (!backend.isAvailable()) {
    throw std::runtime_error("FFT backend not available");
  }

  backend.batchC2c(data, n, batch, direction);

  if (direction == 1 && opts.normalize) {
    real64_t const scale = 1.0 / static_cast<real64_t>(n);
    for (int64_t i = 0; i < n * batch; ++i) {
      data[i] *= scale;
    }
  }
}

// ============================================================================
// 同步
// ============================================================================

void FFT::sync() {
  auto& backend = backend::getFftBackend();
  if (backend.isAvailable()) {
    backend.sync();
  }
}

bool FFT::supports(ScalarType precision, FftTransType type) {
  auto& backend = backend::getFftBackend();
  if (!backend.isAvailable()) return false;
  return backend.supports(precision, type);
}

/// @}

}  // namespace prism::runtime
