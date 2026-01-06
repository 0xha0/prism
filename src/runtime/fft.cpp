/**
 * @file fft.cpp
 * @ingroup runtime
 * @brief FFT Anchor 实现
 *
 * 调用 fft_backend 模板接口执行 FFT
 */

#include "prism/runtime/fft.h"

#include <complex>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include "prism/backend/fft_backend.h"
#include "prism/types.h"

namespace prism::runtime {

/// @addtogroup runtime
/// @{

namespace {

void ensureOutOfPlace(const void* in, const void* out, const char* label) {
  if (in == out) {
    throw std::runtime_error(std::string(label) + " must be out-of-place (in/out cannot alias)");
  }
}

bool isPrecisionCompatible(ScalarType precision, FftTransType type) {
  if (!isFloatType(precision)) return false;
  if (type == FftTransType::C2C) return isComplexType(precision);
  return isRealType(precision);
}

void validateDirection(FftTransType type, int direction) {
  if (type == FftTransType::C2C && direction != -1 && direction != 1) {
    throw std::invalid_argument("C2C direction must be -1 or 1");
  }
  if (type == FftTransType::R2C && direction != -1) {
    throw std::invalid_argument("R2C direction must be -1");
  }
  if (type == FftTransType::C2R && direction != 1) {
    throw std::invalid_argument("C2R direction must be 1");
  }
}

}  // namespace

// ============================================================================
// C2C 正变换
// ============================================================================

void FFT::forwardImpl(std::complex<real32_t>* data, int64_t n) {
  auto& backend = backend::getFftBackend();
  if (!backend.isAvailable()) {
    throw std::runtime_error("FFT backend not available");
  }

  backend.forwardC2c(data, n);
}

void FFT::forwardImpl(std::complex<real64_t>* data, int64_t n) {
  auto& backend = backend::getFftBackend();
  if (!backend.isAvailable()) {
    throw std::runtime_error("FFT backend not available");
  }

  backend.forwardC2c(data, n);
}

// ============================================================================
// C2C 逆变换
// ============================================================================

void FFT::inverseImpl(std::complex<real32_t>* data, int64_t n, bool normalize) {
  auto& backend = backend::getFftBackend();
  if (!backend.isAvailable()) {
    throw std::runtime_error("FFT backend not available");
  }

  backend.inverseC2c(data, n, normalize);
}

void FFT::inverseImpl(std::complex<real64_t>* data, int64_t n, bool normalize) {
  auto& backend = backend::getFftBackend();
  if (!backend.isAvailable()) {
    throw std::runtime_error("FFT backend not available");
  }

  backend.inverseC2c(data, n, normalize);
}

// ============================================================================
// R2C 正变换
// ============================================================================

void FFT::forwardR2cImpl(const real32_t* in, std::complex<real32_t>* out, int64_t n) {
  auto& backend = backend::getFftBackend();
  if (!backend.isAvailable()) {
    throw std::runtime_error("FFT backend not available");
  }

  ensureOutOfPlace(in, out, "R2C");
  backend.forwardR2c(in, out, n);
}

void FFT::forwardR2cImpl(const real64_t* in, std::complex<real64_t>* out, int64_t n) {
  auto& backend = backend::getFftBackend();
  if (!backend.isAvailable()) {
    throw std::runtime_error("FFT backend not available");
  }

  ensureOutOfPlace(in, out, "R2C");
  backend.forwardR2c(in, out, n);
}

// ============================================================================
// C2R 逆变换
// ============================================================================

void FFT::inverseC2rImpl(const std::complex<real32_t>* in, real32_t* out, int64_t n,
                         bool normalize) {
  auto& backend = backend::getFftBackend();
  if (!backend.isAvailable()) {
    throw std::runtime_error("FFT backend not available");
  }

  ensureOutOfPlace(in, out, "C2R");
  backend.inverseC2r(in, out, n, normalize);
}

void FFT::inverseC2rImpl(const std::complex<real64_t>* in, real64_t* out, int64_t n,
                         bool normalize) {
  auto& backend = backend::getFftBackend();
  if (!backend.isAvailable()) {
    throw std::runtime_error("FFT backend not available");
  }

  ensureOutOfPlace(in, out, "C2R");
  backend.inverseC2r(in, out, n, normalize);
}

// ============================================================================
// 批量变换
// ============================================================================

void FFT::batchImpl(std::complex<real32_t>* data, int64_t n, int64_t batch, int direction,
                    bool normalize) {
  auto& backend = backend::getFftBackend();
  if (!backend.isAvailable()) {
    throw std::runtime_error("FFT backend not available");
  }

  backend.batchC2c(data, n, batch, direction);

  if (direction == 1 && normalize) {
    real32_t const scale = 1.0F / static_cast<real32_t>(n);
    for (int64_t i = 0; i < n * batch; ++i) {
      data[i] *= scale;
    }
  }
}

void FFT::batchImpl(std::complex<real64_t>* data, int64_t n, int64_t batch, int direction,
                    bool normalize) {
  auto& backend = backend::getFftBackend();
  if (!backend.isAvailable()) {
    throw std::runtime_error("FFT backend not available");
  }

  backend.batchC2c(data, n, batch, direction);

  if (direction == 1 && normalize) {
    real64_t const scale = 1.0 / static_cast<real64_t>(n);
    for (int64_t i = 0; i < n * batch; ++i) {
      data[i] *= scale;
    }
  }
}

// ============================================================================
// Global sync
// ============================================================================

void FFT::deviceSyncGlobal() {
  auto& backend = backend::getFftBackend();
  if (backend.isAvailable()) {
    backend.sync();
  }
}

bool FFT::supports(ScalarType precision, FftTransType type) {
  auto& backend = backend::getFftBackend();
  if (!backend.isAvailable()) return false;
  if (!isPrecisionCompatible(precision, type)) return false;
  return backend.supports(precision, type);
}

// ============================================================================
// Device buffer API
// ============================================================================

struct FFT::DeviceBuffer::Impl {
  backend::FFTBackend* backend = nullptr;
  backend::FFTBackend::DeviceBuffer buffer = {};
  bool host_dirty = false;    // NOLINT(readability-identifier-naming)
  bool device_dirty = false;  // NOLINT(readability-identifier-naming)
};

FFT::DeviceBuffer::DeviceBuffer(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

FFT::DeviceBuffer::~DeviceBuffer() {
  if (!impl_ || !impl_->backend) return;
  impl_->backend->deviceSync(impl_->buffer);
  impl_->backend->releaseDeviceBuffer(impl_->buffer);
}

bool FFT::DeviceBuffer::valid() const { return impl_ != nullptr && impl_->backend != nullptr; }

void FFT::DeviceBuffer::submit(int direction, bool wait) {
  if (!impl_ || !impl_->backend) {
    throw std::runtime_error("Invalid device buffer");
  }
  validateDirection(impl_->buffer.type, direction);
  if (host_dirty()) {
    copy_to_device();
  }
  impl_->backend->submitDeviceBuffer(impl_->buffer, direction);
  impl_->device_dirty = true;
  impl_->host_dirty = false;
  if (wait) {
    device_sync();
  }
}

void FFT::DeviceBuffer::device_sync() {
  if (!impl_ || !impl_->backend) return;
  impl_->backend->deviceSync(impl_->buffer);
}

void FFT::DeviceBuffer::copy_to_host() {
  if (!impl_ || !impl_->backend) return;
  if (!impl_->device_dirty) return;
  impl_->backend->copyToHost(impl_->buffer);
  impl_->device_dirty = false;
  impl_->host_dirty = false;
}

void FFT::DeviceBuffer::copy_to_device() {
  if (!impl_ || !impl_->backend) return;
  if (!impl_->host_dirty) return;
  impl_->backend->copyToDevice(impl_->buffer);
  impl_->host_dirty = false;
  impl_->device_dirty = false;
}

bool FFT::DeviceBuffer::host_dirty() const { return impl_ && impl_->host_dirty; }

bool FFT::DeviceBuffer::device_dirty() const { return impl_ && impl_->device_dirty; }

void FFT::DeviceBuffer::set_host_dirty(bool v) {
  if (!impl_) return;
  if (v && impl_->device_dirty) {
    throw std::runtime_error("Cannot set host dirty when device is already dirty");
  }
  impl_->host_dirty = v;
}

void FFT::DeviceBuffer::set_device_dirty(bool v) {
  if (!impl_) return;
  if (v && impl_->host_dirty) {
    throw std::runtime_error("Cannot set device dirty when host is already dirty");
  }
  impl_->device_dirty = v;
}

bool FFT::supportsDeviceBuffer(ScalarType precision, FftTransType type) {
  auto& backend = backend::getFftBackend();
  if (!backend.isAvailable()) return false;
  if (!isPrecisionCompatible(precision, type)) return false;
  return backend.supportsDeviceBuffer(precision, type);
}

FFT::DeviceBuffer FFT::acquireDeviceBuffer(ScalarType precision, FftTransType type, int64_t n,
                                           int64_t batch) {
  auto& backend = backend::getFftBackend();
  if (!backend.isAvailable()) {
    throw std::runtime_error("FFT backend not available");
  }
  if (!isPrecisionCompatible(precision, type)) {
    throw std::invalid_argument(
        "FFT precision/type mismatch (C2C requires complex, R2C/C2R require "
        "real)");
  }
  if (!backend.supportsDeviceBuffer(precision, type)) {
    throw std::runtime_error("FFT device buffer not supported for requested precision/type");
  }
  auto impl = std::make_unique<DeviceBuffer::Impl>();
  impl->backend = &backend;
  impl->buffer = backend.acquireDeviceBuffer(precision, type, n, batch);
  return DeviceBuffer(std::move(impl));
}

void FFT::releaseDeviceBuffer(DeviceBuffer& buffer) {
  if (!buffer.impl_ || !buffer.impl_->backend) return;
  buffer.impl_->backend->deviceSync(buffer.impl_->buffer);
  buffer.impl_->backend->releaseDeviceBuffer(buffer.impl_->buffer);
  buffer.impl_.reset();
}

void FFT::executeDeviceBuffer(DeviceBuffer& buffer, int direction, bool wait) {
  buffer.submit(direction, wait);
}

FFT::DeviceBuffer FFT::wrap_native_handle(void* handle, ScalarType precision, FftTransType type,
                                          int64_t n, int64_t batch) {
  auto& backend = backend::getFftBackend();
  if (!backend.isAvailable()) {
    throw std::runtime_error("FFT backend not available");
  }
  if (!isPrecisionCompatible(precision, type)) {
    throw std::invalid_argument(
        "FFT precision/type mismatch (C2C requires complex, R2C/C2R require "
        "real)");
  }
  auto impl = std::make_unique<DeviceBuffer::Impl>();
  impl->backend = &backend;
  impl->buffer = backend.wrapNativeHandle(handle, precision, type, n, batch);
  return DeviceBuffer(std::move(impl));
}

void FFT::detach_native_handle(DeviceBuffer& buffer) {
  if (!buffer.impl_ || !buffer.impl_->backend) return;
  buffer.impl_->backend->detachNativeHandle(buffer.impl_->buffer);
}

void* FFT::get_native_device_ptr(const DeviceBuffer& buffer) {
  if (!buffer.impl_ || !buffer.impl_->backend) return nullptr;
  return buffer.impl_->backend->getNativeDevicePtr(buffer.impl_->buffer);
}

std::string FFT::get_backend_device_info() {
  auto& backend = backend::getFftBackend();
  if (!backend.isAvailable()) return "FFT backend not available";
  return backend.getBackendDeviceInfo();
}

/// @}

}  // namespace prism::runtime
