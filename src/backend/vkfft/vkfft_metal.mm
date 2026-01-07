/**
 * @file vkfft_metal.mm
 * @ingroup backend
 * @brief vkFFT Metal 后端实现
 *
 * 使用 vkFFT 库通过 Metal API 执行 FFT
 * 此后端在 VKFFT_BACKEND=5 时编译
 */

#include "prism/backend/fft_backend.h"

#if defined(PRISM_HAS_VKFFT) && (VKFFT_BACKEND == 5)

#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

#include "vkFFT.h"

#include <algorithm>
#include <cstring>
#include <functional>
#include <memory>
#include <stdexcept>
#include <unordered_map>

namespace prism::backend {

/// @addtogroup backend
/// @{

/**
 * @brief vkFFT Metal FFT 后端实现
 */
class VkFFTMetalBackend : public FFTBackend {
 public:
  VkFFTMetalBackend() : device_(MTL::CreateSystemDefaultDevice()) {
    if (!device_) {
      available_ = false;
      return;
    }

    queue_ = device_->newCommandQueue();
    if (!queue_) {
      device_->release();
      device_ = nullptr;
      available_ = false;
      return;
    }

    available_ = true;
  }

  VkFFTMetalBackend(const VkFFTMetalBackend &) = delete;
  VkFFTMetalBackend &operator=(const VkFFTMetalBackend &) = delete;
  VkFFTMetalBackend(VkFFTMetalBackend &&) = delete;
  VkFFTMetalBackend &operator=(VkFFTMetalBackend &&) = delete;

  // 析构函数在缓存定义后实现

  [[nodiscard]] bool isAvailable() const override { return available_; }

  [[nodiscard]] const char *name() const override { return "vkFFT (Metal)"; }

  [[nodiscard]] bool supports(ScalarType precision, FftTransType /*type*/) const override {
    // Metal on Mac typically supports fp32. double (fp64) is often not
    // supported in standard Metal standard lib or hardware (e.g. M1/M2 double
    // performance or shader support issues). vkFFT Metal backend in this
    // project currently only successfully tests for fp32. We assume C2C/R2C/C2R
    // are all supported if precision is supported.
    if (!isFloatType(precision)) return false;
    return getComponentSize(precision) == 4;
  }

  void forwardC2cImpl(complex32_t *data, int64_t n) override { runC2c<real32_t>(data, n, 1, -1); }

  void forwardC2cImpl(complex64_t *data, int64_t n) override { runC2c<real64_t>(data, n, 1, -1); }

  void inverseC2cImpl(complex32_t *data, int64_t n, bool normalize) override {
    runC2c<real32_t>(data, n, 1, 1, normalize);
  }

  void inverseC2cImpl(complex64_t *data, int64_t n, bool normalize) override {
    runC2c<real64_t>(data, n, 1, 1, normalize);
  }

  void forwardR2cImpl(const real32_t *in, complex32_t *out, int64_t n) override {
    runR2c<real32_t>(in, out, n);
  }

  void forwardR2cImpl(const real64_t *in, complex64_t *out, int64_t n) override {
    runR2c<real64_t>(in, out, n);
  }

  void inverseC2rImpl(const complex32_t *in, real32_t *out, int64_t n, bool normalize) override {
    runC2r<real32_t>(in, out, n, normalize);
  }

  void inverseC2rImpl(const complex64_t *in, real64_t *out, int64_t n, bool normalize) override {
    runC2r<real64_t>(in, out, n, normalize);
  }

 private:
  MTL::Device *device_ = nullptr;
  MTL::CommandQueue *queue_ = nullptr;
  bool available_ = false;
  struct CachedPlan;
  struct PendingBuffer {
    MTL::CommandBuffer *cmd = nullptr;
    DeviceBuffer *buffer = nullptr;
  };
  std::vector<PendingBuffer> pendingBuffers_;
  enum class NormalizeKind : std::uint8_t {
    None,
    Complex32,
    Complex64,
    Real32,
    Real64
    };
  struct PendingHostOp {
    MTL::CommandBuffer *cmd = nullptr;
    CachedPlan *plan = nullptr;
    void *dst = nullptr;
    const void *src = nullptr;
    size_t bytes = 0;
    NormalizeKind normalize = NormalizeKind::None;
    int64_t normalizeCount = 0;
    int64_t normalizeN = 0;
  };
  std::vector<PendingHostOp> pendingHostOps_;

  enum class PlanKind : std::uint8_t { C2C, R2C };

  struct PlanKey {
    // NOLINTBEGIN
    FftTransType type = FftTransType::C2C;
    ScalarType precision = ScalarType::F32;
    int64_t n = 0;
    int64_t batch = 0;
    // NOLINTEND
    bool operator==(const PlanKey &other) const {
      return type == other.type && precision == other.precision && n == other.n &&
             batch == other.batch;
    }
  };

  struct PlanKeyHash {
    std::size_t operator()(const PlanKey &key) const {
      auto hashCombine = [](std::size_t seed, std::size_t value) {
        return seed ^ (value + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2));
      };
      std::size_t h = 0;
      h = hashCombine(h, static_cast<std::size_t>(key.type));
      h = hashCombine(h, static_cast<std::size_t>(key.precision));
      h = hashCombine(h, std::hash<int64_t>{}(key.n));
      h = hashCombine(h, std::hash<int64_t>{}(key.batch));
      return h;
    }
  };

  // ========== vkFFT App 缓存 ==========
  struct CachedPlan {
    VkFFTApplication app = {};
    MTL::Buffer *buffer = nullptr;
    MTL::Buffer *inputBuffer = nullptr;
    uint64_t bufferSize = 0;
    uint64_t inputBufferSize = 0;
    int64_t fftSize = 0;
    int64_t batch = 0;
    PlanKind kind = PlanKind::C2C;
    ScalarType precision = ScalarType::F32;
    bool ownsBuffers = true;
    bool valid = false;
    void *hostComplex = nullptr;
    void *hostReal = nullptr;
  };

  std::unordered_map<PlanKey, std::unique_ptr<CachedPlan>, PlanKeyHash> cache_;

  static PlanKind planKindFor(FftTransType type) {
    return type == FftTransType::C2C ? PlanKind::C2C : PlanKind::R2C;
  }

  template <typename T>
  static ScalarType precisionFor(PlanKind kind) {
    if (kind == PlanKind::C2C) {
      return std::is_same_v<T, real64_t> ? ScalarType::C64 : ScalarType::C32;
    }
    return std::is_same_v<T, real64_t> ? ScalarType::F64 : ScalarType::F32;
  }

  void releasePlan(CachedPlan &plan) {
    if (plan.valid) {
      deleteVkFFT(&plan.app);
    }
    if (plan.ownsBuffers) {
      if (plan.buffer) {
        plan.buffer->release();
        plan.buffer = nullptr;
      }
      if (plan.inputBuffer) {
        plan.inputBuffer->release();
        plan.inputBuffer = nullptr;
      }
    }
    plan.buffer = nullptr;
    plan.inputBuffer = nullptr;
    plan.hostComplex = nullptr;
    plan.hostReal = nullptr;
    plan.valid = false;
  }

  template <typename T>
  CachedPlan &getOrCreatePlan(FftTransType type, int64_t n, int64_t batch) {
    PlanKind const kind = planKindFor(type);
    PlanKey const key{type, precisionFor<T>(kind), n, batch};
    auto it = cache_.find(key);
    if (it != cache_.end() && it->second && it->second->valid) {
      return *it->second;
    }

    auto plan = std::make_unique<CachedPlan>();
    plan->fftSize = n;
    plan->batch = batch;
    plan->kind = kind;
    plan->precision = precisionFor<T>(kind);
    plan->valid = false;

    if (kind == PlanKind::C2C) {
      plan->bufferSize = static_cast<uint64_t>(n * batch * sizeof(std::complex<T>));
      plan->buffer =
          device_->newBuffer(static_cast<size_t>(plan->bufferSize), MTL::ResourceStorageModeShared);
      if (!plan->buffer) {
        throw std::runtime_error("Failed to create Metal buffer");
      }
      plan->hostComplex = plan->buffer->contents();
    } else {
      plan->inputBufferSize = static_cast<uint64_t>(n * sizeof(T) * batch);
      plan->bufferSize = static_cast<uint64_t>((n / 2 + 1) * sizeof(std::complex<T>) * batch);
      plan->inputBuffer = device_->newBuffer(static_cast<size_t>(plan->inputBufferSize),
                                             MTL::ResourceStorageModeShared);
      plan->buffer =
          device_->newBuffer(static_cast<size_t>(plan->bufferSize), MTL::ResourceStorageModeShared);
      if (!plan->inputBuffer || !plan->buffer) {
        if (plan->inputBuffer) plan->inputBuffer->release();
        if (plan->buffer) plan->buffer->release();
        throw std::runtime_error("Failed to create Metal buffers");
      }
      plan->hostReal = plan->inputBuffer->contents();
      plan->hostComplex = plan->buffer->contents();
    }

    VkFFTConfiguration config = {};
    config.FFTdim = 1;
    config.size[0] = static_cast<uint64_t>(n);
    config.numberBatches = static_cast<uint64_t>(batch);
    if constexpr (std::is_same_v<T, real64_t>) config.doublePrecision = 1;
    config.device = device_;
    config.queue = queue_;

    if (kind == PlanKind::C2C) {
      config.buffer = &plan->buffer;
      config.bufferSize = &plan->bufferSize;
    } else {
      config.performR2C = 1;
      config.inverseReturnToInputBuffer = 1;
      config.isInputFormatted = 1;
      config.inputBufferNum = 1;
      config.bufferNum = 1;
      config.inputBufferStride[0] = static_cast<uint64_t>(n);
      config.bufferStride[0] = static_cast<uint64_t>(n / 2 + 1);
      config.inputBuffer = &plan->inputBuffer;
      config.inputBufferSize = &plan->inputBufferSize;
      config.buffer = &plan->buffer;
      config.bufferSize = &plan->bufferSize;
    }

    VkFFTResult const result = initializeVkFFT(&plan->app, config);
    if (result != VKFFT_SUCCESS) {
      releasePlan(*plan);
      throw std::runtime_error("Failed to initialize vkFFT");
    }

    plan->valid = true;
    auto &stored = cache_[key];
    stored = std::move(plan);
    return *stored;
  }

 public:
  void clearCache() {
    flushPendingHostOps(false);
    for (auto &pending : pendingBuffers_) {
      if (pending.cmd) {
        pending.cmd->waitUntilCompleted();
        pending.cmd->release();
      }
      if (pending.buffer) {
        pending.buffer->asyncHandle = nullptr;
      }
    }
    pendingBuffers_.clear();
    for (auto &[key, plan] : cache_) {
      if (plan) {
        releasePlan(*plan);
      }
    }
    cache_.clear();
  }

  ~VkFFTMetalBackend() override {
    flushPendingHostOps(false);
    for (auto &pending : pendingBuffers_) {
      if (pending.cmd) {
        pending.cmd->waitUntilCompleted();
        pending.cmd->release();
      }
      if (pending.buffer) {
        pending.buffer->asyncHandle = nullptr;
      }
    }
    pendingBuffers_.clear();
    clearCache();
    if (queue_) queue_->release();
    if (device_) device_->release();
  }

 private:
  MTL::CommandBuffer *enqueuePlan(CachedPlan &plan, int direction) {
    if (!available_) {
      throw std::runtime_error("vkFFT Metal backend not available");
    }

    MTL::CommandBuffer *cmdBuffer = queue_->commandBuffer();
    if (!cmdBuffer) {
      throw std::runtime_error("Failed to create Metal command buffer");
    }
    MTL::ComputeCommandEncoder *cmdEncoder = cmdBuffer->computeCommandEncoder();
    if (!cmdEncoder) {
      cmdBuffer->release();
      throw std::runtime_error("Failed to create Metal command encoder");
    }

    VkFFTLaunchParams params = {};
    params.buffer = &plan.buffer;
    if (plan.kind == PlanKind::R2C) {
      params.inputBuffer = &plan.inputBuffer;
    }
    params.commandBuffer = cmdBuffer;
    params.commandEncoder = cmdEncoder;

    VkFFTResult const result = VkFFTAppend(&plan.app, direction, &params);
    if (result != VKFFT_SUCCESS) {
      cmdEncoder->endEncoding();
      cmdEncoder->release();
      cmdBuffer->release();
      throw std::runtime_error("Failed to execute vkFFT");
    }

    cmdEncoder->endEncoding();
    cmdEncoder->release();
    cmdBuffer->commit();
    return cmdBuffer;
  }

  void executePlanSync(CachedPlan &plan, int direction) {
    MTL::CommandBuffer *cmdBuffer = enqueuePlan(plan, direction);
    cmdBuffer->waitUntilCompleted();
    cmdBuffer->release();
  }

  static void applyNormalization(const PendingHostOp &op) {
    if (op.normalize == NormalizeKind::None || op.normalizeCount <= 0 || op.normalizeN <= 0 ||
        op.dst == nullptr) {
      return;
    }
    switch (op.normalize) {
      case NormalizeKind::Complex32: {
        auto *out = static_cast<complex32_t *>(op.dst);
        real32_t const scale = static_cast<real32_t>(1.0F / static_cast<real32_t>(op.normalizeN));
        for (int64_t i = 0; i < op.normalizeCount; ++i) {
          out[i] *= scale;
        }
        break;
      }
      case NormalizeKind::Complex64: {
        auto *out = static_cast<complex64_t *>(op.dst);
        real64_t const scale = static_cast<real64_t>(1.0 / static_cast<real64_t>(op.normalizeN));
        for (int64_t i = 0; i < op.normalizeCount; ++i) {
          out[i] *= scale;
        }
        break;
      }
      case NormalizeKind::Real32: {
        auto *out = static_cast<real32_t *>(op.dst);
        real32_t const scale = static_cast<real32_t>(1.0F / static_cast<real32_t>(op.normalizeN));
        for (int64_t i = 0; i < op.normalizeCount; ++i) {
          out[i] *= scale;
        }
        break;
      }
      case NormalizeKind::Real64: {
        auto *out = static_cast<real64_t *>(op.dst);
        real64_t const scale = static_cast<real64_t>(1.0 / static_cast<real64_t>(op.normalizeN));
        for (int64_t i = 0; i < op.normalizeCount; ++i) {
          out[i] *= scale;
        }
        break;
      }
      case NormalizeKind::None:
        break;
    }
  }

  void drainPendingHostOpsForPlan(CachedPlan &plan, bool copyOutput) {
    auto it = pendingHostOps_.begin();
    while (it != pendingHostOps_.end()) {
      if (it->plan != &plan) {
        ++it;
        continue;
      }
      if (it->cmd) {
        it->cmd->waitUntilCompleted();
      }
      if (copyOutput && it->dst != nullptr && it->src != nullptr && it->bytes > 0 &&
          it->dst != it->src) {
        memcpy(it->dst, it->src, it->bytes);
      }
      if (copyOutput) {
        applyNormalization(*it);
      }
      if (it->cmd) {
        it->cmd->release();
      }
      it = pendingHostOps_.erase(it);
    }
  }

  void dropPendingHostOpsForPlan(CachedPlan &plan) {
    auto it = std::remove_if(pendingHostOps_.begin(), pendingHostOps_.end(),
                             [&plan](const PendingHostOp &pending) {
                               if (pending.plan != &plan) return false;
                               if (pending.cmd) {
                                 pending.cmd->release();
                               }
                               return true;
                             });
    pendingHostOps_.erase(it, pendingHostOps_.end());
  }

  void flushPendingHostOps(bool copyOutput) {
    for (auto &pending : pendingHostOps_) {
      if (pending.cmd) {
        pending.cmd->waitUntilCompleted();
      }
      if (copyOutput && pending.dst != nullptr && pending.src != nullptr && pending.bytes > 0 &&
          pending.dst != pending.src) {
        memcpy(pending.dst, pending.src, pending.bytes);
      }
      if (copyOutput) {
        applyNormalization(pending);
      }
      if (pending.cmd) {
        pending.cmd->release();
      }
    }
    pendingHostOps_.clear();
  }

  template <typename T>
  void runC2c(std::complex<T> *data, int64_t n, int64_t batch, int direction,
              bool normalize = false) {
    if (n <= 0 || (n & (n - 1)) != 0) {
      throw std::invalid_argument("FFT length must be power of 2");
    }
    CachedPlan &plan = getOrCreatePlan<T>(FftTransType::C2C, n, batch);
    auto *host = static_cast<std::complex<T> *>(plan.hostComplex);
    auto const dataSize = static_cast<size_t>(n * batch * sizeof(std::complex<T>));
    if (data != host) {
      drainPendingHostOpsForPlan(plan, false);
      memcpy(host, data, dataSize);
    } else {
      dropPendingHostOpsForPlan(plan);
    }
    MTL::CommandBuffer *cmdBuffer = enqueuePlan(plan, direction);
    PendingHostOp pending;
    pending.cmd = cmdBuffer;
    pending.plan = &plan;
    pending.dst = data;
    pending.src = host;
    pending.bytes = dataSize;
    if (normalize) {
      pending.normalize = std::is_same_v<T, real64_t> ? NormalizeKind::Complex64
                                                      : NormalizeKind::Complex32;
      pending.normalizeCount = n * batch;
      pending.normalizeN = n;
    }
    pendingHostOps_.push_back(pending);
  }

  template <typename T>
  void runC2cSync(std::complex<T> *data, int64_t n, int64_t batch, int direction) {
    if (n <= 0 || (n & (n - 1)) != 0) {
      throw std::invalid_argument("FFT length must be power of 2");
    }
    CachedPlan &plan = getOrCreatePlan<T>(FftTransType::C2C, n, batch);
    auto *host = static_cast<std::complex<T> *>(plan.hostComplex);
    auto const dataSize = static_cast<size_t>(n * batch * sizeof(std::complex<T>));
    drainPendingHostOpsForPlan(plan, false);
    if (data != host) {
      memcpy(host, data, dataSize);
    }
    executePlanSync(plan, direction);
    if (data != host) {
      memcpy(data, host, dataSize);
    }
  }

  template <typename T>
  void runR2c(const T *in, std::complex<T> *out, int64_t n) {
    if (n <= 0 || (n & (n - 1)) != 0) {
      throw std::invalid_argument("FFT length must be power of 2");
    }
    CachedPlan &plan = getOrCreatePlan<T>(FftTransType::R2C, n, 1);
    auto *hostReal = static_cast<T *>(plan.hostReal);
    auto *hostComplex = static_cast<std::complex<T> *>(plan.hostComplex);
    if (in != hostReal) {
      drainPendingHostOpsForPlan(plan, false);
      memcpy(hostReal, in, static_cast<size_t>(n * sizeof(T)));
    } else {
      dropPendingHostOpsForPlan(plan);
    }
    MTL::CommandBuffer *cmdBuffer = enqueuePlan(plan, -1);
    PendingHostOp pending;
    pending.cmd = cmdBuffer;
    pending.plan = &plan;
    pending.dst = out;
    pending.src = hostComplex;
    pending.bytes = static_cast<size_t>((n / 2 + 1) * sizeof(std::complex<T>));
    pendingHostOps_.push_back(pending);
  }

  template <typename T>
  void runC2r(const std::complex<T> *in, T *out, int64_t n, bool normalize) {
    if (n <= 0 || (n & (n - 1)) != 0) {
      throw std::invalid_argument("FFT length must be power of 2");
    }
    CachedPlan &plan = getOrCreatePlan<T>(FftTransType::C2R, n, 1);
    auto *hostReal = static_cast<T *>(plan.hostReal);
    auto *hostComplex = static_cast<std::complex<T> *>(plan.hostComplex);
    if (in != hostComplex) {
      drainPendingHostOpsForPlan(plan, false);
      memcpy(hostComplex, in, static_cast<size_t>((n / 2 + 1) * sizeof(std::complex<T>)));
    } else {
      dropPendingHostOpsForPlan(plan);
    }
    MTL::CommandBuffer *cmdBuffer = enqueuePlan(plan, 1);
    PendingHostOp pending;
    pending.cmd = cmdBuffer;
    pending.plan = &plan;
    pending.dst = out;
    pending.src = hostReal;
    pending.bytes = static_cast<size_t>(n * sizeof(T));
    if (normalize) {
      pending.normalize =
          std::is_same_v<T, real64_t> ? NormalizeKind::Real64 : NormalizeKind::Real32;
      pending.normalizeCount = n;
      pending.normalizeN = n;
    }
    pendingHostOps_.push_back(pending);
  }

  // 高性能批量 FFT（使用缓存）
  void batchC2cImpl(complex32_t *data, int64_t n, int64_t batch, int direction) override {
    runC2cSync<real32_t>(data, n, batch, direction);
  }

  void batchC2cImpl(complex64_t *data, int64_t n, int64_t batch, int direction) override {
    runC2cSync<real64_t>(data, n, batch, direction);
  }

  // ========== 零拷贝 GPU 接口 ==========

  // ========== 设备缓冲区接口 ==========

  [[nodiscard]] bool supportsDeviceBuffer(ScalarType precision, FftTransType type) const override {
    if (!available_) return false;
    return supports(precision, type);
  }

  DeviceBuffer acquireDeviceBuffer(ScalarType precision, FftTransType type, int64_t n,
                                   int64_t batch) override {
    if (!available_) {
      throw std::runtime_error("vkFFT Metal backend not available");
    }
    PlanKind const kind = planKindFor(type);
    bool const complex = isComplexType(precision);
    bool const doublePrecision = getComponentSize(precision) == 8;

    if (kind == PlanKind::C2C && !complex) {
      throw std::invalid_argument("C2C requires complex precision");
    }
    if (kind == PlanKind::R2C && complex) {
      throw std::invalid_argument("R2C/C2R requires real precision");
    }

    DeviceBuffer buffer;
    buffer.n = n;
    buffer.batch = batch;
    buffer.precision = precision;
    buffer.type = type;
    buffer.hostVisible = true;

    if (doublePrecision) {
      auto &plan = getOrCreatePlan<real64_t>(type, n, batch);
      buffer.backendHandle = &plan;
      if (kind == PlanKind::C2C) {
        buffer.input = plan.hostComplex;
        buffer.output = plan.hostComplex;
      } else if (type == FftTransType::R2C) {
        buffer.input = plan.hostReal;
        buffer.output = plan.hostComplex;
      } else {
        buffer.input = plan.hostComplex;
        buffer.output = plan.hostReal;
      }
      buffer.nativeHandle = (type == FftTransType::C2R) ? plan.inputBuffer : plan.buffer;
    } else {
      auto &plan = getOrCreatePlan<real32_t>(type, n, batch);
      buffer.backendHandle = &plan;
      if (kind == PlanKind::C2C) {
        buffer.input = plan.hostComplex;
        buffer.output = plan.hostComplex;
      } else if (type == FftTransType::R2C) {
        buffer.input = plan.hostReal;
        buffer.output = plan.hostComplex;
      } else {
        buffer.input = plan.hostComplex;
        buffer.output = plan.hostReal;
      }
      buffer.nativeHandle = (type == FftTransType::C2R) ? plan.inputBuffer : plan.buffer;
    }
    return buffer;
  }

  void releaseDeviceBuffer(DeviceBuffer &buffer) override {
    if (buffer.asyncHandle != nullptr) {
      deviceSync(buffer);
    }
    if (buffer.wrapped && buffer.backendHandle != nullptr) {
      auto *plan = static_cast<CachedPlan *>(buffer.backendHandle);
      releasePlan(*plan);
      delete plan;
    }
    buffer.input = nullptr;
    buffer.output = nullptr;
    buffer.backendHandle = nullptr;
    buffer.asyncHandle = nullptr;
    buffer.nativeHandle = nullptr;
    buffer.wrapped = false;
  }

  void submitDeviceBuffer(DeviceBuffer &buffer, int direction) override {
    if (!available_) {
      throw std::runtime_error("vkFFT Metal backend not available");
    }
    if (buffer.backendHandle == nullptr) {
      throw std::runtime_error("Invalid device buffer handle");
    }
    auto *plan = static_cast<CachedPlan *>(buffer.backendHandle);
    if (!plan->valid) {
      throw std::runtime_error("Device buffer plan is not initialized");
    }
    if (buffer.type == FftTransType::R2C && direction != -1) {
      throw std::invalid_argument("R2C direction must be -1");
    }
    if (buffer.type == FftTransType::C2R && direction != 1) {
      throw std::invalid_argument("C2R direction must be 1");
    }
    if (buffer.asyncHandle != nullptr) {
      auto *prev = static_cast<MTL::CommandBuffer *>(buffer.asyncHandle);
      removePendingBuffer(prev);
      prev->release();
      buffer.asyncHandle = nullptr;
    }
    MTL::CommandBuffer *cmdBuffer = enqueuePlan(*plan, direction);
    pendingBuffers_.push_back({cmdBuffer, &buffer});
    buffer.asyncHandle = cmdBuffer;
  }

  void deviceSync(DeviceBuffer &buffer) override {
    if (buffer.asyncHandle == nullptr) return;
    auto *cmdBuffer = static_cast<MTL::CommandBuffer *>(buffer.asyncHandle);
    cmdBuffer->waitUntilCompleted();
    removePendingBuffer(cmdBuffer);
    cmdBuffer->release();
    buffer.asyncHandle = nullptr;
  }

  void copyToHost(DeviceBuffer &buffer) override {
    if (!buffer.hostVisible) {
      throw std::runtime_error("Device buffer is not host visible");
    }
    deviceSync(buffer);
  }

  void copyToDevice(DeviceBuffer &buffer) override {
    if (!buffer.hostVisible) {
      throw std::runtime_error("Device buffer is not host visible");
    }
    // Shared storage, no explicit copy needed.
  }

  DeviceBuffer wrapNativeHandle(void *handle, ScalarType precision, FftTransType type, int64_t n,
                                int64_t batch) override {
    if (!available_) {
      throw std::runtime_error("vkFFT Metal backend not available");
    }
    PlanKind const kind = planKindFor(type);
    bool const complex = isComplexType(precision);
    if (kind == PlanKind::C2C && !complex) {
      throw std::invalid_argument("C2C requires complex precision");
    }
    if (kind == PlanKind::R2C && complex) {
      throw std::invalid_argument("R2C/C2R requires real precision");
    }

    MTL::Buffer *inputHandle = nullptr;
    MTL::Buffer *outputHandle = nullptr;
    if (kind == PlanKind::C2C) {
      outputHandle = static_cast<MTL::Buffer *>(handle);
      inputHandle = outputHandle;
    } else {
      struct HandlePair {
        void *input;
        void *output;
      };
      auto *pair = static_cast<HandlePair *>(handle);
      if (!pair || !pair->input || !pair->output) {
        throw std::invalid_argument("R2C/C2R requires handle pair {input, output}");
      }
      inputHandle = static_cast<MTL::Buffer *>(pair->input);
      outputHandle = static_cast<MTL::Buffer *>(pair->output);
    }

    auto plan = std::make_unique<CachedPlan>();
    plan->fftSize = n;
    plan->batch = batch;
    plan->kind = kind;
    plan->precision = precision;
    plan->ownsBuffers = false;
    plan->valid = false;

    size_t const realBytes = static_cast<size_t>(getComponentSize(precision));
    size_t const complexBytes = realBytes * 2;
    if (kind == PlanKind::C2C) {
      plan->buffer = outputHandle;
      plan->bufferSize = static_cast<uint64_t>(n * batch * complexBytes);
      plan->hostComplex = plan->buffer->contents();
    } else {
      plan->inputBuffer = inputHandle;
      plan->buffer = outputHandle;
      plan->inputBufferSize = static_cast<uint64_t>(n * batch * realBytes);
      plan->bufferSize = static_cast<uint64_t>((n / 2 + 1) * batch * complexBytes);
      plan->hostReal = plan->inputBuffer->contents();
      plan->hostComplex = plan->buffer->contents();
    }

    VkFFTConfiguration config = {};
    config.FFTdim = 1;
    config.size[0] = static_cast<uint64_t>(n);
    config.numberBatches = static_cast<uint64_t>(batch);
    if (getComponentSize(precision) == 8) config.doublePrecision = 1;
    config.device = device_;
    config.queue = queue_;

    if (kind == PlanKind::C2C) {
      config.buffer = &plan->buffer;
      config.bufferSize = &plan->bufferSize;
    } else {
      config.performR2C = 1;
      config.inverseReturnToInputBuffer = 1;
      config.isInputFormatted = 1;
      config.inputBufferNum = 1;
      config.bufferNum = 1;
      config.inputBufferStride[0] = static_cast<uint64_t>(n);
      config.bufferStride[0] = static_cast<uint64_t>(n / 2 + 1);
      config.inputBuffer = &plan->inputBuffer;
      config.inputBufferSize = &plan->inputBufferSize;
      config.buffer = &plan->buffer;
      config.bufferSize = &plan->bufferSize;
    }

    VkFFTResult const result = initializeVkFFT(&plan->app, config);
    if (result != VKFFT_SUCCESS) {
      releasePlan(*plan);
      throw std::runtime_error("Failed to initialize vkFFT");
    }
    plan->valid = true;

    DeviceBuffer buffer;
    buffer.n = n;
    buffer.batch = batch;
    buffer.precision = precision;
    buffer.type = type;
    buffer.hostVisible =
        (plan->hostComplex != nullptr) && (kind == PlanKind::C2C || plan->hostReal != nullptr);
    if (buffer.hostVisible) {
      buffer.input = (kind == PlanKind::C2C)
                         ? plan->hostComplex
                         : (type == FftTransType::R2C ? plan->hostReal : plan->hostComplex);
      buffer.output = (kind == PlanKind::C2C)
                          ? plan->hostComplex
                          : (type == FftTransType::R2C ? plan->hostComplex : plan->hostReal);
    }
    buffer.backendHandle = plan.release();
    buffer.nativeHandle = outputHandle;
    buffer.wrapped = true;
    return buffer;
  }

  void detachNativeHandle(DeviceBuffer &buffer) override {
    if (!buffer.wrapped || buffer.backendHandle == nullptr) return;
    auto *plan = static_cast<CachedPlan *>(buffer.backendHandle);
    releasePlan(*plan);
    delete plan;
    buffer.backendHandle = nullptr;
    buffer.input = nullptr;
    buffer.output = nullptr;
    buffer.asyncHandle = nullptr;
    buffer.nativeHandle = nullptr;
    buffer.wrapped = false;
  }

  void *getNativeDevicePtr(const DeviceBuffer &buffer) override { return buffer.nativeHandle; }

  std::string getBackendDeviceInfo() const override {
    if (!device_) return name();
    auto nsName = device_->name();
    std::string devName;
    if (nsName) {
      devName = nsName->utf8String();
    }
    if (devName.empty()) return name();
    return std::string("vkFFT (Metal): ") + devName;
  }

  /**
   * @brief 同步等待所有异步操作完成
   */
  void sync() override {
    flushPendingHostOps(true);
    for (auto &pending : pendingBuffers_) {
      if (pending.cmd) {
        pending.cmd->waitUntilCompleted();
        pending.cmd->release();
      }
      if (pending.buffer) {
        pending.buffer->asyncHandle = nullptr;
      }
    }
    pendingBuffers_.clear();
  }

 private:
  void removePendingBuffer(MTL::CommandBuffer *cmdBuffer) {
    auto it = std::remove_if(
        pendingBuffers_.begin(), pendingBuffers_.end(),
        [cmdBuffer](const PendingBuffer &pending) { return pending.cmd == cmdBuffer; });
    pendingBuffers_.erase(it, pendingBuffers_.end());
  }
};

namespace {
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
VkFFTMetalBackend gVkfftMetalBackend;
}  // namespace

// 导出给 fft_backend.cpp 使用
FFTBackend &getVkfftBackend() { return gVkfftMetalBackend; }

/// @}

}  // namespace prism::backend

#endif  // PRISM_HAS_VKFFT && VKFFT_BACKEND == 5
