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

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <unordered_map>
#include <vector>

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

  [[nodiscard]] bool supports(ScalarType precision,
                              FftTransType /*type*/) const override {
    // Metal on Mac typically supports fp32. double (fp64) is often not
    // supported in standard Metal standard lib or hardware (e.g. M1/M2 double
    // performance or shader support issues). vkFFT Metal backend in this
    // project currently only successfully tests for fp32. We assume C2C/R2C/C2R
    // are all supported if precision is supported.
    return precision == ScalarType::F32;
  }

  void forwardC2cImpl(complex32_t *data, int64_t n) override {
    executeC2c<real32_t>(data, n, -1);
  }

  void forwardC2cImpl(complex64_t *data, int64_t n) override {
    executeC2c<real64_t>(data, n, -1);
  }

  void inverseC2cImpl(complex32_t *data, int64_t n, bool normalize) override {
    executeC2c<real32_t>(data, n, 1);
    if (normalize) {
      real32_t const scale = 1.0F / static_cast<real32_t>(n);
      for (int64_t i = 0; i < n; ++i) data[i] *= scale;
    }
  }

  void inverseC2cImpl(complex64_t *data, int64_t n, bool normalize) override {
    executeC2c<real64_t>(data, n, 1);
    if (normalize) {
      real64_t const scale = 1.0 / static_cast<real64_t>(n);
      for (int64_t i = 0; i < n; ++i) data[i] *= scale;
    }
  }

  void forwardR2cImpl(const real32_t *in, complex32_t *out,
                      int64_t n) override {
    std::vector<complex32_t> temp(n);
    for (int64_t i = 0; i < n; ++i) temp[i] = complex32_t(in[i], 0.0F);
    executeC2c<real32_t>(temp.data(), n, -1);
    for (int64_t i = 0; i <= n / 2; ++i) out[i] = temp[i];
  }

  void forwardR2cImpl(const real64_t *in, complex64_t *out,
                      int64_t n) override {
    std::vector<complex64_t> temp(n);
    for (int64_t i = 0; i < n; ++i) temp[i] = complex64_t(in[i], 0.0);
    executeC2c<real64_t>(temp.data(), n, -1);
    for (int64_t i = 0; i <= n / 2; ++i) out[i] = temp[i];
  }

  void inverseC2rImpl(const complex32_t *in, real32_t *out, int64_t n,
                      bool normalize) override {
    std::vector<complex32_t> temp(n);
    temp[0] = in[0];
    for (int64_t i = 1; i < n / 2; ++i) {
      temp[i] = in[i];
      temp[n - i] = std::conj(in[i]);
    }
    if (n / 2 < n) temp[n / 2] = in[n / 2];
    executeC2c<real32_t>(temp.data(), n, 1);
    real32_t const scale = normalize ? 1.0F / static_cast<real32_t>(n) : 1.0F;
    for (int64_t i = 0; i < n; ++i) out[i] = temp[i].real() * scale;
  }

  void inverseC2rImpl(const complex64_t *in, real64_t *out, int64_t n,
                      bool normalize) override {
    std::vector<complex64_t> temp(n);
    temp[0] = in[0];
    for (int64_t i = 1; i < n / 2; ++i) {
      temp[i] = in[i];
      temp[n - i] = std::conj(in[i]);
    }
    if (n / 2 < n) temp[n / 2] = in[n / 2];
    executeC2c<real64_t>(temp.data(), n, 1);
    real64_t const scale = normalize ? 1.0 / static_cast<real64_t>(n) : 1.0;
    for (int64_t i = 0; i < n; ++i) out[i] = temp[i].real() * scale;
  }

 private:
  MTL::Device *device_ = nullptr;
  MTL::CommandQueue *queue_ = nullptr;
  bool available_ = false;

  // ========== vkFFT App 缓存 ==========
  struct CachedPlan {
    VkFFTApplication app = {};
    MTL::Buffer *buffer = nullptr;
    uint64_t bufferSize = 0;
    int64_t fftSize = 0;
    int64_t batch = 0;
    bool valid = false;
  };

  // 缓存：<fft_size, batch> -> CachedPlan
  std::unordered_map<uint64_t, CachedPlan> cache_;

  static uint64_t cacheKey(int64_t n, int64_t batch) {
    constexpr int kBatchShift = 32;
    return (static_cast<uint64_t>(n) << kBatchShift) |
           static_cast<uint64_t>(batch);
  }

  CachedPlan &getOrCreatePlan(int64_t n, int64_t batch) {
    uint64_t const key = cacheKey(n, batch);
    auto it = cache_.find(key);
    if (it != cache_.end() && it->second.valid) {
      return it->second;
    }

    // 创建新 plan - 先存储到 cache_ 再初始化
    CachedPlan &plan = cache_[key];
    plan.fftSize = n;
    plan.batch = batch;
    plan.bufferSize = static_cast<uint64_t>(n * batch * sizeof(complex32_t));
    plan.valid = false;

    plan.buffer = device_->newBuffer(static_cast<size_t>(plan.bufferSize),
                                     MTL::ResourceStorageModeShared);
    if (!plan.buffer) {
      cache_.erase(key);
      throw std::runtime_error("Failed to create Metal buffer");
    }

    VkFFTConfiguration config = {};
    config.FFTdim = 1;
    config.size[0] = static_cast<uint64_t>(n);
    config.numberBatches = static_cast<uint64_t>(batch);
    config.device = device_;
    config.queue = queue_;
    config.buffer = &plan.buffer;
    config.bufferSize = &plan.bufferSize;

    VkFFTResult const result = initializeVkFFT(&plan.app, config);
    if (result != VKFFT_SUCCESS) {
      plan.buffer->release();
      cache_.erase(key);
      throw std::runtime_error("Failed to initialize vkFFT");
    }

    plan.valid = true;
    return plan;
  }

 public:
  void clearCache() {
    for (auto &[key, plan] : cache_) {
      if (plan.valid) {
        deleteVkFFT(&plan.app);
        if (plan.buffer) plan.buffer->release();
      }
    }
    cache_.clear();
  }

  ~VkFFTMetalBackend() override {
    clearCache();
    if (queue_) queue_->release();
    if (device_) device_->release();
  }

 private:
  template <typename T>
  void executeC2c(std::complex<T> *data, int64_t n, int direction) {
    if (!available_) {
      throw std::runtime_error("vkFFT Metal backend not available");
    }
    if (n <= 0 || (n & (n - 1)) != 0) {
      throw std::invalid_argument("FFT length must be power of 2");
    }

    VkFFTConfiguration config = {};
    VkFFTApplication app = {};
    config.FFTdim = 1;
    config.size[0] = static_cast<uint64_t>(n);
    config.numberBatches = 1;
    if constexpr (std::is_same_v<T, real64_t>) config.doublePrecision = 1;

    config.device = device_;
    config.queue = queue_;

    auto bufferSize = static_cast<uint64_t>(n * sizeof(std::complex<T>));
    MTL::Buffer *buffer = device_->newBuffer(static_cast<size_t>(bufferSize),
                                             MTL::ResourceStorageModeShared);
    if (!buffer) throw std::runtime_error("Failed to create Metal buffer");

    memcpy(buffer->contents(), data, static_cast<size_t>(bufferSize));
    config.buffer = &buffer;
    config.bufferSize = &bufferSize;

    VkFFTResult result = initializeVkFFT(&app, config);
    if (result != VKFFT_SUCCESS) {
      buffer->release();
      std::cerr << "[vkFFT] initializeVkFFT failed: " << result << "\n";
      throw std::runtime_error("Failed to initialize vkFFT");
    }

    // 创建 command buffer 和 command encoder
    MTL::CommandBuffer *cmdBuffer = queue_->commandBuffer();
    if (!cmdBuffer) {
      deleteVkFFT(&app);
      buffer->release();
      throw std::runtime_error("Failed to create Metal command buffer");
    }

    MTL::ComputeCommandEncoder *cmdEncoder = cmdBuffer->computeCommandEncoder();
    if (!cmdEncoder) {
      deleteVkFFT(&app);
      buffer->release();
      throw std::runtime_error("Failed to create Metal command encoder");
    }

    VkFFTLaunchParams params = {};
    params.buffer = &buffer;
    params.commandBuffer = cmdBuffer;
    params.commandEncoder = cmdEncoder;

    result = VkFFTAppend(&app, direction, &params);
    if (result != VKFFT_SUCCESS) {
      cmdEncoder->endEncoding();
      cmdEncoder->release();
      cmdBuffer->release();
      deleteVkFFT(&app);
      buffer->release();
      std::cerr << "[vkFFT] VkFFTAppend failed: " << result << "\n";
      throw std::runtime_error("Failed to execute vkFFT");
    }

    // 结束编码并提交
    cmdEncoder->endEncoding();
    cmdBuffer->commit();
    cmdBuffer->waitUntilCompleted();

    // 检查命令 buffer 状态
    auto status = cmdBuffer->status();
    if (status != MTL::CommandBufferStatusCompleted) {
      std::cerr << "[vkFFT] CommandBuffer failed, status: "
                << static_cast<int>(status) << "\n";
      auto *error = cmdBuffer->error();
      if (error) {
        std::cerr << "[vkFFT] Error: "
                  << error->localizedDescription()->utf8String() << "\n";
      }
    }

    memcpy(data, buffer->contents(), bufferSize);

    cmdEncoder->release();
    cmdBuffer->release();
    deleteVkFFT(&app);
    buffer->release();
  }

  // 高性能批量 FFT（使用缓存）
  void batchC2cImpl(complex32_t *data, int64_t n, int64_t batch,
                    int direction) override {
    if (!available_) {
      throw std::runtime_error("vkFFT Metal backend not available");
    }
    if (n <= 0 || (n & (n - 1)) != 0) {
      throw std::invalid_argument("FFT length must be power of 2");
    }

    // 使用缓存的 plan
    CachedPlan &plan = getOrCreatePlan(n, batch);

    // 拷贝输入数据到 buffer
    memcpy(plan.buffer->contents(), data, static_cast<size_t>(plan.bufferSize));

    // 创建 command buffer 和 encoder
    MTL::CommandBuffer *cmdBuffer = queue_->commandBuffer();
    MTL::ComputeCommandEncoder *cmdEncoder = cmdBuffer->computeCommandEncoder();

    VkFFTLaunchParams params = {};
    params.buffer = &plan.buffer;
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
    cmdBuffer->commit();
    cmdBuffer->waitUntilCompleted();

    memcpy(data, plan.buffer->contents(), plan.bufferSize);

    cmdEncoder->release();
    cmdBuffer->release();
  }

  // ========== 零拷贝 GPU 接口 ==========

  /**
   * @brief 获取 GPU buffer 指针（Unified Memory 上直接访问）
   */
  complex32_t *getGpuBuffer(int64_t n, int64_t batch) override {
    if (!available_) return nullptr;
    CachedPlan &plan = getOrCreatePlan(n, batch);
    return reinterpret_cast<complex32_t *>(plan.buffer->contents());
  }

  /**
   * @brief 在 GPU buffer 上执行 FFT（零拷贝）
   * @param wait 是否同步等待完成
   */
  void executeOnBuffer(int64_t n, int64_t batch, int direction) override {
    executeOnBufferAsync(n, batch, direction, true);
  }

  /**
   * @brief 异步执行 FFT（不等待完成）
   */
  void executeOnBufferAsync(int64_t n, int64_t batch, int direction,
                            bool wait) override {
    if (!available_) {
      throw std::runtime_error("vkFFT Metal backend not available");
    }

    CachedPlan &plan = getOrCreatePlan(n, batch);

    // 创建 command buffer 和 encoder
    MTL::CommandBuffer *cmdBuffer = queue_->commandBuffer();
    MTL::ComputeCommandEncoder *cmdEncoder = cmdBuffer->computeCommandEncoder();

    VkFFTLaunchParams params = {};
    params.buffer = &plan.buffer;
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
    cmdBuffer->commit();

    if (wait) {
      cmdBuffer->waitUntilCompleted();
      cmdEncoder->release();
      cmdBuffer->release();
    } else {
      // 保存用于后续同步和释放
      last_cmd_buffer_ = cmdBuffer;
      last_cmd_encoder_ = cmdEncoder;
    }
  }

  /**
   * @brief 同步等待所有异步操作完成
   */
  void sync() override {
    if (last_cmd_buffer_) {
      last_cmd_buffer_->waitUntilCompleted();
      last_cmd_encoder_->release();
      last_cmd_buffer_->release();
      last_cmd_buffer_ = nullptr;
      last_cmd_encoder_ = nullptr;
    }
  }

  MTL::CommandBuffer *last_cmd_buffer_ = nullptr;
  MTL::ComputeCommandEncoder *last_cmd_encoder_ = nullptr;
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
