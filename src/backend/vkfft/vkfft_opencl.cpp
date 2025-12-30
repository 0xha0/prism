/**
 * @file vkfft_opencl.cpp
 * @ingroup backend
 * @brief vkFFT OpenCL 后端实现
 *
 * 使用 vkFFT 库通过 OpenCL API 执行 FFT
 * 此后端在 VKFFT_BACKEND=3 时编译
 */
// NOLINTBEGIN(misc-include-cleaner)
#include <complex>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "prism/backend/fft_backend.h"
#include "prism/types.h"
#include "vkFFT.h"

#if defined(PRISM_HAS_VKFFT) && (VKFFT_BACKEND == 3)

//NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define CL_TARGET_OPENCL_VERSION 120
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

namespace prism::backend {

/// @addtogroup backend
/// @{

/**
 * @brief vkFFT OpenCL FFT 后端实现
 *
 * 支持 C2C、R2C、C2R 变换，包含 plan 缓存、批量处理、零拷贝接口
 */
class VkFFTOpenCLBackend : public FFTBackend {
 public:
  VkFFTOpenCLBackend() {
    // 获取平台
    cl_uint numPlatforms = 0;
    cl_int err = clGetPlatformIDs(0, nullptr, &numPlatforms);
    if (err != CL_SUCCESS || numPlatforms == 0) {
      available_ = false;
      return;
    }

    std::vector<cl_platform_id> platforms(numPlatforms);
    clGetPlatformIDs(numPlatforms, platforms.data(), nullptr);
    platform_ = platforms[0];

    // 获取 GPU 设备
    cl_uint numDevices = 0;
    err =
        clGetDeviceIDs(platform_, CL_DEVICE_TYPE_GPU, 0, nullptr, &numDevices);
    if (err != CL_SUCCESS || numDevices == 0) {
      // 尝试使用 CPU 设备
      err = clGetDeviceIDs(platform_, CL_DEVICE_TYPE_CPU, 0, nullptr,
                           &numDevices);
      if (err != CL_SUCCESS || numDevices == 0) {
        available_ = false;
        return;
      }
      clGetDeviceIDs(platform_, CL_DEVICE_TYPE_CPU, 1, &device_, nullptr);
    } else {
      clGetDeviceIDs(platform_, CL_DEVICE_TYPE_GPU, 1, &device_, nullptr);
    }

    // 创建上下文
    context_ = clCreateContext(nullptr, 1, &device_, nullptr, nullptr, &err);
    if (err != CL_SUCCESS) {
      available_ = false;
      return;
    }

    // 创建命令队列
    queue_ = clCreateCommandQueue(context_, device_, 0, &err);
    if (err != CL_SUCCESS) {
      clReleaseContext(context_);
      available_ = false;
      return;
    }

    available_ = true;

    // Check for double precision support
    cl_ulong fp64Config = 0;
    err = clGetDeviceInfo(device_, CL_DEVICE_DOUBLE_FP_CONFIG,
                          sizeof(fp64Config), &fp64Config, nullptr);
    if (err == CL_SUCCESS && fp64Config != 0) {
      supportsF64_ = true;
    }
  }

  VkFFTOpenCLBackend(const VkFFTOpenCLBackend&) = delete;
  VkFFTOpenCLBackend& operator=(const VkFFTOpenCLBackend&) = delete;
  VkFFTOpenCLBackend(VkFFTOpenCLBackend&&) = delete;
  VkFFTOpenCLBackend& operator=(VkFFTOpenCLBackend&&) = delete;

  ~VkFFTOpenCLBackend() override {
    clearCache();
    if (queue_ != nullptr) clReleaseCommandQueue(queue_);
    if (context_ != nullptr) clReleaseContext(context_);
  }

  [[nodiscard]] bool isAvailable() const override { return available_; }
  [[nodiscard]] const char* name() const override { return "vkFFT (OpenCL)"; }

  [[nodiscard]] bool supports(ScalarType precision,
                              FftTransType /*type*/) const override {
    if (precision == ScalarType::F64) return supportsF64_;
    return precision == ScalarType::F32;
  }

  void forwardC2cImpl(complex32_t* data, int64_t n) override {
    executeC2c<real32_t>(data, n, -1);
  }

  void forwardC2cImpl(complex64_t* data, int64_t n) override {
    executeC2c<real64_t>(data, n, -1);
  }

  void inverseC2cImpl(complex32_t* data, int64_t n, bool normalize) override {
    executeC2c<real32_t>(data, n, 1);
    if (normalize) {
      real32_t const scale = 1.0F / static_cast<real32_t>(n);
      for (int64_t i = 0; i < n; ++i) data[i] *= scale;
    }
  }

  void inverseC2cImpl(complex64_t* data, int64_t n, bool normalize) override {
    executeC2c<real64_t>(data, n, 1);
    if (normalize) {
      real64_t const scale = 1.0 / static_cast<real64_t>(n);
      for (int64_t i = 0; i < n; ++i) data[i] *= scale;
    }
  }

  void forwardR2cImpl(const real32_t* in, complex32_t* out,
                      int64_t n) override {
    std::vector<complex32_t> temp(n);
    for (int64_t i = 0; i < n; ++i) temp[i] = complex32_t(in[i], 0.0F);
    executeC2c<real32_t>(temp.data(), n, -1);
    for (int64_t i = 0; i <= n / 2; ++i) out[i] = temp[i];
  }

  void forwardR2cImpl(const real64_t* in, complex64_t* out,
                      int64_t n) override {
    std::vector<complex64_t> temp(n);
    for (int64_t i = 0; i < n; ++i) temp[i] = complex64_t(in[i], 0.0);
    executeC2c<real64_t>(temp.data(), n, -1);
    for (int64_t i = 0; i <= n / 2; ++i) out[i] = temp[i];
  }

  void inverseC2rImpl(const complex32_t* in, real32_t* out, int64_t n,
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

  void inverseC2rImpl(const complex64_t* in, real64_t* out, int64_t n,
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

  // ========== 批量 FFT 实现 ==========

  void batchC2cImpl(complex32_t* data, int64_t n, int64_t batch,
                    int direction) override {
    executeBatchC2c<real32_t>(data, n, batch, direction);
  }

  void batchC2cImpl(complex64_t* data, int64_t n, int64_t batch,
                    int direction) override {
    executeBatchC2c<real64_t>(data, n, batch, direction);
  }

  // ========== 零拷贝 GPU 接口 ==========

  /**
   * @brief 获取 GPU buffer 指针（映射内存）
   */
  complex32_t* getGpuBuffer(int64_t n, int64_t batch) override {
    if (!available_) return nullptr;
    CachedPlan const& plan = getOrCreatePlan<real32_t>(n, batch);
    return plan.hostPtr32;
  }

  /**
   * @brief 在 GPU buffer 上执行 FFT（同步）
   */
  void executeOnBuffer(int64_t n, int64_t batch, int direction) override {
    executeOnBufferAsync(n, batch, direction, true);
  }

  /**
   * @brief 异步执行 FFT
   */
  void executeOnBufferAsync(int64_t n, int64_t batch, int direction,
                            bool wait) override {
    if (!available_) {
      throw std::runtime_error("vkFFT OpenCL backend not available");
    }

    CachedPlan& plan = getOrCreatePlan<real32_t>(n, batch);

    VkFFTLaunchParams params = {};
    params.buffer = &plan.buffer;
    params.commandQueue = &queue_;

    VkFFTResult const result = VkFFTAppend(&plan.app, direction, &params);
    if (result != VKFFT_SUCCESS) {
      throw std::runtime_error("Failed to execute vkFFT");
    }

    if (wait) {
      clFinish(queue_);
    }
  }

  /**
   * @brief 同步等待所有异步操作完成
   */
  void sync() override {
    if (queue_ != nullptr) {
      clFinish(queue_);
    }
  }

  void clearCache() {
    for (auto& [key, plan] : cache_) {
      if (plan.valid) {
        deleteVkFFT(&plan.app);
        if (plan.buffer != nullptr) clReleaseMemObject(plan.buffer);
      }
    }
    cache_.clear();
  }

 private:
  cl_platform_id platform_ = nullptr;
  cl_device_id device_ = nullptr;
  cl_context context_ = nullptr;
  cl_command_queue queue_ = nullptr;
 bool available_ = false;
 bool supportsF64_ = false;
 // NOLINTNEXTLINE(readability-identifier-naming)
 static constexpr int kPrecisionBitShift = 63;
  // NOLINTNEXTLINE(readability-identifier-naming)
 static constexpr int kBatchShift = 32;

  // ========== vkFFT App 缓存 ==========
  struct CachedPlan {
    VkFFTApplication app = {};
    cl_mem buffer = nullptr;
    uint64_t bufferSize = 0;
    int64_t fftSize = 0;
    int64_t batch = 0;
    bool doublePrecision = false;
    bool valid = false;
    // 映射的主机指针（用于零拷贝）
    complex32_t* hostPtr32 = nullptr;
    complex64_t* hostPtr64 = nullptr;
  };

  std::unordered_map<uint64_t, CachedPlan> cache_;

  static uint64_t cacheKey(int64_t n, int64_t batch, bool doublePrecision) {
    // 高位存储精度标志，中间存储 n，低位存储 batch
    return (static_cast<uint64_t>(doublePrecision ? 1 : 0) << kPrecisionBitShift) |
           (static_cast<uint64_t>(n) << kBatchShift) | static_cast<uint64_t>(batch);
  }

  template <typename T>
  CachedPlan& getOrCreatePlan(int64_t n, int64_t batch) {
    constexpr bool isDouble = std::is_same_v<T, real64_t>;
    uint64_t const key = cacheKey(n, batch, isDouble);
    auto it = cache_.find(key);
    if (it != cache_.end() && it->second.valid) {
      return it->second;
    }

    // 创建新 plan
    CachedPlan& plan = cache_[key];
    plan.fftSize = n;
    plan.batch = batch;
    plan.doublePrecision = isDouble;
    plan.bufferSize =
        static_cast<uint64_t>(n * batch * sizeof(std::complex<T>));
    plan.valid = false;

    // 使用 CL_MEM_ALLOC_HOST_PTR 创建可映射的 buffer
    cl_int err = 0;
    plan.buffer =
        clCreateBuffer(context_, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                       static_cast<size_t>(plan.bufferSize), nullptr, &err);
    if (err != CL_SUCCESS) {
      cache_.erase(key);
      throw std::runtime_error("Failed to create OpenCL buffer");
    }

    // 映射 buffer 到主机内存
    void* mapped = clEnqueueMapBuffer(
        queue_, plan.buffer, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0,
        static_cast<size_t>(plan.bufferSize), 0, nullptr, nullptr, &err);
    if (err != CL_SUCCESS) {
      clReleaseMemObject(plan.buffer);
      cache_.erase(key);
      throw std::runtime_error("Failed to map OpenCL buffer");
    }

    if constexpr (isDouble) {
      plan.hostPtr64 = static_cast<complex64_t*>(mapped);
    } else {
      plan.hostPtr32 = static_cast<complex32_t*>(mapped);
    }

    // 初始化 vkFFT
    VkFFTConfiguration config = {};
    config.FFTdim = 1;
    config.size[0] = static_cast<uint64_t>(n);
    config.numberBatches = static_cast<uint64_t>(batch);
    if constexpr (isDouble) config.doublePrecision = 1;

    config.device = &device_;
    config.context = &context_;
    config.buffer = &plan.buffer;
    config.bufferSize = &plan.bufferSize;

    VkFFTResult const result = initializeVkFFT(&plan.app, config);
    if (result != VKFFT_SUCCESS) {
      clEnqueueUnmapMemObject(queue_, plan.buffer, mapped, 0, nullptr, nullptr);
      clReleaseMemObject(plan.buffer);
      cache_.erase(key);
      throw std::runtime_error("Failed to initialize vkFFT");
    }

    plan.valid = true;
    return plan;
  }

  template <typename T>
  void executeC2c(std::complex<T>* data, int64_t n, int direction) {
    if (!available_) {
      throw std::runtime_error("vkFFT OpenCL backend not available");
    }
    if (n <= 0 || (n & (n - 1)) != 0) {
      throw std::invalid_argument("FFT length must be power of 2");
    }

    // 对于单次 FFT，直接执行而不使用缓存（避免过多缓存）
    VkFFTConfiguration config = {};
    VkFFTApplication app = {};
    config.FFTdim = 1;
    config.size[0] = static_cast<uint64_t>(n);
    config.numberBatches = 1;
    if constexpr (std::is_same_v<T, real64_t>) config.doublePrecision = 1;

    config.device = &device_;
    config.context = &context_;

    auto bufferSize = static_cast<uint64_t>(n * sizeof(std::complex<T>));
    cl_int err = 0;
    cl_mem buffer =
        clCreateBuffer(context_, CL_MEM_READ_WRITE,
                       static_cast<size_t>(bufferSize), nullptr, &err);
    if (err != CL_SUCCESS) {
      throw std::runtime_error("Failed to create OpenCL buffer");
    }

    // 拷贝数据到 GPU
    err = clEnqueueWriteBuffer(queue_, buffer, CL_TRUE, 0,
                               static_cast<size_t>(bufferSize), data, 0,
                               nullptr, nullptr);
    if (err != CL_SUCCESS) {
      clReleaseMemObject(buffer);
      throw std::runtime_error("Failed to write to OpenCL buffer");
    }

    config.buffer = &buffer;
    config.bufferSize = &bufferSize;

    VkFFTResult result = initializeVkFFT(&app, config);
    if (result != VKFFT_SUCCESS) {
      clReleaseMemObject(buffer);
      std::cerr << "[vkFFT] initializeVkFFT failed: " << result << "\n";
      throw std::runtime_error("Failed to initialize vkFFT");
    }

    VkFFTLaunchParams params = {};
    params.buffer = &buffer;
    params.commandQueue = &queue_;

    result = VkFFTAppend(&app, direction, &params);
    if (result != VKFFT_SUCCESS) {
      deleteVkFFT(&app);
      clReleaseMemObject(buffer);
      std::cerr << "[vkFFT] VkFFTAppend failed: " << result << "\n";
      throw std::runtime_error("Failed to execute vkFFT");
    }

    // 等待完成
    clFinish(queue_);

    // 拷贝结果回 CPU
    err = clEnqueueReadBuffer(queue_, buffer, CL_TRUE, 0,
                              static_cast<size_t>(bufferSize), data, 0, nullptr,
                              nullptr);
    if (err != CL_SUCCESS) {
      deleteVkFFT(&app);
      clReleaseMemObject(buffer);
      throw std::runtime_error("Failed to read from OpenCL buffer");
    }

    deleteVkFFT(&app);
    clReleaseMemObject(buffer);
  }

  template <typename T>
  void executeBatchC2c(std::complex<T>* data, int64_t n, int64_t batch,
                       int direction) {
    if (!available_) {
      throw std::runtime_error("vkFFT OpenCL backend not available");
    }
    if (n <= 0 || (n & (n - 1)) != 0) {
      throw std::invalid_argument("FFT length must be power of 2");
    }

    // 使用缓存的 plan
    CachedPlan& plan = getOrCreatePlan<T>(n, batch);

    // 拷贝输入数据到映射的 buffer
    auto const dataSize =
        static_cast<size_t>(n * batch * sizeof(std::complex<T>));
    if constexpr (std::is_same_v<T, real64_t>) {
      std::memcpy(plan.hostPtr64, data, dataSize);
    } else {
      std::memcpy(plan.hostPtr32, data, dataSize);
    }

    // 执行 FFT
    VkFFTLaunchParams params = {};
    params.buffer = &plan.buffer;
    params.commandQueue = &queue_;

    VkFFTResult const result = VkFFTAppend(&plan.app, direction, &params);
    if (result != VKFFT_SUCCESS) {
      throw std::runtime_error("Failed to execute vkFFT");
    }

    // 等待完成
    clFinish(queue_);

    // 拷贝结果回输出
    if constexpr (std::is_same_v<T, real64_t>) {
      std::memcpy(data, plan.hostPtr64, dataSize);
    } else {
      std::memcpy(data, plan.hostPtr32, dataSize);
    }
  }
};

namespace {
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
VkFFTOpenCLBackend gVkfftOpenclBackend;
}  // namespace

// 导出给 fft_backend.cpp 使用
FFTBackend& getVkfftBackend() { return gVkfftOpenclBackend; }

/// @}

}  // namespace prism::backend

#endif  // PRISM_HAS_VKFFT && VKFFT_BACKEND == 3
// NOLINTEND(misc-include-cleaner)
