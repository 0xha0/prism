/**
 * @file vkfft_opencl.cpp
 * @ingroup backend
 * @brief vkFFT OpenCL 后端实现
 *
 * 使用 vkFFT 库通过 OpenCL API 执行 FFT
 * 此后端在 VKFFT_BACKEND=3 时编译
 */
// NOLINTBEGIN(misc-include-cleaner)
#include <algorithm>
#include <complex>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "prism/backend/fft_backend.h"
#include "prism/types.h"
#include "vkFFT.h"

#if defined(PRISM_HAS_VKFFT) && (VKFFT_BACKEND == 3)

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
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
    err = clGetDeviceIDs(platform_, CL_DEVICE_TYPE_GPU, 0, nullptr, &numDevices);
    if (err != CL_SUCCESS || numDevices == 0) {
      // 尝试使用 CPU 设备
      err = clGetDeviceIDs(platform_, CL_DEVICE_TYPE_CPU, 0, nullptr, &numDevices);
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

    cl_device_type deviceType = 0;
    clGetDeviceInfo(device_, CL_DEVICE_TYPE, sizeof(deviceType), &deviceType, nullptr);
    deviceTypeName_ = ((deviceType & CL_DEVICE_TYPE_GPU) != 0U)   ? "GPU"
                      : ((deviceType & CL_DEVICE_TYPE_CPU) != 0U) ? "CPU"
                                                                  : "Other";

    deviceName_ = getDeviceInfoString_(CL_DEVICE_NAME);
    if (deviceName_.empty()) {
      deviceName_ = getDeviceInfoString_(CL_DEVICE_VENDOR);
    }

    name_ = "vkFFT (OpenCL, " + deviceTypeName_;
    if (!deviceName_.empty()) {
      name_ += ": " + deviceName_;
    }
    name_ += ")";

    // Check for double precision support
    cl_ulong fp64Config = 0;
    err = clGetDeviceInfo(device_, CL_DEVICE_DOUBLE_FP_CONFIG, sizeof(fp64Config), &fp64Config,
                          nullptr);
    if (err == CL_SUCCESS && fp64Config != 0) {
      supportsF64_ = true;
    }
  }

  VkFFTOpenCLBackend(const VkFFTOpenCLBackend&) = delete;
  VkFFTOpenCLBackend& operator=(const VkFFTOpenCLBackend&) = delete;
  VkFFTOpenCLBackend(VkFFTOpenCLBackend&&) = delete;
  VkFFTOpenCLBackend& operator=(VkFFTOpenCLBackend&&) = delete;

  ~VkFFTOpenCLBackend() override {
    for (auto& pending : pendingEvents_) {
      if (pending.event != nullptr) {
        clWaitForEvents(1, &pending.event);
        clReleaseEvent(pending.event);
      }
      if (pending.buffer) {
        pending.buffer->asyncHandle = nullptr;
      }
    }
    pendingEvents_.clear();
    clearCache();
    if (queue_ != nullptr) clReleaseCommandQueue(queue_);
    if (context_ != nullptr) clReleaseContext(context_);
  }

  [[nodiscard]] bool isAvailable() const override { return available_; }
  [[nodiscard]] const char* name() const override { return name_.c_str(); }

  [[nodiscard]] bool supports(ScalarType precision, FftTransType /*type*/) const override {
    if (!isFloatType(precision)) return false;
    auto const bytes = getComponentSize(precision);
    if (bytes == 8) return supportsF64_;  // NOLINT
    return bytes == 4;
  }

  void forwardC2cImpl(complex32_t* data, int64_t n) override { runC2c<real32_t>(data, n, 1, -1); }

  void forwardC2cImpl(complex64_t* data, int64_t n) override { runC2c<real64_t>(data, n, 1, -1); }

  void inverseC2cImpl(complex32_t* data, int64_t n, bool normalize) override {
    runC2c<real32_t>(data, n, 1, 1);
    if (normalize) {
      real32_t const scale = 1.0F / static_cast<real32_t>(n);
      for (int64_t i = 0; i < n; ++i) data[i] *= scale;
    }
  }

  void inverseC2cImpl(complex64_t* data, int64_t n, bool normalize) override {
    runC2c<real64_t>(data, n, 1, 1);
    if (normalize) {
      real64_t const scale = 1.0 / static_cast<real64_t>(n);
      for (int64_t i = 0; i < n; ++i) data[i] *= scale;
    }
  }

  void forwardR2cImpl(const real32_t* in, complex32_t* out, int64_t n) override {
    runR2c<real32_t>(in, out, n);
  }

  void forwardR2cImpl(const real64_t* in, complex64_t* out, int64_t n) override {
    runR2c<real64_t>(in, out, n);
  }

  void inverseC2rImpl(const complex32_t* in, real32_t* out, int64_t n, bool normalize) override {
    runC2r<real32_t>(in, out, n, normalize);
  }

  void inverseC2rImpl(const complex64_t* in, real64_t* out, int64_t n, bool normalize) override {
    runC2r<real64_t>(in, out, n, normalize);
  }

  // ========== 批量 FFT 实现 ==========

  void batchC2cImpl(complex32_t* data, int64_t n, int64_t batch, int direction) override {
    runC2c<real32_t>(data, n, batch, direction);
  }

  void batchC2cImpl(complex64_t* data, int64_t n, int64_t batch, int direction) override {
    runC2c<real64_t>(data, n, batch, direction);
  }

  // ========== 设备缓冲区接口 ==========

  [[nodiscard]] bool supportsDeviceBuffer(ScalarType precision, FftTransType type) const override {
    if (!available_) return false;
    return supports(precision, type);
  }

  DeviceBuffer acquireDeviceBuffer(ScalarType precision, FftTransType type, int64_t n,
                                   int64_t batch) override {
    if (!available_) {
      throw std::runtime_error("vkFFT OpenCL backend not available");
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
      auto& plan = getOrCreatePlan<real64_t>(type, n, batch);
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
      auto& plan = getOrCreatePlan<real32_t>(type, n, batch);
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

  void releaseDeviceBuffer(DeviceBuffer& buffer) override {
    if (buffer.asyncHandle != nullptr) {
      deviceSync(buffer);
    }
    if (buffer.wrapped && buffer.backendHandle != nullptr) {
      auto* plan = static_cast<CachedPlan*>(buffer.backendHandle);
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

  void submitDeviceBuffer(DeviceBuffer& buffer, int direction) override {
    if (!available_) {
      throw std::runtime_error("vkFFT OpenCL backend not available");
    }
    if (buffer.backendHandle == nullptr) {
      throw std::runtime_error("Invalid device buffer handle");
    }
    auto* plan = static_cast<CachedPlan*>(buffer.backendHandle);
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
      auto prev = static_cast<cl_event>(buffer.asyncHandle);
      removePendingEvent(prev);
      clReleaseEvent(prev);
      buffer.asyncHandle = nullptr;
    }
    cl_event event = enqueuePlan(*plan, direction);
    pendingEvents_.push_back({event, &buffer});
    buffer.asyncHandle = event;
  }

  void deviceSync(DeviceBuffer& buffer) override {
    if (buffer.asyncHandle == nullptr) return;
    auto event = static_cast<cl_event>(buffer.asyncHandle);
    clWaitForEvents(1, &event);
    removePendingEvent(event);
    clReleaseEvent(event);
    buffer.asyncHandle = nullptr;
  }

  void copyToHost(DeviceBuffer& buffer) override {
    if (!buffer.hostVisible || buffer.output == nullptr) {
      throw std::runtime_error("Device buffer is not host visible");
    }
    deviceSync(buffer);
    auto* plan = static_cast<CachedPlan*>(buffer.backendHandle);
    cl_mem outBuf = nullptr;
    size_t outBytes = 0;
    auto const realBytes = static_cast<size_t>(getComponentSize(buffer.precision));
    size_t const complexBytes = realBytes * 2;
    if (buffer.type == FftTransType::C2C) {
      outBuf = plan->buffer;
      outBytes = static_cast<size_t>(buffer.n * buffer.batch) * complexBytes;
    } else if (buffer.type == FftTransType::R2C) {
      outBuf = plan->buffer;
      outBytes = static_cast<size_t>((buffer.n / 2 + 1) * buffer.batch) * complexBytes;
    } else {
      outBuf = plan->inputBuffer;
      outBytes = static_cast<size_t>(buffer.n * buffer.batch) * realBytes;
    }
    cl_int const err = clEnqueueReadBuffer(queue_, outBuf, CL_TRUE, 0, outBytes, buffer.output, 0,
                                           nullptr, nullptr);
    if (err != CL_SUCCESS) {
      throw std::runtime_error("Failed to read OpenCL buffer");
    }
  }

  void copyToDevice(DeviceBuffer& buffer) override {
    if (!buffer.hostVisible || buffer.input == nullptr) {
      throw std::runtime_error("Device buffer is not host visible");
    }
    auto* plan = static_cast<CachedPlan*>(buffer.backendHandle);
    cl_mem inBuf = nullptr;
    size_t inBytes = 0;
    auto const realBytes = static_cast<size_t>(getComponentSize(buffer.precision));
    size_t const complexBytes = realBytes * 2;
    if (buffer.type == FftTransType::C2C) {
      inBuf = plan->buffer;
      inBytes = static_cast<size_t>(buffer.n * buffer.batch) * complexBytes;
    } else if (buffer.type == FftTransType::R2C) {
      inBuf = plan->inputBuffer;
      inBytes = static_cast<size_t>(buffer.n * buffer.batch) * realBytes;
    } else {
      inBuf = plan->buffer;
      inBytes = static_cast<size_t>((buffer.n / 2 + 1) * buffer.batch) * complexBytes;
    }
    cl_int const err =
        clEnqueueWriteBuffer(queue_, inBuf, CL_TRUE, 0, inBytes, buffer.input, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
      throw std::runtime_error("Failed to write OpenCL buffer");
    }
  }

  DeviceBuffer wrapNativeHandle(void* handle, ScalarType precision, FftTransType type, int64_t n,
                                int64_t batch) override {
    if (!available_) {
      throw std::runtime_error("vkFFT OpenCL backend not available");
    }
    PlanKind const kind = planKindFor(type);
    bool const complex = isComplexType(precision);
    if (kind == PlanKind::C2C && !complex) {
      throw std::invalid_argument("C2C requires complex precision");
    }
    if (kind == PlanKind::R2C && complex) {
      throw std::invalid_argument("R2C/C2R requires real precision");
    }

    cl_mem inputHandle = nullptr;
    cl_mem outputHandle = nullptr;
    if (kind == PlanKind::C2C) {
      outputHandle = static_cast<cl_mem>(handle);
      inputHandle = outputHandle;
    } else {
      struct HandlePair {
        void* input;
        void* output;
      };
      auto* pair = static_cast<HandlePair*>(handle);
      if (!pair || !pair->input || !pair->output) {
        throw std::invalid_argument("R2C/C2R requires handle pair {input, output}");
      }
      inputHandle = static_cast<cl_mem>(pair->input);
      outputHandle = static_cast<cl_mem>(pair->output);
    }

    auto plan = std::make_unique<CachedPlan>();
    plan->fftSize = n;
    plan->batch = batch;
    plan->kind = kind;
    plan->precision = precision;
    plan->ownsBuffers = false;
    plan->valid = false;

    auto const realBytes = static_cast<size_t>(getComponentSize(precision));
    size_t const complexBytes = realBytes * 2;
    if (kind == PlanKind::C2C) {
      plan->buffer = outputHandle;
      plan->bufferSize = static_cast<uint64_t>(n * batch * complexBytes);
      plan->hostComplexStorage.resize(static_cast<size_t>(plan->bufferSize));
      plan->hostComplex = plan->hostComplexStorage.data();
    } else {
      plan->inputBuffer = inputHandle;
      plan->buffer = outputHandle;
      plan->inputBufferSize = static_cast<uint64_t>(n * batch * realBytes);
      plan->bufferSize = static_cast<uint64_t>((n / 2 + 1) * batch * complexBytes);
      plan->hostRealStorage.resize(static_cast<size_t>(plan->inputBufferSize));
      plan->hostComplexStorage.resize(static_cast<size_t>(plan->bufferSize));
      plan->hostReal = plan->hostRealStorage.data();
      plan->hostComplex = plan->hostComplexStorage.data();
    }

    VkFFTConfiguration config = {};
    config.FFTdim = 1;
    config.size[0] = static_cast<uint64_t>(n);
    config.numberBatches = static_cast<uint64_t>(batch);
    // NOLINTNEXTLINE
    if (getComponentSize(precision) == 8) {
      config.doublePrecision = 1;
    }
    config.device = &device_;
    config.context = &context_;

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
      config.bufferStride[0] = static_cast<uint64_t>((n / 2) + 1);
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
    buffer.hostVisible = true;
    buffer.input = (kind == PlanKind::C2C)
                       ? plan->hostComplex
                       : (type == FftTransType::R2C ? plan->hostReal : plan->hostComplex);
    buffer.output = (kind == PlanKind::C2C)
                        ? plan->hostComplex
                        : (type == FftTransType::R2C ? plan->hostComplex : plan->hostReal);
    buffer.backendHandle = plan.release();
    buffer.nativeHandle = outputHandle;
    buffer.wrapped = true;
    return buffer;
  }

  void detachNativeHandle(DeviceBuffer& buffer) override {
    if (!buffer.wrapped || buffer.backendHandle == nullptr) return;
    auto* plan = static_cast<CachedPlan*>(buffer.backendHandle);
    releasePlan(*plan);
    delete plan;
    buffer.backendHandle = nullptr;
    buffer.input = nullptr;
    buffer.output = nullptr;
    buffer.asyncHandle = nullptr;
    buffer.nativeHandle = nullptr;
    buffer.wrapped = false;
  }

  void* getNativeDevicePtr(const DeviceBuffer& buffer) override { return buffer.nativeHandle; }

  [[nodiscard]] std::string getBackendDeviceInfo() const override { return name_; }

  /**
   * @brief 同步等待所有异步操作完成
   */
  void sync() override {
    for (auto& pending : pendingEvents_) {
      if (pending.event != nullptr) {
        clWaitForEvents(1, &pending.event);
        clReleaseEvent(pending.event);
      }
      if (pending.buffer) {
        pending.buffer->asyncHandle = nullptr;
      }
    }
    pendingEvents_.clear();
    if (queue_ != nullptr) clFinish(queue_);
  }

  void clearCache() {
    for (auto& [key, plan] : cache_) {
      if (plan) {
        releasePlan(*plan);
      }
    }
    cache_.clear();
  }

 private:
  [[nodiscard]] std::string getDeviceInfoString_(cl_device_info param) const {
    size_t size = 0;
    if (clGetDeviceInfo(device_, param, 0, nullptr, &size) != CL_SUCCESS || size == 0) {
      return {};
    }
    std::string out(size, '\0');
    if (clGetDeviceInfo(device_, param, size, out.data(), nullptr) != CL_SUCCESS) {
      return {};
    }
    if (!out.empty() && out.back() == '\0') {
      out.pop_back();
    }
    return out;
  }

  cl_platform_id platform_ = nullptr;
  cl_device_id device_ = nullptr;
  cl_context context_ = nullptr;
  cl_command_queue queue_ = nullptr;
  std::string name_ = "vkFFT (OpenCL)";
  std::string deviceName_;
  std::string deviceTypeName_;
  bool available_ = false;
  bool supportsF64_ = false;
  struct PendingEvent {
    cl_event event = nullptr;
    DeviceBuffer* buffer = nullptr;
  };
  std::vector<PendingEvent> pendingEvents_;

  enum class PlanKind : std::uint8_t { C2C, R2C };

  struct PlanKey {
    FftTransType type = FftTransType::C2C;
    ScalarType precision = ScalarType::F32;
    int64_t n = 0;
    int64_t batch = 0;

    bool operator==(const PlanKey& other) const {
      return type == other.type && precision == other.precision && n == other.n &&
             batch == other.batch;
    }
  };

  struct PlanKeyHash {
    std::size_t operator()(const PlanKey& key) const {
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
    cl_mem buffer = nullptr;
    cl_mem inputBuffer = nullptr;
    uint64_t bufferSize = 0;
    uint64_t inputBufferSize = 0;
    int64_t fftSize = 0;
    int64_t batch = 0;
    PlanKind kind = PlanKind::C2C;
    ScalarType precision = ScalarType::F32;
    bool ownsBuffers = true;
    bool valid = false;
    // 主机 staging 缓冲（用于显式拷贝）
    std::vector<uint8_t> hostComplexStorage;
    std::vector<uint8_t> hostRealStorage;
    void* hostComplex = nullptr;
    void* hostReal = nullptr;
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

  void releasePlan(CachedPlan& plan) {
    plan.hostComplex = nullptr;
    plan.hostReal = nullptr;
    plan.hostComplexStorage.clear();
    plan.hostRealStorage.clear();
    if (queue_ != nullptr) clFinish(queue_);
    if (plan.valid) {
      deleteVkFFT(&plan.app);
    }
    if (plan.ownsBuffers) {
      if (plan.buffer != nullptr) {
        clReleaseMemObject(plan.buffer);
        plan.buffer = nullptr;
      }
      if (plan.inputBuffer != nullptr) {
        clReleaseMemObject(plan.inputBuffer);
        plan.inputBuffer = nullptr;
      }
    }
    plan.buffer = nullptr;
    plan.inputBuffer = nullptr;
    plan.valid = false;
  }

  void removePendingEvent(cl_event event) {
    auto it =
        std::remove_if(pendingEvents_.begin(), pendingEvents_.end(),
                       [event](const PendingEvent& pending) { return pending.event == event; });
    pendingEvents_.erase(it, pendingEvents_.end());
  }

  template <typename T>
  CachedPlan& getOrCreatePlan(FftTransType type, int64_t n, int64_t batch) {
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

    cl_int err = 0;
    if (kind == PlanKind::C2C) {
      plan->bufferSize = static_cast<uint64_t>(n * batch * sizeof(std::complex<T>));
      plan->buffer = clCreateBuffer(context_, CL_MEM_READ_WRITE,
                                    static_cast<size_t>(plan->bufferSize), nullptr, &err);
      if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to create OpenCL buffer");
      }
      plan->hostComplexStorage.resize(static_cast<size_t>(plan->bufferSize));
      plan->hostComplex = plan->hostComplexStorage.data();
    } else {
      plan->inputBufferSize = static_cast<uint64_t>(n * sizeof(T) * batch);
      plan->bufferSize = static_cast<uint64_t>((n / 2 + 1) * sizeof(std::complex<T>) * batch);
      plan->inputBuffer = clCreateBuffer(context_, CL_MEM_READ_WRITE,
                                         static_cast<size_t>(plan->inputBufferSize), nullptr, &err);
      if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to create OpenCL input buffer");
      }
      plan->buffer = clCreateBuffer(context_, CL_MEM_READ_WRITE,
                                    static_cast<size_t>(plan->bufferSize), nullptr, &err);
      if (err != CL_SUCCESS) {
        clReleaseMemObject(plan->inputBuffer);
        throw std::runtime_error("Failed to create OpenCL buffer");
      }
      plan->hostRealStorage.resize(static_cast<size_t>(plan->inputBufferSize));
      plan->hostComplexStorage.resize(static_cast<size_t>(plan->bufferSize));
      plan->hostReal = plan->hostRealStorage.data();
      plan->hostComplex = plan->hostComplexStorage.data();
    }

    VkFFTConfiguration config = {};
    config.FFTdim = 1;
    config.size[0] = static_cast<uint64_t>(n);
    config.numberBatches = static_cast<uint64_t>(batch);
    if constexpr (std::is_same_v<T, real64_t>) {
      config.doublePrecision = 1;
    }
    config.device = &device_;
    config.context = &context_;

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
      config.bufferStride[0] = static_cast<uint64_t>((n / 2) + 1);
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
    auto& stored = cache_[key];
    stored = std::move(plan);
    return *stored;
  }

  cl_event enqueuePlan(CachedPlan& plan, int direction) {
    VkFFTLaunchParams params = {};
    params.commandQueue = &queue_;
    params.buffer = &plan.buffer;
    if (plan.kind == PlanKind::R2C) {
      params.inputBuffer = &plan.inputBuffer;
    }

    VkFFTResult const result = VkFFTAppend(&plan.app, direction, &params);
    if (result != VKFFT_SUCCESS) {
      throw std::runtime_error("Failed to execute vkFFT");
    }
    cl_event event = nullptr;
#ifdef CL_VERSION_1_2
    cl_int const err = clEnqueueMarkerWithWaitList(queue_, 0, nullptr, &event);
#else
    cl_int const err = clEnqueueMarker(queue_, &event);
#endif
    if (err != CL_SUCCESS || event == nullptr) {
      throw std::runtime_error("Failed to create OpenCL event");
    }
    return event;
  }

  template <typename T>
  void runC2c(std::complex<T>* data, int64_t n, int64_t batch, int direction) {
    if (!available_) {
      throw std::runtime_error("vkFFT OpenCL backend not available");
    }
    if (n <= 0 || (n & (n - 1)) != 0) {
      throw std::invalid_argument("FFT length must be power of 2");
    }
    CachedPlan& plan = getOrCreatePlan<T>(FftTransType::C2C, n, batch);
    auto const dataSize = static_cast<size_t>(n * batch * sizeof(std::complex<T>));
    cl_int err =
        clEnqueueWriteBuffer(queue_, plan.buffer, CL_TRUE, 0, dataSize, data, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
      throw std::runtime_error("Failed to write OpenCL buffer");
    }
    cl_event event = enqueuePlan(plan, direction);
    clWaitForEvents(1, &event);
    clReleaseEvent(event);
    err = clEnqueueReadBuffer(queue_, plan.buffer, CL_TRUE, 0, dataSize, data, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
      throw std::runtime_error("Failed to read OpenCL buffer");
    }
  }

  template <typename T>
  void runR2c(const T* in, std::complex<T>* out, int64_t n) {
    if (!available_) {
      throw std::runtime_error("vkFFT OpenCL backend not available");
    }
    if (n <= 0 || (n & (n - 1)) != 0) {
      throw std::invalid_argument("FFT length must be power of 2");
    }
    CachedPlan& plan = getOrCreatePlan<T>(FftTransType::R2C, n, 1);
    cl_int err = clEnqueueWriteBuffer(queue_, plan.inputBuffer, CL_TRUE, 0,
                                      static_cast<size_t>(n * sizeof(T)), in, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
      throw std::runtime_error("Failed to write OpenCL input buffer");
    }
    cl_event event = enqueuePlan(plan, -1);
    clWaitForEvents(1, &event);
    clReleaseEvent(event);
    err = clEnqueueReadBuffer(queue_, plan.buffer, CL_TRUE, 0,
                              static_cast<size_t>((n / 2 + 1) * sizeof(std::complex<T>)), out, 0,
                              nullptr, nullptr);
    if (err != CL_SUCCESS) {
      throw std::runtime_error("Failed to read OpenCL output buffer");
    }
  }

  template <typename T>
  void runC2r(const std::complex<T>* in, T* out, int64_t n, bool normalize) {
    if (!available_) {
      throw std::runtime_error("vkFFT OpenCL backend not available");
    }
    if (n <= 0 || (n & (n - 1)) != 0) {
      throw std::invalid_argument("FFT length must be power of 2");
    }
    CachedPlan& plan = getOrCreatePlan<T>(FftTransType::C2R, n, 1);
    cl_int err = clEnqueueWriteBuffer(queue_, plan.buffer, CL_TRUE, 0,
                                      static_cast<size_t>((n / 2 + 1) * sizeof(std::complex<T>)),
                                      in, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
      throw std::runtime_error("Failed to write OpenCL input buffer");
    }
    cl_event event = enqueuePlan(plan, 1);
    clWaitForEvents(1, &event);
    clReleaseEvent(event);
    err = clEnqueueReadBuffer(queue_, plan.inputBuffer, CL_TRUE, 0,
                              static_cast<size_t>(n * sizeof(T)), out, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
      throw std::runtime_error("Failed to read OpenCL output buffer");
    }
    if (normalize) {
      T const scale = static_cast<T>(1) / static_cast<T>(n);
      for (int64_t i = 0; i < n; ++i) {
        out[i] *= scale;
      }
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
