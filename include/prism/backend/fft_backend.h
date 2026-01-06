/**
 * @file fft_backend.h
 * @ingroup backend
 * @brief FFT 后端抽象接口
 *
 * 定义了统一的 FFT 运算接口，屏蔽了底层实现（如 cuFFT, vDSP, VkFFT,
 * PocketFFT）的差异 所有接口通过模板支持 `real32_t` (float) 和 `real64_t`
 * (double) 两种精度 支持 C2C（复数-复数）、R2C（实数-复数）和
 * C2R（复数-实数）变换
 *
 * @note 所有 FFT 长度 N 必须是 2 的幂 ($N = 2^k$)，对于非 2
 * 的幂长度，具体行为取决于后端实现，通常会抛出异常
 */

#ifndef PRISM_FFT_BACKEND_H
#define PRISM_FFT_BACKEND_H

#include <complex>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "prism/runtime/fft.h"
#include "prism/types.h"

namespace prism::backend {

using runtime::FftTransType;

/// @addtogroup backend
/// @{

/**
 * @brief FFT 后端类型定义
 */
enum class FftBackendType : std::uint8_t {
  AUTO,    ///< 自动选择（默认策略）
  VDSP,    ///< macOS Accelerate vDSP
  CUDA,    ///< NVIDIA cuFFT
  HIP,     ///< AMD hipFFT
  VK_FFT,  ///< VkFFT (Vulkan/OpenCL/Metal)
  STUB     ///< 无可用后端 (Stub)
};

/**
 * @brief 设置 FFT 后端偏好
 * @param type 用户希望使用的后端类型
 */
void setFftBackendPreference(FftBackendType type);

/**
 * @brief 获取当前正在使用的 FFT 后端类型
 * @return 后端枚举值
 */
FftBackendType getFftBackendInUseType();

/**
 * @brief 获取当前后端的名称
 * @return 后端名称字符串 (e.g., "CUDA (cuFFT)")
 */
const char* getFftBackendInUseName();

/// @}

/**
 * @brief FFT 后端基类
 *
 * 所有具体的 FFT 实现（如 CudaFFTBackend,
 * VdspFFTBackend）都必须继承此类并实现虚函数
 * 接口设计为模板非虚函数调用受保护的虚实现函数（NVI
 * 模式的变体），以在基类层做类型检查
 *
 * @note 仅支持 `real32_t` 和 `real64_t` 类型的浮点数
 */
class FFTBackend {
 public:
  FFTBackend() = default;
  virtual ~FFTBackend() = default;

  FFTBackend(const FFTBackend&) = delete;
  FFTBackend& operator=(const FFTBackend&) = delete;
  FFTBackend(FFTBackend&&) = delete;
  FFTBackend& operator=(FFTBackend&&) = delete;

  /**
   * @brief 检查后端是否可用
   * @return 若硬件支持且初始化成功则返回 true
   */
  [[nodiscard]] virtual bool isAvailable() const = 0;

  /**
   * @brief 获取后端名称
   * @return 用于调试的名称字符串
   */
  [[nodiscard]] virtual const char* name() const = 0;

  /**
   * @brief 查询是否支持特定精度和变换类型
   * @param precision 标量精度 (F32/F64)
   * @param type 变换类型 (C2C/R2C/C2R)
   * @return 若支持返回 true
   */
  [[nodiscard]] virtual bool supports(ScalarType precision, FftTransType type) const = 0;

  /**
   * @brief 设备侧缓冲区描述符
   *
   * 用于在 Device 内存中持有数据，支持零拷贝和异步调度
   *
   * @note 对于 R2C/C2R，input 和 output 分别指向不同的设备内存区域
   * @note 对于 C2C，input 和 output 通常相同（In-Place），除非后端特殊要求
   */
  struct DeviceBuffer {
    void* input = nullptr;                   ///< 输入数据指针（Device 指针）
    void* output = nullptr;                  ///< 输出数据指针（Device 指针）
    void* backendHandle = nullptr;           ///< 后端私有句柄（如 plan 或 context）
    void* asyncHandle = nullptr;             ///< 异步流/事件句柄（如 cudaStream_t）
    void* nativeHandle = nullptr;            ///< 原生资源句柄（用于互操作）
    bool wrapped = false;                    ///< 是否为封装外部句柄（不拥有内存所有权）
    int64_t n = 0;                           ///< 变换长度
    int64_t batch = 0;                       ///< 批次数
    ScalarType precision = ScalarType::F32;  ///< 数据精度
    FftTransType type = FftTransType::C2C;   ///< 变换类型
    bool hostVisible = false;                ///< 内存是否对 Host 可见（统一内存）
  };

  // ========================================================================
  // C2C 变换（复数 -> 复数）
  // ========================================================================

  /**
   * @brief 执行 C2C 正变换 (Forward)
   * @tparam T 浮点类型 (real32_t / real64_t)
   * @param data 输入/输出复数数组（In-Place）
   * @param n 变换长度
   *
   * @note 执行 $y[k] = \sum_{n=0}^{N-1} x[n] e^{-j 2\pi k n / N}$
   */
  template <typename T>
  void forwardC2c(std::complex<T>* data, int64_t n) {
    static_assert(IS_REAL_TYPE_V<T>, "T must be real32_t or real64_t");
    if constexpr (std::is_same_v<T, real32_t>) {
      forwardC2cImpl(reinterpret_cast<complex32_t*>(data), n);
    } else {
      forwardC2cImpl(reinterpret_cast<complex64_t*>(data), n);
    }
  }

  /**
   * @brief 执行 C2C 逆变换 (Inverse)
   * @tparam T 浮点类型 (real32_t / real64_t)
   * @param data 输入/输出复数数组（In-Place）
   * @param n 变换长度
   * @param normalize 是否归一化（除以 n）
   *
   * @note 执行 $x[n] = \frac{1}{N} \sum_{k=0}^{N-1} y[k] e^{j 2\pi k n / N}$
   * (若 normalize=true)
   */
  template <typename T>
  void inverseC2c(std::complex<T>* data, int64_t n, bool normalize = true) {
    static_assert(IS_REAL_TYPE_V<T>, "T must be real32_t or real64_t");
    if constexpr (std::is_same_v<T, real32_t>) {
      inverseC2cImpl(reinterpret_cast<complex32_t*>(data), n, normalize);
    } else {
      inverseC2cImpl(reinterpret_cast<complex64_t*>(data), n, normalize);
    }
  }

  // ========================================================================
  // R2C 变换（实数 -> 复数）
  // ========================================================================

  /**
   * @brief 执行 R2C 正变换
   * @tparam T 浮点类型
   * @param in 输入实数数组（长度 n）
   * @param out 输出复数数组（长度 n/2 + 1）
   * @param n 实数域长度
   *
   * @note 输出利用了共轭对称性，只存储前一半频率分量
   */
  template <typename T>
  void forwardR2c(const T* in, std::complex<T>* out, int64_t n) {
    static_assert(IS_REAL_TYPE_V<T>, "T must be real32_t or real64_t");
    if constexpr (std::is_same_v<T, real32_t>) {
      forwardR2cImpl(in, reinterpret_cast<complex32_t*>(out), n);
    } else {
      forwardR2cImpl(in, reinterpret_cast<complex64_t*>(out), n);
    }
  }

  // ========================================================================
  // C2R 变换（复数 -> 实数）
  // ========================================================================

  /**
   * @brief 执行 C2R 逆变换
   * @tparam T 浮点类型
   * @param in 输入复数数组（长度 n/2 + 1）
   * @param out 输出实数数组（长度 n）
   * @param n 实数域长度
   * @param normalize 是否归一化
   *
   * @note 输入应满足共轭对称性，否则输出结果为未定义（通常对应实部）
   */
  template <typename T>
  void inverseC2r(const std::complex<T>* in, T* out, int64_t n, bool normalize = true) {
    static_assert(IS_REAL_TYPE_V<T>, "T must be real32_t or real64_t");
    if constexpr (std::is_same_v<T, real32_t>) {
      inverseC2rImpl(reinterpret_cast<const complex32_t*>(in), out, n, normalize);
    } else {
      inverseC2rImpl(reinterpret_cast<const complex64_t*>(in), out, n, normalize);
    }
  }

  // ========================================================================
  // Batch FFT (批量 C2C)
  // ========================================================================

  /**
   * @brief 执行批量 C2C 变换
   * @tparam T 浮点类型
   * @param data 数据数组（总长度 n * batch）
   * @param n 单个 FFT 长度
   * @param batch 批次数
   * @param direction 变换方向（-1: Forward, 1: Inverse）
   */
  template <typename T>
  void batchC2c(std::complex<T>* data, int64_t n, int64_t batch, int direction) {
    static_assert(IS_REAL_TYPE_V<T>, "T must be real32_t or real64_t");
    if constexpr (std::is_same_v<T, real32_t>) {
      batchC2cImpl(reinterpret_cast<complex32_t*>(data), n, batch, direction);
    } else {
      batchC2cImpl(reinterpret_cast<complex64_t*>(data), n, batch, direction);
    }
  }

  // ========================================================================
  // 设备缓冲区操作接口（用于零拷贝/异步）
  // ========================================================================

  /**
   * @brief 查询后端是否支持显式 Device Buffer 操作
   */
  [[nodiscard]] virtual bool supportsDeviceBuffer(ScalarType /*precision*/,
                                                  FftTransType /*type*/) const {
    return false;
  }

  /**
   * @brief 申请设备侧缓冲区
   * @return 若成功返回有效 buffer，否则返回空 buffer
   */
  virtual DeviceBuffer acquireDeviceBuffer(ScalarType /*precision*/, FftTransType /*type*/,
                                           int64_t /*n*/, int64_t /*batch*/) {
    return {};
  }

  /**
   * @brief 释放设备侧缓冲区
   */
  virtual void releaseDeviceBuffer(DeviceBuffer& /*buffer*/) {}

  /**
   * @brief 提交 FFT 任务到计算流（异步）
   * @param buffer 目标 buffer
   * @param direction C2C 方向
   */
  virtual void submitDeviceBuffer(DeviceBuffer& buffer, int direction) {
    (void)buffer;
    (void)direction;
    throw std::runtime_error("Device buffer FFT not supported");
  }

  /**
   * @brief 等待设备任务完成（同步）
   */
  virtual void deviceSync(DeviceBuffer& /*buffer*/) {}

  /**
   * @brief 将结果从 Device 拷贝回 Host
   */
  virtual void copyToHost(DeviceBuffer& buffer) { deviceSync(buffer); }

  /**
   * @brief 将输入从 Host 拷贝到 Device
   */
  virtual void copyToDevice(DeviceBuffer& /*buffer*/) {}

  /**
   * @brief 封装原生句柄为 DeviceBuffer
   */
  virtual DeviceBuffer wrapNativeHandle(void* /*handle*/, ScalarType /*precision*/,
                                        FftTransType /*type*/, int64_t /*n*/, int64_t /*batch*/) {
    throw std::runtime_error("wrap_native_handle not supported");
  }

  /**
   * @brief 解绑原生句柄
   */
  virtual void detachNativeHandle(DeviceBuffer& /*buffer*/) {}

  /**
   * @brief 提取底层原生指针
   */
  virtual void* getNativeDevicePtr(const DeviceBuffer& buffer) { return buffer.nativeHandle; }

  /** @brief 获取详细设备信息串 */
  [[nodiscard]] virtual std::string getBackendDeviceInfo() const { return name(); }
  /** @brief 全局同步 */
  virtual void sync() {}

 protected:
  // ========================================================================
  // 实现函数（虚接口）
  // ========================================================================

  virtual void forwardC2cImpl(complex32_t* data, int64_t n) = 0;
  virtual void forwardC2cImpl(complex64_t* data, int64_t n) = 0;

  virtual void inverseC2cImpl(complex32_t* data, int64_t n, bool normalize) = 0;
  virtual void inverseC2cImpl(complex64_t* data, int64_t n, bool normalize) = 0;

  virtual void forwardR2cImpl(const real32_t* in, complex32_t* out, int64_t n) = 0;
  virtual void forwardR2cImpl(const real64_t* in, complex64_t* out, int64_t n) = 0;

  virtual void inverseC2rImpl(const complex32_t* in, real32_t* out, int64_t n, bool normalize) = 0;
  virtual void inverseC2rImpl(const complex64_t* in, real64_t* out, int64_t n, bool normalize) = 0;

  virtual void batchC2cImpl(complex32_t* data, int64_t n, int64_t batch, int direction) {
    for (int64_t i = 0; i < batch; ++i) {
      if (direction < 0) {
        forwardC2cImpl(data + (i * n), n);
      } else {
        inverseC2cImpl(data + (i * n), n, false);
      }
    }
  }

  virtual void batchC2cImpl(complex64_t* data, int64_t n, int64_t batch, int direction) {
    for (int64_t i = 0; i < batch; ++i) {
      if (direction < 0) {
        forwardC2cImpl(data + (i * n), n);
      } else {
        inverseC2cImpl(data + (i * n), n, false);
      }
    }
  }
};

/**
 * @brief 获取唯一的全局 FFT 后端单例
 */
FFTBackend& getFftBackend();

/// @}

}  // namespace prism::backend

#endif  // PRISM_FFT_BACKEND_H
