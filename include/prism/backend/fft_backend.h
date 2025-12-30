/**
 * @file fft_backend.h
 * @ingroup backend
 * @brief FFT 后端接口
 *
 * 定义统一的 FFT 模板化抽象，屏蔽 vDSP/cuFFT/hipFFT/vkFFT 等差异。
 * 所有长度均假定为 2 的幂，异常由具体实现抛出。
 */

#ifndef PRISM_FFT_BACKEND_H
#define PRISM_FFT_BACKEND_H

#include <complex>
#include <cstdint>

#include "prism/runtime/fft.h"
#include "prism/types.h"

namespace prism::backend {

using runtime::FftTransType;

/// @addtogroup backend
/// @{

/**
 * @brief FFT 后端抽象接口
 *
 * 模板化接口，T 只能为 real32_t 或 real64_t。
 * 支持 C2C (复数到复数), R2C (实数到复数), C2R (复数到实数)。
 */
class FFTBackend {
 public:
  FFTBackend() = default;
  virtual ~FFTBackend() = default;

  FFTBackend(const FFTBackend&) = delete;
  FFTBackend& operator=(const FFTBackend&) = delete;
  FFTBackend(FFTBackend&&) = delete;
  FFTBackend& operator=(FFTBackend&&) = delete;

  /** @brief 是否可用（硬件/驱动检测） */
  [[nodiscard]] virtual bool isAvailable() const = 0;

  /** @brief 后端名称，用于日志与调试 */
  [[nodiscard]] virtual const char* name() const = 0;

  /** @brief 是否支持特定精度和变换类型 */
  [[nodiscard]] virtual bool supports(ScalarType precision,
                                      FftTransType type) const = 0;

  // ========================================================================
  // C2C 变换（模板接口）
  // ========================================================================

  /**
   * @brief 复数 FFT 正变换 (C2C)
   * @tparam T real32_t 或 real64_t
   * @param data 输入/输出数据（原地操作）
   * @param n 数据长度（必须是 2 的幂）
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
   * @brief 复数 FFT 逆变换 (C2C)
   * @tparam T real32_t 或 real64_t
   * @param data 输入/输出数据（原地操作）
   * @param n 数据长度
   * @param normalize 是否归一化 (除以 n)
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
  // R2C 变换（模板接口）
  // ========================================================================

  /**
   * @brief 实数 FFT 正变换 (R2C)
   * @tparam T real32_t 或 real64_t
   * @param in 输入实数数据
   * @param out 输出复数数据（长度 n/2+1）
   * @param n 实数数据长度
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
  // C2R 变换（模板接口）
  // ========================================================================

  /**
   * @brief 复数 FFT 逆变换到实数 (C2R)
   * @tparam T real32_t 或 real64_t
   * @param in 输入复数数据（长度 n/2+1）
   * @param out 输出实数数据
   * @param n 实数数据长度
   * @param normalize 是否归一化
   */
  template <typename T>
  void inverseC2r(const std::complex<T>* in, T* out, int64_t n,
                  bool normalize = true) {
    static_assert(IS_REAL_TYPE_V<T>, "T must be real32_t or real64_t");
    if constexpr (std::is_same_v<T, real32_t>) {
      inverseC2rImpl(reinterpret_cast<const complex32_t*>(in), out, n,
                     normalize);
    } else {
      inverseC2rImpl(reinterpret_cast<const complex64_t*>(in), out, n,
                     normalize);
    }
  }

  // ========================================================================
  // Batch FFT（模板接口）
  // ========================================================================

  /**
   * @brief 批量 C2C FFT
   * @tparam T real32_t 或 real64_t
   * @param data 输入/输出数据（n * batch 个复数）
   * @param n 单个 FFT 长度
   * @param batch 批次数
   * @param direction -1 正变换，1 逆变换
   */
  template <typename T>
  void batchC2c(std::complex<T>* data, int64_t n, int64_t batch,
                int direction) {
    static_assert(IS_REAL_TYPE_V<T>, "T must be real32_t or real64_t");
    if constexpr (std::is_same_v<T, real32_t>) {
      batchC2cImpl(reinterpret_cast<complex32_t*>(data), n, batch, direction);
    } else {
      batchC2cImpl(reinterpret_cast<complex64_t*>(data), n, batch, direction);
    }
  }

  // ========================================================================
  // 零拷贝 GPU 接口
  // ========================================================================

  /**
   * @brief 获取内部 GPU 缓冲区指针
   *
   * 部分后端在设备侧预分配批量缓冲，可通过该接口直接取得指针以避免重复分配。
   * 默认返回 `nullptr`，表示不支持零拷贝路径。
   */
  virtual complex32_t* getGpuBuffer(int64_t /*n*/, int64_t /*batch*/) {
    return nullptr;
  }
  /**
   * @brief 在内部缓冲区上执行批量 FFT（同步）
   * @param n 单个 FFT 长度
   * @param batch 批次数
   * @param direction -1 正变换，1 逆变换
   */
  virtual void executeOnBuffer(int64_t n, int64_t batch, int direction) {}
  /**
   * @brief 在内部缓冲区上执行批量 FFT（可选异步）
   * @param wait 是否在返回前等待执行完成
   */
  virtual void executeOnBufferAsync(int64_t n, int64_t batch, int direction,
                                    bool /*wait*/) {
    executeOnBuffer(n, batch, direction);
  }
  /** @brief 同步等待后端任务（默认空实现） */
  virtual void sync() {}

 protected:
  // ========================================================================
  // 实现接口（子类覆盖）
  // ========================================================================

  virtual void forwardC2cImpl(complex32_t* data, int64_t n) = 0;
  virtual void forwardC2cImpl(complex64_t* data, int64_t n) = 0;

  virtual void inverseC2cImpl(complex32_t* data, int64_t n, bool normalize) = 0;
  virtual void inverseC2cImpl(complex64_t* data, int64_t n, bool normalize) = 0;

  virtual void forwardR2cImpl(const real32_t* in, complex32_t* out,
                              int64_t n) = 0;
  virtual void forwardR2cImpl(const real64_t* in, complex64_t* out,
                              int64_t n) = 0;

  virtual void inverseC2rImpl(const complex32_t* in, real32_t* out, int64_t n,
                              bool normalize) = 0;
  virtual void inverseC2rImpl(const complex64_t* in, real64_t* out, int64_t n,
                              bool normalize) = 0;

  virtual void batchC2cImpl(complex32_t* data, int64_t n, int64_t batch,
                            int direction) {
    for (int64_t i = 0; i < batch; ++i) {
      if (direction < 0) {
        forwardC2cImpl(data + (i * n), n);
      } else {
        inverseC2cImpl(data + (i * n), n, false);
      }
    }
  }

  virtual void batchC2cImpl(complex64_t* data, int64_t n, int64_t batch,
                            int direction) {
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
 * @brief 获取当前可用的 FFT 后端
 */
FFTBackend& getFftBackend();

/// @}

}  // namespace prism::backend

#endif  // PRISM_FFT_BACKEND_H
