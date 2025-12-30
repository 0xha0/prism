/**
 * @file fft.h
 * @ingroup runtime
 * @brief FFT Anchor 接口
 *
 * 提供模板化 FFT 接口，自动识别精度（real32_t/real64_t）。
 * 作为 Anchor 算子直接调用 FFT backend，不参与 Halide 计算图。
 */

#ifndef PRISM_RUNTIME_FFT_H
#define PRISM_RUNTIME_FFT_H

#include <complex>

#include "prism/types.h"

namespace prism::runtime {

/// @addtogroup runtime
/// @{

/**
 * @brief FFT 变换类型
 */
enum class FftTransType : std::uint8_t {
  C2C = 1,  ///< 复数到复数
  R2C = 2,  ///< 实数到复数
  C2R = 3   ///< 复数到实数
};

/**
 * @brief FFT 执行选项
 */
struct FFTOptions {
  bool normalize = true;  ///< 逆变换是否归一化（除以 N）
  bool async = false;     ///< 是否异步执行
};

/**
 * @brief FFT Anchor 接口
 *
 * 模板化接口，T 只能为 real32_t 或 real64_t。
 *
 * @code
 * // C2C 原位变换
 * FFT::forward(complex_data, 1024);
 * FFT::inverse(complex_data, 1024);
 *
 * // R2C / C2R 非原位变换
 * FFT::forward(real_in, complex_out, 1024);
 * FFT::inverse(complex_in, real_out, 1024);
 *
 * // 异步执行
 * FFT::forward(data, 1024, {.async = true});
 * FFT::sync();
 * @endcode
 */
class FFT {
 public:
  // ========================================================================
  // C2C 变换（原位）
  // ========================================================================

  /**
   * @brief C2C 正变换
   * @tparam T real32_t 或 real64_t（自动推导）
   */
  template <typename T>
  static void forward(std::complex<T>* data, int64_t n, FFTOptions opts = {}) {
    static_assert(IS_REAL_TYPE_V<T>, "T must be real32_t or real64_t");
    forwardImpl(data, n, opts);
  }

  /**
   * @brief C2C 逆变换
   */
  template <typename T>
  static void inverse(std::complex<T>* data, int64_t n, FFTOptions opts = {}) {
    static_assert(IS_REAL_TYPE_V<T>, "T must be real32_t or real64_t");
    inverseImpl(data, n, opts);
  }

  // ========================================================================
  // R2C 变换（非原位）
  // ========================================================================

  /**
   * @brief R2C 正变换
   */
  template <typename T>
  static void forward(const T* in, std::complex<T>* out, int64_t n) {
    static_assert(IS_REAL_TYPE_V<T>, "T must be real32_t or real64_t");
    forwardR2cImpl(in, out, n);
  }

  // ========================================================================
  // C2R 变换（非原位）
  // ========================================================================

  /**
   * @brief C2R 逆变换
   */
  template <typename T>
  static void inverse(const std::complex<T>* in, T* out, int64_t n,
                      FFTOptions opts = {}) {
    static_assert(IS_REAL_TYPE_V<T>, "T must be real32_t or real64_t");
    inverseC2rImpl(in, out, n, opts);
  }

  // ========================================================================
  // 批量变换
  // ========================================================================

  /**
   * @brief 批量 C2C 变换
   * @param data 输入/输出复数数组（长度 = n * batch，原地操作）
   * @param n 单个 FFT 的长度（需为 2 的幂）
   * @param batch 批次数
   * @param direction -1 = 正变换，1 = 逆变换
   * @param opts 执行选项（归一化/异步）
   */
  template <typename T>
  static void batch(std::complex<T>* data, int64_t n, int64_t batch,
                    int direction, FFTOptions opts = {}) {
    static_assert(IS_REAL_TYPE_V<T>, "T must be real32_t or real64_t");
    batchImpl(data, n, batch, direction, opts);
  }

  // ========================================================================
  // 同步 & 能力查询
  // ========================================================================

  static void sync();

  /**
   * @brief 查询后端是否支持特定精度和变换类型
   */
  static bool supports(ScalarType precision, FftTransType type);

 private:
  // 实现分派（在 FFT.cpp 中定义）
  static void forwardImpl(std::complex<real32_t>* data, int64_t n,
                          FFTOptions opts);
  static void forwardImpl(std::complex<real64_t>* data, int64_t n,
                          FFTOptions opts);

  static void inverseImpl(std::complex<real32_t>* data, int64_t n,
                          FFTOptions opts);
  static void inverseImpl(std::complex<real64_t>* data, int64_t n,
                          FFTOptions opts);

  static void forwardR2cImpl(const real32_t* in, std::complex<real32_t>* out,
                             int64_t n);
  static void forwardR2cImpl(const real64_t* in, std::complex<real64_t>* out,
                             int64_t n);

  static void inverseC2rImpl(const std::complex<real32_t>* in, real32_t* out,
                             int64_t n, FFTOptions opts);
  static void inverseC2rImpl(const std::complex<real64_t>* in, real64_t* out,
                             int64_t n, FFTOptions opts);

  static void batchImpl(std::complex<real32_t>* data, int64_t n, int64_t batch,
                        int direction, FFTOptions opts);
  static void batchImpl(std::complex<real64_t>* data, int64_t n, int64_t batch,
                        int direction, FFTOptions opts);
};

/// @}

}  // namespace prism::runtime

#endif  // PRISM_RUNTIME_FFT_H
