/**
 * @file fft.h
 * @ingroup runtime
 * @brief FFT Anchor 接口
 *
 * 提供统一的模板化 FFT 调用入口
 * 该接口作为 "Anchor" 算子直接调用底层 FFT 后端（如 cuFFT/vDSP），如果不通过
 * DSL 构图使用，不经过 Halide 调度 自动识别浮点精度（real32_t/real64_t）
 */

#ifndef PRISM_RUNTIME_FFT_H
#define PRISM_RUNTIME_FFT_H

#include <complex>
#include <memory>

#include "prism/types.h"

namespace prism::runtime {

/// @addtogroup runtime
/// @{

/**
 * @brief FFT 变换类型枚举
 */
enum class FftTransType : std::uint8_t {
  C2C = 1,  ///< 复数到复数 (Complex-to-Complex)
  R2C = 2,  ///< 实数到复数 (Real-to-Complex)，输出共轭对称
  C2R = 3   ///< 复数到实数 (Complex-to-Real)，输入需共轭对称
};

/**
 * @brief FFT 静态工具类
 *
 * 提供方便的静态方法调用 FFT，支持自动模板推导
 *
 * @tparam T 通常为 real32_t (float) 或 real64_t (double)
 *
 * @par 使用示例
 * @code
 * // 1. C2C 原地变换
 * std::vector<std::complex<float>> data(1024);
 * // ... 初始化 data ...
 * FFT::forward(data.data(), 1024); // 正变换
 * FFT::inverse(data.data(), 1024); // 逆变换（归一化）
 *
 * // 2. R2C / C2R 非原地变换
 * std::vector<float> realIn(1024);
 * std::vector<std::complex<float>> compOut(513); // N/2+1
 * FFT::forward(realIn.data(), compOut.data(), 1024);
 *
 * // 3. 异步/零拷贝执行
 * auto buf = FFT::acquireDeviceBuffer(ScalarType::C32, FftTransType::C2C,
 * 1024); buf.copy_to_device(); // 或直接映射 Native 指针
 * FFT::executeDeviceBuffer(buf, -1); // 异步提交
 * buf.copy_to_host(); // 同步并拷回
 * @endcode
 */
class FFT {
 public:
  // ========================================================================
  // C2C 变换（原位）
  // ========================================================================

  /**
   * @brief 执行 C2C 正变换
   * @tparam T real32_t 或 real64_t（自动推导）
   * @param data 输入数据指针（变换后被结果覆盖）
   * @param n 变换长度（必须为 2 的幂）
   */
  template <typename T>
  static void forward(std::complex<T>* data, int64_t n) {
    static_assert(IS_REAL_TYPE_V<T>, "T must be real32_t or real64_t");
    forwardImpl(data, n);
  }

  /**
   * @brief 执行 C2C 逆变换
   * @tparam T real32_t 或 real64_t
   * @param data 输入数据指针（变换后被结果覆盖）
   * @param n 变换长度（必须为 2 的幂）
   * @param normalize 变换后是否除以 N 进行归一化（默认为 true）
   */
  template <typename T>
  static void inverse(std::complex<T>* data, int64_t n, bool normalize = true) {
    static_assert(IS_REAL_TYPE_V<T>, "T must be real32_t or real64_t");
    inverseImpl(data, n, normalize);
  }

  // ========================================================================
  // R2C 变换（非原位）
  // ========================================================================

  /**
   * @brief 执行 R2C 正变换
   * @tparam T real32_t 或 real64_t
   * @param in 输入实数数据（长度 N）
   * @param out 输出复数数据（长度 N/2 + 1）
   * @param n 实数逻辑长度 N（必须为 2 的幂）
   *
   * @note 输入和输出不能指向同一块内存
   * @warning out 缓冲区必须足够大，至少容纳 N/2+1 个复数元素
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
   * @brief 执行 C2R 逆变换
   * @tparam T real32_t 或 real64_t
   * @param in 输入复数数据（长度 N/2 + 1）
   * @param out 输出实数数据（长度 N）
   * @param n 实数逻辑长度 N（必须为 2 的幂）
   * @param normalize 是否归一化（默认为 true）
   *
   * @note 输入和输出不能指向同一块内存
   */
  template <typename T>
  static void inverse(const std::complex<T>* in, T* out, int64_t n, bool normalize = true) {
    static_assert(IS_REAL_TYPE_V<T>, "T must be real32_t or real64_t");
    inverseC2rImpl(in, out, n, normalize);
  }

  // ========================================================================
  // 批量变换
  // ========================================================================

  /**
   * @brief 批量 C2C 变换
   * @tparam T real32_t 或 real64_t
   * @param data 数据数组指针（长度 = n * batch，原地操作）
   * @param n 单个 FFT 的长度
   * @param batch 批次数量
   * @param direction -1: 正变换; 1: 逆变换
   * @param normalize 若为逆变换，是否归一化
   */
  template <typename T>
  static void batch(std::complex<T>* data, int64_t n, int64_t batch, int direction,
                    bool normalize = true) {
    static_assert(IS_REAL_TYPE_V<T>, "T must be real32_t or real64_t");
    batchImpl(data, n, batch, direction, normalize);
  }

  // ========================================================================
  // 同步 & 能力查询
  // ========================================================================

  /**
   * @brief 强制全局设备同步
   *
   * 等待所有已提交的异/同步 GPU 任务完成，通常用于性能分析或调试
   */
  static void deviceSyncGlobal();

  /**
   * @brief 查询后端能力
   * @param precision 数据精度
   * @param type 变换类型
   * @return 若后端支持该组合则返回 true
   */
  static bool supports(ScalarType precision, FftTransType type);

  /**
   * @brief 设备侧缓冲区句柄
   *
   * 提供对 Device 内存的高级管理，支持异步提交、零拷贝（Unified
   * Memory）及手动同步 此对象具有 RAII 语义，销毁时自动释放设备资源
   */
  class DeviceBuffer {
   public:
    DeviceBuffer() = default;
    DeviceBuffer(DeviceBuffer&&) noexcept = default;
    DeviceBuffer& operator=(DeviceBuffer&&) noexcept = default;
    ~DeviceBuffer();

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    /** @brief 检查 Buffer 是否已分配且有效 */
    [[nodiscard]] bool valid() const;

    /**
     * @brief 提交 FFT 任务到设备流
     * @param direction -1(Forward) / 1(Inverse)，仅对 C2C 有效
     * @param wait 若为 true，则在函数返回前阻塞等待任务完成
     */
    void submit(int direction, bool wait = false);

    // API 与 Halide 对齐（采用 snake_case）
    // NOLINTBEGIN(readability-identifier-naming)

    /** @brief 等待设备任务完成（同步） */
    void device_sync();
    /** @brief 从 Device 拷贝数据回 Host（隐式同步） */
    void copy_to_host();
    /** @brief 从 Host 拷贝数据到 Device */
    void copy_to_device();

    /** @brief 检查 Host 端数据是否脏（需从 Device 更新） */
    [[nodiscard]] bool host_dirty() const;
    /** @brief 检查 Device 端数据是否脏（需从 Host 更新） */
    [[nodiscard]] bool device_dirty() const;
    /** @brief 标记 Host 端数据为脏 */
    void set_host_dirty(bool v = true);
    /** @brief 标记 Device 端数据为脏 */
    void set_device_dirty(bool v = true);
    // NOLINTEND(readability-identifier-naming)

   private:
    struct Impl;
    std::unique_ptr<Impl> impl_;

    explicit DeviceBuffer(std::unique_ptr<Impl> impl);
    friend class FFT;
  };

  /**
   * @brief 原生句柄对
   *
   * 用于 R2C/C2R 时的非原地操作，分别持有输入和输出的句柄
   */
  struct NativeHandlePair {
    void* input = nullptr;
    void* output = nullptr;
  };

  /**
   * @brief 查询是否支持 DeviceBuffer 路径
   */
  static bool supportsDeviceBuffer(ScalarType precision, FftTransType type);

  /**
   * @brief 申请设备 Buffer
   *
   * @param precision 标量类型（如 F32/C32）
   * @param type 变换类型
   * @param n 变换长度
   * @param batch 批次
   * @return DeviceBuffer 对象
   */
  static DeviceBuffer acquireDeviceBuffer(ScalarType precision, FftTransType type, int64_t n,
                                          int64_t batch = 1);

  /**
   * @brief 释放设备 Buffer
   * @note DeviceBuffer 析构时会自动释放，通常不需要手动调用
   */
  static void releaseDeviceBuffer(DeviceBuffer& buffer);

  /**
   * @brief 在 DeviceBuffer 上执行 FFT（静态辅助函数）
   * @see DeviceBuffer::submit
   */
  static void executeDeviceBuffer(DeviceBuffer& buffer, int direction, bool wait = false);

  /**
   * @brief 包装原生设备指针为 DeviceBuffer（不接管所有权）
   *
   * 允许用户传入已有的 CUDA/Metal 指针，复用 Prism 的 FFT 调度能力
   *
   * @param handle 指针或 NativeHandlePair
   * @param precision 类型
   * @param type 变换类型
   * @param n 长度
   * @param batch 批次
   */
  // NOLINTNEXTLINE(readability-identifier-naming)
  static DeviceBuffer wrap_native_handle(void* handle, ScalarType precision, FftTransType type,
                                         int64_t n, int64_t batch = 1);

  /**
   * @brief 解绑原生句柄（防止析构时释放）
   */
  // NOLINTNEXTLINE(readability-identifier-naming)
  static void detach_native_handle(DeviceBuffer& buffer);

  /**
   * @brief 获取底层的原生指​​针
   */
  // NOLINTNEXTLINE(readability-identifier-naming)
  static void* get_native_device_ptr(const DeviceBuffer& buffer);

  /**
   * @brief 获取描述后端设备的字符串
   */
  // NOLINTNEXTLINE(readability-identifier-naming)
  static std::string get_backend_device_info();

 private:
  // 实现分派（在 FFT.cpp 中定义）
  static void forwardImpl(std::complex<real32_t>* data, int64_t n);
  static void forwardImpl(std::complex<real64_t>* data, int64_t n);

  static void inverseImpl(std::complex<real32_t>* data, int64_t n, bool normalize);
  static void inverseImpl(std::complex<real64_t>* data, int64_t n, bool normalize);

  static void forwardR2cImpl(const real32_t* in, std::complex<real32_t>* out, int64_t n);
  static void forwardR2cImpl(const real64_t* in, std::complex<real64_t>* out, int64_t n);

  static void inverseC2rImpl(const std::complex<real32_t>* in, real32_t* out, int64_t n,
                             bool normalize);
  static void inverseC2rImpl(const std::complex<real64_t>* in, real64_t* out, int64_t n,
                             bool normalize);

  static void batchImpl(std::complex<real32_t>* data, int64_t n, int64_t batch, int direction,
                        bool normalize);
  static void batchImpl(std::complex<real64_t>* data, int64_t n, int64_t batch, int direction,
                        bool normalize);
};

/// @}

}  // namespace prism::runtime

#endif  // PRISM_RUNTIME_FFT_H
