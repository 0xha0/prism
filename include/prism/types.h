/**
 * @file types.h
 * @ingroup core
 * @brief PRISM 基础类型与形状描述
 *
 * 该文件集中定义了 PRISM 在 DSL、Runtime、Backend 之间共享的
 * 标量类型枚举、形状结构体以及常用别名，便于跨模块复用。
 */

#ifndef PRISM_TYPES_H
#define PRISM_TYPES_H

#include <complex>
#include <cstdint>
#include <type_traits>

namespace prism {

/// @addtogroup core
/// @{

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define M_PI_VAL \
  3.14159265358979323846264338327950288  ///< 内置圆周率常量，便于避免系统宏差异

/**
 * @brief 标量类型枚举
 *
 * 描述 DSL 计算图中的标量精度与是否为复数，为后端选择与调度提供依据。
 */
enum class ScalarType : std::uint8_t {
  F16,  ///< 半精度实数（预留，部分后端可能不支持）
  F32,  ///< 单精度实数
  F64,  ///< 双精度实数
  C16,  ///< 半精度复数（预留）
  C32,  ///< 单精度复数
  C64   ///< 双精度复数
};

/**
 * @brief 边界处理策略
 *
 * 用于滤波/滑窗算子在访问越界样本时的取值方式。
 */
enum class BndryMode : std::uint8_t {
  ZERO,    ///< 越界样本按 0 处理
  CLAMP,   ///< 复制边界样本（越界索引 clamp 到 [0, len-1]）
  REFLECT  ///< 对称反射（镜像边界）
};

/**
 * @brief 信号形状描述
 *
 * 目前聚焦 1D 信号处理，同时预留批次与通道两个维度，便于后续扩展到
 * 批量推理或多通道处理的场景。
 */
struct Shape {
  int64_t length = 0;    ///< 样本长度（1D 维度）
  int64_t channels = 1;  ///< 通道数，默认为单通道
  int64_t batch = 1;     ///< 批次数，便于后续批处理优化
};

/** @name 标量与复数类型别名 */
///@{
using real32_t = float;
using real64_t = double;
using complex32_t = std::complex<real32_t>;
using complex64_t = std::complex<real64_t>;
///@}

/// 类型约束：仅允许 real32_t 或 real64_t
template <typename T>
inline constexpr bool IS_REAL_TYPE_V =
    std::is_same_v<T, real32_t> || std::is_same_v<T, real64_t>;

namespace detail {
template <typename T>
struct IsComplex : std::false_type {};
template <typename T>
struct IsComplex<std::complex<T>> : std::true_type {};
}  // namespace detail

/// 类型约束：判断是否为 std::complex
template <typename T>
inline constexpr bool IS_COMPLEX_V = detail::IsComplex<T>::value;

/**
 * @brief 辅助 trait：将 C++ 类型映射为 Halide Buffer 的元素类型
 *
 * complex32_t -> real32_t
 * complex64_t -> real64_t
 * T -> T
 */
template <typename T>
struct ToHalideType {
  using Type = T;
};

template <>
struct ToHalideType<complex32_t> {
  using Type = real32_t;
};

template <>
struct ToHalideType<complex64_t> {
  using Type = real64_t;
};

/// @}

}  // namespace prism

#endif  // PRISM_TYPES_H
