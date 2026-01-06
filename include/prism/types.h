/**
 * @file types.h
 * @ingroup core
 * @brief PRISM 基础类型定义与形状描述
 *
 * 本文件集中定义了 PRISM 在 DSL、Runtime 及 Backend 之间共享的
 * 标量类型枚举、形状结构体以及常用类型别名
 * 这些定义构成了 PRISM 类型系统的基础，用于支持跨模块的类型安全与推导
 */

#ifndef PRISM_TYPES_H
#define PRISM_TYPES_H

#include <complex>
#include <cstdint>
#include <type_traits>
#include <variant>
#include <vector>

namespace prism {

/// @addtogroup core
/// @{

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define M_PI_VAL 3.14159265358979323846264338327950288  ///< 内置圆周率常量 $\pi$，避免平台依赖

/**
 * @brief 标量类型枚举
 *
 * 定义 DSL 计算图中 Tensor 元素的具体数据类型
 * 正确的类型选择对性能至关重要：
 * - GPU 后端通常对 F32/F16 有高度优化的指令路径
 * - 频域处理（FFT/Filter）大量依赖复数类型 (C32/C64)
 *
 * @note 本枚举采用位掩码 (Bitmask) 编码，便于通过位运算快速查表或判断属性
 */
/**
 * @brief 标量类型编码细节 (Bitmask Encoding)
 *
 * 编码方案 (8 bits):
 * - Bits 0-3: 字节大小 (Size in bytes)
 * - Bit  4:   复数标志 (1=Complex, 0=Real)
 * - Bit  5:   浮点标志 (1=Float, 0=Integer)
 * - Bit  6:   符号标志 (1=Signed, 0=Unsigned)
 */
namespace detail {
// NOLINTNEXTLINE(cppcoreguidelines-use-enum-class)
enum TypeMask : std::uint8_t {
  SIZE_MASK = 0x0F,     ///< 掩码：获取类型大小（字节）
  COMPLEX_FLAG = 0x10,  ///< 标志：是否为复数
  FLOAT_FLAG = 0x20,    ///< 标志：是否为浮点数
  SIGNED_FLAG = 0x40    ///< 标志：是否有符号
};
}  // namespace detail

enum class ScalarType : std::uint8_t {
  // 浮点类型 (Float | Size)
  F16 = detail::FLOAT_FLAG | 2,  ///< 16-bit 浮点数 (half)
  F32 = detail::FLOAT_FLAG | 4,  ///< 32-bit 浮点数 (float)
  F64 = detail::FLOAT_FLAG | 8,  ///< 64-bit 浮点数 (double)

  // 复数类型 (Float | Complex | Size)
  C16 = detail::FLOAT_FLAG | detail::COMPLEX_FLAG | 2,  ///< 16-bit 复数 (complex half)
  C32 = detail::FLOAT_FLAG | detail::COMPLEX_FLAG | 4,  ///< 32-bit 复数 (complex float)
  C64 = detail::FLOAT_FLAG | detail::COMPLEX_FLAG | 8,  ///< 64-bit 复数 (complex double)

  // 预留整数类型 (Reserved)
  // I8  = detail::SIGNED_FLAG | 1,
  // U8  = 1
};

/**
 * @brief 边界扩展模式 (Boundary Condition)
 *
 * 指定卷积、滤波或滑动窗口操作在信号边界处的填充策略
 * 假设信号定义在区间 $[0, N-1]$，当访问索引 $i < 0$ 或 $i \ge N$ 时：
 */
enum class BndryMode : std::uint8_t {
  ZERO,    ///< 零填充 (Zero Padding)：越界值视为 0，常用于 FIR 滤波
  CLAMP,   ///< 边缘截断 (Clamp to Edge)：越界取最近的边界值 $x[0]$ 或 $x[N-1]$
  REFLECT  ///< 镜像反射 (Reflect 101)：以边界为轴进行镜像（不重复边界值）
};

/**
 * @brief 信号形状描述 (Shape)
 *
 * 描述 Tensor 在各维度的尺寸
 * PRISM 主要处理 1D 信号流，但支持 Batch 和 Channel 维度用于并行处理
 * 数据布局通常为 (Batch, Channel, Length)
 */
struct Shape {
  size_t length = 0;    ///< 信号长度（时域/频域样本数）
  size_t channels = 1;  ///< 通道数 (Channels)，例如 MIMO 系统中的天线数
  size_t batch = 1;     ///< 批次大小 (Batch Size)，用于数据并行
};

/** @name 常用类型别名 */
///@{
using real32_t = float;                      ///< 单精度实数
using real64_t = double;                     ///< 双精度实数
using complex32_t = std::complex<real32_t>;  ///< 单精度复数
using complex64_t = std::complex<real64_t>;  ///< 双精度复数
///@}

/**
 * @brief 算子参数容器 (OpParam)
 *
 * 通用变体类型，用于存储算子的配置参数
 * 支持标量（整型/浮点/复数）及其对应的 std::vector 形式
 */
using OpParam =
    std::variant<int32_t, int64_t, real32_t, real64_t, complex32_t, complex64_t,
                 std::vector<int32_t>, std::vector<int64_t>, std::vector<real32_t>,
                 std::vector<real64_t>, std::vector<complex32_t>, std::vector<complex64_t>>;

/// 类型约束：仅允许浮点实数类型 (float/double)
template <typename T>
inline constexpr bool IS_REAL_TYPE_V = std::is_same_v<T, real32_t> || std::is_same_v<T, real64_t>;

namespace detail {
template <typename T>
struct IsComplex : std::false_type {};
template <typename T>
struct IsComplex<std::complex<T>> : std::true_type {};

template <typename T, bool IsComplex>
struct ToHalideTypeImpl {
  using Type = T;
};

template <typename T>
struct ToHalideTypeImpl<T, true> {
  using Type = typename T::value_type;
};
}  // namespace detail

/// 类型约束：判断类型是否为 std::complex 特化
template <typename T>
inline constexpr bool IS_COMPLEX_V = detail::IsComplex<T>::value;

/**
 * @brief 类型映射 Trait：C++ 类型 -> Halide 基础类型
 *
 * 将 `std::complex<T>` 映射为基础类型 `T`，非复数类型保持不变
 * 用于 Halide Type 构造时的基础类型提取
 */
template <typename T>
struct ToHalideType {
  using Decayed = std::decay_t<T>;
  using Type = typename detail::ToHalideTypeImpl<Decayed, IS_COMPLEX_V<Decayed>>::Type;
};

// ============================================================================
// ScalarType 辅助函数 (Bitwise Implementation)
// ============================================================================

/**
 * @brief 判断是否为复数类型
 * @param type 待检查的标量类型
 * @return true 如果是复数类型 (C16/C32/C64)
 */
constexpr bool isComplexType(ScalarType type) {
  return (static_cast<std::uint8_t>(type) & detail::COMPLEX_FLAG) != 0;
}

/**
 * @brief 判断是否为实数类型
 * @param type 待检查的标量类型
 * @return true 如果是实数类型 (F16/F32/F64 等)
 */
constexpr bool isRealType(ScalarType type) { return !isComplexType(type); }

/**
 * @brief 判断是否为浮点类型
 * @param type 待检查的标量类型
 * @return true 如果是浮点数 (Float/Complex)
 */
constexpr bool isFloatType(ScalarType type) {
  return (static_cast<std::uint8_t>(type) & detail::FLOAT_FLAG) != 0;
}

/**
 * @brief 类型提升：将实数类型转为同精度的复数类型
 * @param type 输入类型 (e.g., F32)
 * @return 对应的复数类型 (e.g., C32)，如果已是复数则保持不变
 */
constexpr ScalarType toComplexType(ScalarType type) {
  return static_cast<ScalarType>(static_cast<std::uint8_t>(type) | detail::COMPLEX_FLAG);
}

/**
 * @brief 类型降级：将复数类型转为同精度的实数类型
 * @param type 输入类型 (e.g., C32)
 * @return 对应的实数类型 (e.g., F32)，如果已是实数则保持不变
 */
constexpr ScalarType toRealType(ScalarType type) {
  return static_cast<ScalarType>(static_cast<std::uint8_t>(type) & ~detail::COMPLEX_FLAG);
}

/**
 * @brief 获取类型大小（字节数）
 * @param type 输入类型
 * @return 类型占用的字节数 (e.g., F32/C32 均为 4) -> 注意这里指的是 component
 * size
 * @note 复数类型返回其实部/虚部单个分量的大小
 */
constexpr int getComponentSize(ScalarType type) {
  return static_cast<std::uint8_t>(type) & detail::SIZE_MASK;
}

/**
 * @brief 检查两个类型是否具有匹配的精度与基础格式
 * @return true 如果两者的大小与浮点属性一致
 */
constexpr bool isPrecisionMatch(ScalarType a, ScalarType b) {
  return getComponentSize(a) == getComponentSize(b) && isFloatType(a) == isFloatType(b);
}

/**
 * @brief 二元运算类型推导
 *
 * 规则：
 * 1. 如果任一操作数为复数，结果为复数
 * 2. 否则结果为实数
 * @note 目前只处理复数提升，假设输入精度已对齐
 */
constexpr ScalarType promoteTypes(ScalarType a, ScalarType b) {
  if (isComplexType(a) || isComplexType(b)) {
    return toComplexType(a);
  }
  return a;
}

/**
 * @brief 运行时类型获取辅助
 * 内名空间 detail 封装具体的类型提取逻辑
 */
namespace detail {
template <typename T>
struct GetValueType {
  using Type = std::decay_t<T>;
};
template <typename T, typename A>
struct GetValueType<std::vector<T, A>> {
  using Type = std::decay_t<T>;
};

template <typename>
inline constexpr bool DEPENDENT_FALSE_V = false;
}  // namespace detail

/**
 * @brief 编译期类型 -> 运行时 ScalarType 枚举映射
 *
 * 根据 C++ 模板参数 T 返回对应的 ScalarType枚举值
 * 支持 float, double, complex<float>, complex<double> 及其 vector 容器
 * @tparam T C++ 数据类型
 * @return 对应的 ScalarType 枚举值
 */
template <typename T>
constexpr ScalarType getScalarType() {
  using VT = typename detail::GetValueType<std::decay_t<T>>::Type;
  if constexpr (std::is_same_v<VT, real32_t>) {
    return ScalarType::F32;
  } else if constexpr (std::is_same_v<VT, real64_t>) {
    return ScalarType::F64;
  } else if constexpr (std::is_same_v<VT, complex32_t>) {
    return ScalarType::C32;
  } else if constexpr (std::is_same_v<VT, complex64_t>) {
    return ScalarType::C64;
  } else {
    static_assert(detail::DEPENDENT_FALSE_V<VT>, "不支持的类型，无法推导 ScalarType");
    return ScalarType::F32;
  }
}

/// @}

}  // namespace prism

#endif  // PRISM_TYPES_H
