/**
 * @file ops.h
 * @ingroup dsl
 * @brief 基础数学与信号处理算子定义
 *
 * 定义 DSL 层的无状态算子工厂函数
 * 所有函数均以 **惰性 (Lazy)** 方式返回新的 @ref prism::dsl::Signal
 * 节点，不产生即时计算开销 实际的数值运算将在 `Executor` 运行时由 Halide
 * 调度执行
 */

#ifndef PRISM_DSL_OPS_H
#define PRISM_DSL_OPS_H

#include "prism/dsl/signal.h"
#include "prism/types.h"

namespace prism::dsl {

/// @addtogroup dsl
/// @{

/** @brief 逐元素加法：$output[i] = a[i] + b[i]$ */
Signal add(const Signal& a, const Signal& b);

/** @brief 逐元素减法：$output[i] = a[i] - b[i]$ */
Signal sub(const Signal& a, const Signal& b);

/** @brief 逐元素乘法：$output[i] = a[i] \times b[i]$ */
Signal mul(const Signal& a, const Signal& b);

/** @brief 逐元素除法：$output[i] = a[i] / b[i]$ */
Signal div(const Signal& a, const Signal& b);

/**
 * @brief 标量缩放 (Scalar Scaling)
 *
 * 将信号的每个元素乘以一个常数系数 $v$
 * $y[n] = x[n] \times v$
 *
 * @tparam T 标量类型（自动推导）
 * @param a 源信号
 * @param v 缩放因子（支持实数/复数，但精度必须与信号匹配，例如 F32 信号不能乘以
 * F64 系数）
 * @return 缩放后的新信号
 * @throw std::runtime_error 如果输入信号与缩放因子的精度不匹配
 */
template <typename T>
Signal scale(const Signal& a, T v) {
  auto node = std::make_shared<detail::Node>();
  node->kind = OpKind::SCALE;
  node->inputs = {a.node()};
  node->shape = a.shape();
  node->inputType = a.type();

  ScalarType const vType = getScalarType<T>();
  if (!isPrecisionMatch(a.type(), vType)) {
    throw std::runtime_error("Scale: precision mismatch");
  }
  node->outputType = promoteTypes(a.type(), vType);
  node->param = v;
  return Signal(node);
}

/** @brief 逐元素取绝对值：$y[n] = |x[n]|$（输入复数时计算模值，输出为实数） */
Signal abs(const Signal& a);

/** @brief 逐元素取负：$y[n] = -x[n]$ */
Signal negative(const Signal& a);

/** @brief 逐元素取复共轭：$y[n] = x[n]^*$ */
Signal conj(const Signal& a);

/**
 * @brief 一维离散卷积 (Discrete Convolution)
 *
 * 计算公式：
 * $(x * h)[n] = \sum_{m=0}^{K-1} x[n-m] \cdot h[m]$
 *
 * 采用 Full 模式，输出长度为 $L_x + L_h - 1$
 *
 * @param a 输入信号 $x$
 * @param kernel 卷积核 $h$（通常较短）
 * @return 卷积结果信号
 */
Signal convolve(const Signal& a, const Signal& kernel);

/**
 * @brief 克罗内克积 (Kronecker Product) / 扩频
 *
 * 将输入序列 $a$ 的每个样本扩展为整个序列 $b$ 的加权副本
 * 常用于 DSSS (Direct Sequence Spread Spectrum) 系统中生成扩频信号
 *
 * 例：$a = [s_0, s_1], \quad b = [c_0, c_1]$
 * 结果：$[s_0 c_0, \ s_0 c_1, \ s_1 c_0, \ s_1 c_1]$
 *
 * @param a 数据符号序列 (Symbol Sequence)
 * @param b 扩频码/伪随机码 (Spreading Code)
 * @return 扩频后的高速流，长度 = $Length(a) \times Length(b)$
 */
Signal kron(const Signal& a, const Signal& b);

/**
 * @brief 复数打包 (Complex Packing)
 *
 * 将两路独立的实数信号合并为一路复数信号
 * $y[n] = i[n] + j \cdot q[n]$
 *
 * @param i 同相分量 I (In-phase)，作为实部
 * @param q 正交分量 Q (Quadrature)，作为虚部
 * @return 复数信号 (Complex64/Complex32)，长度需与输入一致
 */
Signal complexPack(const Signal& i, const Signal& q);

/**
 * @brief 提取实部 (Real Component)
 *
 * $y[n] = \text{Re}(x[n])$
 *
 * @param iq 复数输入信号
 * @return 实数信号 (Real64/Real32)
 */
Signal real(const Signal& iq);

/**
 * @brief 提取虚部 (Imag Component)
 *
 * $y[n] = \text{Im}(x[n])$
 *
 * @param iq 复数输入信号
 * @return 实数信号 (Real64/Real32)
 */
Signal imag(const Signal& iq);

/**
 * @brief 零值上采样 (Upsample by Zero Stuffing)
 *
 * 在相邻样本之间插入 $L-1$ 个零值
 *
 * @param a 输入信号
 * @param factor 上采样倍数 $L$ (Integer $\ge 1$)
 * @param offset 初始采样相移 (0 $\le$ offset < factor)
 * @return 扩展后的信号，长度 = $Length(a) \times L$
 */
Signal upsample(const Signal& a, int factor, int offset = 0);

/**
 * @brief 抽取/下采样 (Downsample / Decimation)
 *
 * 每隔 $M$ 个样本保留一个值
 * $y[n] = x[n \cdot M + \text{offset}]$
 *
 * @param a 输入信号
 * @param factor 抽取倍数 $M$ (Integer $\ge 1$)
 * @param offset 初始采样相移 (0 $\le$ offset < factor)
 * @return 抽取后的信号，长度 $\approx Length(a) / M$
 */
Signal downsample(const Signal& a, int factor, int offset = 0);

/// 语法糖：支持 `Signal` 直接使用 C++ 标准算术运算符重载
Signal operator+(const Signal& a, const Signal& b);
Signal operator-(const Signal& a, const Signal& b);
Signal operator*(const Signal& a, const Signal& b);
Signal operator/(const Signal& a, const Signal& b);
Signal operator-(const Signal& a);

/// @}

}  // namespace prism::dsl

#endif  // PRISM_DSL_OPS_H
