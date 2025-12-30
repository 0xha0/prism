/**
 * @file ops.h
 * @ingroup dsl
 * @brief 基础算子定义
 *
 * 定义 DSL 层的无状态算子工厂函数，均以惰性方式返回新的 @ref prism::dsl::Signal
 * 节点，不产生即时计算开销。
 */

#ifndef PRISM_DSL_OPS_H
#define PRISM_DSL_OPS_H

#include "prism/dsl/signal.h"
#include "prism/types.h"

namespace prism::dsl {

/// @addtogroup dsl
/// @{

/** @brief 逐元素加法 */
Signal add(const Signal& a, const Signal& b);
/** @brief 逐元素减法 */
Signal sub(const Signal& a, const Signal& b);
/** @brief 逐元素乘法 */
Signal mul(const Signal& a, const Signal& b);
/** @brief 逐元素除法 */
Signal div(const Signal& a, const Signal& b);
/**
 * @brief 缩放运算
 * @param a 输入信号
 * @param v 缩放系数
 */
Signal scale(const Signal& a, real64_t v);
/** @brief 取绝对值（模值） */
Signal abs(const Signal& a);

/**
 * @brief 卷积运算
 * @param a 输入信号
 * @param kernel 卷积核
 * @return 输出长度为 a.length + kernel.length - 1
 */
Signal convolve(const Signal& a, const Signal& kernel);

/**
 * @brief 克罗内克积（扩频用）
 * @param a 输入信号
 * @param b 扩频码
 * @return 输出长度 = a.length * b.length
 */
Signal kron(const Signal& a, const Signal& b);

// ============================================================================
// I/Q 交织辅助
// ============================================================================

/**
 * @brief 将两路实数 I/Q 合并为交织序列
 *
 * 输出格式：[I0, Q0, I1, Q1, ...]
 *
 * @param i I 分量序列
 * @param q Q 分量序列
 * @return 交织后的 I/Q 序列（长度为输入的 2 倍）
 */
Signal iqPack(const Signal& i, const Signal& q);

/**
 * @brief 从交织 I/Q 序列提取 I 分量
 * @param iq 交织序列 [I0, Q0, I1, Q1, ...]
 * @return I 分量序列（长度为输入的一半）
 */
Signal iqI(const Signal& iq);

/**
 * @brief 从交织 I/Q 序列提取 Q 分量
 * @param iq 交织序列 [I0, Q0, I1, Q1, ...]
 * @return Q 分量序列（长度为输入的一半）
 */
Signal iqQ(const Signal& iq);

/**
 * @brief 上采样（插零）
 * @param a 输入信号
 * @param factor 上采样因子（>0）
 * @return 输出长度 = a.length * factor
 */
Signal upsample(const Signal& a, int factor);

/**
 * @brief 下采样（抽取）
 * @param a 输入信号
 * @param factor 抽取因子（>0）
 * @param offset 起始偏移（从 offset 位置开始每隔 factor 取样）
 * @return 输出长度约为 ceil((a.length - offset) / factor)
 */
Signal downsample(const Signal& a, int factor, int offset = 0);

/// 语法糖：支持 `Signal` 直接使用算术运算符
Signal operator+(const Signal& a, const Signal& b);
Signal operator-(const Signal& a, const Signal& b);
Signal operator*(const Signal& a, const Signal& b);
Signal operator/(const Signal& a, const Signal& b);

/// @}

}  // namespace prism::dsl

#endif  // PRISM_DSL_OPS_H
