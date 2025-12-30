/**
 * @file modem.h
 * @ingroup dsl
 * @brief 调制解调 Routines
 *
 * 定义常见混频、QAM 与 PSK 调制/解调节点，所有函数均返回惰性 @ref
 * prism::dsl::Signal
 */

#ifndef PRISM_DSL_MODEM_H
#define PRISM_DSL_MODEM_H

#include "prism/dsl/signal.h"
#include "prism/types.h"

namespace prism::dsl::modem {

/// @addtogroup dsl
/// @{

// ============================================================================
// 混频器
// ============================================================================

/**
 * @brief 混频器（上/下变频）
 * @param x 输入信号
 * @param freq 频移（Hz，正值为上变频）
 * @param sampleRate 采样率（Hz）
 * @return 变频后的信号，保持与输入相同的长度与标量类型
 */
Signal mixer(const Signal& x, real64_t freq, real64_t sampleRate);

// ============================================================================
// QAM 调制解调
// ============================================================================

/**
 * @brief QAM 符号映射
 *
 * 将符号索引映射到 I/Q 交织输出。
 *
 * @param bits 输入符号索引（0 ~ order-1）
 * @param order 阶数（4/16/64/256...必须为完全平方数）
 * @return I/Q 交织输出 [I0, Q0, I1, Q1, ...]，长度为输入的 2 倍
 */
Signal qamMap(const Signal& bits, int order);

/**
 * @brief QAM 符号解映射（硬判决）
 *
 * 将 I/Q 交织输入解映射回符号索引。
 *
 * @param symbols I/Q 交织输入
 * @param order 阶数（与映射保持一致）
 * @return 符号索引，长度为输入的一半
 */
Signal qamDemap(const Signal& symbols, int order);

// ============================================================================
// PSK 调制解调
// ============================================================================

/**
 * @brief PSK 符号映射
 *
 * 将符号索引映射到单位圆上的 I/Q 点。
 * 相位公式：θ = 2π * k / M + π/M（k = 0, 1, ..., M-1）
 *
 * @param bits 输入符号索引（0 ~ order-1）
 * @param order 阶数（2=BPSK, 4=QPSK, 8=8PSK）
 * @return I/Q 交织输出 [I0, Q0, I1, Q1, ...]，长度为输入的 2 倍
 *
 * @note
 * - BPSK (order=2): 相位 ±π/2，点在 (0, ±1)
 * - QPSK (order=4): 相位 π/4, 3π/4, 5π/4, 7π/4
 * - 8PSK (order=8): 8 个等间隔相位点
 */
Signal pskMap(const Signal& bits, int order);

/**
 * @brief PSK 符号解映射（硬判决）
 *
 * 根据接收的 I/Q 值计算相位并量化到最近的星座点。
 *
 * @param symbols I/Q 交织输入
 * @param order 阶数
 * @return 符号索引，长度为输入的一半
 */
Signal pskDemap(const Signal& symbols, int order);

/// @}

}  // namespace prism::dsl::modem

#endif  // PRISM_DSL_MODEM_H
