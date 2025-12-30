/**
 * @file filter.h
 * @ingroup dsl
 * @brief 滤波 Routines
 *
 * 提供 FIR/IIR 滤波器的 DSL 构造函数，系数由用户提供。
 * 返回惰性 @ref prism::dsl::Signal 节点，实际计算由 Runtime 完成。
 */

#ifndef PRISM_DSL_FILTER_H
#define PRISM_DSL_FILTER_H

#include "prism/dsl/signal.h"

namespace prism::dsl::filter {

/// @addtogroup dsl
/// @{

// ============================================================================
// 核心滤波器（系数由用户提供）
// ============================================================================

/**
 * @brief FIR 滤波器
 * @param x 输入信号
 * @param taps 滤波器系数（b 系数）
 * @param mode 边界处理策略
 * @return 输出长度为 `x.length + taps.size() - 1`
 * @note 支持单精度与双精度两种系数形态，内部自动匹配 Signal 类型。
 */
Signal fir(const Signal& x, const std::vector<real32_t>& taps,
           BndryMode mode = BndryMode::ZERO);
Signal fir(const Signal& x, const std::vector<real64_t>& taps,
           BndryMode mode = BndryMode::ZERO);

/**
 * @brief IIR 滤波器 (Direct Form II Transposed)
 * @param x 输入信号
 * @param b 前向系数（分子）
 * @param a 反馈系数（分母），a[0] 必须为 1
 * @param mode 边界处理策略
 * @return IIR 滤波后的信号节点
 * @note 系数长度应保持一致；`a` 的首元素用于归一化，调用方需保证有效。
 */
Signal iir(const Signal& x, const std::vector<real32_t>& b,
           const std::vector<real32_t>& a, BndryMode mode = BndryMode::ZERO);
Signal iir(const Signal& x, const std::vector<real64_t>& b,
           const std::vector<real64_t>& a, BndryMode mode = BndryMode::ZERO);

// ============================================================================
// 滑动窗口滤波器
// ============================================================================

/**
 * @brief 移动平均滤波器
 * @param x 输入信号
 * @param window 窗口长度（样本数）
 * @param mode 边界处理策略
 * @return 平滑后的信号
 */
Signal movingAverage(const Signal& x, int window,
                     BndryMode mode = BndryMode::ZERO);

/**
 * @brief 中值滤波器
 * @param x 输入信号
 * @param window 窗口长度（必须为正奇数，当前支持 3/5/7）
 * @param mode 边界处理策略
 * @return 去除尖噪点后的信号
 * @note 其他窗口长度会在构图阶段抛出异常
 */
Signal median(const Signal& x, int window, BndryMode mode = BndryMode::ZERO);

/// @}

}  // namespace prism::dsl::filter

#endif  // PRISM_DSL_FILTER_H
