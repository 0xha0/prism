/**
 * @file filter.h
 * @ingroup dsl
 * @brief 滤波器构造函数
 *
 * 提供各类型滤波器的 DSL 构造接口，包括 FIR、滑动平均与中值滤波
 * 返回惰性 @ref prism::dsl::Signal 节点，实际计算（如 FFT 卷积或时域卷积）由
 * Runtime 优化决定
 */

#ifndef PRISM_DSL_FILTER_H
#define PRISM_DSL_FILTER_H

#include "prism/dsl/signal.h"

namespace prism::dsl::filter {

/// @addtogroup dsl
/// @{

/**
 * @brief 构造通用 FIR 滤波器 (Finite Impulse Response)
 *
 * 计算公式：$y[n] = \sum_{k=0}^{N-1} h[k] \cdot x[n-k]$
 *
 * @tparam T 系数数据类型，支持 real32/64 或 complex32/64
 * @param x 输入信号
 * @param taps 滤波器系数 $h$ (Impulse Response)
 * @param mode 边界处理模式，决定如何填充信号边界外的样本 (e.g. Zero Pad,
 * Reflect)
 * @return 滤波后的信号，长度通常与输入一致（取决于实现细节，通常保持 input
 * length）
 */
template <typename T>
Signal fir(const Signal& x, const std::vector<T>& taps, BndryMode mode = BndryMode::ZERO);

/**
 * @brief 滑动平均滤波器 (Moving Average / Boxcar)
 *
 * 一种简单的低通滤波器，计算窗口内样本的算术平均值
 * 等效于系数全为 $1/N$ 的 FIR 滤波器
 *
 * $$ y[n] = \frac{1}{W} \sum_{k=0}^{W-1} x[n-k] $$
 *
 * @param x 输入信号
 * @param window 窗口长度 $W$（样本数），值越大越平滑，但群延迟 (Group Delay)
 * 越大
 * @param mode 边界处理策略
 * @return 平滑后的信号
 */
Signal movingAverage(const Signal& x, int window, BndryMode mode = BndryMode::ZERO);

/**
 * @brief 中值滤波器 (Median Filter)
 *
 * 非线性滤波器，取窗口内样本排序后的中位数作为输出
 *
 * 特点：
 * - 极佳的去 **椒盐噪声 (Salt & Pepper noise)** 能力
 * - 相比线性滤波，能更好地 **保留信号的边缘特征 (Edge Preserving)**
 *
 * @param x 输入信号
 * @param window 窗口长度 $W$（必须为正奇数，e.g. 3, 5, 7...）
 * @param mode 边界处理策略
 * @return 去噪后的信号
 *
 * @warning 性能提示：相比 FIR，中值滤波无法利用 FFT 加速且涉及排序操作 ($O(N
 * \cdot W \log W)$)， 大窗口下的计算成本极高，请谨慎设置窗口大小
 */
Signal median(const Signal& x, int window, BndryMode mode = BndryMode::ZERO);

/// @}

}  // namespace prism::dsl::filter

#endif  // PRISM_DSL_FILTER_H
