/**
 * @file modem.h
 * @ingroup dsl
 * @brief 调制解调 (Modem) 算法库
 *
 * 提供常用的数字通信处理模块，包括：
 * - 频率搬移 (Mixer/NCO)
 * - 映射与解映射 (Mapper/Demapper): QAM, PSK
 *
 * 所有函数均返回惰性 @ref prism::dsl::Signal 对象
 */

#ifndef PRISM_DSL_MODEM_H
#define PRISM_DSL_MODEM_H

#include "prism/dsl/signal.h"
#include "prism/types.h"

namespace prism::dsl::modem {

/// @addtogroup dsl
/// @{

// ============================================================================
// 混频器 (Mixer / Frequency Shifter)
// ============================================================================

/**
 * @brief 数字混频器 / 数控振荡器 (NCO)
 *
 * 将输入信号乘以复指数信号，实现频谱搬移
 * 核心公式：
 * $$ y[n] = x[n] \times e^{-j \frac{2\pi f}{F_s} n} $$
 *
 * @note 负号约定：PRISM 遵循工程惯例，在该函数中正频率 $f > 0$ 对应下变频
 * (Down-conversion) 或
 *       频谱左移，如果用于上变频需传入负频率，具体取决于后续滤波器的通带位置
 *
 * @param x 输入信号
 * @param freq 频移量 $f$ (Hz)
 * @param sampleRate 系统采样率 $F_s$ (Hz)，用于归一化角频率
 * @return 变频后的信号，数据类型通常提升为复数
 */
Signal mixer(const Signal& x, real64_t freq, real64_t sampleRate);

// ============================================================================
// QAM 调制解调
// ============================================================================

/**
 * @brief 正交幅度调制 (QAM) 映射
 *
 * 将输入的比特/符号索引映射到复平面的星座点上
 * 支持矩形星座图 (Rectangular QAM)，采用格雷码 (Gray Code) 映射以最小化误比特率
 *
 * @param bits 输入符号索引流，每个元素的取值必须在 $[0, M-1]$ 之间
 * @param order 调制阶数 $M$，必须为完全平方数 (e.g., 4, 16, 64, 256)
 * @return 输出为 Complex 信号 (I+jQ)，长度与输入一致
 */
Signal qamMap(const Signal& bits, int order);

/**
 * @brief QAM 符号解映射 (Soft/Hard Demapper)
 *
 * QAM 映射的逆过程
 * 当前实现为 **硬判决 (Hard Decision)**：根据接收符号的 I/Q 坐标，
 * 寻找欧氏距离最近的标准星座点，并输出该点的符号索引
 *
 * @param symbols Complex 接收信号序列
 * @param order 调制阶数 $M$ (需与发送端一致)
 * @return 解调出的符号索引流 (Integer)，长度与 symbols 一致
 */
Signal qamDemap(const Signal& symbols, int order);

/**
 * @brief QAM 符号解映射 (I/Q 分离输入版)
 *
 * @param i 同相分量信号
 * @param q 正交分量信号
 * @param order 调制阶数 $M$
 * @return 符号索引流
 */
Signal qamDemap(const Signal& i, const Signal& q, int order);

// ============================================================================
// PSK 调制解调
// ============================================================================

/**
 * @brief 相移键控 (PSK) 映射
 *
 * 将符号索引映射到单位圆上的均匀分布点
 * 映射公式：
 * $$ \theta_k = \frac{2\pi k}{M} + \phi_0 $$
 * $$ s_k = e^{j \theta_k} $$
 * 其中 $M$ 为阶数，$k$ 为输入符号
 *
 * 典型星座：
 * - BPSK ($M=2$): 相位 $\{0, \pi\}$ -> $(1, 0), (-1, 0)$
 * - QPSK ($M=4$): 相位 $\{\pi/4, 3\pi/4, 5\pi/4, 7\pi/4\}$
 * - 8PSK ($M=8$): 相位每 $45^\circ$ 分布
 *
 * @param bits 输入符号索引 ($0 \sim M-1$)
 * @param order 调制阶数 $M$ (e.g. 2, 4, 8)
 * @return Complex 输出信号，模值为 1
 */
Signal pskMap(const Signal& bits, int order);

/**
 * @brief PSK 符号解映射 (Hard Decision)
 *
 * 计算接收点的相位角 `atan2(Q, I)`，并量化到最近的理想相位点，判决出符号索引
 *
 * @param symbols Complex 接收信号
 * @param order 调制阶数 $M$
 * @return 符号索引流
 */
Signal pskDemap(const Signal& symbols, int order);

/**
 * @brief PSK 符号解映射 (I/Q 分离输入版)
 *
 * @param i 同相分量信号
 * @param q 正交分量信号
 * @param order 调制阶数 $M$
 * @return 符号索引流
 */
Signal pskDemap(const Signal& i, const Signal& q, int order);

/// @}

}  // namespace prism::dsl::modem

#endif  // PRISM_DSL_MODEM_H
