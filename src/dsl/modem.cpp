/**
 * @file modem.cpp
 * @ingroup dsl
 * @brief 调制解调算子实现
 */

#include "prism/dsl/modem.h"

#include <memory>

#include "prism/dsl/signal.h"
#include "prism/types.h"

namespace prism::dsl::modem {

Signal mixer(const Signal& x, real64_t freq, real64_t sampleRate) {
  auto node = std::make_shared<detail::Node>();
  node->kind = OpKind::MIXER;
  node->inputs = {x.node()};
  node->shape = x.shape();
  node->type = x.type();
  node->freq = freq;
  node->sampleRate = sampleRate;
  return Signal(node);
}

/**
 * @brief QAM 符号映射
 *
 * 输入：符号索引序列（长度 N）
 * 输出：I/Q 交织序列（长度 2N）：[I0, Q0, I1, Q1, ...]
 *
 * @note 这是一个 breaking change，输出长度变为输入的两倍
 * @return I/Q 交织输出，长度等于输入长度的两倍
 */
Signal qamMap(const Signal& bits, int order) {
  auto node = std::make_shared<detail::Node>();
  node->kind = OpKind::QAM_MAP;
  node->inputs = {bits.node()};
  // 输出长度是输入的两倍（I/Q 交织）
  Shape outShape = bits.shape();
  outShape.length *= 2;
  node->shape = outShape;
  node->type = bits.type();
  node->modOrder = order;
  return Signal(node);
}

/**
 * @brief QAM 符号解映射
 *
 * 输入：I/Q 交织序列（长度 2N）：[I0, Q0, I1, Q1, ...]
 * 输出：符号索引序列（长度 N）
 *
 * @return 符号索引信号
 */
Signal qamDemap(const Signal& symbols, int order) {
  auto node = std::make_shared<detail::Node>();
  node->kind = OpKind::QAM_DEMAP;
  node->inputs = {symbols.node()};
  // 输出长度是输入的一半（从 I/Q 对恢复符号）
  Shape outShape = symbols.shape();
  outShape.length /= 2;
  node->shape = outShape;
  node->type = symbols.type();
  node->modOrder = order;
  return Signal(node);
}

// ============================================================================
// PSK 调制解调
// ============================================================================

/**
 * @brief PSK 符号映射
 *
 * 将符号索引映射到单位圆上的 I/Q 点。
 * 相位公式：θ = 2π * k / M + π/M
 *
 * 输入：符号索引序列（长度 N）
 * 输出：I/Q 交织序列（长度 2N）：[I0, Q0, I1, Q1, ...]
 *
 * @return I/Q 交织输出
 */
Signal pskMap(const Signal& bits, int order) {
  auto node = std::make_shared<detail::Node>();
  node->kind = OpKind::PSK_MAP;
  node->inputs = {bits.node()};
  // 输出长度是输入的两倍（I/Q 交织）
  Shape outShape = bits.shape();
  outShape.length *= 2;
  node->shape = outShape;
  node->type = bits.type();
  node->modOrder = order;
  return Signal(node);
}

/**
 * @brief PSK 符号解映射
 *
 * 根据 I/Q 值计算相位并量化到最近的星座点。
 *
 * 输入：I/Q 交织序列（长度 2N）：[I0, Q0, I1, Q1, ...]
 * 输出：符号索引序列（长度 N）
 *
 * @return 解映射后的符号索引
 */
Signal pskDemap(const Signal& symbols, int order) {
  auto node = std::make_shared<detail::Node>();
  node->kind = OpKind::PSK_DEMAP;
  node->inputs = {symbols.node()};
  // 输出长度是输入的一半
  Shape outShape = symbols.shape();
  outShape.length /= 2;
  node->shape = outShape;
  node->type = symbols.type();
  node->modOrder = order;
  return Signal(node);
}

}  // namespace prism::dsl::modem
