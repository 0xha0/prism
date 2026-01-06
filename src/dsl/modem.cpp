/**
 * @file modem.cpp
 * @ingroup dsl
 * @brief 调制解调算子实现
 */

#include "prism/dsl/modem.h"

#include <memory>
#include <stdexcept>
#include <vector>

#include "prism/dsl/signal.h"
#include "prism/types.h"

namespace prism::dsl::modem {

/**
 * @brief 数字混频器 implementation
 *
 * 构建 OpKind::MIXER 节点
 * 频率参数 stored in `node->freq` (Hz), 采样率 in `node->sampleRate` (Hz).
 * Runtime 后端将据此生成正弦波表或计算实时相位
 */
Signal mixer(const Signal& x, real64_t freq, real64_t sampleRate) {
  auto node = std::make_shared<detail::Node>();
  node->kind = OpKind::MIXER;
  node->inputs = {x.node()};
  node->shape = x.shape();
  node->inputType = x.type();
  node->outputType = x.type();
  node->param = std::vector<real64_t>{freq, sampleRate};
  return Signal(node);
}

/**
 * @brief QAM 符号映射 implementation
 *
 * 构建 OpKind::QAM_MAP 节点
 * 调制阶数 M (e.g. 16, 64) 存储于 `node->modOrder`
 *
 * 输入: Real 符号索引 [N]
 * 输出: Complex I/Q 星座点 [N] (实际 Buffer 为 [2, N])
 *
 * @throws std::invalid_argument 如果输入不是 Real 类型
 */
Signal qamMap(const Signal& bits, int order) {
  if (isComplexType(bits.type())) {
    throw std::invalid_argument("qamMap: input must be Real, not Complex");
  }

  auto node = std::make_shared<detail::Node>();
  node->kind = OpKind::QAM_MAP;
  node->inputs = {bits.node()};
  node->shape = bits.shape();
  node->inputType = bits.type();
  node->outputType = toComplexType(bits.type());
  node->param = order;
  return Signal(node);
}

/**
 * @brief QAM 符号解映射 implementation
 *
 * 构建 OpKind::QAM_DEMAP 节点
 *
 * 输入: Complex I/Q 星座点 [N]
 * 输出: Real 符号索引 [N]
 *
 * @throws std::invalid_argument 如果输入不是 Complex 类型
 */
Signal qamDemap(const Signal& symbols, int order) {
  if (!isComplexType(symbols.type())) {
    throw std::invalid_argument("qamDemap: input must be Complex type");
  }

  auto node = std::make_shared<detail::Node>();
  node->kind = OpKind::QAM_DEMAP;
  node->inputs = {symbols.node()};
  node->shape = symbols.shape();
  node->inputType = symbols.type();
  node->outputType = toRealType(symbols.type());
  node->param = order;
  return Signal(node);
}

// ============================================================================
// PSK 调制解调
// ============================================================================

/**
 * @brief PSK 符号映射 implementation
 *
 * 构建 OpKind::PSK_MAP 节点
 * 相位分布逻辑由后端实现，这里仅记录 order
 *
 * 输入: Real 符号索引 [N]
 * 输出: Complex I/Q 星座点 [N]
 *
 * @throws std::invalid_argument 如果输入不是 Real 类型
 */
Signal pskMap(const Signal& bits, int order) {
  if (isComplexType(bits.type())) {
    throw std::invalid_argument("pskMap: input must be Real, not Complex");
  }

  auto node = std::make_shared<detail::Node>();
  node->kind = OpKind::PSK_MAP;
  node->inputs = {bits.node()};
  node->shape = bits.shape();
  node->inputType = bits.type();
  node->outputType = toComplexType(bits.type());
  node->param = order;
  return Signal(node);
}

/**
 * @brief PSK 符号解映射 implementation
 *
 * 构建 OpKind::PSK_DEMAP 节点
 *
 * 输入: Complex I/Q 星座点 [N]
 * 输出: Real 符号索引 [N]
 *
 * @throws std::invalid_argument 如果输入不是 Complex 类型
 */
Signal pskDemap(const Signal& symbols, int order) {
  if (!isComplexType(symbols.type())) {
    throw std::invalid_argument("pskDemap: input must be Complex type");
  }

  auto node = std::make_shared<detail::Node>();
  node->kind = OpKind::PSK_DEMAP;
  node->inputs = {symbols.node()};
  node->shape = symbols.shape();
  node->inputType = symbols.type();
  node->outputType = toRealType(symbols.type());
  node->param = order;
  return Signal(node);
}

Signal qamDemap(const Signal& i, const Signal& q, int order) {
  if (i.type() != q.type()) {
    throw std::invalid_argument("qamDemap(i, q): inputs must have same type");
  }
  if (i.shape().length != q.shape().length || i.shape().channels != q.shape().channels ||
      i.shape().batch != q.shape().batch) {
    throw std::invalid_argument("qamDemap(i, q): inputs must have same shape");
  }
  if (isComplexType(i.type())) {
    throw std::invalid_argument("qamDemap(i, q): inputs must be Real");
  }
  auto node = std::make_shared<detail::Node>();
  node->kind = OpKind::QAM_DEMAP;
  node->inputs = {i.node(), q.node()};
  node->shape = i.shape();
  node->inputType = i.type();
  node->outputType = toRealType(i.type());  // output is indices (real)
  node->param = order;
  return Signal(node);
}

Signal pskDemap(const Signal& i, const Signal& q, int order) {
  if (i.type() != q.type()) {
    throw std::invalid_argument("pskDemap(i, q): inputs must have same type");
  }
  if (i.shape().length != q.shape().length || i.shape().channels != q.shape().channels ||
      i.shape().batch != q.shape().batch) {
    throw std::invalid_argument("pskDemap(i, q): inputs must have same shape");
  }
  if (isComplexType(i.type())) {
    throw std::invalid_argument("pskDemap(i, q): inputs must be Real");
  }
  auto node = std::make_shared<detail::Node>();
  node->kind = OpKind::PSK_DEMAP;
  node->inputs = {i.node(), q.node()};
  node->shape = i.shape();
  node->inputType = i.type();
  node->outputType = toRealType(i.type());
  node->param = order;
  return Signal(node);
}

}  // namespace prism::dsl::modem
