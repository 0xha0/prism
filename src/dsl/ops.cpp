/**
 * @file ops.cpp
 * @ingroup dsl
 * @brief 基础 DSL 算子实现
 */

#include "prism/dsl/ops.h"

#include <memory>

#include "prism/dsl/signal.h"
#include "prism/types.h"

namespace prism::dsl {

namespace {

/// 构建二元算子节点
Signal makeBinary(OpKind kind, const Signal& a, const Signal& b) {
  auto node = std::make_shared<detail::Node>();
  node->kind = kind;
  node->inputs = {a.node(), b.node()};
  node->shape = a.shape();
  node->type = a.type();
  return Signal(node);
}

/// 构建一元算子节点
Signal makeUnary(OpKind kind, const Signal& a) {
  auto node = std::make_shared<detail::Node>();
  node->kind = kind;
  node->inputs = {a.node()};
  node->shape = a.shape();
  node->type = a.type();
  return Signal(node);
}

}  // anonymous namespace

/** @brief 逐元素加法，shape/type 以左输入为准 */
Signal add(const Signal& a, const Signal& b) {
  return makeBinary(OpKind::ADD, a, b);
}

/** @brief 逐元素减法 */
Signal sub(const Signal& a, const Signal& b) {
  return makeBinary(OpKind::SUB, a, b);
}

/** @brief 逐元素乘法 */
Signal mul(const Signal& a, const Signal& b) {
  return makeBinary(OpKind::MUL, a, b);
}

/** @brief 逐元素除法 */
Signal div(const Signal& a, const Signal& b) {
  return makeBinary(OpKind::DIV, a, b);
}

/**
 * @brief 缩放运算
 * @param a 输入信号
 * @param v 缩放系数
 */
Signal scale(const Signal& a, real64_t v) {
  auto node = std::make_shared<detail::Node>();
  node->kind = OpKind::SCALE;
  node->inputs = {a.node()};
  node->shape = a.shape();
  node->type = a.type();
  node->scalar = v;
  return Signal(node);
}

/** @brief 逐元素取绝对值（模值） */
Signal abs(const Signal& a) { return makeUnary(OpKind::ABS, a); }

/**
 * @brief 卷积运算
 * @param a 输入信号
 * @param kernel 卷积核（长度需与算子匹配）
 * @return 输出长度 = a.length + kernel.length - 1
 */
Signal convolve(const Signal& a, const Signal& kernel) {
  auto node = std::make_shared<detail::Node>();
  node->kind = OpKind::CONVOLVE;
  node->inputs = {a.node(), kernel.node()};
  node->shape.length = a.shape().length + kernel.shape().length - 1;
  node->shape.channels = a.shape().channels;
  node->shape.batch = a.shape().batch;
  node->type = a.type();
  return Signal(node);
}

/**
 * @brief 克罗内克积（扩频）
 * @param a 输入信号
 * @param b 扩频码/重复向量
 * @return 输出长度 = a.length * b.length
 */
Signal kron(const Signal& a, const Signal& b) {
  auto node = std::make_shared<detail::Node>();
  node->kind = OpKind::KRON;
  node->inputs = {a.node(), b.node()};
  node->shape.length = a.shape().length * b.shape().length;
  node->shape.channels = a.shape().channels;
  node->shape.batch = a.shape().batch;
  node->type = a.type();
  return Signal(node);
}

// ============================================================================
// I/Q 交织辅助
// ============================================================================

/**
 * @brief 将两路实数 I/Q 合并为交织序列
 */
Signal iqPack(const Signal& i, const Signal& q) {
  auto node = std::make_shared<detail::Node>();
  node->kind = OpKind::IQ_PACK;
  node->inputs = {i.node(), q.node()};
  Shape outShape = i.shape();
  outShape.length *= 2;
  node->shape = outShape;
  node->type = i.type();
  return Signal(node);
}

/**
 * @brief 从交织 I/Q 序列提取 I 分量
 */
Signal iqI(const Signal& iq) {
  auto node = std::make_shared<detail::Node>();
  node->kind = OpKind::IQ_I;
  node->inputs = {iq.node()};
  Shape outShape = iq.shape();
  outShape.length /= 2;
  node->shape = outShape;
  node->type = iq.type();
  return Signal(node);
}

/**
 * @brief 从交织 I/Q 序列提取 Q 分量
 */
Signal iqQ(const Signal& iq) {
  auto node = std::make_shared<detail::Node>();
  node->kind = OpKind::IQ_Q;
  node->inputs = {iq.node()};
  Shape outShape = iq.shape();
  outShape.length /= 2;
  node->shape = outShape;
  node->type = iq.type();
  return Signal(node);
}

/**
 * @brief 上采样（插零）
 */
Signal upsample(const Signal& a, int factor) {
  auto node = std::make_shared<detail::Node>();
  node->kind = OpKind::UPSAMPLE;
  node->inputs = {a.node()};
  node->shape = a.shape();
  if (factor > 0) {
    node->shape.length = a.shape().length * factor;
  } else {
    node->shape.length = 0;
  }
  node->type = a.type();
  node->step = factor;
  node->offset = 0;
  return Signal(node);
}

/**
 * @brief 下采样（抽取）
 */
Signal downsample(const Signal& a, int factor, int offset) {
  auto node = std::make_shared<detail::Node>();
  node->kind = OpKind::DOWNSAMPLE;
  node->inputs = {a.node()};
  node->shape = a.shape();
  if (factor > 0 && offset >= 0 && offset < a.shape().length) {
    node->shape.length = (a.shape().length - offset + factor - 1) / factor;
  } else {
    node->shape.length = 0;
  }
  node->type = a.type();
  node->step = factor;
  node->offset = offset;
  return Signal(node);
}

Signal operator+(const Signal& a, const Signal& b) { return add(a, b); }
Signal operator-(const Signal& a, const Signal& b) { return sub(a, b); }
Signal operator*(const Signal& a, const Signal& b) { return mul(a, b); }
Signal operator/(const Signal& a, const Signal& b) { return div(a, b); }

}  // namespace prism::dsl
