/**
 * @file ops.cpp
 * @ingroup dsl
 * @brief 基础 DSL 算子实现
 */

#include "prism/dsl/ops.h"

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

#include "prism/dsl/signal.h"
#include "prism/types.h"

namespace prism::dsl {

namespace {

/// 统一二元运算类型逻辑
/// 1. 检查精度是否匹配 (F32/C32 or F64/C64)
/// 2. 推导输出类型 (Real+Complex -> Complex)
void unifyBinaryType(const Signal& a, const Signal& b, std::shared_ptr<detail::Node>& node) {
  if (!isPrecisionMatch(a.type(), b.type())) {
    throw std::invalid_argument(
        "Binary op precision mismatch: input types must verify "
        "isPrecisionMatch");
  }
  node->outputType = promoteTypes(a.type(), b.type());
}

/// 构建二元算子节点
Signal makeBinary(OpKind kind, const Signal& a, const Signal& b) {
  if (a.shape().length != b.shape().length || a.shape().channels != b.shape().channels ||
      a.shape().batch != b.shape().batch) {
    throw std::invalid_argument(
        "Binary op shape mismatch: inputs must have same shape (no "
        "broadcasting)");
  }
  auto node = std::make_shared<detail::Node>();
  node->kind = kind;
  node->inputs = {a.node(), b.node()};
  node->shape = a.shape();
  node->inputType = a.type();  // 仅记录主输入类型
  unifyBinaryType(a, b, node);
  return Signal(node);
}
}  // anonymous namespace

/** @brief 逐元素加法 */
Signal add(const Signal& a, const Signal& b) { return makeBinary(OpKind::ADD, a, b); }

/** @brief 逐元素减法 */
Signal sub(const Signal& a, const Signal& b) { return makeBinary(OpKind::SUB, a, b); }

/** @brief 逐元素乘法 */
Signal mul(const Signal& a, const Signal& b) { return makeBinary(OpKind::MUL, a, b); }

/** @brief 逐元素除法 */
Signal div(const Signal& a, const Signal& b) { return makeBinary(OpKind::DIV, a, b); }

// scale implementation moved to header template

/** @brief 逐元素取绝对值（模值） */
Signal abs(const Signal& a) {
  auto node = std::make_shared<detail::Node>();
  node->kind = OpKind::ABS;
  node->inputs = {a.node()};
  node->shape = a.shape();
  node->inputType = a.type();
  node->outputType = toRealType(a.type());
  return Signal(node);
}

/** @brief 逐元素取负 */
Signal negative(const Signal& a) {
  auto node = std::make_shared<detail::Node>();
  node->kind = OpKind::NEG;
  node->inputs = {a.node()};
  node->shape = a.shape();
  node->inputType = a.type();
  node->outputType = a.type();
  return Signal(node);
}

/** @brief 逐元素取共轭 */
Signal conj(const Signal& a) {
  auto node = std::make_shared<detail::Node>();
  node->kind = OpKind::CONJ;
  node->inputs = {a.node()};
  node->shape = a.shape();
  node->inputType = a.type();
  node->outputType = a.type();
  return Signal(node);
}

/**
 * @brief 卷积运算
 *
 * 构建 OpKind::CONVOLVE 节点
 * 输出长度遵循 full convolution 规则
 */
Signal convolve(const Signal& a, const Signal& kernel) {
  auto node = std::make_shared<detail::Node>();
  node->kind = OpKind::CONVOLVE;
  node->inputs = {a.node(), kernel.node()};
  node->shape.length = a.shape().length + kernel.shape().length - 1;
  node->shape.channels = a.shape().channels;
  node->shape.batch = a.shape().batch;
  node->inputType = a.type();
  unifyBinaryType(a, kernel, node);
  return Signal(node);
}

/**
 * @brief 克罗内克积
 *
 * 构建 OpKind::KRON 节点
 * 常用于将每个输入符号扩展为整个扩频码序列
 */
Signal kron(const Signal& a, const Signal& b) {
  auto node = std::make_shared<detail::Node>();
  node->kind = OpKind::KRON;
  node->inputs = {a.node(), b.node()};
  node->shape.length = a.shape().length * b.shape().length;
  node->shape.channels = a.shape().channels;
  node->shape.batch = a.shape().batch;
  node->inputType = a.type();
  unifyBinaryType(a, b, node);
  return Signal(node);
}

/**
 * @brief 将两路实数信号合并为复数信号
 *
 * 构建 OpKind::CPLX_PACK 节点
 * 输出为 Complex 类型，长度与输入一致
 *
 * @throws std::invalid_argument 如果输入类型不匹配或不是 Real 类型
 */
Signal complexPack(const Signal& i, const Signal& q) {
  // 类型检查：两个输入必须是相同的 Real 类型
  if (i.type() != q.type()) {
    throw std::invalid_argument("complexPack: I and Q must have the same type");
  }
  if (isComplexType(i.type())) {
    throw std::invalid_argument("complexPack: inputs must be Real, not Complex");
  }

  auto node = std::make_shared<detail::Node>();
  node->kind = OpKind::CPLX_PACK;
  node->inputs = {i.node(), q.node()};
  node->shape = i.shape();  // Complex 布局，长度不变
  node->inputType = i.type();
  node->outputType = toComplexType(i.type());  // Real → Complex
  return Signal(node);
}

/**
 * @brief 从 Complex 信号提取 I 分量
 *
 * 构建 OpKind::REAL 节点
 * 输入必须是 Complex 类型，输出为 Real 类型
 *
 * @throws std::invalid_argument 如果输入不是 Complex 类型
 */
Signal real(const Signal& iq) {
  if (!isComplexType(iq.type())) {
    throw std::invalid_argument("real: input must be Complex type");
  }

  auto node = std::make_shared<detail::Node>();
  node->kind = OpKind::REAL;
  node->inputs = {iq.node()};
  node->shape = iq.shape();
  node->inputType = iq.type();
  node->outputType = toRealType(iq.type());  // Complex → Real
  return Signal(node);
}

/**
 * @brief 从 Complex 信号提取 Q 分量
 *
 * 构建 OpKind::IMAG 节点
 * 输入必须是 Complex 类型，输出为 Real 类型
 *
 * @throws std::invalid_argument 如果输入不是 Complex 类型
 */
Signal imag(const Signal& iq) {
  if (!isComplexType(iq.type())) {
    throw std::invalid_argument("imag: input must be Complex type");
  }

  auto node = std::make_shared<detail::Node>();
  node->kind = OpKind::IMAG;
  node->inputs = {iq.node()};
  node->shape = iq.shape();
  node->inputType = iq.type();
  node->outputType = toRealType(iq.type());  // Complex → Real
  return Signal(node);
}

/**
 * @brief 上采样（插零）
 *
 * 构建 OpKind::UPSAMPLE 节点
 * 参数 factor/offset 存储于 param 字段
 */
Signal upsample(const Signal& a, int factor, int offset) {
  auto node = std::make_shared<detail::Node>();
  node->kind = OpKind::UPSAMPLE;
  node->inputs = {a.node()};
  node->shape = a.shape();
  if (factor > 0 && offset >= 0 && offset < factor) {
    node->shape.length = a.shape().length * factor;
  } else {
    node->shape.length = 0;
  }
  node->inputType = a.type();
  node->outputType = a.type();
  node->param = std::vector<int64_t>{static_cast<int64_t>(factor), static_cast<int64_t>(offset)};
  return Signal(node);
}

/**
 * @brief 下采样（抽取）
 *
 * 构建 OpKind::DOWNSAMPLE 节点
 * 参数 factor 存储于 step 字段，offset 存储于 offset 字段
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
  node->inputType = a.type();
  node->outputType = a.type();
  node->param = std::vector<int64_t>{static_cast<int64_t>(factor), static_cast<int64_t>(offset)};
  return Signal(node);
}

Signal operator+(const Signal& a, const Signal& b) { return add(a, b); }
Signal operator-(const Signal& a, const Signal& b) { return sub(a, b); }
Signal operator*(const Signal& a, const Signal& b) { return mul(a, b); }
Signal operator/(const Signal& a, const Signal& b) { return div(a, b); }
Signal operator-(const Signal& a) { return negative(a); }
Signal operator~(const Signal& a) { return conj(a); }

}  // namespace prism::dsl
