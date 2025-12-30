/**
 * @file signal.cpp
 * @ingroup dsl
 * @brief Signal 工厂函数实现
 *
 * 提供输入/常量节点的构造与内部节点封装。
 */

#include "prism/dsl/signal.h"

#include <cstdint>
#include <memory>
#include <utility>

#include "prism/types.h"

namespace prism::dsl {

/**
 * @brief 以内部节点直接构造 Signal
 * @param node 已填充元信息的节点指针
 *
 * 该构造函数只在算子实现内部使用，外部建议走工厂接口。
 */
Signal::Signal(std::shared_ptr<detail::Node> node) : node_(std::move(node)) {}

/**
 * @brief 创建输入节点
 * @param length 样本长度
 * @param type 标量类型
 * @return 对应的 Signal 句柄
 *
 * 不做边界检查，调用方需保证 length > 0。
 */
Signal Signal::input(int64_t length, ScalarType type) {
  auto node = std::make_shared<detail::Node>();
  node->kind = OpKind::INPUT;
  node->shape.length = length;
  node->type = type;
  return Signal(node);
}

/**
 * @brief 创建常量节点
 * @param value 常量值
 * @param length 输出长度
 * @param type 标量类型
 */
Signal Signal::constant(real64_t value, int64_t length, ScalarType type) {
  auto node = std::make_shared<detail::Node>();
  node->kind = OpKind::CONSTANT;
  node->scalar = value;
  node->shape.length = length;
  node->type = type;
  return Signal(node);
}

/**
 * @brief 从已有节点生成 Signal
 * @param node 共享节点
 * @return 轻量句柄副本
 */
Signal Signal::fromNode(const std::shared_ptr<detail::Node>& node) {
  return Signal(node);
}

}  // namespace prism::dsl
