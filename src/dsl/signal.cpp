/**
 * @file signal.cpp
 * @ingroup dsl
 * @brief Signal 工厂函数实现
 *
 * 提供输入/常量节点的构造与内部节点封装
 */

#include "prism/dsl/signal.h"

#include <cstddef>
#include <memory>
#include <utility>

#include "prism/types.h"

namespace prism::dsl {

/*
 * 以内部节点直接构造 Signal (Private)
 * 该构造函数只在算子实现内部使用，外部建议走工厂接口
 */
Signal::Signal(std::shared_ptr<detail::Node> node) : node_(std::move(node)) {}

/*
 * 创建输入节点 (Factory)
 * 构建一个 OpKind::INPUT 类型的叶子节点
 */
Signal Signal::input(size_t length, ScalarType type) {
  auto node = std::make_shared<detail::Node>();
  node->kind = OpKind::INPUT;
  node->shape.length = length;
  node->inputType = type;
  node->outputType = type;
  return Signal(node);
}

/*
 * 从已有节点生成 Signal (Wrap)
 * 用于将底层 Node 指针重新封装为 DSL 层的 Signal 对象
 */
Signal Signal::fromNode(const std::shared_ptr<detail::Node>& node) { return Signal(node); }

}  // namespace prism::dsl
