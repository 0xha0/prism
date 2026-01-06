/**
 * @file op_handler.h
 * @ingroup runtime
 * @brief 算子处理器接口与注册表
 *
 * 定义了将 DSL 算子映射为 Halide Func 的标准化接口（OpHandler）
 * 提供了算子注册机制（REGISTER_OP 宏），允许不同模块解耦注册实现
 */

#pragma once

#include <Halide.h>

#include <functional>
#include <map>
#include <unordered_map>

#include "prism/dsl/signal.h"
#include "prism/types.h"

namespace prism::runtime {

/// @addtogroup runtime
/// @{

// ============================================================================
// Unified Op Context
// ============================================================================

/**
 * @brief 算子构建上下文
 *
 * 在计算图构建过程中传递，用于维护状态
 *
 * @tparam T 标量类型
 */
template <typename T>
struct OpContext {
  /// 编译路径 (Compile-Time)：将 DSL 输入节点映射到 Halide ImageParam
  std::map<const dsl::detail::Node*, Halide::ImageParam*> inputParams;

  /// JIT 路径 (Run-Time)：将 DSL 输入节点映射到实际的 Halide Buffer
  std::map<const dsl::detail::Node*, const Halide::Buffer<typename ToHalideType<T>::Type>*>
      inputBuffers;

  /// 函数缓存：避免对同一节点重复构建 Halide::Func，保证 DAG 结构正确性
  std::unordered_map<const dsl::detail::Node*, Halide::Func> funcCache;

  /// 递归构建器回调：允许算子 Handler 请求构建其子节点（输入信号）
  std::function<Halide::Func(const dsl::Signal&)> buildFunc;

  /// GPU 标志：指示是否针对 GPU 目标进行构建（影响算子内部调度策略）
  bool useGpu = false;
};

// ============================================================================
// Handler Function Type
// ============================================================================

/**
 * @brief 算子处理器函数签名
 *
 * 所有算子实现必须符合此签名
 *
 * @param node 当前处理的 DSL 节点
 * @param ctx 构建上下文
 * @param args Halide 变量列表（实数 {x}, 复数 {c, x}）
 * @return 构建好的 Halide::Func
 */
template <typename T>
using OpHandler = Halide::Func (*)(const dsl::detail::Node* node, OpContext<T>& ctx,
                                   const std::vector<Halide::Var>& args);

// ============================================================================
// Op Registry
// ============================================================================

/**
 * @brief 算子注册表（单例模式）
 *
 * 负责管理 OpKind 到 OpHandler 的映射
 *
 * @tparam T 标量类型
 */
template <typename T>
class OpRegistry {
 public:
  static OpRegistry& instance() {
    static OpRegistry registry;
    return registry;
  }

  /**
   * @brief 注册一个算子 Handler
   * @param kind 算子类型 ID
   * @param handler 处理函数
   * @note 后注册的 Handler 会覆盖先注册的（允许用户重写算子）
   */
  void registerHandler(dsl::OpKind kind, OpHandler<T> handler) { handlers_[kind] = handler; }

  /**
   * @brief 查找算子 Handler
   * @param kind 算子类型 ID
   * @return 对应的处理函数，若未找到返回 nullptr
   */
  [[nodiscard]] OpHandler<T> getHandler(dsl::OpKind kind) const {
    auto it = handlers_.find(kind);
    if (it != handlers_.end()) {
      return it->second;
    }
    return nullptr;
  }

 private:
  OpRegistry() = default;
  std::unordered_map<dsl::OpKind, OpHandler<T>> handlers_;
};

// ============================================================================
// Registration Macro
// ============================================================================

/**
 * @def REGISTER_OP
 * @brief 算子注册宏
 *
 * 自动为 real32_t, real64_t, complex32_t, complex64_t 四种类型注册算子 Handler
 * 使用静态变量初始化机制，在 `main` 执行前完成注册
 *
 * @param Kind DSL OpKind 枚举名 (e.g., Add)
 * @param Func 模板函数名 (e.g., handle_add)
 */
// NOLINTBEGIN(bugprone-macro-parentheses)
#define REGISTER_OP(Kind, Func)                                                        \
  namespace {                                                                          \
  static const bool _reg_##Kind##_real32_t =                                           \
      (::prism::runtime::OpRegistry<::prism::real32_t>::instance().registerHandler(    \
           ::prism::dsl::OpKind::Kind, Func<::prism::real32_t>),                       \
       true);                                                                          \
  static const bool _reg_##Kind##_real64_t =                                           \
      (::prism::runtime::OpRegistry<::prism::real64_t>::instance().registerHandler(    \
           ::prism::dsl::OpKind::Kind, Func<::prism::real64_t>),                       \
       true);                                                                          \
  static const bool _reg_##Kind##_complex32_t =                                        \
      (::prism::runtime::OpRegistry<::prism::complex32_t>::instance().registerHandler( \
           ::prism::dsl::OpKind::Kind, Func<::prism::complex32_t>),                    \
       true);                                                                          \
  static const bool _reg_##Kind##_complex64_t =                                        \
      (::prism::runtime::OpRegistry<::prism::complex64_t>::instance().registerHandler( \
           ::prism::dsl::OpKind::Kind, Func<::prism::complex64_t>),                    \
       true);                                                                          \
  }
// NOLINTEND(bugprone-macro-parentheses)

/// @}

}  // namespace prism::runtime
