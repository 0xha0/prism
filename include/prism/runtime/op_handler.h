/**
 * @file op_handler.h
 * @ingroup runtime
 * @brief Handler 接口与注册表
 *
 * Runtime 通过 Handler 将 DSL 节点映射为 Halide Func，本文件定义了
 * 统一上下文、函数签名以及注册工具宏。
 */

#pragma once

#include <Halide.h>

#include <functional>
#include <map>
#include <tuple>
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
 * @brief Halide 构图上下文
 *
 * 同时支持编译阶段的 ImageParam 与运行阶段的 Buffer，内部维护构图缓存，
 * 避免重复构建同一节点。
 *
 * @tparam T 标量类型（real32_t/real64_t/complex32_t/complex64_t）
 */
template <typename T>
struct OpContext {
  /// 编译模式：ImageParam 输入
  Halide::ImageParam* inputParam = nullptr;

  /// 运行模式：Buffer 输入
  /// 注意：对于复数类型，这里存储的是 reinterpret_cast 后的实数 Buffer 指针
  const Halide::Buffer<typename ToHalideType<T>::Type>* inputBuffer = nullptr;

  /// 缓存，避免重复构建相同节点
  std::unordered_map<const dsl::detail::Node*, Halide::Func> funcCache;

  /// 递归构建函数（由 buildFunc 填充，用于 handler 调度子节点）
  std::function<Halide::Func(const dsl::Signal&)> buildFunc;

  // 系数缓存（同一次构图调用内复用）
  std::map<std::vector<real32_t>, Halide::Buffer<real32_t>> firTapsCache32;
  std::map<std::vector<real64_t>, Halide::Buffer<real64_t>> firTapsCache64;
  std::map<std::tuple<int, std::vector<real32_t>, std::vector<real32_t>>,
           Halide::Buffer<real32_t>>
      iirCoeffCache32;
  std::map<std::tuple<int, std::vector<real64_t>, std::vector<real64_t>>,
           Halide::Buffer<real64_t>>
      iirCoeffCache64;
};

// ============================================================================
// Handler Function Type
// ============================================================================

/**
 * @brief Handler 函数签名
 *
 * 负责将 DSL 节点转换为 Halide::Func，并可以递归调用子节点。
 * args: 对于实数对应 {x}，对于复数对应 {c, x}
 */
template <typename T>
using OpHandler = Halide::Func (*)(const dsl::detail::Node* node,
                                   OpContext<T>& ctx,
                                   const std::vector<Halide::Var>& args);

// ============================================================================
// Op Registry
// ============================================================================

/**
 * @brief Handler 注册表（单例）
 *
 * 提供按 @ref prism::dsl::OpKind 查找对应 Handler 的能力。
 * @tparam T 标量类型（real32_t/real64_t）
 */
template <typename T>
class OpRegistry {
 public:
  static OpRegistry& instance() {
    static OpRegistry registry;
    return registry;
  }

  /** @brief 注册算子 Handler（同一算子后注册者覆盖前者） */
  void registerHandler(dsl::OpKind kind, OpHandler<T> handler) {
    handlers_[kind] = handler;
  }

  /**
   * @brief 获取指定算子的 Handler
   * @return 若未注册返回 nullptr
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
 * @brief Handler 注册宏（同时注册 real32_t/real64_t 版本）
 *
 * 用法示例：`REGISTER_OP(Add, handle_add);`
 */
// NOLINTBEGIN(bugprone-macro-parentheses)
#define REGISTER_OP(Kind, Func)                                               \
  namespace {                                                                 \
  static const bool _reg_##Kind##_real32_t =                                  \
      (::prism::runtime::OpRegistry<real32_t>::instance().registerHandler(    \
           ::prism::dsl::OpKind::Kind, Func<real32_t>),                       \
       true);                                                                 \
  static const bool _reg_##Kind##_real64_t =                                  \
      (::prism::runtime::OpRegistry<real64_t>::instance().registerHandler(    \
           ::prism::dsl::OpKind::Kind, Func<real64_t>),                       \
       true);                                                                 \
  static const bool _reg_##Kind##_complex32_t =                               \
      (::prism::runtime::OpRegistry<complex32_t>::instance().registerHandler( \
           ::prism::dsl::OpKind::Kind, Func<complex32_t>),                    \
       true);                                                                 \
  static const bool _reg_##Kind##_complex64_t =                               \
      (::prism::runtime::OpRegistry<complex64_t>::instance().registerHandler( \
           ::prism::dsl::OpKind::Kind, Func<complex64_t>),                    \
       true);                                                                 \
  }
// NOLINTEND(bugprone-macro-parentheses)

/// @}

}  // namespace prism::runtime
