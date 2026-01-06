/**
 * @file halide_builder.h
 * @ingroup runtime
 * @brief DSL 到 Halide Func 的转换器
 *
 * 负责遍历 DSL 节点并构建对应的 Halide::Func 对象
 * 处理 INPUT、CONSTANT 以及 OpHandler 的递归调用
 */

#pragma once

#include <Halide.h>

#include <stdexcept>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

#include "prism/dsl/signal.h"
#include "prism/runtime/op_handler.h"
#include "prism/types.h"

namespace prism::runtime {
namespace detail {

/// @cond DOXYGEN_SKIP

template <typename T>
struct IsVector : std::false_type {};

template <typename T, typename A>
struct IsVector<std::vector<T, A>> : std::true_type {
  using ValueType = T;
};

// 验证输入 Buffer 的维度是否符合预期
template <typename T>
inline void validateInputBufferShape(const prism::dsl::detail::Node* node,
                                     const Halide::Buffer<typename ToHalideType<T>::Type>& buffer) {
  bool const nodeIsComplex = isComplexType(node->outputType);
  int const dims = buffer.dimensions();
  if (nodeIsComplex) {
    // 复数必须是 2D (C, N)，且 C=2
    if (dims != 2 || buffer.dim(0).extent() != 2) {
      throw std::runtime_error("Input buffer must be 2D with c=2 for complex INPUT node");
    }
  } else {
    // 实数必须是 1D (N)
    if (dims != 1) {
      throw std::runtime_error("Input buffer must be 1D for real INPUT node");
    }
  }
}

// 构建 Input 节点的 Func
template <typename T>
inline Halide::Func buildInputFunc(const prism::dsl::detail::Node* node, OpContext<T>& ctx,
                                   const std::vector<Halide::Var>& args) {
  Halide::Func func;
  std::vector<Halide::Expr> argsExpr;
  argsExpr.reserve(args.size());
  for (const auto& var : args) {
    argsExpr.emplace_back(var);
  }

  // 1. 尝试从 ImageParam 查找 (编译模式)
  auto paramIt = ctx.inputParams.find(node);
  if (paramIt != ctx.inputParams.end() && paramIt->second != nullptr) {
    func(args) = (*paramIt->second)(argsExpr);
    return func;
  }

  // 2. 尝试从 Buffer 查找 (JIT 运行模式)
  auto bufIt = ctx.inputBuffers.find(node);
  if (bufIt != ctx.inputBuffers.end() && bufIt->second != nullptr) {
    validateInputBufferShape<T>(node, *bufIt->second);
    func(args) = (*bufIt->second)(argsExpr);
    return func;
  }

  throw std::runtime_error("Input buffer/param not found for INPUT node");
}

// 处理标量常量
template <typename T, typename ArgT>
inline Halide::Func handleScalarConstant(const ArgT& val, const prism::dsl::detail::Node* node,
                                         const std::vector<Halide::Var>& args) {
  Halide::Func func;
  using HT = typename ToHalideType<T>::Type;
  bool const nodeIsComplex = isComplexType(node->outputType);

  if (getScalarType<ArgT>() != node->outputType) {
    throw std::runtime_error("CONSTANT param type does not match node output type");
  }

  Halide::Expr valReal;
  Halide::Expr valImag;

  if constexpr (IS_COMPLEX_V<ArgT>) {
    valReal = Halide::cast<HT>(Halide::Expr(val.real()));
    valImag = Halide::cast<HT>(Halide::Expr(val.imag()));
  } else {
    valReal = Halide::cast<HT>(Halide::Expr(val));
    valImag = Halide::cast<HT>(0);
  }

  if (nodeIsComplex) {
    const Halide::Var& c = args[0];
    const Halide::Var& x = args[1];
    func(c, x) = Halide::select(c == 0, valReal, valImag);
  } else {
    const Halide::Var& x = args[0];
    func(x) = valReal;
  }
  return func;
}

// 处理向量常量
template <typename T, typename VecT>
inline Halide::Func handleVectorConstant(const std::vector<VecT>& vec,
                                         const prism::dsl::detail::Node* node,
                                         const std::vector<Halide::Var>& args) {
  Halide::Func func;
  using HT = typename ToHalideType<T>::Type;

  if (getScalarType<VecT>() != node->outputType) {
    throw std::runtime_error("CONSTANT param type does not match node output type");
  }

  size_t const len = vec.size();
  if constexpr (IS_COMPLEX_V<VecT>) {
    Halide::Buffer<HT> cBuf(2, static_cast<int>(len));
    for (size_t i = 0; i < len; ++i) {
      cBuf(0, static_cast<int>(i)) = static_cast<HT>(vec[i].real());
      cBuf(1, static_cast<int>(i)) = static_cast<HT>(vec[i].imag());
    }
    const Halide::Var& c = args[0];
    const Halide::Var& x = args[1];
    func(c, x) = cBuf(c, x);
  } else {
    Halide::Buffer<HT> rBuf(static_cast<int>(len));
    for (size_t i = 0; i < len; ++i) {
      rBuf(static_cast<int>(i)) = static_cast<HT>(vec[i]);
    }
    const Halide::Var& x = args[0];
    func(x) = rBuf(x);
  }
  return func;
}

// 构建 Constant 节点的 Func
template <typename T>
inline Halide::Func buildConstantFunc(const prism::dsl::detail::Node* node,
                                      const std::vector<Halide::Var>& args) {
  return std::visit(
      [&](auto&& arg) -> Halide::Func {
        using ArgT = std::decay_t<decltype(arg)>;

        if constexpr (IS_REAL_TYPE_V<ArgT> || IS_COMPLEX_V<ArgT>) {
          return handleScalarConstant<T>(arg, node, args);
        } else if constexpr (IsVector<ArgT>::value) {
          using VecValT = typename IsVector<ArgT>::ValueType;
          if constexpr (IS_REAL_TYPE_V<VecValT> || IS_COMPLEX_V<VecValT>) {
            return handleVectorConstant<T>(arg, node, args);
          } else {
            throw std::runtime_error("CONSTANT only supports real/complex vector params");
          }
        } else {
          throw std::runtime_error("CONSTANT only supports real/complex scalar params");
        }
      },
      node->param);
}

// 递归构建 Signal Func 的内部实现
template <typename T>
inline Halide::Func buildSignalFunc(const prism::dsl::Signal& signal, OpContext<T>& ctx) {
  const auto* node = signal.node().get();
  if (!node) {
    throw std::runtime_error("Signal node is null");
  }

  // 检查缓存
  auto it = ctx.funcCache.find(node);
  if (it != ctx.funcCache.end()) {
    return it->second;
  }

  // 绑定递归回调使得 OpHandler 可以调用 buildSignalFunc
  ctx.buildFunc = [&ctx](const prism::dsl::Signal& s) {
    return ::prism::runtime::detail::buildSignalFunc<T>(s, ctx);
  };

  Halide::Func func;
  std::vector<Halide::Var> args;

  bool const nodeIsComplex = isComplexType(node->outputType);
  if (nodeIsComplex) {
    // 复数信号: (c, x)
    args.emplace_back("c");
    args.emplace_back("x");
  } else {
    // 实数信号: (x)
    args.emplace_back("x");
  }

  // 根据 OpKind 分发
  if (node->kind == prism::dsl::OpKind::INPUT) {
    func = buildInputFunc<T>(node, ctx, args);
  } else if (node->kind == prism::dsl::OpKind::CONSTANT) {
    func = buildConstantFunc<T>(node, args);
  } else {
    // 使用 Registry 查找 Handler
    auto handler = OpRegistry<T>::instance().getHandler(node->kind);
    if (!handler) {
      throw std::runtime_error("OpKind not implemented: " +
                               std::to_string(static_cast<int>(node->kind)));
    }
    func = handler(node, ctx, args);
  }

  // 更新缓存
  ctx.funcCache.emplace(node, func);
  return func;
}

/// @endcond

}  // namespace detail

/**
 * @brief 构建 Signal 对应的 Halide Func
 *
 * 这是 DSL 到 Halide IR 的核心入口点，它会递归遍历 DSL 节点树，
 * 利用缓存机制避免重复构建，并调用 OpRegistry 分发具体算子的 Handler
 *
 * @tparam T 标量类型
 * @param signal DSL 信号对象
 * @param ctx 构建上下文（包含输入绑定、缓存等）
 * @return 构建完成的 Halide::Func
 */
template <typename T>
inline Halide::Func buildSignalFunc(const prism::dsl::Signal& signal, OpContext<T>& ctx) {
  return detail::buildSignalFunc<T>(signal, ctx);
}
}  // namespace prism::runtime
