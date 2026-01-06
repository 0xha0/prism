/**
 * @file convolution_handlers.cpp
 * @ingroup runtime
 * @brief 卷积相关 Handler：Convolve、Kron
 *
 * 实现了线性卷积和克罗内克积算子的 Halide 转换
 */

#include <Halide.h>

#include <vector>

#include "prism/dsl/signal.h"
#include "prism/runtime/op_handler.h"
#include "prism/types.h"

namespace prism::runtime {

/// @addtogroup runtime
/// @{

namespace {
// 边界条件处理辅助函数
template <typename T>
Halide::Func makeBoundedInput(const Halide::Func& inputFunc, int len, bool isComplex) {
  using ElemT = typename ToHalideType<T>::Type;
  if (isComplex) {
    Halide::Region const bounds = {{0, 2}, {0, len}};
    // 使用 constant_exterior（补零）作为默认边界条件
    Halide::Func bounded =
        Halide::BoundaryConditions::constant_exterior(inputFunc, Halide::cast<ElemT>(0), bounds);
    auto const args = bounded.args();
    if (!args.empty()) {
      bounded.fold_storage(args.back(), len);
    }
    return bounded;
  }
  Halide::Func bounded =
      Halide::BoundaryConditions::constant_exterior(inputFunc, Halide::cast<ElemT>(0), 0, len);
  auto const args = bounded.args();
  if (!args.empty()) {
    bounded.fold_storage(args.back(), len);
  }
  return bounded;
}
}  // namespace

/**
 * @brief Handle Convolve 算子
 *
 * $ y[n] = (x * h)[n] = \sum_{m} x[n-m] h[m] $
 *
 * 采用零填充边界条件
 * 支持 Real/Complex 卷积，包括混合类型（Real * Complex 等）
 */
template <typename T>
Halide::Func handleConvolve(const dsl::detail::Node* node, OpContext<T>& ctx,
                            const std::vector<Halide::Var>& args) {
  auto a = ctx.buildFunc(dsl::Signal::fromNode(node->inputs.at(0)));
  auto kernel = ctx.buildFunc(dsl::Signal::fromNode(node->inputs.at(1)));
  int const kernelLen = static_cast<int>(node->inputs.at(1)->shape.length);
  int const aLen = static_cast<int>(node->inputs.at(0)->shape.length);

  Halide::RDom const r(0, kernelLen);
  Halide::Func func;
  using ElemT = typename ToHalideType<T>::Type;
  bool const outputComplex = isComplexType(node->outputType);
  bool const isAComplex = isComplexType(node->inputs[0]->outputType);
  bool const isKComplex = isComplexType(node->inputs[1]->outputType);

  // (a * k)[n] = sum(a[n-m] * k[m])
  Halide::Func const bounded = makeBoundedInput<T>(a, aLen, isAComplex);

  if (outputComplex) {
    Halide::Var const& c = args[0];
    Halide::Var const& x = args[1];
    Halide::Expr const zero = Halide::cast<ElemT>(0);

    if (isAComplex && isKComplex) {
      // C * C
      Halide::Expr const ar = bounded(0, x - r);
      Halide::Expr const ai = bounded(1, x - r);
      Halide::Expr const kr = kernel(0, r);
      Halide::Expr const ki = kernel(1, r);
      func(c, x) = Halide::mux(c, {Halide::sum(ar * kr - ai * ki), Halide::sum(ar * ki + ai * kr)});
    } else if (isAComplex) {
      // C * R
      Halide::Expr const k = kernel(r);
      func(c, x) = Halide::sum(bounded(c, x - r) * k);
    } else if (isKComplex) {
      // R * C
      Halide::Expr const aVal = bounded(x - r);
      Halide::Expr const kr = kernel(0, r);
      Halide::Expr const ki = kernel(1, r);
      func(c, x) = Halide::mux(c, {Halide::sum(aVal * kr), Halide::sum(aVal * ki)});
    } else {
      // R * R -> C (理论上不会发生，除非 outputType 被强制指定为 Complex)
      Halide::Expr const aVal = bounded(x - r);
      Halide::Expr const k = kernel(r);
      func(c, x) = Halide::mux(c, {Halide::sum(aVal * k), zero});
    }
  } else {
    Halide::Var const& x = args[0];
    func(x) = Halide::sum(bounded(x - r) * kernel(r));
  }
  return func;
}

REGISTER_OP(CONVOLVE, handleConvolve);

/**
 * @brief Handle Kron (Kronecker Product) 算子
 *
 * $ Y = A \otimes B $
 *
 * 对于 1D 信号，结果长度为 $Len(A) \times Len(B)$
 * 实现上，我们将 $A$ 的每个元素扩展为 $A[i] \times B$
 */
template <typename T>
Halide::Func handleKron(const dsl::detail::Node* node, OpContext<T>& ctx,
                        const std::vector<Halide::Var>& args) {
  auto a = ctx.buildFunc(dsl::Signal::fromNode(node->inputs.at(0)));
  auto b = ctx.buildFunc(dsl::Signal::fromNode(node->inputs.at(1)));
  int const bLen = static_cast<int>(node->inputs.at(1)->shape.length);

  Halide::Func func;
  using ElemT = typename ToHalideType<T>::Type;
  bool const outputComplex = isComplexType(node->outputType);
  bool const isAComplex = isComplexType(node->inputs[0]->outputType);
  bool const isBComplex = isComplexType(node->inputs[1]->outputType);

  if (outputComplex) {
    Halide::Var const& c = args[0];
    Halide::Var const& x = args[1];

    Halide::Expr const aIdx = x / bLen;
    Halide::Expr const bIdx = x % bLen;

    Halide::Expr const zero = Halide::cast<ElemT>(0);
    Halide::Expr ar = zero;
    Halide::Expr ai = zero;
    Halide::Expr br = zero;
    Halide::Expr bi = zero;
    if (isAComplex) {
      ar = a(0, aIdx);
      ai = a(1, aIdx);
    } else {
      ar = a(aIdx);
    }
    if (isBComplex) {
      br = b(0, bIdx);
      bi = b(1, bIdx);
    } else {
      br = b(bIdx);
    }

    func(c, x) = Halide::mux(c, {ar * br - ai * bi, ar * bi + ai * br});
  } else {
    Halide::Var const& x = args[0];
    func(x) = a(x / bLen) * b(x % bLen);
  }
  return func;
}

REGISTER_OP(KRON, handleKron);

// 显式注册函数（供 Executor.cpp 调用，确保目标文件被链接）
void registerConvolutionHandlers() {
  // Handler 已通过 REGISTER_OP 自动注册
}

/// @}

}  // namespace prism::runtime
