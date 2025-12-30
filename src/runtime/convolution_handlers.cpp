/**
 * @file convolution_handlers.cpp
 * @ingroup runtime
 * @brief 卷积相关 Handler：Convolve、Kron
 */

#include <Halide.h>

#include <vector>

#include "prism/dsl/signal.h"
#include "prism/runtime/op_handler.h"
#include "prism/types.h"

namespace prism::runtime {

/// @addtogroup runtime
/// @{

// ============================================================================
// Convolve
// ============================================================================

/// 线性卷积（零填充边界）
template <typename T>
Halide::Func handleConvolve(const dsl::detail::Node* node, OpContext<T>& ctx,
                            const std::vector<Halide::Var>& args) {
  auto a = ctx.buildFunc(dsl::Signal::fromNode(node->inputs.at(0)));
  auto kernel = ctx.buildFunc(dsl::Signal::fromNode(node->inputs.at(1)));
  int const kernelLen = static_cast<int>(node->inputs.at(1)->shape.length);
  int const aLen = static_cast<int>(node->inputs.at(0)->shape.length);

  Halide::RDom const r(0, kernelLen);
  Halide::Func func;

  if constexpr (IS_COMPLEX_V<T>) {
    // check if kernel is complex or real?
    // Current Signal system types whole pipeline as T generally,
    // but inputs can be mixed? dsl::Signal has `type()` but OpContext<T> fixes
    // compilation to T. If T=Complex, we assume both are complex for safety, or
    // implicit broadcast if one is simpler. Let's implement fully Complex *
    // Complex convolution.

    Halide::Var const& c = args[0];
    Halide::Var const& x = args[1];

    // (a * k)[n] = sum(a[n-m] * k[m])
    // Complex mult sum.

    Halide::Expr const idx = x - r;
    Halide::Expr const inBounds = (idx >= 0 && idx < aLen);
    Halide::Expr const boundedX = Halide::clamp(idx, 0, aLen - 1);

    Halide::Expr const ar = a(0, boundedX);
    Halide::Expr const ai = a(1, boundedX);
    Halide::Expr const kr = kernel(0, r);
    Halide::Expr const ki = kernel(1, r);

    // mul = (ar*kr - ai*ki) + i(ar*ki + ai*kr)
    Halide::Expr const realPart = Halide::sum(
        Halide::select(inBounds, ar * kr - ai * ki,
                       Halide::cast<typename ToHalideType<T>::Type>(0)));
    Halide::Expr const imagPart = Halide::sum(
        Halide::select(inBounds, ar * ki + ai * kr,
                       Halide::cast<typename ToHalideType<T>::Type>(0)));

    func(c, x) = Halide::select(c == 0, realPart, imagPart);

  } else {
    Halide::Var const& x = args[0];
    func(x) = Halide::sum(Halide::select(
        x - r >= 0 && x - r < aLen,
        a(Halide::clamp(x - r, 0, aLen - 1)) * kernel(r), Halide::cast<T>(0)));
  }
  return func;
}

REGISTER_OP(CONVOLVE, handleConvolve);

// ============================================================================
// Kron
// ============================================================================

/// 克罗内克积：按 b 展开复制 a
template <typename T>
Halide::Func handleKron(const dsl::detail::Node* node, OpContext<T>& ctx,
                        const std::vector<Halide::Var>& args) {
  auto a = ctx.buildFunc(dsl::Signal::fromNode(node->inputs.at(0)));
  auto b = ctx.buildFunc(dsl::Signal::fromNode(node->inputs.at(1)));
  int const bLen = static_cast<int>(node->inputs.at(1)->shape.length);

  Halide::Func func;
  if constexpr (IS_COMPLEX_V<T>) {
    // A (x) B
    // For complex, elementwise mul logic applies if A element is complex.
    // But Kron usually is block expansion.
    // If A is matrix, B is matrix... here 1D signals.
    // [a0, a1] (x) [b0, b1] = [a0b0, a0b1, a1b0, a1b1]
    // Complex: a0 is pair (r,i). b0 is pair (r,i).

    // Let's implement strictly structure:
    // index n = x
    // a_idx = x / bLen
    // b_idx = x % bLen
    // val = a(a_idx) * b(b_idx) -> Complex Mul!

    Halide::Var const& c = args[0];
    Halide::Var const& x = args[1];

    Halide::Expr const aIdx = x / bLen;
    Halide::Expr const bIdx = x % bLen;

    Halide::Expr const ar = a(0, aIdx);
    Halide::Expr const ai = a(1, aIdx);
    Halide::Expr const br = b(0, bIdx);
    Halide::Expr const bi = b(1, bIdx);

    func(c, x) = Halide::select(c == 0, ar * br - ai * bi, ar * bi + ai * br);
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
