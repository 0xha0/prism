/**
 * @file arithmetic_handlers.cpp
 * @ingroup runtime
 * @brief 基础代数算子 Handler（Add/Sub/Mul/Div/Scale/Abs）
 *
 * 将 DSL 中的算子节点转换为 Halide::Func，支持 real32_t/real64_t 两种精度。
 */

#include <Halide.h>

#include <stdexcept>
#include <vector>

#include "prism/dsl/signal.h"
#include "prism/runtime/op_handler.h"
#include "prism/types.h"

namespace prism::runtime {

/// @addtogroup runtime
/// @{

// ============================================================================
// Add
// ============================================================================

/// Add：逐元素相加
template <typename T>
Halide::Func handleAdd(const dsl::detail::Node* node, OpContext<T>& ctx,
                       const std::vector<Halide::Var>& args) {
  auto a = ctx.buildFunc(dsl::Signal::fromNode(node->inputs.at(0)));
  auto b = ctx.buildFunc(dsl::Signal::fromNode(node->inputs.at(1)));
  Halide::Func func;
  func(args) = a(args) + b(args);
  return func;
}

REGISTER_OP(ADD, handleAdd);

// ============================================================================
// Sub
// ============================================================================

/// Sub：逐元素相减
template <typename T>
Halide::Func handleSub(const dsl::detail::Node* node, OpContext<T>& ctx,
                       const std::vector<Halide::Var>& args) {
  auto a = ctx.buildFunc(dsl::Signal::fromNode(node->inputs.at(0)));
  auto b = ctx.buildFunc(dsl::Signal::fromNode(node->inputs.at(1)));
  Halide::Func func;
  func(args) = a(args) - b(args);
  return func;
}

REGISTER_OP(SUB, handleSub);

// ============================================================================
// Mul
// ============================================================================

/// Mul：逐元素相乘
template <typename T>
Halide::Func handleMul(const dsl::detail::Node* node, OpContext<T>& ctx,
                       const std::vector<Halide::Var>& args) {
  auto a = ctx.buildFunc(dsl::Signal::fromNode(node->inputs.at(0)));
  auto b = ctx.buildFunc(dsl::Signal::fromNode(node->inputs.at(1)));
  Halide::Func func;

  if constexpr (IS_COMPLEX_V<T>) {
    // args = {c, x}
    Halide::Var const& c = args[0];
    Halide::Var const& x = args[1];

    // (Ar + i Ai) * (Br + i Bi) = (ArBr - AiBi) + i (ArBi + AiBr)
    Halide::Expr const ar = a(0, x);
    Halide::Expr const ai = a(1, x);
    Halide::Expr const br = b(0, x);
    Halide::Expr const bi = b(1, x);

    func(c, x) = Halide::select(c == 0, ar * br - ai * bi, ar * bi + ai * br);
  } else {
    func(args) = a(args) * b(args);
  }
  return func;
}

REGISTER_OP(MUL, handleMul);

// ============================================================================
// Div
// ============================================================================

/// Div：逐元素相除
template <typename T>
Halide::Func handleDiv(const dsl::detail::Node* node, OpContext<T>& ctx,
                       const std::vector<Halide::Var>& args) {
  auto a = ctx.buildFunc(dsl::Signal::fromNode(node->inputs.at(0)));
  auto b = ctx.buildFunc(dsl::Signal::fromNode(node->inputs.at(1)));
  Halide::Func func;

  if constexpr (IS_COMPLEX_V<T>) {
    Halide::Var const& c = args[0];
    Halide::Var const& x = args[1];

    Halide::Expr const ar = a(0, x);
    Halide::Expr const ai = a(1, x);
    Halide::Expr const br = b(0, x);
    Halide::Expr const bi = b(1, x);

    Halide::Expr const denom = br * br + bi * bi;

    // (Ar + i Ai) / (Br + i Bi) = [(ArBr + AiBi) + i (AiBr - ArBi)] / (Br^2 +
    // Bi^2)
    func(c, x) = Halide::select(c == 0, (ar * br + ai * bi) / denom,
                                (ai * br - ar * bi) / denom);
  } else {
    func(args) = a(args) / b(args);
  }
  return func;
}

REGISTER_OP(DIV, handleDiv);

// ============================================================================
// Scale
// ============================================================================

/// Scale：乘以标量
template <typename T>
Halide::Func handleScale(const dsl::detail::Node* node, OpContext<T>& ctx,
                         const std::vector<Halide::Var>& args) {
  auto a = ctx.buildFunc(dsl::Signal::fromNode(node->inputs.at(0)));
  Halide::Func func;
  // Ensure we use the correct precision for the scalar
  // node->scalar is stored as real64_t
  using HT = typename ToHalideType<T>::Type;
  func(args) = a(args) * Halide::Expr(static_cast<HT>(node->scalar));
  return func;
}

REGISTER_OP(SCALE, handleScale);

// ============================================================================
// Abs
// ============================================================================

/// Abs：取绝对值（模值）
template <typename T>
Halide::Func handleAbs(const dsl::detail::Node* node, OpContext<T>& ctx,
                       const std::vector<Halide::Var>& args) {
  auto a = ctx.buildFunc(dsl::Signal::fromNode(node->inputs.at(0)));
  Halide::Func func;

  if constexpr (IS_COMPLEX_V<T>) {
    // For complex, Abs returns Magnitude, which is REAL.
    // So the output of Abs(Complex) is Real.
    // BUT, DSL Node currently defines generic type T for "Abs" based on input
    // usually? Actually DSL Ops.h defines output type. Signal::abs(a) should
    // strictly return real. However, our DSL node system currently might
    // inherit T. If the OpNode type is Complex, Abs returning Real doesn't fit
    // standard pipeline unless we change downstream expectations.
    //
    // Standard: abs(complex) -> real.
    // If we return Real Func, we essentially drop the simple 'T' propagation.
    // But OpContext is templated on T. If T is complex, context expects Complex
    // output buffer.
    //
    // This is a design friction. "Abs" of complex *changes type* to Real.
    // But our Executor runs for "T".
    // If we run `Executor::run<complex32_t>(y)`, and y uses Abs, y technically
    // becomes real. This implies DSL type inference needs to handle this, OR we
    // return Complex with Imag=0?

    // Let's assume for now Abs on Complex returns Complex with Real=Mag,
    // Imag=0. This keeps the pipeline T consistent.
    Halide::Var const& c = args[0];
    Halide::Var const& x = args[1];

    Halide::Expr const ar = a(0, x);
    Halide::Expr const ai = a(1, x);
    Halide::Expr const mag = Halide::sqrt(ar * ar + ai * ai);

    func(c, x) = Halide::select(
        c == 0, mag, Halide::cast<typename ToHalideType<T>::Type>(0));
  } else {
    func(args) = Halide::abs(a(args));
  }
  return func;
}

REGISTER_OP(ABS, handleAbs);

// ============================================================================
// Upsample
// ============================================================================

/// Upsample：插零上采样
template <typename T>
Halide::Func handleUpsample(const dsl::detail::Node* node, OpContext<T>& ctx,
                            const std::vector<Halide::Var>& args) {
  auto inputFunc = ctx.buildFunc(dsl::Signal::fromNode(node->inputs.at(0)));
  int const inputLen = static_cast<int>(node->inputs.at(0)->shape.length);
  int const factor = static_cast<int>(node->step);
  if (factor <= 0) {
    throw std::runtime_error("Upsample factor must be > 0");
  }

  Halide::Func func;
  if constexpr (IS_COMPLEX_V<T>) {
    Halide::Var const& c = args[0];
    Halide::Var const& x = args[1];
    Halide::Expr const idx = x / factor;
    Halide::Expr const inRange = idx >= 0 && idx < inputLen;
    Halide::Expr const isHit = (x % factor) == 0;
    func(c, x) =
        Halide::select(isHit && inRange, inputFunc(c, idx),
                       Halide::cast<typename ToHalideType<T>::Type>(0));
  } else {
    Halide::Var const& x = args[0];
    Halide::Expr const idx = x / factor;
    Halide::Expr const inRange = idx >= 0 && idx < inputLen;
    Halide::Expr const isHit = (x % factor) == 0;
    func(x) =
        Halide::select(isHit && inRange, inputFunc(idx), Halide::cast<T>(0));
  }
  return func;
}

REGISTER_OP(UPSAMPLE, handleUpsample);

// ============================================================================
// Downsample
// ============================================================================

/// Downsample：抽取
template <typename T>
Halide::Func handleDownsample(const dsl::detail::Node* node, OpContext<T>& ctx,
                              const std::vector<Halide::Var>& args) {
  auto inputFunc = ctx.buildFunc(dsl::Signal::fromNode(node->inputs.at(0)));
  int const inputLen = static_cast<int>(node->inputs.at(0)->shape.length);
  int const factor = static_cast<int>(node->step);
  int const offset = static_cast<int>(node->offset);
  if (factor <= 0) {
    throw std::runtime_error("Downsample factor must be > 0");
  }
  if (offset < 0) {
    throw std::runtime_error("Downsample offset must be >= 0");
  }

  Halide::Func func;
  if constexpr (IS_COMPLEX_V<T>) {
    Halide::Var const& c = args[0];
    Halide::Var const& x = args[1];
    Halide::Expr const idx = x * factor + offset;
    Halide::Expr const inRange = idx >= 0 && idx < inputLen;
    func(c, x) =
        Halide::select(inRange, inputFunc(c, idx),
                       Halide::cast<typename ToHalideType<T>::Type>(0));
  } else {
    Halide::Var const& x = args[0];
    Halide::Expr const idx = x * factor + offset;
    Halide::Expr const inRange = idx >= 0 && idx < inputLen;
    func(x) = Halide::select(inRange, inputFunc(idx), Halide::cast<T>(0));
  }
  return func;
}

REGISTER_OP(DOWNSAMPLE, handleDownsample);

// ============================================================================
// I/Q 交织辅助
// ============================================================================

/// IQ Pack：输出交织序列 [I0, Q0, I1, Q1, ...]
template <typename T>
Halide::Func handleIqPack(const dsl::detail::Node* node, OpContext<T>& ctx,
                          const std::vector<Halide::Var>& args) {
  auto i = ctx.buildFunc(dsl::Signal::fromNode(node->inputs.at(0)));
  auto q = ctx.buildFunc(dsl::Signal::fromNode(node->inputs.at(1)));

  Halide::Func func;
  if constexpr (IS_COMPLEX_V<T>) {
    Halide::Var const& c = args[0];
    Halide::Var const& x = args[1];
    Halide::Expr const idx = x / 2;
    func(c, x) = Halide::select((x % 2) == 0, i(c, idx), q(c, idx));
  } else {
    Halide::Var const& x = args[0];
    Halide::Expr const idx = x / 2;
    func(x) = Halide::select((x % 2) == 0, i(idx), q(idx));
  }
  return func;
}

REGISTER_OP(IQ_PACK, handleIqPack);

/// IQ I：out[k] = in[2k]
template <typename T>
Halide::Func handleIqI(const dsl::detail::Node* node, OpContext<T>& ctx,
                       const std::vector<Halide::Var>& args) {
  auto inputFunc = ctx.buildFunc(dsl::Signal::fromNode(node->inputs.at(0)));
  int const inputLen = static_cast<int>(node->inputs.at(0)->shape.length);

  Halide::Func func;
  if constexpr (IS_COMPLEX_V<T>) {
    Halide::Var const& c = args[0];
    Halide::Var const& x = args[1];
    Halide::Expr const idx = Halide::clamp(2 * x, 0, inputLen - 1);
    func(c, x) = inputFunc(c, idx);
  } else {
    Halide::Var const& x = args[0];
    Halide::Expr const idx = Halide::clamp(2 * x, 0, inputLen - 1);
    func(x) = inputFunc(idx);
  }
  return func;
}

REGISTER_OP(IQ_I, handleIqI);

/// IQ Q：out[k] = in[2k+1]
template <typename T>
Halide::Func handleIqQ(const dsl::detail::Node* node, OpContext<T>& ctx,
                       const std::vector<Halide::Var>& args) {
  auto inputFunc = ctx.buildFunc(dsl::Signal::fromNode(node->inputs.at(0)));
  int const inputLen = static_cast<int>(node->inputs.at(0)->shape.length);

  Halide::Func func;
  if constexpr (IS_COMPLEX_V<T>) {
    Halide::Var const& c = args[0];
    Halide::Var const& x = args[1];
    Halide::Expr const idx = Halide::clamp((2 * x) + 1, 0, inputLen - 1);
    func(c, x) = inputFunc(c, idx);
  } else {
    Halide::Var const& x = args[0];
    Halide::Expr const idx = Halide::clamp((2 * x) + 1, 0, inputLen - 1);
    func(x) = inputFunc(idx);
  }
  return func;
}

REGISTER_OP(IQ_Q, handleIqQ);

// 显式注册函数（供 Executor.cpp 调用，确保目标文件被链接）
void registerArithmeticHandlers() {
  // Handler 已通过 REGISTER_OP 自动注册
}

/// @}

}  // namespace prism::runtime
