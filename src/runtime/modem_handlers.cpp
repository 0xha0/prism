/**
 * @file modem_handlers.cpp
 * @ingroup runtime
 * @brief 调制解调相关 Handler：Mixer、QAMMap、QAMDemap
 */

#include <Halide.h>

#include <cmath>
#include <vector>

#include "prism/dsl/signal.h"
#include "prism/runtime/op_handler.h"
#include "prism/types.h"

namespace prism::runtime {

/// @addtogroup runtime
/// @{

// ============================================================================
// 混频器
// ============================================================================

/// 混频：乘以旋转相位
template <typename T>
Halide::Func handleMixer(const dsl::detail::Node* node, OpContext<T>& ctx,
                         const std::vector<Halide::Var>& args) {
  auto a = ctx.buildFunc(dsl::Signal::fromNode(node->inputs.at(0)));
  real64_t const freq = node->freq;
  real64_t const sampleRate = node->sampleRate;
  Halide::Expr const omega = Halide::Expr(2.0 * M_PI_VAL * freq / sampleRate);

  Halide::Func func;

  if constexpr (IS_COMPLEX_V<T>) {
    Halide::Var const& c = args[0];
    Halide::Var const& x = args[1];

    // a(x) * exp(j * omega * n)
    // exp(...) = cos(...) + j sin(...)
    // (ar + j ai)(cos + j sin) = (ar cos - ai sin) + j (ar sin + ai cos)

    Halide::Expr const ar = a(0, x);
    Halide::Expr const ai = a(1, x);
    Halide::Expr const cosV =
        Halide::cos(Halide::cast<typename ToHalideType<T>::Type>(x) *
                    Halide::cast<typename ToHalideType<T>::Type>(omega));
    Halide::Expr const sinV =
        Halide::sin(Halide::cast<typename ToHalideType<T>::Type>(x) *
                    Halide::cast<typename ToHalideType<T>::Type>(omega));

    func(c, x) =
        Halide::select(c == 0, ar * cosV - ai * sinV, ar * sinV + ai * cosV);
  } else {
    Halide::Var const& x = args[0];
    func(x) =
        a(x) * Halide::cast<T>(Halide::cos(Halide::cast<real32_t>(x) *
                                           Halide::cast<real32_t>(omega)));
  }
  return func;
}

REGISTER_OP(MIXER, handleMixer);

// ============================================================================
// QAMMap（符号映射）—— 完整 I/Q 交织输出
// ============================================================================

/**
 * @brief QAM 映射：符号索引 -> I/Q 交织输出
 *
 * 输入：长度为 N 的符号索引序列
 * 输出：长度为 2N 的 I/Q 交织序列 [I0, Q0, I1, Q1, ...]
 *
 * 映射规则（以 QAM16 为例）：
 * - symbol % sqrt(M) -> I 索引
 * - symbol / sqrt(M) -> Q 索引
 * - 索引归一化到 [-1, 1]
 */
template <typename T>
Halide::Func handleQamMap(const dsl::detail::Node* node, OpContext<T>& ctx,
                          const std::vector<Halide::Var>& args) {
  auto inputFunc = ctx.buildFunc(dsl::Signal::fromNode(node->inputs.at(0)));
  int const order = node->modOrder;
  int const sqrtOrder = static_cast<int>(std::sqrt(order));
  int const inputLen = static_cast<int>(node->inputs.at(0)->shape.length);

  Halide::Func func;

  if constexpr (IS_COMPLEX_V<T>) {
    // Output is Complex(c, x)
    // Input is Symbol Index (Real usually)
    Halide::Var const& c = args[0];
    Halide::Var const& x = args[1];  // x is symbol index effectively

    // Input is scalar signal of symbols.
    // inputFunc(x) gives symbol.
    // We map symbol -> (I, Q)

    Halide::Expr symbol = Halide::cast<int>(inputFunc(Halide::clamp(
        x, 0,
        inputLen - 1)));  // Assuming input is not args dependent in same way?
    // Wait, if input is Real, InputFunc is 1D.
    // But OpContext<T> expects T. If T is Complex, input must be Complex?
    // No, Mapping takes Real Symbol -> Complex I/Q.
    // The Input "Symbol" signal should explicitly be Real?
    // But DSL type propagation might set Mapping output as Complex.
    // So T=Complex.
    // Input Signal might be Real. But `ctx.buildFunc` returns `Func` for Input.
    // If Input is Real, its Func is 1D `f(args[0])`.
    // If we are in Complex context, `args` has 2 vars.
    // Using `inputFunc({x})` (constructing vector) is needed?
    // `buildFunc` returns a `Func`. We can call it with any args we want.
    // If input is Real, it expects 1 arg.

    // THIS IS A KEY PROBLEM: `inputFunc` dimensionality depends on Input Signal
    // Type, not T (Output Type). `OpContext<T>` is for Output Type? No,
    // OpContext is generic. But `buildFunc` implementation in `executor.cpp`
    // uses `T` for everything? Step 767: `buildFunc<T>` calls `handler(node,
    // ctx, args)`. It recursively compiles inputs with `buildFunc<T>`. This
    // implicitly forces the entire pipeline to be type `T`.
    //
    // So if T=Complex, Input is also treated as Complex?
    // If Symbol Input is treated as Complex, then `inputFunc` expects (c, x).
    // But Symbol is scalar. Real part = symbol, Imag part = 0?
    // Or we just look at Real part?

    std::vector<Halide::Var> const inputArgs = {args[0],
                                                args[1]};  // Pass same args
    // But if symbol, we want real part `c=0`.
    // Actually `inputFunc(0, x)` is safer if input is complex.

    symbol = Halide::cast<int>(inputFunc(0, Halide::clamp(x, 0, inputLen - 1)));

    Halide::Expr const iIdx = symbol % sqrtOrder;
    Halide::Expr const qIdx = symbol / sqrtOrder;

    Halide::Expr const normFactor =
        Halide::cast<typename ToHalideType<T>::Type>(sqrtOrder - 1);
    Halide::Expr const iVal =
        (Halide::cast<typename ToHalideType<T>::Type>(2 * iIdx) - normFactor) /
        normFactor;
    Halide::Expr const qVal =
        (Halide::cast<typename ToHalideType<T>::Type>(2 * qIdx) - normFactor) /
        normFactor;

    func(c, x) = Halide::select(c == 0, iVal, qVal);

  } else {
    // T=Real. Output is Interleaved Real.
    Halide::Var const& x = args[0];
    Halide::Expr const symIdx = x / 2;
    Halide::Expr const isI = (x % 2) == 0;

    Halide::Expr const symbol =
        Halide::cast<int>(inputFunc(Halide::clamp(symIdx, 0, inputLen - 1)));

    Halide::Expr const iIdx = symbol % sqrtOrder;
    Halide::Expr const qIdx = symbol / sqrtOrder;

    Halide::Expr const normFactor = Halide::cast<T>(sqrtOrder - 1);
    Halide::Expr const iVal =
        (Halide::cast<T>(2 * iIdx) - normFactor) / normFactor;
    Halide::Expr const qVal =
        (Halide::cast<T>(2 * qIdx) - normFactor) / normFactor;

    func(x) = Halide::select(isI, iVal, qVal);
  }

  return func;
}

REGISTER_OP(QAM_MAP, handleQamMap);

// ============================================================================
// QAMDemap（符号解映射）—— I/Q 交织输入
// ============================================================================

/**
 * @brief QAM 解映射：I/Q 交织输入 -> 符号索引
 *
 * 输入：长度为 2N 的 I/Q 交织序列 [I0, Q0, I1, Q1, ...]
 * 输出：长度为 N 的符号索引序列
 *
 * 硬判决规则：
 * - 量化 I/Q 到最近的星座点
 * - symbol = q_idx * sqrt(M) + i_idx
 */
template <typename T>
Halide::Func handleQamDemap(const dsl::detail::Node* node, OpContext<T>& ctx,
                            const std::vector<Halide::Var>& args) {
  auto inputFunc = ctx.buildFunc(dsl::Signal::fromNode(node->inputs.at(0)));
  int const order = node->modOrder;
  int const sqrtOrder = static_cast<int>(std::sqrt(order));
  int const inputLen = static_cast<int>(node->inputs.at(0)->shape.length);

  Halide::Func func;

  if constexpr (IS_COMPLEX_V<T>) {
    // T=Complex (Output is Complex? No, Output is Symbol Index (Real)).
    // This is the Friction again.
    // If T=Complex, `func` must be 2D `(c, x)`.
    // But Symbol Index is 1D scalar.
    // So we return `(symbol, 0)`?

    Halide::Var const& c = args[0];
    Halide::Var const& x = args[1];

    // Input is Complex Signal (I, Q)
    Halide::Expr const iVal = inputFunc(0, Halide::clamp(x, 0, inputLen - 1));
    Halide::Expr const qVal = inputFunc(1, Halide::clamp(x, 0, inputLen - 1));

    Halide::Expr const normFactor =
        Halide::cast<typename ToHalideType<T>::Type>(sqrtOrder - 1);
    Halide::Expr const iScaled = iVal * normFactor;
    Halide::Expr const qScaled = qVal * normFactor;

    Halide::Expr const iIdx = Halide::clamp(
        Halide::cast<int>(
            Halide::round((iScaled + normFactor) /
                          Halide::cast<typename ToHalideType<T>::Type>(2))),
        0, sqrtOrder - 1);
    Halide::Expr const qIdx = Halide::clamp(
        Halide::cast<int>(
            Halide::round((qScaled + normFactor) /
                          Halide::cast<typename ToHalideType<T>::Type>(2))),
        0, sqrtOrder - 1);

    Halide::Expr const symbol = qIdx * sqrtOrder + iIdx;

    func(c, x) = Halide::select(
        c == 0, Halide::cast<typename ToHalideType<T>::Type>(symbol),
        Halide::cast<typename ToHalideType<T>::Type>(0));

  } else {
    Halide::Var const& x = args[0];
    Halide::Expr const iPos = 2 * x;
    Halide::Expr const qPos = 2 * x + 1;

    Halide::Expr const iVal = inputFunc(Halide::clamp(iPos, 0, inputLen - 1));
    Halide::Expr const qVal = inputFunc(Halide::clamp(qPos, 0, inputLen - 1));

    Halide::Expr const normFactor = Halide::cast<T>(sqrtOrder - 1);
    Halide::Expr const iScaled = iVal * normFactor;
    Halide::Expr const qScaled = qVal * normFactor;

    Halide::Expr const iIdx =
        Halide::clamp(Halide::cast<int>(Halide::round((iScaled + normFactor) /
                                                      Halide::cast<T>(2))),
                      0, sqrtOrder - 1);
    Halide::Expr const qIdx =
        Halide::clamp(Halide::cast<int>(Halide::round((qScaled + normFactor) /
                                                      Halide::cast<T>(2))),
                      0, sqrtOrder - 1);

    Halide::Expr const symbol = qIdx * sqrtOrder + iIdx;

    func(x) = Halide::cast<T>(symbol);
  }
  return func;
}

REGISTER_OP(QAM_DEMAP, handleQamDemap);

// ============================================================================
// PSKMap（PSK 符号映射）
// ============================================================================

/**
 * @brief PSK 映射：符号索引 -> I/Q 交织输出
 *
 * 将符号索引 k 映射到单位圆上的点：
 * - θ = 2π * k / M + π/M（偏移 π/M 避免轴上点）
 * - I = cos(θ), Q = sin(θ)
 *
 * 输出格式：[I0, Q0, I1, Q1, ...]
 */
template <typename T>
Halide::Func handlePskMap(const dsl::detail::Node* node, OpContext<T>& ctx,
                          const std::vector<Halide::Var>& args) {
  auto inputFunc = ctx.buildFunc(dsl::Signal::fromNode(node->inputs.at(0)));
  int const order = node->modOrder;
  int const inputLen = static_cast<int>(node->inputs.at(0)->shape.length);

  Halide::Func func;
  auto const twoPiVal = static_cast<real32_t>(2.0 * M_PI_VAL);
  auto const phaseOffsetVal = static_cast<real32_t>(M_PI_VAL / order);

  if constexpr (IS_COMPLEX_V<T>) {
    Halide::Var const& c = args[0];
    Halide::Var const& x = args[1];

    Halide::Expr const symbol =
        Halide::cast<int>(inputFunc(0, Halide::clamp(x, 0, inputLen - 1)));
    Halide::Expr const twoPi =
        Halide::cast<typename ToHalideType<T>::Type>(twoPiVal);
    Halide::Expr const phaseOffset =
        Halide::cast<typename ToHalideType<T>::Type>(phaseOffsetVal);
    Halide::Expr const theta =
        (twoPi * Halide::cast<typename ToHalideType<T>::Type>(symbol) /
         Halide::cast<typename ToHalideType<T>::Type>(order)) +
        phaseOffset;

    Halide::Expr const iVal = Halide::cos(theta);
    Halide::Expr const qVal = Halide::sin(theta);

    func(c, x) = Halide::select(c == 0, iVal, qVal);
  } else {
    Halide::Var const& x = args[0];
    Halide::Expr const symIdx = x / 2;
    Halide::Expr const isI = (x % 2) == 0;

    Halide::Expr const symbol =
        Halide::cast<int>(inputFunc(Halide::clamp(symIdx, 0, inputLen - 1)));
    Halide::Expr const twoPi = Halide::cast<T>(twoPiVal);
    Halide::Expr const phaseOffset = Halide::cast<T>(phaseOffsetVal);
    Halide::Expr const theta =
        (twoPi * Halide::cast<T>(symbol) / Halide::cast<T>(order)) +
        phaseOffset;

    Halide::Expr const iVal = Halide::cos(theta);
    Halide::Expr const qVal = Halide::sin(theta);

    func(x) = Halide::select(isI, iVal, qVal);
  }
  return func;
}

REGISTER_OP(PSK_MAP, handlePskMap);

// ============================================================================
// PSKDemap（PSK 符号解映射）
// ============================================================================

/**
 * @brief PSK 解映射：I/Q 交织输入 -> 符号索引
 *
 * 根据 I/Q 计算相位并量化到最近的星座点：
 * - θ = atan2(Q, I)
 * - k = round((θ - π/M) * M / 2π) mod M
 */
template <typename T>
Halide::Func handlePskDemap(const dsl::detail::Node* node, OpContext<T>& ctx,
                            const std::vector<Halide::Var>& args) {
  auto inputFunc = ctx.buildFunc(dsl::Signal::fromNode(node->inputs.at(0)));
  int const order = node->modOrder;
  int const inputLen = static_cast<int>(node->inputs.at(0)->shape.length);

  Halide::Func func;
  auto const twoPiVal = static_cast<real32_t>(2.0 * M_PI_VAL);
  auto const phaseOffsetVal = static_cast<real32_t>(M_PI_VAL / order);

  if constexpr (IS_COMPLEX_V<T>) {
    Halide::Var const& c = args[0];
    Halide::Var const& x = args[1];

    Halide::Expr const iVal = inputFunc(0, Halide::clamp(x, 0, inputLen - 1));
    Halide::Expr const qVal = inputFunc(1, Halide::clamp(x, 0, inputLen - 1));

    Halide::Expr const theta = Halide::atan2(qVal, iVal);

    Halide::Expr const twoPi =
        Halide::cast<typename ToHalideType<T>::Type>(twoPiVal);
    Halide::Expr const thetaNorm = Halide::select(
        theta < Halide::cast<typename ToHalideType<T>::Type>(0.0F),
        theta + twoPi, theta);

    Halide::Expr const phaseOffset =
        Halide::cast<typename ToHalideType<T>::Type>(phaseOffsetVal);
    Halide::Expr const kNum =
        (thetaNorm - phaseOffset) *
        Halide::cast<typename ToHalideType<T>::Type>(order) / twoPi;
    Halide::Expr k = Halide::cast<int>(Halide::round(kNum));

    k = Halide::select(k < 0, k + order, k);
    k = Halide::select(k >= order, k - order, k);

    func(c, x) =
        Halide::select(c == 0, Halide::cast<typename ToHalideType<T>::Type>(k),
                       Halide::cast<typename ToHalideType<T>::Type>(0));

  } else {
    Halide::Var const& x = args[0];
    Halide::Expr const iPos = 2 * x;
    Halide::Expr const qPos = 2 * x + 1;

    Halide::Expr const iVal = inputFunc(Halide::clamp(iPos, 0, inputLen - 1));
    Halide::Expr const qVal = inputFunc(Halide::clamp(qPos, 0, inputLen - 1));

    Halide::Expr const theta = Halide::atan2(qVal, iVal);
    Halide::Expr const twoPi = Halide::cast<T>(twoPiVal);
    Halide::Expr const thetaNorm =
        Halide::select(theta < Halide::cast<T>(0.0F), theta + twoPi, theta);

    Halide::Expr const phaseOffset = Halide::cast<T>(phaseOffsetVal);
    Halide::Expr const kNum =
        (thetaNorm - phaseOffset) * Halide::cast<T>(order) / twoPi;
    Halide::Expr k = Halide::cast<int>(Halide::round(kNum));

    k = Halide::select(k < 0, k + order, k);
    k = Halide::select(k >= order, k - order, k);

    func(x) = Halide::cast<T>(k);
  }
  return func;
}

REGISTER_OP(PSK_DEMAP, handlePskDemap);

// 显式注册函数（供 Executor.cpp 调用，确保目标文件被链接）
void registerModemHandlers() {
  // Handler 已通过 REGISTER_OP 自动注册
}

/// @}

}  // namespace prism::runtime
