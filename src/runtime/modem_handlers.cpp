/**
 * @file modem_handlers.cpp
 * @ingroup runtime
 * @brief 调制解调相关 Handler 实现：Mixer、QAMMap、QAMDemap
 *
 * 实现了混频、QAM/PSK 星座图映射及硬判决解映射的 Halide 转换
 */

#include <Halide.h>

#include <cmath>
#include <cstdint>
#include <vector>

#include "prism/dsl/signal.h"
#include "prism/runtime/op_handler.h"
#include "prism/types.h"

namespace prism::runtime {

/// @addtogroup runtime
/// @{

/**
 * @brief Handle Mixer 算子
 *
 * $y[n] = x[n] \cdot e^{j \omega n}$
 *
 * 其中 $\omega = 2\pi \cdot \frac{f}{f_s}$
 * 实现了复数相乘的展开形式：
 * $ (Ar + jAi)(\cos + j\sin) = (Ar\cos - Ai\sin) + j(Ar\sin + Ai\cos) $
 */
template <typename T>
Halide::Func handleMixer(const dsl::detail::Node* node, OpContext<T>& ctx,
                         const std::vector<Halide::Var>& args) {
  auto a = ctx.buildFunc(dsl::Signal::fromNode(node->inputs.at(0)));
  const auto& params = std::get<std::vector<real64_t>>(node->param);
  real64_t const freq = params[0];
  real64_t const sampleRate = params[1];
  Halide::Expr const omega = Halide::Expr(2.0 * M_PI_VAL * freq / sampleRate);

  Halide::Func func;

  using ElemT = typename ToHalideType<T>::Type;
  bool const outputComplex = isComplexType(node->outputType);
  bool const inputComplex = isComplexType(node->inputs[0]->outputType);

  if (outputComplex) {
    Halide::Var const& c = args[0];
    Halide::Var const& x = args[1];

    // a(x) * exp(j * omega * n)
    // exp(...) = cos(...) + j sin(...)
    // (ar + j ai)(cos + j sin) = (ar cos - ai sin) + j (ar sin + ai cos)
    Halide::Expr const ar = inputComplex ? a(0, x) : a(x);
    Halide::Expr const ai = inputComplex ? a(1, x) : Halide::cast<ElemT>(0);
    Halide::Expr const cosV = Halide::cos(Halide::cast<ElemT>(x) * Halide::cast<ElemT>(omega));
    Halide::Expr const sinV = Halide::sin(Halide::cast<ElemT>(x) * Halide::cast<ElemT>(omega));

    func(c, x) = Halide::mux(c, {ar * cosV - ai * sinV, ar * sinV + ai * cosV});
  } else {
    Halide::Var const& x = args[0];
    Halide::Expr const cosV = Halide::cos(Halide::cast<ElemT>(x) * Halide::cast<ElemT>(omega));
    func(x) = a(x) * Halide::cast<ElemT>(cosV);
  }
  return func;
}

REGISTER_OP(MIXER, handleMixer);

/**
 * @brief Handle QAM Map 算子
 *
 * 符号索引 -> I/Q 星座点
 *
 * 输入：Real 符号索引 [N]
 * 输出：Complex I/Q 星座点 [N] (Halide 布局: (c, x))
 *
 * 映射规则（矩形 QAM）：
 * - $I_{idx} = symbol \% \sqrt{M}$
 * - $Q_{idx} = symbol / \sqrt{M}$
 * - 归一化到 [-1, 1]: $Val = \frac{2 \cdot idx - (\sqrt{M}-1)}{\sqrt{M}-1}$
 */
template <typename T>
Halide::Func handleQamMap(const dsl::detail::Node* node, OpContext<T>& ctx,
                          const std::vector<Halide::Var>& args) {
  auto inputFunc = ctx.buildFunc(dsl::Signal::fromNode(node->inputs.at(0)));
  int const order = std::get<int32_t>(node->param);
  int const sqrtOrder = static_cast<int>(std::sqrt(order));
  int const inputLen = static_cast<int>(node->inputs.at(0)->shape.length);

  Halide::Func func;

  // qamMap: Real 输入 → Complex 输出
  // 输出使用 Complex (c, x) 布局
  Halide::Var const& c = args[0];
  Halide::Var const& x = args[1];

  // 输入是 Real (1D)
  Halide::Expr const symbol = Halide::cast<int>(inputFunc(Halide::clamp(x, 0, inputLen - 1)));

  Halide::Expr const iIdx = symbol % sqrtOrder;
  Halide::Expr const qIdx = symbol / sqrtOrder;

  using RealT = typename ToHalideType<T>::Type;
  Halide::Expr const normFactor = Halide::cast<RealT>(sqrtOrder - 1);
  Halide::Expr const iVal = (Halide::cast<RealT>(2 * iIdx) - normFactor) / normFactor;
  Halide::Expr const qVal = (Halide::cast<RealT>(2 * qIdx) - normFactor) / normFactor;

  func(c, x) = Halide::mux(c, {iVal, qVal});

  return func;
}

REGISTER_OP(QAM_MAP, handleQamMap);

/**
 * @brief Handle QAM Demap 算子
 *
 * I/Q 星座点 -> 符号索引（硬判决）
 *
 * 输入：Complex I/Q 星座点 [N]
 * 输出：Real 符号索引 [N]
 *
 * 判决规则：
 * - 将 I/Q 值反归一化并四舍五入到最近的整数索引
 * - $symbol = Q_{idx} \cdot \sqrt{M} + I_{idx}$
 */
template <typename T>
Halide::Func handleQamDemap(const dsl::detail::Node* node, OpContext<T>& ctx,
                            const std::vector<Halide::Var>& args) {
  auto inputFunc = ctx.buildFunc(dsl::Signal::fromNode(node->inputs.at(0)));
  int const order = std::get<int32_t>(node->param);
  int const sqrtOrder = static_cast<int>(std::sqrt(order));
  int const inputLen = static_cast<int>(node->inputs.at(0)->shape.length);

  Halide::Func func;

  // qamDemap: Complex 输入 → Real 输出
  // 输出使用 Real (x) 布局
  Halide::Var const& x = args[0];

  Halide::Expr iVal;
  Halide::Expr qVal;
  if (node->inputs.size() == 2) {
    // Dual Real Input (I, Q)
    auto iFunc = ctx.buildFunc(dsl::Signal::fromNode(node->inputs.at(0)));
    auto qFunc = ctx.buildFunc(dsl::Signal::fromNode(node->inputs.at(1)));
    iVal = iFunc(Halide::clamp(x, 0, inputLen - 1));
    qVal = qFunc(Halide::clamp(x, 0, inputLen - 1));
  } else {
    // Single Complex Input
    Halide::Expr const iValRaw = inputFunc(0, Halide::clamp(x, 0, inputLen - 1));
    Halide::Expr const qValRaw = inputFunc(1, Halide::clamp(x, 0, inputLen - 1));
    iVal = iValRaw;
    qVal = qValRaw;
  }

  using RealT = typename ToHalideType<T>::Type;
  Halide::Expr const normFactor = Halide::cast<RealT>(sqrtOrder - 1);
  Halide::Expr const iScaled = iVal * normFactor;
  Halide::Expr const qScaled = qVal * normFactor;

  Halide::Expr const iIdx = Halide::clamp(
      Halide::cast<int>(Halide::round((iScaled + normFactor) / Halide::cast<RealT>(2))), 0,
      sqrtOrder - 1);
  Halide::Expr const qIdx = Halide::clamp(
      Halide::cast<int>(Halide::round((qScaled + normFactor) / Halide::cast<RealT>(2))), 0,
      sqrtOrder - 1);

  Halide::Expr const symbol = qIdx * sqrtOrder + iIdx;

  func(x) = Halide::cast<RealT>(symbol);

  return func;
}

REGISTER_OP(QAM_DEMAP, handleQamDemap);

/**
 * @brief Handle PSK Map 算子
 *
 * 符号索引 -> I/Q 星座点
 *
 * 输入：Real 符号索引 [N]
 * 输出：Complex I/Q 星座点 [N]
 *
 * 映射规则：
 * - $\theta = \frac{2\pi k}{M} + \frac{\pi}{M}$
 * - $I = \cos(\theta), Q = \sin(\theta)$
 */
template <typename T>
Halide::Func handlePskMap(const dsl::detail::Node* node, OpContext<T>& ctx,
                          const std::vector<Halide::Var>& args) {
  auto inputFunc = ctx.buildFunc(dsl::Signal::fromNode(node->inputs.at(0)));
  int const order = std::get<int32_t>(node->param);
  int const inputLen = static_cast<int>(node->inputs.at(0)->shape.length);

  Halide::Func func;
  auto const twoPiVal = static_cast<real32_t>(2.0 * M_PI_VAL);
  auto const phaseOffsetVal = static_cast<real32_t>(M_PI_VAL / order);

  using RealT = typename ToHalideType<T>::Type;

  // pskMap: Real 输入 → Complex 输出
  Halide::Var const& c = args[0];
  Halide::Var const& x = args[1];

  // 输入是 Real (1D)
  Halide::Expr const symbol = Halide::cast<int>(inputFunc(Halide::clamp(x, 0, inputLen - 1)));

  Halide::Expr const twoPi = Halide::cast<RealT>(twoPiVal);
  Halide::Expr const phaseOffset = Halide::cast<RealT>(phaseOffsetVal);
  Halide::Expr const theta =
      (twoPi * Halide::cast<RealT>(symbol) / Halide::cast<RealT>(order)) + phaseOffset;

  Halide::Expr const iVal = Halide::cos(theta);
  Halide::Expr const qVal = Halide::sin(theta);

  func(c, x) = Halide::mux(c, {iVal, qVal});

  return func;
}

REGISTER_OP(PSK_MAP, handlePskMap);

/**
 * @brief Handle PSK Demap 算子
 *
 * I/Q 星座点 -> 符号索引（硬判决）
 *
 * 判决规则：
 * - 计算相位 $\theta = \text{atan2}(Q, I)$
 * - 反解 k，并处理周期性
 */
template <typename T>
Halide::Func handlePskDemap(const dsl::detail::Node* node, OpContext<T>& ctx,
                            const std::vector<Halide::Var>& args) {
  auto inputFunc = ctx.buildFunc(dsl::Signal::fromNode(node->inputs.at(0)));
  int const order = std::get<int32_t>(node->param);
  int const inputLen = static_cast<int>(node->inputs.at(0)->shape.length);

  Halide::Func func;
  auto const twoPiVal = static_cast<real32_t>(2.0 * M_PI_VAL);
  auto const phaseOffsetVal = static_cast<real32_t>(M_PI_VAL / order);

  using RealT = typename ToHalideType<T>::Type;

  // pskDemap: Complex 输入 → Real 输出
  Halide::Var const& x = args[0];

  Halide::Expr iVal;
  Halide::Expr qVal;
  if (node->inputs.size() == 2) {
    // Dual Real Input
    auto iFunc = ctx.buildFunc(dsl::Signal::fromNode(node->inputs.at(0)));
    auto qFunc = ctx.buildFunc(dsl::Signal::fromNode(node->inputs.at(1)));
    iVal = iFunc(Halide::clamp(x, 0, inputLen - 1));
    qVal = qFunc(Halide::clamp(x, 0, inputLen - 1));
  } else {
    // Single Complex Input
    iVal = inputFunc(0, Halide::clamp(x, 0, inputLen - 1));
    qVal = inputFunc(1, Halide::clamp(x, 0, inputLen - 1));
  }

  Halide::Expr const theta = Halide::atan2(qVal, iVal);
  Halide::Expr const twoPi = Halide::cast<RealT>(twoPiVal);
  Halide::Expr const thetaNorm =
      Halide::select(theta < Halide::cast<RealT>(0.0F), theta + twoPi, theta);

  Halide::Expr const phaseOffset = Halide::cast<RealT>(phaseOffsetVal);
  Halide::Expr const kNum = (thetaNorm - phaseOffset) * Halide::cast<RealT>(order) / twoPi;
  Halide::Expr k = Halide::cast<int>(Halide::round(kNum));

  k = Halide::select(k < 0, k + order, k);
  k = Halide::select(k >= order, k - order, k);

  func(x) = Halide::cast<RealT>(k);

  return func;
}

REGISTER_OP(PSK_DEMAP, handlePskDemap);

// 显式注册函数（供 Executor.cpp 调用，确保目标文件被链接）
void registerModemHandlers() {
  // Handler 已通过 REGISTER_OP 自动注册
}

/// @}

}  // namespace prism::runtime
