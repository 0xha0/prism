/**
 * @file arithmetic_handlers.cpp
 * @ingroup runtime
 * @brief 基础代数算子 Handler 实现
 *
 * 包含加减乘除、缩放、取负、共轭、取模、上/下采样以及复数辅助操作的 Halide 实现
 * 支持 real32_t/real64_t 标量类型及其复数形式
 */

#include <Halide.h>

#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "prism/dsl/signal.h"
#include "prism/runtime/op_handler.h"
#include "prism/types.h"

namespace prism::runtime {

/// @addtogroup runtime
/// @{

// ============================================================================
// Add: 逐元素相加
// ============================================================================
/**
 * @brief Handle Add 算子
 *
 * $y[n] = a[n] + b[n]$
 *
 * 支持的操作数类型组合：
 * - Real + Real -> Real
 * - Complex + Complex -> Complex where $(a_r+ib_r) + (b_r+ib_i) = (a_r+b_r) +
 * i(a_i+b_i)$
 * - Mixed (Real+Complex) -> Complex (Real 视为虚部为 0)
 */
template <typename T>
Halide::Func handleAdd(const dsl::detail::Node* node, OpContext<T>& ctx,
                       const std::vector<Halide::Var>& args) {
  auto a = ctx.buildFunc(dsl::Signal::fromNode(node->inputs.at(0)));
  auto b = ctx.buildFunc(dsl::Signal::fromNode(node->inputs.at(1)));
  Halide::Func func;
  using ElemT = typename ToHalideType<T>::Type;
  bool const outputComplex = isComplexType(node->outputType);
  bool const isAComplex = isComplexType(node->inputs[0]->outputType);
  bool const isBComplex = isComplexType(node->inputs[1]->outputType);

  if (outputComplex) {
    Halide::Var const& c = args[0];
    Halide::Var const& x = args[1];
    Halide::Expr const zero = Halide::cast<ElemT>(0);
    Halide::Expr ar = zero;
    Halide::Expr ai = zero;
    Halide::Expr br = zero;
    Halide::Expr bi = zero;
    if (isAComplex) {
      ar = a(0, x);
      ai = a(1, x);
    } else {
      ar = a(x);
    }
    if (isBComplex) {
      br = b(0, x);
      bi = b(1, x);
    } else {
      br = b(x);
    }
    func(c, x) = Halide::mux(c, {ar + br, ai + bi});
  } else {
    Halide::Var const& x = args[0];
    func(x) = a(x) + b(x);
  }
  return func;
}

REGISTER_OP(ADD, handleAdd);

// ============================================================================
// Sub: 逐元素相减
// ============================================================================
/**
 * @brief Handle Sub 算子
 *
 * $y[n] = a[n] - b[n]$
 *
 * 支持 Real/Complex 及其混合运算
 */
template <typename T>
Halide::Func handleSub(const dsl::detail::Node* node, OpContext<T>& ctx,
                       const std::vector<Halide::Var>& args) {
  auto a = ctx.buildFunc(dsl::Signal::fromNode(node->inputs.at(0)));
  auto b = ctx.buildFunc(dsl::Signal::fromNode(node->inputs.at(1)));
  Halide::Func func;
  using ElemT = typename ToHalideType<T>::Type;
  bool const outputComplex = isComplexType(node->outputType);
  bool const isAComplex = isComplexType(node->inputs[0]->outputType);
  bool const isBComplex = isComplexType(node->inputs[1]->outputType);

  if (outputComplex) {
    Halide::Var const& c = args[0];
    Halide::Var const& x = args[1];
    Halide::Expr const zero = Halide::cast<ElemT>(0);
    Halide::Expr ar = zero;
    Halide::Expr ai = zero;
    Halide::Expr br = zero;
    Halide::Expr bi = zero;
    if (isAComplex) {
      ar = a(0, x);
      ai = a(1, x);
    } else {
      ar = a(x);
    }
    if (isBComplex) {
      br = b(0, x);
      bi = b(1, x);
    } else {
      br = b(x);
    }
    func(c, x) = Halide::mux(c, {ar - br, ai - bi});
  } else {
    Halide::Var const& x = args[0];
    func(x) = a(x) - b(x);
  }
  return func;
}

REGISTER_OP(SUB, handleSub);

// ============================================================================
// Mul: 逐元素相乘
// ============================================================================
/**
 * @brief Handle Mul 算子
 *
 * $y[n] = a[n] \times b[n]$
 *
 * 复数乘法公式：
 * $(a_r + j a_i) (b_r + j b_i) = (a_r b_r - a_i b_i) + j (a_r b_i + a_i b_r)$
 */
template <typename T>
Halide::Func handleMul(const dsl::detail::Node* node, OpContext<T>& ctx,
                       const std::vector<Halide::Var>& args) {
  auto a = ctx.buildFunc(dsl::Signal::fromNode(node->inputs.at(0)));
  auto b = ctx.buildFunc(dsl::Signal::fromNode(node->inputs.at(1)));
  Halide::Func func;

  using ElemT = typename ToHalideType<T>::Type;
  bool const outputComplex = isComplexType(node->outputType);
  bool const isAComplex = isComplexType(node->inputs[0]->outputType);
  bool const isBComplex = isComplexType(node->inputs[1]->outputType);

  if (outputComplex) {
    Halide::Var const& c = args[0];
    Halide::Var const& x = args[1];
    Halide::Expr const zero = Halide::cast<ElemT>(0);
    Halide::Expr ar = zero;
    Halide::Expr ai = zero;
    Halide::Expr br = zero;
    Halide::Expr bi = zero;
    if (isAComplex) {
      ar = a(0, x);
      ai = a(1, x);
    } else {
      ar = a(x);
    }
    if (isBComplex) {
      br = b(0, x);
      bi = b(1, x);
    } else {
      br = b(x);
    }
    func(c, x) = Halide::mux(c, {ar * br - ai * bi, ar * bi + ai * br});
  } else {
    Halide::Var const& x = args[0];
    func(x) = a(x) * b(x);
  }
  return func;
}

REGISTER_OP(MUL, handleMul);

// ============================================================================
// Div: 逐元素相除
// ============================================================================
/**
 * @brief Handle Div 算子
 *
 * $y[n] = a[n] / b[n]$
 *
 * 复数除法公式：
 * $\frac{a_r + j a_i}{b_r + j b_i} = \frac{(a_r b_r + a_i b_i) + j (a_i b_r -
 * a_r b_i)}{b_r^2 + b_i^2}$
 */
template <typename T>
Halide::Func handleDiv(const dsl::detail::Node* node, OpContext<T>& ctx,
                       const std::vector<Halide::Var>& args) {
  auto a = ctx.buildFunc(dsl::Signal::fromNode(node->inputs.at(0)));
  auto b = ctx.buildFunc(dsl::Signal::fromNode(node->inputs.at(1)));
  Halide::Func func;

  using ElemT = typename ToHalideType<T>::Type;
  bool const outputComplex = isComplexType(node->outputType);
  bool const isAComplex = isComplexType(node->inputs[0]->outputType);
  bool const isBComplex = isComplexType(node->inputs[1]->outputType);

  if (outputComplex) {
    Halide::Var const& c = args[0];
    Halide::Var const& x = args[1];
    Halide::Expr const zero = Halide::cast<ElemT>(0);
    Halide::Expr ar = zero;
    Halide::Expr ai = zero;
    Halide::Expr br = zero;
    Halide::Expr bi = zero;
    if (isAComplex) {
      ar = a(0, x);
      ai = a(1, x);
    } else {
      ar = a(x);
    }
    if (isBComplex) {
      br = b(0, x);
      bi = b(1, x);
    } else {
      br = b(x);
    }
    Halide::Expr const denom = br * br + bi * bi;
    func(c, x) = Halide::mux(c, {(ar * br + ai * bi) / denom, (ai * br - ar * bi) / denom});
  } else {
    Halide::Var const& x = args[0];
    func(x) = a(x) / b(x);
  }
  return func;
}

REGISTER_OP(DIV, handleDiv);

// ============================================================================
// Scale: 乘以标量
// ============================================================================
/**
 * @brief Handle Scale 算子
 *
 * $y[n] = a[n] \times s$
 *
 * s 由 node->param 提供，可以是实数或复数标量
 */
template <typename T>
Halide::Func handleScale(const dsl::detail::Node* node, OpContext<T>& ctx,
                         const std::vector<Halide::Var>& args) {
  auto a = ctx.buildFunc(dsl::Signal::fromNode(node->inputs.at(0)));
  Halide::Func func;
  using ElemT = typename ToHalideType<T>::Type;
  bool const outputComplex = isComplexType(node->outputType);
  bool const inputComplex = isComplexType(node->inputs[0]->outputType);

  bool scalarComplex = false;
  auto const scalar = std::visit(
      [&](auto&& arg) -> std::pair<ElemT, ElemT> {
        using ArgT = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<ArgT, real32_t> || std::is_same_v<ArgT, real64_t>) {
          return {static_cast<ElemT>(arg), static_cast<ElemT>(0)};
        }
        if constexpr (IS_COMPLEX_V<ArgT>) {
          scalarComplex = true;
          return {static_cast<ElemT>(arg.real()), static_cast<ElemT>(arg.imag())};
        }
        throw std::runtime_error("Scale: param must be real/complex scalar");
        return {static_cast<ElemT>(0), static_cast<ElemT>(0)};
      },
      node->param);

  bool const expectedComplex = inputComplex || scalarComplex;
  if (outputComplex != expectedComplex) {
    throw std::runtime_error("Scale: output type must match promoted input/scalar type");
  }

  Halide::Expr const valR = Halide::Expr(scalar.first);
  Halide::Expr const valI = Halide::Expr(scalar.second);

  if (outputComplex) {
    Halide::Var const& c = args[0];
    Halide::Var const& x = args[1];
    Halide::Expr ar = Halide::cast<ElemT>(0);
    Halide::Expr ai = Halide::cast<ElemT>(0);
    if (inputComplex) {
      ar = a(0, x);
      ai = a(1, x);
    } else {
      ar = a(x);
    }
    // (Ar + j Ai) * (Vr + j Vi)
    func(c, x) = Halide::mux(c, {ar * valR - ai * valI, ar * valI + ai * valR});
  } else {
    Halide::Var const& x = args[0];
    func(x) = a(x) * valR;
  }
  return func;
}

REGISTER_OP(SCALE, handleScale);

// ============================================================================
// Negate: 逐元素取负
// ============================================================================
/**
 * @brief Handle Negate 算子
 *
 * $y[n] = -a[n]$
 */
template <typename T>
Halide::Func handleNeg(const dsl::detail::Node* node, OpContext<T>& ctx,
                       const std::vector<Halide::Var>& args) {
  auto a = ctx.buildFunc(dsl::Signal::fromNode(node->inputs.at(0)));
  Halide::Func func;
  bool const outputComplex = isComplexType(node->outputType);
  bool const inputComplex = isComplexType(node->inputs[0]->outputType);
  if (outputComplex != inputComplex) {
    throw std::runtime_error("Negate: output type must match input type");
  }

  if (outputComplex) {
    Halide::Var const& c = args[0];
    Halide::Var const& x = args[1];
    func(c, x) = -a(c, x);
  } else {
    Halide::Var const& x = args[0];
    func(x) = -a(x);
  }
  return func;
}

REGISTER_OP(NEG, handleNeg);

// ============================================================================
// Conjugate: 取复共轭
// ============================================================================
/**
 * @brief Handle Conj 算子
 *
 * $y[n] = a_r[n] - j a_i[n]$
 */
template <typename T>
Halide::Func handleConj(const dsl::detail::Node* node, OpContext<T>& ctx,
                        const std::vector<Halide::Var>& args) {
  auto a = ctx.buildFunc(dsl::Signal::fromNode(node->inputs.at(0)));
  Halide::Func func;
  bool const outputComplex = isComplexType(node->outputType);
  bool const inputComplex = isComplexType(node->inputs[0]->outputType);
  if (outputComplex != inputComplex) {
    throw std::runtime_error("Conj: output type must match input type");
  }

  if (outputComplex) {
    Halide::Var const& c = args[0];
    Halide::Var const& x = args[1];
    func(c, x) = Halide::mux(c, {a(0, x), -a(1, x)});
  } else {
    Halide::Var const& x = args[0];
    func(x) = a(x);
  }
  return func;
}

REGISTER_OP(CONJ, handleConj);

// ============================================================================
// Abs: 取绝对值（模值）
// ============================================================================
/**
 * @brief Handle Abs 算子
 *
 * $y[n] = |a[n]| = \sqrt{a_r^2 + a_i^2}$
 *
 * 输出始终为实数
 */
template <typename T>
Halide::Func handleAbs(const dsl::detail::Node* node, OpContext<T>& ctx,
                       const std::vector<Halide::Var>& args) {
  auto a = ctx.buildFunc(dsl::Signal::fromNode(node->inputs.at(0)));
  Halide::Func func;
  bool const outputComplex = isComplexType(node->outputType);
  bool const inputComplex = isComplexType(node->inputs[0]->outputType);

  if (outputComplex) {
    throw std::runtime_error("Abs: output type must be real");
  }

  Halide::Var const& x = args[0];
  if (inputComplex) {
    Halide::Expr const ar = a(0, x);
    Halide::Expr const ai = a(1, x);
    func(x) = Halide::hypot(ar, ai);
  } else {
    func(x) = Halide::abs(a(x));
  }
  return func;
}

REGISTER_OP(ABS, handleAbs);

// ============================================================================
// Upsample: 插零上采样
// ============================================================================
/**
 * @brief Handle Upsample 算子
 *
 * $y[n] = x[n/L]$ 如果 $n \equiv \text{offset} \pmod L$，否则 $0$
 *
 * L 由 param 参数指定（int64_t 或 vector<int64_t> {L, offset}）
 */
template <typename T>
Halide::Func handleUpsample(const dsl::detail::Node* node, OpContext<T>& ctx,
                            const std::vector<Halide::Var>& args) {
  auto inputFunc = ctx.buildFunc(dsl::Signal::fromNode(node->inputs.at(0)));
  int const inputLen = static_cast<int>(node->inputs.at(0)->shape.length);
  int factor = 0;
  int offset = 0;
  if (auto const* scalar = std::get_if<int64_t>(&node->param)) {
    factor = static_cast<int>(*scalar);
  } else if (auto const* params = std::get_if<std::vector<int64_t>>(&node->param)) {
    if (params->empty()) {
      throw std::runtime_error("Upsample factor missing");
    }
    factor = static_cast<int>(params->at(0));
    if (params->size() > 1) {
      offset = static_cast<int>(params->at(1));
    }
  } else {
    throw std::runtime_error("Upsample param must be int64 or vector<int64>");
  }
  if (factor <= 0) {
    throw std::runtime_error("Upsample factor must be > 0");
  }
  if (offset < 0 || offset >= factor) {
    throw std::runtime_error("Upsample offset must be in [0, factor)");
  }

  Halide::Func func;
  using ElemT = typename ToHalideType<T>::Type;
  bool const outputComplex = isComplexType(node->outputType);
  bool const inputComplex = isComplexType(node->inputs[0]->outputType);
  if (outputComplex != inputComplex) {
    throw std::runtime_error("Upsample: output type must match input type");
  }

  if (outputComplex) {
    Halide::Var const& c = args[0];
    Halide::Var const& x = args[1];
    Halide::Expr const t = x - offset;
    Halide::Expr const rawIdx = t / factor;
    Halide::Expr const idx = Halide::clamp(rawIdx, 0, inputLen - 1);
    Halide::Expr const inRange = t >= 0 && rawIdx < inputLen;
    Halide::Expr const isHit = (t % factor) == 0;
    func(c, x) = Halide::select(isHit && inRange, inputFunc(c, idx), Halide::cast<ElemT>(0));
  } else {
    Halide::Var const& x = args[0];
    Halide::Expr const t = x - offset;
    Halide::Expr const rawIdx = t / factor;
    Halide::Expr const idx = Halide::clamp(rawIdx, 0, inputLen - 1);
    Halide::Expr const inRange = t >= 0 && rawIdx < inputLen;
    Halide::Expr const isHit = (t % factor) == 0;
    func(x) = Halide::select(isHit && inRange, inputFunc(idx), Halide::cast<ElemT>(0));
  }
  return func;
}

REGISTER_OP(UPSAMPLE, handleUpsample);

// ============================================================================
// Downsample: 抽取下采样
// ============================================================================
/**
 * @brief Handle Downsample 算子
 *
 * $y[n] = x[n \times M + \text{offset}]$
 *
 * M 由 param 参数指定（vector<int64_t> {M, offset}）
 */
template <typename T>
Halide::Func handleDownsample(const dsl::detail::Node* node, OpContext<T>& ctx,
                              const std::vector<Halide::Var>& args) {
  auto inputFunc = ctx.buildFunc(dsl::Signal::fromNode(node->inputs.at(0)));
  int const inputLen = static_cast<int>(node->inputs.at(0)->shape.length);
  const auto& params = std::get<std::vector<int64_t>>(node->param);
  int const factor = static_cast<int>(params[0]);
  int const offset = static_cast<int>(params[1]);
  if (factor <= 0) {
    throw std::runtime_error("Downsample factor must be > 0");
  }
  if (offset < 0) {
    throw std::runtime_error("Downsample offset must be >= 0");
  }

  Halide::Func func;
  using ElemT = typename ToHalideType<T>::Type;
  bool const outputComplex = isComplexType(node->outputType);
  bool const inputComplex = isComplexType(node->inputs[0]->outputType);

  if (outputComplex) {
    Halide::Var const& c = args[0];
    Halide::Var const& x = args[1];
    Halide::Expr const idx = x * factor + offset;
    Halide::Expr const inRange = idx >= 0 && idx < inputLen;
    Halide::Expr val = Halide::cast<ElemT>(0);
    if (inputComplex) {
      val = inputFunc(c, idx);
    } else {
      val = Halide::select(c == 0, inputFunc(idx), Halide::cast<ElemT>(0));
    }
    func(c, x) = Halide::select(inRange, val, Halide::cast<ElemT>(0));
  } else {
    Halide::Var const& x = args[0];
    Halide::Expr const idx = x * factor + offset;
    Halide::Expr const inRange = idx >= 0 && idx < inputLen;
    func(x) = Halide::select(inRange, inputFunc(idx), Halide::cast<ElemT>(0));
  }
  return func;
}

REGISTER_OP(DOWNSAMPLE, handleDownsample);

// ============================================================================
// Complex 辅助
// ============================================================================

/**
 * @brief Handle I/Q Pack
 *
 * 将两个实数信号（或复数信号）组合成一个复数信号
 * out[n] = i[n] + j * q[n]
 */
template <typename T>
Halide::Func handleIqPack(const dsl::detail::Node* node, OpContext<T>& ctx,
                          const std::vector<Halide::Var>& args) {
  auto i = ctx.buildFunc(dsl::Signal::fromNode(node->inputs.at(0)));
  auto q = ctx.buildFunc(dsl::Signal::fromNode(node->inputs.at(1)));

  Halide::Func func;
  bool const outputComplex = isComplexType(node->outputType);
  bool const iComplex = isComplexType(node->inputs[0]->outputType);
  bool const qComplex = isComplexType(node->inputs[1]->outputType);

  if (outputComplex) {
    Halide::Var const& c = args[0];
    Halide::Var const& x = args[1];
    Halide::Expr const iVal = iComplex ? i(0, x) : i(x);
    Halide::Expr const qVal = qComplex ? q(0, x) : q(x);
    func(c, x) = Halide::select(c == 0, iVal, qVal);
  } else {
    // 降级为只取 I 路（不常见，仅防御式编程）
    Halide::Var const& x = args[0];
    func(x) = iComplex ? i(0, x) : i(x);
  }
  return func;
}

REGISTER_OP(CPLX_PACK, handleIqPack);

/**
 * @brief Handle Real Extraction
 *
 * 提取复数信号的实部
 */
template <typename T>
Halide::Func handleReal(const dsl::detail::Node* node, OpContext<T>& ctx,
                        const std::vector<Halide::Var>& args) {
  auto inputFunc = ctx.buildFunc(dsl::Signal::fromNode(node->inputs.at(0)));
  int const inputLen = static_cast<int>(node->inputs.at(0)->shape.length);
  bool const inputComplex = isComplexType(node->inputs[0]->outputType);

  Halide::Func func;
  Halide::Var const& x = args[0];
  Halide::Expr const idx = Halide::clamp(x, 0, inputLen - 1);
  if (inputComplex) {
    func(x) = inputFunc(0, idx);
  } else {
    func(x) = inputFunc(idx);
  }
  return func;
}

REGISTER_OP(REAL, handleReal);

/**
 * @brief Handle Imag Extraction
 *
 * 提取复数信号的虚部
 */
template <typename T>
Halide::Func handleImag(const dsl::detail::Node* node, OpContext<T>& ctx,
                        const std::vector<Halide::Var>& args) {
  auto inputFunc = ctx.buildFunc(dsl::Signal::fromNode(node->inputs.at(0)));
  int const inputLen = static_cast<int>(node->inputs.at(0)->shape.length);
  bool const inputComplex = isComplexType(node->inputs[0]->outputType);

  Halide::Func func;
  Halide::Var const& x = args[0];
  Halide::Expr const idx = Halide::clamp(x, 0, inputLen - 1);
  if (inputComplex) {
    func(x) = inputFunc(1, idx);
  } else {
    // 实数信号虚部为 0
    // 注意：需确保类型匹配
    using ET = typename ToHalideType<T>::Type;
    func(x) = Halide::cast<ET>(0);
  }
  return func;
}

REGISTER_OP(IMAG, handleImag);

/**
 * @brief 显式注册函数
 *
 * 供 Executor.cpp 调用，确保目标文件被链接器包含
 * 实际 Handler 已通过 REGISTER_OP 宏自动注册
 */
void registerArithmeticHandlers() {
  // Handler 已通过 REGISTER_OP 自动注册
}

/// @}

}  // namespace prism::runtime
