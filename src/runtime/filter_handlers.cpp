/**
 * @file filter_handlers.cpp
 * @ingroup runtime
 * @brief 滤波相关 Handler：FIR/IIR/MovingAverage/Median
 *
 * FIR/IIR 系数由用户提供，IIR 使用脉冲响应截断近似实现。
 */

#include <Halide.h>

#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#include "prism/dsl/signal.h"
#include "prism/runtime/op_handler.h"
#include "prism/types.h"

namespace prism::runtime {

/// @addtogroup runtime
/// @{

namespace {
// NOLINTBEGIN(readability-identifier-naming)
constexpr double kIirA0Tol64 = 1e-12;
constexpr float kIirA0Tol32 = 1e-6F;
constexpr int kIirImpulseMinLen = 128;
constexpr int kIirImpulseScale = 4;
constexpr int kMedianWindowSmall = 3;
constexpr int kMedianWindowMedium = 5;
constexpr int kMedianWindowLarge = 7;
//NOLINTEND(readability-identifier-naming)

template <typename T>
void validateIirTaps(const std::vector<T>& b, const std::vector<T>& a, T tol) {
  if (b.empty()) {
    throw std::invalid_argument("IIR requires non-empty b taps");
  }
  if (!a.empty() && std::abs(a[0] - static_cast<T>(1)) > tol) {
    throw std::invalid_argument("IIR requires a[0] == 1.0");
  }
}

template <typename T>
bool isFirTaps(const std::vector<T>& a, T tol) {
  if (a.empty()) {
    return true;
  }
  if (a.size() == 1) {
    return std::abs(a[0] - static_cast<T>(1)) < tol;
  }
  return false;
}

template <typename T>
void fillIirImpulseResponse(Halide::Buffer<T>& coeffsBuf,
                            const std::vector<T>& b,
                            const std::vector<T>& a, bool isFir) {
  int const coeffsLen = coeffsBuf.dim(0).extent();
  int const bLen = static_cast<int>(b.size());
  if (isFir || a.empty()) {
    for (int i = 0; i < coeffsLen; ++i) {
      coeffsBuf(i) = (i < bLen) ? b[i] : static_cast<T>(0);
    }
    return;
  }

  int const aLen = static_cast<int>(a.size());
  std::vector<T> state(std::max(bLen, aLen), static_cast<T>(0));
  for (int n = 0; n < coeffsLen; ++n) {
    T const xN = (n == 0) ? static_cast<T>(1) : static_cast<T>(0);
    T const yN = (b[0] * xN) + state[0];

    for (int i = 0; i < static_cast<int>(state.size()) - 1; ++i) {
      T const bTerm = (i + 1 < bLen) ? b[i + 1] * xN : static_cast<T>(0);
      T const aTerm = (i + 1 < aLen) ? a[i + 1] * yN : static_cast<T>(0);
      state[i] = bTerm - aTerm + state[i + 1];
    }
    state.back() = static_cast<T>(0);
    coeffsBuf(n) = yN;
  }
}

template <typename T, typename CoeffT>
void defineIirFunc(Halide::Func& func, const Halide::Func& inputFunc,
                   const Halide::Buffer<CoeffT>& coeffsBuf, int inputLen,
                   BndryMode boundary, const std::vector<Halide::Var>& args,
                   int coeffsLen) {
  Halide::RDom const r(0, coeffsLen);
  if constexpr (IS_COMPLEX_V<T>) {
    Halide::Var const& c = args[0];
    Halide::Var const& x = args[1];
    auto sample = [&](const Halide::Expr& idx) { return inputFunc(c, idx); };
    auto const zero = Halide::cast<typename ToHalideType<T>::Type>(0);
    func(c, x) = Halide::sum(
        sampleWithBoundary(sample, x - r, inputLen, boundary, zero) *
        Halide::cast<typename ToHalideType<T>::Type>(coeffsBuf(r)));
  } else {
    Halide::Var const& x = args[0];
    auto sample = [&](const Halide::Expr& idx) { return inputFunc(idx); };
    auto const zero = Halide::cast<T>(0);
    func(x) = Halide::sum(
        sampleWithBoundary(sample, x - r, inputLen, boundary, zero) *
        Halide::cast<T>(coeffsBuf(r)));
  }
}

Halide::Expr reflectIndex(Halide::Expr idx, int len) {
  if (len <= 1) {
    return {0};
  }
  Halide::Expr const period = Halide::Expr(2 * (len - 1));
  Halide::Expr const x = Halide::abs(std::move(idx)) % period;
  return Halide::select(x > (len - 1), period - x, x);
}

template <typename SampleFunc>
Halide::Expr sampleWithBoundary(const SampleFunc& sample,
                                const Halide::Expr& idx, int len,
                                BndryMode mode, Halide::Expr zero) {
  if (len <= 0) {
    return zero;
  }
  Halide::Expr const safe = Halide::clamp(idx, 0, len - 1);
  Halide::Expr const inRange = idx >= 0 && idx < len;

  switch (mode) {
    case BndryMode::ZERO:
      return Halide::select(inRange, sample(safe), zero);
    case BndryMode::CLAMP:
      return sample(safe);
    case BndryMode::REFLECT:
      return sample(reflectIndex(idx, len));
    default:
      return Halide::select(inRange, sample(safe), zero);
  }
}
}  // namespace

// ============================================================================
// FIR 滤波器
// ============================================================================

/// FIR 滤波器：直接卷积
template <typename T>
Halide::Func handleFir(const dsl::detail::Node* node, OpContext<T>& ctx,
                       const std::vector<Halide::Var>& args) {
  auto inputFunc = ctx.buildFunc(dsl::Signal::fromNode(node->inputs.at(0)));

  const bool useDouble = !node->taps64.empty();
  int const tapsLen = useDouble ? static_cast<int>(node->taps64.size())
                                : static_cast<int>(node->taps.size());

  if (tapsLen == 0) {
    Halide::Func func;
    func(args) = inputFunc(args);
    return func;
  }

  int const inputLen = static_cast<int>(node->inputs.at(0)->shape.length);

  // Use Helper to abstract Tap Access?
  // Easier to just branch logic or templated buffer?
  // Halide::Buffer can be Type-erased or we use If-Else.

  Halide::Func func;
  Halide::RDom const r(0, tapsLen);

  if (useDouble) {
    auto it = ctx.firTapsCache64.find(node->taps64);
    if (it == ctx.firTapsCache64.end()) {
      Halide::Buffer<double> tapsBuf(tapsLen);
      for (int i = 0; i < tapsLen; ++i) tapsBuf(i) = node->taps64[i];
      it = ctx.firTapsCache64.emplace(node->taps64, tapsBuf).first;
    }
    auto& tapsBuf = it->second;

    if constexpr (IS_COMPLEX_V<T>) {
      Halide::Var c = args[0];
      Halide::Var const& x = args[1];
      auto sample = [&](const Halide::Expr& idx) { return inputFunc(c, idx); };
      auto const zero = Halide::cast<typename ToHalideType<T>::Type>(0);
      func(c, x) = Halide::sum(
          sampleWithBoundary(sample, x - r, inputLen, node->boundary, zero) *
          Halide::cast<typename ToHalideType<T>::Type>(tapsBuf(r)));
    } else {
      Halide::Var const& x = args[0];
      auto sample = [&](const Halide::Expr& idx) { return inputFunc(idx); };
      auto const zero = Halide::cast<T>(0);
      func(x) = Halide::sum(
          sampleWithBoundary(sample, x - r, inputLen, node->boundary, zero) *
          Halide::cast<T>(tapsBuf(r)));
    }
  } else {
    auto it = ctx.firTapsCache32.find(node->taps);
    if (it == ctx.firTapsCache32.end()) {
      Halide::Buffer<float> tapsBuf(tapsLen);
      for (int i = 0; i < tapsLen; ++i) tapsBuf(i) = node->taps[i];
      it = ctx.firTapsCache32.emplace(node->taps, tapsBuf).first;
    }
    auto& tapsBuf = it->second;

    if constexpr (IS_COMPLEX_V<T>) {
      Halide::Var c = args[0];
      Halide::Var const& x = args[1];
      auto sample = [&](const Halide::Expr& idx) { return inputFunc(c, idx); };
      auto const zero = Halide::cast<typename ToHalideType<T>::Type>(0);
      func(c, x) = Halide::sum(
          sampleWithBoundary(sample, x - r, inputLen, node->boundary, zero) *
          Halide::cast<typename ToHalideType<T>::Type>(tapsBuf(r)));
    } else {
      Halide::Var const& x = args[0];
      auto sample = [&](const Halide::Expr& idx) { return inputFunc(idx); };
      auto const zero = Halide::cast<T>(0);
      func(x) = Halide::sum(
          sampleWithBoundary(sample, x - r, inputLen, node->boundary, zero) *
          Halide::cast<T>(tapsBuf(r)));
    }
  }

  return func;
}

REGISTER_OP(FIR, handleFir);

// ============================================================================
// IIR 滤波器（脉冲响应截断近似）
// ============================================================================

/// IIR 滤波器：使用预计算脉冲响应实现
/// IIR 滤波器：使用预计算脉冲响应实现
template <typename T>
Halide::Func handleIir(const dsl::detail::Node* node, OpContext<T>& ctx,
                       const std::vector<Halide::Var>& args) {
  auto inputFunc = ctx.buildFunc(dsl::Signal::fromNode(node->inputs.at(0)));
  int const inputLen = static_cast<int>(node->inputs.at(0)->shape.length);

  bool const useDouble = !node->taps64.empty() || !node->tapsA64.empty();
  int const bLen = useDouble ? static_cast<int>(node->taps64.size())
                             : static_cast<int>(node->taps.size());
  int const aLen = useDouble ? static_cast<int>(node->tapsA64.size())
                             : static_cast<int>(node->tapsA.size());

  if (useDouble) {
    validateIirTaps(node->taps64, node->tapsA64, kIirA0Tol64);
  } else {
    validateIirTaps(node->taps, node->tapsA, kIirA0Tol32);
  }
  // TODO: 增加极点稳定性检测与更清晰的错误提示。

  // 检查是否为纯 FIR（a 仅有 a[0]=1）
  bool const isFir = useDouble ? isFirTaps(node->tapsA64, kIirA0Tol64)
                               : isFirTaps(node->tapsA, kIirA0Tol32);

  // Common Buffer / Func creation logic...
  // Since we need to compute IR, we need compile-time (or run-time) IR
  // calculation. Here we do it at graph-build time (DSL -> Halide).

  Halide::Func func;
  // TODO: 采用能量阈值自适应截断长度，避免过短或过长。
  int const coeffsLen = std::max(
      {kIirImpulseMinLen, kIirImpulseScale * std::max(bLen, aLen), inputLen});

  if (useDouble) {
    auto key = std::make_tuple(coeffsLen, node->taps64, node->tapsA64);
    auto it = ctx.iirCoeffCache64.find(key);
    if (it == ctx.iirCoeffCache64.end()) {
      Halide::Buffer<double> coeffsBuf(coeffsLen);
      fillIirImpulseResponse(coeffsBuf, node->taps64, node->tapsA64, isFir);
      it = ctx.iirCoeffCache64.emplace(std::move(key), coeffsBuf).first;
    }
    auto& coeffsBuf = it->second;
    defineIirFunc<T>(func, inputFunc, coeffsBuf, inputLen, node->boundary, args,
                     coeffsLen);
  } else {
    auto key = std::make_tuple(coeffsLen, node->taps, node->tapsA);
    auto it = ctx.iirCoeffCache32.find(key);
    if (it == ctx.iirCoeffCache32.end()) {
      Halide::Buffer<float> coeffsBuf(coeffsLen);
      fillIirImpulseResponse(coeffsBuf, node->taps, node->tapsA, isFir);
      it = ctx.iirCoeffCache32.emplace(std::move(key), coeffsBuf).first;
    }
    auto& coeffsBuf = it->second;
    defineIirFunc<T>(func, inputFunc, coeffsBuf, inputLen, node->boundary, args,
                     coeffsLen);
  }

  return func;
}

REGISTER_OP(IIR, handleIir);

// ============================================================================
// 移动平均
// ============================================================================

/// 移动平均：窗口内求和并归一化
template <typename T>
Halide::Func handleMovingAverage(const dsl::detail::Node* node,
                                 OpContext<T>& ctx,
                                 const std::vector<Halide::Var>& args) {
  auto inputFunc = ctx.buildFunc(dsl::Signal::fromNode(node->inputs.at(0)));
  int const window = static_cast<int>(node->scalar);
  int const len = static_cast<int>(node->inputs.at(0)->shape.length);

  Halide::Func func;
  if (window <= 0) {
    throw std::invalid_argument("MovingAverage window must be positive");
  }
  if (window == 1) {
    func(args) = inputFunc(args);
    return func;
  }

  constexpr int kSmallWindow = 31;
  // TODO: 根据目标设备自动选择阈值（CPU/GPU 行为不同）。
  bool const useDirectSum =
      (window <= kSmallWindow) || (node->boundary != BndryMode::ZERO);

  if constexpr (IS_COMPLEX_V<T>) {
    Halide::Var c = args[0];
    Halide::Var const& x = args[1];
    auto sample = [&](const Halide::Expr& idx) { return inputFunc(c, idx); };
    auto const zero = Halide::cast<typename ToHalideType<T>::Type>(0);

    if (useDirectSum) {
      Halide::RDom const r(0, window);
      func(c, x) = Halide::sum(sampleWithBoundary(sample, x - r, len,
                                                  node->boundary, zero)) /
                   Halide::cast<typename ToHalideType<T>::Type>(window);
    } else {
      Halide::Func const prefix("ma_prefix");
      prefix(c, x) = zero;
      prefix(c, 0) = sampleWithBoundary(sample, 0, len, node->boundary, zero);
      Halide::RDom const r(1, len - 1);
      prefix(c, r) = prefix(c, r - 1) +
                     sampleWithBoundary(sample, r, len, node->boundary, zero);

      Halide::Expr const left =
          Halide::select(x >= window, prefix(c, x - window), zero);
      func(c, x) = (prefix(c, x) - left) /
                   Halide::cast<typename ToHalideType<T>::Type>(window);
    }
  } else {
    Halide::Var const& x = args[0];
    auto sample = [&](const Halide::Expr& idx) { return inputFunc(idx); };
    auto const zero = Halide::cast<T>(0);

    if (useDirectSum) {
      Halide::RDom const r(0, window);
      func(x) = Halide::sum(sampleWithBoundary(sample, x - r, len,
                                               node->boundary, zero)) /
                Halide::cast<T>(window);
    } else {
      Halide::Func const prefix("ma_prefix");
      prefix(x) = zero;
      prefix(0) = sampleWithBoundary(sample, 0, len, node->boundary, zero);
      Halide::RDom const r(1, len - 1);
      prefix(r) = prefix(r - 1) +
                  sampleWithBoundary(sample, r, len, node->boundary, zero);

      Halide::Expr const left =
          Halide::select(x >= window, prefix(x - window), zero);
      func(x) = (prefix(x) - left) / Halide::cast<T>(window);
    }
  }
  return func;
}

REGISTER_OP(MOVING_AVERAGE, handleMovingAverage);

// ============================================================================
// 中值滤波
// ============================================================================

/// 中值滤波：仅支持 3/5/7 窗口，使用排序网络求中值
template <typename T>
Halide::Func handleMedian(const dsl::detail::Node* node, OpContext<T>& ctx,
                          const std::vector<Halide::Var>& args) {
  auto inputFunc = ctx.buildFunc(dsl::Signal::fromNode(node->inputs.at(0)));
  int const window = static_cast<int>(node->scalar);
  int const len = static_cast<int>(node->inputs.at(0)->shape.length);

  Halide::Func func;

  if (window <= 0 || (window % 2) == 0) {
    throw std::invalid_argument("Median window must be positive odd");
  }
  if (window != kMedianWindowSmall && window != kMedianWindowMedium &&
      window != kMedianWindowLarge) {
    throw std::invalid_argument("Median window only supports 3/5/7");
  }

  // Helper for generic 1D/2D
  auto getVal = [&](const std::vector<Halide::Var>& idxs, int offset) {
    if constexpr (IS_COMPLEX_V<T>) {
      Halide::Expr const idx = idxs[1] + offset;
      auto sample = [&](const Halide::Expr& x) {
        return inputFunc(idxs[0], x);
      };
      auto const zero = Halide::cast<typename ToHalideType<T>::Type>(0);
      return sampleWithBoundary(sample, idx, len, node->boundary, zero);
    } else {
      Halide::Expr const idx = idxs[0] + offset;
      auto sample = [&](const Halide::Expr& x) { return inputFunc(x); };
      auto const zero = Halide::cast<T>(0);
      return sampleWithBoundary(sample, idx, len, node->boundary, zero);
    }
  };

  int const radius = window / 2;
  std::vector<Halide::Expr> values;
  values.reserve(window);
  for (int offset = -radius; offset <= radius; ++offset) {
    values.push_back(getVal(args, offset));
  }

  auto compareSwap = [](Halide::Expr& a, Halide::Expr& b) {
    Halide::Expr const lo = Halide::min(a, b);
    Halide::Expr const hi = Halide::max(a, b);
    a = lo;
    b = hi;
  };

  // 使用排序网络求中值（窗口限定 3/5/7）
  for (size_t i = 0; i < values.size(); ++i) {
    for (size_t j = 0; j + 1 < values.size() - i; ++j) {
      compareSwap(values[j], values[j + 1]);
    }
  }

  func(args) = values[values.size() / 2];

  return func;
}

REGISTER_OP(MEDIAN, handleMedian);

// 显式注册函数
void registerFilterHandlers() {}

/// @}

}  // namespace prism::runtime
