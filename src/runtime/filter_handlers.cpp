/**
 * @file filter_handlers.cpp
 * @ingroup runtime
 * @brief 滤波相关 Handler 实现：FIR/MovingAverage/Median
 *
 * 包含通用 FIR
 * 滤波（支持对称系数优化）、移动平均（支持前缀和优化）以及中值滤波（排序网络）的
 * Halide 实现
 */

#include <Halide.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "prism/dsl/signal.h"
#include "prism/runtime/op_handler.h"
#include "prism/types.h"

namespace prism::runtime {

/// @addtogroup runtime
/// @{

/**
 * @brief FIR 滤波器系数对称性类型
 */
enum class FirSymmetry : std::uint8_t {
  NONE,       ///< 无对称性
  EVEN,       ///< 偶对称 (symmetric, h[n] = h[M-1-n])
  ODD,        ///< 奇对称 (antisymmetric, h[n] = -h[M-1-n])
  CONJ_EVEN,  ///< 共轭偶对称 (Hermitian symmetric, h[n] = conj(h[M-1-n]))
  CONJ_ODD    ///< 共轭奇对称 (Hermitian antisymmetric, h[n] = -conj(h[M-1-n]))
};

/**
 * @brief 获取对称性检测的容差 epsilon
 *
 * @tparam T 数据类型
 * @return constexpr T 容差值
 */
template <typename T>
constexpr T symmetryEps() {
  if constexpr (std::is_same_v<T, float>) {
    return static_cast<T>(1e-5F);  // NOLINT
  }
  return static_cast<T>(1e-12F);  // NOLINT
}

/**
 * @brief 标量近似相等检查
 *
 * @tparam T 标量类型
 * @param a 值 a
 * @param b 值 b
 * @return true 如果 abs(a-b) <= eps
 * @return false 否则
 */
template <typename T>
inline bool approxEqualScalar(T a, T b) {
  return std::abs(a - b) <= symmetryEps<T>();
}

/**
 * @brief 标量近似零检查
 *
 * @tparam T 标量类型
 * @param a 值 a
 * @return true 如果 abs(a) <= eps
 * @return false 否则
 */
template <typename T>
inline bool approxZeroScalar(T a) {
  return std::abs(a) <= symmetryEps<T>();
}

/**
 * @brief 复数近似相等检查
 *
 * 分别检查实部和虚部
 *
 * @tparam T 基础标量类型
 * @param a 复数 a
 * @param b 复数 b
 * @return true 如果实部和虚部均近似相等
 * @return false 否则
 */
template <typename T>
inline bool approxEqualComplex(const std::complex<T>& a, const std::complex<T>& b) {
  return approxEqualScalar(a.real(), b.real()) && approxEqualScalar(a.imag(), b.imag());
}

/**
 * @brief 复数近似零检查
 *
 * 分别检查实部和虚部是否近似为零
 *
 * @tparam T 基础标量类型
 * @param a 复数 a
 * @return true 如果实部和虚部均近似为零
 * @return false 否则
 */
template <typename T>
inline bool approxZeroComplex(const std::complex<T>& a) {
  return approxZeroScalar(a.real()) && approxZeroScalar(a.imag());
}

// ----------------------------------------------------------------------------
// 对称性检测辅助函数 (Symmetry Detection Helpers)
// ----------------------------------------------------------------------------

template <typename T>
void updateSymmetryFlags(const T& a, const T& b, bool& even, bool& odd, bool& conjEven,
                         bool& conjOdd) {
  if (even && !approxEqualComplex(a, b)) even = false;
  if (odd && !approxEqualComplex(a, -b)) odd = false;
  if (conjEven && !approxEqualComplex(a, std::conj(b))) conjEven = false;
  if (conjOdd && !approxEqualComplex(a, -std::conj(b))) conjOdd = false;
}

template <typename TapT>
FirSymmetry detectFirSymmetryComplex(const std::vector<TapT>& taps) {
  int const len = static_cast<int>(taps.size());
  bool even = true;
  bool odd = true;
  bool conjEven = true;
  bool conjOdd = true;
  for (int i = 0; i < len / 2; ++i) {
    updateSymmetryFlags(taps[i], taps[len - 1 - i], even, odd, conjEven, conjOdd);
    if (!even && !odd && !conjEven && !conjOdd) {
      return FirSymmetry::NONE;
    }
  }
  if ((len % 2) == 1) {
    auto const& center = taps[len / 2];
    if (!approxZeroComplex(center)) {
      odd = false;
    }
    if (!approxZeroScalar(center.imag())) {
      conjEven = false;
    }
    if (!approxZeroScalar(center.real())) {
      conjOdd = false;
    }
  }

  if (even) return FirSymmetry::EVEN;
  if (odd) return FirSymmetry::ODD;
  if (conjEven) return FirSymmetry::CONJ_EVEN;
  if (conjOdd) return FirSymmetry::CONJ_ODD;
  return FirSymmetry::NONE;
}

template <typename TapT>
FirSymmetry detectFirSymmetryReal(const std::vector<TapT>& taps) {
  int const len = static_cast<int>(taps.size());
  bool even = true;
  bool odd = true;
  for (int i = 0; i < len / 2; ++i) {
    auto const& a = taps[i];
    auto const& b = taps[len - 1 - i];
    if (even && !approxEqualScalar(a, b)) {
      even = false;
    }
    if (odd && !approxEqualScalar(a, -b)) {
      odd = false;
    }
    if (!even && !odd) {
      return FirSymmetry::NONE;
    }
  }
  if ((len % 2) == 1) {
    if (!approxZeroScalar(taps[len / 2])) {
      odd = false;
    }
  }
  if (even) return FirSymmetry::EVEN;
  if (odd) return FirSymmetry::ODD;
  return FirSymmetry::NONE;
}

// 自动检测 FIR 系数的对称性以优化计算量
template <typename TapT>
FirSymmetry detectFirSymmetry(const std::vector<TapT>& taps) {
  if (taps.size() <= 1) {
    return FirSymmetry::NONE;
  }
  if constexpr (prism::IS_COMPLEX_V<TapT>) {
    return detectFirSymmetryComplex(taps);
  } else {
    return detectFirSymmetryReal(taps);
  }
}

template <typename T>
Halide::Func buildFirPassThrough(const Halide::Func& inputFunc,
                                 const std::vector<Halide::Var>& args, bool inputComplex,
                                 bool outputComplex) {
  using ElemT = typename prism::ToHalideType<T>::Type;
  Halide::Func passthrough("fir_passthrough");
  if (outputComplex) {
    Halide::Var const& c = args[0];
    Halide::Var const& x = args[1];
    Halide::Expr val = Halide::cast<ElemT>(0);
    if (inputComplex) {
      val = inputFunc(c, x);
    } else {
      val = Halide::select(c == 0, inputFunc(x), Halide::cast<ElemT>(0));
    }
    passthrough(c, x) = val;
  } else {
    Halide::Var const& x = args[0];
    if (inputComplex) {
      passthrough(x) = inputFunc(0, x);
    } else {
      passthrough(x) = inputFunc(x);
    }
  }
  return passthrough;
}

template <typename T>
Halide::Func makeZeroFunc(bool isComplex) {
  using ET = typename prism::ToHalideType<T>::Type;
  Halide::Func func;
  if (isComplex) {
    Halide::Var const c("c");
    Halide::Var const x("x");
    func(c, x) = Halide::cast<ET>(0);
  } else {
    Halide::Var const x("x");
    func(x) = Halide::cast<ET>(0);
  }
  return func;
}

inline Halide::Region makeBounds(bool isComplex, int len) {
  Halide::Region bounds;
  if (isComplex) {
    bounds.emplace_back(Halide::Expr(0), Halide::Expr(2));
  }
  bounds.emplace_back(Halide::Expr(0), Halide::Expr(len));
  return bounds;
}

inline Halide::Target getScheduleTarget(bool useGpu) {
  Halide::Target target = Halide::get_host_target();
  if (!useGpu) {
    return target;
  }
#ifdef PRISM_GPU_BACKEND_Metal
  target.set_feature(Halide::Target::Metal);
#elif defined(PRISM_GPU_BACKEND_CUDA)
  target.set_feature(Halide::Target::CUDA);
#elif defined(PRISM_GPU_BACKEND_OpenCL)
  target.set_feature(Halide::Target::OpenCL);
#endif
  return target;
}

// 决策阈值：何时从直接求和切换到前缀和优化
constexpr int GPU_DIRECT_SUM_THRESHOLD = 64;

inline int getMovingAvgDirectSumThreshold(int len, bool useGpu, Halide::Type elemType) {
  if (useGpu) {
    return GPU_DIRECT_SUM_THRESHOLD;
  }

  Halide::Target const target = getScheduleTarget(false);
  int const vectorWidth = std::max(1, target.natural_vector_size(elemType));
  int const base = std::max(1, 16 * vectorWidth);
  return std::min(std::max(base, vectorWidth), std::max(1, len));
}

template <typename T>
Halide::Func makeBoundedInput(const Halide::Func& inputFunc, prism::BndryMode mode, int len,
                              bool isComplex) {
  using ET = typename prism::ToHalideType<T>::Type;
  if (len <= 0) {
    return makeZeroFunc<T>(isComplex);
  }
  if (mode == prism::BndryMode::REFLECT && len <= 1) {
    mode = prism::BndryMode::CLAMP;
  }

  Halide::Region const bounds = makeBounds(isComplex, len);
  Halide::Func bounded;
  switch (mode) {
    case prism::BndryMode::ZERO:
      bounded =
          Halide::BoundaryConditions::constant_exterior(inputFunc, Halide::cast<ET>(0), bounds);
      break;
    case prism::BndryMode::CLAMP:
      bounded = Halide::BoundaryConditions::repeat_edge(inputFunc, bounds);
      break;
    case prism::BndryMode::REFLECT:
      bounded = Halide::BoundaryConditions::mirror_interior(inputFunc, bounds);
      break;
    default:
      bounded =
          Halide::BoundaryConditions::constant_exterior(inputFunc, Halide::cast<ET>(0), bounds);
      break;
  }

  auto const args = bounded.args();
  if (!args.empty()) {
    bounded.fold_storage(args.back(), len);
  }
  return bounded;
}

// ----------------------------------------------------------------------------
// FIR Implementation Helpers
// ----------------------------------------------------------------------------

template <typename T, typename TapType>
Halide::Func buildSymmetricFir(
    const std::vector<Halide::Var>& args, const Halide::Func& bounded,
    const Halide::Buffer<
        typename prism::ToHalideType<typename std::vector<TapType>::value_type>::Type>& tapsBuf,
    int tapsLen, FirSymmetry symmetry, bool inputComplex, bool outputComplex) {
  using ElemT = typename prism::ToHalideType<T>::Type;
  Halide::Func func("fir_sym");
  int const halfLen = tapsLen / 2;
  Halide::RDom const rPair(0, halfLen);

  if (!outputComplex) {
    // Real Result
    if (symmetry == FirSymmetry::EVEN || symmetry == FirSymmetry::ODD) {
      Halide::Var const& x = args[0];
      Halide::Expr const idx1 = x - rPair;
      Halide::Expr const idx2 = x - (tapsLen - 1 - rPair);
      Halide::Expr const xr1 = bounded(idx1);
      Halide::Expr const xr2 = bounded(idx2);
      Halide::Expr const hr = Halide::cast<ElemT>(tapsBuf(rPair));
      Halide::Expr const pair = (symmetry == FirSymmetry::EVEN) ? (xr1 + xr2) : (xr1 - xr2);
      Halide::Expr sum = Halide::sum(hr * pair);
      if ((tapsLen % 2) == 1) {
        Halide::Expr const centerIdx = x - halfLen;
        sum += Halide::cast<ElemT>(Halide::Expr(tapsBuf(halfLen))) * bounded(centerIdx);
      }
      func(x) = sum;
      return func;
    }
    // Should not happen for Real output with Complex taps or other symmetries
    // if validated
    return {};
  }

  // Complex Result
  Halide::Var const& c = args[0];
  Halide::Var const& x = args[1];
  Halide::Expr const zero = Halide::cast<ElemT>(0);
  Halide::Expr const idx1 = x - rPair;
  Halide::Expr const idx2 = x - (tapsLen - 1 - rPair);

  Halide::Expr const xr1 = inputComplex ? bounded(0, idx1) : bounded(idx1);
  Halide::Expr const xi1 = inputComplex ? bounded(1, idx1) : zero;
  Halide::Expr const xr2 = inputComplex ? bounded(0, idx2) : bounded(idx2);
  Halide::Expr const xi2 = inputComplex ? bounded(1, idx2) : zero;

  Halide::Expr hr = zero;
  Halide::Expr hi = zero;  // NOLINT(performance-unnecessary-copy-initialization)
  if constexpr (prism::IS_COMPLEX_V<TapType>) {
    hr = Halide::cast<ElemT>(tapsBuf(0, rPair));
    hi = Halide::cast<ElemT>(tapsBuf(1, rPair));
  } else {
    hr = Halide::cast<ElemT>(tapsBuf(rPair));
  }

  Halide::Expr realTerm = zero;
  Halide::Expr imagTerm = zero;

  switch (symmetry) {
    case FirSymmetry::EVEN: {
      Halide::Expr const xrSum = xr1 + xr2;
      Halide::Expr const xiSum = xi1 + xi2;
      realTerm = hr * xrSum - hi * xiSum;
      imagTerm = hr * xiSum + hi * xrSum;
      break;
    }
    case FirSymmetry::ODD: {
      Halide::Expr const xrDiff = xr1 - xr2;
      Halide::Expr const xiDiff = xi1 - xi2;
      realTerm = hr * xrDiff - hi * xiDiff;
      imagTerm = hr * xiDiff + hi * xrDiff;
      break;
    }
    case FirSymmetry::CONJ_EVEN: {
      Halide::Expr const xrSum = xr1 + xr2;
      Halide::Expr const xiSum = xi1 + xi2;
      Halide::Expr const xrDiff = xr1 - xr2;
      Halide::Expr const xiDiff = xi1 - xi2;
      realTerm = hr * xrSum - hi * xiDiff;
      imagTerm = hr * xiSum + hi * xrDiff;
      break;
    }
    case FirSymmetry::CONJ_ODD: {
      Halide::Expr const xrSum = xr1 + xr2;
      Halide::Expr const xiSum = xi1 + xi2;
      Halide::Expr const xrDiff = xr1 - xr2;
      Halide::Expr const xiDiff = xi1 - xi2;
      realTerm = hr * xrDiff - hi * xiSum;
      imagTerm = hr * xiDiff + hi * xrSum;
      break;
    }
    default:
      break;
  }

  Halide::Expr realSum = Halide::sum(realTerm);
  Halide::Expr imagSum = Halide::sum(imagTerm);

  if ((tapsLen % 2) == 1) {
    Halide::Expr const centerIdx = x - halfLen;
    Halide::Expr const xrC = inputComplex ? bounded(0, centerIdx) : bounded(centerIdx);
    Halide::Expr const xiC = inputComplex ? bounded(1, centerIdx) : zero;
    Halide::Expr hrC = zero;
    Halide::Expr hiC = zero;  // NOLINT(performance-unnecessary-copy-initialization)
    if constexpr (prism::IS_COMPLEX_V<TapType>) {
      hrC = Halide::cast<ElemT>(Halide::Expr(tapsBuf(0, halfLen)));
      hiC = Halide::cast<ElemT>(Halide::Expr(tapsBuf(1, halfLen)));
    } else {
      hrC = Halide::cast<ElemT>(Halide::Expr(tapsBuf(halfLen)));
    }
    realSum += hrC * xrC - hiC * xiC;
    imagSum += hrC * xiC + hiC * xrC;
  }

  func(c, x) = Halide::mux(c, {realSum, imagSum});
  return func;
}

template <typename T, typename TapType>
Halide::Func buildGeneralFir(
    const std::vector<Halide::Var>& args, const Halide::Func& bounded,
    const Halide::Buffer<
        typename prism::ToHalideType<typename std::vector<TapType>::value_type>::Type>& tapsBuf,
    int tapsLen, bool inputComplex, bool outputComplex) {
  using ElemT = typename prism::ToHalideType<T>::Type;
  Halide::Func func("fir_general");
  Halide::RDom const r(0, tapsLen);
  bool const isTapsComplex = prism::IS_COMPLEX_V<TapType>;

  if (outputComplex) {
    Halide::Var const& c = args[0];
    Halide::Var const& x = args[1];
    Halide::Expr const zero = Halide::cast<ElemT>(0);

    Halide::Expr xr = zero;
    Halide::Expr xi = zero;
    if (inputComplex) {
      xr = bounded(0, x - r);
      xi = bounded(1, x - r);
    } else {
      xr = bounded(x - r);
    }

    Halide::Expr hr = zero;
    Halide::Expr hi = zero;
    if (isTapsComplex) {
      hr = tapsBuf(0, r);
      hi = tapsBuf(1, r);
    } else {
      hr = tapsBuf(r);
    }

    func(c, x) = Halide::mux(c, {Halide::sum(xr * hr - xi * hi), Halide::sum(xr * hi + xi * hr)});
  } else {
    Halide::Var const& x = args[0];
    func(x) = Halide::sum(bounded(x - r) * Halide::cast<ElemT>(tapsBuf(r)));
  }
  return func;
}

/**
 * @brief Handle FIR 滤波器算子
 *
 * $y[n] = \sum_{k=0}^{M-1} h[k] x[n-k]$
 *
 * 优化特性：
 * - 自动利用系数对称性（偶对称、奇对称、共轭对称）减少乘法次数（减半）
 * - 支持所有 Real/Complex 输入输出组合
 */
template <typename T, typename TapsVecT>
Halide::Func handleFirImpl(const TapsVecT& tapsVec, const prism::dsl::detail::Node* node,
                           const Halide::Func& inputFunc, const std::vector<Halide::Var>& args) {
  using ArgT = std::decay_t<TapsVecT>;
  // Check if it is a vector
  if constexpr (!std::is_same_v<ArgT, std::vector<prism::real32_t>> &&
                !std::is_same_v<ArgT, std::vector<prism::real64_t>> &&
                !std::is_same_v<ArgT, std::vector<prism::complex32_t>> &&
                !std::is_same_v<ArgT, std::vector<prism::complex64_t>>) {
    throw std::invalid_argument("FIR taps must be a vector");
  } else {
    using TapType = typename ArgT::value_type;
    const int tapsLen = static_cast<int>(tapsVec.size());
    const int inputLen = static_cast<int>(node->inputs.at(0)->shape.length);

    bool const inputComplex = prism::isComplexType(node->inputs[0]->outputType);
    bool const outputComplex = prism::isComplexType(node->outputType);

    auto const makePassThrough = [&]() -> Halide::Func {
      return buildFirPassThrough<T>(inputFunc, args, inputComplex, outputComplex);
    };

    if (tapsLen == 0) {
      return makePassThrough();
    }

    Halide::Func const bounded =
        makeBoundedInput<T>(inputFunc, node->boundary, inputLen, inputComplex);

    using TapRealT = typename prism::ToHalideType<TapType>::Type;
    Halide::Buffer<TapRealT> tapsBuf;
    if constexpr (prism::IS_COMPLEX_V<TapType>) {
      tapsBuf = Halide::Buffer<TapRealT>(2, tapsLen);
      for (int i = 0; i < tapsLen; ++i) {
        tapsBuf(0, i) = static_cast<TapRealT>(tapsVec[i].real());
        tapsBuf(1, i) = static_cast<TapRealT>(tapsVec[i].imag());
      }
    } else {
      tapsBuf = Halide::Buffer<TapRealT>(tapsLen);
      for (int i = 0; i < tapsLen; ++i) {
        tapsBuf(i) = static_cast<TapRealT>(tapsVec[i]);
      }
    }

    FirSymmetry const symmetry = detectFirSymmetry(tapsVec);

    // ------------------------------------------------------------------
    // 对称性优化路径 (Tap Symmetry Optimization)
    // ------------------------------------------------------------------
    if (symmetry != FirSymmetry::NONE && tapsLen > 1) {
      auto symFunc = buildSymmetricFir<T, TapType>(args, bounded, tapsBuf, tapsLen, symmetry,
                                                   inputComplex, outputComplex);
      if (symFunc.defined()) {
        return symFunc;
      }
    }

    // ------------------------------------------------------------------
    // 通用卷积路径 (General Path)
    // ------------------------------------------------------------------
    return buildGeneralFir<T, TapType>(args, bounded, tapsBuf, tapsLen, inputComplex,
                                       outputComplex);
  }
}

template <typename T>
Halide::Func handleFir(const prism::dsl::detail::Node* node, prism::runtime::OpContext<T>& ctx,
                       const std::vector<Halide::Var>& args) {
  auto inputFunc = ctx.buildFunc(prism::dsl::Signal::fromNode(node->inputs.at(0)));

  // 使用 std::visit 统一处理不同精度的 taps (from node->param)
  return std::visit(
      [&](const auto& tapsVec) -> Halide::Func {
        return handleFirImpl<T>(tapsVec, node, inputFunc, args);
      },
      node->param);
}

REGISTER_OP(FIR, handleFir);

/**
 * @brief Handle Moving Average 算子
 *
 * $y[n] = \frac{1}{W} \sum_{k=0}^{W-1} x[n-k]$
 *
 * 优化策略：
 * - 小窗口：Direct Sum (向量化性能好)
 * - 大窗口：Prefix Sum (复用性好，复杂度 O(N))
 */
template <typename T>
Halide::Func handleMovingAverage(const prism::dsl::detail::Node* node,
                                 prism::runtime::OpContext<T>& ctx,
                                 const std::vector<Halide::Var>& args) {
  auto inputFunc = ctx.buildFunc(prism::dsl::Signal::fromNode(node->inputs.at(0)));

  // Extract window from param (scalar)
  int const window = std::visit(
      [](auto&& arg) -> int {
        using ArgT = std::decay_t<decltype(arg)>;
        if constexpr (std::is_arithmetic_v<ArgT>) return static_cast<int>(arg);
        return 1;
      },
      node->param);
  int const len = static_cast<int>(node->inputs.at(0)->shape.length);
  bool const outputComplex = prism::isComplexType(node->outputType);
  bool const inputComplex = prism::isComplexType(node->inputs.at(0)->outputType);
  if (outputComplex != inputComplex) {
    throw std::invalid_argument("MovingAverage: output type must match input type");
  }
  using ElemT = typename prism::ToHalideType<T>::Type;
  Halide::Func const bounded = makeBoundedInput<T>(inputFunc, node->boundary, len, inputComplex);

  Halide::Func func;
  if (window <= 0) {
    throw std::invalid_argument("MovingAverage window must be positive");
  }
  if (window == 1) {
    func(args) = inputFunc(args);
    return func;
  }

  Halide::Type const elemType = Halide::type_of<typename prism::ToHalideType<T>::Type>();
  int const vectorWidth = std::max(1, Halide::get_host_target().natural_vector_size(elemType));
  int const kSmallWindow = getMovingAvgDirectSumThreshold(len, ctx.useGpu, elemType);
  bool const useDirectSum = (window <= kSmallWindow) || (node->boundary != prism::BndryMode::ZERO);

  if (outputComplex) {
    Halide::Var const& c = args[0];
    Halide::Var const& x = args[1];
    auto const zero = Halide::cast<ElemT>(0);

    if (useDirectSum) {
      Halide::RDom const r(0, window);
      func(c, x) = Halide::sum(bounded(c, x - r)) / Halide::cast<ElemT>(window);

    } else {
      // Prefix Sum Approach
      Halide::Func prefix("ma_prefix");
      prefix(c, x) = zero;
      prefix(c, 0) = bounded(c, 0);
      Halide::RDom const r(1, len - 1);
      prefix(c, r) = prefix(c, r - 1) + bounded(c, r);
      prefix.fold_storage(x, vectorWidth);
      prefix.update(0).unscheduled();
      prefix.update(1).unscheduled();

      Halide::Expr const left = Halide::select(x >= window, prefix(c, x - window), zero);
      func(c, x) = (prefix(c, x) - left) / Halide::cast<ElemT>(window);
    }
  } else {
    Halide::Var const& x = args[0];
    auto const zero = Halide::cast<ElemT>(0);

    if (useDirectSum) {
      Halide::RDom const r(0, window);
      func(x) = Halide::sum(bounded(x - r)) / Halide::cast<ElemT>(window);

    } else {
      Halide::Func prefix("ma_prefix");
      prefix(x) = zero;
      prefix(0) = bounded(0);
      Halide::RDom const r(1, len - 1);
      prefix(r) = prefix(r - 1) + bounded(r);
      prefix.fold_storage(x, vectorWidth);
      prefix.update(0).unscheduled();
      prefix.update(1).unscheduled();

      Halide::Expr const left = Halide::select(x >= window, prefix(x - window), zero);
      func(x) = (prefix(x) - left) / Halide::cast<ElemT>(window);
    }
  }
  return func;
}

REGISTER_OP(MOVING_AVERAGE, handleMovingAverage);

/**
 * @brief Handle Median 算子
 *
 * 使用冒泡排序网络（Bitonic Sort 思想）在常数窗口内寻找中值
 * 针对 Halide 向量化进行了优化
 */
template <typename T>
Halide::Func handleMedian(const prism::dsl::detail::Node* node, prism::runtime::OpContext<T>& ctx,
                          const std::vector<Halide::Var>& args) {
  auto inputFunc = ctx.buildFunc(prism::dsl::Signal::fromNode(node->inputs.at(0)));
  int const window = std::visit(
      [](auto&& arg) -> int {
        using ArgT = std::decay_t<decltype(arg)>;
        if constexpr (std::is_arithmetic_v<ArgT>) return static_cast<int>(arg);
        return 1;
      },
      node->param);
  int const len = static_cast<int>(node->inputs.at(0)->shape.length);
  bool const inputComplex = prism::isComplexType(node->inputs.at(0)->outputType);
  bool const outputComplex = prism::isComplexType(node->outputType);
  if (outputComplex != inputComplex) {
    throw std::invalid_argument("Median: output type must match input type");
  }
  Halide::Func const bounded = makeBoundedInput<T>(inputFunc, node->boundary, len, inputComplex);

  Halide::Func func;

  if (window <= 0 || (window % 2) == 0) {
    throw std::invalid_argument("Median window must be positive odd");
  }

  int const radius = window / 2;
  if (inputComplex) {
    Halide::Var const& c = args[0];
    Halide::Var const& x = args[1];
    std::vector<Halide::Expr> valuesR;
    std::vector<Halide::Expr> valuesI;
    valuesR.reserve(window);
    valuesI.reserve(window);
    for (int offset = -radius; offset <= radius; ++offset) {
      Halide::Expr const idx = x + offset;
      valuesR.push_back(bounded(0, idx));
      valuesI.push_back(bounded(1, idx));
    }

    auto compareSwapComplex = [](Halide::Expr& r1, Halide::Expr& i1, Halide::Expr& r2,
                                 Halide::Expr& i2) {
      Halide::Expr const r1Old = r1;
      Halide::Expr const i1Old = i1;
      Halide::Expr const r2Old = r2;
      Halide::Expr const i2Old = i2;
      Halide::Expr const mag1 = r1Old * r1Old + i1Old * i1Old;
      Halide::Expr const mag2 = r2Old * r2Old + i2Old * i2Old;
      Halide::Expr const swap = mag1 > mag2;
      r1 = Halide::select(swap, r2Old, r1Old);
      i1 = Halide::select(swap, i2Old, i1Old);
      r2 = Halide::select(swap, r1Old, r2Old);
      i2 = Halide::select(swap, i1Old, i2Old);
    };

    for (size_t i = 0; i < valuesR.size(); ++i) {
      for (size_t j = 0; j + 1 < valuesR.size() - i; ++j) {
        compareSwapComplex(valuesR[j], valuesI[j], valuesR[j + 1], valuesI[j + 1]);
      }
    }

    Halide::Expr const outR = valuesR[valuesR.size() / 2];
    Halide::Expr const outI = valuesI[valuesI.size() / 2];
    func(c, x) = Halide::mux(c, {outR, outI});
  } else {
    Halide::Var const& x = args[0];
    std::vector<Halide::Expr> values;
    values.reserve(window);
    for (int offset = -radius; offset <= radius; ++offset) {
      Halide::Expr const idx = x + offset;
      values.push_back(bounded(idx));
    }

    auto compareSwap = [](Halide::Expr& a, Halide::Expr& b) {
      Halide::Expr const lo = Halide::min(a, b);
      Halide::Expr const hi = Halide::max(a, b);
      a = lo;
      b = hi;
    };

    for (size_t i = 0; i < values.size(); ++i) {
      for (size_t j = 0; j + 1 < values.size() - i; ++j) {
        compareSwap(values[j], values[j + 1]);
      }
    }

    func(x) = values[values.size() / 2];
  }

  return func;
}

REGISTER_OP(MEDIAN, handleMedian);

// 显式注册函数
void registerFilterHandlers() {
  // Handler 已通过 REGISTER_OP 自动注册
}

/// @}

}  // namespace prism::runtime
