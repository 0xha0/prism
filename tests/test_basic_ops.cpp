/**
 * @file test_basic_ops.cpp
 * @ingroup tests
 * @brief 基础算子单元测试
 *
 * 本文件涵盖了 DSL 基础算子的正确性验证，包括：
 * - 基础四则运算 (Add, Sub, Mul, Div)
 * - 单目运算 (Scale, Abs, Negate, Conj)
 * - 采样率变换 (Upsample, Downsample)
 * - 数据格式转换 (IQ Pack/Unpack)
 *
 * ## 测试策略
 * - **分组测试**：每个操作覆盖所有精度 (F32, F64) 和数域 (Real, Complex)
 * - **二元运算**：统一测试 a+b, a-b, a*b, a/b
 * - **混合运算**：测试实数与复数混合运算 (如 Real * Complex)
 * - **类型适配**：使用 `TestTraits` 统一处理 Buffer 和元素类型差异
 */

#include <Halide.h>

#include <cassert>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>
#include <vector>

#include "prism/dsl/ops.h"
#include "prism/dsl/signal.h"
#include "prism/runtime/executor.h"
#include "prism/types.h"
#include "test_utils.h"

using prism::complex32_t;
using prism::complex64_t;
using prism::real32_t;
using prism::real64_t;
using namespace prism::dsl;
using namespace prism::runtime;
using namespace prism::tests;

/// @addtogroup tests
/// @{

// ==============================================================================
// 测试数据生成 (Test Data Generation)
// ==============================================================================

/** @brief 生成二元运算测试数据 A */
template <typename T>
std::vector<T> getBinaryDataA() {
  if constexpr (TestTraits<T>::IS_COMPLEX) {
    return {{5.0, 3.0}, {8.0, 4.0}, {10.0, 6.0}, {12.0, 8.0}};
  } else {
    return {static_cast<T>(5.0), static_cast<T>(8.0), static_cast<T>(10.0), static_cast<T>(12.0)};
  }
}

/** @brief 生成二元运算测试数据 B */
template <typename T>
std::vector<T> getBinaryDataB() {
  if constexpr (TestTraits<T>::IS_COMPLEX) {
    return {{1.0, 1.0}, {2.0, 2.0}, {2.0, 2.0}, {4.0, 4.0}};
  } else {
    return {static_cast<T>(1.0), static_cast<T>(2.0), static_cast<T>(2.0), static_cast<T>(4.0)};
  }
}

// ======================================================================
// 校验工具 (Validation Helpers)
// ======================================================================

/**
 * @brief 检查 Buffer 内容是否匹配期望值
 * @tparam T 元素类型
 * @param result 计算结果 Buffer
 * @param expected 期望向量
 * @param tol 允许的最大误差 (Tolerance)
 */
template <typename T>
bool checkBufferMatches(const Halide::Buffer<typename TestTraits<T>::BufferElemType>& result,
                        const std::vector<T>& expected, double tol = 1e-4) {
  for (size_t i = 0; i < expected.size(); ++i) {
    T const actual = TestTraits<T>::getElement(result, static_cast<int>(i));
    auto const diff = std::abs(actual - expected[i]);
    if (diff > static_cast<decltype(diff)>(tol)) {
      return false;
    }
  }
  return true;
}

// ==============================================================================
// 统一测试模板 (Unified Test Templates)
// ==============================================================================

enum class BinaryOp : std::uint8_t { ADD, SUB, MUL, DIV };
enum class UnaryOp : std::uint8_t { ABS, NEG, CONJ };

inline const char* binaryOpName(BinaryOp op) {
  switch (op) {
    case BinaryOp::ADD:
      return "Add";
    case BinaryOp::SUB:
      return "Sub";
    case BinaryOp::MUL:
      return "Mul";
    case BinaryOp::DIV:
      return "Div";
  }
  return "Unknown";
}

inline const char* unaryOpName(UnaryOp op) {
  switch (op) {
    case UnaryOp::ABS:
      return "Abs";
    case UnaryOp::NEG:
      return "Negate";
    case UnaryOp::CONJ:
      return "Conj";
  }
  return "Unknown";
}

/**
 * @brief 将输入值转换为期望输出类型 (支持实数->复数提升)
 */
template <typename OutT, typename InT>
OutT castToOut(const InT& v) {
  static_assert(!(TestTraits<InT>::IS_COMPLEX && !TestTraits<OutT>::IS_COMPLEX),
                "Invalid cast: complex to real");
  if constexpr (TestTraits<OutT>::IS_COMPLEX) {
    using RealT = typename TestTraits<OutT>::BufferElemType;
    if constexpr (TestTraits<InT>::IS_COMPLEX) {
      return OutT(static_cast<RealT>(v.real()), static_cast<RealT>(v.imag()));
    } else {
      return OutT(static_cast<RealT>(v), static_cast<RealT>(0));
    }
  } else {
    return static_cast<OutT>(v);
  }
}

/** @brief 执行二元运算 (host 端参考实现) */
template <BinaryOp Op, typename OutT>
OutT applyBinaryOp(const OutT& a, const OutT& b) {
  if constexpr (Op == BinaryOp::ADD) {
    return a + b;
  } else if constexpr (Op == BinaryOp::SUB) {
    return a - b;
  } else if constexpr (Op == BinaryOp::MUL) {
    return a * b;
  } else {
    return a / b;
  }
}

/** @brief 二元运算类型特征推导 */
template <typename A, typename B>
struct BinaryTraits {
  using RealA = typename TestTraits<A>::BufferElemType;
  using RealB = typename TestTraits<B>::BufferElemType;
  static_assert(std::is_same_v<RealA, RealB>, "Binary test requires matching precision");
  using Real = RealA;
  using Out = std::conditional_t<TestTraits<A>::IS_COMPLEX || TestTraits<B>::IS_COMPLEX,
                                 std::complex<Real>, Real>;
};

/** @brief 标量乘法类型特征推导 */
template <typename InT, typename ScalarT>
struct ScaleTraits {
  using RealIn = typename TestTraits<InT>::BufferElemType;
  using RealScalar = typename TestTraits<ScalarT>::BufferElemType;
  static_assert(std::is_same_v<RealIn, RealScalar>, "Scale test requires matching precision");
  using Real = RealIn;
  using Out = std::conditional_t<TestTraits<InT>::IS_COMPLEX || TestTraits<ScalarT>::IS_COMPLEX,
                                 std::complex<Real>, Real>;
};

template <BinaryOp Op, typename A, typename B>
std::string binaryTestName() {
  std::string const aName = TypeName<A>::get();
  std::string const bName = TypeName<B>::get();
  std::string name = std::string(binaryOpName(Op)) + " (";
  name += aName;
  if (aName != bName) {
    name += "," + bName;
  }
  name += ")";
  return name;
}

template <UnaryOp Op, typename InT>
std::string unaryTestName() {
  return std::string(unaryOpName(Op)) + " (" + TypeName<InT>::get() + ")";
}

template <typename InT, typename ScalarT>
std::string scaleTestName() {
  return "Scale (" + TypeName<InT>::get() + "," + TypeName<ScalarT>::get() + ")";
}

template <typename T>
std::vector<T> getUnaryData() {
  if constexpr (TestTraits<T>::IS_COMPLEX) {
    return {{3.0, 4.0}, {0.0, 1.0}, {1.0, 0.0}, {-3.0, -4.0}};
  } else {
    return {static_cast<T>(3.0), static_cast<T>(0.0), static_cast<T>(1.0), static_cast<T>(-3.0)};
  }
}

template <typename T>
std::vector<T> getScaleData() {
  if constexpr (TestTraits<T>::IS_COMPLEX) {
    return {{1.0, 1.0}, {2.0, -2.0}, {-3.0, 3.0}, {-4.0, -4.0}};
  } else {
    return {static_cast<T>(1.0), static_cast<T>(-2.0), static_cast<T>(0.0), static_cast<T>(3.0)};
  }
}

template <typename ScalarT>
ScalarT scaleScalarValue() {
  if constexpr (TestTraits<ScalarT>::IS_COMPLEX) {
    using RealT = typename ScalarT::value_type;
    return ScalarT(static_cast<RealT>(0.5), static_cast<RealT>(-0.5));
  } else {
    return static_cast<ScalarT>(2.0);
  }
}

// ==============================================================================
// 二元运算测试
// ==============================================================================

template <BinaryOp Op, typename A, typename B>
void testBinaryOp() {
  using TraitsA = TestTraits<A>;
  using TraitsB = TestTraits<B>;
  using BT = BinaryTraits<A, B>;
  using OutT = typename BT::Out;
  using RealT = typename BT::Real;
  std::string const name = binaryTestName<Op, A, B>();

  auto dataA = getBinaryDataA<A>();
  auto dataB = getBinaryDataB<B>();
  int const len = static_cast<int>(dataA.size());

  auto bufA = TraitsA::makeBuffer(len);
  auto bufB = TraitsB::makeBuffer(len);
  TraitsA::fillBuffer(bufA, dataA);
  TraitsB::fillBuffer(bufB, dataB);

  auto a = Signal::input(len, TraitsA::scalarType());
  auto b = Signal::input(len, TraitsB::scalarType());
  Signal y;
  if constexpr (Op == BinaryOp::ADD) {
    y = a + b;
  } else if constexpr (Op == BinaryOp::SUB) {
    y = a - b;
  } else if constexpr (Op == BinaryOp::MUL) {
    y = a * b;
  } else {
    y = a / b;
  }

  std::vector<Halide::Buffer<RealT>> const inputs = {bufA, bufB};
  auto result = Executor::run<OutT>(y, inputs);

  std::vector<OutT> expected(len);
  for (int i = 0; i < len; ++i) {
    OutT const av = castToOut<OutT>(dataA[i]);
    OutT const bv = castToOut<OutT>(dataB[i]);
    expected[i] = applyBinaryOp<Op>(av, bv);
  }

  bool const pass = checkBufferMatches<OutT>(result, expected);
  TestPrinter::printTestResult(name, pass);
  assert(pass);
}

// ==============================================================================
// 单目运算测试
// ==============================================================================

template <UnaryOp Op, typename InT>
void testUnaryOp() {
  using Traits = TestTraits<InT>;
  using RealT = typename Traits::BufferElemType;
  using OutT = std::conditional_t<Op == UnaryOp::ABS, RealT, InT>;
  std::string const name = unaryTestName<Op, InT>();

  auto data = getUnaryData<InT>();
  int const len = static_cast<int>(data.size());
  auto buf = Traits::makeBuffer(len);
  Traits::fillBuffer(buf, data);

  auto x = Signal::input(len, Traits::scalarType());
  Signal y;
  if constexpr (Op == UnaryOp::ABS) {
    y = abs(x);
  } else if constexpr (Op == UnaryOp::NEG) {
    y = negative(x);
  } else {
    y = conj(x);
  }

  auto result = Executor::run<OutT>(y, buf);

  std::vector<OutT> expected(len);
  for (int i = 0; i < len; ++i) {
    if constexpr (Op == UnaryOp::ABS) {
      expected[i] = static_cast<OutT>(std::abs(data[i]));
    } else if constexpr (Op == UnaryOp::NEG) {
      expected[i] = -data[i];
    } else {
      if constexpr (Traits::IS_COMPLEX) {
        expected[i] = std::conj(data[i]);
      } else {
        expected[i] = data[i];
      }
    }
  }

  bool const pass = checkBufferMatches<OutT>(result, expected);
  TestPrinter::printTestResult(name, pass);
  assert(pass);
}

template <typename InT, typename ScalarT>
void testScale() {
  using Traits = TestTraits<InT>;
  using ST = ScaleTraits<InT, ScalarT>;
  using OutT = typename ST::Out;
  std::string const name = scaleTestName<InT, ScalarT>();

  auto data = getScaleData<InT>();
  int const len = static_cast<int>(data.size());
  auto buf = Traits::makeBuffer(len);
  Traits::fillBuffer(buf, data);

  auto const scalar = scaleScalarValue<ScalarT>();
  auto x = Signal::input(len, Traits::scalarType());
  auto y = scale(x, scalar);

  auto result = Executor::run<OutT>(y, buf);

  std::vector<OutT> expected(len);
  OutT const sv = castToOut<OutT>(scalar);
  for (int i = 0; i < len; ++i) {
    expected[i] = castToOut<OutT>(data[i]) * sv;
  }

  bool const pass = checkBufferMatches<OutT>(result, expected);
  TestPrinter::printTestResult(name, pass);
  assert(pass);
}

// ==============================================================================
// Main Entry
// ==============================================================================

int main() {
  TestPrinter::printSuiteHeader("DSL Ops Tests");

  // --- 二元运算 ---
  TestPrinter::printSection("Add (Binary: a + b)");
  runTest([] { testBinaryOp<BinaryOp::ADD, real32_t, real32_t>(); },
          binaryTestName<BinaryOp::ADD, real32_t, real32_t>());
  runTest([] { testBinaryOp<BinaryOp::ADD, real64_t, real64_t>(); },
          binaryTestName<BinaryOp::ADD, real64_t, real64_t>());
  runTest([] { testBinaryOp<BinaryOp::ADD, complex32_t, complex32_t>(); },
          binaryTestName<BinaryOp::ADD, complex32_t, complex32_t>());
  runTest([] { testBinaryOp<BinaryOp::ADD, complex64_t, complex64_t>(); },
          binaryTestName<BinaryOp::ADD, complex64_t, complex64_t>());

  TestPrinter::printSection("Sub (Binary: a - b)");
  runTest([] { testBinaryOp<BinaryOp::SUB, real32_t, real32_t>(); },
          binaryTestName<BinaryOp::SUB, real32_t, real32_t>());
  runTest([] { testBinaryOp<BinaryOp::SUB, real64_t, real64_t>(); },
          binaryTestName<BinaryOp::SUB, real64_t, real64_t>());
  runTest([] { testBinaryOp<BinaryOp::SUB, complex32_t, complex32_t>(); },
          binaryTestName<BinaryOp::SUB, complex32_t, complex32_t>());
  runTest([] { testBinaryOp<BinaryOp::SUB, complex64_t, complex64_t>(); },
          binaryTestName<BinaryOp::SUB, complex64_t, complex64_t>());

  TestPrinter::printSection("Mul (Binary: a * b)");
  runTest([] { testBinaryOp<BinaryOp::MUL, real32_t, real32_t>(); },
          binaryTestName<BinaryOp::MUL, real32_t, real32_t>());
  runTest([] { testBinaryOp<BinaryOp::MUL, real64_t, real64_t>(); },
          binaryTestName<BinaryOp::MUL, real64_t, real64_t>());
  runTest([] { testBinaryOp<BinaryOp::MUL, complex32_t, complex32_t>(); },
          binaryTestName<BinaryOp::MUL, complex32_t, complex32_t>());
  runTest([] { testBinaryOp<BinaryOp::MUL, complex64_t, complex64_t>(); },
          binaryTestName<BinaryOp::MUL, complex64_t, complex64_t>());

  TestPrinter::printSection("Div (Binary: a / b)");
  runTest([] { testBinaryOp<BinaryOp::DIV, real32_t, real32_t>(); },
          binaryTestName<BinaryOp::DIV, real32_t, real32_t>());
  runTest([] { testBinaryOp<BinaryOp::DIV, real64_t, real64_t>(); },
          binaryTestName<BinaryOp::DIV, real64_t, real64_t>());
  runTest([] { testBinaryOp<BinaryOp::DIV, complex32_t, complex32_t>(); },
          binaryTestName<BinaryOp::DIV, complex32_t, complex32_t>());
  runTest([] { testBinaryOp<BinaryOp::DIV, complex64_t, complex64_t>(); },
          binaryTestName<BinaryOp::DIV, complex64_t, complex64_t>());

  // --- 单目运算 ---
  TestPrinter::printSection("Scale (Unary: a * scalar)");
  runTest([] { testScale<real32_t, real32_t>(); }, scaleTestName<real32_t, real32_t>());
  runTest([] { testScale<real64_t, real64_t>(); }, scaleTestName<real64_t, real64_t>());
  runTest([] { testScale<complex32_t, complex32_t>(); }, scaleTestName<complex32_t, complex32_t>());
  runTest([] { testScale<complex64_t, complex64_t>(); }, scaleTestName<complex64_t, complex64_t>());
  runTest([] { testScale<real32_t, complex32_t>(); }, scaleTestName<real32_t, complex32_t>());
  runTest([] { testScale<real64_t, complex64_t>(); }, scaleTestName<real64_t, complex64_t>());
  runTest([] { testScale<complex32_t, real32_t>(); }, scaleTestName<complex32_t, real32_t>());
  runTest([] { testScale<complex64_t, real64_t>(); }, scaleTestName<complex64_t, real64_t>());

  TestPrinter::printSection("Abs (Unary: |a|)");
  runTest([] { testUnaryOp<UnaryOp::ABS, real32_t>(); }, unaryTestName<UnaryOp::ABS, real32_t>());
  runTest([] { testUnaryOp<UnaryOp::ABS, real64_t>(); }, unaryTestName<UnaryOp::ABS, real64_t>());
  runTest([] { testUnaryOp<UnaryOp::ABS, complex32_t>(); },
          unaryTestName<UnaryOp::ABS, complex32_t>());
  runTest([] { testUnaryOp<UnaryOp::ABS, complex64_t>(); },
          unaryTestName<UnaryOp::ABS, complex64_t>());

  TestPrinter::printSection("Negate (Unary: -a)");
  runTest([] { testUnaryOp<UnaryOp::NEG, real32_t>(); }, unaryTestName<UnaryOp::NEG, real32_t>());
  runTest([] { testUnaryOp<UnaryOp::NEG, real64_t>(); }, unaryTestName<UnaryOp::NEG, real64_t>());
  runTest([] { testUnaryOp<UnaryOp::NEG, complex32_t>(); },
          unaryTestName<UnaryOp::NEG, complex32_t>());
  runTest([] { testUnaryOp<UnaryOp::NEG, complex64_t>(); },
          unaryTestName<UnaryOp::NEG, complex64_t>());

  TestPrinter::printSection("Conj (Unary: conj(a))");
  runTest([] { testUnaryOp<UnaryOp::CONJ, real32_t>(); }, unaryTestName<UnaryOp::CONJ, real32_t>());
  runTest([] { testUnaryOp<UnaryOp::CONJ, real64_t>(); }, unaryTestName<UnaryOp::CONJ, real64_t>());
  runTest([] { testUnaryOp<UnaryOp::CONJ, complex32_t>(); },
          unaryTestName<UnaryOp::CONJ, complex32_t>());
  runTest([] { testUnaryOp<UnaryOp::CONJ, complex64_t>(); },
          unaryTestName<UnaryOp::CONJ, complex64_t>());

  // --- 混合二元运算 ---
  TestPrinter::printSection("Mixed Binary Ops (Real/Complex)");
  runTest([] { testBinaryOp<BinaryOp::ADD, real32_t, complex32_t>(); },
          binaryTestName<BinaryOp::ADD, real32_t, complex32_t>());
  runTest([] { testBinaryOp<BinaryOp::ADD, complex32_t, real32_t>(); },
          binaryTestName<BinaryOp::ADD, complex32_t, real32_t>());
  runTest([] { testBinaryOp<BinaryOp::SUB, real32_t, complex32_t>(); },
          binaryTestName<BinaryOp::SUB, real32_t, complex32_t>());
  runTest([] { testBinaryOp<BinaryOp::SUB, complex32_t, real32_t>(); },
          binaryTestName<BinaryOp::SUB, complex32_t, real32_t>());
  runTest([] { testBinaryOp<BinaryOp::MUL, real32_t, complex32_t>(); },
          binaryTestName<BinaryOp::MUL, real32_t, complex32_t>());
  runTest([] { testBinaryOp<BinaryOp::MUL, complex32_t, real32_t>(); },
          binaryTestName<BinaryOp::MUL, complex32_t, real32_t>());
  runTest([] { testBinaryOp<BinaryOp::DIV, real32_t, complex32_t>(); },
          binaryTestName<BinaryOp::DIV, real32_t, complex32_t>());
  runTest([] { testBinaryOp<BinaryOp::DIV, complex32_t, real32_t>(); },
          binaryTestName<BinaryOp::DIV, complex32_t, real32_t>());

  runTest([] { testBinaryOp<BinaryOp::ADD, real64_t, complex64_t>(); },
          binaryTestName<BinaryOp::ADD, real64_t, complex64_t>());
  runTest([] { testBinaryOp<BinaryOp::ADD, complex64_t, real64_t>(); },
          binaryTestName<BinaryOp::ADD, complex64_t, real64_t>());
  runTest([] { testBinaryOp<BinaryOp::SUB, real64_t, complex64_t>(); },
          binaryTestName<BinaryOp::SUB, real64_t, complex64_t>());
  runTest([] { testBinaryOp<BinaryOp::SUB, complex64_t, real64_t>(); },
          binaryTestName<BinaryOp::SUB, complex64_t, real64_t>());
  runTest([] { testBinaryOp<BinaryOp::MUL, real64_t, complex64_t>(); },
          binaryTestName<BinaryOp::MUL, real64_t, complex64_t>());
  runTest([] { testBinaryOp<BinaryOp::MUL, complex64_t, real64_t>(); },
          binaryTestName<BinaryOp::MUL, complex64_t, real64_t>());
  runTest([] { testBinaryOp<BinaryOp::DIV, real64_t, complex64_t>(); },
          binaryTestName<BinaryOp::DIV, real64_t, complex64_t>());
  runTest([] { testBinaryOp<BinaryOp::DIV, complex64_t, real64_t>(); },
          binaryTestName<BinaryOp::DIV, complex64_t, real64_t>());
  TestPrinter::printSummary();
  return 0;
}

/// @}
