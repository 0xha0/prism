/**
 * @file test_basic_ops.cpp
 * @ingroup tests
 * @brief 基础算子单元测试
 *
 * 覆盖加减乘除、缩放、绝对值、卷积、克罗内克积等 DSL 基础算子，
 * 针对实数/复数、单精度/双精度进行模板化验证。
 */

#include <Halide.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <string>
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
using prism::ScalarType;
using namespace prism::dsl;
using namespace prism::runtime;
using namespace prism::tests;

/// @addtogroup tests
/// @{

// ==============================================================================
// 测试数据生成器
// ==============================================================================

/** @brief 卷积输入样本：1,2,3,4 */
template <typename T>
std::vector<T> getConvInput() {
  return {static_cast<T>(1.0), static_cast<T>(2.0), static_cast<T>(3.0),
          static_cast<T>(4.0)};
}

/** @brief 卷积核：1,0,-1 */
template <typename T>
std::vector<T> getConvKernel() {
  return {static_cast<T>(1.0), static_cast<T>(0.0), static_cast<T>(-1.0)};
}

/** @brief 卷积期望输出 */
template <typename T>
std::vector<T> getConvExpected() {
  return {static_cast<T>(1.0), static_cast<T>(2.0),  static_cast<T>(2.0),
          static_cast<T>(2.0), static_cast<T>(-3.0), static_cast<T>(-4.0)};
}

/** @brief 克罗内克积输入 A */
template <typename T>
std::vector<T> getKronA() {
  return {static_cast<T>(1.0), static_cast<T>(2.0), static_cast<T>(3.0)};
}

/** @brief 克罗内克积输入 B */
template <typename T>
std::vector<T> getKronB() {
  return {static_cast<T>(4.0), static_cast<T>(5.0)};
}

/** @brief 克罗内克积期望输出 */
template <typename T>
std::vector<T> getKronExpected() {
  return {static_cast<T>(4.0),  static_cast<T>(5.0),  static_cast<T>(8.0),
          static_cast<T>(10.0), static_cast<T>(12.0), static_cast<T>(15.0)};
}

// ==============================================================================
// 基础算子测试
// ==============================================================================

/**
 * @brief 验证逐元素加法
 * @tparam T 精度类型
 */
template <typename T>
void testAdd() {
  std::string const name = "Add (" + TypeName<T>::get() + ")";
  Halide::Buffer<T> buf(4);
  for (int i = 0; i < 4; ++i) buf(i) = static_cast<T>(i);  // [0,1,2,3]

  Signal const x = Signal::input(4);
  Signal const two = Signal::constant(2.0, 4);
  Signal const y = x + two;

  auto result = Executor::run<T>(y, buf);

  bool pass = true;
  if (!approxEqual(result(0), static_cast<T>(2.0))) pass = false;
  if (!approxEqual(result(1), static_cast<T>(3.0))) pass = false;
  if (!approxEqual(result(2), static_cast<T>(4.0))) pass = false;
  if (!approxEqual(result(3), static_cast<T>(5.0))) pass = false;

  TestPrinter::printTestResult(name, pass);
  assert(pass);
}

/**
 * @brief 验证逐元素减法
 * @tparam T 精度类型
 */
template <typename T>
void testSub() {
  std::string const name = "Sub (" + TypeName<T>::get() + ")";
  Halide::Buffer<T> buf(4);
  for (int i = 0; i < 4; ++i) buf(i) = static_cast<T>(i + 5);  // [5,6,7,8]

  Signal const x = Signal::input(4);
  Signal const one = Signal::constant(1.0, 4);
  Signal const y = x - one;

  auto result = Executor::run<T>(y, buf);

  bool pass = true;
  if (!approxEqual(result(0), static_cast<T>(4.0))) pass = false;
  if (!approxEqual(result(1), static_cast<T>(5.0))) pass = false;
  if (!approxEqual(result(2), static_cast<T>(6.0))) pass = false;
  if (!approxEqual(result(3), static_cast<T>(7.0))) pass = false;

  TestPrinter::printTestResult(name, pass);
  assert(pass);
}

/**
 * @brief 验证逐元素乘法
 * @tparam T 精度类型
 */
template <typename T>
void testMul() {
  std::string const name = "Mul (" + TypeName<T>::get() + ")";
  Halide::Buffer<T> buf(4);
  for (int i = 0; i < 4; ++i) buf(i) = static_cast<T>(i + 1);  // [1,2,3,4]

  Signal const x = Signal::input(4);
  Signal const three = Signal::constant(3.0, 4);
  Signal const y = x * three;

  auto result = Executor::run<T>(y, buf);

  bool pass = true;
  if (!approxEqual(result(0), static_cast<T>(3.0))) pass = false;
  if (!approxEqual(result(1), static_cast<T>(6.0))) pass = false;
  if (!approxEqual(result(2), static_cast<T>(9.0))) pass = false;
  if (!approxEqual(result(3), static_cast<T>(12.0))) pass = false;

  TestPrinter::printTestResult(name, pass);
  assert(pass);
}

/**
 * @brief 验证逐元素除法
 * @tparam T 精度类型
 */
template <typename T>
void testDiv() {
  std::string const name = "Div (" + TypeName<T>::get() + ")";
  Halide::Buffer<T> buf(4);
  for (int i = 0; i < 4; ++i) {
    buf(i) = static_cast<T>((i + 1) * 4);  // [4,8,12,16]
  }

  Signal const x = Signal::input(4);
  Signal const two = Signal::constant(2.0, 4);
  Signal const y = x / two;

  auto result = Executor::run<T>(y, buf);

  bool pass = true;
  if (!approxEqual(result(0), static_cast<T>(2.0))) pass = false;
  if (!approxEqual(result(1), static_cast<T>(4.0))) pass = false;
  if (!approxEqual(result(2), static_cast<T>(6.0))) pass = false;
  if (!approxEqual(result(3), static_cast<T>(8.0))) pass = false;

  TestPrinter::printTestResult(name, pass);
  assert(pass);
}

/**
 * @brief 验证缩放算子
 * @tparam T 精度类型
 */
template <typename T>
void testScale() {
  std::string const name = "Scale (" + TypeName<T>::get() + ")";
  Halide::Buffer<T> buf(4);
  for (int i = 0; i < 4; ++i) buf(i) = static_cast<T>(i + 1);  // [1,2,3,4]

  Signal const x = Signal::input(4);
  Signal const y = scale(x, 0.5);

  auto result = Executor::run<T>(y, buf);

  bool pass = true;
  if (!approxEqual(result(0), static_cast<T>(0.5))) pass = false;
  if (!approxEqual(result(1), static_cast<T>(1.0))) pass = false;
  if (!approxEqual(result(2), static_cast<T>(1.5))) pass = false;
  if (!approxEqual(result(3), static_cast<T>(2.0))) pass = false;

  TestPrinter::printTestResult(name, pass);
  assert(pass);
}

/**
 * @brief 验证绝对值算子
 * @tparam T 精度类型
 */
template <typename T>
void testAbs() {
  std::string const name = "Abs (" + TypeName<T>::get() + ")";
  Halide::Buffer<T> buf(4);
  buf(0) = static_cast<T>(-3.0);
  buf(1) = static_cast<T>(2.0);
  buf(2) = static_cast<T>(-1.0);
  buf(3) = static_cast<T>(0.0);

  Signal const x = Signal::input(4);
  Signal const y = abs(x);

  auto result = Executor::run<T>(y, buf);

  bool pass = true;
  if (!approxEqual(result(0), static_cast<T>(3.0))) pass = false;
  if (!approxEqual(result(1), static_cast<T>(2.0))) pass = false;
  if (!approxEqual(result(2), static_cast<T>(1.0))) pass = false;
  if (!approxEqual(result(3), static_cast<T>(0.0))) pass = false;

  TestPrinter::printTestResult(name, pass);
  assert(pass);
}

/**
 * @brief 验证取负算子（通过 scale -1）
 * @tparam T 精度类型
 */
template <typename T>
void testNegate() {
  std::string const name = "Negate (" + TypeName<T>::get() + ")";
  Halide::Buffer<T> buf(4);
  for (int i = 0; i < 4; ++i) buf(i) = static_cast<T>(i - 1);  // [-1,0,1,2]

  Signal const x = Signal::input(4);
  Signal const y = scale(x, -1.0);

  auto result = Executor::run<T>(y, buf);

  bool pass = true;
  if (!approxEqual(result(0), static_cast<T>(1.0))) pass = false;
  if (!approxEqual(result(1), static_cast<T>(0.0))) pass = false;
  if (!approxEqual(result(2), static_cast<T>(-1.0))) pass = false;
  if (!approxEqual(result(3), static_cast<T>(-2.0))) pass = false;

  TestPrinter::printTestResult(name, pass);
  assert(pass);
}

// ==============================================================================
// Composite Ops Tests
// ==============================================================================

/**
 * @brief 验证卷积算子与常量卷积核的输出
 * @tparam T 精度类型
 */
template <typename T>
void testConvolve() {
  std::string const name = "Convolve (" + TypeName<T>::get() + ")";
  auto inputData = getConvInput<T>();
  auto kernelData = getConvKernel<T>();
  auto expected = getConvExpected<T>();

  Halide::Buffer<T> buf(inputData.size());
  for (size_t i = 0; i < inputData.size(); ++i) buf(i) = inputData[i];

  Signal const x = Signal::input(inputData.size());
  Signal const k = Signal::constant(
      1.0, 3);  // Keeping it simple for DSL graph, but using pre-calc expected
  // Note: Signal::constant(1.0, 3) implies [1,1,1] kernel?
  // Wait, original test said: Signal const k = Signal::constant(1.0, 3); //
  // 简化: [1,1,1] But hardcoded CONV_KERNEL was {1, 0, -1}. Let's stick to the
  // original test logic which used [1,1,1] constant signal in DSL but verified
  // against... wait. Original: Signal const k = Signal::constant(1.0, 3); ->
  // [1,1,1] Original Verification: [1,2,3,4] * [1,1,1] = [1, 3, 6, 9, 7, 4] My
  // GenConvExpected was for [1,0,-1]? No, I should respect the original test
  // logic. Let's recreate the expected for [1,1,1] here locally or fix the
  // generator.

  std::vector<T> const localExpected = {
      static_cast<T>(1.0), static_cast<T>(3.0), static_cast<T>(6.0),
      static_cast<T>(9.0), static_cast<T>(7.0), static_cast<T>(4.0)};

  Signal const y = convolve(x, k);

  auto result = Executor::run<T>(y, buf);

  T err = maxError(result, localExpected);
  bool const pass = err < 1e-4;

  TestPrinter::printTestResult(name, pass);
  assert(pass);
}

/**
 * @brief 验证克罗内克积算子
 * @tparam T 精度类型
 */
template <typename T>
void testKron() {
  std::string const name = "Kron (" + TypeName<T>::get() + ")";
  auto aData = getKronA<T>();
  // DSL: b = Constant(1.0, 2) -> [1, 1]

  Halide::Buffer<T> buf(aData.size());
  for (size_t i = 0; i < aData.size(); ++i) buf(i) = aData[i];

  Signal const a = Signal::input(aData.size());
  Signal const b = Signal::constant(1.0, 2);
  Signal const y = kron(a, b);

  auto result = Executor::run<T>(y, buf);

  // Expected for [1,2,3] x [1,1] -> [1,1, 2,2, 3,3]
  std::vector<T> const localExpected = {
      static_cast<T>(1.0), static_cast<T>(1.0), static_cast<T>(2.0),
      static_cast<T>(2.0), static_cast<T>(3.0), static_cast<T>(3.0)};

  T err = maxError(result, localExpected);
  bool const pass = err < 1e-4;

  TestPrinter::printTestResult(name, pass);
  assert(pass);
}

// ==============================================================================
// 上/下采样算子测试
// ==============================================================================

/**
 * @brief 验证上采样（插零）
 * @tparam T 精度类型
 */
template <typename T>
void testUpsample() {
  std::string const name = "Upsample (" + TypeName<T>::get() + ")";
  std::vector<T> input = {static_cast<T>(1.0), static_cast<T>(2.0),
                          static_cast<T>(3.0)};

  Halide::Buffer<T> buf(input.size());
  for (size_t i = 0; i < input.size(); ++i) buf(i) = input[i];

  Signal const x = Signal::input(input.size());
  Signal const y = upsample(x, 3);

  auto result = Executor::run<T>(y, buf);

  std::vector<T> const expected = {
      static_cast<T>(1.0), static_cast<T>(0.0), static_cast<T>(0.0),
      static_cast<T>(2.0), static_cast<T>(0.0), static_cast<T>(0.0),
      static_cast<T>(3.0), static_cast<T>(0.0), static_cast<T>(0.0)};

  T err = maxError(result, expected);
  bool const pass = err < static_cast<T>(1e-4);

  TestPrinter::printTestResult(name, pass);
  assert(pass);
}

/**
 * @brief 验证下采样（抽取）
 * @tparam T 精度类型
 */
template <typename T>
void testDownsample() {
  std::string const name = "Downsample (" + TypeName<T>::get() + ")";
  std::vector<T> input = {static_cast<T>(1.0), static_cast<T>(2.0),
                          static_cast<T>(3.0), static_cast<T>(4.0),
                          static_cast<T>(5.0), static_cast<T>(6.0),
                          static_cast<T>(7.0)};

  Halide::Buffer<T> buf(input.size());
  for (size_t i = 0; i < input.size(); ++i) buf(i) = input[i];

  Signal const x = Signal::input(input.size());
  Signal const y = downsample(x, 2, 1);

  auto result = Executor::run<T>(y, buf);

  std::vector<T> const expected = {static_cast<T>(2.0), static_cast<T>(4.0),
                                   static_cast<T>(6.0)};

  T err = maxError(result, expected);
  bool const pass = err < static_cast<T>(1e-4);

  TestPrinter::printTestResult(name, pass);
  assert(pass);
}

// ==============================================================================
// I/Q 拆装测试
// ==============================================================================

/**
 * @brief 验证 I/Q 交织与拆分
 * @tparam T 精度类型
 */
template <typename T>
void testIqPackRoundtrip() {
  std::string const name = "IQ Pack/Unpack (" + TypeName<T>::get() + ")";
  std::vector<T> input = {static_cast<T>(1.0), static_cast<T>(2.0),
                          static_cast<T>(3.0), static_cast<T>(4.0)};
  std::vector<T> qExpected(input.size());
  for (size_t i = 0; i < input.size(); ++i) {
    qExpected[i] = -input[i];
  }

  Halide::Buffer<T> buf(input.size());
  for (size_t i = 0; i < input.size(); ++i) {
    buf(i) = input[i];
  }

  Signal const x = Signal::input(input.size());
  Signal const qSig = scale(x, -1.0);
  Signal const iq = iqPack(x, qSig);
  Signal const iOut = iqI(iq);
  Signal const qOut = iqQ(iq);

  auto iResult = Executor::run<T>(iOut, buf);
  auto qResult = Executor::run<T>(qOut, buf);

  T errI = maxError(iResult, input);
  T errQ = maxError(qResult, qExpected);
  bool const pass = errI < static_cast<T>(1e-4) && errQ < static_cast<T>(1e-4);

  TestPrinter::printTestResult(name, pass);
  assert(pass);
}

// ==============================================================================
// Chained Ops Tests
// ==============================================================================

/**
 * @brief 验证复合链路 (x+1)*2
 * @tparam T 精度类型
 */
template <typename T>
void testChain() {
  std::string const name = "Chained Ops (" + TypeName<T>::get() + ")";
  Halide::Buffer<T> buf(4);
  for (int i = 0; i < 4; ++i) buf(i) = static_cast<T>(i + 1);

  // (x + 1) * 2
  Signal const x = Signal::input(4);
  Signal const one = Signal::constant(1.0, 4);
  Signal const two = Signal::constant(2.0, 4);
  Signal const y = (x + one) * two;

  auto result = Executor::run<T>(y, buf);

  bool pass = true;
  if (!approxEqual(result(0), static_cast<T>(4.0))) pass = false;
  if (!approxEqual(result(1), static_cast<T>(6.0))) pass = false;
  if (!approxEqual(result(2), static_cast<T>(8.0))) pass = false;
  if (!approxEqual(result(3), static_cast<T>(10.0))) pass = false;

  TestPrinter::printTestResult(name, pass);
  assert(pass);
}

/**
 * @brief 验证复合链路 ((x-1)*3+2)/2
 * @tparam T 精度类型
 */
template <typename T>
void testComplexChain() {
  std::string const name = "Complex Chain (" + TypeName<T>::get() + ")";
  Halide::Buffer<T> buf(4);
  for (int i = 0; i < 4; ++i) buf(i) = static_cast<T>(i);

  // ((x - 1) * 3 + 2) / 2
  Signal const x = Signal::input(4);
  Signal const one = Signal::constant(1.0, 4);
  Signal const two = Signal::constant(2.0, 4);
  Signal const three = Signal::constant(3.0, 4);
  Signal const y = ((x - one) * three + two) / two;

  auto result = Executor::run<T>(y, buf);

  bool pass = true;
  if (!approxEqual(result(0), static_cast<T>(-0.5))) pass = false;
  if (!approxEqual(result(1), static_cast<T>(1.0))) pass = false;
  if (!approxEqual(result(2), static_cast<T>(2.5))) pass = false;
  if (!approxEqual(result(3), static_cast<T>(4.0))) pass = false;

  TestPrinter::printTestResult(name, pass);
  assert(pass);
}

// ==============================================================================
// Complex Arithmetic Tests
// ==============================================================================

/**
 * @brief 复数加法测试（real/imag 分通道）
 * @tparam ComplexT 复数类型
 */
template <typename ComplexT>
void testComplexAdd() {
  using RealT = typename ComplexT::value_type;
  std::string const name = "Complex Add (" + TypeName<ComplexT>::get() + ")";

  std::vector<ComplexT> data = {{1.0, 1.0}, {2.0, 2.0}, {3.0, 3.0}, {4.0, 4.0}};
  int const len = 4;

  Halide::Buffer<RealT> inBuf(2, len);
  fillComplexBuffer(inBuf, data);

  ScalarType const stype =
      (sizeof(RealT) == 8) ? ScalarType::C64 : ScalarType::C32;
  auto s = Signal::input(len, stype);
  auto y = s + s;

  auto result = Executor::run<ComplexT>(y, inBuf);

  bool pass = true;
  for (int i = 0; i < len; ++i) {
    ComplexT expected = data[i] + data[i];
    ComplexT actual;
    if constexpr (std::is_same_v<RealT, double>) {
      actual = getComplex64(result, i);
    } else {
      actual = getComplex32(result, i);
    }

    if (std::abs(actual - expected) > 1e-5) pass = false;
  }
  TestPrinter::printTestResult(name, pass);
  assert(pass);
}

/**
 * @brief 复数减法测试
 * @tparam ComplexT 复数类型
 */
template <typename ComplexT>
void testComplexSub() {
  using RealT = typename ComplexT::value_type;
  std::string const name = "Complex Sub (" + TypeName<ComplexT>::get() + ")";

  std::vector<ComplexT> const data = {
      {1.0, 1.0}, {2.0, 2.0}, {3.0, 3.0}, {4.0, 4.0}};
  int const len = 4;

  Halide::Buffer<RealT> inBuf(2, len);
  fillComplexBuffer(inBuf, data);

  ScalarType const stype =
      (sizeof(RealT) == 8) ? ScalarType::C64 : ScalarType::C32;
  auto s = Signal::input(len, stype);
  // NOLINTNEXTLINE(misc-redundant-expression
  auto y = s - s;

  auto result = Executor::run<ComplexT>(y, inBuf);

  bool pass = true;
  for (int i = 0; i < len; ++i) {
    ComplexT expected = {0, 0};
    ComplexT actual;
    if constexpr (std::is_same_v<RealT, double>) {
      actual = getComplex64(result, i);
    } else {
      actual = getComplex32(result, i);
    }

    if (std::abs(actual - expected) > 1e-5) pass = false;
  }
  TestPrinter::printTestResult(name, pass);
  assert(pass);
}

/**
 * @brief 复数乘法测试
 * @tparam ComplexT 复数类型
 */
template <typename ComplexT>
void testComplexMul() {
  using RealT = typename ComplexT::value_type;
  std::string const name = "Complex Mul (" + TypeName<ComplexT>::get() + ")";

  std::vector<ComplexT> data = {{1.0, 1.0}, {2.0, 2.0}, {3.0, 3.0}, {4.0, 4.0}};
  int const len = 4;

  Halide::Buffer<RealT> inBuf(2, len);
  fillComplexBuffer(inBuf, data);

  ScalarType const stype =
      (sizeof(RealT) == 8) ? ScalarType::C64 : ScalarType::C32;
  auto s = Signal::input(len, stype);
  auto y = s * s;

  auto result = Executor::run<ComplexT>(y, inBuf);

  bool pass = true;
  for (int i = 0; i < len; ++i) {
    ComplexT expected = data[i] * data[i];
    ComplexT actual;
    if constexpr (std::is_same_v<RealT, double>) {
      actual = getComplex64(result, i);
    } else {
      actual = getComplex32(result, i);
    }

    if (std::abs(actual - expected) > 1e-5) pass = false;
  }
  TestPrinter::printTestResult(name, pass);
  assert(pass);
}

/**
 * @brief 复数除法测试
 * @tparam ComplexT 复数类型
 */
template <typename ComplexT>
void testComplexDiv() {
  using RealT = typename ComplexT::value_type;
  std::string const name = "Complex Div (" + TypeName<ComplexT>::get() + ")";

  std::vector<ComplexT> const data = {
      {1.0, 1.0}, {2.0, 2.0}, {3.0, 3.0}, {4.0, 4.0}};
  int const len = 4;

  Halide::Buffer<RealT> inBuf(2, len);
  fillComplexBuffer(inBuf, data);

  ScalarType const stype =
      (sizeof(RealT) == 8) ? ScalarType::C64 : ScalarType::C32;
  auto s = Signal::input(len, stype);
  // NOLINTNEXTLINE(misc-redundant-expression
  auto y = s / s;

  auto result = Executor::run<ComplexT>(y, inBuf);

  bool pass = true;
  for (int i = 0; i < len; ++i) {
    ComplexT expected = {1.0, 0.0};
    ComplexT actual;
    if constexpr (std::is_same_v<RealT, double>) {
      actual = getComplex64(result, i);
    } else {
      actual = getComplex32(result, i);
    }

    if (std::abs(actual - expected) > 1e-5) pass = false;
  }
  TestPrinter::printTestResult(name, pass);
  assert(pass);
}

/**
 * @brief 复数绝对值测试，期望虚部清零
 * @tparam ComplexT 复数类型
 */
template <typename ComplexT>
void testComplexAbs() {
  using RealT = typename ComplexT::value_type;
  std::string const name = "Complex Abs (" + TypeName<ComplexT>::get() + ")";

  std::vector<ComplexT> data = {{1.0, 1.0}, {2.0, 2.0}, {3.0, 3.0}, {4.0, 4.0}};
  int const len = 4;

  Halide::Buffer<RealT> inBuf(2, len);
  fillComplexBuffer(inBuf, data);

  ScalarType const stype =
      (sizeof(RealT) == 8) ? ScalarType::C64 : ScalarType::C32;
  auto s = Signal::input(len, stype);
  auto y = abs(s);

  auto result = Executor::run<ComplexT>(y, inBuf);

  bool pass = true;
  for (int i = 0; i < len; ++i) {
    RealT const mag = std::abs(data[i]);
    ComplexT expected = {mag, 0};
    ComplexT actual;
    if constexpr (std::is_same_v<RealT, double>) {
      actual = getComplex64(result, i);
    } else {
      actual = getComplex32(result, i);
    }

    if (std::abs(actual - expected) > 1e-5) pass = false;
  }
  TestPrinter::printTestResult(name, pass);
  assert(pass);
}

/**
 * @brief 复数缩放测试
 * @tparam ComplexT 复数类型
 */
template <typename ComplexT>
void testComplexScale() {
  using RealT = typename ComplexT::value_type;
  std::string const name = "Complex Scale (" + TypeName<ComplexT>::get() + ")";

  std::vector<ComplexT> data = {{1.0, 1.0}, {2.0, 2.0}, {3.0, 3.0}, {4.0, 4.0}};
  int const len = 4;

  Halide::Buffer<RealT> inBuf(2, len);
  fillComplexBuffer(inBuf, data);

  ScalarType const stype =
      (sizeof(RealT) == 8) ? ScalarType::C64 : ScalarType::C32;
  auto s = Signal::input(len, stype);
  auto y = scale(s, 2.0);

  auto result = Executor::run<ComplexT>(y, inBuf);

  bool pass = true;
  for (int i = 0; i < len; ++i) {
    ComplexT expected = data[i] * static_cast<RealT>(2.0);
    ComplexT actual;
    if constexpr (std::is_same_v<RealT, double>) {
      actual = getComplex64(result, i);
    } else {
      actual = getComplex32(result, i);
    }

    if (std::abs(actual - expected) > 1e-5) pass = false;
  }
  TestPrinter::printTestResult(name, pass);
  assert(pass);
}

// ==============================================================================
// Stress Test
// ==============================================================================

/**
 * @brief 大规模信号性能与正确性冒烟
 * @tparam T 精度类型
 */
template <typename T>
void testLargeSignal() {
  std::string const name = "Large Signal (" + TypeName<T>::get() + ")";
  const int n = 1024;
  Halide::Buffer<T> buf(n);
  for (int i = 0; i < n; ++i) buf(i) = std::sin(i * 0.01);

  Signal const x = Signal::input(n);
  Signal const y = scale(x, 2.0);

  auto result = Executor::run<T>(y, buf);

  T maxErr = 0.0;
  for (int i = 0; i < n; ++i) {
    T const expected = static_cast<T>(2.0 * std::sin(i * 0.01));
    maxErr = std::max(maxErr, std::abs(result(i) - expected));
  }

  bool const pass = maxErr < 1e-5;
  TestPrinter::printTestResult(name, pass, "maxErr=" + std::to_string(maxErr));
  assert(pass);
}

// ==============================================================================
// Main
// ==============================================================================

int main() {
  TestPrinter::printSuiteHeader("DSL Ops Tests (Refactored)");

  // --- Real32 ---
  TestPrinter::printSection("Basic Ops (Real32)");
  runTest([] { testAdd<real32_t>(); }, "Add (Real32)");
  runTest([] { testSub<real32_t>(); }, "Sub (Real32)");
  runTest([] { testMul<real32_t>(); }, "Mul (Real32)");
  runTest([] { testDiv<real32_t>(); }, "Div (Real32)");
  runTest([] { testScale<real32_t>(); }, "Scale (Real32)");
  runTest([] { testAbs<real32_t>(); }, "Abs (Real32)");
  runTest([] { testNegate<real32_t>(); }, "Negate (Real32)");

  TestPrinter::printSection("Composite Ops (Real32)");
  runTest([] { testConvolve<real32_t>(); }, "Convolve (Real32)");
  runTest([] { testKron<real32_t>(); }, "Kron (Real32)");
  runTest([] { testUpsample<real32_t>(); }, "Upsample (Real32)");
  runTest([] { testDownsample<real32_t>(); }, "Downsample (Real32)");
  runTest([] { testIqPackRoundtrip<real32_t>(); }, "IQ Pack/Unpack (Real32)");

  TestPrinter::printSection("Chained Ops (Real32)");
  runTest([] { testChain<real32_t>(); }, "Chain (Real32)");
  runTest([] { testComplexChain<real32_t>(); }, "Complex Chain (Real32)");

  TestPrinter::printSection("Complex Ops (Complex32)");
  runTest([] { testComplexAdd<complex32_t>(); }, "Add (Complex32)");
  runTest([] { testComplexSub<complex32_t>(); }, "Sub (Complex32)");
  runTest([] { testComplexMul<complex32_t>(); }, "Mul (Complex32)");
  runTest([] { testComplexDiv<complex32_t>(); }, "Div (Complex32)");
  runTest([] { testComplexScale<complex32_t>(); }, "Scale (Complex32)");
  runTest([] { testComplexAbs<complex32_t>(); }, "Abs (Complex32)");

  TestPrinter::printSection("Stress Test (Real32)");
  runTest([] { testLargeSignal<real32_t>(); }, "Large Signal (Real32)");

  // --- Real64 ---
  TestPrinter::printSection("Basic Ops (Real64)");
  runTest([] { testAdd<real64_t>(); }, "Add (Real64)");
  runTest([] { testSub<real64_t>(); }, "Sub (Real64)");
  runTest([] { testMul<real64_t>(); }, "Mul (Real64)");
  runTest([] { testDiv<real64_t>(); }, "Div (Real64)");
  runTest([] { testScale<real64_t>(); }, "Scale (Real64)");
  runTest([] { testAbs<real64_t>(); }, "Abs (Real64)");
  runTest([] { testNegate<real64_t>(); }, "Negate (Real64)");

  TestPrinter::printSection("Composite Ops (Real64)");
  runTest([] { testConvolve<real64_t>(); }, "Convolve (Real64)");
  runTest([] { testKron<real64_t>(); }, "Kron (Real64)");
  runTest([] { testUpsample<real64_t>(); }, "Upsample (Real64)");
  runTest([] { testDownsample<real64_t>(); }, "Downsample (Real64)");
  runTest([] { testIqPackRoundtrip<real64_t>(); }, "IQ Pack/Unpack (Real64)");

  TestPrinter::printSection("Chained Ops (Real64)");
  runTest([] { testChain<real64_t>(); }, "Chain (Real64)");
  runTest([] { testComplexChain<real64_t>(); }, "Complex Chain (Real64)");

  TestPrinter::printSection("Complex Ops (Complex64)");
  runTest([] { testComplexAdd<complex64_t>(); }, "Add (Complex64)");
  runTest([] { testComplexSub<complex64_t>(); }, "Sub (Complex64)");
  runTest([] { testComplexMul<complex64_t>(); }, "Mul (Complex64)");
  runTest([] { testComplexDiv<complex64_t>(); }, "Div (Complex64)");
  runTest([] { testComplexScale<complex64_t>(); }, "Scale (Complex64)");
  runTest([] { testComplexAbs<complex64_t>(); }, "Abs (Complex64)");

  TestPrinter::printSection("Stress Test (Real64)");
  runTest([] { testLargeSignal<real64_t>(); }, "Large Signal (Real64)");

  TestPrinter::printSummary();
  return 0;
}

/// @}
