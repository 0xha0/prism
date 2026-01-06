/**
 * @file test_advanced_ops.cpp
 * @ingroup tests
 * @brief 复合算子与复杂链路单元测试
 *
 * 验证多算子组合的复杂 Pipeline 正确性与边界条件处理主要包括：
 * - **算子级联 (Cascaded Ops)**：如 FIR 滤波后接缩放 (Convolve -> Scale)
 * - **混合算子 (Mixed Ops)**：Convolve/Kron 的实/复混合组合
 * - **边界条件 (Boundary Conditions)**：如长度为 1 的信号、极大长度信号 (100K+)
 * 的吞吐验证
 *
 * 支持 real32_t 和 real64_t 双精度测试覆盖
 */

#include <Halide.h>

#include <cassert>
#include <cmath>
#include <complex>
#include <cstddef>
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

// ============================================================================
// Convolve/Kron 参考实现 (Reference Implementations)
// ============================================================================

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

/** @brief 将实数向量转换为复数向量 (虚部为0) */
template <typename RealT>
std::vector<std::complex<RealT>> toComplexVector(const std::vector<RealT>& data) {
  std::vector<std::complex<RealT>> out;
  out.reserve(data.size());
  for (auto v : data) {
    out.emplace_back(v, static_cast<RealT>(0));
  }
  return out;
}

/** @brief 复数向量原样返回 (用于模板重载) */
template <typename RealT>
std::vector<std::complex<RealT>> toComplexVector(const std::vector<std::complex<RealT>>& data) {
  return data;
}

/** @brief 复数卷积参考实现 (Naive $O(N^2)$) */
template <typename ComplexT>
std::vector<ComplexT> convolveFull(const std::vector<ComplexT>& a, const std::vector<ComplexT>& b) {
  size_t const outLen = a.size() + b.size() - 1;
  std::vector<ComplexT> out(outLen, ComplexT{});
  for (size_t n = 0; n < outLen; ++n) {
    ComplexT acc = {};
    for (size_t m = 0; m < b.size(); ++m) {
      if (n >= m && (n - m) < a.size()) {
        acc += a[n - m] * b[m];
      }
    }
    out[n] = acc;
  }
  return out;
}

/** @brief Kronecker 积参考实现 */
template <typename ComplexT>
std::vector<ComplexT> kronFull(const std::vector<ComplexT>& a, const std::vector<ComplexT>& b) {
  std::vector<ComplexT> out;
  out.reserve(a.size() * b.size());
  for (const auto& av : a) {
    for (const auto& bv : b) {
      out.push_back(av * bv);
    }
  }
  return out;
}

// ============================================================================
// Convolve/Kron 测试用例
// ============================================================================

template <typename A, typename B>
std::string mixedOpName(std::string const& op) {
  return op + " (" + TypeName<A>::get() + ", " + TypeName<B>::get() + ")";
}

template <typename RealT>
struct MixedData {
  using ComplexT = std::complex<RealT>;
  std::vector<RealT> aR;
  std::vector<RealT> bR;
  std::vector<ComplexT> aC;
  std::vector<ComplexT> bC;
};

template <typename RealT>
MixedData<RealT> makeConvolveData() {
  using ComplexT = std::complex<RealT>;
  MixedData<RealT> data;
  data.aR = {static_cast<RealT>(1.0), static_cast<RealT>(2.0), static_cast<RealT>(3.0)};
  data.bR = {static_cast<RealT>(0.5), static_cast<RealT>(-1.0)};
  data.aC = {ComplexT{static_cast<RealT>(1.0), static_cast<RealT>(1.0)},
             ComplexT{static_cast<RealT>(0.5), static_cast<RealT>(-0.5)},
             ComplexT{static_cast<RealT>(-1.0), static_cast<RealT>(0.0)}};
  data.bC = {ComplexT{static_cast<RealT>(2.0), static_cast<RealT>(-1.0)},
             ComplexT{static_cast<RealT>(0.5), static_cast<RealT>(0.5)}};
  return data;
}

template <typename RealT>
MixedData<RealT> makeKronData() {
  using ComplexT = std::complex<RealT>;
  MixedData<RealT> data;
  data.aR = {static_cast<RealT>(1.0), static_cast<RealT>(-2.0)};
  data.bR = {static_cast<RealT>(0.5), static_cast<RealT>(2.0)};
  data.aC = {ComplexT{static_cast<RealT>(1.0), static_cast<RealT>(1.0)},
             ComplexT{static_cast<RealT>(0.0), static_cast<RealT>(-1.0)}};
  data.bC = {ComplexT{static_cast<RealT>(2.0), static_cast<RealT>(0.0)},
             ComplexT{static_cast<RealT>(-0.5), static_cast<RealT>(0.5)}};
  return data;
}

template <typename T, typename RealT>
decltype(auto) selectAData(const MixedData<RealT>& data) {
  if constexpr (TestTraits<T>::IS_COMPLEX) {
    return (data.aC);
  } else {
    return (data.aR);
  }
}

template <typename T, typename RealT>
decltype(auto) selectBData(const MixedData<RealT>& data) {
  if constexpr (TestTraits<T>::IS_COMPLEX) {
    return (data.bC);
  } else {
    return (data.bR);
  }
}

template <typename RealT, typename A, typename B>
void testConvolveCase() {
  using ComplexT = std::complex<RealT>;
  using OutT =
      std::conditional_t<TestTraits<A>::IS_COMPLEX || TestTraits<B>::IS_COMPLEX, ComplexT, RealT>;
  std::string const name = mixedOpName<A, B>("Convolve");
  auto const data = makeConvolveData<RealT>();
  auto const& aData = selectAData<A>(data);
  auto const& bData = selectBData<B>(data);

  auto bufA = TestTraits<A>::makeBuffer(aData.size());
  auto bufB = TestTraits<B>::makeBuffer(bData.size());
  TestTraits<A>::fillBuffer(bufA, aData);
  TestTraits<B>::fillBuffer(bufB, bData);

  auto sigA = Signal::input(aData.size(), TestTraits<A>::scalarType());
  auto sigB = Signal::input(bData.size(), TestTraits<B>::scalarType());

  auto y = convolve(sigA, sigB);
  std::vector<Halide::Buffer<RealT>> const inputs = {bufA, bufB};
  auto out = Executor::run<OutT>(y, inputs);

  auto expC = convolveFull(toComplexVector(aData), toComplexVector(bData));
  bool pass = true;
  if constexpr (TestTraits<OutT>::IS_COMPLEX) {
    pass = checkBufferMatches<OutT>(out, expC);
  } else {
    std::vector<RealT> expected(expC.size());
    for (size_t i = 0; i < expC.size(); ++i) expected[i] = expC[i].real();
    pass = checkBufferMatches<RealT>(out, expected);
  }

  TestPrinter::printTestResult(name, pass);
  assert(pass);
}

template <typename RealT, typename A, typename B>
void testKronCase() {
  using ComplexT = std::complex<RealT>;
  using OutT =
      std::conditional_t<TestTraits<A>::IS_COMPLEX || TestTraits<B>::IS_COMPLEX, ComplexT, RealT>;
  std::string const name = mixedOpName<A, B>("Kron");
  auto const data = makeKronData<RealT>();
  auto const& aData = selectAData<A>(data);
  auto const& bData = selectBData<B>(data);

  auto bufA = TestTraits<A>::makeBuffer(aData.size());
  auto bufB = TestTraits<B>::makeBuffer(bData.size());
  TestTraits<A>::fillBuffer(bufA, aData);
  TestTraits<B>::fillBuffer(bufB, bData);

  auto sigA = Signal::input(aData.size(), TestTraits<A>::scalarType());
  auto sigB = Signal::input(bData.size(), TestTraits<B>::scalarType());

  auto y = kron(sigA, sigB);
  std::vector<Halide::Buffer<RealT>> const inputs = {bufA, bufB};
  auto out = Executor::run<OutT>(y, inputs);

  auto expC = kronFull(toComplexVector(aData), toComplexVector(bData));
  bool pass = true;
  if constexpr (TestTraits<OutT>::IS_COMPLEX) {
    pass = checkBufferMatches<OutT>(out, expC);
  } else {
    std::vector<RealT> expected(expC.size());
    for (size_t i = 0; i < expC.size(); ++i) expected[i] = expC[i].real();
    pass = checkBufferMatches<RealT>(out, expected);
  }

  TestPrinter::printTestResult(name, pass);
  assert(pass);
}

// ==============================================================================
// 采样率变换测试
// ==============================================================================

/**
 * @brief [Test] 上采样 (Upsample)
 */
template <typename T>
void testUpsample() {
  using Traits = TestTraits<T>;
  std::string const name = "Upsample (" + TypeName<T>::get() + ")";

  std::vector<T> input;
  if constexpr (Traits::IS_COMPLEX) {
    input = {T(1.0, 1.0), T(2.0, -1.0), T(-1.0, 0.5)};
  } else {
    input = {static_cast<T>(1.0), static_cast<T>(2.0), static_cast<T>(3.0)};
  }
  int const factor = 3;
  int const offset = 1;
  std::vector<T> expected(input.size() * factor, T{});
  for (size_t i = 0; i < input.size(); ++i) {
    expected[offset + (i * factor)] = input[i];
  }

  auto buf = Traits::makeBuffer(input.size());
  Traits::fillBuffer(buf, input);

  auto x = Signal::input(input.size(), Traits::scalarType());
  auto y = upsample(x, factor, offset);

  auto result = Executor::run<T>(y, buf);

  bool const pass = checkBufferMatches<T>(result, expected);
  TestPrinter::printTestResult(name, pass);
  assert(pass);
}

/**
 * @brief [Test] 下采样 (Downsample)
 */
template <typename T>
void testDownsample() {
  using Traits = TestTraits<T>;
  std::string const name = "Downsample (" + TypeName<T>::get() + ")";

  std::vector<T> input;
  if constexpr (Traits::IS_COMPLEX) {
    input = {T(1.0, 0.5),  T(2.0, -1.0), T(3.0, 0.0), T(4.0, 1.0),
             T(5.0, -0.5), T(6.0, 2.0),  T(7.0, -2.0)};
  } else {
    input = {static_cast<T>(1.0), static_cast<T>(2.0), static_cast<T>(3.0), static_cast<T>(4.0),
             static_cast<T>(5.0), static_cast<T>(6.0), static_cast<T>(7.0)};
  }
  int const factor = 2;
  int const offset = 1;
  int const outLen = static_cast<int>((input.size() - offset + factor - 1) / factor);
  std::vector<T> expected;
  expected.reserve(outLen);
  for (int i = offset; i < static_cast<int>(input.size()); i += factor) {
    expected.push_back(input[i]);
  }

  auto buf = Traits::makeBuffer(input.size());
  Traits::fillBuffer(buf, input);

  auto x = Signal::input(input.size(), Traits::scalarType());
  auto y = downsample(x, factor, offset);

  auto result = Executor::run<T>(y, buf);

  bool const pass = checkBufferMatches<T>(result, expected);
  TestPrinter::printTestResult(name, pass);
  assert(pass);
}

// ==============================================================================
// Complex Pack/Unpack 测试
// ==============================================================================

/**
 * @brief [Test] Complex Pack/Round 往返测试
 */
template <typename T>
void testComplexPackRoundtrip() {
  static_assert(!TestTraits<T>::IS_COMPLEX, "Complex Pack only supports real types");
  std::string const name = "Complex Pack Roundtrip (" + TypeName<T>::get() + ")";

  std::vector<T> rData = {static_cast<T>(1.0), static_cast<T>(2.0), static_cast<T>(3.0),
                          static_cast<T>(4.0)};
  std::vector<T> iData = {static_cast<T>(-1.0), static_cast<T>(-2.0), static_cast<T>(-3.0),
                          static_cast<T>(-4.0)};
  int const len = 4;

  Halide::Buffer<T> iBuf(len);
  Halide::Buffer<T> qBuf(len);
  for (int i = 0; i < len; ++i) {
    iBuf(i) = rData[i];
    qBuf(i) = iData[i];
  }

  // 两个独立的 Signal
  auto rSig = Signal::input(len, TestTraits<T>::scalarType());
  auto iSig = Signal::input(len, TestTraits<T>::scalarType());
  auto packed = complexPack(rSig, iSig);
  auto rOut = real(packed);
  auto iOut = imag(packed);

  // 使用多输入 run
  std::vector<Halide::Buffer<T>> const inputs = {iBuf, qBuf};
  using ComplexT = std::complex<T>;
  std::vector<ComplexT> expected;
  expected.reserve(rData.size());
  for (size_t i = 0; i < rData.size(); ++i) {
    expected.emplace_back(rData[i], iData[i]);
  }
  auto packedResult = Executor::run<ComplexT>(packed, inputs);
  auto rResult = Executor::run<T>(rOut, inputs);
  auto iResult = Executor::run<T>(iOut, inputs);

  T errI = maxError(rResult, rData);
  T errQ = maxError(iResult, iData);
  bool const pass = checkBufferMatches<ComplexT>(packedResult, expected) &&
                    errI < static_cast<T>(1e-4) && errQ < static_cast<T>(1e-4);

  TestPrinter::printTestResult(name, pass);
  assert(pass);
}

// ============================================================================
// AOT 多输入测试 (CPU)
// ============================================================================

template <typename RealT>
void testAotConvolveRR() {
  std::string const name = "AOT Convolve (R,R) (" + TypeName<RealT>::get() + ")";

  auto const data = makeConvolveData<RealT>();
  auto const& aData = data.aR;
  auto const& bData = data.bR;

  auto bufA = TestTraits<RealT>::makeBuffer(aData.size());
  auto bufB = TestTraits<RealT>::makeBuffer(bData.size());
  TestTraits<RealT>::fillBuffer(bufA, aData);
  TestTraits<RealT>::fillBuffer(bufB, bData);

  auto sigA = Signal::input(aData.size(), TestTraits<RealT>::scalarType());
  auto sigB = Signal::input(bData.size(), TestTraits<RealT>::scalarType());
  auto y = convolve(sigA, sigB);

  std::vector<Halide::Buffer<RealT>> const inputs = {bufA, bufB};
  auto pipeline = Executor::compile<RealT>(y, ExecMode::CPU);
  auto out = pipeline.run(inputs);

  auto expC = convolveFull(toComplexVector(aData), toComplexVector(bData));
  std::vector<RealT> expected(expC.size());
  for (size_t i = 0; i < expC.size(); ++i) expected[i] = expC[i].real();

  bool const pass = checkBufferMatches<RealT>(out, expected);
  TestPrinter::printTestResult(name, pass);
  assert(pass);
}

template <typename RealT>
void testAotComplexPack() {
  using ComplexT = std::complex<RealT>;
  std::string const name = "AOT ComplexPack (I,Q) (" + TypeName<RealT>::get() + ")";

  std::vector<RealT> rData = {static_cast<RealT>(1.0), static_cast<RealT>(2.0),
                              static_cast<RealT>(3.0), static_cast<RealT>(4.0)};
  std::vector<RealT> iData = {static_cast<RealT>(-1.0), static_cast<RealT>(-2.0),
                              static_cast<RealT>(-3.0), static_cast<RealT>(-4.0)};

  auto rBuf = TestTraits<RealT>::makeBuffer(rData.size());
  auto iBuf = TestTraits<RealT>::makeBuffer(iData.size());
  TestTraits<RealT>::fillBuffer(rBuf, rData);
  TestTraits<RealT>::fillBuffer(iBuf, iData);

  auto rSig = Signal::input(rData.size(), TestTraits<RealT>::scalarType());
  auto iSig = Signal::input(iData.size(), TestTraits<RealT>::scalarType());
  auto packed = complexPack(rSig, iSig);

  std::vector<Halide::Buffer<RealT>> const inputs = {rBuf, iBuf};
  auto pipeline = Executor::compile<ComplexT>(packed, ExecMode::CPU);
  auto out = pipeline.run(inputs);

  std::vector<ComplexT> expected;
  expected.reserve(rData.size());
  for (size_t i = 0; i < rData.size(); ++i) {
    expected.emplace_back(rData[i], iData[i]);
  }

  bool const pass = checkBufferMatches<ComplexT>(out, expected);
  TestPrinter::printTestResult(name, pass);
  assert(pass);
}

// ============================================================================
// 复合链路测试
// ============================================================================

template <typename RealT>
void testLongCompositeChain() {
  using ComplexT = std::complex<RealT>;
  using RealTraits = TestTraits<RealT>;
  using ComplexTraits = TestTraits<ComplexT>;
  std::string const name =
      "LongChain (Kron+Convolve+Add+Scale+Abs) (" + TypeName<RealT>::get() + ")";

  std::vector<RealT> const aR = {static_cast<RealT>(1.0), static_cast<RealT>(-2.0),
                                 static_cast<RealT>(0.5)};
  std::vector<ComplexT> const bC = {{static_cast<RealT>(0.5), static_cast<RealT>(-1.0)},
                                    {static_cast<RealT>(2.0), static_cast<RealT>(0.25)}};
  std::vector<RealT> const cR = {static_cast<RealT>(1.0), static_cast<RealT>(-0.5),
                                 static_cast<RealT>(2.0)};

  auto kronRef = kronFull(toComplexVector(aR), bC);
  auto convRef = convolveFull(kronRef, toComplexVector(cR));

  std::vector<RealT> dR(convRef.size());
  for (size_t i = 0; i < dR.size(); ++i) {
    dR[i] = static_cast<RealT>(0.1) * static_cast<RealT>(static_cast<int>(i % 5) - 2);
  }
  ComplexT const scaleVal(static_cast<RealT>(0.75), static_cast<RealT>(-0.25));

  std::vector<RealT> expected(convRef.size());
  for (size_t i = 0; i < convRef.size(); ++i) {
    ComplexT val = convRef[i] + ComplexT(dR[i], static_cast<RealT>(0));
    val *= scaleVal;
    expected[i] = std::abs(val);
  }

  auto bufA = RealTraits::makeBuffer(aR.size());
  auto bufB = ComplexTraits::makeBuffer(bC.size());
  auto bufC = RealTraits::makeBuffer(cR.size());
  auto bufD = RealTraits::makeBuffer(dR.size());
  RealTraits::fillBuffer(bufA, aR);
  ComplexTraits::fillBuffer(bufB, bC);
  RealTraits::fillBuffer(bufC, cR);
  RealTraits::fillBuffer(bufD, dR);

  auto sigA = Signal::input(aR.size(), RealTraits::scalarType());
  auto sigB = Signal::input(bC.size(), ComplexTraits::scalarType());
  auto sigC = Signal::input(cR.size(), RealTraits::scalarType());
  auto sigD = Signal::input(dR.size(), RealTraits::scalarType());

  auto y = abs(scale(convolve(kron(sigA, sigB), sigC) + sigD, scaleVal));
  std::vector<Halide::Buffer<RealT>> const inputs = {bufA, bufB, bufC, bufD};
  auto result = Executor::run<RealT>(y, inputs);

  bool const pass = checkBufferMatches<RealT>(result, expected);
  TestPrinter::printTestResult(name, pass);
  assert(pass);
}

// ============================================================================
// 边界条件测试组 (Boundary Condition Tests)
// ============================================================================

/**
 * @brief [Test] 极短信号测试 (Length-1)
 *
 * 验证 Pipeline 处理长度为 1 的缓冲区时的正确性
 * 防止出现除零、越界访问或对齐错误
 * 测试逻辑：y = x * 2.0, len(x)=1
 */
template <typename T>
void testLengthOneSignal() {
  using Traits = TestTraits<T>;
  std::string const name = "Length-1 Signal (" + TypeName<T>::get() + ")";
  auto inputBuf = Traits::makeBuffer(1);
  std::vector<T> const data = {static_cast<T>(42.0)};
  Traits::fillBuffer(inputBuf, data);

  Signal const x = Signal::input(1, Traits::scalarType());
  Signal const y = scale(x, T(2.0));

  auto result = Executor::run<T>(y, inputBuf);

  bool pass = true;
  if (result.width() != 1) pass = false;
  auto val = Traits::getElement(result, 0);
  if (!approxEqual(val, static_cast<T>(84.0))) pass = false;

  TestPrinter::printTestResult(name, pass);
  assert(pass);
}

/**
 * @brief [Test] 大规模信号测试 (100K Points)
 *
 * 验证 Pipeline 处理较大缓冲区 (100,000 点) 时的稳定性
 * 确保内存分配、分块处理 Loop 等机制正常工作
 * 测试逻辑：y = x * 0.5
 */
template <typename T>
void testLargeSignal() {
  using Traits = TestTraits<T>;
  std::string const name = "Large Signal (100K) (" + TypeName<T>::get() + ")";
  int const len = 100000;
  auto inputBuf = Traits::makeBuffer(len);

  // Consider direct fill for speed instead of vector copy for large buf?
  // But standard pattern uses vector. optimize loop later if slow.
  // 100k floats is small (400KB).
  std::vector<T> const data(len, static_cast<T>(1.0));
  Traits::fillBuffer(inputBuf, data);

  Signal const x = Signal::input(len, Traits::scalarType());
  Signal const y = scale(x, T(0.5));

  auto result = Executor::run<T>(y, inputBuf);

  bool pass = true;
  if (result.width() != len) pass = false;
  // 抽样检查首尾数据
  if (!approxEqual(Traits::getElement(result, 0), static_cast<T>(0.5))) {
    pass = false;
  }
  if (!approxEqual(Traits::getElement(result, len - 1), static_cast<T>(0.5))) {
    pass = false;
  }

  TestPrinter::printTestResult(name, pass);
  assert(pass);
}

// ============================================================================
// Main Entry
// ============================================================================

int main() {
  TestPrinter::printSuiteHeader("Advanced Ops Tests");

  // --- 采样率变换 ---
  TestPrinter::printSection("Upsample");
  runTest([] { testUpsample<real32_t>(); }, "Upsample (Real32)");
  runTest([] { testUpsample<real64_t>(); }, "Upsample (Real64)");
  runTest([] { testUpsample<complex32_t>(); }, "Upsample (Complex32)");
  runTest([] { testUpsample<complex64_t>(); }, "Upsample (Complex64)");

  TestPrinter::printSection("Downsample");
  runTest([] { testDownsample<real32_t>(); }, "Downsample (Real32)");
  runTest([] { testDownsample<real64_t>(); }, "Downsample (Real64)");
  runTest([] { testDownsample<complex32_t>(); }, "Downsample (Complex32)");
  runTest([] { testDownsample<complex64_t>(); }, "Downsample (Complex64)");

  // --- Complex 操作（仅实数）---
  TestPrinter::printSection("Complex Pack (Binary: i, q)");
  runTest([] { testComplexPackRoundtrip<real32_t>(); }, "Complex Roundtrip (Real32)");
  runTest([] { testComplexPackRoundtrip<real64_t>(); }, "Complex Roundtrip (Real64)");

  TestPrinter::printSection("AOT Multi-Input (CPU)");
  testAotConvolveRR<real32_t>();
  testAotConvolveRR<real64_t>();
  testAotComplexPack<real32_t>();
  testAotComplexPack<real64_t>();

  // --- Convolve 混合类型 ---
  TestPrinter::printSection("Convolve (Mixed Types)");
  runTest([]() { testConvolveCase<real32_t, real32_t, real32_t>(); },
          mixedOpName<real32_t, real32_t>("Convolve"));
  runTest([]() { testConvolveCase<real32_t, real32_t, complex32_t>(); },
          mixedOpName<real32_t, complex32_t>("Convolve"));
  runTest([]() { testConvolveCase<real32_t, complex32_t, real32_t>(); },
          mixedOpName<complex32_t, real32_t>("Convolve"));
  runTest([]() { testConvolveCase<real32_t, complex32_t, complex32_t>(); },
          mixedOpName<complex32_t, complex32_t>("Convolve"));

  runTest([]() { testConvolveCase<real64_t, real64_t, real64_t>(); },
          mixedOpName<real64_t, real64_t>("Convolve"));
  runTest([]() { testConvolveCase<real64_t, real64_t, complex64_t>(); },
          mixedOpName<real64_t, complex64_t>("Convolve"));
  runTest([]() { testConvolveCase<real64_t, complex64_t, real64_t>(); },
          mixedOpName<complex64_t, real64_t>("Convolve"));
  runTest([]() { testConvolveCase<real64_t, complex64_t, complex64_t>(); },
          mixedOpName<complex64_t, complex64_t>("Convolve"));

  // --- Kron 混合类型 ---
  TestPrinter::printSection("Kron (Mixed Types)");
  runTest([]() { testKronCase<real32_t, real32_t, real32_t>(); },
          mixedOpName<real32_t, real32_t>("Kron"));
  runTest([]() { testKronCase<real32_t, real32_t, complex32_t>(); },
          mixedOpName<real32_t, complex32_t>("Kron"));
  runTest([]() { testKronCase<real32_t, complex32_t, real32_t>(); },
          mixedOpName<complex32_t, real32_t>("Kron"));
  runTest([]() { testKronCase<real32_t, complex32_t, complex32_t>(); },
          mixedOpName<complex32_t, complex32_t>("Kron"));

  runTest([]() { testKronCase<real64_t, real64_t, real64_t>(); },
          mixedOpName<real64_t, real64_t>("Kron"));
  runTest([]() { testKronCase<real64_t, real64_t, complex64_t>(); },
          mixedOpName<real64_t, complex64_t>("Kron"));
  runTest([]() { testKronCase<real64_t, complex64_t, real64_t>(); },
          mixedOpName<complex64_t, real64_t>("Kron"));
  runTest([]() { testKronCase<real64_t, complex64_t, complex64_t>(); },
          mixedOpName<complex64_t, complex64_t>("Kron"));

  TestPrinter::printSection("Long Composite Chain");
  runTest([]() { testLongCompositeChain<real32_t>(); },
          "Long Chain (Kron+Convolve+Add+Scale+Abs) (Real32)");
  runTest([]() { testLongCompositeChain<real64_t>(); },
          "Long Chain (Kron+Convolve+Add+Scale+Abs) (Real64)");

  TestPrinter::printSection("Boundary Tests");
  runTest([]() { testLengthOneSignal<real32_t>(); }, "Length-1 (Real32)");
  runTest([]() { testLengthOneSignal<real64_t>(); }, "Length-1 (Real64)");
  runTest([]() { testLargeSignal<real32_t>(); }, "Large Signal (Real32)");
  runTest([]() { testLargeSignal<real64_t>(); }, "Large Signal (Real64)");

  TestPrinter::printSummary();
  return 0;
}
