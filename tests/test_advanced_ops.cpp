/**
 * @file test_advanced_ops.cpp
 * @ingroup tests
 * @brief 复合算子链路单元测试
 *
 * 验证多算子组合与边界条件，支持多种标量类型。
 */

#include <Halide.h>

#include <cassert>
#include <string>
#include <vector>

#include "prism/dsl/filter.h"
#include "prism/dsl/modem.h"
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
// 复合链路测试
// ============================================================================

/** @brief FIR + Scale 链路响应测试 */
template <typename T>
void testFirScaleChain() {
  std::string const name = "FIR + Scale Chain (" + TypeName<T>::get() + ")";

  std::vector<T> const taps = {static_cast<T>(0.25), static_cast<T>(0.5),
                               static_cast<T>(0.25)};
  int const len = 20;

  Halide::Buffer<T> inputBuf(len);
  for (int i = 0; i < len; ++i) {
    inputBuf(i) = (i == len / 2) ? static_cast<T>(1.0) : static_cast<T>(0.0);
  }

  Signal const x = Signal::input(len);
  // Need to handle type-specific taps? The DSL expects std::vector<float> or
  // double. We can convert our generic vector to the specific type if needed,
  // but if T is real32_t/real64_t it matches directly.

  Signal filtered;
  // NOLINTNEXTLINE(bugprone-branch-clone)
  if constexpr (std::is_same_v<T, real64_t>) {
    // Need to be careful if dsl::filter::fir has overloads.
    // Yes, it has overloads for vector<real32_t> and vector<real64_t>.
    filtered = filter::fir(x, taps);
  } else {
    // Convert T vector to real32_t vector if T is real32_t.
    // Actually taps IS vector<T> so it matches directly.
    filtered = filter::fir(x, taps);
  }

  Signal const scaled = scale(filtered, 2.0);

  auto result = Executor::run<T>(scaled, inputBuf);

  // Find peak
  T maxVal = static_cast<T>(0.0);
  int peakIdx = -1;
  for (int i = 0; i < result.width(); ++i) {
    if (result(i) > maxVal) {
      maxVal = result(i);
      peakIdx = i;
    }
  }

  bool pass = true;
  // Impulse response: taps * 2 => Peak: 0.5 * 2 = 1.0
  if (!approxEqual(maxVal, static_cast<T>(1.0), static_cast<T>(1e-2))) {
    pass = false;
  }

  // Sides: 0.25 * 2 = 0.5
  if (peakIdx > 0) {
    if (!approxEqual(result(peakIdx - 1), static_cast<T>(0.5),
                     static_cast<T>(1e-2))) {
      pass = false;
    }
  }
  if (peakIdx < result.width() - 1) {
    if (!approxEqual(result(peakIdx + 1), static_cast<T>(0.5),
                     static_cast<T>(1e-2))) {
      pass = false;
    }
  }

  TestPrinter::printTestResult(name, pass);
  assert(pass);
}

/** @brief QAM 映射 + 噪声扰动的解映射稳定性 */
template <typename T>
void testQamNoiseTolerance() {
  std::string const name = "QAM Noise Tolerance (" + TypeName<T>::get() + ")";
  int const nSyms = 4;
  Halide::Buffer<T> inputBuf(nSyms);
  for (int i = 0; i < nSyms; ++i) {
    inputBuf(i) = static_cast<T>(i);
  }

  Signal const x = Signal::input(nSyms);
  Signal const mapped = modem::qamMap(x, 4);

  auto iq = Executor::run<T>(mapped, inputBuf);

  // Add small noise
  Halide::Buffer<T> noisyIq(iq.width());
  for (int i = 0; i < iq.width(); ++i) {
    noisyIq(i) =
        iq(i) + (static_cast<T>(0.1) *
                 (((i % 2) != 0) ? static_cast<T>(1.0) : static_cast<T>(-1.0)));
  }

  Signal const noisyIn = Signal::input(iq.width());
  Signal const demapped = modem::qamDemap(noisyIn, 4);

  auto result = Executor::run<T>(demapped, noisyIq);

  // Small noise should not affect hard decision
  bool pass = true;
  for (int i = 0; i < nSyms; ++i) {
    auto const expected = static_cast<T>(i);
    if (!approxEqual(result(i), expected)) pass = false;
  }

  TestPrinter::printTestResult(name, pass);
  assert(pass);
}

// ============================================================================
// Boundary Condition Tests
// ============================================================================

/** @brief 长度为 1 的信号路径覆盖 */
template <typename T>
void testLengthOneSignal() {
  std::string const name = "Length-1 Signal (" + TypeName<T>::get() + ")";
  Halide::Buffer<T> inputBuf(1);
  inputBuf(0) = static_cast<T>(42.0);

  Signal const x = Signal::input(1);
  Signal const y = scale(x, 2.0);

  auto result = Executor::run<T>(y, inputBuf);

  bool pass = true;
  if (result.width() != 1) pass = false;
  if (!approxEqual(result(0), static_cast<T>(84.0))) pass = false;

  TestPrinter::printTestResult(name, pass);
  assert(pass);
}

/** @brief 10 万点长信号的吞吐冒烟 */
template <typename T>
void testLargeSignal() {
  std::string const name = "Large Signal (100K) (" + TypeName<T>::get() + ")";
  int const len = 100000;
  Halide::Buffer<T> inputBuf(len);
  // Using explicit loop or generic fill if available?
  // We'll stick to loop for clarity and direct T usage
  for (int i = 0; i < len; ++i) {
    inputBuf(i) = static_cast<T>(1.0);
  }

  Signal const x = Signal::input(len);
  Signal const y = scale(x, 0.5);

  auto result = Executor::run<T>(y, inputBuf);

  bool pass = true;
  if (result.width() != len) pass = false;
  if (!approxEqual(result(0), static_cast<T>(0.5))) pass = false;
  if (!approxEqual(result(len - 1), static_cast<T>(0.5))) pass = false;

  TestPrinter::printTestResult(name, pass);
  assert(pass);
}

// ============================================================================
// Main
// ============================================================================

int main() {
  TestPrinter::printSuiteHeader("Advanced Ops Tests (Refactored)");

  // Helper macro to run parameter-less tests
  auto run = [](auto func, const std::string& name) {
    TestPrinter::printSection(name + " [Running on CPU & GPU]");
    runTest(func, name);
  };

  run([]() { testFirScaleChain<real32_t>(); }, "FIR+Scale (Real32)");
  run([]() { testQamNoiseTolerance<real32_t>(); }, "QAM Noise (Real32)");

  run([]() { testFirScaleChain<real64_t>(); }, "FIR+Scale (Real64)");
  run([]() { testQamNoiseTolerance<real64_t>(); }, "QAM Noise (Real64)");

  run([]() { testLengthOneSignal<real32_t>(); }, "Length-1 (Real32)");
  run([]() { testLargeSignal<real32_t>(); }, "Large Signal (Real32)");

  run([]() { testLengthOneSignal<real64_t>(); }, "Length-1 (Real64)");
  run([]() { testLargeSignal<real64_t>(); }, "Large Signal (Real64)");

  TestPrinter::printSummary();
  return 0;
}

/// @}
