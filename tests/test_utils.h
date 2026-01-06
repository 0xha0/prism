/**
 * @file test_utils.h
 * @ingroup tests
 * @brief 测试辅助工具集
 *
 * 包含用于单元测试的通用工具类和函数，主要功能包括：
 * - 类型特征 (Traits) 检测与元编程辅助
 * - Halide Buffer 数据填充与读取的统一接口
 * - 浮点数/复数误差计算与近似相等判断
 * - 测试结果格式化输出 (TestPrinter)
 */

#include <Halide.h>

#include <algorithm>
#include <complex>
#include <iomanip>
#include <iostream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "prism/runtime/executor.h"
#include "prism/types.h"

namespace prism::tests {

/// @addtogroup tests
/// @{

// ============================================================================
// 类型特征 (Type Traits)
// ============================================================================
/**
 * @brief 获取类型的字符串表示
 *
 * 用于测试日志输出
 */
template <typename T>
struct TypeName {
  static std::string get() {
    if constexpr (std::is_same_v<T, real32_t>) {
      return "Real32";
    } else if constexpr (std::is_same_v<T, real64_t>) {
      return "Real64";
    } else if constexpr (std::is_same_v<T, complex32_t>) {
      return "Complex32";
    } else if constexpr (std::is_same_v<T, complex64_t>) {
      return "Complex64";
    } else {
      return "Unknown";
    }
  }
};

// ============================================================================
// 复数 Buffer 读写 (Buffer Access Helpers)
// ============================================================================

/**
 * @brief 从 Halide Buffer 读取复数元素
 *
 * 兼容处理 1D Buffer (交织存储? 视具体实现而定) 和 2D Buffer (维度0 为 I/Q
 * 通道)
 *
 * @tparam ComplexT 目标复数类型 (complex32_t 或 complex64_t)
 * @tparam BufT Buffer 元素类型
 * @param buf 输入 Buffer
 * @param idx 元素索引
 * @return ComplexT 读取的复数值
 */
template <typename ComplexT, typename BufT>
ComplexT getComplex(const Halide::Buffer<BufT>& buf, int idx) {
  using RealT = typename ComplexT::value_type;
  if (buf.dimensions() < 2) {
    // 假设 1D 情况为实部 (或出错，视具体逻辑)，这里保留原逻辑
    return ComplexT(static_cast<RealT>(buf(idx)), 0);
  }
  // 维度 0 必须是 size 2 (I, Q)
  return {static_cast<RealT>(buf(0, idx)), static_cast<RealT>(buf(1, idx))};
}

/**
 * @brief 将复数向量写入 Halide Buffer
 *
 * 默认将数据写入 Buffer 的前两行 (Dim 0)，分别对应实部和虚部
 *
 * @tparam RealT Buffer 基础数据类型
 * @tparam ComplexT 输入向量元素类型
 */
template <typename RealT, typename ComplexT>
void fillComplexBuffer(Halide::Buffer<RealT>& buf, const std::vector<ComplexT>& data) {
  static_assert(std::is_same_v<RealT, typename ComplexT::value_type>,
                "Buffer element type must match complex value_type");
  for (size_t i = 0; i < data.size(); ++i) {
    buf(0, static_cast<int>(i)) = data[i].real();
    buf(1, static_cast<int>(i)) = data[i].imag();
  }
}

// ============================================================================
// 测试适配器 (Test Traits Adapter)
// ============================================================================

/**
 * @brief 测试类型适配器 - 实数特化
 *
 * 统一不同数据类型在测试中的 Buffer 创建、填充和读取行为
 */
template <typename T>
struct TestTraits {
  using ElementType = T;
  using BufferElemType = T;
  static constexpr bool IS_COMPLEX = false;

  static ScalarType scalarType() {
    return std::is_same_v<T, real32_t> ? ScalarType::F32 : ScalarType::F64;
  }

  static Halide::Buffer<T> makeBuffer(size_t len) {
    return Halide::Buffer<T>(static_cast<int>(len));
  }

  static void fillBuffer(Halide::Buffer<T>& buf, const std::vector<T>& data) {
    for (size_t i = 0; i < data.size(); ++i) {
      buf(static_cast<int>(i)) = data[i];
    }
  }

  static T getElement(const Halide::Buffer<T>& buf, int idx) { return buf(idx); }
};

/**
 * @brief 测试类型适配器 - Complex32 特化
 */
template <>
struct TestTraits<complex32_t> {
  using ElementType = complex32_t;
  using BufferElemType = real32_t;
  static constexpr bool IS_COMPLEX = true;

  static ScalarType scalarType() { return ScalarType::C32; }

  static Halide::Buffer<real32_t> makeBuffer(size_t len) {
    // 创建 (2, N) 形状的 Buffer，Dim 0 为 I/Q
    return Halide::Buffer<real32_t>(2, static_cast<int>(len));
  }

  static void fillBuffer(Halide::Buffer<real32_t>& buf, const std::vector<complex32_t>& data) {
    fillComplexBuffer(buf, data);
  }

  static complex32_t getElement(const Halide::Buffer<real32_t>& buf, int idx) {
    return getComplex<complex32_t>(buf, idx);
  }
};

/**
 * @brief 测试类型适配器 - Complex64 特化
 */
template <>
struct TestTraits<complex64_t> {
  using ElementType = complex64_t;
  using BufferElemType = real64_t;
  static constexpr bool IS_COMPLEX = true;

  static ScalarType scalarType() { return ScalarType::C64; }

  static Halide::Buffer<real64_t> makeBuffer(size_t len) {
    return Halide::Buffer<real64_t>(2, static_cast<int>(len));
  }

  static void fillBuffer(Halide::Buffer<real64_t>& buf, const std::vector<complex64_t>& data) {
    fillComplexBuffer(buf, data);
  }

  static complex64_t getElement(const Halide::Buffer<real64_t>& buf, int idx) {
    return getComplex<complex64_t>(buf, idx);
  }
};

// ============================================================================
// 误差计算 (Error Calculation)
// ============================================================================

/**
 * @brief 计算两个 std::vector 之间的最大绝对误差 (L-inf Norm)
 */
template <typename T>
auto maxError(const std::vector<T>& a, const std::vector<T>& b) {
  using ResultT = decltype(std::abs(a[0] - b[0]));
  ResultT maxErr = 0;
  for (size_t i = 0; i < std::min(a.size(), b.size()); ++i) {
    maxErr = std::max(maxErr, std::abs(a[i] - b[i]));
  }
  return maxErr;
}

/**
 * @brief 计算 Halide Buffer 与 期望向量 之间的最大绝对误差
 */
template <typename T>
T maxError(const Halide::Buffer<T>& result, const std::vector<T>& expected) {
  T maxErr = 0;
  int checkLen = std::min(result.width(), static_cast<int>(expected.size()));
  for (int i = 0; i < checkLen; ++i) {
    maxErr = std::max(maxErr, std::abs(result(i) - expected[i]));
  }
  return maxErr;
}

/**
 * @brief 判断两个浮点数是否近似相等
 * @param epsilon 允许的最大绝对误差
 */
template <typename T>
bool approxEqual(T a, T b, double epsilon = 1e-4) {
  return std::abs(a - b) < static_cast<decltype(std::abs(a))>(epsilon);
}

// ============================================================================
// 测试日志打印 (Test Logging)
// ============================================================================

/**
 * @brief 简单的测试结果打印器
 *
 * 负责输出格式化的测试套件头、测试用例结果（PASS/FAIL/SKIP）以及最终统计摘要
 */
class TestPrinter {
 public:
  inline static int sTotalCount = 0;
  inline static int sPassedCount = 0;
  inline static int sFailedCount = 0;
  inline static int sSkippedCount = 0;

  static void printSuiteHeader(const std::string& name) {
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "  TEST SUITE: " << name << "\n";
    std::cout << std::string(70, '=') << "\n\n";
  }

  static void printSection(const std::string& name) {
    std::cout << "\n" << std::string(50, '-') << "\n";
    std::cout << "  " << name << "\n";
    std::cout << std::string(50, '-') << "\n";
  }

  static void printInfo(const std::string& key, const std::string& value) {
    std::cout << "  > " << std::left << std::setw(15) << key << ": " << value << "\n";
  }

  static void printTestResult(const std::string& name, bool passed, const std::string& extra = "",
                              const std::string& label = "") {
    sTotalCount++;
    passed ? sPassedCount++ : sFailedCount++;
    std::string const suffix = label.empty() ? "" : (" " + label);
    constexpr int kWidth = 50;
    std::cout << "  " << std::left << std::setw(kWidth) << std::setfill('.')
              << (name + suffix + " ") << std::setfill(' ') << " ";
    std::cout << (passed ? "[ \033[32mPASSED\033[0m ]" : "[ \033[31mFAILED\033[0m ]");
    if (!extra.empty()) std::cout << " " << extra;
    std::cout << "\n";
  }

  static void printSkip(const std::string& name, const std::string& reason) {
    sTotalCount++;
    sSkippedCount++;
    constexpr int kWidth = 50;
    std::cout << "  " << std::left << std::setw(kWidth) << std::setfill('.') << (name + " ")
              << std::setfill(' ') << " ";
    std::cout << "[ \033[33mSKIP\033[0m   ] (" << reason << ")\n";
  }

  static void printSummary() {
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "  SUMMARY\n";
    std::cout << "  Total:  " << sTotalCount << "\n";
    std::cout << "  Passed: " << sPassedCount << "\n";
    std::cout << "  Failed: " << sFailedCount << "\n";
    if (sSkippedCount > 0) std::cout << "  Skipped: " << sSkippedCount << "\n";
    std::cout << std::string(70, '=') << "\n";
  }
};

// ============================================================================
// 测试运行封装 (Test Runner)
// ============================================================================

/**
 * @brief 在 CPU 模式下运行测试函数
 *
 * 封装了异常捕获和结果打印逻辑，确保测试套件即使遇到异常也能继续运行 (Soft
 * Fail)
 *
 * @tparam Func 测试函数类型 (无参, void 返回)
 * @param f 测试执行体
 * @param name 测试用例名称
 */
template <typename Func>
void runTest(Func&& f, const std::string& name) {
  // 强制与默认使用 CPU 后端，避免环境差异
  prism::runtime::Executor::setMode(prism::runtime::ExecMode::CPU);
  try {
    std::forward<Func>(f)();
  } catch (const Halide::CompileError& e) {
    TestPrinter::printTestResult(name, false, std::string("CompileError: ") + e.what());
  } catch (const Halide::RuntimeError& e) {
    TestPrinter::printTestResult(name, false, std::string("RuntimeError: ") + e.what());
  } catch (const std::exception& e) {
    TestPrinter::printTestResult(name, false, std::string("Exception: ") + e.what());
  }
  prism::runtime::Executor::setMode(prism::runtime::ExecMode::AUTO);
}

/// @}

}  // namespace prism::tests
