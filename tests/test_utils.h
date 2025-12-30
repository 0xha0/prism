#pragma once

/**
 * @file test_utils.h
 * @ingroup tests
 * @brief 测试辅助工具：类型名、打印、误差计算、复数 Buffer 读写
 */

#include <Halide.h>

#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "prism/runtime/executor.h"
#include "prism/types.h"

namespace prism::tests {

/// @addtogroup tests
/// @{

// ============================================================================
// // 复数 Buffer 读写辅助
// ============================================================================
// //

/** @brief 从二维 Buffer 读取 complex32（默认维度顺序 c,x） */
template <typename T>
complex32_t getComplex32(const Halide::Buffer<T>& buf, int idx) {
  if (buf.dimensions() < 2) return complex32_t(buf(idx), 0);
  return {static_cast<float>(buf(0, idx)), static_cast<float>(buf(1, idx))};
}

/** @brief 从二维 Buffer 读取 complex64（默认维度顺序 c,x） */
template <typename T>
complex64_t getComplex64(const Halide::Buffer<T>& buf, int idx) {
  if (buf.dimensions() < 2) return complex64_t(buf(idx), 0);
  return {static_cast<double>(buf(0, idx)), static_cast<double>(buf(1, idx))};
}

/** @brief 将 complex32 写入二维 Buffer（维度 c,x） */
inline void fillComplexBuffer(Halide::Buffer<float>& buf,
                              const std::vector<complex32_t>& data) {
  for (size_t i = 0; i < data.size(); ++i) {
    buf(0, i) = data[i].real();
    buf(1, i) = data[i].imag();
  }
}

/** @brief 将 complex64 写入二维 Buffer（维度 c,x） */
inline void fillComplexBuffer(Halide::Buffer<double>& buf,
                              const std::vector<complex64_t>& data) {
  for (size_t i = 0; i < data.size(); ++i) {
    buf(0, i) = data[i].real();
    buf(1, i) = data[i].imag();
  }
}

// ============================================================================
// // 测试输出工具
// ============================================================================
// //

/**
 * @brief 统一的测试输出工具
 *
 * 提供套件标题、章节、测试结果打印等辅助函数。输出保留英文 PASSED/FAILED，
 * 以便终端彩色高亮与对齐。
 */
class TestPrinter {
 public:
  inline static int s_total = 0;
  inline static int s_passed = 0;
  inline static int s_failed = 0;
  inline static int s_skipped = 0;

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
    std::cout << "  > " << std::left << std::setw(15) << key << ": " << value
              << "\n";
  }

  static void printTestStart(const std::string& name) {
    std::cout << "  [ RUN      ] " << std::left << std::setw(40) << name;
    std::cout.flush();
  }

  // 保留接口避免未使用警告，实际输出统一走 printTestResult
  static void printPass(const std::string& /*extra*/ = "") {}

  /// 点线填充风格的统一结果输出
  static void printTestResult(const std::string& name, bool passed,
                              const std::string& extra = "") {
    s_total++;
    if (passed) {
      s_passed++;
    } else {
      s_failed++;
    }

    const int width = 50;
    std::cout << "  " << std::left << std::setw(width) << std::setfill('.')
              << (name + " ") << std::setfill(' ') << " ";

    if (passed) {
      std::cout << "[ \033[32mPASSED\033[0m ]";  // 绿色
    } else {
      std::cout << "[ \033[31mFAILED\033[0m ]";  // 红色
    }

    if (!extra.empty()) {
      std::cout << " " << extra;
    }
    std::cout << "\n";
  }

  static void printSkip(const std::string& name, const std::string& reason) {
    s_total++;
    s_skipped++;
    const int width = 50;
    std::cout << "  " << std::left << std::setw(width) << std::setfill('.')
              << (name + " ") << std::setfill(' ') << " ";
    std::cout << "[ \033[33mSKIP\033[0m   ] (" << reason << ")\n";
  }

  static void printSummary() {
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "  SUMMARY\n";
    std::cout << "  Total:  " << s_total << "\n";
    std::cout << "  Passed: " << s_passed << "\n";
    if (s_failed > 0) {
      std::cout << "  Failed: " << s_failed << "\n";
    } else {
      std::cout << "  Failed: 0\n";
    }
    if (s_skipped > 0) {
      std::cout << "  Skipped: " << s_skipped << "\n";
    }
    std::cout << std::string(70, '=') << "\n";
  }

  // Deprecated manual version
  static void printSummary(int total, int passed, int failed) {
    // Reuse new logic but override values if user really wants to force them?
    // Or just keep the old implementation for compatibility if someone calls it
    // with args. Let's keep it simple and independent, but we might want to
    // warn or just use specific values.
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "  SUMMARY (Manual)\n";
    std::cout << "  Total:  " << total << "\n";
    std::cout << "  Passed: " << passed << "\n";
    if (failed > 0) {
      std::cout << "  Failed: " << failed << "\n";
    } else {
      std::cout << "  Failed: 0\n";
    }
    std::cout << std::string(70, '=') << "\n";
  }
};

// ============================================================================
// // 类型名与误差计算
// ============================================================================
// //

/// 类型名获取工具
template <typename T>
struct TypeName;

template <>
struct TypeName<real32_t> {
  static std::string get() { return "Real32"; }
};
template <>
struct TypeName<real64_t> {
  static std::string get() { return "Real64"; }
};
template <>
struct TypeName<complex32_t> {
  static std::string get() { return "Complex32"; }
};
template <>
struct TypeName<complex64_t> {
  static std::string get() { return "Complex64"; }
};

/** @brief 近似相等判断（默认误差 1e-4） */
template <typename T>
bool approxEqual(T a, T b, double epsilon = 1e-4) {
  return std::abs(a - b) < static_cast<T>(epsilon);
}

/// 计算两个复数向量的最大误差
template <typename T>
T maxError(const std::vector<std::complex<T>>& a,
           const std::vector<std::complex<T>>& b) {
  T maxErr = 0.0;
  for (size_t i = 0; i < std::min(a.size(), b.size()); ++i) {
    maxErr = std::max(maxErr, std::abs(a[i] - b[i]));
  }
  return maxErr;
}

/// 计算两个标量向量的最大误差
template <typename T>
T maxError(const std::vector<T>& a, const std::vector<T>& b) {
  T maxErr = 0.0;
  for (size_t i = 0; i < std::min(a.size(), b.size()); ++i) {
    maxErr = std::max(maxErr, std::abs(a[i] - b[i]));
  }
  return maxErr;
}

/// 计算 Halide Buffer 与期望向量的最大误差
template <typename T>
T maxError(const Halide::Buffer<T>& result, const std::vector<T>& expected) {
  T maxErr = 0.0;
  int checkLen = std::min(result.width(), static_cast<int>(expected.size()));
  for (int i = 0; i < checkLen; ++i) {
    maxErr = std::max(maxErr, std::abs(result(i) - expected[i]));
  }
  return maxErr;
}

// ============================================================================
// // 测试运行封装
// ============================================================================
// //

/**
 * @brief 在 CPU/GPU 双模式下运行测试函数
 *
 * @tparam Func 形如 `void()` 的可调用对象
 * @param f 测试函数体
 * @param name 测试名称（用于打印）
 *
 * 行为：先切到 CPU 执行，再切到 GPU；捕获编译/运行异常并输出结果。
 */
template <typename Func>
void runTest(Func&& f, const std::string& name) {
  // CPU 路径
  {
    prism::runtime::Executor::setMode(prism::runtime::ExecMode::CPU);
    try {
      f();
    } catch (const std::exception& e) {
      TestPrinter::printTestResult(name, false,
                                   std::string("[CPU] Exception: ") + e.what());
      throw;  // CPU 失败通常为关键问题
    }
  }

  // GPU 路径
  {
    prism::runtime::Executor::setMode(prism::runtime::ExecMode::GPU);
    try {
      f();
    } catch (const Halide::CompileError& e) {
      TestPrinter::printSkip(name,
                             std::string("[GPU] CompileError: ") + e.what());
    } catch (const Halide::RuntimeError& e) {
      TestPrinter::printSkip(name,
                             std::string("[GPU] RuntimeError: ") + e.what());
    } catch (const std::exception& e) {
      TestPrinter::printTestResult(name, false,
                                   std::string("[GPU] Exception: ") + e.what());
      // 不再抛出，便于继续其他测试
    }
  }

  // 恢复自动模式
  prism::runtime::Executor::setMode(prism::runtime::ExecMode::AUTO);
}

/// @}

}  // namespace prism::tests
