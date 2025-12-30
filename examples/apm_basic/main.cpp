/**
 * @file main.cpp
 * @ingroup examples
 * @brief QAM/PSK 基础链路示例
 *
 * 演示符号映射/解映射的正确性、CPU/GPU 性能对比与 BER 统计。
 * 默认阶数为 2（BPSK），也可配置为 4/16/64 等 QAM。
 */

#include <Halide.h>

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cxxopts.hpp>
#include <exception>
#include <iomanip>
#include <iostream>
#include <random>
#include <ratio>
#include <sstream>
#include <string>
#include <vector>

#include "prism/dsl/modem.h"
#include "prism/dsl/signal.h"
#include "prism/prism.h"
#include "prism/runtime/executor.h"
#include "prism/simulation/channel.h"
#include "prism/simulation/rng.h"
#include "prism/types.h"

using prism::real32_t;
using prism::dsl::Signal;
using prism::runtime::ExecMode;
using prism::runtime::Executor;
using prism::simulation::RNG;
using namespace prism::dsl;

namespace {

/** @brief 调制方式枚举 */
enum class ModemScheme : std::uint8_t { AUTO, QAM, PSK };

/** @brief 示例参数集合 */
struct ExampleArgs {
  int order = 2;                           ///< 调制阶数（默认 BPSK）
  int symbols = 4096;                      ///< 每次仿真的符号数量
  int iters = 50;                          ///< BER 统计迭代次数
  int perfIters = 50;                      ///< 性能统计迭代次数
  uint64_t seed = 42;                      ///< 随机种子
  bool enableGpu = true;                   ///< 是否尝试 GPU 模式
  ModemScheme scheme = ModemScheme::AUTO;  ///< 调制方式
  std::vector<double> snrList = {0.0, 5.0, 10.0, 15.0, 20.0};  ///< SNR 列表
};

/** @brief 解析逗号分隔的 SNR 列表 */
bool parseSnrList(const std::string& text, std::vector<double>& out) {
  out.clear();
  std::stringstream ss(text);
  std::string token;
  while (std::getline(ss, token, ',')) {
    if (token.empty()) continue;
    try {
      out.push_back(std::stod(token));
    } catch (const std::exception&) {
      return false;
    }
  }
  return !out.empty();
}

/** @brief 判断是否为 2 的幂 */
bool isPowerOfTwo(int value) { return value > 0 && (value & (value - 1)) == 0; }

/** @brief 判断是否为完全平方数 */
bool isPerfectSquare(int value) {
  if (value <= 0) return false;
  int const root = static_cast<int>(std::sqrt(value));
  return root * root == value;
}

/** @brief 计算每符号比特数（假设 order 为 2 的幂） */
int bitsPerSymbol(int order) {
  int bits = 0;
  while (order > 1) {
    order >>= 1;
    ++bits;
  }
  return bits;
}

/** @brief 生成随机符号索引 */
std::vector<real32_t> generateSymbols(int count, int order, RNG& rng) {
  std::uniform_int_distribution<int> dist(0, order - 1);
  std::vector<real32_t> symbols(count);
  for (int i = 0; i < count; ++i) {
    symbols[i] = static_cast<real32_t>(dist(rng.engine()));
  }
  return symbols;
}

/** @brief std::vector -> Halide::Buffer */
Halide::Buffer<real32_t> vectorToBuffer(const std::vector<real32_t>& data) {
  Halide::Buffer<real32_t> buf(static_cast<int>(data.size()));
  for (int i = 0; i < buf.width(); ++i) {
    buf(i) = data[static_cast<size_t>(i)];
  }
  return buf;
}

/** @brief Halide::Buffer -> std::vector */
std::vector<real32_t> bufferToVector(const Halide::Buffer<real32_t>& buf) {
  std::vector<real32_t> data(static_cast<size_t>(buf.width()));
  for (int i = 0; i < buf.width(); ++i) {
    data[static_cast<size_t>(i)] = buf(i);
  }
  return data;
}

/** @brief 统计 32 位整型的 bit 数 */
int popcount32(uint32_t value) {
  int count = 0;
  while (value != 0U) {
    count += static_cast<int>(value & 1U);
    value >>= 1U;
  }
  return count;
}

/** @brief 符号/比特误差统计 */
struct ErrorStats {
  int symbolErrors = 0;
  int64_t bitErrors = 0;
  int64_t totalBits = 0;
};

/** @brief 比较解调结果并统计误差 */
ErrorStats compareSymbols(const std::vector<real32_t>& expected,
                          const Halide::Buffer<real32_t>& demapped, int bits) {
  ErrorStats stats;
  uint32_t const mask = (bits >= 32) ? 0xFFFFFFFFU : ((1U << bits) - 1U);
  stats.totalBits = static_cast<int64_t>(expected.size()) * bits;

  for (int i = 0; i < demapped.width(); ++i) {
    int const expSym = static_cast<int>(expected[static_cast<size_t>(i)]);
    int const gotSym = static_cast<int>(std::lround(demapped(i)));
    if (expSym != gotSym) {
      stats.symbolErrors++;
    }
    uint32_t const diff = static_cast<uint32_t>(expSym ^ gotSym) & mask;
    stats.bitErrors += popcount32(diff);
  }
  return stats;
}

/** @brief 计时工具（返回平均毫秒） */
template <typename Func>
double measureMs(Func&& func, int iterations) {
  func();
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iterations; ++i) {
    func();
  }
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration<double, std::milli>(end - start).count() /
         iterations;
}

/** @brief 判断是否启用了 GPU 后端 */
bool gpuAvailable() {
#if defined(PRISM_GPU_BACKEND_Metal) || defined(PRISM_GPU_BACKEND_CUDA) || \
    defined(PRISM_GPU_BACKEND_OpenCL)
  return true;
#else
  return false;
#endif
}

/** @brief 输出性能统计 */
void printPerf(const std::string& label, double ms, int symbols) {
  double const symPerSec = (symbols / (ms / 1000.0)) / 1e6;
  std::cout << "  " << std::setw(12) << label << ": " << std::fixed
            << std::setprecision(3) << ms << " ms, " << std::setprecision(2)
            << symPerSec << " MSym/s\n";
}

}  // namespace

int main(int argc, char** argv) {
  ExampleArgs args;
  const std::string defaultSnr = "0,5,10,15,20";

  cxxopts::Options options("example_apm_basic", "QAM/PSK 基础链路示例");
  options.add_options()("order", "调制阶数",
                        cxxopts::value<int>()->default_value("2"))(
      "scheme", "auto/qam/psk",
      cxxopts::value<std::string>()->default_value("auto"))(
      "symbols", "每轮符号数量", cxxopts::value<int>()->default_value("4096"))(
      "snr", "SNR 列表 (dB, 逗号分隔)",
      cxxopts::value<std::string>()->default_value(defaultSnr))(
      "iters", "BER 迭代次数", cxxopts::value<int>()->default_value("50"))(
      "perf-iters", "性能统计迭代次数",
      cxxopts::value<int>()->default_value("50"))(
      "seed", "随机种子", cxxopts::value<uint64_t>()->default_value("42"))(
      "no-gpu", "跳过 GPU 模式",
      cxxopts::value<bool>()->default_value("false")->implicit_value("true"))(
      "h,help", "显示帮助");

  cxxopts::ParseResult result;
  try {
    result = options.parse(argc, argv);
  } catch (const cxxopts::exceptions::exception& e) {
    std::cerr << "参数解析失败: " << e.what() << "\n";
    std::cout << options.help() << "\n";
    return 1;
  }

  if (result.contains("help")) {
    std::cout << options.help() << "\n";
    return 0;
  }

  args.order = result["order"].as<int>();
  args.symbols = result["symbols"].as<int>();
  args.iters = result["iters"].as<int>();
  args.perfIters = result["perf-iters"].as<int>();
  args.seed = result["seed"].as<uint64_t>();
  args.enableGpu = !result["no-gpu"].as<bool>();

  std::string const schemeText = result["scheme"].as<std::string>();
  if (schemeText == "auto") {
    args.scheme = ModemScheme::AUTO;
  } else if (schemeText == "qam") {
    args.scheme = ModemScheme::QAM;
  } else if (schemeText == "psk") {
    args.scheme = ModemScheme::PSK;
  } else {
    std::cerr << "未知 scheme: " << schemeText << "\n";
    return 1;
  }

  std::string const snrText = result["snr"].as<std::string>();
  if (!parseSnrList(snrText, args.snrList)) {
    std::cerr << "SNR 列表解析失败\n";
    return 1;
  }

  if (args.order <= 1 || !isPowerOfTwo(args.order)) {
    std::cerr << "调制阶数必须是 2 的幂且大于 1\n";
    return 1;
  }
  if (args.symbols <= 0 || args.iters <= 0 || args.perfIters <= 0) {
    std::cerr << "symbols/iters/perf-iters 必须为正数\n";
    return 1;
  }

  ModemScheme scheme = args.scheme;
  if (scheme == ModemScheme::AUTO) {
    scheme = (args.order == 2) ? ModemScheme::PSK : ModemScheme::QAM;
  }
  if (scheme == ModemScheme::QAM && !isPerfectSquare(args.order)) {
    std::cerr << "QAM 阶数必须是完全平方数\n";
    return 1;
  }

  prism::initialize();

  int const bits = bitsPerSymbol(args.order);
  int const iqLen = args.symbols * 2;

  Signal const input = Signal::input(args.symbols);
  Signal const mapSig = (scheme == ModemScheme::PSK)
                            ? modem::pskMap(input, args.order)
                            : modem::qamMap(input, args.order);
  Signal const demapSig =
      (scheme == ModemScheme::PSK)
          ? modem::pskDemap(Signal::input(iqLen), args.order)
          : modem::qamDemap(Signal::input(iqLen), args.order);

  auto mapCpu = Executor::compile<real32_t>(mapSig, ExecMode::CPU);
  auto demapCpu = Executor::compile<real32_t>(demapSig, ExecMode::CPU);

  std::cout << "=== PRISM 示例: QAM/PSK 基础链路 ===\n\n";
  std::cout << "配置:\n";
  std::cout << "  scheme: " << ((scheme == ModemScheme::PSK) ? "PSK" : "QAM")
            << "\n";
  std::cout << "  order: " << args.order << " (bits/sym=" << bits << ")\n";
  std::cout << "  symbols: " << args.symbols << "\n";
  std::cout << "  iters: " << args.iters << "\n";
  std::cout << "  perf iters: " << args.perfIters << "\n";
  std::cout << "  seed: " << args.seed << "\n";
  std::cout << "  backend: " << prism::getBackendName() << "\n";
  std::cout << "  gpu: " << (gpuAvailable() ? "可用" : "未启用") << "\n\n";

  // --------------------------------------------------------------------------
  // 正确性验证（无噪声往返）
  // --------------------------------------------------------------------------
  RNG rng(args.seed);
  auto symbols = generateSymbols(args.symbols, args.order, rng);
  auto inputBuf = vectorToBuffer(symbols);

  auto mapped = mapCpu.run(inputBuf);
  auto demapped = demapCpu.run(mapped);

  ErrorStats const roundtrip = compareSymbols(symbols, demapped, bits);
  bool const pass = (roundtrip.symbolErrors == 0);

  std::cout << "正确性验证: " << (pass ? "PASS" : "FAIL") << "\n";
  std::cout << "  symbol errors: " << roundtrip.symbolErrors << "\n";
  std::cout << "  bit errors: " << roundtrip.bitErrors << "\n\n";

  // --------------------------------------------------------------------------
  // 性能对比
  // --------------------------------------------------------------------------
  std::cout << "性能对比 (CPU):\n";
  double const mapCpuMs =
      measureMs([&]() { mapCpu.run(inputBuf); }, args.perfIters);
  double const demapCpuMs =
      measureMs([&]() { demapCpu.run(mapped); }, args.perfIters);
  double const e2eCpuMs = measureMs(
      [&]() {
        auto iq = mapCpu.run(inputBuf);
        demapCpu.run(iq);
      },
      args.perfIters);
  printPerf("Map", mapCpuMs, args.symbols);
  printPerf("Demap", demapCpuMs, args.symbols);
  printPerf("End-to-End", e2eCpuMs, args.symbols);

  if (args.enableGpu && gpuAvailable()) {
    try {
      auto mapGpu = Executor::compile<real32_t>(mapSig, ExecMode::GPU);
      auto demapGpu = Executor::compile<real32_t>(demapSig, ExecMode::GPU);

      std::cout << "\n性能对比 (GPU):\n";
      double const mapGpuMs =
          measureMs([&]() { mapGpu.run(inputBuf); }, args.perfIters);
      double const demapGpuMs =
          measureMs([&]() { demapGpu.run(mapped); }, args.perfIters);
      double const e2eGpuMs = measureMs(
          [&]() {
            auto iq = mapGpu.run(inputBuf);
            demapGpu.run(iq);
          },
          args.perfIters);

      printPerf("Map", mapGpuMs, args.symbols);
      printPerf("Demap", demapGpuMs, args.symbols);
      printPerf("End-to-End", e2eGpuMs, args.symbols);
    } catch (const std::exception& e) {
      std::cout << "\nGPU 模式不可用: " << e.what() << "\n";
    }
  } else {
    std::cout << "\nGPU 模式跳过\n";
  }

  // --------------------------------------------------------------------------
  // BER 仿真
  // --------------------------------------------------------------------------
  std::cout << "\nBER 仿真:\n";
  std::cout << "  SNR(dB)      BER        BitErrors/TotalBits\n";

  for (double const snrDb : args.snrList) {
    int64_t totalBits = 0;
    int64_t totalErrors = 0;

    for (int iter = 0; iter < args.iters; ++iter) {
      auto symbolsIter = generateSymbols(args.symbols, args.order, rng);
      auto inputIter = vectorToBuffer(symbolsIter);
      auto iq = mapCpu.run(inputIter);

      auto iqVec = bufferToVector(iq);
      auto noisy = prism::simulation::awgn(iqVec, snrDb, &rng);
      auto noisyBuf = vectorToBuffer(noisy);
      auto demapOut = demapCpu.run(noisyBuf);

      ErrorStats const stats = compareSymbols(symbolsIter, demapOut, bits);
      totalBits += stats.totalBits;
      totalErrors += stats.bitErrors;
    }

    double const ber = (totalBits == 0) ? 0.0
                                        : static_cast<double>(totalErrors) /
                                              static_cast<double>(totalBits);
    std::cout << "  " << std::setw(6) << std::fixed << std::setprecision(1)
              << snrDb << "   " << std::scientific << std::setprecision(3)
              << ber << "   " << std::fixed << totalErrors << "/" << totalBits
              << "\n";
  }

  prism::shutdown();
  return 0;
}
