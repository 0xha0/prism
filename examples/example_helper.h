/**
 * @file example_helper.h
 * @ingroup examples
 * @brief 示例通用辅助接口与工具类
 *
 * 本文件提供了一套通用的辅助工具，用于简化 PRISM 示例程序的编写
 * 主要功能包括：
 * - 配置文件解析 (TOML)
 * - 命令行参数与配置参数的统一管理 (StandardArgs)
 * - 通用数字信号处理工具 (滤波器设计、混频、信道仿真)
 * - 性能基准测试与误码率 (BER) 统计框架
 * - 分步调试与数据导出工具 (StandardInspector)
 *
 * 通过使用 example_helper，示例程序可以聚焦于核心的 DSL 管道定义，
 * 而无需重复编写繁琐的配置加载、数据生成和验证逻辑
 */

#ifndef PRISM_EXAMPLES_EXAMPLE_HELPER_H
#define PRISM_EXAMPLES_EXAMPLE_HELPER_H

#include <Halide.h>
#include <toml++/toml.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iomanip>
#include <iostream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "halide_benchmark.h"
#include "prism/dsl/signal.h"
#include "prism/prism.h"
#include "prism/simulation/rng.h"
#include "prism/simulation/source.h"
#include "prism/types.h"

namespace prism::runtime {
enum class ExecMode : std::uint8_t;
struct SchedulerConfig;
}  // namespace prism::runtime

namespace prism::examples {

// SFINAE helper to detect .device_sync()
template <typename T, typename = void>
struct HAS_DEVICE_SYNC : std::false_type {};

template <typename T>
struct HAS_DEVICE_SYNC<T, std::void_t<decltype(std::declval<T>().device_sync())>> : std::true_type {
};

/// @addtogroup examples
/// @{

/**
 * @brief 调制方式枚举
 *
 * 用于指定物理层链路采用的调制类型
 *
 * @note AUTO 模式将根据阶数 (Order) 自动推断：
 * - 如果阶数是完全平方数 (如 4, 16, 64)，默认为 QAM
 * - 否则默认为 PSK
 */
enum class ModemScheme : std::uint8_t {
  AUTO,  ///< 自动推断 (基于 Order)
  QAM,   ///< 正交幅度调制 (Quadrature Amplitude Modulation)
  PSK    ///< 相移键控 (Phase Shift Keying)
};

/**
 * @brief 成形滤波器配置参数
 *
 * 定义发射端脉冲成形 (Pulse Shaping) 和接收端匹配滤波 (Matched Filter) 的参数
 */
struct FilterArgs {
  std::string mode = "rrc";  ///< 滤波器类型: "rrc" (根升余弦), "rc" (升余弦), "none" (直通)
  int span = 8;              ///< 滤波器截断长度 (符号数)，总 Taps 数通常为 span *
                             ///< samplesPerSymbol + 1
  double rolloff = 0.35;     ///< 滚降系数 (Beta)，决定带宽占用与时域振铃的平衡
  bool normalize = true;     ///< 是否归一化滤波器能量，保证滤波前后信号总能量基本不变
};

/**
 * @brief 信道仿真参数
 *
 * 用于配置软件信道模拟器的各种损伤模型
 */
struct ChannelArgs {
  bool enableAwgn = true;     ///< 是否启用高斯白噪声 (AWGN)
  bool enableFading = false;  ///< 是否启用多径衰落 (Flat Fading / Multipath)
  double gain = 1.0;          ///< 信道基础增益 (线性值)
  std::vector<real32_t> fadingTaps = {
      1.0F};                            ///< 多径信道系数 (复数模或实部均可定义，具体取决于实现)
  std::vector<int> fadingDelays = {0};  ///< 多径时延 (单位：采样点)
  double dopplerHz = 0.0;               ///< 最大多普勒频移 (Hz)
  double dopplerStartHz = 0.0;          ///< 动态多普勒起始频率 (Hz)
  double dopplerEndHz = 0.0;            ///< 动态多普勒终止频率 (Hz)
  double cfoHz = 0.0;                   ///< 载波频率偏差 (Carrier Frequency Offset, Hz)
  double phaseNoiseStd = 0.0;           ///< 相位噪声标准差 (弧度)
};

/**
 * @brief 数据导出配置
 *
 * 控制仿真过程中关键节点数据的导出，用于外部分析 (如 Python/MATLAB)
 */
struct OutputArgs {
  bool enable = false;             ///< 是否启用数据导出功能
  std::string dir = "outputs";     ///< 导出文件的存储目录
  std::vector<std::string> steps;  ///< 需要导出的步骤名称列表（若为空，则导出所有注册的步骤）
};

/**
 * @brief 误码统计结果
 *
 * 包含符号误差和比特误差的统计信息
 */
struct ErrorStats {
  int symbolErrors = 0;     ///< 错误符号总数
  int64_t bitErrors = 0;    ///< 错误比特总数
  int64_t totalBits = 0;    ///< 参与比较的总比特数
  int symbolsCompared = 0;  ///< 参与比较的符号数
};

/**
 * @brief 读取 TOML 数组并转换为指定类型的 std::vector
 *
 * @tparam Value 目标元素类型
 * @tparam Node TOML 节点类型
 * @param node TOML 数组节点视图
 * @param out [out] 输出向量
 * @return true 成功读取且数组不为空
 * @return false 读取失败或节点非数组
 */
template <typename Value, typename Node>
bool readArray(const toml::node_view<Node>& node, std::vector<Value>& out) {
  if (!node || !node.is_array()) return false;
  auto arr = node.as_array();
  if (!arr) return false;

  out.clear();
  out.reserve(arr->size());

  for (auto&& item : *arr) {
    if (auto v = item.template value<Value>()) {
      out.push_back(*v);
    } else {
      return false;
    }
  }
  return true;
}

/**
 * @brief 解析调制方式字符串
 *
 * 支持 "qam", "psk", "auto" (不区分大小写)
 *
 * @param text 输入字符串
 * @param scheme [out] 解析后的枚举值
 * @param err [out] 错误信息
 * @return true 解析成功
 */
bool parseModemScheme(const std::string& text, ModemScheme& scheme, std::string& err);

/**
 * @brief 解析 TOML 配置文件
 *
 * @param path 配置文件路径
 * @param cfg [out] 解析后的 TOML 表格对象
 * @param err [out] 错误信息
 * @return true 解析成功
 */
bool parseConfigFile(const std::string& path, toml::table& cfg, std::string& err);

/**
 * @brief 读取 [modem] 配置段
 *
 * @param cfg TOML 配置表
 * @param order [out] 调制阶数
 * @param symbols [out] 符号总数
 * @param scheme [out] 调制方式
 * @param err [out] 错误信息
 * @return true 读取成功
 */
bool loadModemConfig(const toml::table& cfg, int& order, size_t& symbols, ModemScheme& scheme,
                     std::string& err);

/**
 * @brief 读取 [sim] 仿真参数配置段
 *
 * @param cfg TOML 配置表
 * @param iters [out] BER 仿真迭代次数
 * @param perfMinTimeMs [out] 性能测试最小计时窗口 (ms)
 * @param seed [out] 随机数种子
 * @param enableGpu [out] 是否启用 GPU 加速
 * @param snrList [out] 信噪比 (dB) 列表
 * @param err [out] 错误信息
 * @return true 读取成功
 */
bool loadSimConfig(const toml::table& cfg, int& iters, double& perfMinTimeMs, uint64_t& seed,
                   bool& enableGpu, std::vector<double>& snrList, std::string& err);

/**
 * @brief 读取 [sampling] 采样率配置段
 *
 * @param cfg TOML 配置表
 * @param samplesPer [out] 过采样率 (SPS)
 * @param symbolRateHz [out] 符号率 (Hz)
 * @param sampleRateHz [out] 采样率 (Hz, 若为 0 则根据 SPS 计算)
 * @param carrierHz [out] 载波频率 (Hz)
 * @param rxLoOffsetHz [out] 接收端本振频偏 (Hz)
 * @param txPhaseRad [out] 发送端相位偏移 (rad)
 * @param rxPhaseRad [out] 接收端相位偏移 (rad)
 */
void loadSamplingConfig(const toml::table& cfg, size_t& samplesPer, double& symbolRateHz,
                        double& sampleRateHz, double& carrierHz, double& rxLoOffsetHz,
                        double& txPhaseRad, double& rxPhaseRad);

/** @brief 读取 [filter] 滤波器配置段 */
void loadFilterConfig(const toml::table& cfg, FilterArgs& filter);

/** @brief 读取 [lpf] 低通滤波器配置段 */
void loadLpfConfig(const toml::table& cfg, int& order);

/** @brief 读取 [channel] 信道配置段 */
bool loadChannelConfig(const toml::table& cfg, ChannelArgs& channel, std::string& err);

/** @brief 读取 [output] 输出配置段 */
bool loadOutputConfig(const toml::table& cfg, OutputArgs& output, std::string& err);

/**
 * @brief 根据阶数解析最终的调制方式
 *
 * 如果 inputs 为 AUTO，则根据 order 决定 (完全平方数 -> QAM, 否则 -> PSK)
 */
ModemScheme resolveModemScheme(ModemScheme scheme, int order);

/** @brief 获取调制方式的名称字符串 */
const char* modemSchemeName(ModemScheme scheme);

/** @brief 判断整数是否为 2 的幂 */
bool isPowerOfTwo(int value);

/** @brief 判断整数是否为完全平方数 */
bool isPerfectSquare(int value);

/**
 * @brief 计算每符号比特数 (Bits Per Symbol)
 *
 * @param order 调制阶数 (必须为 2 的幂)
 * @return int log2(order)
 */
int bitsPerSymbol(int order);

/**
 * @brief 校验并计算最终的采样率
 *
 * 确保符号率、采样率和过采样率之间满足关系：sampleRate = symbolRate *
 * samplesPer * rateFactor
 *
 * @param symbolRateHz [in,out] 符号率
 * @param sampleRateHz [in,out] 采样率 (若输入为 0 则自动计算)
 * @param samplesPer [in] 每个符号(或码片)的采样数
 * @param rateFactor [in] 速率倍乘因子 (例如 DSSS 中为 ChipLen，普通调制为 1)
 * @param err [out] 错误信息
 * @return true 校验通过
 */
bool finalizeSampleRates(double& symbolRateHz, double& sampleRateHz, int samplesPer, int rateFactor,
                         std::string& err);

/**
 * @brief 生成 DSSS 扩频伪随机码 (PN Code)
 *
 * 生成双极性序列 (+1, -1)
 */
std::vector<real32_t> generatePnCode(int length, simulation::RNG& rng);

/**
 * @brief 将 std::vector 转换为 Halide::Buffer
 *
 * 执行内存拷贝，生成的 Buffer 拥有自己的内存副本
 */
inline Halide::Buffer<real32_t> vectorToBuffer(const std::vector<real32_t>& data) {
  Halide::Buffer<real32_t> buf(static_cast<int>(data.size()));
  std::memcpy(buf.data(), data.data(), data.size() * sizeof(real32_t));
  buf.set_host_dirty();
  return buf;
}

/**
 * @brief 将复数向量转换为 Complex Halide::Buffer (2, N)
 */
inline Halide::Buffer<real32_t> complexToBuffer(const std::vector<complex32_t>& data) {
  Halide::Buffer<real32_t> buf(2, static_cast<int>(data.size()));
  for (size_t i = 0; i < data.size(); ++i) {
    buf(0, static_cast<int>(i)) = static_cast<real32_t>(data[i].real());
    buf(1, static_cast<int>(i)) = static_cast<real32_t>(data[i].imag());
  }
  buf.set_host_dirty();
  return buf;
}

/**
 * @brief 将 Halide::Buffer 转换为 std::vector
 *
 * 执行 device_sync 和 copy_to_host，确保数据已同步到 CPU
 */
inline std::vector<real32_t> bufferToVector(Halide::Buffer<real32_t> buf) {
  buf.copy_to_host();
  return {buf.begin(), buf.end()};
}

/**
 * @brief 获取 Buffer 的信号长度（兼容 Real 与 Complex 布局）
 */
inline int bufferLength(const Halide::Buffer<real32_t>& buf) {
  if (buf.dimensions() == 2 && buf.dim(0).extent() == 2) {
    return buf.dim(1).extent();
  }
  return buf.dim(0).extent();
}

/**
 * @brief 将交织的 I/Q 序列 (I0, Q0, I1, Q1...) 转换为复数 std::vector
 */
std::vector<complex32_t> interleavedToComplex(const std::vector<real32_t>& iq);

/**
 * @brief 将复数 std::vector 转换为交织的 I/Q 序列
 */
std::vector<real32_t> complexToInterleaved(const std::vector<complex32_t>& signal);

/**
 * @brief 比较解调/解映射后的符号与期望符号，统计误码率
 *
 * @param expected 期望的符号索引序列 (从 randomSymbols 获取)
 * @param demapped 解映射后的符号索引 (通常为 float，需四舍五入)
 * @param bits 每个符号代表的比特数
 * @return ErrorStats 误符号和误比特统计
 */
ErrorStats compareSymbols(const std::vector<real32_t>& expected, Halide::Buffer<real32_t> demapped,
                          int bits);

/**
 * @brief 测量函数执行时间的通用工具 (毫秒)
 *
 * 使用 Halide 官方 benchmark 做自适应采样
 * 自动检测返回值是否支持 `.device_sync()`，以确保 GPU 异步执行被正确计时
 *
 * @tparam Func 可调用对象类型
 * @param func 待测函数
 * @param minTimeMs 最小计时窗口 (ms)
 * @return double 平均每次执行的毫秒数
 */
template <typename Func>
double measureMs(Func&& func, double minTimeMs) {
  using ResultType = std::invoke_result_t<Func>;
  constexpr bool canSync = HAS_DEVICE_SYNC<ResultType>::value;

  const auto benchOp = [&]() {
    if constexpr (canSync) {
      auto out = func();
      out.device_sync();
    } else {
      func();
    }
  };

  // Warmup
  benchOp();

  Halide::Tools::BenchmarkConfig config;
  config.min_time = std::max(1e-3, minTimeMs / 1000.0);
  config.max_time = config.min_time * 4;
  auto result = Halide::Tools::benchmark(benchOp, config);
  return result.wall_time * 1000.0;
}

/** @brief 检查当前环境是否有可用的 GPU 设备 (Metal/CUDA/OpenCL) */
bool gpuAvailable();

/** @brief 打印标准格式的性能测试结果 (吞吐率等) */
void printPerf(const std::string& label, double ms, int symbols);

/** @brief 检查指定步骤是否在 OutputArgs 配置的导出列表中 */
bool shouldDumpStep(const OutputArgs& output, const std::string& step);

/** @brief 解析命令行参数中的配置文件路径，若未提供则使用默认值 */
std::string resolveConfigPath(int argc, char** argv, const std::string& defaultPath);

/** @brief 根据配置生成成形滤波器 (RRC/RC) 系数 */
std::vector<real32_t> buildShapingTaps(const FilterArgs& filter, int sps, std::string& err,
                                       const std::string& spsKey);

/** @brief 生成 Hamming 窗函数的低通滤波器系数 (Windowed-Sinc) */
std::vector<real32_t> buildLowpassTaps(int order, double cutoffHz, double sampleRateHz,
                                       std::string& err);

/** @brief 对复数序列进行数字混频 (上变频或下变频) */
std::vector<complex32_t> mixComplex(const std::vector<complex32_t>& in, double freqHz,
                                    double sampleRateHz, double phaseRad);

/** @brief 导出实数数据到 CSV 文件 */
bool dumpCsv(const OutputArgs& output, const std::string& step, const std::vector<real32_t>& data,
             std::string& err);

/** @brief 导出复数数据到 CSV 文件 (实部虚部分列) */
bool dumpCsv(const OutputArgs& output, const std::string& step,
             const std::vector<complex32_t>& data, std::string& err);

/** @brief 导出交织 I/Q 数据到 CSV 文件 */
bool dumpInterleavedIqCsv(const OutputArgs& output, const std::string& step,
                          const std::vector<real32_t>& iq, std::string& err);

/** @brief 对复数基带信号应用信道损伤模型 (噪声、衰落、频偏等) */
std::vector<complex32_t> applyChannel(const std::vector<complex32_t>& in,
                                      const ChannelArgs& channel, double sampleRateHz, double snrDb,
                                      simulation::RNG& rng);

/// @}

/**
 * @brief 标准示例参数集合
 *
 * 聚合了 Modem, Sampling, Filter, Channel, Output 等所有配置，
 * 是 example_helper 的主要配置对象
 */
struct StandardArgs {
  int iters = 50;                          ///< BER 统计的帧迭代次数
  double perfMinTimeMs = 100.0;            ///< 性能测试最小计时窗口 (ms)
  bool enableGpu = true;                   ///< 是否优先使用 GPU 后端
  int order = 2;                           ///< 调制阶数 (M-ary)
  size_t symbols = 4096;                   ///< 每帧仿真的符号数
  uint64_t seed = 42;                      ///< 随机数生成器种子
  ModemScheme scheme = ModemScheme::AUTO;  ///< 调制方式
  std::vector<double> snrList;             ///< 需要扫描的 SNR 点列表 (dB)

  size_t samplesPerSymbol = 8;  ///< 过采样率 (Samples Per Symbol/Chip)
  double symbolRateHz = 1e6;    ///< 符号率 (Baud Rate)
  double sampleRateHz = 0.0;    ///< 采样率 (通常由 symbolRate * sps 决定)
  double carrierHz = 2e6;       ///< 载波频率 (用于混频仿真)
  double rxLoOffsetHz = 0.0;    ///< 接收机本振频率偏差 (模拟 CFO)
  double txPhaseRad = 0.0;      ///< 发送机初始相位
  double rxPhaseRad = 0.0;      ///< 接收机初始相位 (模拟相偏)

  FilterArgs filter;    ///< 成形滤波器配置
  int lpfOrder = 63;    ///< 接收端低通滤波器阶数
  ChannelArgs channel;  ///< 信道模型配置
  OutputArgs output;    ///< 数据导出配置
  // Default scheduler (used as fallback for TX/RX when not specified).
  runtime::SchedulerConfig cpuSchedule{runtime::SchedulerKind::AUTO};
  runtime::SchedulerConfig gpuSchedule{runtime::SchedulerKind::AUTO};
  // Per-chain scheduler overrides.
  runtime::SchedulerConfig cpuScheduleTx{runtime::SchedulerKind::AUTO};
  runtime::SchedulerConfig gpuScheduleTx{runtime::SchedulerKind::AUTO};
  runtime::SchedulerConfig cpuScheduleRx{runtime::SchedulerKind::AUTO};
  runtime::SchedulerConfig gpuScheduleRx{runtime::SchedulerKind::AUTO};
};

/**
 * @brief DSSS 示例参数
 *
 * 在 StandardArgs 基础上追加 DSSS 扩频相关配置
 */
struct DsssArgs : public StandardArgs {
  int chipLen = 32;
  uint64_t pnSeed = 7;
  std::vector<real32_t> pnCode;
};

/** @brief 从文件加载 DSSS 配置 */
bool loadDsssConfig(const std::string& path, DsssArgs& args, std::string& err);

/** @brief 校验并完善 DSSS 参数 */
bool finalizeDsssArgs(DsssArgs& args, std::string& err);

/** @brief 从文件加载完整标准配置到 StandardArgs */
bool loadStandardConfig(const std::string& path, StandardArgs& args, std::string& err);

/** @brief 校验并完善 StandardArgs (计算依赖参数) */
bool finalizeStandardArgs(StandardArgs& args, std::string& err, int rateFactor = 1);

/** @brief 打印 StandardArgs 到控制台 */
void printStandardConfig(const StandardArgs& args);

/** @brief 标准滤波器组系数容器 */
struct StandardFilters {
  std::vector<real32_t> shapingTaps;  ///< 发送/匹配滤波器系数
  std::vector<real32_t> lpfTaps;      ///< 接收端抗混叠/噪声抑制滤波器系数
  int downModelDelay = 0;             ///< 下采样最佳相位延迟 (用于 downsample 算子)
};

/** @brief 根据 Args 初始化 StandardFilters */
bool setupStandardFilters(const StandardArgs& args, StandardFilters& out, std::string& err);

// ----------------------------------------------------------------------------
// 模板化公共流程
// ----------------------------------------------------------------------------

/**
 * @brief 运行标准正确性验证流程 (CPU 模式)
 *
 * 构建并运行一次 TX 和 RX 管道，比对输入输出符号，验证逻辑正确性
 *
 * @tparam TxFunc 发送链路函数对象
 * @tparam RxFunc 接收链路函数对象
 * @return true 误符号率为 0
 */
template <typename TxFunc, typename RxFunc>
bool runStandardVerification(const StandardArgs& args, TxFunc&& txFunc, RxFunc&& rxFunc) {
  simulation::RNG rng(args.seed);
  int const bits = bitsPerSymbol(args.order);

  auto symbols = simulation::randomSymbols<real32_t>(args.symbols, args.order, &rng);
  auto inputBuf = vectorToBuffer(symbols);

  // TX 链路 (理想发射)
  auto txSig = txFunc(inputBuf);

  // 理想信道 (直通，无损)
  auto const& rxSig = txSig;

  // RX 链路 (理想接收)
  auto rxSymbols = rxFunc(rxSig);
  ErrorStats const stat = compareSymbols(symbols, rxSymbols, bits);

  bool const pass = (stat.symbolErrors == 0);
  std::cout << "正确性验证(理想链路): " << (pass ? "PASS" : "FAIL") << "\n";
  std::cout << "  symbol errors: " << stat.symbolErrors << "\n";
  std::cout << "  bit errors: " << stat.bitErrors << "\n";
  std::cout << "  symbols used: " << stat.symbolsCompared << "\n\n";

  return pass;
}

/**
 * @brief 运行标准性能基准测试
 *
 * 分别测量 TX 链路、RX 链路以及端到端 (End-to-End) 的执行时间
 *
 * @param backendNameProp 手动指定的 Halide 后端名称 (用于显示)
 */
template <typename TxFunc, typename RxFunc>
void runStandardBenchmarks(const StandardArgs& args, TxFunc&& txFunc, RxFunc&& rxFunc,
                           const Halide::Buffer<real32_t>& inputBuf,
                           const Halide::Buffer<real32_t>& txOutBuf,
                           const std::string& backendNameProp = "") {
  if (args.perfMinTimeMs <= 0.0) return;

  std::string const fftBe = prism::getFftBackendName();
  std::string const halideBe =
      backendNameProp.empty() ? prism::getHalideBackendName() : backendNameProp;

  std::cout << "\n性能对比 [" << halideBe << ", " << fftBe << "]:\n";

  // 测量 TX
  double const txMs = measureMs([&]() { return txFunc(inputBuf); }, args.perfMinTimeMs);

  // 测量 RX (输入为预计算的 TX 输出)
  double const rxMs = measureMs([&]() { return rxFunc(txOutBuf); }, args.perfMinTimeMs);

  // 测量端到端 (TX + RX)
  double const e2eMs = measureMs(
      [&]() {
        auto tx = txFunc(inputBuf);
        return rxFunc(tx);
      },
      args.perfMinTimeMs);

  printPerf("TX Chain", txMs, args.symbols);
  printPerf("RX Chain", rxMs, args.symbols);
  printPerf("End-to-End", e2eMs, args.symbols);
}

/**
 * @brief 运行标准误码率 (BER) 仿真循环
 *
 * 在指定的一组 SNR 点上，进行多次迭代仿真，统计 BER
 * 包含软件信道模型 (applyChannel) 的调用
 *
 * @tparam TxChain 已编译的发送链路
 * @tparam RxChain 已编译的接收链路
 */
template <typename TxChain, typename RxChain>
void runStandardBer(const StandardArgs& args, TxChain& txChain, RxChain& rxChain) {
  std::cout << "\nBER 仿真 [" << txChain.targetName() << ", " << prism::getFftBackendName()
            << "]:\n";
  std::cout << "  SNR(dB)      BER        BitErrors/TotalBits\n";

  simulation::RNG rng(args.seed);
  int const bits = bitsPerSymbol(args.order);

  for (double const snrDb : args.snrList) {
    int64_t totalBits = 0;
    int64_t totalErrors = 0;

    for (int iter = 0; iter < args.iters; ++iter) {
      auto symbols = simulation::randomSymbols<real32_t>(args.symbols, args.order, &rng);
      auto inputIter = vectorToBuffer(symbols);

      // 1. TX 执行
      auto txOut = txChain.run(inputIter);

      // 2. 信道仿真 (Software)
      // 需要将 Halide Buffer 转为 host vector 进行复数运算
      auto txVec = bufferToVector(txOut);
      auto baseband = interleavedToComplex(txVec);

      // 上变频 -> 信道 -> 下变频
      auto up = mixComplex(baseband, args.carrierHz, args.sampleRateHz, args.txPhaseRad);
      auto ch = applyChannel(up, args.channel, args.sampleRateHz, snrDb, rng);
      auto down =
          mixComplex(ch, -(args.carrierHz + args.rxLoOffsetHz), args.sampleRateHz, args.rxPhaseRad);

      auto downBuf = complexToBuffer(down);

      // 3. RX 执行
      auto demapOut = rxChain.run(downBuf);

      // 确保 GPU 数据同步回 CPU
      if (demapOut.device_dirty()) {
        demapOut.device_sync();
        demapOut.copy_to_host();
      }

      // 4. 误码统计
      ErrorStats const stat = compareSymbols(symbols, demapOut, bits);
      totalBits += stat.totalBits;
      totalErrors += stat.bitErrors;
    }

    double const ber =
        (totalBits == 0) ? 0.0 : static_cast<double>(totalErrors) / static_cast<double>(totalBits);
    std::cout << "  " << std::setw(6) << std::fixed << std::setprecision(1) << snrDb << "   "
              << std::scientific << std::setprecision(3) << ber << "   " << std::fixed
              << totalErrors << "/" << totalBits << "\n";
  }
}

/**
 * @brief Bench 步骤配置
 *
 * 每个步骤描述如何从输入 Signal 构建当前阶段的计算图
 */
struct BenchStepSpec {
  std::string name;
  ScalarType inputType = ScalarType::C32;
  bool isIq = true;
  std::function<prism::dsl::Signal(const prism::dsl::Signal&)> build;
};

/**
 * @brief 构建并运行分步 benchmark（GPU/CPU 通用）
 */
void runBenchSteps(const StandardArgs& args, const std::string& backendLabel,
                   runtime::ExecMode mode, const runtime::SchedulerConfig& schedule,
                   const Halide::Buffer<real32_t>& initialInput,
                   const std::vector<BenchStepSpec>& steps);

/**
 * @brief 分步调试与数据导出检查器
 *
 * 允许注册 Pipeline 中的中间步骤，并统一管理它们的基准测试运行和数据导出
 * 对于调试复杂的 DSL 管道非常有用
 */
class StandardInspector {
 public:
  /** @brief 步骤执行函数类型: Buffer input -> Buffer output */
  using StepFunc = std::function<Halide::Buffer<real32_t>(Halide::Buffer<real32_t>)>;

  /** @brief 构造函数，需指定后端标签 */
  explicit StandardInspector(const std::string& halideBackendLabel);

  /**
   * @brief 注册一个检查步骤
   *
   * @param name 步骤名称 (用于显示和文件命名)
   * @param func 执行该步骤的 lambda
   * @param isIq 该步骤输出是否为 IQ 数据 (用于决定导出格式)
   */
  void addStep(std::string name, StepFunc func, bool isIq = true);

  /** @brief 运行所有注册步骤的基准测试并打印结果 */
  void runStepBenchmarks(const StandardArgs& args, const Halide::Buffer<real32_t>& initialInput);

  /** @brief 运行所有注册步骤并根据 OutputArgs 导出数据 */
  void exportStepData(const StandardArgs& args, const Halide::Buffer<real32_t>& initialInput);

 private:
  std::string backendLabel_;
  struct Step {
    std::string name;
    StepFunc func;
    bool isIq;
  };
  std::vector<Step> steps_;
};

/// @}

}  // namespace prism::examples

#endif  // PRISM_EXAMPLES_EXAMPLE_HELPER_H
