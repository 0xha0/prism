/**
 * @file example_helper.cpp
 * @ingroup examples
 * @brief 示例通用辅助实现
 *
 * 包含用于构建复杂物理层示例的共享工具函数，如:
 * - 滤波器设计 (RRC, RC)
 * - 信号质量评估 (EVM, BER)
 * - 配置文件解析 (TOML)
 * - 数据导出 (CSV)
 */

#include "example_helper.h"

#include <Halide.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "prism/prism.h"
#include "prism/runtime/executor.h"
#include "prism/runtime/schedule_config.h"
#include "prism/simulation/channel.h"
#include "prism/simulation/rng.h"
#include "prism/types.h"
#include "toml++/impl/node.hpp"
#include "toml++/impl/parse_error.hpp"
#include "toml++/impl/parser.hpp"
#include "toml++/impl/table.hpp"

namespace prism::examples {

constexpr double EXAMPLE_ESPS = 1e-12;

/**
 * @brief 计算归一化的 sinc 函数: sin(pi*x) / (pi*x)
 * @param x 输入值
 * @return sinc(x)
 */
double sinc(double x) {
  if (std::abs(x) < EXAMPLE_ESPS) {
    return 1.0;
  }
  return std::sin(M_PI_VAL * x) / (M_PI_VAL * x);
}

/**
 * @brief 设计根升余弦 (Root-Raised Cosine, RRC) 滤波器
 * @param span 滤波器跨度 (symbols)
 * @param sps 每个符号的采样数 (Samples Per Symbol)
 * @param rolloff 滚降系数 (beta), 范围 [0, 1]
 * @return 32位浮点滤波器系数
 *
 * RRC 滤波器是匹配滤波器的常用选择发送端和接收端各使用一个 RRC，
 * 级联后形成一个满足奈奎斯特第一准则的 RC (升余弦) 滤波器，从而消除 ISI
 */
std::vector<real32_t> designRrc(int span, int sps, double rolloff) {
  int const length = (span * sps) + 1;
  int const mid = length / 2;
  std::vector<real32_t> taps(static_cast<size_t>(length));

  if (rolloff <= 0.0) {
    for (int i = 0; i < length; ++i) {
      double const t = (i - mid) / static_cast<double>(sps);
      taps[static_cast<size_t>(i)] = static_cast<real32_t>(sinc(t));
    }
    return taps;
  }

  for (int i = 0; i < length; ++i) {
    double const t = (i - mid) / static_cast<double>(sps);
    double value = 0.0;
    if (std::abs(t) < EXAMPLE_ESPS) {
      value = 1.0 + (rolloff * (4.0 / M_PI_VAL - 1.0));
    } else if (std::abs(std::abs(4.0 * rolloff * t) - 1.0) < EXAMPLE_ESPS) {
      double const term1 = (1.0 + 2.0 / M_PI_VAL) * std::sin(M_PI_VAL / (4.0 * rolloff));
      double const term2 = (1.0 - 2.0 / M_PI_VAL) * std::cos(M_PI_VAL / (4.0 * rolloff));
      value = (rolloff / std::sqrt(2.0)) * (term1 + term2);
    } else {
      double const numerator = std::sin(M_PI_VAL * t * (1.0 - rolloff)) +
                               (4.0 * rolloff * t * std::cos(M_PI_VAL * t * (1.0 + rolloff)));
      double const denominator = M_PI_VAL * t * (1.0 - std::pow(4.0 * rolloff * t, 2.0));
      value = numerator / denominator;
    }
    taps[static_cast<size_t>(i)] = static_cast<real32_t>(value);
  }
  return taps;
}

/**
 * @brief 设计升余弦 (Raised Cosine, RC) 滤波器
 * @param span 滤波器跨度 (symbols)
 * @param sps 每个符号的采样数
 * @param rolloff 滚降系数
 * @return 32位浮点滤波器系数
 *
 * RC 滤波器本身满足无码间干扰 (Zero-ISI) 条件，通常用于理论验证
 */
std::vector<real32_t> designRc(int span, int sps, double rolloff) {
  int const length = (span * sps) + 1;
  int const mid = length / 2;
  std::vector<real32_t> taps(static_cast<size_t>(length));

  if (rolloff <= 0.0) {
    for (int i = 0; i < length; ++i) {
      double const t = (i - mid) / static_cast<double>(sps);
      taps[static_cast<size_t>(i)] = static_cast<real32_t>(sinc(t));
    }
    return taps;
  }

  for (int i = 0; i < length; ++i) {
    double const t = (i - mid) / static_cast<double>(sps);
    double value = 0.0;
    double const denom = 1.0 - std::pow(2.0 * rolloff * t, 2.0);
    if (std::abs(t) < EXAMPLE_ESPS) {
      value = 1.0;
    } else if (std::abs(denom) < EXAMPLE_ESPS) {
      value = (rolloff / 2.0) * std::sin(M_PI_VAL / (2.0 * rolloff));
    } else {
      value = sinc(t) * std::cos(M_PI_VAL * rolloff * t) / denom;
    }
    taps[static_cast<size_t>(i)] = static_cast<real32_t>(value);
  }
  return taps;
}

/**
 * @brief 归一化滤波器能量
 * @param taps 滤波器系数
 *
 * 使得 sum(taps[i]^2) = 1，这在匹配滤波系统中常用，确保总增益为 1
 */
void normalizeEnergy(std::vector<real32_t>& taps) {
  double energy = 0.0;
  for (real32_t const v : taps) {
    energy += static_cast<double>(v) * static_cast<double>(v);
  }
  if (energy <= 0.0) {
    return;
  }
  double const scale = 1.0 / std::sqrt(energy);
  for (auto& v : taps) {
    v = static_cast<real32_t>(static_cast<double>(v) * scale);
  }
}

/**
 * @brief 归一化滤波器直流增益
 * @param taps 滤波器系数
 *
 * 使得 sum(taps[i]) = 1，常用于低通滤波器保持信号幅值
 */
void normalizeDcGain(std::vector<real32_t>& taps) {
  double sum = 0.0;
  for (real32_t const v : taps) {
    sum += static_cast<double>(v);
  }
  if (std::abs(sum) < EXAMPLE_ESPS) {
    return;
  }
  for (auto& v : taps) {
    v = static_cast<real32_t>(static_cast<double>(v) / sum);
  }
}

/**
 * @brief 计算 32 位无符号整数中 set bits (1) 的数量
 * @param value 输入整数
 * @return 1 的个数
 */
int popcount32(uint32_t value) {
  int count = 0;
  while (value != 0U) {
    count += static_cast<int>(value & 1U);
    value >>= 1U;
  }
  return count;
}

bool ensureOutputDir(const std::string& dir, std::string& err) {
  if (dir.empty()) {
    err = "输出目录为空";
    return false;
  }
  std::error_code ec;
  std::filesystem::path const outDir(dir);
  if (std::filesystem::exists(outDir, ec)) {
    if (!std::filesystem::is_directory(outDir, ec)) {
      err = "输出路径不是目录: " + outDir.string();
      return false;
    }
    return true;
  }
  if (!std::filesystem::create_directories(outDir, ec)) {
    err = "无法创建输出目录: " + outDir.string();
    return false;
  }
  return true;
}

bool openCsvStream(const std::string& path, std::ofstream& os, std::string& err) {
  os.open(path, std::ios::out | std::ios::trunc);
  if (!os) {
    err = "无法写入输出文件: " + path;
    return false;
  }
  os << std::scientific << std::setprecision(std::numeric_limits<double>::max_digits10);
  return true;
}

bool parseModemScheme(const std::string& text, ModemScheme& scheme, std::string& err) {
  if (text == "auto") {
    scheme = ModemScheme::AUTO;
    return true;
  }
  if (text == "qam") {
    scheme = ModemScheme::QAM;
    return true;
  }
  if (text == "psk") {
    scheme = ModemScheme::PSK;
    return true;
  }
  err = "未知 scheme: " + text;
  return false;
}

bool parseConfigFile(const std::string& path, toml::table& cfg, std::string& err) {
  try {
    cfg = toml::parse_file(path);
  } catch (const toml::parse_error& e) {
    std::ostringstream oss;
    oss << "TOML 解析失败: " << e.description() << " (" << e.source().begin << ")";
    err = oss.str();
    return false;
  }
  return true;
}

bool parseSchedulerKind(const std::string& text, runtime::SchedulerKind& kind, std::string& err) {
  if (text == "none") {
    kind = runtime::SchedulerKind::NONE;
    return true;
  }
  if (text == "auto") {
    kind = runtime::SchedulerKind::AUTO;
    return true;
  }
  if (text == "manual") {
    kind = runtime::SchedulerKind::MANUAL;
    return true;
  }
  err = "未知 scheduler.kind: " + text;
  return false;
}

bool tomlValueToString(const toml::node& node, std::string& out) {
  if (auto v = node.value<std::string>()) {
    out = *v;
    return true;
  }
  if (auto v = node.value<int64_t>()) {
    out = std::to_string(*v);
    return true;
  }
  if (auto v = node.value<double>()) {
    std::ostringstream oss;
    oss << *v;
    out = oss.str();
    return true;
  }
  if (auto v = node.value<bool>()) {
    out = *v ? "1" : "0";
    return true;
  }
  return false;
}

bool loadSchedulerTable(const toml::table& table, runtime::SchedulerConfig& cfg, std::string& err) {
  if (auto kindText = table["kind"].value<std::string>()) {
    if (!parseSchedulerKind(*kindText, cfg.kind, err)) {
      return false;
    }
  }
  if (auto nameText = table["name"].value<std::string>()) {
    cfg.name = *nameText;
  }

  auto extraNode = table["extra"];
  if (extraNode) {
    const auto* extraTable = extraNode.as_table();
    if (!extraTable) {
      err = "scheduler.extra 必须是 table";
      return false;
    }
    for (const auto& [key, value] : *extraTable) {
      std::string text;
      if (!tomlValueToString(value, text)) {
        err = "scheduler.extra." + std::string(key.str()) + " 必须是 string/number/bool";
        return false;
      }
      if (text.empty()) {
        continue;  // 忽略空值
      }
      cfg.extra[std::string(key.str())] = text;
    }
  }
  return true;
}

bool loadSchedulerConfig(const toml::table& cfg, StandardArgs& args, std::string& err) {
  auto schedNode = cfg["scheduler"];
  const auto* schedTable = schedNode.as_table();
  auto loadPair = [&](const toml::table& table, runtime::SchedulerConfig& cpu,
                      runtime::SchedulerConfig& gpu) {
    const auto* cpuTable = table.get_as<toml::table>("cpu");
    const auto* gpuTable = table.get_as<toml::table>("gpu");
    if (cpuTable) {
      if (!loadSchedulerTable(*cpuTable, cpu, err)) {
        return false;
      }
    }
    if (gpuTable) {
      if (!loadSchedulerTable(*gpuTable, gpu, err)) {
        return false;
      }
    }
    if (!cpuTable && !gpuTable) {
      if (!loadSchedulerTable(table, cpu, err)) {
        return false;
      }
      gpu = cpu;
    }
    return true;
  };

  if (!schedTable) {
    args.cpuScheduleTx = args.cpuSchedule;
    args.gpuScheduleTx = args.gpuSchedule;
    args.cpuScheduleRx = args.cpuSchedule;
    args.gpuScheduleRx = args.gpuSchedule;
    return true;
  }

  if (!loadPair(*schedTable, args.cpuSchedule, args.gpuSchedule)) {
    return false;
  }

  args.cpuScheduleTx = args.cpuSchedule;
  args.gpuScheduleTx = args.gpuSchedule;
  args.cpuScheduleRx = args.cpuSchedule;
  args.gpuScheduleRx = args.gpuSchedule;

  auto txNode = (*schedTable)["tx"];
  if (txNode) {
    const auto* txTable = txNode.as_table();
    if (!txTable) {
      err = "scheduler.tx 必须是 table";
      return false;
    }
    if (!loadPair(*txTable, args.cpuScheduleTx, args.gpuScheduleTx)) {
      return false;
    }
  }
  auto rxNode = (*schedTable)["rx"];
  if (rxNode) {
    const auto* rxTable = rxNode.as_table();
    if (!rxTable) {
      err = "scheduler.rx 必须是 table";
      return false;
    }
    if (!loadPair(*rxTable, args.cpuScheduleRx, args.gpuScheduleRx)) {
      return false;
    }
  }
  return true;
}

bool loadModemConfig(const toml::table& cfg, int& order, size_t& symbols, ModemScheme& scheme,
                     std::string& err) {
  order = cfg["modem"]["order"].value_or(order);
  symbols = cfg["modem"]["symbols"].value_or(symbols);
  std::string const schemeText = cfg["modem"]["scheme"].value_or(std::string("auto"));
  return parseModemScheme(schemeText, scheme, err);
}

bool loadBerSimConfig(const toml::table& cfg, bool& enable, int& iters, uint64_t& seed,
                      bool& useGpu, std::vector<double>& snrList, std::string& err) {
  enable = cfg["ber_sim"]["enable"].value_or(enable);
  iters = cfg["ber_sim"]["iters"].value_or(iters);
  seed = cfg["ber_sim"]["seed"].value_or(seed);
  useGpu = cfg["ber_sim"]["use_gpu"].value_or(useGpu);

  auto snrNode = cfg["ber_sim"]["snr_db"];
  if (snrNode) {
    if (!readArray(snrNode, snrList) || snrList.empty()) {
      err = "SNR 列表读取失败";
      return false;
    }
  }
  return true;
}

void loadSamplingConfig(const toml::table& cfg, size_t& samplesPerSymbol, double& symbolRateHz,
                        double& sampleRateHz, double& carrierHz, double& rxLoOffsetHz,
                        double& txPhaseRad, double& rxPhaseRad) {
  samplesPerSymbol = cfg["sampling"]["samples_per_symbol"].value_or(samplesPerSymbol);
  symbolRateHz = cfg["sampling"]["symbol_rate_hz"].value_or(symbolRateHz);
  sampleRateHz = cfg["sampling"]["sample_rate_hz"].value_or(sampleRateHz);
  carrierHz = cfg["sampling"]["carrier_hz"].value_or(carrierHz);
  rxLoOffsetHz = cfg["sampling"]["rx_lo_offset_hz"].value_or(rxLoOffsetHz);
  txPhaseRad = cfg["sampling"]["tx_phase_rad"].value_or(txPhaseRad);
  rxPhaseRad = cfg["sampling"]["rx_phase_rad"].value_or(rxPhaseRad);
}

void loadFilterConfig(const toml::table& cfg, FilterArgs& filter) {
  filter.mode = cfg["filter"]["mode"].value_or(filter.mode);
  filter.span = cfg["filter"]["span"].value_or(filter.span);
  filter.rolloff = cfg["filter"]["rolloff"].value_or(filter.rolloff);
  filter.normalize = cfg["filter"]["normalize"].value_or(filter.normalize);
}

void loadLpfConfig(const toml::table& cfg, int& order) {
  order = cfg["lpf"]["order"].value_or(order);
}

bool loadChannelConfig(const toml::table& cfg, ChannelArgs& channel, std::string& err) {
  channel.enableAwgn = cfg["channel"]["enable_awgn"].value_or(channel.enableAwgn);
  channel.enableFading = cfg["channel"]["enable_fading"].value_or(channel.enableFading);
  channel.gain = cfg["channel"]["gain"].value_or(channel.gain);
  channel.dopplerHz = cfg["channel"]["doppler_hz"].value_or(channel.dopplerHz);
  channel.dopplerStartHz = cfg["channel"]["doppler_start_hz"].value_or(channel.dopplerStartHz);
  channel.dopplerEndHz = cfg["channel"]["doppler_end_hz"].value_or(channel.dopplerEndHz);
  channel.cfoHz = cfg["channel"]["cfo_hz"].value_or(channel.cfoHz);
  channel.phaseNoiseStd = cfg["channel"]["phase_noise_std"].value_or(channel.phaseNoiseStd);

  auto tapsNode = cfg["channel"]["fading_taps"];
  if (tapsNode) {
    std::vector<double> taps;
    if (!readArray(tapsNode, taps) || taps.empty()) {
      err = "fading_taps 读取失败";
      return false;
    }
    channel.fadingTaps.clear();
    channel.fadingTaps.reserve(taps.size());
    for (double v : taps) {
      channel.fadingTaps.push_back(static_cast<real32_t>(v));
    }
  }

  auto delaysNode = cfg["channel"]["fading_delays"];
  if (delaysNode) {
    std::vector<int> delays;
    if (!readArray(delaysNode, delays)) {
      err = "fading_delays 读取失败";
      return false;
    }
    channel.fadingDelays = delays;
  }
  return true;
}

bool loadOutputConfig(const toml::table& cfg, OutputArgs& output, std::string& err) {
  output.enable = cfg["output"]["enable"].value_or(output.enable);
  output.dir = cfg["output"]["dir"].value_or(output.dir);
  auto stepsNode = cfg["output"]["steps"];
  if (stepsNode) {
    std::vector<std::string> steps;
    if (!readArray(stepsNode, steps)) {
      err = "output.steps 读取失败";
      return false;
    }
    output.steps = steps;
  }
  return true;
}

const char* modemSchemeName(ModemScheme scheme) {
  switch (scheme) {
    case ModemScheme::PSK:
      return "PSK";
    case ModemScheme::QAM:
      return "QAM";
    case ModemScheme::AUTO:
    default:
      return "AUTO";
  }
}

bool isPowerOfTwo(int value) { return value > 0 && (value & (value - 1)) == 0; }

bool isPerfectSquare(int value) {
  if (value <= 0) {
    return false;
  }
  int const root = static_cast<int>(std::sqrt(value));
  return (root % 2 == 0);
}

inline int bitsPerSymbol(int order) {
  int bits = 0;
  while (order > 1) {
    order >>= 1;
    ++bits;
  }
  return bits;
}

bool finalizeSampleRates(double& symbolRateHz, double& sampleRateHz, int samplesPer, int rateFactor,
                         std::string& err) {
  if (symbolRateHz <= 0.0 && sampleRateHz <= 0.0) {
    err = "symbol_rate_hz/sample_rate_hz 至少提供一个";
    return false;
  }
  double const denom = static_cast<double>(samplesPer) * static_cast<double>(rateFactor);
  if (sampleRateHz <= 0.0) {
    sampleRateHz = symbolRateHz * denom;
  }
  if (symbolRateHz <= 0.0) {
    symbolRateHz = sampleRateHz / denom;
  }
  return true;
}

std::vector<real32_t> generatePnCode(int length, simulation::RNG& rng) {
  std::vector<real32_t> code(length);
  for (size_t i = 0; i < length; ++i) {
    code[i] = (rng.bit() == 0) ? -1.0F : 1.0F;
  }
  return code;
}

std::vector<complex32_t> interleavedToComplex(const std::vector<real32_t>& iq) {
  size_t const pairs = iq.size() / 2;
  std::vector<complex32_t> out(pairs);
  const real32_t* p = iq.data();
  for (size_t i = 0; i < pairs; ++i) {
    out[i] = complex32_t(p[0], p[1]);
    p += 2;
  }
  return out;
}

std::vector<real32_t> complexToInterleaved(const std::vector<complex32_t>& signal) {
  std::vector<real32_t> out(signal.size() * 2);
  real32_t* p = out.data();
  for (auto i : signal) {
    p[0] = static_cast<real32_t>(i.real());
    p[1] = static_cast<real32_t>(i.imag());
    p += 2;
  }
  return out;
}

ErrorStats compareSymbols(const std::vector<real32_t>& expected, Halide::Buffer<real32_t> demapped,
                          int bits) {
  demapped.copy_to_host();
  ErrorStats stats;
  int const compareCount = std::min(static_cast<int>(expected.size()), demapped.width());
  if (compareCount <= 0) {
    return stats;
  }

  uint32_t const mask = (bits >= 32) ? 0xFFFFFFFFU : ((1U << bits) - 1U);
  stats.totalBits = static_cast<int64_t>(compareCount) * bits;
  stats.symbolsCompared = compareCount;

  for (size_t i = 0; i < compareCount; ++i) {
    int const expSym = static_cast<int>(expected[i]);
    int const gotSym = static_cast<int>(std::lround(demapped(i)));
    if (expSym != gotSym) {
      stats.symbolErrors++;
    }
    uint32_t const diff = static_cast<uint32_t>(expSym ^ gotSym) & mask;
    stats.bitErrors += popcount32(diff);
  }
  return stats;
}

bool gpuAvailable() {
#if defined(PRISM_GPU_BACKEND_Metal) || defined(PRISM_GPU_BACKEND_CUDA) || \
    defined(PRISM_GPU_BACKEND_OpenCL)
  return true;
#else
  return false;
#endif
}

void printPerf(const std::string& label, double ms, int symbols) {
  double const symPerSec = (symbols / (ms / 1000.0)) / 1e6;
  std::cout << "  " << std::setw(12) << label << ": " << std::fixed << std::setprecision(3) << ms
            << " ms, " << std::setprecision(2) << symPerSec << " MSym/s\n";
}

bool shouldDumpStep(const OutputArgs& output, const std::string& step) {
  if (!output.enable) {
    return false;
  }
  if (output.steps.empty()) {
    return true;
  }
  return std::find(output.steps.begin(), output.steps.end(), step) != output.steps.end();
}

std::string resolveConfigPath(int argc, char** argv, const std::string& defaultPath) {
  if (argc > 1 && argv[1] != nullptr && !std::string(argv[1]).empty()) {
    return argv[1];
  }
  if (std::filesystem::exists(defaultPath)) {
    return defaultPath;
  }
  std::filesystem::path const alt = std::filesystem::path("..") / defaultPath;
  if (std::filesystem::exists(alt)) {
    return alt.string();
  }
  return defaultPath;
}

std::vector<real32_t> buildShapingTaps(const FilterArgs& filter, int sps, std::string& err,
                                       const std::string& spsKey) {
  if (filter.mode == "none") {
    return {1.0F};
  }
  if (filter.span <= 0 || sps <= 0) {
    err = "filter.span/" + spsKey + " 必须为正数";
    return {};
  }
  if (filter.rolloff < 0.0 || filter.rolloff > 1.0) {
    err = "filter.rolloff 必须在 [0, 1] 范围";
    return {};
  }

  std::vector<real32_t> taps;
  if (filter.mode == "rrc") {
    taps = designRrc(filter.span, sps, filter.rolloff);
  } else if (filter.mode == "rc") {
    taps = designRc(filter.span, sps, filter.rolloff);
  } else {
    err = "未知 filter.mode: " + filter.mode;
    return {};
  }
  if (filter.normalize) {
    normalizeEnergy(taps);
  }
  return taps;
}

std::vector<real32_t> buildLowpassTaps(int order, double cutoffHz, double sampleRateHz,
                                       std::string& err) {
  if (order <= 1) {
    return {1.0F};
  }
  if (sampleRateHz <= 0.0) {
    err = "sample_rate_hz 必须为正数";
    return {};
  }
  double const nyquist = 0.5 * sampleRateHz;
  if (cutoffHz <= 0.0 || cutoffHz >= nyquist) {
    err = "lpf 截止频率需在 (0, Nyquist) 内";
    return {};
  }

  int const length = (order % 2 == 0) ? (order + 1) : order;
  int const mid = length / 2;
  double const fc = cutoffHz / sampleRateHz;  // cycles/sample

  std::vector<real32_t> taps(static_cast<size_t>(length));
  for (int i = 0; i < length; ++i) {
    auto const n = static_cast<double>(i - mid);
    double const ideal = 2.0 * fc * sinc(2.0 * fc * n);
    double const w = 0.54 - (0.46 * std::cos(2.0 * M_PI_VAL * static_cast<double>(i) /
                                             static_cast<double>(length - 1)));
    taps[static_cast<size_t>(i)] = static_cast<real32_t>(ideal * w);
  }
  normalizeDcGain(taps);
  return taps;
}

/**
 * @brief 复数混频 (Mixer)
 * @param in 输入复数信号
 * @param freqHz 混频频率 (Hz)
 * @param sampleRateHz 采样率 (Hz)
 * @param phaseRad 初始相位 (radians)
 * @return 混频后的信号
 *
 *数学公式: out[t] = in[t] * exp(j * (2*pi*freq*t + phase))
 */
std::vector<complex32_t> mixComplex(const std::vector<complex32_t>& in, double freqHz,
                                    double sampleRateHz, double phaseRad) {
  if (in.empty()) {
    return {};
  }
  std::vector<complex32_t> out(in.size());

  const double omega = 2.0 * M_PI_VAL * freqHz / sampleRateHz;

  for (size_t i = 0; i < in.size(); ++i) {
    // 显式添加括号明确计算优先级，避免 Lint 警告
    const double ph = phaseRad + (omega * static_cast<double>(i));
    const complex32_t rot((real32_t)std::cos(ph), (real32_t)std::sin(ph));
    out[i] = in[i] * rot;
  }
  return out;
}

/**
 * @brief 将实数数据导出为 CSV 文件
 * @param output 输出配置
 * @param step 步骤名称 (也是文件名)
 * @param data 实数数据向量
 * @param err 错误信息输出
 * @return 成功返回 true
 */
bool dumpCsv(const OutputArgs& output, const std::string& step, const std::vector<real32_t>& data,
             std::string& err) {
  if (!shouldDumpStep(output, step)) {
    return true;
  }
  if (!ensureOutputDir(output.dir, err)) {
    return false;
  }

  std::filesystem::path const outPath = std::filesystem::path(output.dir) / (step + ".csv");
  std::ofstream os;
  if (!openCsvStream(outPath.string(), os, err)) {
    return false;
  }

  for (auto v : data) {
    os << static_cast<double>(v) << '\n';
  }
  return true;
}

/**
 * @brief 将复数数据导出为 CSV 文件 (Real, Imag)
 * @param output 输出配置
 * @param step 步骤名称
 * @param data 复数数据向量
 * @param err 错误信息输出
 * @return 成功返回 true
 */
bool dumpCsv(const OutputArgs& output, const std::string& step,
             const std::vector<complex32_t>& data, std::string& err) {
  if (!shouldDumpStep(output, step)) {
    return true;
  }
  if (!ensureOutputDir(output.dir, err)) {
    return false;
  }

  std::filesystem::path const outPath = std::filesystem::path(output.dir) / (step + ".csv");
  std::ofstream os;
  if (!openCsvStream(outPath.string(), os, err)) {
    return false;
  }

  for (const auto& v : data) {
    os << static_cast<double>(v.real()) << "," << static_cast<double>(v.imag()) << '\n';
  }
  return true;
}

/**
 * @brief 将交织的 IQ 数据导出为 CSV 文件
 * @param output 输出配置
 * @param step 步骤名称
 * @param iq 交织 IQ 数据 (I, Q, I, Q...)
 * @param err 错误信息输出
 * @return 成功返回 true
 */
bool dumpInterleavedIqCsv(const OutputArgs& output, const std::string& step,
                          const std::vector<real32_t>& iq, std::string& err) {
  if (!shouldDumpStep(output, step)) {
    return true;
  }
  if (iq.size() % 2 != 0) {
    err = "交织 I/Q 长度不是 2 的倍数";
    return false;
  }
  if (!ensureOutputDir(output.dir, err)) {
    return false;
  }

  std::filesystem::path const outPath = std::filesystem::path(output.dir) / (step + ".csv");
  std::ofstream os;
  if (!openCsvStream(outPath.string(), os, err)) {
    return false;
  }

  for (size_t i = 0; i + 1 < iq.size(); i += 2) {
    os << static_cast<double>(iq[i]) << "," << static_cast<double>(iq[i + 1]) << '\n';
  }
  return true;
}

/**
 * @brief 应用信道模型 (增益, 衰落, 多普勒, 频偏, 相噪, AWGN)
 * @param in 输入复数信号
 * @param channel 信道参数配置
 * @param sampleRateHz 采样率 (Hz)
 * @param snrDb 信噪比 (dB)
 * @param rng 随机数生成器
 * @return 经过信道后的信号
 *
 * 依次应用以下效应:
 * 1. 增益 (Gain)
 * 2. 多径衰落 (Multipath Fading) - 可选
 * 3. 多普勒频移/线性变化 (Doppler) - 可选
 * 4. 载波频偏 (CFO) - 可选
 * 5. 相位噪声 (Phase Noise) - 可选
 * 6. 加性高斯白噪声 (AWGN) - 可选
 */
std::vector<complex32_t> applyChannel(const std::vector<complex32_t>& in,
                                      const ChannelArgs& channel, double sampleRateHz, double snrDb,
                                      simulation::RNG& rng) {
  std::vector<complex32_t> out = in;
  if (out.empty()) {
    return out;
  }

  if (channel.gain != 1.0) {
    for (auto& v : out) {
      v = complex32_t(static_cast<real32_t>(v.real() * channel.gain),
                      static_cast<real32_t>(v.imag() * channel.gain));
    }
  }

  if (channel.enableFading && !channel.fadingTaps.empty()) {
    std::vector<complex32_t> taps;
    taps.reserve(channel.fadingTaps.size());
    for (auto v : channel.fadingTaps) {
      taps.emplace_back(v, 0.0F);
    }
    out = prism::simulation::fading(out, taps, channel.fadingDelays);
  }

  if (channel.dopplerStartHz != 0.0 || channel.dopplerEndHz != 0.0) {
    out = prism::simulation::dopplerRamp(out, channel.dopplerStartHz, channel.dopplerEndHz,
                                         sampleRateHz);
  } else if (channel.dopplerHz != 0.0) {
    out = prism::simulation::doppler(out, channel.dopplerHz, sampleRateHz);
  }

  if (channel.cfoHz != 0.0) {
    out = prism::simulation::frequencyOffset(out, channel.cfoHz, sampleRateHz);
  }

  if (channel.phaseNoiseStd > 0.0) {
    out = prism::simulation::phaseNoise(out, channel.phaseNoiseStd, &rng);
  }

  if (channel.enableAwgn) {
    out = prism::simulation::awgn(out, snrDb, &rng);
  }

  return out;
}

// ----------------------------------------------------------------------------
// Standard Helpers
// ----------------------------------------------------------------------------

bool loadStandardConfig(const std::string& path, StandardArgs& args, std::string& err) {
  toml::table cfg;
  if (!parseConfigFile(path, cfg, err)) {
    return false;
  }

  if (auto minTimeMs = cfg["perf_min_time_ms"].value<double>()) {
    args.perfMinTimeMs = *minTimeMs;
  } else if (auto minTimeMsInt = cfg["perf_min_time_ms"].value<int64_t>()) {
    args.perfMinTimeMs = static_cast<double>(*minTimeMsInt);
  }
  args.enableGpu = cfg["enable_gpu"].value_or(args.enableGpu);
  args.enableCpu = cfg["enable_cpu"].value_or(args.enableCpu);
  args.enableInspector = cfg["enable_inspector"].value_or(args.enableInspector);

  if (!loadModemConfig(cfg, args.order, args.symbols, args.scheme, err)) {
    return false;
  }
  if (!loadBerSimConfig(cfg, args.berEnable, args.iters, args.seed, args.berUseGpu, args.snrList,
                        err)) {
    return false;
  }
  if (!loadSchedulerConfig(cfg, args, err)) {
    return false;
  }
  loadSamplingConfig(cfg, args.samplesPerSymbol, args.symbolRateHz, args.sampleRateHz,
                     args.carrierHz, args.rxLoOffsetHz, args.txPhaseRad, args.rxPhaseRad);
  loadFilterConfig(cfg, args.filter);
  loadLpfConfig(cfg, args.lpfOrder);
  if (!loadChannelConfig(cfg, args.channel, err)) {
    return false;
  }
  if (!loadOutputConfig(cfg, args.output, err)) {
    return false;
  }
  return true;
}

bool finalizeStandardArgs(StandardArgs& args, std::string& err, int rateFactor) {
  if (args.order <= 1 || !isPowerOfTwo(args.order)) {
    err = "调制阶数必须是 2 的幂且大于 1";
    return false;
  }
  if (args.scheme == ModemScheme::QAM && !isPerfectSquare(args.order)) {
    err = "QAM 阶数必须是完全平方数 log(order) == 2N\n";
    return false;
  }
  if (args.scheme == ModemScheme::AUTO) {
    args.scheme =
        (args.order > 8) && isPerfectSquare(args.order) ? ModemScheme::QAM : ModemScheme::PSK;
  }
  if (args.symbols <= 0) {
    err = "symbols 必须为正数";
    return false;
  }
  if (args.perfMinTimeMs < 0.0) {
    err = "perf_min_time_ms 必须为非负数";
    return false;
  }
  if (!args.enableCpu && !args.enableGpu) {
    err = "enable_cpu 和 enable_gpu 不能同时为 false";
    return false;
  }
  if (args.samplesPerSymbol <= 0) {
    err = "samples_per_symbol 必须为正数";
    return false;
  }
  if (rateFactor <= 0) {
    err = "rateFactor 必须为正数";
    return false;
  }
  if (!finalizeSampleRates(args.symbolRateHz, args.sampleRateHz, args.samplesPerSymbol, rateFactor,
                           err)) {
    return false;
  }
  if (args.berEnable) {
    if (args.iters <= 0) {
      err = "ber_sim.iters 必须为正数";
      return false;
    }
    if (args.snrList.empty()) {
      err = "ber_sim.snr_db 列表不能为空";
      return false;
    }
  }
  return true;
}

bool setupStandardFilters(const StandardArgs& args, StandardFilters& out, std::string& err) {
  out.shapingTaps = buildShapingTaps(args.filter, args.samplesPerSymbol, err, "samples_per_symbol");
  if (!err.empty()) {
    return false;
  }
  if (out.shapingTaps.empty()) {
    err = "成形滤波器系数为空";
    return false;
  }

  if (args.symbols <= args.filter.span && args.filter.mode != "none") {
    err = "symbols 需大于 filter.span 以保证输出长度";
    return false;
  }

  double const rolloff = (args.filter.mode == "none") ? 0.0 : args.filter.rolloff;
  double const symRateHz = args.sampleRateHz / static_cast<double>(args.samplesPerSymbol);
  double const lpfCutoffHz = 0.5 * symRateHz * (1.0 + rolloff);

  out.lpfTaps = buildLowpassTaps(args.lpfOrder, lpfCutoffHz, args.sampleRateHz, err);
  if (!err.empty()) {
    return false;
  }
  if (out.lpfTaps.empty()) {
    err = "低通滤波器系数为空";
    return false;
  }

  int const lpfDelay = (static_cast<int>(out.lpfTaps.size()) - 1) / 2;
  out.downModelDelay = static_cast<int>(out.shapingTaps.size()) - 1 + lpfDelay;
  return true;
}

bool loadDsssConfig(const std::string& path, DsssArgs& args, std::string& err) {
  toml::table cfg;
  if (!parseConfigFile(path, cfg, err)) return false;
  if (!loadStandardConfig(path, args, err)) return false;
  args.chipLen = cfg["dsss"]["chip_len"].value_or(args.chipLen);
  args.pnSeed = cfg["dsss"]["pn_seed"].value_or(args.pnSeed);
  auto codeNode = cfg["dsss"]["code"];
  if (codeNode) {
    std::vector<double> code;
    if (!readArray(codeNode, code) || code.empty()) {
      err = "dsss.code 读取失败";
      return false;
    }
    args.pnCode.clear();
    for (double v : code) args.pnCode.push_back(static_cast<real32_t>(v));
  }
  return true;
}

bool finalizeDsssArgs(DsssArgs& args, std::string& err) {
  if (args.chipLen <= 0) {
    err = "chip_len 必须为正数";
    return false;
  }
  if (!finalizeStandardArgs(args, err, args.chipLen)) return false;
  if (!args.pnCode.empty() && static_cast<int>(args.pnCode.size()) != args.chipLen) {
    err = "dsss.code 长度需等于 chip_len";
    return false;
  }
  return true;
}

void printStandardConfig(const StandardArgs& args) {
  auto formatHz = [](double value) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << value << " Hz";
    return oss.str();
  };
  auto line = [](const std::string& key, const std::string& value) {
    std::ostringstream oss;
    oss << "  " << std::left << std::setw(16) << key << ": " << value;
    std::cout << oss.str() << "\n";
  };

  std::cout << "配置:\n";
  line("scheme", args.scheme == ModemScheme::PSK ? "PSK" : "QAM");
  {
    std::ostringstream oss;
    oss << args.order << " (bits/sym=" << bitsPerSymbol(args.order) << ")";
    line("order", oss.str());
  }
  line("symbols", std::to_string(args.symbols));
  line("sps(chip)", std::to_string(args.samplesPerSymbol));
  line("symbol rate", formatHz(args.symbolRateHz));
  double const sr =
      args.sampleRateHz > 0 ? args.sampleRateHz : (args.symbolRateHz * args.samplesPerSymbol);
  line("sample rate", formatHz(sr));
  line("carrier", formatHz(args.carrierHz));
  line("rx lo offset", formatHz(args.rxLoOffsetHz));
  auto scheduleLabel = [](const runtime::SchedulerConfig& cfg) {
    std::string label;
    switch (cfg.kind) {
      case runtime::SchedulerKind::NONE:
        label = "none";
        break;
      case runtime::SchedulerKind::AUTO:
        label = cfg.name.empty() ? "auto(default)" : "auto(" + cfg.name + ")";
        break;
      case runtime::SchedulerKind::MANUAL:
        label = "manual";
        break;
    }
    if (!cfg.extra.empty()) {
      label += " +extra(" + std::to_string(cfg.extra.size()) + ")";
    }
    return label;
  };
  line("cpu sched tx", scheduleLabel(args.cpuScheduleTx));
  line("cpu sched rx", scheduleLabel(args.cpuScheduleRx));
  line("gpu sched tx", scheduleLabel(args.gpuScheduleTx));
  line("gpu sched rx", scheduleLabel(args.gpuScheduleRx));

  // Filter info
  {
    std::ostringstream oss;
    oss << args.filter.mode << " (span=" << args.filter.span << ", rolloff=" << args.filter.rolloff
        << ")";
    line("filter", oss.str());
  }
  {
    std::ostringstream oss;
    oss << args.lpfOrder << " (cutoff=" << std::fixed << std::setprecision(2) << (sr / 2 * 0.9)
        << " Hz est.)";
    line("lpf order", oss.str());
  }

  line("cpu", args.enableCpu ? "启用" : "禁用");
  line("gpu", args.enableGpu ? "启用" : "禁用");
  line("inspector", args.enableInspector ? "启用" : "禁用");
  line("ber", args.berEnable ? (args.berUseGpu ? "启用(GPU)" : "启用(CPU)") : "禁用");
  line("iters", std::to_string(args.iters));
  {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(1) << args.perfMinTimeMs << " ms";
    line("perf min time", oss.str());
  }
  line("seed", std::to_string(args.seed));
  line("fft backend", prism::getFftBackendName());
  std::cout << "\n";
}

// ----------------------------------------------------------------------------
// Inspector Implementation
// ----------------------------------------------------------------------------

StandardInspector::StandardInspector(const std::string& halideBackendLabel) {
  if (halideBackendLabel.empty()) {
    backendLabel_ = prism::getHalideBackendName() + ", " + prism::getFftBackendName();
  } else {
    backendLabel_ = halideBackendLabel + ", " + prism::getFftBackendName();
  }
}

void runBenchSteps(const StandardArgs& args, const std::string& backendLabel,
                   runtime::ExecMode mode, const runtime::SchedulerConfig& schedule,
                   const Halide::Buffer<real32_t>& initialInput,
                   const std::vector<BenchStepSpec>& steps) {
  if (!args.enableInspector || args.perfMinTimeMs <= 0.0) return;

  StandardInspector inspector(backendLabel);
  Halide::Buffer<real32_t> currentInput = initialInput;

  for (const auto& step : steps) {
    auto signal = step.build(prism::dsl::Signal::input(bufferLength(currentInput), step.inputType));
    auto compiled = prism::runtime::Executor::compile<real32_t>(signal, mode, schedule);
    using ExecHandle = decltype(compiled);
    auto handle = std::make_shared<ExecHandle>(std::move(compiled));

    auto out = handle->run(currentInput);
    inspector.addStep(
        step.name, [handle](const Halide::Buffer<real32_t>& in) { return handle->run(in); },
        step.isIq);
    currentInput = out;
  }

  inspector.runStepBenchmarks(args, initialInput);
}

void StandardInspector::addStep(std::string name, StepFunc func, bool isIq) {
  steps_.push_back({std::move(name), std::move(func), isIq});
}

void StandardInspector::runStepBenchmarks(const StandardArgs& args,
                                          const Halide::Buffer<real32_t>& initialInput) {
  if (args.perfMinTimeMs <= 0.0) return;

  std::cout << "\n步骤计时 [" << backendLabel_ << "]:\n";

  Halide::Buffer<real32_t> currentInput = initialInput;

  for (const auto& step : steps_) {
    // Warmup
    auto outWarm = step.func(currentInput);

    // Bench using measureMs
    double const avgMs = measureMs([&]() { return step.func(currentInput); }, args.perfMinTimeMs);

    // Print
    std::cout << "  " << std::left << std::setw(13) << step.name << ": " << std::right << std::fixed
              << std::setprecision(3) << avgMs << " ms"
              << "\n";

    // Update input for next step
    currentInput = outWarm;
  }
}

void StandardInspector::exportStepData(const StandardArgs& args,
                                       const Halide::Buffer<real32_t>& initialInput) {
  if (!args.enableInspector || !args.output.enable) return;

  std::cout << "\n正在导出分步仿真数据 (SNR=" << args.snrList[0] << "dB) [" << backendLabel_
            << "]...\n";

  std::string err;
  bool fail = false;

  // Dump Initial Input if needed (usually 'symbols')
  if (shouldDumpStep(args.output, "symbols")) {
    auto vec = bufferToVector(initialInput);
    if (!dumpCsv(args.output, "symbols", vec, err)) fail = true;
  }

  Halide::Buffer<real32_t> currentInput = initialInput;

  for (const auto& step : steps_) {
    if (fail) break;

    // Execute
    auto output = step.func(currentInput);

    // Sync
    if (output.device_dirty()) {
      output.device_sync();
      output.copy_to_host();
    }

    // Dump
    if (shouldDumpStep(args.output, step.name)) {
      auto vec = bufferToVector(output);
      if (step.isIq) {
        if (!dumpInterleavedIqCsv(args.output, step.name, vec, err)) {
          fail = true;
        }
      } else {
        if (!dumpCsv(args.output, step.name, vec, err)) {
          fail = true;
        }
      }
    }

    currentInput = output;
  }

  if (fail) {
    std::cerr << "导出数据失败: " << err << "\n";
  } else {
    // Only print if enable is true, which is checked at top
    std::cout << "\n数据导出完成.\n";
  }
}

}  // namespace prism::examples
