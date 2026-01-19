/**
 * @file main.cpp
 * @brief QAM/PSK + DSSS + FIR 均衡仿真示例
 *
 * 本示例演示了如何基于 PRISM DSL 构建 DSSS 物理链路，并加入理想 FIR 时域均衡
 *
 * ## 系统架构
 * 扩频通信通过将每个符号扩展为高速率的伪随机码片 (Chips)
 * 序列，显著增加信号带宽，从而降低功率谱密度并提高抗窄带干扰能力
 *
 * ### 发射链路 (TX)
 * 1. **符号映射 (ModMap)**: Bits -> Symbols (QAM/PSK 调制)
 * 2. **扩频 (Spreading)**: Symbols -> Chips
 *    - 每个符号被长度为 $L$ (扩频因子) 的伪随机码 (PN Code) 调制
 *    - 实现上利用 `upsample` (插值) 和 `fir` (与 PN 序列卷积) 完成
 * 3. **脉冲成形 (Pulse Shaping)**: Chips -> Waveform
 *    - 对码片序列进行 RRC 滤波，限制发射带宽
 * 4. **上变频 (Upconversion)**: 基带 -> 射频载波仿真
 *
 * ### 接收链路 (RX) - 理想均衡 + 解扩
 * 1. **低通滤波 (LPF)**: 抑制带外噪声
 * 2. **FIR 均衡 (Ideal ZF)**: 使用已知信道响应设计时域逆滤波
 * 3. **匹配滤波 (Matched Filter)**: Waveform -> Soft Chips
 * 4. **码片级同步与下采样**: 恢复到 Chip Rate
 * 5. **解扩 (Despreading)**: Soft Chips * PN Code -> Soft Symbols
 * 6. **符号级下采样**: 抽取相关峰值
 * 7. **解映射 (Demap)**: Soft Symbols -> Bits
 *
 * ## 关键参数
 * - `chip_len` (扩频增益): 决定带宽扩展倍数和处理增益 ($G_p = 10 \log_{10}(L)$)
 * - `pn_seed`: 控制伪随机序列生成的种子
 */

#include <Halide.h>

#include <algorithm>
#include <cmath>
#include <exception>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

#include "example_helper.h"
#include "prism/dsl/filter.h"
#include "prism/dsl/modem.h"
#include "prism/dsl/ops.h"
#include "prism/dsl/signal.h"
#include "prism/prism.h"
#include "prism/runtime/executor.h"
#include "prism/simulation/rng.h"
#include "prism/simulation/source.h"
#include "prism/types.h"

using prism::real32_t;
using prism::dsl::Signal;
using prism::runtime::ExecMode;
using prism::runtime::Executor;
using namespace prism::dsl;
using namespace prism::examples;
using namespace prism::simulation;

namespace {

std::vector<real32_t> buildChannelImpulse(const ChannelArgs& channel, int& maxDelay) {
  bool const useFading = channel.enableFading && !channel.fadingTaps.empty();
  maxDelay = 0;
  if (useFading) {
    for (int const d : channel.fadingDelays) {
      maxDelay = std::max(maxDelay, d);
    }
  }
  if (maxDelay < 0) {
    maxDelay = 0;
  }

  std::vector<real32_t> h(static_cast<size_t>(maxDelay + 1), 0.0F);
  if (useFading) {
    for (size_t i = 0; i < channel.fadingTaps.size(); ++i) {
      int const delay = (i < channel.fadingDelays.size()) ? channel.fadingDelays[i] : 0;
      if (delay < 0 || delay > maxDelay) {
        continue;
      }
      h[static_cast<size_t>(delay)] += channel.fadingTaps[i];
    }
  } else {
    h[0] = 1.0F;
  }

  if (channel.gain != 1.0) {
    for (auto& v : h) {
      v = static_cast<real32_t>(v * channel.gain);
    }
  }

  bool nonZero = false;
  for (auto v : h) {
    if (std::abs(v) > 1e-12F) {
      nonZero = true;
      break;
    }
  }
  if (!nonZero) {
    h.assign(1, 1.0F);
    maxDelay = 0;
  }
  return h;
}

std::vector<double> solveLinearSystem(std::vector<double> a, std::vector<double> b, int n,
                                      bool& ok) {
  ok = true;
  for (int i = 0; i < n; ++i) {
    int pivot = i;
    double maxAbs = std::abs(a[static_cast<size_t>(i * n + i)]);
    for (int r = i + 1; r < n; ++r) {
      double const v = std::abs(a[static_cast<size_t>(r * n + i)]);
      if (v > maxAbs) {
        maxAbs = v;
        pivot = r;
      }
    }
    if (maxAbs < 1e-12) {
      ok = false;
      return {};
    }
    if (pivot != i) {
      for (int c = i; c < n; ++c) {
        std::swap(a[static_cast<size_t>(i * n + c)],
                  a[static_cast<size_t>(pivot * n + c)]);
      }
      std::swap(b[static_cast<size_t>(i)], b[static_cast<size_t>(pivot)]);
    }

    double const diag = a[static_cast<size_t>(i * n + i)];
    for (int c = i; c < n; ++c) {
      a[static_cast<size_t>(i * n + c)] /= diag;
    }
    b[static_cast<size_t>(i)] /= diag;

    for (int r = 0; r < n; ++r) {
      if (r == i) {
        continue;
      }
      double const factor = a[static_cast<size_t>(r * n + i)];
      if (std::abs(factor) < 1e-12) {
        continue;
      }
      for (int c = i; c < n; ++c) {
        a[static_cast<size_t>(r * n + c)] -= factor * a[static_cast<size_t>(i * n + c)];
      }
      b[static_cast<size_t>(r)] -= factor * b[static_cast<size_t>(i)];
    }
  }
  return b;
}

std::vector<real32_t> designIdealEqTaps(const std::vector<real32_t>& h, int eqLen,
                                        int targetDelay, real32_t reg, std::string& err) {
  if (eqLen <= 0) {
    err = "均衡器长度必须为正数";
    return {1.0F};
  }
  int const hLen = static_cast<int>(h.size());
  if (hLen <= 0) {
    err = "信道冲激响应为空";
    return {1.0F};
  }
  int const totalLen = hLen + eqLen - 1;
  if (targetDelay < 0 || targetDelay >= totalLen) {
    targetDelay = std::clamp(targetDelay, 0, totalLen - 1);
  }

  std::vector<double> ata(static_cast<size_t>(eqLen * eqLen), 0.0);
  std::vector<double> atb(static_cast<size_t>(eqLen), 0.0);

  for (int i = 0; i < eqLen; ++i) {
    int const idx = targetDelay - i;
    if (idx >= 0 && idx < hLen) {
      atb[static_cast<size_t>(i)] = static_cast<double>(h[static_cast<size_t>(idx)]);
    }
  }

  for (int i = 0; i < eqLen; ++i) {
    for (int j = 0; j < eqLen; ++j) {
      double sum = 0.0;
      for (int m = 0; m < totalLen; ++m) {
        int const idxI = m - i;
        int const idxJ = m - j;
        double const hi =
            (idxI >= 0 && idxI < hLen) ? static_cast<double>(h[static_cast<size_t>(idxI)]) : 0.0;
        double const hj =
            (idxJ >= 0 && idxJ < hLen) ? static_cast<double>(h[static_cast<size_t>(idxJ)]) : 0.0;
        sum += hi * hj;
      }
      ata[static_cast<size_t>(i * eqLen + j)] = sum;
    }
  }

  for (int i = 0; i < eqLen; ++i) {
    ata[static_cast<size_t>(i * eqLen + i)] += reg;
  }

  bool ok = false;
  auto sol = solveLinearSystem(ata, atb, eqLen, ok);
  if (!ok || sol.size() != static_cast<size_t>(eqLen)) {
    err = "均衡器求解失败";
    return {1.0F};
  }

  std::vector<real32_t> taps(static_cast<size_t>(eqLen));
  for (int i = 0; i < eqLen; ++i) {
    taps[static_cast<size_t>(i)] = static_cast<real32_t>(sol[static_cast<size_t>(i)]);
  }
  return taps;
}

}  // namespace

int main(int argc, char** argv) {
  DsssArgs args;
  std::string const configPath = resolveConfigPath(argc, argv, "examples/apm_dsss_eq/config.toml");

  std::string err;
  if (!loadDsssConfig(configPath, args, err)) {
    std::cerr << err << "\n";
    return 1;
  }
  if (!finalizeDsssArgs(args, err)) {
    std::cerr << err << "\n";
    return 1;
  }

  // PN Code
  RNG pnRng(args.pnSeed);
  if (args.pnCode.empty()) {
    args.pnCode = generatePnCode(args.chipLen, pnRng);
  }
  std::vector<real32_t> const pnCodeRev(args.pnCode.rbegin(), args.pnCode.rend());

  // 滤波器（将码片视为射频成形的符号）
  StandardFilters filters;
  if (!setupStandardFilters(args, filters, err)) {
    std::cerr << err << "\n";
    return 1;
  }
  // 检查 DSSS 要求
  if (args.symbols * args.chipLen <= args.filter.span && args.filter.mode != "none") {
    std::cerr << "symbols*chip_len 需大于 filter.span 以保证输出长度\n";
    return 1;
  }

  int maxDelay = 0;
  auto channelImpulse = buildChannelImpulse(args.channel, maxDelay);
  int const eqLen = std::max(1, static_cast<int>(channelImpulse.size()));
  int eqDelay = maxDelay;
  std::string eqErr;
  std::vector<real32_t> const eqTaps =
      designIdealEqTaps(channelImpulse, eqLen, eqDelay, 1e-6F, eqErr);
  if (!eqErr.empty()) {
    std::cout << "均衡器设计警告: " << eqErr << "\n";
    eqDelay = 0;
  }

  // --------------------------------------------------------------------------
  // Initial Information / 配置信息显示
  // --------------------------------------------------------------------------
  std::cout << "=== PRISM 示例: QAM/PSK + DSSS + FIR 均衡 ===\n\n";
  std::cout << "配置文件: " << configPath << "\n";
  std::cout << "后端: " << prism::getFftBackendName() << "\n\n";
  printStandardConfig(args);
  std::cout << "均衡器: Ideal FIR (taps=" << eqTaps.size() << ", delay=" << eqDelay << ")\n\n";

  prism::initialize();

  // --------------------------------------------------------------------------
  // Pipeline
  // --------------------------------------------------------------------------
  auto const& scheme = args.scheme;
  auto defineTx = [&](const Signal& txInput) {
    // 1. 符号映射 (Symbol Mapping)
    // 将输入比特流映射为复数符号 (Symbol Rate)
    Signal const mapSig = (scheme == ModemScheme::PSK) ? modem::pskMap(txInput, args.order)
                                                       : modem::qamMap(txInput, args.order);

    // 2. 扩频 (Spreading)
    // 将每个符号扩展为 chip_len 个码片 (Chip Rate)
    // 实现方式：先上采样 (插0) 到 chip_len 倍，然后与 PN 序列卷积 (相当于
    // Kronecker 积) PN 序列作为 FIR 滤波器系数，长度等于扩频因子
    Signal const spreadI = filter::fir(upsample(real(mapSig), args.chipLen), args.pnCode);
    Signal const spreadQ = filter::fir(upsample(imag(mapSig), args.chipLen), args.pnCode);

    // 3. 脉冲成形 (Pulse Shaping)
    // 将码片序列上采样并进行 RRC 滤波 (Sample Rate)
    // SPS 这里指 Samples Per Chip，即每个码片的采样点数
    Signal const upI = upsample(spreadI, args.samplesPerSymbol);
    Signal const upQ = upsample(spreadQ, args.samplesPerSymbol);
    Signal const shapeI = filter::fir(upI, filters.shapingTaps);
    Signal const shapeQ = filter::fir(upQ, filters.shapingTaps);

    // 4. I/Q 打包
    // 合成最终的复数基带信号
    return complexPack(shapeI, shapeQ);
  };

  auto defineRx = [&](const Signal& rxInput) {
    // 1. 接收前端滤波与匹配
    // LPF: 滤除带外噪声 (Anti-aliasing)
    Signal const rxLpfI = filter::fir(real(rxInput), filters.lpfTaps);
    Signal const rxLpfQ = filter::fir(imag(rxInput), filters.lpfTaps);
    Signal const rxLpf = complexPack(rxLpfI, rxLpfQ);

    // 2. 理想时域均衡
    // 使用已知信道冲激响应设计 ZF 逆滤波
    Signal const eqSig = filter::fir(rxLpf, eqTaps);

    // 3. 匹配滤波 (Matched Filter)
    Signal const mfI = filter::fir(real(eqSig), filters.shapingTaps);
    Signal const mfQ = filter::fir(imag(eqSig), filters.shapingTaps);

    // 4. 下采样到码片速率 (Sample Rate -> Chip Rate)
    // 选择最佳采样点 (downModelDelay) 进行抽取
    int const rxDownDelay = filters.downModelDelay + eqDelay;
    Signal const chipI = downsample(mfI, args.samplesPerSymbol, rxDownDelay);
    Signal const chipQ = downsample(mfQ, args.samplesPerSymbol, rxDownDelay);
    Signal const chipSig = complexPack(chipI, chipQ);

    // 5. 解扩 (Despreading / Correlation)
    // 将接收到的码片序列与本地 PN 码 (翻转后) 进行卷积，实现滑动相关运算
    // 相关峰值将出现在符号边界处，幅度增强 L 倍
    Signal const corrI = filter::fir(real(chipSig), pnCodeRev);
    Signal const corrQ = filter::fir(imag(chipSig), pnCodeRev);

    // 6. 符号级下采样 (Chip Rate -> Symbol Rate)
    // 在相关峰值处抽取符号，延迟通常为 ChipLen - 1 (卷积带来的延迟) + 系统延迟
    Signal const symI = downsample(corrI, args.chipLen, args.chipLen - 1);
    Signal const symQ = downsample(corrQ, args.chipLen, args.chipLen - 1);

    // 7. 幅度归一化 (Normalization)
    // 扩频增益导致相关峰值幅度为 ChipLen，需归一化以匹配星座图
    Signal const normI = scale(symI, 1.0F / static_cast<real32_t>(args.chipLen));
    Signal const normQ = scale(symQ, 1.0F / static_cast<real32_t>(args.chipLen));

    // 8. 解映射 (Demapping)
    // 将软符号判决为比特
    return (scheme == ModemScheme::PSK) ? modem::pskDemap(normI, normQ, args.order)
                                        : modem::qamDemap(normI, normQ, args.order);
  };

  auto txInLen = args.symbols;
  auto rxInLen = args.symbols * args.chipLen * args.samplesPerSymbol;

  // CPU
  auto txChainCpu = Executor::compile<real32_t>(defineTx(Signal::input(txInLen)), ExecMode::CPU,
                                                args.cpuScheduleTx);
  auto rxChainCpu = Executor::compile<real32_t>(
      defineRx(Signal::input(rxInLen, prism::ScalarType::C32)), ExecMode::CPU, args.cpuScheduleRx);

  // GPU
  using ExecHandle = decltype(txChainCpu);
  std::optional<ExecHandle> txChainGpu;
  std::optional<ExecHandle> rxChainGpu;

  if (args.enableGpu && gpuAvailable()) {
    try {
      txChainGpu = Executor::compile<real32_t>(defineTx(Signal::input(txInLen)), ExecMode::GPU,
                                               args.gpuScheduleTx);
      rxChainGpu =
          Executor::compile<real32_t>(defineRx(Signal::input(rxInLen, prism::ScalarType::C32)),
                                      ExecMode::GPU, args.gpuScheduleRx);
    } catch (const std::exception& e) {
      std::cout << "GPU compilation failed: " << e.what() << "\n";
    }
  }

  // --------------------------------------------------------------------------
  // 1. 参数检查 (带信道但不加噪声)
  // --------------------------------------------------------------------------
  if (args.enableCpu || (args.enableGpu && txChainGpu && rxChainGpu)) {
    RNG rng(args.seed);
    ChannelArgs channel = args.channel;
    channel.enableAwgn = false;
    double const snrDb0 = args.snrList.empty() ? 0.0 : args.snrList[0];

    auto symbols = randomSymbols<real32_t>(args.symbols, args.order, &rng);
    auto inputBuf = vectorToBuffer(symbols);

    auto txOut = args.enableGpu && !args.enableCpu && txChainGpu ? txChainGpu->run(inputBuf)
                                                                 : txChainCpu.run(inputBuf);
    auto txVec = bufferToVector(txOut);
    auto baseband = interleavedToComplex(txVec);

    auto up = mixComplex(baseband, args.carrierHz, args.sampleRateHz, args.txPhaseRad);
    auto ch = applyChannel(up, channel, args.sampleRateHz, snrDb0, rng);
    auto down = mixComplex(ch, -(args.carrierHz + args.rxLoOffsetHz), args.sampleRateHz,
                           args.rxPhaseRad);
    auto downBuf = complexToBuffer(down);

    auto rxSymbols = args.enableGpu && !args.enableCpu && rxChainGpu ? rxChainGpu->run(downBuf)
                                                                     : rxChainCpu.run(downBuf);
    if (rxSymbols.device_dirty()) {
      rxSymbols.device_sync();
      rxSymbols.copy_to_host();
    }
    ErrorStats const stat = compareSymbols(symbols, rxSymbols, bitsPerSymbol(args.order));
    bool const pass = (stat.symbolErrors == 0);
    std::cout << "正确性验证(理想均衡, 无噪声): " << (pass ? "PASS" : "FAIL") << "\n";
    std::cout << "  symbol errors: " << stat.symbolErrors << "\n";
    std::cout << "  bit errors: " << stat.bitErrors << "\n";
    std::cout << "  symbols used: " << stat.symbolsCompared << "\n\n";
  }

  // --------------------------------------------------------------------------
  // 2. 性能测试
  // --------------------------------------------------------------------------
  if (args.perfMinTimeMs > 0.0) {
    RNG rng(args.seed);
    auto symbols = vectorToBuffer(randomSymbols<real32_t>(args.symbols, args.order, &rng));

    auto defineTxMap = [&](const Signal& in) {
      return (scheme == ModemScheme::PSK) ? modem::pskMap(in, args.order)
                                          : modem::qamMap(in, args.order);
    };
    auto defineTxSpread = [&](const Signal& in) {
      return complexPack(filter::fir(upsample(real(in), args.chipLen), args.pnCode),
                         filter::fir(upsample(imag(in), args.chipLen), args.pnCode));
    };
    auto defineTxUp = [&](const Signal& in) {
      return complexPack(upsample(real(in), args.samplesPerSymbol),
                         upsample(imag(in), args.samplesPerSymbol));
    };
    auto defineTxShape = [&](const Signal& in) {
      return complexPack(filter::fir(real(in), filters.shapingTaps),
                         filter::fir(imag(in), filters.shapingTaps));
    };
    auto defineRxLpf = [&](const Signal& in) {
      return complexPack(filter::fir(real(in), filters.lpfTaps),
                         filter::fir(imag(in), filters.lpfTaps));
    };
    auto defineRxEq = [&](const Signal& in) { return filter::fir(in, eqTaps); };
    auto defineRxMatch = [&](const Signal& in) {
      return complexPack(filter::fir(real(in), filters.shapingTaps),
                         filter::fir(imag(in), filters.shapingTaps));
    };
    auto defineRxDown = [&](const Signal& in) {  // Samples -> Chips
      int const rxDownDelay = filters.downModelDelay + eqDelay;
      return complexPack(downsample(real(in), args.samplesPerSymbol, rxDownDelay),
                         downsample(imag(in), args.samplesPerSymbol, rxDownDelay));
    };
    auto defineRxDespread = [&](const Signal& in) {
      return complexPack(filter::fir(real(in), pnCodeRev), filter::fir(imag(in), pnCodeRev));
    };
    auto defineRxSymDecim = [&](const Signal& in) {  // Chips -> Symbols
      Signal const i =
          scale(downsample(real(in), args.chipLen, args.chipLen - 1), 1.0F / args.chipLen);
      Signal const q =
          scale(downsample(imag(in), args.chipLen, args.chipLen - 1), 1.0F / args.chipLen);
      return complexPack(i, q);
    };
    auto defineRxDemap = [&](const Signal& in) {
      return (scheme == ModemScheme::PSK) ? modem::pskDemap(real(in), imag(in), args.order)
                                          : modem::qamDemap(real(in), imag(in), args.order);
    };

    std::vector<BenchStepSpec> chainSteps;
    chainSteps.push_back({"TX Map", prism::ScalarType::F32, true, defineTxMap});
    chainSteps.push_back({"TX Spread", prism::ScalarType::C32, true, defineTxSpread});
    chainSteps.push_back({"TX Upsample", prism::ScalarType::C32, true, defineTxUp});
    chainSteps.push_back({"TX Shaping", prism::ScalarType::C32, true, defineTxShape});
    chainSteps.push_back({"RX LPF", prism::ScalarType::C32, true, defineRxLpf});
    chainSteps.push_back({"RX EQ", prism::ScalarType::C32, true, defineRxEq});
    chainSteps.push_back({"RX Matched", prism::ScalarType::C32, true, defineRxMatch});
    chainSteps.push_back({"RX Downsample", prism::ScalarType::C32, true, defineRxDown});
    chainSteps.push_back({"RX Despread", prism::ScalarType::C32, true, defineRxDespread});
    chainSteps.push_back({"RX SymDecim", prism::ScalarType::C32, true, defineRxSymDecim});
    chainSteps.push_back({"RX Demap", prism::ScalarType::C32, false, defineRxDemap});

    if (args.enableCpu) {
      auto txOutCpu = txChainCpu.run(symbols);
      runStandardBenchmarks(
          args, [&](const Halide::Buffer<real32_t>& in) { return txChainCpu.run(in); },
          [&](const Halide::Buffer<real32_t>& in) { return rxChainCpu.run(in); }, symbols,
          txOutCpu, txChainCpu.targetName());
      runBenchSteps(args, txChainCpu.targetName(), ExecMode::CPU, args.cpuScheduleTx, symbols,
                    chainSteps);
    }

    if (args.enableGpu && gpuAvailable() && txChainGpu && rxChainGpu) {
      auto txOutGpu = txChainGpu->run(symbols);
      runStandardBenchmarks(
          args, [&](const Halide::Buffer<real32_t>& in) { return txChainGpu->run(in); },
          [&](const Halide::Buffer<real32_t>& in) { return rxChainGpu->run(in); }, symbols,
          txOutGpu, txChainGpu->targetName());
      runBenchSteps(args, txChainGpu->targetName(), ExecMode::GPU, args.gpuScheduleTx, symbols,
                    chainSteps);
    }
  }
  // --------------------------------------------------------------------------
  // 3. BER
  // --------------------------------------------------------------------------
  if (args.berEnable) {
    if (args.berUseGpu && args.enableGpu && txChainGpu && rxChainGpu) {
      runStandardBer(args, *txChainGpu, *rxChainGpu);
    } else if (args.enableCpu) {
      runStandardBer(args, txChainCpu, rxChainCpu);
    } else {
      std::cout << "BER 仿真已跳过: CPU/GPU 路径未启用或不可用\n";
    }
  }

  // --------------------------------------------------------------------------
  // 4. Dump Data
  // --------------------------------------------------------------------------
  if (args.output.enable) {
    double const snrDb0 = args.snrList.empty() ? 0.0 : args.snrList[0];
    std::cout << "\n正在导出仿真数据 (SNR=" << snrDb0 << "dB)...\n";
    bool const canUseGpu = args.enableGpu && txChainGpu;
    bool const canUseCpu = args.enableCpu;
    if (!canUseGpu && !canUseCpu) {
      std::cout << "导出已跳过: CPU/GPU 路径未启用或不可用\n";
      prism::shutdown();
      return 0;
    }
    // Dump Static PN Code first
    std::string dumpErr;
    if (shouldDumpStep(args.output, "pn_code")) {
      dumpCsv(args.output, "pn_code", args.pnCode, dumpErr);
    }

    RNG rng(args.seed);
    auto symbols = vectorToBuffer(randomSymbols<real32_t>(args.symbols, args.order, &rng));

    std::string const backendName =
        canUseGpu ? txChainGpu->targetName() : txChainCpu.targetName();
    StandardInspector inspector(backendName);
    ExecMode const mode = canUseGpu ? ExecMode::GPU : ExecMode::CPU;

    // 1. Map
    auto stepMap = Executor::compile<real32_t>(
        (scheme == ModemScheme::PSK) ? modem::pskMap(Signal::input(args.symbols), args.order)
                                     : modem::qamMap(Signal::input(args.symbols), args.order),
        mode, (mode == ExecMode::GPU) ? args.gpuScheduleTx : args.cpuScheduleTx);
    inspector.addStep("mapped_iq", [&](const auto& in) { return stepMap.run(in); }, true);

    // 2. Spread
    auto defineSpread = [&](const Signal& in) {
      Signal const spreadI = filter::fir(upsample(real(in), args.chipLen), args.pnCode);
      Signal const spreadQ = filter::fir(upsample(imag(in), args.chipLen), args.pnCode);
      return complexPack(spreadI, spreadQ);
    };
    auto stepSpread = Executor::compile<real32_t>(
        defineSpread(Signal::input(args.symbols, prism::ScalarType::C32)), mode,
        (mode == ExecMode::GPU) ? args.gpuScheduleTx : args.cpuScheduleTx);
    inspector.addStep("spread_iq", [&](const auto& in) { return stepSpread.run(in); }, true);

    // 3. Up + Shape (TX)
    auto defineTxShape = [&](const Signal& in) {
      Signal const upI = upsample(real(in), args.samplesPerSymbol);
      Signal const upQ = upsample(imag(in), args.samplesPerSymbol);
      Signal const shapeI = filter::fir(upI, filters.shapingTaps);
      Signal const shapeQ = filter::fir(upQ, filters.shapingTaps);
      return complexPack(shapeI, shapeQ);
    };
    auto stepTx = Executor::compile<real32_t>(
        defineTxShape(Signal::input(args.symbols * args.chipLen, prism::ScalarType::C32)), mode,
        (mode == ExecMode::GPU) ? args.gpuScheduleTx : args.cpuScheduleTx);
    inspector.addStep("tx_shaped_iq", [&](const auto& in) { return stepTx.run(in); }, true);

    // 4. Channel (Software)
    inspector.addStep(
        "rx_iq",
        [&](const auto& in) {
          RNG r(args.seed);
          auto vec = bufferToVector(in);
          auto bb = interleavedToComplex(vec);
          auto up = mixComplex(bb, args.carrierHz, args.sampleRateHz, args.txPhaseRad);
          auto ch = applyChannel(up, args.channel, args.sampleRateHz, snrDb0, r);
          auto down = mixComplex(ch, -(args.carrierHz + args.rxLoOffsetHz), args.sampleRateHz,
                                 args.rxPhaseRad);
          return complexToBuffer(down);
        },
        true);

    // 5. Equalizer (LPF + EQ)
    auto defineEq = [&](const Signal& in) {
      Signal const lpfI = filter::fir(real(in), filters.lpfTaps);
      Signal const lpfQ = filter::fir(imag(in), filters.lpfTaps);
      Signal const lpf = complexPack(lpfI, lpfQ);
      return filter::fir(lpf, eqTaps);
    };
    auto stepEq = Executor::compile<real32_t>(
        defineEq(Signal::input(rxInLen, prism::ScalarType::C32)), mode,
        (mode == ExecMode::GPU) ? args.gpuScheduleRx : args.cpuScheduleRx);
    inspector.addStep("eq_iq", [&](const auto& in) { return stepEq.run(in); }, true);

    // 6. RX
    if (mode == ExecMode::GPU) {
      inspector.addStep("demapped", [&](const auto& in) { return rxChainGpu->run(in); }, false);
    } else {
      inspector.addStep("demapped", [&](const auto& in) { return rxChainCpu.run(in); }, false);
    }

    inspector.exportStepData(args, symbols);
  }
  prism::shutdown();
  return 0;
}
