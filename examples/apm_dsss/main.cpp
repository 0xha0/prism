/**
 * @file main.cpp
 * @brief QAM/PSK + DSSS 直接序列扩频仿真示例
 *
 * 本示例演示了如何基于 PRISM DSL 实现抗干扰能力更强的直接序列扩频 (Direct
 * Sequence Spread Spectrum) 系统
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
 * ### 接收链路 (RX) - RAKE 接收机简化版
 * 1. **匹配滤波 (Matched Filter)**: Waveform -> Soft Chips
 *    - 与发射端滤波器匹配，最大化码片信噪比
 * 2. **码片级同步与下采样**: 恢复到 Chip Rate
 * 3. **解扩 (Despreading)**: Soft Chips * PN Code -> Soft Symbols
 *    - 利用滑动相关 (Sliding Correlation) 或匹配滤波捕获相关峰值
 *    - 相关峰值处即为符号的最佳判决时刻
 * 4. **符号级下采样**: 抽取相关峰值
 * 5. **解映射 (Demap)**: Soft Symbols -> Bits
 *
 * ## 关键参数
 * - `chip_len` (扩频增益): 决定带宽扩展倍数和处理增益 ($G_p = 10 \log_{10}(L)$)
 * - `pn_seed`: 控制伪随机序列生成的种子
 */

#include <Halide.h>

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

int main(int argc, char** argv) {
  DsssArgs args;
  std::string const configPath = resolveConfigPath(argc, argv, "examples/apm_dsss/config.toml");

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
  prism::simulation::RNG pnRng(args.pnSeed);
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

  // --------------------------------------------------------------------------
  // Initial Information / 配置信息显示
  // --------------------------------------------------------------------------
  std::cout << "=== PRISM 示例: QAM/PSK 物理链路 (Refactored) ===\n\n";
  std::cout << "配置文件: " << configPath << "\n";
  std::cout << "后端: " << prism::getFftBackendName() << "\n\n";
  printStandardConfig(args);

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

    // MF: 匹配发送端的脉冲成形 (RRC)，最大化码片信噪比
    Signal const mfI = filter::fir(rxLpfI, filters.shapingTaps);
    Signal const mfQ = filter::fir(rxLpfQ, filters.shapingTaps);

    // 2. 下采样到码片速率 (Sample Rate -> Chip Rate)
    // 选择最佳采样点 (downModelDelay) 进行抽取
    Signal const chipI = downsample(mfI, args.samplesPerSymbol, filters.downModelDelay);
    Signal const chipQ = downsample(mfQ, args.samplesPerSymbol, filters.downModelDelay);
    Signal const chipSig = complexPack(chipI, chipQ);

    // 3. 解扩 (Despreading / Correlation)
    // 将接收到的码片序列与本地 PN 码 (翻转后) 进行卷积，实现滑动相关运算
    // 相关峰值将出现在符号边界处，幅度增强 L 倍
    Signal const corrI = filter::fir(real(chipSig), pnCodeRev);
    Signal const corrQ = filter::fir(imag(chipSig), pnCodeRev);

    // 4. 符号级下采样 (Chip Rate -> Symbol Rate)
    // 在相关峰值处抽取符号，延迟通常为 ChipLen - 1 (卷积带来的延迟) + 系统延迟
    Signal const symI = downsample(corrI, args.chipLen, args.chipLen - 1);
    Signal const symQ = downsample(corrQ, args.chipLen, args.chipLen - 1);

    // 5. 幅度归一化 (Normalization)
    // 扩频增益导致相关峰值幅度为 ChipLen，需归一化以匹配星座图
    Signal const normI = scale(symI, 1.0F / static_cast<real32_t>(args.chipLen));
    Signal const normQ = scale(symQ, 1.0F / static_cast<real32_t>(args.chipLen));

    // 6. 解映射 (Demapping)
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

  if (gpuAvailable()) {
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
  // 1. 参数检查
  // --------------------------------------------------------------------------
  bool const pass = runStandardVerification(
      args, [&](const Halide::Buffer<real32_t>& txIn) { return txChainCpu.run(txIn); },
      [&](const Halide::Buffer<real32_t>& rxIn) { return rxChainCpu.run(rxIn); });
  if (!pass) {
    // 验证失败，但通常我们会继续进行性能测试和误码率测试
  }

  // --------------------------------------------------------------------------
  // 2. 性能测试
  // --------------------------------------------------------------------------
  if (args.perfMinTimeMs > 0.0) {
    prism::simulation::RNG rng(args.seed);
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
    auto defineRxMatch = [&](const Signal& in) {
      return complexPack(filter::fir(real(in), filters.shapingTaps),
                         filter::fir(imag(in), filters.shapingTaps));
    };
    auto defineRxDown = [&](const Signal& in) {  // Samples -> Chips
      return complexPack(downsample(real(in), args.samplesPerSymbol, filters.downModelDelay),
                         downsample(imag(in), args.samplesPerSymbol, filters.downModelDelay));
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
    chainSteps.push_back({"RX Matched", prism::ScalarType::C32, true, defineRxMatch});
    chainSteps.push_back({"RX Downsample", prism::ScalarType::C32, true, defineRxDown});
    chainSteps.push_back({"RX Despread", prism::ScalarType::C32, true, defineRxDespread});
    chainSteps.push_back({"RX SymDecim", prism::ScalarType::C32, true, defineRxSymDecim});
    chainSteps.push_back({"RX Demap", prism::ScalarType::C32, false, defineRxDemap});

    auto txOutCpu = txChainCpu.run(symbols);
    runStandardBenchmarks(
        args, [&](const Halide::Buffer<real32_t>& in) { return txChainCpu.run(in); },
        [&](const Halide::Buffer<real32_t>& in) { return rxChainCpu.run(in); }, symbols, txOutCpu,
        txChainCpu.targetName());
    runBenchSteps(args, txChainCpu.targetName(), ExecMode::CPU, args.cpuScheduleTx, symbols,
                  chainSteps);

    if (gpuAvailable() && txChainGpu && rxChainGpu) {
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
  if (args.enableGpu && txChainGpu && rxChainGpu) {
    runStandardBer(args, *txChainGpu, *rxChainGpu);
  } else {
    runStandardBer(args, txChainCpu, rxChainCpu);
  }

  // --------------------------------------------------------------------------
  // 4. Dump Data
  // --------------------------------------------------------------------------
  if (args.output.enable) {
    std::cout << "\n正在导出仿真数据 (SNR=" << args.snrList[0] << "dB)...\n";
    // Dump Static PN Code first
    std::string dumpErr;
    if (shouldDumpStep(args.output, "pn_code")) {
      dumpCsv(args.output, "pn_code", args.pnCode, dumpErr);
    }

    prism::simulation::RNG rng(args.seed);
    auto symbols = vectorToBuffer(randomSymbols<real32_t>(args.symbols, args.order, &rng));

    std::string backendName =
        args.enableGpu && txChainGpu ? txChainGpu->targetName() : txChainCpu.targetName();
    StandardInspector inspector(backendName);
    ExecMode const mode = (args.enableGpu && txChainGpu) ? ExecMode::GPU : ExecMode::CPU;

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
          prism::simulation::RNG r(args.seed);
          auto vec = bufferToVector(in);
          auto bb = interleavedToComplex(vec);
          auto up = mixComplex(bb, args.carrierHz, args.sampleRateHz, args.txPhaseRad);
          auto ch = applyChannel(up, args.channel, args.sampleRateHz, args.snrList[0], r);
          auto down = mixComplex(ch, -(args.carrierHz + args.rxLoOffsetHz), args.sampleRateHz,
                                 args.rxPhaseRad);
          return complexToBuffer(down);
        },
        true);

    // 5. RX
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
