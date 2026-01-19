/**
 * @file main.cpp
 * @brief QAM/PSK 物理链路仿真示例
 *
 * 本示例演示了如何使用 PRISM DSL 构建一个完整的单载波通信链路仿真系统
 *
 * ## 主要功能
 * - **调制方式**：支持 M-PSK 和 M-QAM (配置可调)
 * - **链路流程**：
 *   - **发射端 (TX)**：符号映射 -> 上采样 (插零) -> 脉冲成形 (RRC 滤波) ->
 * 混频/打包
 *   - **信道 (Channel)**：支持 AWGN (高斯白噪声)、多径衰落、频偏等损伤模型
 * (软件仿真)
 *   - **接收端 (RX)**：相关/匹配滤波 -> 符号定时同步 (下采样) -> 解映射
 * - **高性能执行**：利用 Halide 后端 (CPU/GPU) 自动优化计算图
 * - **基准测试**：内置 BER (误码率) 统计和吞吐率性能测试工具
 *
 * ## 配置与运行
 * 默认读取 `config.toml` 文件中的参数；支持通过命令行参数指定配置文件路径
 */

#include <Halide.h>

#include <cstdint>
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
  StandardArgs args;
  std::string const configPath = resolveConfigPath(argc, argv, "examples/apm_basic/config.toml");

  std::string err;
  if (!loadStandardConfig(configPath, args, err)) {
    std::cerr << err << "\n";
    return 1;
  }
  if (!finalizeStandardArgs(args, err)) {
    std::cerr << err << "\n";
    return 1;
  }

  // Filters
  StandardFilters filters;
  if (!setupStandardFilters(args, filters, err)) {
    std::cerr << err << "\n";
    return 1;
  }

  // --------------------------------------------------------------------------
  // Initial Information / 配置信息显示
  // --------------------------------------------------------------------------
  std::cout << "=== PRISM 示例: QAM/PSK ===\n\n";
  std::cout << "配置文件: " << configPath << "\n";
  std::cout << "后端: " << prism::getFftBackendName() << "\n\n";
  printStandardConfig(args);

  prism::initialize();

  // --------------------------------------------------------------------------
  // Pipeline Definitions / DSL 计算图构建
  // --------------------------------------------------------------------------

  // 定义发射链路 (Transmitter Pipeline)
  // -----------------------------------------------------------
  // 流程：符号映射 -> 上采样 -> 脉冲成型滤波 -> I/Q 打包
  auto const& scheme = args.scheme;
  auto defineTx = [&](const Signal& input) {
    // 1. 符号映射 (Symbol Mapping)
    // 将输入比特索引映射为星座图上的复数符号 (Symbol Rate)
    Signal const mapSig = (scheme == ModemScheme::PSK) ? modem::pskMap(input, args.order)
                                                       : modem::qamMap(input, args.order);

    // 2. 上采样 (Upsampling)
    // 插入零值以提高采样率，为脉冲成形防止混叠做准备 (Sample Rate = Symbol Rate
    // * SPS)
    Signal const upI = upsample(real(mapSig), args.samplesPerSymbol);
    Signal const upQ = upsample(imag(mapSig), args.samplesPerSymbol);

    // 3. 脉冲成型 (Pulse Shaping)
    // 限制信号带宽，通常使用根升余弦 (RRC)
    // 滤波器以满足奈奎斯特准则并消除码间串扰 (ISI)
    Signal const shapeI = filter::fir(upI, filters.shapingTaps);
    Signal const shapeQ = filter::fir(upQ, filters.shapingTaps);

    // 4. 正交打包 (I/Q Packing)
    // 合并同相 (I) 和正交 (Q) 分量为复数信号
    return complexPack(shapeI, shapeQ);
  };

  // 定义接收链路 (Receiver Pipeline)
  // -----------------------------------------------------------
  // 流程：抗混叠滤波 -> 匹配滤波 -> 下采样/抽取 -> 解映射
  auto defineRx = [&](const Signal& rxInput) {
    // 1. 接收前端滤波 (Front-end Filtering)
    // 低通滤波以抑制带外噪声和干扰 (Anti-aliasing / Noise Rejection)
    Signal const rxLpfI = filter::fir(real(rxInput), filters.lpfTaps);
    Signal const rxLpfQ = filter::fir(imag(rxInput), filters.lpfTaps);

    // 2. 匹配滤波 (Matched Filtering)
    // 与发送端 RRC 匹配，最大化信噪比 (SNR) 并消除 ISI
    Signal const mfI = filter::fir(rxLpfI, filters.shapingTaps);
    Signal const mfQ = filter::fir(rxLpfQ, filters.shapingTaps);

    // 3. 下采样/抽取 (Downsampling / Decimation)
    // 在最佳采样时刻抽取符号，恢复到符号速率
    // 需要准确的定时同步 (这里假设理想定时，通过 delay 参数控制)
    Signal const downI = downsample(mfI, args.samplesPerSymbol, filters.downModelDelay);
    Signal const downQ = downsample(mfQ, args.samplesPerSymbol, filters.downModelDelay);

    // 4. 解映射 (Demapping)
    // 根据星座图判决，将复数符号还原为比特/符号索引 (Soft/Hard Decision)
    return (scheme == ModemScheme::PSK) ? modem::pskDemap(downI, downQ, args.order)
                                        : modem::qamDemap(downI, downQ, args.order);
  };

  // Compile
  // Note: We need ExecHandles for the verification/benchmark helpers
  auto txChainCpu = Executor::compile<real32_t>(defineTx(Signal::input(args.symbols)),
                                                ExecMode::CPU, args.cpuScheduleTx);
  int64_t const rxInputSize = static_cast<int64_t>(args.symbols) * args.samplesPerSymbol;
  auto rxChainCpu =
      Executor::compile<real32_t>(defineRx(Signal::input(rxInputSize, prism::ScalarType::C32)),
                                  ExecMode::CPU, args.cpuScheduleRx);

  using ExecHandle = decltype(txChainCpu);
  std::optional<ExecHandle> txChainGpu;
  std::optional<ExecHandle> rxChainGpu;

  if (args.enableGpu && gpuAvailable()) {
    try {
      txChainGpu = Executor::compile<real32_t>(defineTx(Signal::input(args.symbols)), ExecMode::GPU,
                                               args.gpuScheduleTx);
      rxChainGpu =
          Executor::compile<real32_t>(defineRx(Signal::input(rxInputSize, prism::ScalarType::C32)),
                                      ExecMode::GPU, args.gpuScheduleRx);
    } catch (const std::exception& e) {
      std::cout << "GPU compilation failed: " << e.what() << "\n";
    }
  }

  // --------------------------------------------------------------------------
  // Execution
  // --------------------------------------------------------------------------

  if (args.enableCpu) {
    bool const pass = runStandardVerification(
        args, [&](const Halide::Buffer<real32_t>& in) { return txChainCpu.run(in); },
        [&](const Halide::Buffer<real32_t>& in) { return rxChainCpu.run(in); });
    if (!pass) {
      // verification failed, but we continue to perf/BER usually
    }
  } else if (args.enableGpu && txChainGpu && rxChainGpu) {
    bool const pass = runStandardVerification(
        args, [&](const Halide::Buffer<real32_t>& in) { return txChainGpu->run(in); },
        [&](const Halide::Buffer<real32_t>& in) { return rxChainGpu->run(in); });
    if (!pass) {
      // verification failed, but we continue to perf/BER usually
    }
  }

  // --------------------------------------------------------------------------
  // 2. Benchmarks (Chain & Step-wise)
  // --------------------------------------------------------------------------
  if (args.perfMinTimeMs > 0.0) {
    prism::simulation::RNG rng(args.seed);
    auto symbols = vectorToBuffer(randomSymbols<real32_t>(args.symbols, args.order, &rng));

    auto defineTxMap = [&](const Signal& in) {
      return (scheme == ModemScheme::PSK) ? modem::pskMap(in, args.order)
                                          : modem::qamMap(in, args.order);
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
    auto defineRxDown = [&](const Signal& in) {
      return complexPack(downsample(real(in), args.samplesPerSymbol, filters.downModelDelay),
                         downsample(imag(in), args.samplesPerSymbol, filters.downModelDelay));
    };
    auto defineRxDemap = [&](const Signal& in) {
      return (scheme == ModemScheme::PSK) ? modem::pskDemap(real(in), imag(in), args.order)
                                          : modem::qamDemap(real(in), imag(in), args.order);
    };

    std::vector<BenchStepSpec> chainSteps;
    chainSteps.push_back({"TX Map", prism::ScalarType::F32, true, defineTxMap});
    chainSteps.push_back({"TX Upsample", prism::ScalarType::C32, true, defineTxUp});
    chainSteps.push_back({"TX Shaping", prism::ScalarType::C32, true, defineTxShape});
    chainSteps.push_back({"RX LPF", prism::ScalarType::C32, true, defineRxLpf});
    chainSteps.push_back({"RX Matched", prism::ScalarType::C32, true, defineRxMatch});
    chainSteps.push_back({"RX Downsample", prism::ScalarType::C32, true, defineRxDown});
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

    prism::simulation::RNG rng(args.seed);
    auto symbols = vectorToBuffer(randomSymbols<real32_t>(args.symbols, args.order, &rng));

    ExecMode const mode = canUseGpu ? ExecMode::GPU : ExecMode::CPU;
    std::string const backendName =
        canUseGpu ? txChainGpu->targetName() : txChainCpu.targetName();
    StandardInspector inspector(backendName);

    // 1. Map
    auto stepMap = Executor::compile<real32_t>(
        (scheme == ModemScheme::PSK) ? modem::pskMap(Signal::input(args.symbols), args.order)
                                     : modem::qamMap(Signal::input(args.symbols), args.order),
        mode, (mode == ExecMode::GPU) ? args.gpuScheduleTx : args.cpuScheduleTx);
    inspector.addStep("mapped_iq", [&](const auto& in) { return stepMap.run(in); }, true);

    // 2. TX (Upsample + Shape)
    auto defineTxShape = [&](const Signal& in) {
      Signal const upI = upsample(real(in), args.samplesPerSymbol);
      Signal const upQ = upsample(imag(in), args.samplesPerSymbol);
      Signal const shapeI = filter::fir(upI, filters.shapingTaps);
      Signal const shapeQ = filter::fir(upQ, filters.shapingTaps);
      return complexPack(shapeI, shapeQ);
    };
    auto stepTx = Executor::compile<real32_t>(
        defineTxShape(Signal::input(args.symbols, prism::ScalarType::C32)), mode,
        (mode == ExecMode::GPU) ? args.gpuScheduleTx : args.cpuScheduleTx);
    inspector.addStep("tx_shaped_iq", [&](const auto& in) { return stepTx.run(in); }, true);

    // 3. Upconvert
    inspector.addStep(
        "upconverted",
        [&](const auto& in) {
          auto vec = bufferToVector(in);
          auto bb = interleavedToComplex(vec);
          auto up = mixComplex(bb, args.carrierHz, args.sampleRateHz, args.txPhaseRad);
          return complexToBuffer(up);
        },
        true);

    // 4. Channel
    inspector.addStep(
        "channel",
        [&](const auto& in) {
          prism::simulation::RNG r(args.seed);
          auto vec = bufferToVector(in);
          auto bb = interleavedToComplex(vec);
          auto ch = applyChannel(bb, args.channel, args.sampleRateHz, snrDb0, r);
          return complexToBuffer(ch);
        },
        true);

    // 5. Downconvert
    inspector.addStep(
        "downconverted",
        [&](const auto& in) {
          auto vec = bufferToVector(in);
          auto bb = interleavedToComplex(vec);
          auto down = mixComplex(bb, -(args.carrierHz + args.rxLoOffsetHz), args.sampleRateHz,
                                 args.rxPhaseRad);
          return complexToBuffer(down);
        },
        true);

    // 6. RX IQ (baseband)
    inspector.addStep("rx_iq", [&](const auto& in) { return in; }, true);

    // 7. Demap
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
