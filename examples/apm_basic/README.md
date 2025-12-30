# QAM/PSK 基础链路示例

本示例演示 PRISM 的 QAM/PSK 符号映射与解映射流程，包含三部分：

- 正确性验证（无噪声往返）
- CPU/GPU 性能对比（Halide 调度）
- BER 仿真（多 SNR 统计）

## 构建

```bash
cmake -S . -B build
cmake --build build --target example_apm_basic
```

## 运行

```bash
./build/example_apm_basic
```

常用参数：

```bash
./build/example_apm_basic --order 16 --scheme qam --symbols 8192 --snr 0,5,10,15 --iters 100 --perf-iters 50
```

参数说明：

- `--order`：调制阶数，必须为 2 的幂；QAM 要求为完全平方数。
- `--scheme`：`auto/qam/psk`，默认 `auto`（当阶数为 2 时自动使用 BPSK）。
- `--symbols`：每轮仿真符号数。
- `--snr`：SNR 列表（dB），逗号分隔。
- `--iters`：BER 统计轮数。
- `--perf-iters`：性能统计迭代轮数。
- `--seed`：随机种子。
- `--no-gpu`：跳过 GPU 模式。

## 输出说明

输出包含三段：

1. 正确性验证：无噪声情况下，映射后解调应完全一致。
2. 性能对比：CPU/GPU 的 Map、Demap 与端到端延迟。
3. BER 仿真：输出每个 SNR 的 BER 与错误计数。

## 示例输出

```text
=== PRISM 示例: QAM/PSK 基础链路 ===

配置:
  scheme: QAM
  order: 16 (bits/sym=4)
  symbols: 4096
  iters: 50
  perf iters: 50
  seed: 42
  backend: vDSP
  gpu: 可用

正确性验证: PASS
  symbol errors: 0
  bit errors: 0

性能对比 (CPU):
  Map         : 0.320 ms, 12.80 MSym/s
  Demap       : 0.410 ms, 9.98 MSym/s
  End-to-End  : 0.780 ms, 5.25 MSym/s

性能对比 (GPU):
  Map         : 0.120 ms, 34.11 MSym/s
  Demap       : 0.180 ms, 22.72 MSym/s
  End-to-End  : 0.350 ms, 11.70 MSym/s

BER 仿真:
  SNR(dB)      BER        BitErrors/TotalBits
     0.0   3.200e-01   26240/81920
     5.0   1.210e-01   9920/81920
    10.0   1.400e-02   1147/81920
    15.0   4.000e-04   33/81920
```
