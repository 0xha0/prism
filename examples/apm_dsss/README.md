# QAM/PSK + DSSS 示例

本示例在基础 QAM/PSK 链路中加入 DSSS（直接序列扩频），展示扩频/解扩、
CPU/GPU 性能以及 BER 仿真结果。

## 构建

```bash
cmake -S . -B build
cmake --build build --target example_apm_dsss
```

## 运行

```bash
./build/example_apm_dsss
```

常用参数：

```bash
./build/example_apm_dsss --order 4 --scheme qam --chip-len 11 --symbols 4096 --snr 0,5,10 --iters 80
```

参数说明：

- `--order`：调制阶数，必须为 2 的幂；QAM 要求为完全平方数。
- `--scheme`：`auto/qam/psk`，默认 `auto`（当阶数为 2 时自动使用 BPSK）。
- `--chip-len`：扩频长度（处理增益）。
- `--symbols`：每轮仿真符号数。
- `--snr`：SNR 列表（dB），逗号分隔。
- `--iters`：BER 统计轮数。
- `--perf-iters`：性能统计迭代轮数。
- `--seed`：随机种子。
- `--no-gpu`：跳过 GPU 模式。

## 输出说明

输出包含三段：

1. 正确性验证：无噪声情况下，扩频/解扩后仍可正确解调。
2. 性能对比：CPU/GPU 的 Map、Demap、Spread、Despread 与端到端延迟。
3. BER 仿真：输出每个 SNR 的 BER 与错误计数。

> 说明：本示例的 AWGN 噪声在扩频后的序列上添加，因此 SNR 的定义是“按 chip 级别”。

## 示例输出

```text
=== PRISM 示例: QAM/PSK + DSSS ===

配置:
  scheme: PSK
  order: 2 (bits/sym=1)
  symbols: 2048
  chip len: 8
  iters: 30
  perf iters: 30
  seed: 42
  backend: vDSP
  gpu: 未启用

正确性验证: PASS
  symbol errors: 0
  bit errors: 0

性能对比 (CPU):
  Map         : 0.210 ms, 9.75 MSym/s
  Demap       : 0.230 ms, 8.91 MSym/s
  Spread      : 0.450 ms, 4.55 MSym/s
  Despread    : 0.480 ms, 4.27 MSym/s
  End-to-End  : 1.520 ms, 1.35 MSym/s

BER 仿真 (SNR 以扩频后序列为基准):
  SNR(dB)      BER        BitErrors/TotalBits
     0.0   2.100e-01   12924/61440
     5.0   6.400e-02   3938/61440
    10.0   5.000e-03   307/61440
```
