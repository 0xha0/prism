# QAM/PSK 物理链路示例

本示例在 QAM/PSK 基础链路上加入上采样、成形滤波、上/下变频与信道，
用于 BER 统计与性能对比。参数通过 TOML 配置。

## 构建

```bash
cmake -S . -B build
cmake --build build --target example_apm_basic
```

## 运行

```bash
./build/example_apm_basic
```

自定义配置路径：

```bash
./build/example_apm_basic examples/apm_basic/config.toml
```

> 若在 `build/` 目录内运行，请显式传入配置路径。

## 依赖

- Halide（运行时与 autoscheduler 库）
- FFT 后端（vDSP/cuFFT/hipFFT/vkFFT 之一，按需启用）
- GPU 路径需对应驱动与 Halide GPU target（如 Metal/CUDA/HIP/OpenCL）

## 使用流程

1. 读取 TOML 配置并生成派生参数。
2. 编译 CPU 链路；若 `enable_gpu=true` 且 GPU 可用则编译 GPU 链路。
3. 进行正确性验证、CPU 性能测试；GPU 可用时追加 GPU 性能测试。
4. 执行 BER 仿真；如开启 `output.enable` 则导出分步数据。

## 配置说明 (TOML)

### `[modem]` 调制参数

| 参数名 | 说明 |
| :--- | :--- |
| `order` | 调制阶数 (e.g., 2, 4, 16) |
| `scheme` | 调制方式 (`psk`, `qam`) |
| `symbols` | 仿真符号总数 |

### `[sim]` 仿真控制

| 参数名 | 说明 |
| :--- | :--- |
| `snr_db` | 信噪比 (dB) 列表 |
| `iters` | 每个 SNR 点的仿真次数 |
| `perf_min_time_ms` | 性能测试最小运行时长 (ms) |
| `seed` | 随机数种子 |
| `enable_gpu` | 是否启用 GPU 加速 |

### `[sampling]` 采样与频率

| 参数名 | 说明 |
| :--- | :--- |
| `samples_per_symbol` | 每个符号的采样点数 (过采样率) |
| `symbol_rate_hz` | 符号率 (Hz) |
| `sample_rate_hz` | 采样率 (Hz) |
| `carrier_hz` | 载波频率 (Hz) |
| `rx_lo_offset_hz` | 接收端本振频偏 (Hz) |
| `tx_phase_rad` | 发射端初始相位 (rad) |
| `rx_phase_rad` | 接收端初始相位 (rad) |

### `[filter]` 成形滤波

| 参数名 | 说明 |
| :--- | :--- |
| `mode` | 滤波器类型 (`rrc`: 根升余弦, `rc`: 升余弦, `none`: 无) |
| `rolloff` | 滚降系数 (0.0 - 1.0) |
| `span` | 滤波器长度 (符号数) |
| `normalize` | 是否归一化 |

### `[lpf]` 低通滤波

| 参数名 | 说明 |
| :--- | :--- |
| `order` | 低通 FIR 阶数 (tap 数，建议奇数) |
| *截止频率* | 自动根据`采样率`和`rolloff`生成 |

### `[channel]` 信道模型

| 参数名 | 说明 |
| :--- | :--- |
| `enable_awgn` | 是否启用高斯白噪声 |
| `enable_fading` | 是否启用多径衰落 |
| `fading_taps` | 衰落路径数 |
| `fading_delays` | 各路径延迟 (sample) |
| `doppler_hz` | 多普勒频移 (Hz) |
| `cfo_hz` | 载波频偏 (Hz) |
| `phase_noise_std` | 相位噪声标准差 |
| `gain` | 信道增益 |

### `[scheduler]` 调度配置

分别配置 `tx` / `rx` 的 `cpu` / `gpu`：

| 参数名 | 说明 |
| :--- | :--- |
| `kind` | 调度器类型 (`autoscheduler` 等) |
| `name` | 调度器名称 (`Anderson2021`, `Adams2019` 等) |
| `extra.weights_path` | (可选) Autotune 权重文件路径 |

### `[output]` 数据导出

| 参数名 | 说明 |
| :--- | :--- |
| `enable` | 是否启用数据导出 |
| `dir` | 导出目录 |
| `steps` | 需导出的步骤列表 (为空导出所有) |

`perf_min_time_ms` 对应 RunGen 的 `--benchmark_min_time`。

`perf_min_time_ms` 对应 RunGen 的 `--benchmark_min_time`，使用 Halide 官方 benchmark 进行计时。

## 输出说明

输出包含三段：

1. 正确性验证（理想链路往返）。
2. 性能对比（TX 成形 + RX 解调）。
3. BER 仿真（按 SNR 列表统计）。
