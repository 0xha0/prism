<div align="center">
  <img src="docs/images/prism_logo.svg" alt="PRISM Logo" width="200">

  <h1>PRISM</h1>

  <p><b>Parallel RF Instructions for Signal Manipulation</b></p>
  <p>面向无线通信与信号处理的 C++/Halide 计算库</p>

  <p>
    <img src="https://img.shields.io/badge/C%2B%2B-17-blue.svg" alt="C++17">
    <img src="https://img.shields.io/badge/Platform-macOS%20|%20Linux%20|%20Windows-lightgrey.svg" alt="Platform">
  </p>
</div>

---

PRISM 是一个面向无线通信与数字信号处理的 C++17 库。它通过 DSL 构建惰性计算图，运行时结合 Halide JIT/AOT 与 FFT Vendor 后端完成跨平台高性能执行。常规算子统一由 Halide 调度，FFT/IFFT Anchor 节点可选择 vDSP / cuFFT / hipFFT / vkFFT。

## 特性 Highlights

- **流式 DSL**：`Signal` + 算子组合描述链路，不阻塞、不立即计算
- **Anchor 管线**：FFT/IFFT 强制走 Vendor API，性能与稳定性兼顾
- **后端可插拔**：自动探测或手动指定 vDSP / cuFFT / hipFFT / vkFFT
- **跨平台构建**：CMake + C++17，依赖简单
- **仿真工具集**：随机源、信道、噪声模型开箱即用

## 环境与依赖

必需：

- CMake ≥ 3.20
- C++17 编译器（clang/clang++ 推荐）
- [Halide](https://halide-lang.org/)（`find_package(Halide REQUIRED)`）

可选（按需启用）：

- macOS Accelerate/vDSP
- NVIDIA CUDA Toolkit（cuFFT）
- AMD ROCm / hipFFT
- vkFFT 源码（`external/vkfft` 子模块，Metal/CUDA/HIP/OpenCL）
- tomlplusplus（`external/tomlplusplus` 子模块，示例配置解析）
- Graphviz（生成文档图形）
- Doxygen（生成 API 文档）

> 后端开关：在 `cmake/local.cmake` 设置 `PRISM_USE_VDSP/PRISM_USE_CUFFT/PRISM_USE_HIPFFT/PRISM_USE_VKFFT` 为 `AUTO/ON/OFF`

## 快速开始

```bash
git clone https://github.com/<your-org>/prism.git
cd prism
git submodule update --init --recursive   # 确保 vkFFT/tomlplusplus 就绪

cmake -S . -B build
cmake --build build
```

最小示例：

```cpp
#include <Halide.h>

#include <prism/prism.h>
#include <prism/dsl/ops.h>
#include <prism/runtime/executor.h>

using namespace prism::dsl;
using namespace prism::runtime;

int main() {
    prism::initialize();

    Signal x = Signal::input(1024);
    Signal y = scale(x, prism::real32_t{0.5F});

    Halide::Buffer<prism::real32_t> input(1024);
    input.fill(1.0F);
    auto out = Executor::run<prism::real32_t>(y, input);

    prism::shutdown();
    return 0;
}
```

## 构建与安装

### 标准构建

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

### 选择后端

推荐使用本地配置文件，便于复用：

```bash
cp cmake/local.cmake.example cmake/local.cmake
```

在 `cmake/local.cmake` 中设置后端：

```cmake
set(PRISM_USE_CUFFT "ON")   # 示例：强制启用 CUDA FFT
set(PRISM_USE_HIPFFT "OFF")
set(PRISM_USE_VDSP "OFF")
set(PRISM_USE_VKFFT "AUTO")
# 可选：CUDAToolkit_ROOT / ROCm 路径等
```

也可以在 `cmake/local.cmake` 中设置构建开关：

```cmake
set(PRISM_BUILD_TESTS ON)
set(PRISM_BUILD_EXAMPLES ON)
set(PRISM_BUILD_BENCHMARKS ON)
```

然后正常构建：

```bash
cmake -S . -B build
cmake --build build
```

### 运行测试

```bash
cmake --build build --target test_basic_ops test_advanced_ops test_fft test_simulation
ctest --test-dir build
```

### 运行基准

```bash
cmake --build build --target bench_ops bench_fft bench_filter bench_modem bench_stress
./build/bench_fft    # 其他基准同理
```

## 示例（apm_basic / apm_dsss / apm_dsss_eq）

依赖补充：

- Halide 运行时与 autoscheduler 库
- FFT 后端（vDSP/cuFFT/hipFFT/vkFFT 之一）
- GPU 路径需对应驱动与 Halide GPU target（如 Metal/CUDA/HIP/OpenCL）

使用流程：

1. 读取 TOML 配置并生成派生参数
2. 编译 CPU 链路；`enable_gpu=true` 且 GPU 可用时再编译 GPU 链路
3. 进行正确性验证、CPU 性能测试；GPU 可用时追加 GPU 性能测试
4. 执行 BER 仿真；若 `output.enable=true` 则导出分步数据

配置要点：

- `scheduler.tx`/`scheduler.rx` 可分别配置 `cpu`/`gpu` 的 `kind`/`name`/`extra`
- `extra.weights_path` 默认注释，填入 autotune 输出即可启用

## Autotune（高级）

PRISM 提供 Halide 官方 autoscheduler 的 autotune 脚本（Adams2019 / Anderson2021）。
**默认不会自动触发**，需要用户显式调用，避免编译时间过长。

### 前置条件

通用要求：

- 已安装 Halide（`find_package(Halide REQUIRED)` 可通过）
- 先构建 autotune 生成器：

  ```bash
  cmake --build build --target example_apm_basic_autotune example_apm_dsss_autotune example_apm_dsss_eq_autotune
  ```

- 权重文件在 `misc/`：
  - `misc/adams2019_baseline.weights`
  - `misc/anderson2021_baseline.weights`

macOS 额外：

- `gtimeout`（coreutils），否则脚本会提示安装。

Anderson2021 额外：

- 需要 `nvidia-smi`（缺失时 CMake 会自动生成 fake 版本，仅用于 GPU 数量检测）
- 需要 `libpng-config` 与 `libjpeg`（用于构建 RunGen 可执行文件）

### CMake 准备环境（推荐）

执行一次即可，CMake 会在 `build/autotune/` 下创建以下目录并建立软链接：

- `autosched_bin/`（autoscheduler 相关工具与 so）
- `halide_dist/`（`include/` 与 `tools/RunGenMain.cpp` 的 shim）
- `samples/`（输出样本）
- `env.sh`（导出上述路径）
- 可选：`bin/nvidia-smi` fake 脚本（仅在系统缺少 `nvidia-smi` 时生成）

```bash
cmake --build build --target prism_autotune_setup
```

### 使用封装脚本（推荐）

封装脚本会调用原始 loop 脚本，并使用 CMake 生成的目录（需要存在 `build/autotune/env.sh`）：

```bash
misc/adams2019.sh
misc/anderson2021.sh
```

常用可覆盖变量（环境变量）：

- 通用：`GENERATOR`、`PIPELINE`、`HALIDE_TARGET`、`WEIGHTS_FILE`
- 目录：`AUTOTUNE_DIR`、`AUTOSCHED_BIN`、`HALIDE_DISTRIB_PATH`、`HALIDE_TOOLS_DIR`、`HALIDE_BUILD_DIR`、`SAMPLES_OUT`
- 生成器参数：`GENERATOR_ARGS_SETS`（空格分组、分号分隔）
- Anderson2021：`PARALLELISM`、`TRAIN_ONLY`、`CXX`

示例（传入 base + 方向）：

```bash
misc/adams2019.sh apm_basic rx
```

### 直接调用 loop 脚本（手动）

```bash
build/autotune/tools/adams2019_autotune_loop.sh \
  build/example_apm_basic_autotune \
  apm_basic_tx \
  host \
  misc/adams2019_baseline.weights \
  build/autotune/autosched_bin \
  build/autotune/halide_dist \
  build/autotune/samples/apm_basic_tx_adams2019
```

Anderson2021 若无真实 GPU，可通过自动生成的 fake `nvidia-smi` 让脚本通过检测

### 生成文档

```bash
cmake --build build --target docs   # 需 Doxygen + Graphviz（dot）
open docs/generated/html/index.html
```

文档包含中文 API 手册、架构总览，并提供专门的 [测试导航](docs/tests.dox) 与 [基准导航](docs/benchmarks.dox) 页面，可直接跳转到源码；若未安装 Graphviz（`dot`），流程图将被跳过

## 后续计划

- **FFT 后端完善**：补齐 cuFFT/hipFFT/vkFFT(CUDA/HIP)，对齐批处理与可用性检测（参考现有 Metal/vDSP 结构）
- **文档与 CI**：完善后端选择、Anchor 行为、算子限制、零拷贝与 schedule 示例；通过 GitHub Actions 自动生成文档
- **算子与错误处理**：统一实/复数与精度接口，规划 fp16 路径；补充长度/形状/后端不可用等明确异常
- **信道与编译码路线**：先做 ZF/MMSE 与短 FIR 均衡（CPU/GPU 可选）；FEC 从 Hamming/CRC 起步，进阶到短约束卷积码+硬判决 Viterbi
- **示例矩阵**：PSK/QAM → +DSSS → +均衡 → +编译码 → +组帧/同步；每个示例含正确性、BER、CPU/GPU 性能对比与 README
- **推进顺序**：均衡+编译码+示例 → FFT 后端 → fp16 与类型统一 → 调度/错误处理 → CI Gate

## 目录结构

- `include/`：公共头文件（DSL、Runtime、Backend、Simulation）
- `src/`：对应实现
- `examples/`：综合应用示例
- `benchmark/`：性能与压力基准
- `tests/`：单元测试
- `docs/`：Doxygen 配置与主页（生成物在 `docs/generated`）
- `cmake/`：构建脚本与工具链配置
- `external/`：第三方依赖（例如 vkFFT、tomlplusplus）

## 后端说明

- **Halide JIT/AOT**：覆盖常规算子（Add/Filter/Modem 等），CPU/GPU 自动调度。
- **FFT Anchor**：FFT/IFFT 节点强制调用后端，优先级 vDSP > cuFFT > hipFFT > vkFFT > Stub。
- **手动控制**：`prism::initialize()` 默认自动选择；可通过 CMake 选项或编译宏覆盖

## 支持与贡献

- 提 Issue 时请附：系统/编译器版本、CMake 配置、启用后端、复现步骤
- 欢迎 PR：遵循现有风格，提交前跑 `ctest` 与相关基准
