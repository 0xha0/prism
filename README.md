<p align="center">
  <img src="https://content.gatsby.icu/image/prism_logo.svg" alt="PRISM Logo" width="200">
</p>

<h1 align="center">PRISM</h1>
<p align="center"><b>Parallel RF Instructions for Signal Manipulation</b></p>
<p align="center">基于 Halide 的跨平台信号处理加速库</p>

<p align="center">
  <img src="https://img.shields.io/badge/C%2B%2B-17-blue.svg" alt="C++17">
  <img src="https://img.shields.io/badge/Platform-macOS%20|%20Linux%20|%20Windows-lightgrey.svg" alt="Platform">
</p>

---

PRISM 通过 DSL 构建惰性计算图，运行时结合 Halide JIT 与 FFT Vendor 后端实现高效无线信号处理。常规算子统一由 Halide 调度，FFT/IFFT Anchor 节点可选择 vDSP / cuFFT / hipFFT / vkFFT。

## 特性 Highlights

- **流式 DSL**：`Signal` + 算子组合描述链路，不阻塞、不立即计算。
- **Anchor 管线**：FFT/IFFT 强制走 Vendor API，性能与稳定性兼顾。
- **后端可插拔**：自动探测或手动指定 vDSP / cuFFT / hipFFT / vkFFT。
- **跨平台构建**：CMake + C++17，依赖简单。
- **仿真工具集**：随机源、信道、噪声模型开箱即用。

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
- cxxopts（`external/cxxopts` 子模块，示例 CLI）
- Graphviz（生成文档图形）
- Doxygen（生成 API 文档）

> 后端开关：在 `cmake/local.cmake` 设置 `PRISM_USE_VDSP/PRISM_USE_CUFFT/PRISM_USE_HIPFFT/PRISM_USE_VKFFT` 为 `AUTO/ON/OFF`。

## 快速开始

```bash
git clone https://github.com/<your-org>/prism.git
cd prism
git submodule update --init --recursive   # 确保 vkFFT/cxxopts 就绪

cmake -S . -B build
cmake --build build
```

最小示例：

```cpp
#include <prism/prism.h>
#include <prism/dsl/Ops.h>
#include <prism/runtime/Executor.h>

using namespace prism::dsl;
using namespace prism::runtime;

int main() {
    prism::initialize();

    Signal x = Signal::input(1024);
    Signal y = Scale(x, 0.5);
    auto out = Executor::run<prism::real32_t>(y);

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

### 生成文档

```bash
cmake --build build --target docs   # 需 Doxygen + Graphviz
open docs/generated/html/index.html
```

文档包含中文 API 手册、架构总览，并提供专门的 [测试导航](docs/tests.dox) 与 [基准导航](docs/benchmarks.dox) 页面，可直接跳转到源码。

## 后续计划

- **FFT 后端完善**：补齐 cuFFT/hipFFT/vkFFT(CUDA/HIP)，对齐批处理与可用性检测（参考现有 Metal/vDSP 结构）。
- **文档与 CI**：新增后端选择/Anchor 行为/算子限制/零拷贝与 schedule 示例；CI 加入 clang-tidy/Doxygen/格式化 gate。
- **算子与错误处理**：统一实/复数与精度接口，规划 fp16 路径；补充长度/形状/后端不可用等明确异常。
- **信道与编译码路线**：先做 ZF/MMSE 与短 FIR 均衡（CPU/GPU 可选）；FEC 从 Hamming/CRC 起步，进阶到短约束卷积码+硬判决 Viterbi。
- **示例矩阵**：PSK/QAM → +DSSS → +均衡 → +编译码 → +组帧/同步；每个示例含正确性、BER、CPU/GPU 性能对比与 README。
- **推进顺序**：均衡+编译码+示例 → FFT 后端 → fp16 与类型统一 → 调度/错误处理 → CI Gate。

## 目录结构

- `include/`：公共头文件（DSL、Runtime、Backend、Simulation）
- `src/`：对应实现
- `benchmark/`：性能与压力基准
- `tests/`：单元测试
- `docs/`：Doxygen 配置与主页（生成物在 `docs/generated`）
- `external/`：第三方依赖（例如 vkFFT、cxxopts）

## 后端说明

- **Halide JIT**：覆盖常规算子（Add/Filter/Modem 等），CPU/GPU 自动调度。
- **FFT Anchor**：FFT/IFFT 节点强制调用后端，优先级 vDSP > cuFFT > hipFFT > vkFFT > Stub。
- **手动控制**：`prism::initialize()` 默认自动选择；可通过 CMake 选项或编译宏覆盖。

## 支持与贡献

- 提 Issue 时请附：系统/编译器版本、CMake 配置、启用后端、复现步骤。
- 欢迎 PR：遵循现有风格，提交前跑 `ctest` 与相关基准。
