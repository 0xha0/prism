/**
 * @file schedule_config.cpp
 * @ingroup runtime
 * @brief Halide 调度策略配置实现
 *
 * 实现自适应调度策略选择和 Halide Func 调度应用。
 */

#include "prism/runtime/schedule_config.h"

#include <Halide.h>

#include <algorithm>
#include <cstdint>
#include <vector>

namespace prism::runtime {

/// @addtogroup runtime
/// @{

// ============================================================================
// 自动调度策略阈值常量
// ============================================================================
// 这些阈值基于典型 GPU 架构的 warp/wavefront 大小和寄存器压力经验值选取

namespace {
/// @name GPU Tile 大小选择阈值
/// @{
constexpr int64_t GPU_EXTENT_SMALL = 256;    ///< <= 此值使用小 tile
constexpr int64_t GPU_EXTENT_MEDIUM = 1024;  ///< <= 此值使用中 tile
constexpr int64_t GPU_EXTENT_LARGE = 4096;   ///< <= 此值使用标准 tile
/// @}

/// @name GPU Tile 大小
/// @{
constexpr int GPU_TILE_SMALL = 64;    ///< 小数据 tile
constexpr int GPU_TILE_MEDIUM = 128;  ///< 中等数据 tile
constexpr int GPU_TILE_LARGE = 256;   ///< 大数据 tile
constexpr int GPU_TILE_XLARGE = 512;  ///< 巨型数据 tile
/// @}

/// @name CPU 向量化阈值
/// @{
constexpr int64_t CPU_EXTENT_MIN_SIMD = 8;    ///< 最小 SIMD 阈值
constexpr int64_t CPU_EXTENT_WIDE_SIMD = 32;  ///< 宽 SIMD 阈值 (AVX)
/// @}

/// @name CPU 向量化宽度
/// @{
constexpr int CPU_VECTOR_NONE = 1;    ///< 不向量化
constexpr int CPU_VECTOR_NARROW = 4;  ///< SSE/NEON
constexpr int CPU_VECTOR_WIDE = 8;    ///< AVX/AVX2
/// @}
}  // namespace

// ============================================================================
// 自动 GPU Tile 大小选择策略
// ============================================================================
//
// 设计原理:
// - 小数据 (<64): 不使用 tiling，调度开销大于收益
// - 中等数据 (64-256): 使用较小 tile (64)，保持合理占用率
// - 大数据 (257-1024): 中等 tile (128)
// - 超大数据 (1025-4096): 标准 tile (256)
// - 巨型数据 (>4096): 大 tile (512)，最大化吞吐量
//
// 这些阈值基于典型 GPU 架构（Metal/CUDA）的 warp/wavefront 大小和
// 寄存器压力经验值选取，用户可通过自定义配置覆盖。
// ============================================================================

int ScheduleConfig::getEffectiveGpuTileSize(int64_t extent) const {
  // 数据太小，不使用 tiling
  if (extent < gpuMinExtent) {
    return 0;
  }

  // 用户显式指定，直接使用
  if (gpuTileSize > 0) {
    // 确保 tile 不超过数据长度
    return std::min(gpuTileSize, static_cast<int>(extent));
  }

  // 自动选择策略
  if (extent <= GPU_EXTENT_SMALL) {
    return GPU_TILE_SMALL;
  }
  if (extent <= GPU_EXTENT_MEDIUM) {
    return GPU_TILE_MEDIUM;
  }
  if (extent <= GPU_EXTENT_LARGE) {
    return GPU_TILE_LARGE;
  }
  return GPU_TILE_XLARGE;
}

// ============================================================================
// 自动 CPU 向量化宽度选择策略
// ============================================================================
//
// 设计原则：
// - 数据 <8: 不向量化
// - 数据 8-31: 4-wide SIMD (SSE/NEON)
// - 数据 >=32: 8-wide SIMD (AVX/AVX2)
//
// AVX-512 可显式设置 cpuVectorWidth = 16。
// ============================================================================

int ScheduleConfig::getEffectiveCpuVectorWidth(int64_t extent) const {
  // 用户显式指定
  if (cpuVectorWidth > 0) {
    return cpuVectorWidth;
  }

  // 自动选择策略
  if (extent < CPU_EXTENT_MIN_SIMD) {
    return CPU_VECTOR_NONE;
  }
  if (extent < CPU_EXTENT_WIDE_SIMD) {
    return CPU_VECTOR_NARROW;  // SSE/NEON
  }
  return CPU_VECTOR_WIDE;  // AVX/AVX2
}

bool ScheduleConfig::shouldEnableCpuParallel(int64_t extent) const {
  return cpuParallel && extent >= cpuParallelThreshold;
}

// ============================================================================
// GPU 调度应用
// ============================================================================

void ScheduleConfig::applyGpuSchedule(Halide::Func& result,
                                      int64_t extent) const {
  int const tileSize = getEffectiveGpuTileSize(extent);

  if (tileSize <= 0) {
    // 数据太小或用户禁用 tiling，不应用调度
    // Halide 将使用默认的逐元素 GPU kernel
    return;
  }

  // 取出 x 维（约定最后一维为 x）
  std::vector<Halide::Var> args = result.args();
  if (args.empty()) return;

  Halide::Var const& x = args.back();

  if (args.size() > 1 && args[0].name() == "c") {
    result.unroll(args[0], 2);
  }

  Halide::Var const xo("xo");
  Halide::Var const xi("xi");

  // 应用 GPU tiling
  // gpu_tile 会将循环分割为 tile 并映射到 GPU block/thread
  result.gpu_tile(x, xo, xi, tileSize);
}

// ============================================================================
// CPU 调度应用
// ============================================================================

void ScheduleConfig::applyCpuSchedule(Halide::Func& result,
                                      int64_t extent) const {
  int const vectorWidth = getEffectiveCpuVectorWidth(extent);
  bool const enableParallel = shouldEnableCpuParallel(extent);

  std::vector<Halide::Var> args = result.args();
  if (args.empty()) return;

  Halide::Var const& x = args.back();

  // 处理复数通道维度 c
  if (args.size() > 1 && args[0].name() == "c") {
    // 展开固定长度的 c 维（长度 2），提升缓存局部性
    result.unroll(args[0]);
  }

  if (enableParallel && vectorWidth > 1) {
    // 并行 + 向量化: 外层并行，内层向量化
    Halide::Var const xo("xo");
    Halide::Var const xi("xi");
    result.split(x, xo, xi, vectorWidth);
    result.parallel(xo);
    result.vectorize(xi);
  } else if (enableParallel) {
    // 仅并行（数据太小无法向量化）
    result.parallel(x);
  } else if (vectorWidth > 1) {
    // 仅向量化（数据不足以并行）
    result.vectorize(x, vectorWidth);
  }
}

/// @}

}  // namespace prism::runtime
