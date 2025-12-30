/**
 * @file executor.cpp
 * @ingroup runtime
 * @brief Halide JIT 执行器实现
 *
 * 支持 GPU 执行（Metal/CUDA/HIP/OpenGL）和 CPU 回退
 * 支持预编译模式避免运行时 JIT 开销
 */

#include "prism/runtime/executor.h"

#include <Halide.h>

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "prism/dsl/signal.h"
#include "prism/runtime/op_handler.h"
#include "prism/runtime/schedule_config.h"
#include "prism/types.h"

// 通过显式引用符号强制链接各 Handler 目标文件
namespace prism::runtime {

/// @addtogroup runtime
/// @{

// 定义在各 Handler 源文件中
extern void registerArithmeticHandlers();
extern void registerConvolutionHandlers();
extern void registerFilterHandlers();
extern void registerModemHandlers();

namespace {
struct HandlerInitializer {
  HandlerInitializer() {
    registerArithmeticHandlers();
    registerConvolutionHandlers();
    registerFilterHandlers();
    registerModemHandlers();
  }
};
const HandlerInitializer GLOBAL_HANDLER_INIT;
}  // namespace
}  // namespace prism::runtime

namespace prism::runtime {

// ============================================================================
// 全局执行模式
// ============================================================================

namespace {
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
ExecMode gExecutionMode = ExecMode::AUTO;

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
ScheduleConfig gScheduleConfig;
}  // namespace

void Executor::setMode(ExecMode mode) { gExecutionMode = mode; }

ExecMode Executor::getMode() { return gExecutionMode; }

void Executor::setScheduleConfig(const ScheduleConfig& config) {
  gScheduleConfig = config;
}

const ScheduleConfig& Executor::getScheduleConfig() { return gScheduleConfig; }

namespace {

// ============================================================================
// GPU 目标选择
// ============================================================================

Halide::Target getGpuTarget() {
  Halide::Target target = Halide::get_host_target();

#ifdef PRISM_GPU_BACKEND_Metal
  target.set_feature(Halide::Target::Metal);
#elif defined(PRISM_GPU_BACKEND_CUDA)
  target.set_feature(Halide::Target::CUDA);
#elif defined(PRISM_GPU_BACKEND_OpenCL)
  target.set_feature(Halide::Target::OpenCL);
#endif
  return target;
}

Halide::Target getTargetForMode(ExecMode mode, int64_t extent) {
  bool useGpu = false;
  switch (mode) {
    case ExecMode::GPU:
      useGpu = true;
      break;
    case ExecMode::CPU:
      useGpu = false;
      break;
    case ExecMode::AUTO:
    default:
      constexpr int64_t gpuThreshold = 1024;
#ifdef PRISM_GPU_BACKEND_CPU
      useGpu = false;
#else
      useGpu = extent >= gpuThreshold;
#endif
      break;
  }
  return useGpu ? getGpuTarget() : Halide::get_host_target();
}

// ============================================================================
// 统一构图器
// ============================================================================

template <typename T>
Halide::Func buildFunc(const prism::dsl::Signal& signal, OpContext<T>& ctx);

template <typename T>
Halide::Func buildFunc(const prism::dsl::Signal& signal, OpContext<T>& ctx) {
  const auto* node = signal.node().get();
  if (!node) {
    throw std::runtime_error("Signal node is null");
  }

  // 缓存命中直接返回
  auto it = ctx.funcCache.find(node);
  if (it != ctx.funcCache.end()) {
    return it->second;
  }

  // 设置递归构图器，便于 Handler 调用子节点
  ctx.buildFunc = [&ctx](const prism::dsl::Signal& s) {
    return buildFunc<T>(s, ctx);
  };

  Halide::Func func;
  std::vector<Halide::Var> args;

  if constexpr (IS_COMPLEX_V<T>) {
    args.emplace_back("c");
    args.emplace_back("x");
  } else {
    args.emplace_back("x");
  }

  // Input / Constant 特判
  if (node->kind == prism::dsl::OpKind::INPUT) {
    std::vector<Halide::Expr> argsExpr;
    argsExpr.reserve(args.size());
    for (const auto& var : args) {
      argsExpr.emplace_back(var);
    }

    if (ctx.inputParam) {
      func(args) = (*ctx.inputParam)(argsExpr);
    } else if (ctx.inputBuffer) {
      func(args) = (*ctx.inputBuffer)(argsExpr);
    } else {
      throw std::runtime_error("Input buffer/param required");
    }
  } else if (node->kind == prism::dsl::OpKind::CONSTANT) {
    // 对于复数, 常量只填充实部 (imag=0)
    if constexpr (IS_COMPLEX_V<T>) {
      const Halide::Var& c = args[0];
      func(args) =
          Halide::select(c == 0,
                         Halide::cast<typename ToHalideType<T>::Type>(
                             static_cast<real32_t>(node->scalar)),
                         Halide::cast<typename ToHalideType<T>::Type>(0));
    } else {
      func(args) = Halide::cast<typename ToHalideType<T>::Type>(
          static_cast<real32_t>(node->scalar));
    }
  } else {
    // 分发到已注册的 Handler
    auto handler = OpRegistry<T>::instance().getHandler(node->kind);
    if (!handler) {
      throw std::runtime_error("OpKind not implemented: " +
                               std::to_string(static_cast<int>(node->kind)));
    }
    func = handler(node, ctx, args);
  }

  ctx.funcCache.emplace(node, func);
  return func;
}

}  // anonymous namespace

// ============================================================================
// CompiledPipeline 实现
// ============================================================================

template <typename T>
CompiledPipeline<T>::CompiledPipeline(Halide::Callable callable, int extent)
    : callable_(std::move(callable)), extent_(extent) {}

template <typename T>
Halide::Buffer<typename ToHalideType<T>::Type> CompiledPipeline<T>::run(
    const Halide::Buffer<typename ToHalideType<T>::Type>& input) {
  Halide::Buffer<typename ToHalideType<T>::Type> output(extent_);
  // If T is complex, output needs correct dimensions?
  // extent_ is flatten length of Signal.
  // If T is complex, result func is (c, x).
  // Halide::Buffer constructor (extent_) creates 1D alloc?
  // We need 2D buffer if T is complex!
  // CompiledPipeline saves extent_. But shape?

  if constexpr (IS_COMPLEX_V<T>) {
    // We need to reallocate output with (2, extent)
    // Halide::Buffer constructor taking vector of ints?
    // Or just use pre-allocated output mechanism?
    // This simplified CompiledPipeline might be too simple for Complex.
    // But wait! CompiledPipeline::run uses `callable_(input, output)`.
    // If output has wrong dimensions, callable will complain?
    // Or we construct correct buffer here.
    std::vector<int> const dims = {2, extent_};
    output = Halide::Buffer<typename ToHalideType<T>::Type>(dims);
  } else {
    // 1D
    output = Halide::Buffer<typename ToHalideType<T>::Type>(extent_);
  }

  callable_(input, output);
  return output;
}

// ============================================================================
// Executor::compile 实现
// ============================================================================

template <typename T>
CompiledPipeline<T> Executor::compile(const prism::dsl::Signal& signal,
                                      ExecMode mode) {
  const auto& shape = signal.shape();
  const int64_t extent = shape.length * shape.channels * shape.batch;
  if (extent <= 0) {
    return CompiledPipeline<T>();
  }

  // 创建 ImageParam 作为输入
  // 对于复数，使用对应的实数类型，维度+1
  // 创建 ImageParam 作为输入
  // 对于复数，使用对应的实数类型，维度+1
  // NOLINTNEXTLINE(misc-const-correctness)
  Halide::Type type = Halide::type_of<typename ToHalideType<T>::Type>();
  // NOLINTNEXTLINE(misc-const-correctness)
  int inputDims = 1;
  if constexpr (IS_COMPLEX_V<T>) {
    type = Halide::type_of<typename T::value_type>();
    inputDims = 2;  // (c, x)
  }

  Halide::ImageParam inputParam(type, inputDims, "input");

  // 构建计算图
  OpContext<T> ctx;
  ctx.inputParam = &inputParam;
  Halide::Func result = buildFunc<T>(signal, ctx);

  // 获取目标
  Halide::Target const target = getTargetForMode(mode, extent);
  bool const useGpu = target.has_feature(Halide::Target::Metal) ||
                      target.has_feature(Halide::Target::CUDA);

  // 应用调度策略
  if (useGpu) {
    gScheduleConfig.applyGpuSchedule(result, extent);
  } else {
    gScheduleConfig.applyCpuSchedule(result, extent);
  }

  // 编译为 Callable
  Halide::Callable callable = result.compile_to_callable({inputParam}, target);

  return CompiledPipeline<T>(std::move(callable), static_cast<int>(extent));
}

// ============================================================================
// Executor::run 实现（JIT 模式）
// ============================================================================

template <typename T>
Halide::Buffer<typename ToHalideType<T>::Type> Executor::run(
    const prism::dsl::Signal& signal) {
  return run<T>(signal, Halide::Buffer<typename ToHalideType<T>::Type>());
}

template <typename T>
Halide::Buffer<typename ToHalideType<T>::Type> Executor::run(
    const prism::dsl::Signal& signal,
    const Halide::Buffer<typename ToHalideType<T>::Type>& input) {
  const auto& shape = signal.shape();
  const int64_t extent = shape.length * shape.channels * shape.batch;
  if (extent <= 0) {
    return Halide::Buffer<typename ToHalideType<T>::Type>();
  }
  if (extent > static_cast<int64_t>(std::numeric_limits<int>::max())) {
    throw std::runtime_error("Signal extent exceeds Halide int limit");
  }

  OpContext<T> ctx;
  if (input.defined()) {
    ctx.inputBuffer = &input;
  }

  Halide::Func result = buildFunc<T>(signal, ctx);

  Halide::Target const target = getTargetForMode(gExecutionMode, extent);
  bool const useGpu = target.has_feature(Halide::Target::Metal) ||
                      target.has_feature(Halide::Target::CUDA);

  // 应用调度策略
  if (useGpu) {
    gScheduleConfig.applyGpuSchedule(result, extent);
  } else {
    gScheduleConfig.applyCpuSchedule(result, extent);
  }

  if constexpr (IS_COMPLEX_V<T>) {
    // (2, extent)
    // Explicitly bound 'c' to [0, 2] so that unroll(c) strategies work
    result.bound(result.args()[0], 0, 2);
    return result.realize({2, static_cast<int>(extent)}, target);
  } else {
    return result.realize({static_cast<int>(extent)}, target);
  }
}

// ============================================================================
// 显式模板实例化
// ============================================================================

/// @cond DOXYGEN_SKIP
template class CompiledPipeline<real32_t>;
template class CompiledPipeline<real64_t>;
template class CompiledPipeline<complex32_t>;
template class CompiledPipeline<complex64_t>;

template CompiledPipeline<real32_t> Executor::compile<real32_t>(
    const prism::dsl::Signal&, ExecMode);
template CompiledPipeline<real64_t> Executor::compile<real64_t>(
    const prism::dsl::Signal&, ExecMode);
template CompiledPipeline<complex32_t> Executor::compile<complex32_t>(
    const prism::dsl::Signal&, ExecMode);
template CompiledPipeline<complex64_t> Executor::compile<complex64_t>(
    const prism::dsl::Signal&, ExecMode);

template Halide::Buffer<real32_t> Executor::run<real32_t>(
    const prism::dsl::Signal&);
template Halide::Buffer<real32_t> Executor::run<real32_t>(
    const prism::dsl::Signal&, const Halide::Buffer<real32_t>&);
template Halide::Buffer<real64_t> Executor::run<real64_t>(
    const prism::dsl::Signal&);
template Halide::Buffer<real64_t> Executor::run<real64_t>(
    const prism::dsl::Signal&, const Halide::Buffer<real64_t>&);
template Halide::Buffer<real32_t> Executor::run<complex32_t>(
    const prism::dsl::Signal&);
template Halide::Buffer<real32_t> Executor::run<complex32_t>(
    const prism::dsl::Signal&, const Halide::Buffer<real32_t>&);
template Halide::Buffer<real64_t> Executor::run<complex64_t>(
    const prism::dsl::Signal&);
template Halide::Buffer<real64_t> Executor::run<complex64_t>(
    const prism::dsl::Signal&, const Halide::Buffer<real64_t>&);
/// @endcond

/// @}

}  // namespace prism::runtime
