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
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <functional>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

#include "prism/dsl/signal.h"
#include "prism/runtime/halide_builder.h"
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

std::string autoschedulerPluginPath(const std::string& name) {
#ifdef PRISM_AUTOSCHEDULER_MULLAPUDI2016_PATH
  if (name == "Mullapudi2016") {
    return PRISM_AUTOSCHEDULER_MULLAPUDI2016_PATH;
  }
#endif
#ifdef PRISM_AUTOSCHEDULER_ADAMS2019_PATH
  if (name == "Adams2019") {
    return PRISM_AUTOSCHEDULER_ADAMS2019_PATH;
  }
#endif
#ifdef PRISM_AUTOSCHEDULER_ANDERSON2021_PATH
  if (name == "Anderson2021") {
    return PRISM_AUTOSCHEDULER_ANDERSON2021_PATH;
  }
#endif
  return {};
}

bool loadAutoschedulerPlugin(const std::string& name) {
  static std::unordered_set<std::string> loaded;
  if (loaded.count(name) != 0) {
    return true;
  }
  const std::string path = autoschedulerPluginPath(name);
  if (path.empty() || !std::filesystem::exists(path)) {
    return false;
  }
  Halide::load_plugin(path);
  loaded.insert(name);
  return true;
}

SchedulerConfig makeJitSchedule(const SchedulerConfig& schedule) {
  SchedulerConfig config = schedule;
  if (config.kind == SchedulerKind::NONE) {
    config.kind = SchedulerKind::AUTO;
    config.name = "Mullapudi2016";
    return config;
  }
  if (config.kind == SchedulerKind::AUTO && config.name.empty()) {
    config.name = "Mullapudi2016";
  }
  return config;
}

std::string getGpuBackendName(const Halide::Target* target) {
  if (target) {
    if (target->has_feature(Halide::Target::Metal)) {
      return "Metal";
    }
    if (target->has_feature(Halide::Target::CUDA)) {
      return "CUDA";
    }
    if (target->has_feature(Halide::Target::OpenCL)) {
      return "OpenCL";
    }
  }
#ifdef PRISM_GPU_BACKEND_Metal
  return "Metal";
#elif defined(PRISM_GPU_BACKEND_CUDA)
  return "CUDA";
#elif defined(PRISM_GPU_BACKEND_OpenCL)
  return "OpenCL";
#else
  return "";
#endif
}

std::string formatTargetName(ExecMode mode, const Halide::Target* target) {
  if (mode == ExecMode::AUTO) {
    return "Auto";
  }

  if (mode == ExecMode::CPU) {
    Halide::Target const cpuTarget = target ? *target : Halide::get_host_target();
    return "CPU (" + cpuTarget.to_string() + ")";
  }

  if (mode == ExecMode::GPU) {
    std::string name = "GPU";
    std::string const backend = getGpuBackendName(target);
    if (!backend.empty()) {
      name += " (" + backend + ")";
    }
    return name;
  }

  return "Unknown";
}
}  // namespace

void Executor::setMode(ExecMode mode) { gExecutionMode = mode; }

ExecMode Executor::getMode() { return gExecutionMode; }

std::string Executor::getCurrentTargetName() { return formatTargetName(gExecutionMode, nullptr); }

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

bool scheduleEnabled() {
  const char* env = std::getenv("PRISM_DISABLE_SCHEDULE");
  if (!env || env[0] == '\0') {
    return true;
  }
  if (std::strcmp(env, "0") == 0) {
    return true;
  }
  return false;
}

Halide::Type getElementType(ScalarType type) {
  int const bytes = getComponentSize(type);
  int const bits = bytes * 8;
  if (isFloatType(type)) {
    switch (bits) {
      case 16:  // NOLINT
      case 32:  // NOLINT
      case 64:  // NOLINT
        return Halide::Float(bits);
      default:
        throw std::runtime_error("Unsupported float ScalarType size");
    }
  }

  bool const isSigned = (static_cast<std::uint8_t>(type) & ::prism::detail::SIGNED_FLAG) != 0;
  switch (bits) {
    case 8:   // NOLINT
    case 16:  // NOLINT
    case 32:  // NOLINT
    case 64:  // NOLINT
      return isSigned ? Halide::Int(bits) : Halide::UInt(bits);
    default:
      throw std::runtime_error("Unsupported integer ScalarType size");
  }
}

template <typename T>
void ensureOutputTypeMatch(const prism::dsl::Signal& signal) {
  ScalarType const expected = getScalarType<T>();
  if (!isPrecisionMatch(signal.type(), expected)) {
    throw std::runtime_error("Executor: output precision mismatch");
  }
}

}  // anonymous namespace

// ============================================================================
// CompiledPipeline 实现
// ============================================================================

template <typename T>
CompiledPipeline<T>::CompiledPipeline(
    Halide::Callable callable, int extent, bool outputComplex, std::vector<ScalarType> inputTypes,
    std::string targetName, SchedulerConfig scheduleConfig,
    std::optional<Halide::AutoSchedulerResults> autoScheduleResults)
    : callable_(std::move(callable)),
      extent_(extent),
      outputComplex_(outputComplex),
      inputTypes_(std::move(inputTypes)),
      targetName_(std::move(targetName)),
      scheduleConfig_(std::move(scheduleConfig)),
      autoScheduleResults_(std::move(autoScheduleResults)) {}

template <typename T>
Halide::Buffer<typename ToHalideType<T>::Type> CompiledPipeline<T>::run(
    const Halide::Buffer<typename ToHalideType<T>::Type>& input, bool copyToHost) {
  if (inputTypes_.empty()) {
    return run(std::vector<Halide::Buffer<typename ToHalideType<T>::Type>>{}, copyToHost);
  }
  if (inputTypes_.size() > 1) {
    throw std::runtime_error("CompiledPipeline: use run(inputs) for multi-input pipelines");
  }
  return run(std::vector<Halide::Buffer<typename ToHalideType<T>::Type>>{input}, copyToHost);
}

template <typename T>
Halide::Buffer<typename ToHalideType<T>::Type> CompiledPipeline<T>::run(
    const std::vector<Halide::Buffer<typename ToHalideType<T>::Type>>& inputs, bool copyToHost) {
  if (inputs.size() != inputTypes_.size()) {
    throw std::runtime_error("CompiledPipeline: input count mismatch");
  }
  for (size_t i = 0; i < inputs.size(); ++i) {
    if (!inputs[i].defined()) {
      throw std::runtime_error("CompiledPipeline: input buffer undefined");
    }
    bool const inputComplex = isComplexType(inputTypes_[i]);
    int const dims = inputs[i].dimensions();
    if (inputComplex) {
      if (dims != 2 || inputs[i].dim(0).extent() != 2) {
        throw std::runtime_error("Input buffer must be 2D with c=2 for complex input");
      }
    } else {
      if (dims != 1) {
        throw std::runtime_error("Input buffer must be 1D for real input");
      }
    }
  }

  // 根据信号类型确定输出 Buffer 维度：
  // 1. 复数信号采用 (c, x) 布局，其中 c=2 (实部与虚部)
  // 2. 实数信号采用 (x) 一维布局
  Halide::Buffer<typename ToHalideType<T>::Type> output(extent_);
  if (outputComplex_) {
    std::vector<int> const dims = {2, extent_};
    output = Halide::Buffer<typename ToHalideType<T>::Type>(dims);
  } else {
    output = Halide::Buffer<typename ToHalideType<T>::Type>(extent_);
  }

  Halide::JITUserContext const jitCtx;
  Halide::JITUserContext const* ctxPtr = &jitCtx;
  std::vector<const void*> argv;
  argv.reserve(inputs.size() + 2);
  argv.push_back(&ctxPtr);
  for (const auto& buf : inputs) {
    argv.push_back(buf.raw_buffer());
  }
  argv.push_back(output.raw_buffer());
  callable_.call_argv_fast(argv.size(), argv.data());
  if (copyToHost) {
    int const copyErr = output.copy_to_host();
    if (copyErr != halide_error_code_success) {
      throw std::runtime_error("CompiledPipeline: copy_to_host failed");
    }
  }
  return output;
}

// ============================================================================
// Executor::compile 实现 (AOT/JIT 预编译)
// ============================================================================

template <typename T>
CompiledPipeline<T> Executor::compile(const prism::dsl::Signal& signal, ExecMode mode) {
  SchedulerConfig config;
  config.kind = SchedulerKind::AUTO;
  return compile<T>(signal, mode, config);
}

namespace {
void setAutoschedulerEstimates(
    Halide::Func& result, bool outputComplex, int extent,
    std::map<const prism::dsl::detail::Node*, Halide::ImageParam>& inputParamStorage) {
  const auto& args = result.args();
  if (!args.empty()) {
    if (outputComplex && args.size() >= 2) {
      result.set_estimate(args[0], 0, 2);
      result.set_estimate(args[1], 0, extent);
    } else {
      result.set_estimate(args.back(), 0, extent);
    }
  }

  for (auto& entry : inputParamStorage) {
    const auto* node = entry.first;
    auto& param = entry.second;
    int const inputExtent = node ? static_cast<int>(node->shape.length) : extent;
    if (param.dimensions() >= 2) {
      param.dim(0).set_estimate(0, 2);
      param.dim(1).set_estimate(0, inputExtent);
    } else if (param.dimensions() == 1) {
      param.dim(0).set_estimate(0, inputExtent);
    }
  }
}
}  // namespace

template <typename T>
CompiledPipeline<T> Executor::compile(const prism::dsl::Signal& signal, ExecMode mode,
                                      const SchedulerConfig& schedule) {
  try {
    ensureOutputTypeMatch<T>(signal);
    const auto& shape = signal.shape();
    const int64_t extent = shape.length * shape.channels * shape.batch;
    if (extent <= 0) {
      return CompiledPipeline<T>();
    }

    // 1. 收集计算图中所有唯一的 INPUT 节点（按 DFS 顺序）
    // 用于为每个输入创建对应的 ImageParam
    std::vector<const prism::dsl::detail::Node*> inputNodes;
    std::set<const prism::dsl::detail::Node*> visited;
    std::function<void(const prism::dsl::Signal&)> collectInputs =
        [&](const prism::dsl::Signal& s) {
          const auto* n = s.node().get();
          if (!n || visited.count(n)) return;
          visited.insert(n);
          if (n->kind == prism::dsl::OpKind::INPUT) {
            inputNodes.push_back(n);
          }
          for (const auto& inp : n->inputs) {
            collectInputs(prism::dsl::Signal::fromNode(inp));
          }
        };
    collectInputs(signal);

    // 2. 准备输入参数 (ImageParam)
    // 编译模式下，我们不知道具体输入数据的地址，因此使用 Halide::ImageParam
    // 占位，对每个输入按自身类型决定维度：复数为 (2, N)，实数为 (N)
    std::map<const prism::dsl::detail::Node*, Halide::ImageParam> inputParamStorage;
    std::vector<Halide::Argument> allParams;
    std::vector<ScalarType> inputTypes;
    inputTypes.reserve(inputNodes.size());
    for (size_t i = 0; i < inputNodes.size(); ++i) {
      std::string const name = "input" + std::to_string(i);
      ScalarType const inputType = inputNodes[i]->outputType;
      Halide::Type const type = getElementType(inputType);
      int const inputDims = isComplexType(inputType) ? 2 : 1;
      inputParamStorage.emplace(inputNodes[i], Halide::ImageParam(type, inputDims, name));
      allParams.push_back(inputParamStorage.at(inputNodes[i]));
      inputTypes.push_back(inputType);
    }

    // 3. 目标设备选择
    // 根据用户偏好 mode (CPU/GPU/AUTO) 和数据规模来决策
    Halide::Target const target = getTargetForMode(mode, extent);
    bool const useGpu = target.has_feature(Halide::Target::Metal) ||
                        target.has_feature(Halide::Target::CUDA) ||
                        target.has_feature(Halide::Target::OpenCL);

    // 4. 构建 Halide 函数图
    OpContext<T> ctx;
    for (auto& [nodePtr, param] : inputParamStorage) {
      ctx.inputParams[nodePtr] = &param;
    }
    bool const outputComplex = isComplexType(signal.type());

    ctx.useGpu = useGpu;
    Halide::Func result = buildSignalFunc<T>(signal, ctx);

    if (outputComplex) {
      // Bind complex channel for correct mux bounds during compilation.
      result.bound(result.args()[0], 0, 2);
    }

    SchedulerConfig scheduleUsed = schedule;
    Halide::Pipeline pipeline(result);
    std::optional<Halide::AutoSchedulerResults> autoResults;

    if (!scheduleEnabled() || scheduleUsed.kind == SchedulerKind::NONE) {
      scheduleUsed.kind = SchedulerKind::NONE;
    } else if (scheduleUsed.kind == SchedulerKind::MANUAL) {
      throw std::runtime_error("Manual scheduler is not supported yet");
    } else if (scheduleUsed.kind == SchedulerKind::AUTO) {
      if (scheduleUsed.name.empty()) {
        scheduleUsed.name = useGpu ? "Anderson2021" : "Adams2019";
      }
      auto const& schedName = scheduleUsed.name;
      auto& schedExtra = scheduleUsed.extra;
      for (auto it = schedExtra.begin(); it != schedExtra.end();) {
        if (auto& [k, v] = *it; v.empty()) {
          it = schedExtra.erase(it);
        } else {
          ++it;
        }
      }

      const std::string pluginPath = autoschedulerPluginPath(schedName);
      if (!pluginPath.empty()) {
        (void)loadAutoschedulerPlugin(schedName);
      }
      setAutoschedulerEstimates(result, outputComplex, static_cast<int>(extent), inputParamStorage);

      Halide::AutoschedulerParams const params(schedName, schedExtra);
      try {
        autoResults = pipeline.apply_autoscheduler(target, params);
      } catch (const std::exception& e) {
        std::string hint;
        if (!pluginPath.empty()) {
          hint = " (autoscheduler plugin: " + pluginPath + ")";
        }
        throw std::runtime_error("Autoscheduler failed: " + schedName + hint +
                                 ". Detail: " + e.what());
      }
    }

    // 6. 编译为 Callable (JIT 编译)
    // 生成可重用的机器码/GPU Kernel，绑定到 inputParams
    Halide::Callable callable;
    try {
      callable = pipeline.compile_to_callable(allParams, target);
    } catch (const std::exception& e) {
      std::string msg = "Halide compile failed (target=" + target.to_string();
      if (scheduleUsed.kind == SchedulerKind::AUTO && !scheduleUsed.name.empty()) {
        msg += ", autoscheduler=" + scheduleUsed.name;
      }
      msg += ")";
      throw std::runtime_error(msg + ". Detail: " + e.what());
    }

    ExecMode const targetMode = useGpu ? ExecMode::GPU : ExecMode::CPU;
    std::string const targetName = formatTargetName(targetMode, &target);

    return CompiledPipeline<T>(std::move(callable), static_cast<int>(extent), outputComplex,
                               std::move(inputTypes), std::move(targetName),
                               std::move(scheduleUsed), std::move(autoResults));
  } catch (const std::exception& e) {
    throw std::runtime_error(std::string("Executor::compile failed: ") + e.what());
  }
}

// ============================================================================
// Executor::run 实现（JIT 模式）
// ============================================================================

template <typename T>
Halide::Buffer<typename ToHalideType<T>::Type> Executor::run(const prism::dsl::Signal& signal,
                                                             const SchedulerConfig& schedule) {
  return run<T>(signal, Halide::Buffer<typename ToHalideType<T>::Type>(), schedule);
}

template <typename T>
Halide::Buffer<typename ToHalideType<T>::Type> Executor::run(
    const prism::dsl::Signal& signal, const Halide::Buffer<typename ToHalideType<T>::Type>& input,
    const SchedulerConfig& schedule) {
  if (schedule.kind == SchedulerKind::MANUAL) {
    throw std::runtime_error("Manual scheduler is not supported yet");
  }
  ensureOutputTypeMatch<T>(signal);
  const auto& shape = signal.shape();
  const int64_t extent = shape.length * shape.channels * shape.batch;
  if (extent <= 0) {
    return Halide::Buffer<typename ToHalideType<T>::Type>();
  }
  if (extent > static_cast<int64_t>(std::numeric_limits<int>::max())) {
    throw std::runtime_error("Signal extent exceeds Halide int limit");
  }

  // 收集图中 INPUT 节点（DFS 顺序）
  std::vector<const prism::dsl::detail::Node*> inputNodes;
  std::set<const prism::dsl::detail::Node*> visited;
  std::function<void(const prism::dsl::Signal&)> collectInputs = [&](const prism::dsl::Signal& s) {
    const auto* n = s.node().get();
    if (!n || visited.count(n)) return;
    visited.insert(n);
    if (n->kind == prism::dsl::OpKind::INPUT) {
      inputNodes.push_back(n);
    }
    for (const auto& inp : n->inputs) {
      collectInputs(prism::dsl::Signal::fromNode(inp));
    }
  };
  collectInputs(signal);

  SchedulerConfig const scheduleUsed = makeJitSchedule(schedule);
  if (scheduleEnabled() && scheduleUsed.kind == SchedulerKind::AUTO) {
    auto compiled = compile<T>(signal, gExecutionMode, scheduleUsed);
    return compiled.run(input, true);
  }

  OpContext<T> ctx;
  if (input.defined() && !inputNodes.empty()) {
    // 单输入绑定到第一个 INPUT 节点
    ctx.inputBuffers[inputNodes[0]] = &input;
  }

  Halide::Target const target = getTargetForMode(gExecutionMode, extent);
  bool const useGpu = target.has_feature(Halide::Target::Metal) ||
                      target.has_feature(Halide::Target::CUDA) ||
                      target.has_feature(Halide::Target::OpenCL);
  bool const outputComplex = isComplexType(signal.type());

  ctx.useGpu = useGpu;
  Halide::Func result = buildSignalFunc<T>(signal, ctx);

  if (outputComplex) {
    // (2, extent)
    // Explicitly bound 'c' to [0, 2] so that unroll(c) strategies work
    result.bound(result.args()[0], 0, 2);
    return result.realize({2, static_cast<int>(extent)}, target);
  }
  return result.realize({static_cast<int>(extent)}, target);
}

// ============================================================================
// Executor::run
// ============================================================================

template <typename T>
Halide::Buffer<typename ToHalideType<T>::Type> Executor::run(
    const prism::dsl::Signal& signal,
    const std::vector<Halide::Buffer<typename ToHalideType<T>::Type>>& inputs,
    const SchedulerConfig& schedule) {
  if (schedule.kind == SchedulerKind::MANUAL) {
    throw std::runtime_error("Manual scheduler is not supported yet");
  }
  ensureOutputTypeMatch<T>(signal);
  const auto& shape = signal.shape();
  const int64_t extent = shape.length * shape.channels * shape.batch;
  if (extent <= 0) {
    return Halide::Buffer<typename ToHalideType<T>::Type>();
  }
  if (extent > static_cast<int64_t>(std::numeric_limits<int>::max())) {
    throw std::runtime_error("Signal extent exceeds Halide int limit");
  }

  // 收集图中 INPUT 节点（DFS 顺序）
  std::vector<const prism::dsl::detail::Node*> inputNodes;
  std::set<const prism::dsl::detail::Node*> visited;
  std::function<void(const prism::dsl::Signal&)> collectInputs = [&](const prism::dsl::Signal& s) {
    const auto* n = s.node().get();
    if (!n || visited.count(n)) return;
    visited.insert(n);
    if (n->kind == prism::dsl::OpKind::INPUT) {
      inputNodes.push_back(n);
    }
    for (const auto& inp : n->inputs) {
      collectInputs(prism::dsl::Signal::fromNode(inp));
    }
  };
  collectInputs(signal);

  SchedulerConfig const scheduleUsed = makeJitSchedule(schedule);
  if (scheduleEnabled() && scheduleUsed.kind == SchedulerKind::AUTO) {
    auto compiled = compile<T>(signal, gExecutionMode, scheduleUsed);
    return compiled.run(inputs, true);
  }

  OpContext<T> ctx;
  for (size_t i = 0; i < inputs.size() && i < inputNodes.size(); ++i) {
    if (inputs[i].defined()) {
      ctx.inputBuffers[inputNodes[i]] = &inputs[i];
    }
  }

  Halide::Target const target = getTargetForMode(gExecutionMode, extent);
  bool const useGpu = target.has_feature(Halide::Target::Metal) ||
                      target.has_feature(Halide::Target::CUDA) ||
                      target.has_feature(Halide::Target::OpenCL);
  bool const outputComplex = isComplexType(signal.type());

  ctx.useGpu = useGpu;
  Halide::Func result = buildSignalFunc<T>(signal, ctx);

  if (outputComplex) {
    result.bound(result.args()[0], 0, 2);
    return result.realize({2, static_cast<int>(extent)}, target);
  }
  return result.realize({static_cast<int>(extent)}, target);
}

// ============================================================================
// 显式模板实例化
// ============================================================================

/// @cond DOXYGEN_SKIP
template class CompiledPipeline<real32_t>;
template class CompiledPipeline<real64_t>;
template class CompiledPipeline<complex32_t>;
template class CompiledPipeline<complex64_t>;

template CompiledPipeline<real32_t> Executor::compile<real32_t>(const prism::dsl::Signal&,
                                                                ExecMode);
template CompiledPipeline<real64_t> Executor::compile<real64_t>(const prism::dsl::Signal&,
                                                                ExecMode);
template CompiledPipeline<complex32_t> Executor::compile<complex32_t>(const prism::dsl::Signal&,
                                                                      ExecMode);
template CompiledPipeline<complex64_t> Executor::compile<complex64_t>(const prism::dsl::Signal&,
                                                                      ExecMode);
template CompiledPipeline<real32_t> Executor::compile<real32_t>(const prism::dsl::Signal&, ExecMode,
                                                                const SchedulerConfig&);
template CompiledPipeline<real64_t> Executor::compile<real64_t>(const prism::dsl::Signal&, ExecMode,
                                                                const SchedulerConfig&);
template CompiledPipeline<complex32_t> Executor::compile<complex32_t>(const prism::dsl::Signal&,
                                                                      ExecMode,
                                                                      const SchedulerConfig&);
template CompiledPipeline<complex64_t> Executor::compile<complex64_t>(const prism::dsl::Signal&,
                                                                      ExecMode,
                                                                      const SchedulerConfig&);

template Halide::Buffer<real32_t> Executor::run<real32_t>(const prism::dsl::Signal&,
                                                          const SchedulerConfig&);
template Halide::Buffer<real32_t> Executor::run<real32_t>(const prism::dsl::Signal&,
                                                          const Halide::Buffer<real32_t>&,
                                                          const SchedulerConfig&);
template Halide::Buffer<real32_t> Executor::run<real32_t>(
    const prism::dsl::Signal&, const std::vector<Halide::Buffer<real32_t>>&,
    const SchedulerConfig&);
template Halide::Buffer<real64_t> Executor::run<real64_t>(const prism::dsl::Signal&,
                                                          const SchedulerConfig&);
template Halide::Buffer<real64_t> Executor::run<real64_t>(const prism::dsl::Signal&,
                                                          const Halide::Buffer<real64_t>&,
                                                          const SchedulerConfig&);
template Halide::Buffer<real64_t> Executor::run<real64_t>(
    const prism::dsl::Signal&, const std::vector<Halide::Buffer<real64_t>>&,
    const SchedulerConfig&);
template Halide::Buffer<real32_t> Executor::run<complex32_t>(const prism::dsl::Signal&,
                                                             const SchedulerConfig&);
template Halide::Buffer<real32_t> Executor::run<complex32_t>(const prism::dsl::Signal&,
                                                             const Halide::Buffer<real32_t>&,
                                                             const SchedulerConfig&);
template Halide::Buffer<real32_t> Executor::run<complex32_t>(
    const prism::dsl::Signal&, const std::vector<Halide::Buffer<real32_t>>&,
    const SchedulerConfig&);
template Halide::Buffer<real64_t> Executor::run<complex64_t>(const prism::dsl::Signal&,
                                                             const SchedulerConfig&);
template Halide::Buffer<real64_t> Executor::run<complex64_t>(const prism::dsl::Signal&,
                                                             const Halide::Buffer<real64_t>&,
                                                             const SchedulerConfig&);
template Halide::Buffer<real64_t> Executor::run<complex64_t>(
    const prism::dsl::Signal&, const std::vector<Halide::Buffer<real64_t>>&,
    const SchedulerConfig&);
/// @endcond

/// @}

}  // namespace prism::runtime
