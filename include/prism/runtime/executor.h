/**
 * @file executor.h
 * @ingroup runtime
 * @brief 运行时执行引擎
 *
 * 提供 DSL 计算图的编译与执行入口，支持 Halide JIT 即时编译和
 * 面向性能优化的预编译（CompiledPipeline）模式
 * 负责管理计算设备（CPU/GPU）的选择与调度策略配置
 */

#ifndef PRISM_RUNTIME_EXECUTOR_H
#define PRISM_RUNTIME_EXECUTOR_H

#include <Halide.h>

#include <optional>
#include <string>
#include <vector>

#include "prism/dsl/signal.h"
#include "prism/runtime/schedule_config.h"
#include "prism/types.h"

namespace prism::runtime {

/// @addtogroup runtime
/// @{

/**
 * @brief 执行设备模式
 *
 * 控制计算图在何种硬件设备上执行
 */
enum class ExecMode : std::uint8_t {
  AUTO,  ///< 自动选择（启发式策略：若数据量 >= 1K 且有 GPU 则优先使用
         ///< GPU，否则使用 CPU）
  GPU,   ///< 强制使用 GPU 执行（若无可用 GPU 可能回退或报错）
  CPU    ///< 强制使用 CPU 执行
};

/**
 * @brief 预编译管线
 *
 * 表示一个已编译但尚未执行的 Halide 管道
 * 相比直接运行 `Executor::run`，此类将编译阶段与执行阶段分离，
 * 适用于只需编译一次但通过不同输入数据多次执行的场景（"Compile-Once-Run-Many"），
 * 可显著减少重复编译带来的 JIT 开销
 *
 * @tparam T 标量数据类型 (e.g., prism::real32_t)
 *
 * @par 使用示例
 * @code
 * auto x = Signal::input(1024);
 * auto y = Scale(x, 2.0) | Abs();
 *
 * // 1. 预编译阶段（高开销）
 * auto pipeline = Executor::compile<real32_t>(y, ExecMode::GPU);
 *
 * // 2. 执行阶段（低开销）
 * Halide::Buffer<float> inputBuf(1024);
 * // ... 填充 inputBuf ...
 *
 * for (int i = 0; i < 1000; ++i) {
 *     // 零编译开销，直接复用已编译的可执行代码
 *     auto result = pipeline.run(inputBuf);
 * }
 * @endcode
 */
template <typename T>
class CompiledPipeline {
 public:
  CompiledPipeline() = default;

  /**
   * @brief 构造函数（内部使用）
   * @param callable Halide 可调用对象
   * @param extent 输出信号长度
   * @param outputComplex 输出是否为复数
   * @param inputTypes 输入信号的标量类型列表
   * @param targetName 目标设备名称
   * @param scheduleConfig 调度配置
   * @param autoScheduleResults 自动调度结果（可选）
   */
  CompiledPipeline(Halide::Callable callable, int extent, bool outputComplex,
                   std::vector<ScalarType> inputTypes, std::string targetName = "Unknown",
                   SchedulerConfig scheduleConfig = SchedulerConfig{},
                   std::optional<Halide::AutoSchedulerResults> autoScheduleResults = std::nullopt);

  /**
   * @brief 执行管线（单输入）
   *
   * @param input 输入 Buffer数据，如果 DSL 图中没有 INPUT 节点，此参数将被忽略
   *              若有输入节点但未提供 Buffer，将使用默认值或抛出异常
   * @param copyToHost 是否将结果立即拷贝回 Host 内存
   *                   - `false`: 结果可能仍驻留在 GPU 显存中（DeviceDirty）
   *                   - `true`: 强制同步并拷贝数据到 CPU 内存
   * @return Halide::Buffer<T> 输出 Buffer，长度与 DSL 信号定义的 extent 一致
   *
   * @note 输入 Buffer 的类型必须与编译时指定的类型匹配
   */
  Halide::Buffer<typename ToHalideType<T>::Type> run(
      const Halide::Buffer<typename ToHalideType<T>::Type>& input, bool copyToHost = false);

  /**
   * @brief 执行管线（多输入）
   *
   * @param inputs 输入 Buffer 列表
   *               顺序必须与 `Signal::input()` 在 DSL 中的声明顺序（DFS
   * 遍历顺序）一致
   * @param copyToHost 是否将结果立即拷贝回 Host 内存
   * @return Halide::Buffer<T> 输出 Buffer
   */
  Halide::Buffer<typename ToHalideType<T>::Type> run(
      const std::vector<Halide::Buffer<typename ToHalideType<T>::Type>>& inputs,
      bool copyToHost = false);

  /**
   * @brief 检查管线是否有效
   * @return 若已成功编译且 extent > 0，返回 true
   */
  [[nodiscard]] bool valid() const { return extent_ > 0; }

  /**
   * @brief 获取编译目标名称
   * @return 如 "host-cuda", "host-metal", "x86-64-osx-avx2" 等
   */
  [[nodiscard]] std::string targetName() const { return targetName_; }

  /**
   * @brief 获取调度配置
   * @return 编译时使用的完整调度配置
   */
  [[nodiscard]] const SchedulerConfig& scheduleConfig() const { return scheduleConfig_; }

  /**
   * @brief 获取自动调度结果
   * @return 若使用了 Autoscheduler，返回生成的调度文件与源码信息
   */
  [[nodiscard]] const std::optional<Halide::AutoSchedulerResults>& autoScheduleResults() const {
    return autoScheduleResults_;
  }

 private:
  Halide::Callable callable_;
  int extent_ = 0;
  bool outputComplex_ = false;
  std::vector<ScalarType> inputTypes_;
  std::string targetName_ = "Unknown";
  SchedulerConfig scheduleConfig_{};
  std::optional<Halide::AutoSchedulerResults> autoScheduleResults_;
};

/**
 * @brief 全局执行器
 *
 * 负责 DSL 图的编译、调度与执行管理
 */
class Executor {
 public:
  /**
   * @brief 设置全局默认执行模式
   * @param mode 目标模式 (AUTO/CPU/GPU)
   */
  static void setMode(ExecMode mode);

  /**
   * @brief 获取当前全局执行模式
   * @return 当前模式
   */
  static ExecMode getMode();

  /**
   * @brief 编译 DSL 图（自动调度）
   *
   * @tparam T 计算精度类型 (real32_t/real64_t/...)
   * @param signal 要计算的 DSL 信号节点
   * @param mode 执行模式（默认 AUTO）
   * @return CompiledPipeline<T> 已编译的可重用管线对象
   *
   * @note 默认会自动选择合适的 Autoscheduler（CPU 用 Adams2019，GPU 用
   * Anderson2021）
   */
  template <typename T>
  static CompiledPipeline<T> compile(const prism::dsl::Signal& signal,
                                     ExecMode mode = ExecMode::AUTO);

  /**
   * @brief 编译 DSL 图（指定调度配置）
   *
   * @tparam T 计算精度类型
   * @param signal DSL 信号节点
   * @param mode 执行模式
   * @param schedule 自定义调度配置
   * @return CompiledPipeline<T> 已编译的可重用管线对象
   */
  template <typename T>
  static CompiledPipeline<T> compile(const prism::dsl::Signal& signal, ExecMode mode,
                                     const SchedulerConfig& schedule);

  /**
   * @brief 立即执行 DSL 图（JIT 编译 + 运行）
   *
   * 适用于仅运行一次的场景，内部会先编译再执行
   *
   * @tparam T 计算精度类型
   * @param signal DSL 信号节点
   * @param schedule 调度配置（可选）
   * @return Halide::Buffer<T> 计算结果
   *
   * @note 默认使用 Mullapudi2016 简易自动调度
   */
  template <typename T>
  static Halide::Buffer<typename ToHalideType<T>::Type> run(
      const prism::dsl::Signal& signal, const SchedulerConfig& schedule = SchedulerConfig{});

  /**
   * @brief 立即执行 DSL 图（单输入）
   *
   * @tparam T 计算精度类型
   * @param signal DSL 信号节点
   * @param input 输入数据 Buffer
   * @param schedule 调度配置（可选）
   * @return Halide::Buffer<T> 计算结果
   */
  template <typename T>
  static Halide::Buffer<typename ToHalideType<T>::Type> run(
      const prism::dsl::Signal& signal, const Halide::Buffer<typename ToHalideType<T>::Type>& input,
      const SchedulerConfig& schedule = SchedulerConfig{});

  /**
   * @brief 立即执行 DSL 图（多输入）
   *
   * @tparam T 计算精度类型
   * @param signal DSL 信号节点
   * @param inputs 输入数据 Buffer 列表
   * @param schedule 调度配置（可选）
   * @return Halide::Buffer<T> 计算结果
   *
   * @par 示例
   * @code
   * auto a = Signal::input(1024);
   * auto b = Signal::input(1024);
   * auto y = a + b;
   *
   * Halide::Buffer<float> bufA(1024), bufB(1024);
   * // ... 填充数据 ...
   * auto result = Executor::run<float>(y, {bufA, bufB});
   * @endcode
   */
  template <typename T>
  static Halide::Buffer<typename ToHalideType<T>::Type> run(
      const prism::dsl::Signal& signal,
      const std::vector<Halide::Buffer<typename ToHalideType<T>::Type>>& inputs,
      const SchedulerConfig& schedule = SchedulerConfig{});

  /**
   * @brief 获取当前生效的 Halide Target 名称
   * @return 目标字符串 (e.g., "x86-64-osx-avx2-f16c-fma-sse41")
   */
  static std::string getCurrentTargetName();
};

/// @cond DOXYGEN_SKIP
// 显式模板实例化声明（供编译期优化使用，文档可忽略）
extern template class CompiledPipeline<real32_t>;
extern template class CompiledPipeline<real64_t>;
extern template class CompiledPipeline<complex32_t>;
extern template class CompiledPipeline<complex64_t>;

extern template CompiledPipeline<real32_t> Executor::compile<real32_t>(const prism::dsl::Signal&,
                                                                       ExecMode);
extern template CompiledPipeline<real64_t> Executor::compile<real64_t>(const prism::dsl::Signal&,
                                                                       ExecMode);
extern template CompiledPipeline<complex32_t> Executor::compile<complex32_t>(
    const prism::dsl::Signal&, ExecMode);
extern template CompiledPipeline<complex64_t> Executor::compile<complex64_t>(
    const prism::dsl::Signal&, ExecMode);
extern template CompiledPipeline<real32_t> Executor::compile<real32_t>(const prism::dsl::Signal&,
                                                                       ExecMode,
                                                                       const SchedulerConfig&);
extern template CompiledPipeline<real64_t> Executor::compile<real64_t>(const prism::dsl::Signal&,
                                                                       ExecMode,
                                                                       const SchedulerConfig&);
extern template CompiledPipeline<complex32_t> Executor::compile<complex32_t>(
    const prism::dsl::Signal&, ExecMode, const SchedulerConfig&);
extern template CompiledPipeline<complex64_t> Executor::compile<complex64_t>(
    const prism::dsl::Signal&, ExecMode, const SchedulerConfig&);

extern template Halide::Buffer<real32_t> Executor::run<real32_t>(const prism::dsl::Signal&,
                                                                 const SchedulerConfig&);
extern template Halide::Buffer<real32_t> Executor::run<real32_t>(const prism::dsl::Signal&,
                                                                 const Halide::Buffer<real32_t>&,
                                                                 const SchedulerConfig&);
extern template Halide::Buffer<real32_t> Executor::run<real32_t>(
    const prism::dsl::Signal&, const std::vector<Halide::Buffer<real32_t>>&,
    const SchedulerConfig&);
extern template Halide::Buffer<real64_t> Executor::run<real64_t>(const prism::dsl::Signal&,
                                                                 const SchedulerConfig&);
extern template Halide::Buffer<real64_t> Executor::run<real64_t>(const prism::dsl::Signal&,
                                                                 const Halide::Buffer<real64_t>&,
                                                                 const SchedulerConfig&);
extern template Halide::Buffer<real64_t> Executor::run<real64_t>(
    const prism::dsl::Signal&, const std::vector<Halide::Buffer<real64_t>>&,
    const SchedulerConfig&);
extern template Halide::Buffer<real32_t> Executor::run<complex32_t>(const prism::dsl::Signal&,
                                                                    const SchedulerConfig&);
extern template Halide::Buffer<real32_t> Executor::run<complex32_t>(const prism::dsl::Signal&,
                                                                    const Halide::Buffer<real32_t>&,
                                                                    const SchedulerConfig&);
extern template Halide::Buffer<real32_t> Executor::run<complex32_t>(
    const prism::dsl::Signal&, const std::vector<Halide::Buffer<real32_t>>&,
    const SchedulerConfig&);
extern template Halide::Buffer<real64_t> Executor::run<complex64_t>(const prism::dsl::Signal&,
                                                                    const SchedulerConfig&);
extern template Halide::Buffer<real64_t> Executor::run<complex64_t>(const prism::dsl::Signal&,
                                                                    const Halide::Buffer<real64_t>&,
                                                                    const SchedulerConfig&);
extern template Halide::Buffer<real64_t> Executor::run<complex64_t>(
    const prism::dsl::Signal&, const std::vector<Halide::Buffer<real64_t>>&,
    const SchedulerConfig&);
/// @endcond

/// @}

}  // namespace prism::runtime

#endif  // PRISM_RUNTIME_EXECUTOR_H
