/**
 * @file executor.h
 * @ingroup runtime
 * @brief 运行时执行入口
 *
 * 将 DSL 构建的计算图通过 Halide JIT 或预编译模式执行。
 * 提供 CPU/GPU 执行模式控制与零开销重复调用。
 */

#ifndef PRISM_RUNTIME_EXECUTOR_H
#define PRISM_RUNTIME_EXECUTOR_H

#include <Halide.h>

#include "prism/dsl/signal.h"
#include "prism/runtime/schedule_config.h"

namespace prism::runtime {

/// @addtogroup runtime
/// @{

/** @brief 执行模式 */
enum class ExecMode : std::uint8_t {
  AUTO,  ///< 自动选择（数据量 >= 1K 用 GPU）
  GPU,   ///< 强制 GPU
  CPU    ///< 强制 CPU
};

/**
 * @brief 预编译的管线
 *
 * 编译一次，多次执行，避免 JIT 开销。
 *
 * @par 使用示例
 * @code
 * auto x = Signal::input(1024);
 * auto y = Scale(x, 2.0) | Abs();
 *
 * // 预编译
 * auto pipeline = Executor::compile<real32_t>(y, ExecMode::GPU);
 *
 * // 多次执行（零编译开销）
 * for (int i = 0; i < 1000; ++i) {
 *     auto result = pipeline.run(inputBuf);
 * }
 * @endcode
 */
template <typename T>
class CompiledPipeline {
 public:
  CompiledPipeline() = default;
  CompiledPipeline(Halide::Callable callable, int extent);

  /**
   * @brief 执行预编译的管线
   * @param input 输入 Buffer（未定义则使用常量/内部源）
   * @return 输出 Buffer（长度等于 Signal extent）
   * @tparam T 标量类型（real32_t/real64_t）
   */
  Halide::Buffer<typename ToHalideType<T>::Type> run(
      const Halide::Buffer<typename ToHalideType<T>::Type>& input);

  /**
   * @brief 检查是否有效
   * @return extent 大于 0 则有效
   */
  [[nodiscard]] bool valid() const { return extent_ > 0; }

 private:
  Halide::Callable callable_;
  int extent_ = 0;
};

/**
 * @brief 计算图执行器
 *
 * 负责在不同执行模式下构建 Halide Pipeline，并提供 JIT/预编译两种调用方式。
 */
class Executor {
 public:
  /**
   * @brief 设置执行模式
   * @param mode @ref ExecMode
   */
  static void setMode(ExecMode mode);

  /**
   * @brief 获取当前执行模式
   * @return 当前模式
   */
  static ExecMode getMode();

  /**
   * @brief 预编译计算图（推荐用于重复执行的场景）
   * @param signal DSL 计算图
   * @param mode 执行模式（默认自动）
   * @tparam T 标量类型（real32_t/real64_t）
   */
  template <typename T>
  static CompiledPipeline<T> compile(const prism::dsl::Signal& signal,
                                     ExecMode mode = ExecMode::AUTO);

  /**
   * @brief 执行计算图（JIT 模式，每次编译）
   * @param signal DSL 计算图
   * @tparam T 标量类型（real32_t/real64_t）
   */
  template <typename T>
  static Halide::Buffer<typename ToHalideType<T>::Type> run(
      const prism::dsl::Signal& signal);

  /**
   * @brief 执行计算图（带外部输入）
   * @param signal DSL 计算图
   * @param input 输入 Buffer（长度需与计算图匹配）
   * @tparam T 标量类型（real32_t/real64_t）
   */
  template <typename T>
  static Halide::Buffer<typename ToHalideType<T>::Type> run(
      const prism::dsl::Signal& signal,
      const Halide::Buffer<typename ToHalideType<T>::Type>& input);

  /**
   * @brief 设置全局调度配置
   *
   * 配置 GPU tile 大小、CPU 向量化/并行等调度参数。
   * 传入默认构造的 ScheduleConfig 可恢复自动模式。
   *
   * @param config 调度配置
   *
   * @par 使用示例
   * @code
   * ScheduleConfig config;
   * config.gpuTileSize = 512;   // 自定义 GPU tile
   * config.cpuVectorWidth = 16; // AVX-512
   * Executor::setScheduleConfig(config);
   *
   * // 恢复自动模式
   * Executor::setScheduleConfig(ScheduleConfig{});
   * @endcode
   */
  static void setScheduleConfig(const ScheduleConfig& config);

  /**
   * @brief 获取当前调度配置
   * @return 当前调度配置的常量引用
   */
  static const ScheduleConfig& getScheduleConfig();
};

/// @cond DOXYGEN_SKIP
// 显式模板实例化声明（供编译期优化使用，文档可忽略）
extern template class CompiledPipeline<real32_t>;
extern template class CompiledPipeline<real64_t>;

extern template CompiledPipeline<real32_t> Executor::compile<real32_t>(
    const prism::dsl::Signal&, ExecMode);
extern template CompiledPipeline<real64_t> Executor::compile<real64_t>(
    const prism::dsl::Signal&, ExecMode);

extern template Halide::Buffer<real32_t> Executor::run<real32_t>(
    const prism::dsl::Signal&);
extern template Halide::Buffer<real32_t> Executor::run<real32_t>(
    const prism::dsl::Signal&, const Halide::Buffer<real32_t>&);
extern template Halide::Buffer<real64_t> Executor::run<real64_t>(
    const prism::dsl::Signal&);
extern template Halide::Buffer<real64_t> Executor::run<real64_t>(
    const prism::dsl::Signal&, const Halide::Buffer<real64_t>&);
/// @endcond

/// @}

}  // namespace prism::runtime

#endif  // PRISM_RUNTIME_EXECUTOR_H
