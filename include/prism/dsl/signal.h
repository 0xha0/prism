/**
 * @file signal.h
 * @ingroup dsl
 * @brief Signal 计算图节点与操作种类
 *
 * DSL 层通过 Signal 以惰性方式描述信号处理流程，每个节点记录算子类型、
 * 输入依赖、数据形状与参数，供 Runtime 构建 Halide 计算图。
 */

#ifndef PRISM_DSL_SIGNAL_H
#define PRISM_DSL_SIGNAL_H

#include <memory>
#include <vector>

#include "prism/types.h"

namespace prism::dsl {

/// @addtogroup dsl
/// @{

/**
 * @brief DSL 支持的算子种类
 *
 * 新增算子时需要同时在 Runtime 注册对应 Handler。
 */
enum class OpKind : std::uint8_t {
  INPUT,     ///< 输入端口（占位，不做计算）
  CONSTANT,  ///< 常量源
  ADD,
  SUB,
  MUL,
  DIV,       ///< 基本代数算子
  SCALE,     ///< 缩放（乘以标量）
  ABS,       ///< 绝对值
  CONVOLVE,  ///< 通用卷积
  KRON,      ///< 克罗内克积（扩频）
  // Filter
  FIR,             ///< FIR 滤波器
  IIR,             ///< IIR 滤波器
  MOVING_AVERAGE,  ///< 移动平均
  MEDIAN,          ///< 中值滤波
  // Modem
  MIXER,      ///< 混频器
  QAM_MAP,    ///< QAM 映射
  QAM_DEMAP,  ///< QAM 解映射
  PSK_MAP,    ///< PSK 映射
  PSK_DEMAP,  ///< PSK 解映射
  IQ_PACK,    ///< I/Q 合并为交织序列
  IQ_I,       ///< 交织序列提取 I
  IQ_Q,       ///< 交织序列提取 Q
  UPSAMPLE,   ///< 上采样（插零）
  DOWNSAMPLE  ///< 下采样（抽取）
};

namespace detail {

/**
 * @brief Signal 内部节点描述
 *
 * 保存算子类型、输入依赖、参数与元信息。通常由 DSL 工厂函数构建。
 */
struct Node {
  OpKind kind = OpKind::INPUT;                ///< 算子类型
  std::vector<std::shared_ptr<Node>> inputs;  ///< 输入节点列表
  Shape shape;                                ///< 输出形状
  ScalarType type = ScalarType::F32;          ///< 输出标量类型

  // 通用参数区域
  real64_t scalar = 0.0;                 ///< 标量参数（缩放系数/窗口等）
  real64_t freq = 0.0;                   ///< 频率相关参数（Hz）
  real64_t sampleRate = 0.0;             ///< 采样率（Hz）
  int modOrder = 0;                      ///< 调制阶数（QAM/PSK）
  int64_t step = 0;                      ///< 采样步长（上/下采样因子）
  int64_t offset = 0;                    ///< 采样偏移（下采样起点）
  std::vector<real32_t> taps;            ///< FIR/IIR b 系数 (float)
  std::vector<real32_t> tapsA;           ///< IIR a 系数 (float)
  std::vector<real64_t> taps64;          ///< FIR/IIR b 系数 (double)
  std::vector<real64_t> tapsA64;         ///< IIR a 系数 (double)
  BndryMode boundary = BndryMode::ZERO;  ///< 边界策略（滤波/滑窗）
};

}  // namespace detail

/**
 * @brief DSL 信号句柄
 *
 * 以共享指针持有内部节点，支持不可变链式构建，线程安全地描述计算图。
 */
class Signal {
 public:
  Signal() = default;

  /**
   * @brief 构造输入信号节点
   * @param length 样本长度
   * @param type 标量类型（默认单精度实数）
   * @return 对应的 Signal 句柄
   */
  static Signal input(int64_t length, ScalarType type = ScalarType::F32);

  /**
   * @brief 构造常量信号节点
   * @param value 常量值
   * @param length 重复次数（常量长度）
   * @param type 标量类型
   */
  static Signal constant(real64_t value, int64_t length,
                         ScalarType type = ScalarType::F32);

  /**
   * @brief 从已有节点封装 Signal
   * @param node 已构建的节点指针
   */
  static Signal fromNode(const std::shared_ptr<detail::Node>& node);

  /**
   * @brief 获取输出形状
   */
  [[nodiscard]] const Shape& shape() const { return node_->shape; }

  /**
   * @brief 获取标量类型
   */
  [[nodiscard]] ScalarType type() const { return node_->type; }

  /**
   * @brief 访问底层节点指针（Runtime 与 Handler 使用）
   */
  [[nodiscard]] const std::shared_ptr<detail::Node>& node() const {
    return node_;
  }

  /**
   * @brief 从节点构造 Signal（用于算子内部）
   * @param node 计算图节点
   */
  explicit Signal(std::shared_ptr<detail::Node> node);

 private:
  std::shared_ptr<detail::Node> node_;
};

/// @}

}  // namespace prism::dsl

#endif  // PRISM_DSL_SIGNAL_H
