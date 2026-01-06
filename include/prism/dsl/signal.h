/**
 * @file signal.h
 * @ingroup dsl
 * @brief Signal 计算图节点与算子类型定义
 *
 * DSL 层通过 `Signal` 类以 **惰性 (Lazy)** 方式描述信号处理流程
 * 每个 `Signal` 对象实际上是一个轻量级句柄，指向内部的图节点
 * (`prism::detail::Node`) 调用算子（如 `Add`,
 * `FFT`）不会立即执行计算，而是构建计算图，最终由 `Executor` 编译并运行
 */

#ifndef PRISM_DSL_SIGNAL_H
#define PRISM_DSL_SIGNAL_H

#include <cstddef>
#include <memory>
#include <vector>

#include "prism/types.h"

namespace prism::dsl {

/// @addtogroup dsl
/// @{

/**
 * @brief DSL 支持的算子种类枚举 (Operator Kinds)
 *
 * 定义了 PRISM DSL 能够表达的所有计算操作，涵盖基础代数运算与高级信号处理算法
 *
 * @note 新增算子流程：
 * 1. 在此枚举中添加新的 OpKind
 * 2. 在 @c prism::detail::Node 中确认是否需要扩展参数字段
 * 3. 在 Runtime 层的 `HandlerRegistry` 中注册对应的计算实现 (Handler)
 */
enum class OpKind : std::uint8_t {
  INPUT,     ///< 占位符 (Placeholder)：计算图输入端口，无计算逻辑
  CONSTANT,  ///< 常量源 (Constant)：生成全相同数值的信号 $y[n] = c$
  ADD,       ///< 逐元素加法 (Element-wise Add): $y[n] = a[n] + b[n]$
  SUB,       ///< 逐元素减法 (Element-wise Sub): $y[n] = a[n] - b[n]$
  MUL,       ///< 逐元素乘法 (Element-wise Mul): $y[n] = a[n] \times b[n]$
  DIV,       ///< 逐元素除法 (Element-wise Div): $y[n] = a[n] / b[n]$
  SCALE,     ///< 标量缩放 (Scalar Scale): $y[n] = x[n] \times \alpha$
  NEG,       ///< 取负 (Negation): $y[n] = -x[n]$
  CONJ,      ///< 复共轭 (Conjugate): $y[n] = x[n]^*$
  ABS,       ///< 取模 (Absolute/Magnitude): $y[n] = |x[n]|$
  CONVOLVE,  ///< 卷积 (Convolution): $y[n] = (x * h)[n] = \sum_k x[k] h[n-k]$
  KRON,      ///< 克罗内克积 (Kronecker Product): 常用于扩频序列生成
  // Filter
  FIR,             ///< 有限脉冲响应滤波器 (FIR Filter)
  MOVING_AVERAGE,  ///< 滑动平均滤波 (Moving Average)
  MEDIAN,          ///< 中值滤波 (Median Filter)：非线性去噪，保留边缘
  // Modem
  MIXER,      ///< 数字混频 (Mixer/NCO): $y[n] = x[n] \cdot e^{-j(\omega n + \phi)}$
  QAM_MAP,    ///< QAM 映射 (Mapper): 比特流 -> 星座点符号
  QAM_DEMAP,  ///< QAM 解映射 (Demapper/LLR): 符号 -> 对数似然比
  PSK_MAP,    ///< PSK 映射 (Mapper)
  PSK_DEMAP,  ///< PSK 解映射 (Demapper)
  CPLX_PACK,  ///< 复数打包 (Complex Pack): 将两路实信号合并为一路复信号 $I +
              ///< jQ$
  REAL,       ///< 提取实部 (Real Component): $y[n] = \text{Re}(x[n])$
  IMAG,       ///< 提取虚部 (Imag Component): $y[n] = \text{Im}(x[n])$
  UPSAMPLE,   ///< 上采样 (Upsample): 在样本间插入零值
  DOWNSAMPLE  ///< 下采样 (Downsample): 每隔 $M$ 个样本取一个值
};

namespace detail {

/**
 * @brief Signal 内部图节点 (Internal AST Node)
 *
 * 存储算子的元数据，是 DSL 前端与 Runtime 后端交互的核心数据结构
 * 通常由 DSL 工厂函数自动构建并以 `std::shared_ptr` 管理，用户不可见
 */
struct Node {
  OpKind kind = OpKind::INPUT;                ///< 算子具体类型
  std::vector<std::shared_ptr<Node>> inputs;  ///< 上游依赖节点（入度边）
  Shape shape;                                ///< 计算结果的 Tensor 形状
  ScalarType inputType = ScalarType::F32;     ///< 输入数据类型
  ScalarType outputType = ScalarType::F32;    ///< 输出数据类型

  // 通用参数区域 (Union-like implementation)
  OpParam param;                         ///< 算子参数（标量、系数向量、配置项等）
  BndryMode boundary = BndryMode::ZERO;  ///< 边界处理模式（仅用于卷积/滑窗类算子）
};

}  // namespace detail

/**
 * @brief DSL 信号句柄 (Signal Handle)
 *
 * 代表计算图中的一个逻辑信号（Edge/Node Wrapper）
 * 采用 Pimpl (Pointer to Implementation) 惯用法变体（共享 `Node` 指针），
 * 支持轻量级拷贝与传值，表现类似于智能指针
 *
 * @note **不可变性 (Immutability)**：Signal
 * 对象一旦创建，其代表的计算逻辑不可更改 所有算子操作都会返回一个新的 Signal
 * 对象，从而构建出有向无环图 (DAG)
 */
class Signal {
 public:
  Signal() = default;

  /**
   * @brief 创建一个图输入节点 (Input Placeholder)
   *
   * 定义计算图的输入端口，运行时需要向其馈送数据
   *
   * @param length 信号的时间维度长度 (Samples)
   * @param type 信号的数据类型 (默认为 F32)
   * @return 新创建的 Signal 输入句柄
   */
  static Signal input(size_t length, ScalarType type = ScalarType::F32);

  /**
   * @brief 创建一个常量节点 (Constant)
   *
   * 生成一个长度为 `length` 的信号，所有样本值均等于 `value`
   * 常用于偏置 (Bias)、固定增益控制等场景
   *
   * @tparam T C++ 标量类型 (自动推导)
   * @param value 常量数值
   * @param length 信号长度
   * @return 新创建的常量 Signal
   */
  template <typename T>
  static Signal constant(const T& value, size_t length) {
    auto node = std::make_shared<detail::Node>();
    auto type = getScalarType<T>();
    node->kind = OpKind::CONSTANT;
    node->param = value;
    node->shape.length = length;
    node->inputType = type;
    node->outputType = type;
    return Signal(node);
  }

  /**
   * @brief 从底层 Node 指针封装 Signal
   *
   * @warning 这是一个内部底层 API，仅限于开发自定义 DSL 扩展时使用
   * @param node 已经构造好的 Node 智能指针
   */
  static Signal fromNode(const std::shared_ptr<detail::Node>& node);

  /**
   * @brief 获取信号的形状信息
   * @return Shape 结构体引用
   */
  [[nodiscard]] const Shape& shape() const { return node_->shape; }

  /**
   * @brief 获取信号的数据类型
   * @return ScalarType 枚举值
   */
  [[nodiscard]] ScalarType type() const { return node_->outputType; }

  /**
   * @brief 访问底层 Node 指针
   *
   * 主要供 Runtime 的 Compiler/Executor 遍历计算图时使用
   */
  [[nodiscard]] const std::shared_ptr<detail::Node>& node() const { return node_; }

  /**
   * @brief 内部构造函数
   * @param node 托管的 Node 指针
   */
  explicit Signal(std::shared_ptr<detail::Node> node);

 private:
  std::shared_ptr<detail::Node> node_;
};

/// @}

}  // namespace prism::dsl

#endif  // PRISM_DSL_SIGNAL_H
