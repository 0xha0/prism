/**
 * @file filter.cpp
 * @ingroup dsl
 * @brief 滤波算子实现
 */

#include "prism/dsl/filter.h"

#include <memory>
#include <vector>

#include "prism/dsl/signal.h"
#include "prism/types.h"

namespace prism::dsl::filter {

// ============================================================================
// 核心滤波器
// ============================================================================

/**
 * @brief 构建单精度 FIR 节点
 * @param x 输入信号
 * @param taps b 系数
 * @param mode 边界策略
 */
Signal fir(const Signal& x, const std::vector<real32_t>& taps, BndryMode mode) {
  auto node = std::make_shared<detail::Node>();
  node->kind = OpKind::FIR;
  node->inputs = {x.node()};
  node->shape = x.shape();
  node->type = x.type();
  node->taps = taps;
  node->boundary = mode;
  return Signal(node);
}

/** @brief 构建双精度 FIR 节点 */
Signal fir(const Signal& x, const std::vector<real64_t>& taps, BndryMode mode) {
  auto node = std::make_shared<detail::Node>();
  node->kind = OpKind::FIR;
  node->inputs = {x.node()};
  node->shape = x.shape();
  node->type = x.type();
  node->taps64 = taps;
  node->boundary = mode;
  return Signal(node);
}

/**
 * @brief 构建单精度 IIR 节点
 * @param b 前向系数
 * @param a 反馈系数（a[0] 需为 1）
 * @param mode 边界策略
 */
Signal iir(const Signal& x, const std::vector<real32_t>& b,
           const std::vector<real32_t>& a, BndryMode mode) {
  auto node = std::make_shared<detail::Node>();
  node->kind = OpKind::IIR;
  node->inputs = {x.node()};
  node->shape = x.shape();
  node->type = x.type();
  node->taps = b;
  node->tapsA = a;
  node->boundary = mode;
  return Signal(node);
}

/** @brief 构建双精度 IIR 节点 */
Signal iir(const Signal& x, const std::vector<real64_t>& b,
           const std::vector<real64_t>& a, BndryMode mode) {
  auto node = std::make_shared<detail::Node>();
  node->kind = OpKind::IIR;
  node->inputs = {x.node()};
  node->shape = x.shape();
  node->type = x.type();
  node->taps64 = b;
  node->tapsA64 = a;
  node->boundary = mode;
  return Signal(node);
}

// ============================================================================
// 滑动窗口滤波器
// ============================================================================

/**
 * @brief 移动平均（窗口长度存入 scalar）
 * @param window 窗口长度（样本数）
 * @param mode 边界策略
 */
Signal movingAverage(const Signal& x, int window, BndryMode mode) {
  auto node = std::make_shared<detail::Node>();
  node->kind = OpKind::MOVING_AVERAGE;
  node->inputs = {x.node()};
  node->shape = x.shape();
  node->type = x.type();
  node->scalar = static_cast<real64_t>(window);
  node->boundary = mode;
  return Signal(node);
}

/**
 * @brief 中值滤波器
 * @param window 窗口长度（建议奇数）
 * @param mode 边界策略
 */
Signal median(const Signal& x, int window, BndryMode mode) {
  auto node = std::make_shared<detail::Node>();
  node->kind = OpKind::MEDIAN;
  node->inputs = {x.node()};
  node->shape = x.shape();
  node->type = x.type();
  node->scalar = static_cast<real64_t>(window);
  node->boundary = mode;
  return Signal(node);
}

}  // namespace prism::dsl::filter
