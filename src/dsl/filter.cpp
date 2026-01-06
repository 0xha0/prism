/**
 * @file filter.cpp
 * @ingroup dsl
 * @brief 滤波算子实现
 */

#include "prism/dsl/filter.h"

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

#include "prism/dsl/signal.h"
#include "prism/types.h"

namespace prism::dsl::filter {

/**
 * @brief 构建 FIR 节点 (模板版本)
 */
template <typename T>
Signal fir(const Signal& x, const std::vector<T>& taps, BndryMode mode) {
  auto node = std::make_shared<detail::Node>();
  node->kind = OpKind::FIR;
  node->inputs = {x.node()};
  node->shape = x.shape();
  node->inputType = x.type();
  ScalarType const tapsType = getScalarType<T>();
  if (!isPrecisionMatch(x.type(), tapsType)) {
    throw std::invalid_argument("FIR precision mismatch: input and taps must have same precision");
  }
  node->outputType = promoteTypes(x.type(), tapsType);
  node->param = taps;
  node->boundary = mode;
  return Signal(node);
}

template Signal fir<real32_t>(const Signal&, const std::vector<real32_t>&, BndryMode);
template Signal fir<real64_t>(const Signal&, const std::vector<real64_t>&, BndryMode);
template Signal fir<complex32_t>(const Signal&, const std::vector<complex32_t>&, BndryMode);
template Signal fir<complex64_t>(const Signal&, const std::vector<complex64_t>&, BndryMode);

/**
 * @brief 移动平均滤波器工厂
 *
 * 构建 OpKind::MOVING_AVERAGE 节点
 * 窗口大小 N 暂存入 `node->param`
 */
Signal movingAverage(const Signal& x, int window, BndryMode mode) {
  auto node = std::make_shared<detail::Node>();
  node->kind = OpKind::MOVING_AVERAGE;
  node->inputs = {x.node()};
  node->shape = x.shape();
  node->inputType = x.type();
  node->outputType = x.type();
  node->param = static_cast<int32_t>(window);
  node->boundary = mode;
  return Signal(node);
}

/**
 * @brief 中值滤波器工厂
 *
 * 构建 OpKind::MEDIAN 节点
 * 窗口大小 W 暂存入 `node->param`
 */
Signal median(const Signal& x, int window, BndryMode mode) {
  auto node = std::make_shared<detail::Node>();
  node->kind = OpKind::MEDIAN;
  node->inputs = {x.node()};
  node->shape = x.shape();
  node->inputType = x.type();
  node->outputType = x.type();
  node->param = static_cast<int32_t>(window);
  node->boundary = mode;
  return Signal(node);
}

}  // namespace prism::dsl::filter
