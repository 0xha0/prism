/**
 * @file autotune.cpp
 * @ingroup examples
 * @brief APM DSSS + EQ autotune generators (TX/RX only)
 */

#include <Halide.h>

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "example_helper.h"
#include "prism/dsl/filter.h"
#include "prism/dsl/modem.h"
#include "prism/dsl/ops.h"
#include "prism/dsl/signal.h"
#include "prism/runtime/halide_builder.h"
#include "prism/simulation/rng.h"
#include "prism/simulation/source.h"
#include "prism/types.h"

using prism::real32_t;
using prism::dsl::Signal;
using namespace prism::dsl;
using namespace prism::examples;
using namespace prism::simulation;

namespace {

std::vector<real32_t> buildChannelImpulse(const ChannelArgs& channel, int& maxDelay) {
  bool const useFading = channel.enableFading && !channel.fadingTaps.empty();
  maxDelay = 0;
  if (useFading) {
    for (int const d : channel.fadingDelays) {
      maxDelay = std::max(maxDelay, d);
    }
  }
  if (maxDelay < 0) {
    maxDelay = 0;
  }

  std::vector<real32_t> h(static_cast<size_t>(maxDelay + 1), 0.0F);
  if (useFading) {
    for (size_t i = 0; i < channel.fadingTaps.size(); ++i) {
      int const delay = (i < channel.fadingDelays.size()) ? channel.fadingDelays[i] : 0;
      if (delay < 0 || delay > maxDelay) {
        continue;
      }
      h[static_cast<size_t>(delay)] += channel.fadingTaps[i];
    }
  } else {
    h[0] = 1.0F;
  }

  if (channel.gain != 1.0) {
    for (auto& v : h) {
      v = static_cast<real32_t>(v * channel.gain);
    }
  }

  bool nonZero = false;
  for (auto v : h) {
    if (std::abs(v) > 1e-12F) {
      nonZero = true;
      break;
    }
  }
  if (!nonZero) {
    h.assign(1, 1.0F);
    maxDelay = 0;
  }
  return h;
}

std::vector<double> solveLinearSystem(std::vector<double> a, std::vector<double> b, int n,
                                      bool& ok) {
  ok = true;
  for (int i = 0; i < n; ++i) {
    int pivot = i;
    double maxAbs = std::abs(a[static_cast<size_t>(i * n + i)]);
    for (int r = i + 1; r < n; ++r) {
      double const v = std::abs(a[static_cast<size_t>(r * n + i)]);
      if (v > maxAbs) {
        maxAbs = v;
        pivot = r;
      }
    }
    if (maxAbs < 1e-12) {
      ok = false;
      return {};
    }
    if (pivot != i) {
      for (int c = i; c < n; ++c) {
        std::swap(a[static_cast<size_t>(i * n + c)],
                  a[static_cast<size_t>(pivot * n + c)]);
      }
      std::swap(b[static_cast<size_t>(i)], b[static_cast<size_t>(pivot)]);
    }

    double const diag = a[static_cast<size_t>(i * n + i)];
    for (int c = i; c < n; ++c) {
      a[static_cast<size_t>(i * n + c)] /= diag;
    }
    b[static_cast<size_t>(i)] /= diag;

    for (int r = 0; r < n; ++r) {
      if (r == i) {
        continue;
      }
      double const factor = a[static_cast<size_t>(r * n + i)];
      if (std::abs(factor) < 1e-12) {
        continue;
      }
      for (int c = i; c < n; ++c) {
        a[static_cast<size_t>(r * n + c)] -= factor * a[static_cast<size_t>(i * n + c)];
      }
      b[static_cast<size_t>(r)] -= factor * b[static_cast<size_t>(i)];
    }
  }
  return b;
}

std::vector<real32_t> designIdealEqTaps(const std::vector<real32_t>& h, int eqLen,
                                        int targetDelay, real32_t reg, std::string& err) {
  if (eqLen <= 0) {
    err = "均衡器长度必须为正数";
    return {1.0F};
  }
  int const hLen = static_cast<int>(h.size());
  if (hLen <= 0) {
    err = "信道冲激响应为空";
    return {1.0F};
  }
  int const totalLen = hLen + eqLen - 1;
  if (targetDelay < 0 || targetDelay >= totalLen) {
    targetDelay = std::clamp(targetDelay, 0, totalLen - 1);
  }

  std::vector<double> ata(static_cast<size_t>(eqLen * eqLen), 0.0);
  std::vector<double> atb(static_cast<size_t>(eqLen), 0.0);

  for (int i = 0; i < eqLen; ++i) {
    int const idx = targetDelay - i;
    if (idx >= 0 && idx < hLen) {
      atb[static_cast<size_t>(i)] = static_cast<double>(h[static_cast<size_t>(idx)]);
    }
  }

  for (int i = 0; i < eqLen; ++i) {
    for (int j = 0; j < eqLen; ++j) {
      double sum = 0.0;
      for (int m = 0; m < totalLen; ++m) {
        int const idxI = m - i;
        int const idxJ = m - j;
        double const hi =
            (idxI >= 0 && idxI < hLen) ? static_cast<double>(h[static_cast<size_t>(idxI)]) : 0.0;
        double const hj =
            (idxJ >= 0 && idxJ < hLen) ? static_cast<double>(h[static_cast<size_t>(idxJ)]) : 0.0;
        sum += hi * hj;
      }
      ata[static_cast<size_t>(i * eqLen + j)] = sum;
    }
  }

  for (int i = 0; i < eqLen; ++i) {
    ata[static_cast<size_t>(i * eqLen + i)] += reg;
  }

  bool ok = false;
  auto sol = solveLinearSystem(ata, atb, eqLen, ok);
  if (!ok || sol.size() != static_cast<size_t>(eqLen)) {
    err = "均衡器求解失败";
    return {1.0F};
  }

  std::vector<real32_t> taps(static_cast<size_t>(eqLen));
  for (int i = 0; i < eqLen; ++i) {
    taps[static_cast<size_t>(i)] = static_cast<real32_t>(sol[static_cast<size_t>(i)]);
  }
  return taps;
}

bool loadDsssArgs(const std::string& path, DsssArgs& args, StandardFilters& filters,
                  std::vector<real32_t>& pnCodeRev, std::string& err) {
  if (!loadDsssConfig(path, args, err) || !finalizeDsssArgs(args, err)) {
    return false;
  }
  if (args.pnCode.empty()) {
    RNG pnRng(args.pnSeed);
    args.pnCode = generatePnCode(args.chipLen, pnRng);
  }
  pnCodeRev.assign(args.pnCode.rbegin(), args.pnCode.rend());
  return setupStandardFilters(args, filters, err);
}

class ApmDsssEqTxGenerator : public Halide::Generator<ApmDsssEqTxGenerator> {
 public:
  Halide::GeneratorParam<std::string> configPath{"configPath",
                                                 "examples/apm_dsss_eq/config.toml"};

  Input<Halide::Buffer<real32_t, 1>> input{"input"};
  Output<Halide::Buffer<real32_t, 2>> output{"output"};

  void generate() {
    DsssArgs args;
    StandardFilters filters;
    std::vector<real32_t> pnCodeRev;
    std::string err;
    if (!loadDsssArgs(configPath, args, filters, pnCodeRev, err)) {
      _halide_user_error << err;
    }

    Signal const inputSig = Signal::input(args.symbols);
    Signal const mapSig = (args.scheme == ModemScheme::PSK) ? modem::pskMap(inputSig, args.order)
                                                            : modem::qamMap(inputSig, args.order);
    Signal const spreadI = filter::fir(upsample(real(mapSig), args.chipLen), args.pnCode);
    Signal const spreadQ = filter::fir(upsample(imag(mapSig), args.chipLen), args.pnCode);
    Signal const upI = upsample(spreadI, args.samplesPerSymbol);
    Signal const upQ = upsample(spreadQ, args.samplesPerSymbol);
    Signal const shapeI = filter::fir(upI, filters.shapingTaps);
    Signal const shapeQ = filter::fir(upQ, filters.shapingTaps);
    Signal const txSig = complexPack(shapeI, shapeQ);

    prism::runtime::OpContext<real32_t> ctx;
    Halide::ImageParam inputParam = input;
    ctx.inputParams[inputSig.node().get()] = &inputParam;

    Halide::Func func = prism::runtime::buildSignalFunc<real32_t>(txSig, ctx);
    func.bound(func.args()[0], 0, 2);
    output = func;

    input.dim(0).set_estimate(0, static_cast<int>(args.symbols));
    output.dim(0).set_estimate(0, 2);
    output.dim(1).set_estimate(0, static_cast<int>(txSig.shape().length));
  }
};

class ApmDsssEqRxGenerator : public Halide::Generator<ApmDsssEqRxGenerator> {
 public:
  Halide::GeneratorParam<std::string> configPath{"configPath",
                                                 "examples/apm_dsss_eq/config.toml"};

  Input<Halide::Buffer<real32_t, 2>> input{"input"};
  Output<Halide::Buffer<real32_t, 1>> output{"output"};

  void generate() {
    DsssArgs args;
    StandardFilters filters;
    std::vector<real32_t> pnCodeRev;
    std::string err;
    if (!loadDsssArgs(configPath, args, filters, pnCodeRev, err)) {
      _halide_user_error << err;
    }

    int maxDelay = 0;
    auto channelImpulse = buildChannelImpulse(args.channel, maxDelay);
    int const eqLen = std::max(1, static_cast<int>(channelImpulse.size()));
    int eqDelay = maxDelay;
    std::string eqErr;
    auto eqTaps = designIdealEqTaps(channelImpulse, eqLen, eqDelay, 1e-6F, eqErr);
    if (!eqErr.empty()) {
      _halide_user_error << eqErr;
    }

    int const rxInputSize =
        static_cast<int>(args.symbols) * args.chipLen * static_cast<int>(args.samplesPerSymbol);
    Signal const inputSig = Signal::input(rxInputSize, prism::ScalarType::C32);

    Signal const rxLpfI = filter::fir(real(inputSig), filters.lpfTaps);
    Signal const rxLpfQ = filter::fir(imag(inputSig), filters.lpfTaps);
    Signal const rxLpf = complexPack(rxLpfI, rxLpfQ);
    Signal const eqSig = filter::fir(rxLpf, eqTaps);
    Signal const mfI = filter::fir(real(eqSig), filters.shapingTaps);
    Signal const mfQ = filter::fir(imag(eqSig), filters.shapingTaps);
    int const rxDownDelay = filters.downModelDelay + eqDelay;
    Signal const chipI = downsample(mfI, args.samplesPerSymbol, rxDownDelay);
    Signal const chipQ = downsample(mfQ, args.samplesPerSymbol, rxDownDelay);
    Signal const corrI = filter::fir(chipI, pnCodeRev);
    Signal const corrQ = filter::fir(chipQ, pnCodeRev);
    Signal const symI = downsample(corrI, args.chipLen, args.chipLen - 1);
    Signal const symQ = downsample(corrQ, args.chipLen, args.chipLen - 1);
    Signal const normI = scale(symI, 1.0F / static_cast<real32_t>(args.chipLen));
    Signal const normQ = scale(symQ, 1.0F / static_cast<real32_t>(args.chipLen));

    Signal const rxSig = (args.scheme == ModemScheme::PSK)
                             ? modem::pskDemap(normI, normQ, args.order)
                             : modem::qamDemap(normI, normQ, args.order);

    prism::runtime::OpContext<real32_t> ctx;
    Halide::ImageParam inputParam = input;
    ctx.inputParams[inputSig.node().get()] = &inputParam;

    Halide::Func func = prism::runtime::buildSignalFunc<real32_t>(rxSig, ctx);
    output = func;

    input.dim(0).set_estimate(0, 2);
    input.dim(1).set_estimate(0, rxInputSize);
    output.dim(0).set_estimate(0, static_cast<int>(rxSig.shape().length));
  }
};

}  // namespace

HALIDE_REGISTER_GENERATOR(ApmDsssEqTxGenerator, apm_dsss_eq_tx)
HALIDE_REGISTER_GENERATOR(ApmDsssEqRxGenerator, apm_dsss_eq_rx)
