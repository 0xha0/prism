/**
 * @file autotune.cpp
 * @ingroup examples
 * @brief APM DSSS autotune generators (TX/RX only)
 */

#include <Halide.h>

#include <string>

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

class ApmDsssTxGenerator : public Halide::Generator<ApmDsssTxGenerator> {
 public:
  Halide::GeneratorParam<std::string> configPath{"configPath", "examples/apm_dsss/config.toml"};

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

class ApmDsssRxGenerator : public Halide::Generator<ApmDsssRxGenerator> {
 public:
  Halide::GeneratorParam<std::string> configPath{"configPath", "examples/apm_dsss/config.toml"};

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

    int const rxInputSize =
        static_cast<int>(args.symbols) * args.chipLen * static_cast<int>(args.samplesPerSymbol);
    Signal const inputSig = Signal::input(rxInputSize, prism::ScalarType::C32);

    Signal const rxLpfI = filter::fir(real(inputSig), filters.lpfTaps);
    Signal const rxLpfQ = filter::fir(imag(inputSig), filters.lpfTaps);
    Signal const mfI = filter::fir(rxLpfI, filters.shapingTaps);
    Signal const mfQ = filter::fir(rxLpfQ, filters.shapingTaps);
    Signal const chipI = downsample(mfI, args.samplesPerSymbol, filters.downModelDelay);
    Signal const chipQ = downsample(mfQ, args.samplesPerSymbol, filters.downModelDelay);
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

HALIDE_REGISTER_GENERATOR(ApmDsssTxGenerator, apm_dsss_tx)
HALIDE_REGISTER_GENERATOR(ApmDsssRxGenerator, apm_dsss_rx)
