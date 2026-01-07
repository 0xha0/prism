/**
 * @file autotune.cpp
 * @ingroup examples
 * @brief APM basic autotune generators
 */

#include <Halide.h>

#include <string>

#include "example_helper.h"
#include "prism/dsl/filter.h"
#include "prism/dsl/modem.h"
#include "prism/dsl/ops.h"
#include "prism/dsl/signal.h"
#include "prism/runtime/halide_builder.h"
#include "prism/types.h"

using prism::real32_t;
using prism::dsl::Signal;
using namespace prism::dsl;
using namespace prism::examples;

namespace {

class ApmBasicTxGenerator : public Halide::Generator<ApmBasicTxGenerator> {
 public:
  // NOLINTBEGIN
  Halide::GeneratorParam<std::string> configPath{"configPath", "examples/apm_basic/config.toml"};
  Input<Halide::Buffer<real32_t, 1>> input{"input"};
  Output<Halide::Buffer<real32_t, 2>> output{"output"};
  // NOLINTEND

  void generate() {
    StandardArgs args;
    std::string err;
    if (!loadStandardConfig(configPath, args, err) || !finalizeStandardArgs(args, err)) {
      _halide_user_error << err;
    }

    StandardFilters filters;
    if (!setupStandardFilters(args, filters, err)) {
      _halide_user_error << err;
    }

    Signal const inputSig = Signal::input(args.symbols);
    Signal const mapSig = (args.scheme == ModemScheme::PSK) ? modem::pskMap(inputSig, args.order)
                                                            : modem::qamMap(inputSig, args.order);
    Signal const upI = upsample(real(mapSig), args.samplesPerSymbol);
    Signal const upQ = upsample(imag(mapSig), args.samplesPerSymbol);
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

class ApmBasicRxGenerator : public Halide::Generator<ApmBasicRxGenerator> {
 public:
  Halide::GeneratorParam<std::string> configPath{"configPath", "examples/apm_basic/config.toml"};

  Input<Halide::Buffer<real32_t, 2>> input{"input"};
  Output<Halide::Buffer<real32_t, 1>> output{"output"};

  void generate() {
    StandardArgs args;
    std::string err;
    if (!loadStandardConfig(configPath, args, err) || !finalizeStandardArgs(args, err)) {
      _halide_user_error << err;
    }

    StandardFilters filters;
    if (!setupStandardFilters(args, filters, err)) {
      _halide_user_error << err;
    }

    int const rxInputSize =
        static_cast<int>(args.symbols) * static_cast<int>(args.samplesPerSymbol);
    Signal const inputSig = Signal::input(rxInputSize, prism::ScalarType::C32);

    Signal const rxLpfI = filter::fir(real(inputSig), filters.lpfTaps);
    Signal const rxLpfQ = filter::fir(imag(inputSig), filters.lpfTaps);
    Signal const mfI = filter::fir(rxLpfI, filters.shapingTaps);
    Signal const mfQ = filter::fir(rxLpfQ, filters.shapingTaps);
    Signal const downI = downsample(mfI, args.samplesPerSymbol, filters.downModelDelay);
    Signal const downQ = downsample(mfQ, args.samplesPerSymbol, filters.downModelDelay);

    Signal const rxSig = (args.scheme == ModemScheme::PSK)
                             ? modem::pskDemap(downI, downQ, args.order)
                             : modem::qamDemap(downI, downQ, args.order);

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

HALIDE_REGISTER_GENERATOR(ApmBasicTxGenerator, apm_basic_tx)
HALIDE_REGISTER_GENERATOR(ApmBasicRxGenerator, apm_basic_rx)
