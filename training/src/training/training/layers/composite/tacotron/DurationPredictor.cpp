// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "DurationPredictor.h"

#include <training/api/API.h>

#include <training/layers/activations/SoftPlusActivation.h>
#include <training/layers/basic/ConcatenationLayer.h>
#include <training/layers/basic/ConvertPrecisionLayer.h>
#include <training/layers/basic/ElementWiseMulLayer.h>
#include <training/layers/basic/PositionalEncoding.h>
#include <training/layers/basic/ReshapeLayer.h>
#include <training/layers/basic/RoundLayer.h>
#include <training/layers/basic/ScaleLayer.h>
#include <training/layers/basic/TensorLayer.h>
#include <training/layers/basic/TileLayer.h>
#include <training/layers/basic/trainable/LinearLayer.h>
#include <training/layers/composite/GaussianUpsamplingLayer.h>
#include <training/layers/composite/rnn/BidirectionalLSTMFunc.h>

namespace raul::tacotron
{
using namespace std;

void AddDurationPredictor(Workflow* work, const Name& name, const BasicParams& params, const TacotronParams& tparams)
{
    auto encoder_output = params.getInputs()[0];
    auto input_lengths = params.getInputs()[1];
    auto duration_targets = params.getInputs()[2];
    auto mask = params.getInputs().size() > 3 ? params.getInputs()[3] : Name{};

    auto duration_input = encoder_output;

    auto [durations_output, durations, upsampled_output, positional_embeddings] = std::make_tuple(params.getOutputs()[0], params.getOutputs()[1], params.getOutputs()[2], params.getOutputs()[3]);

    auto height = work->getHeight(encoder_output);

    if (!mask.empty())
    {
        auto tiled_mask = mask;
        if (/*work->getExecutionTarget() == ExecutionTarget::GPU &&*/ work->getWidth(encoder_output) != work->getWidth(mask))
        {
            tiled_mask = name / "tiled_mask";
            work->add<TileLayer>(name / "tile_mask", TilingParams{ mask, tiled_mask, work->getWidth(encoder_output), Dimension::Width });
        }
        work->add<ElementWiseMulLayer>(name / "mul_encoder", ElementWiseLayerParams{ { encoder_output, tiled_mask }, { name / "encoder_output" } });
        duration_input = name / "encoder_output";
    }

    work->add<TensorLayer>(name / "zero_state",
                           TensorParams{ { name / "zero_state" / "1" / "h", name / "zero_state" / "1" / "c", name / "zero_state" / "2" / "h", name / "zero_state" / "2" / "c" },
                                         WShape{ BS(), 1u, 1u, tparams.durationPredictorLstmUnits },
                                         0_dt });

    BidirectionalLSTMFunc(name / "duration_predictor_LSTM" / "1",
                          LSTMParams{ { duration_input, input_lengths },
                                      { name / "duration_predictor_rnn_output" / "1" },
                                      tparams.durationPredictorLstmUnits,
                                      tparams.getLayerExecutionTarget(),
                                      true,
                                      tparams.frozen,
                                      false,
                                      tparams.zoneoutRate,
                                      true,
                                      1_dt },
                          work->getNetworkParameters());

    BidirectionalLSTMFunc(name / "duration_predictor_LSTM" / "2",
                          LSTMParams{ { name / "duration_predictor_rnn_output" / "1", input_lengths },
                                      { name / "duration_predictor_rnn_output" },
                                      tparams.durationPredictorLstmUnits,
                                      tparams.getLayerExecutionTarget(),
                                      true,
                                      tparams.frozen,
                                      false,
                                      tparams.zoneoutRate,
                                      true,
                                      1.f },
                          work->getNetworkParameters());

    if (!mask.empty())
    {
        work->add<LinearLayer>(name / "duration_predictor_projection",
                               LinearParams{ name / "duration_predictor_rnn_output", name / "duration_projection_output", 1, tparams.getLayerExecutionTarget() });
        work->add<ElementWiseMulLayer>(name / "mul_duration_predictor_projection", ElementWiseLayerParams{ { name / "duration_projection_output", mask }, { durations } });
    }
    else
    {
        work->add<LinearLayer>(name / "duration_predictor_projection", LinearParams{ name / "duration_predictor_rnn_output", durations, 1 });
    }

    work->add<ScaleLayer>(name / "scale_targets", ScaleParams{ { duration_targets }, { name / "scaled_duration_targets" }, 100_dt / TODTYPE(tparams.outputsPerStep) });
    work->add<RoundLayer>(name / "round_targets", BasicParams{ { name / "scaled_duration_targets" }, { name / "duration_targets_in_frames" } });

    work->add<ReshapeLayer>(name / "reshape_targets", ViewParams{ name / "duration_targets_in_frames", name / "expanded_duration_targets_in_frames", 1, static_cast<int>(height), 1 });
    work->add<ConcatenationLayer>(name / "concat_targets",
                                  BasicParamsWithDim{ { duration_input, name / "expanded_duration_targets_in_frames" }, { name / "range_predictor_input" }, Dimension::Width });

    BidirectionalLSTMFunc(name / "range_predictor_LSTM" / "1",
                          LSTMParams{ { name / "range_predictor_input", input_lengths },
                                      { name / "range_predictor_rnn_output" / "1" },
                                      tparams.durationPredictorLstmUnits,
                                      tparams.getLayerExecutionTarget(),
                                      true,
                                      tparams.frozen,
                                      false,
                                      tparams.zoneoutRate,
                                      true,
                                      1_dt },
                          work->getNetworkParameters());

    BidirectionalLSTMFunc(name / "range_predictor_LSTM" / "2",
                          LSTMParams{ { name / "range_predictor_rnn_output" / "1", input_lengths },
                                      { name / "range_predictor_rnn_output" },
                                      tparams.durationPredictorLstmUnits,
                                      tparams.getLayerExecutionTarget(),
                                      true,
                                      tparams.frozen,
                                      false,
                                      tparams.zoneoutRate,
                                      true,
                                      1_dt },
                          work->getNetworkParameters());

    if (!mask.empty())
    {
        work->add<LinearLayer>(name / "range_predictor_projection", LinearParams{ name / "range_predictor_rnn_output", name / "range_projection_output" / "1", 1, tparams.getLayerExecutionTarget() });
        work->add<SoftPlusActivation>(name / "range_predictor_projection_softplus",
                                      SoftPlusActivationParams{ name / "range_projection_output" / "1", name / "range_projection_output" / "2", 1_dt, std::numeric_limits<dtype>::max() });
        work->add<ElementWiseMulLayer>(name / "mul_range", ElementWiseLayerParams{ { name / "range_projection_output" / "2", mask }, { name / "range_projection_output" } });
    }
    else
    {
        work->add<LinearLayer>(name / "range_predictor_projection", LinearParams{ name / "range_predictor_rnn_output", name / "range_projection_output" / "1", 1, tparams.getLayerExecutionTarget() });
        work->add<SoftPlusActivation>(name / "range_predictor_projection_softplus",
                                      SoftPlusActivationParams{ name / "range_projection_output" / "1", name / "range_projection_output", 1_dt, std::numeric_limits<dtype>::max() });
    }

    auto duration_targets_in_frames_h = name / "duration_targets_in_frames_h";

    work->add<ReshapeLayer>(name / "reshape_targets_in_frames", ViewParams{ name / "duration_targets_in_frames", duration_targets_in_frames_h, 1, static_cast<int>(height), -1 });
    auto range_projection_output = name / "range_projection_output";

    Name inputGaussian = duration_input;
    Name inputGaussian2 = duration_targets_in_frames_h;
    Name inputGaussian3 = range_projection_output;
    Name outputGaussian = upsampled_output;

    if (work->getExecutionTarget() == raul::ExecutionTarget::CPUFP16)
    {
        work->overrideLayerExecutionTarget(raul::LayerExecutionTarget::CPU);
        work->add<raul::ConvertPrecisionLayer>("g_c1", raul::ConvertPrecisionParams{ { duration_input }, { duration_input + "_fp32" }, false });
        work->add<raul::ConvertPrecisionLayer>("g_c2", raul::ConvertPrecisionParams{ { duration_targets_in_frames_h }, { duration_targets_in_frames_h + "_fp32" }, false });
        work->add<raul::ConvertPrecisionLayer>("g_c3", raul::ConvertPrecisionParams{ { range_projection_output }, { range_projection_output + "_fp32" }, false });

        inputGaussian = duration_input + "_fp32";
        inputGaussian2 = duration_targets_in_frames_h + "_fp32";
        inputGaussian3 = range_projection_output + "_fp32";
        outputGaussian = upsampled_output + "_fp32";
    }
    GaussianUpsamplingLayer(name / "upsampling",
                            GaussianUpsamplingParams{ { inputGaussian, inputGaussian2, inputGaussian3 }, outputGaussian, tparams.maxMelFrames / tparams.outputsPerStep },
                            work->getNetworkParameters());
    if (work->getExecutionTarget() == raul::ExecutionTarget::CPUFP16)
    {
        work->add<raul::ConvertPrecisionLayer>("g_c4", raul::ConvertPrecisionParams{ { upsampled_output + "_fp32" }, { upsampled_output }, true });
        work->resetLayerExecutionTargetOverride();
    }

    work->add<PositionalEncoding>(
        name / "positional_encoding",
        PositionalEncodingParams{
            name / "duration_targets_in_frames", positional_embeddings, tparams.positionalEmbeddingDim, tparams.maxMelLength, true, tparams.maxMelFrames / tparams.outputsPerStep });

    work->add<ConcatenationLayer>(name / "concat_output", BasicParamsWithDim{ { upsampled_output, positional_embeddings }, { durations_output }, Dimension::Width });
}

} // namespace raul::tacotron
