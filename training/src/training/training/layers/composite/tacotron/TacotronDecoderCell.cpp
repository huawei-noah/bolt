// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "TacotronDecoderCell.h"
#include "TacotronDecoderRNN.h"

#include <algorithm>

#include <training/api/API.h>
#include <training/layers/activations/ReLUActivation.h>
#include <training/layers/basic/ConcatenationLayer.h>
#include <training/layers/basic/DropoutLayer.h>
#include <training/layers/basic/MatMulLayer.h>
#include <training/layers/basic/ReshapeLayer.h>
#include <training/layers/basic/SplitterLayer.h>
#include <training/layers/basic/trainable/LinearLayer.h>
#include <training/layers/composite/BahdanauMonotonicAttentionLayer.h>
#include <training/layers/composite/DynamicConvolutionAttentionLayer.h>
#include <training/layers/composite/rnn/LSTMCellLayer.h>

#include <training/tools/Profiler.h>

namespace
{
using namespace std;
using namespace raul;

void AddPreNet(Workflow* work, const Name& name, const BasicParams& params, const TacotronParams& tparams)
{
    string inp = params.getInputs().front();
    bool isShared = !params.getSharedLayer().empty();
    Name parentLayer;
    if (isShared)
    {
        parentLayer = params.getSharedLayer();
    }

    for (size_t i = 0; i < tparams.prenetLayers.size(); ++i)
    {
        auto suffix = "[" + to_string(i + 1) + "]";
        if (isShared)
        {
            work->add<LinearLayer>(
                name / "dense" + suffix,
                LinearParams{ { { inp }, { name / "dense" + suffix }, parentLayer / "dense" + suffix, tparams.getLayerExecutionTarget() }, tparams.prenetLayers[i], true, tparams.frozen });
        }
        else
        {
            work->add<LinearLayer>(name / "dense" + suffix, LinearParams{ inp, name / "dense" + suffix, tparams.prenetLayers[i], tparams.getLayerExecutionTarget(), true, tparams.frozen });
        }
        work->add<ReLUActivation>(name / "relu" + suffix, BasicParams{ { name / "dense" + suffix }, { name / "relu" + suffix } });
        string outp = name / "dropout" + suffix;
        if (i == tparams.prenetLayers.size() - 1)
        {
            outp = params.getOutputs().front();
        }
        work->add<DropoutLayer>(name / "dropout" + suffix, DropoutParams{ { name / "relu" + suffix }, { outp }, tparams.dropoutRate });
        inp = outp;
    }
}
}

namespace raul
{
namespace tacotron
{

void AddDecoderCellDurationPrediction(Workflow* work, const Name& name, const BasicParams& params, const TacotronParams& tparams)
{
    const auto& inputs = params.getInputs();
    const auto& outputs = params.getOutputs();
    const auto& parentLayer = params.getSharedLayer();

    bool isShared = !parentLayer.empty();

    auto sharedName = parentLayer.empty() ? name : parentLayer;

    auto input = inputs[0];
    auto duration_prediction = inputs[1];

    auto rnn_cell_state = Names(inputs.begin() + 2, inputs.begin() + 2 + 2 * tparams.decoderLstmUnits.size());

    Name condition = inputs.size() == (2 + 2 * tparams.decoderLstmUnits.size() + 1) ? inputs.back() : Name();
    if (!condition.empty())
    {
        THROW("Tacotron", name, "conditions not supported with duration prediction");
    }

    auto cell_outputs = outputs[0];
    auto next_cell_state = Names(outputs.begin() + 1, outputs.end());

    // PreNet
    AddPreNet(work, name / "prenet", { { input }, { name / "prenet_output" }, isShared ? parentLayer / "prenet" : Name() }, tparams);
    // ConcatenationLayer (prenet_output, duration_prediction) -> (LSTM_input)
    work->add<ConcatenationLayer>(name / "concat", BasicParamsWithDim{ { name / "prenet_output", duration_prediction }, { name / "lstm_input" }, "width" });

    // DecoderRNN (LSTM_input) -> LSTM_output
    auto decoder_inputs = rnn_cell_state;
    decoder_inputs.insert(decoder_inputs.begin(), name / "lstm_input");
    AddDecoderRNN(work, name / "decoder_LSTM", { decoder_inputs, { name / "lstm_output" }, isShared ? parentLayer / "decoder_LSTM" : Name() }, next_cell_state, tparams);

    // ConcatenationLayer (LSTM_output, duration_prediction) -> (projections_input)
    work->add<ConcatenationLayer>(name / "concat2", BasicParamsWithDim{ { name / "lstm_output", duration_prediction }, { name / "projections_input" }, "width" });

    // Frame Projection (projections_input) -> cell_outputs
    work->add<LinearLayer>(name / "frame_projection",
                           LinearParams{ { { name / "projections_input" }, { cell_outputs }, isShared ? parentLayer / "frame_projection" : Name(), tparams.getLayerExecutionTarget() },
                                         tparams.numMels * tparams.outputsPerStep,
                                         true,
                                         tparams.frozen });
}

/**
 * inputs:
 *   if useDurationPrediction:
 *     - previous_step_output: encoder output or output from previous decoding step (zero for first step)
 *     - duration_prediction
 *     - [state]: optional previous state tensors (will be initialized if not provided):
 *       - rnn cell state
 *         - h
 *         - c
 *   else:
 *   - encoder_output
 *   - input_lengths: Tensor with shape [N, 1, 1, 1] where N is batch size and values are the lengths
 *    of each sequence in inputs.
 *   - previous_step_output: encoder output or output from previous decoding step
 *   - state: optional previous state tensors:
 *     - attention (initial values - zero)
 *     - alignments (initial values - Dirac distribution)
 *     - rnn cell state:
 *       - rnn_cell_h (initial values - zero)
 *       - rnn_cell_c (initial values - zero)
 * outputs:
 *   - cell_outputs
 *   - stop_tokens if !useDurationPrediction
 *   - next state:
 *     - attention if !useDurationPrediction
 *     - alignments if !useDurationPrediction
 *     - rnn cell state:
 *       - rnn_cell_h
 *       - rnn_cell_c
 */
void AddDecoderCell(Workflow* work, const Name& name, const BasicParams& params, const TacotronParams& tparams)
{
    if (tparams.useDurationPrediction)
    {
        AddDecoderCellDurationPrediction(work, name, params, tparams);
        return;
    }
    const auto& inputs = params.getInputs();
    const auto& outputs = params.getOutputs();
    const auto& parentLayer = params.getSharedLayer();

    bool isShared = !parentLayer.empty();

    const auto& encoder_output = inputs[0];
    const auto& input_lengths = inputs[1];

    auto sharedName = parentLayer.empty() ? name : parentLayer;

    auto input = inputs[2];
    auto state_attention = inputs[3];
    auto state_alignments = inputs[4];

    auto rnn_cell_state = Names(inputs.begin() + (tparams.useAttentionRnn ? 7 : 5), inputs.begin() + (tparams.useAttentionRnn ? 7 : 5) + 2 * tparams.decoderLstmUnits.size());

    Name condition = inputs.size() == ((tparams.useAttentionRnn ? 7 : 5) + 2 * tparams.decoderLstmUnits.size() + 1) ? inputs.back() : Name();

    auto cell_outputs = outputs[0];
    auto stop_tokens = outputs[1];
    auto next_attention = outputs[2];
    auto next_alignments = outputs[3];
    auto next_cell_state = Names(outputs.begin() + (tparams.useAttentionRnn ? 6 : 4), outputs.end());

    // PreNet
    AddPreNet(work, name / "prenet", { { input }, { name / "prenet_output" }, isShared ? parentLayer / "prenet" : Name() }, tparams);
    // ConcatenationLayer (prenet_output, previous_context_vector from attention) -> (concat_output)
    auto lstm_input = name / "lstm_input";
    if (condition.empty())
    {
        work->add<ConcatenationLayer>(name / "concat1", BasicParamsWithDim{ { name / "prenet_output", state_attention }, { lstm_input }, "width" });
    }
    else
    {
        work->add<ConcatenationLayer>(name / "concat1", BasicParamsWithDim{ { name / "prenet_output", state_attention, condition }, { lstm_input }, "width" });
    }
    // DecoderRNN (concat_output) -> LSTM_output
    auto lstm_output = name / "lstm_output";

    if (tparams.useAttentionRnn)
    {
        auto attention_rnn_state = Names(inputs.begin() + 5, inputs.begin() + 7);
        auto next_attention_rnn_state = Names(outputs.begin() + 4, outputs.begin() + 6);

        LSTMCellParams lstmParams{ { { lstm_input, attention_rnn_state[0], attention_rnn_state[1] },
                                     { next_attention_rnn_state[0], next_attention_rnn_state[1] },
                                     isShared ? parentLayer / "attention_LSTM" : Name(),
                                     tparams.getLayerExecutionTarget() },
                                   true,
                                   tparams.zoneoutRate,
                                   true,
                                   1.f,
                                   tparams.frozen };

        LSTMCellLayer(name / "attention_LSTM", lstmParams, work->getNetworkParameters());
        work->add<SplitterLayer>(name / "attention_rnn_splitter", BasicParams{ { next_attention_rnn_state[0] }, { lstm_output / "1", name / "attention_rnn_output" } });
    }
    else
    {
        auto decoder_inputs = rnn_cell_state;
        decoder_inputs.insert(decoder_inputs.begin(), lstm_input);
        AddDecoderRNN(work, name / "decoder_LSTM", { decoder_inputs, { lstm_output }, isShared ? parentLayer / "decoder_LSTM" : Name() }, next_cell_state, tparams);
        work->add<SplitterLayer>(name / "lstm_splitter", BasicParams{ { lstm_output }, { lstm_output / "1", lstm_output / "2" } });
    }

    // begin _compute_attention implementation
    // Attention (LSTM_output, previous_alignments) -> (context_vector, alignments)
    Names attentionOutput{ name / "alignments" };
    if (!isShared)
    {
        attentionOutput.push_back(name / "attention_values");
    }

    if (tparams.attentionType == "StepwiseMonotonic")
    {
        BahdanauAttentionParams attentionParams = { { { lstm_output / "1", state_alignments, encoder_output, input_lengths },
                                                      attentionOutput, // next_attention_state for Bahdanau is same as alignments
                                                      isShared ? parentLayer / "attention_mechanism" : Name() },
                                                    tparams.attentionDim,
                                                    tparams.attentionNormalizeEnergy,
                                                    tparams.attentionSigmoidNoise,
                                                    tparams.attentionScoreBiasInit,
                                                    tparams.attentionMode,
                                                    true,
                                                    true,
                                                    tparams.frozen };

        BahdanauMonotonicAttentionLayer(name / "attention_mechanism", attentionParams, work->getNetworkParameters(), tparams.getLayerExecutionTarget());
    }
    else if (tparams.attentionType == "DynamicConvolutional")
    {
        Names dcaInput = { lstm_output / "1", state_alignments, encoder_output, input_lengths };
        DynamicConvolutionAttentionParams attentionParams1 = {
            dcaInput,
            attentionOutput, // next_attention_state for DCA is same as alignments
            isShared ? parentLayer / "attention_mechanism" : Name(),
            tparams.attentionDim,
            DynamicConvolutionAttentionParams::hparams{
                tparams.attentionFilters, tparams.attentionKernelSize, tparams.attentionPriorFilterSize, tparams.attentionWindowSize, tparams.attentionPriorAlpha, tparams.attentionPriorBeta },
            false,
            tparams.frozen
        };

        DynamicConvolutionAttentionLayer(name / "attention_mechanism", attentionParams1, work->getNetworkParameters(), tparams.getLayerExecutionTarget());
    }
    else
    {
        THROW("Tacotron", name, "unsupported attention type '" + tparams.attentionType + "'");
    }
    work->add<SplitterLayer>(name / "alignments_splitter", BasicParams{ { name / "alignments" }, { name / "alignments" / "1", next_alignments } });
    work->add<MatMulLayer>(name / "mul_context", MatMulParams{ { name / "alignments" / "1", isShared ? parentLayer / "attention_values" : name / "attention_values" }, name / "context_vector" });

    work->add<SplitterLayer>(name / "context_splitter", BasicParams{ { name / "context_vector" }, { name / "context_vector" / "1", next_attention } });
    // end _compute_attention implementation

    if (tparams.useAttentionRnn)
    {
        work->add<ConcatenationLayer>(name / "concat_attn_rnn", BasicParamsWithDim{ { name / "attention_rnn_output", name / "context_vector" / "1" }, { name / "lstm_input" / "1" }, "width" });
        auto decoder_inputs = rnn_cell_state;
        decoder_inputs.insert(decoder_inputs.begin(), name / "lstm_input" / "1");
        AddDecoderRNN(work, name / "decoder_LSTM", { decoder_inputs, { lstm_output / "2" }, isShared ? parentLayer / "decoder_LSTM" : Name() }, next_cell_state, tparams);
    }

    // ConcatenationLayer (LSTM_output, context_vector) -> (projections_input)
    work->add<ConcatenationLayer>(name / "concat2", BasicParamsWithDim{ { lstm_output / "2", name / "context_vector" / "1" }, { name / "projections_input" }, "width" });

    // Frame Projection (projections_input) -> cell_outputs
    work->add<LinearLayer>(name / "frame_projection",
                           LinearParams{ { { name / "projections_input" }, { cell_outputs }, isShared ? parentLayer / "frame_projection" : Name(), tparams.getLayerExecutionTarget() },
                                         tparams.numMels * tparams.outputsPerStep,
                                         true,
                                         tparams.frozen });

    // Stop Projection (projections_input) -> stop_tokens
    work->add<LinearLayer>(name / "stop_projection_dense",
                           LinearParams{ { { name / "projections_input" }, { stop_tokens }, isShared ? parentLayer / "stop_projection_dense" : Name(), tparams.getLayerExecutionTarget() },
                                         tparams.outputsPerStep,
                                         true,
                                         tparams.frozen });
}

} // namespace tacotron
} // namespace raul
