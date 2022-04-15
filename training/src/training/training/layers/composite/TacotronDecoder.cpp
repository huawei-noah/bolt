// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "TacotronDecoder.h"
#include "tacotron/TacotronDataInitializationLayer.h"
#include "tacotron/TacotronDecoderCell.h"
#include "tacotron/TargetsReductionLayer.h"

#include <algorithm>

#include <training/api/API.h>
#include <training/layers/basic/ConcatenationLayer.h>
#include <training/layers/basic/ReshapeLayer.h>
#include <training/layers/basic/SlicerLayer.h>
#include <training/layers/basic/SplitterLayer.h>

namespace raul
{
namespace tacotron
{
/**
 * @brief
 * @param name
 * @param params
 *  inputs:
 *    - embedded encoder_output
 *    - [conditions] - multiple embedding tensors that would be concatenated to prenet output
 *    - input_lengths
 *    - mel_targets: float32 Tensor with shape [N, 1, T_out, M] where N is batch size,
 *                   T_out is number of steps in the output time series, M is num_mels,
 *                   and values are entries in the mel spectrogram.
 *                   Only needed for training.
 *  outputs:
 *    - frame prediction
 *    - [stop token prediction] if not tparams.useDurationPrediction
 *    - decoder output (= reshaped frame prediction)
 *    - final alignments
 *
 */
void AddDecoder(Workflow* work, const Name& name, const BasicParams& params, const TacotronParams& tparams)
{
    using namespace std;
    auto prefix = "TacotronDecoder[" + name + "::ctor]: ";
    if (tparams.teacherForcingRatio < 1.f)
    {
        THROW("TacotronDecoder", name, "teacher_forcing_ratio < 1 not supported");
    }
    if (tparams.useAttentionRnn && tparams.useDurationPrediction)
    {
        THROW("TacotronDecoder", name, "attention rnn and duration prediction can't be used together");
    }

    const auto& inputs = params.getInputs();
    const auto& outputs = params.getOutputs();
    if (inputs.size() < 3)
    {
        THROW("TacotronDecoder", name, "wrong number of input names");
    }
    if (inputs.size() != 3 && tparams.concatConditionsWithEncoderOutput)
    {
        THROW("TacotronDecoder", name, "concatConditionsWithEncoderOutput=true is incompatible with decoder conditions");
    }
    if ((tparams.useDurationPrediction && outputs.size() != 1) || (!tparams.useDurationPrediction && outputs.size() != 4))
    {
        THROW("TacotronDecoder", name, "wrong number of output names");
    }

    const auto encoderOutput = tparams.useDurationPrediction ? Name() : inputs[0];
    const auto durationPrediction = tparams.useDurationPrediction ? inputs[0] : Name();
    const auto inputLengths = *(inputs.end() - 2);
    const auto melTargetsName = inputs.back();

    Names conditions;
    copy(inputs.begin() + 1, inputs.end() - 2, back_inserter(conditions));

    const Name initialDecoderInputs = name / "zero_input";
    const Name initialDurations = name / "zero_durations";
    const Name initialAlignmentsName = name / "initial_alignments";
    const Name initialAttentionName = name / "initial_attention";
    Names initialRnnState;
    for (size_t i = 0; i < tparams.decoderLstmUnits.size(); ++i)
    {
        initialRnnState.emplace_back(name / "initial_rnn_state_h[" + to_string(i) + "]");
        initialRnnState.emplace_back(name / "initial_rnn_state_c[" + to_string(i) + "]");
    }

    Names initialData = tparams.useDurationPrediction ? Names{ initialDecoderInputs, initialDurations } : Names{ initialDecoderInputs, initialAttentionName, initialAlignmentsName };
    if (tparams.useAttentionRnn)
    {
        initialData.emplace_back(name / "initial_attention_rnn_h");
        initialData.emplace_back(name / "initial_attention_rnn_c");
    }
    copy(initialRnnState.begin(), initialRnnState.end(), back_inserter(initialData));

    const auto reducedMelTargetsName = name / "reduced_mel_targets";
    work->add<TacotronDataInitializationLayer>(name / "initializer", BasicParams{ { inputs[0] }, { initialData } }, tparams);
    work->add<TargetsReductionLayer>(name / "targets_reducer", BasicParams{ { melTargetsName }, { reducedMelTargetsName } }, tparams);

    size_t max_decoder_iterations = work->getHeight(reducedMelTargetsName);
    Names mels;
    Names durations;
    for (size_t time = 0; time < max_decoder_iterations - 1; ++time)
    {
        mels.push_back(name / "mels[" + to_string(time) + "]");
    }
    vector<int> sliceSizes(max_decoder_iterations - 1, 1);
    work->add<SlicerLayer>(name / "mel_extractor", SlicingParams{ reducedMelTargetsName, mels, "height", sliceSizes });
    if (tparams.useDurationPrediction)
    {
        for (size_t time = 0; time < max_decoder_iterations - 1; ++time)
        {
            durations.push_back(name / "durations[" + to_string(time) + "]");
        }
        work->add<SlicerLayer>(name / "durations_extractor", SlicingParams{ durationPrediction, durations, "height", sliceSizes });
    }

    Names stepInputs = tparams.useDurationPrediction ? Names{} : Names{ encoderOutput, inputLengths };
    copy(initialData.begin(), initialData.end(), back_inserter(stepInputs));

    if (!conditions.empty())
    {
        work->add<ConcatenationLayer>(name / "concat_conditions", BasicParamsWithDim{ conditions, { name / "condition" }, "width" });
        stepInputs.push_back(name / "condition");
    }

    auto mainCellName = name / "_cell";
    auto cellName = mainCellName;
    Names cellFramesPrediction;
    Names cellStopToken;

    for (size_t time = 0; time < max_decoder_iterations; ++time)
    {
        bool lastIteration = time == max_decoder_iterations - 1;

        /**
         *  - decoder_cell_output:
         *    - frames_prediction
         *    - [stop_token] if !tparams.useDurationPrediction
         *  - decoder_state:
         *    if !tparams.useDurationPrediction:
         *      - attention
         *      - alignments
         *      - [attention_rnn_state] [optional - if tparams.useAttentionRnn == true]
         *        - h
         *        - c
         *    - rnn cell state (list of size tparams.decoderLayers)
         *      - h
         *      - c
         */

        Names stepOutputs = 
              tparams.useDurationPrediction ? 
                Names{
                    cellName / "frames_prediction",
                } :
                Names{
                    cellName / "frames_prediction",
                    cellName / "stop_token",
                    cellName / "state" / "attention",
                    lastIteration ? outputs[3] : cellName / "state" / "alignments",
                };

        if (tparams.useAttentionRnn)
        {
            stepOutputs.emplace_back(cellName / "state" / "attention_rnn_h");
            stepOutputs.emplace_back(cellName / "state" / "attention_rnn_c");
        }

        for (size_t i = 0; i < tparams.decoderLstmUnits.size(); ++i)
        {
            stepOutputs.emplace_back(cellName / "state" / "rnn_cell_state_h[" + to_string(i) + "]");
            stepOutputs.emplace_back(cellName / "state" / "rnn_cell_state_c[" + to_string(i) + "]");
        }

        if (time == 0)
        {
            AddDecoderCell(work, cellName, { stepInputs, stepOutputs }, tparams);
        }
        else
        {
            AddDecoderCell(work, cellName, { stepInputs, stepOutputs, mainCellName }, tparams);
        }

        cellFramesPrediction.push_back(cellName / "frames_prediction");
        if (!tparams.useDurationPrediction)
        {
            cellStopToken.push_back(cellName / "stop_token");
        }

        if (!lastIteration)
        {

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
             *     - encoder_output
             *     - input_lengths: int32 Tensor with shape [N] where N is batch size and values are the lengths
             *      of each sequence in inputs.
             *     - [previous_step_output]: encoder output or output from previous decoding step
             *     - [state]: optional previous state tensors (will be initialized if not provided):
             *       - attention
             *       - alignments
             *       - attention_rnn_state [optional - if tparams.useAttentionRnn == true]
             *         - h
             *         - c
             *       - rnn cell state
             *         - h
             *         - c
             */
            stepInputs = tparams.useDurationPrediction ? 
                Names {
                    mels[time], // teacher_forcing = 1
                    durations[time],
                } :
                Names{
                    encoderOutput,
                    inputLengths,
                    mels[time], // teacher_forcing = 1
                };

            if (tparams.useDurationPrediction)
            {
                std::copy(stepOutputs.begin() + 1, stepOutputs.end(), back_inserter(stepInputs));
            }
            else
            {
                std::copy(stepOutputs.begin() + 2, stepOutputs.end(), back_inserter(stepInputs));
            }

            if (!conditions.empty())
            {
                stepInputs.push_back(name / "condition");
            }

            cellName = name / ("_cell[" + to_string(time + 1) + "]");
        }
    }

    work->add<ConcatenationLayer>(name / "concat_frame_prediction", BasicParamsWithDim{ cellFramesPrediction, { name / "frames_prediction" }, "height" });

    if (!tparams.useDurationPrediction)
    {
        work->add<ConcatenationLayer>(name / "concat_stop_token", BasicParamsWithDim{ cellStopToken, { name / "stop_token" }, "height" });
        work->add<SplitterLayer>(name / "frame_prediction_splitter", BasicParams{ { name / "frames_prediction" }, { name / "frames_prediction" / "1", outputs[0] } });
        work->add<ReshapeLayer>(name / "reshape_frame_prediction", ViewParams{ name / "frames_prediction" / "1", outputs[2], 1, -1, static_cast<int>(tparams.numMels) });
        work->add<ReshapeLayer>(name / "reshape_stop_token", ViewParams{ name / "stop_token", outputs[1], 1, -1, 1 });
    }
    else
    {
        work->add<ReshapeLayer>(name / "reshape_frame_prediction", ViewParams{ name / "frames_prediction", outputs[0], 1, -1, static_cast<int>(tparams.numMels) });
    }
}
}
}
