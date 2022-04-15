// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "TacotronModel.h"

#include <algorithm>

#include <training/api/API.h>
#include <training/layers/basic/BatchExpanderLayer.h>
#include <training/layers/basic/ConcatenationLayer.h>
#include <training/layers/basic/ElementWiseSumLayer.h>
#include <training/layers/basic/NonZeroMaskLayer.h>
#include <training/layers/basic/SplitterLayer.h>
#include <training/layers/basic/TensorLayer.h>
#include <training/layers/basic/TileLayer.h>
#include <training/layers/basic/trainable/LinearLayer.h>

#include "TacotronDecoder.h"
#include "tacotron/DurationPredictor.h"
#include "tacotron/PostNet.h"

namespace raul
{
void AddSingleSpeakerFslTacotronModel(Workflow* work, const Name& name, const TacotronParams& tparams)
{
    using namespace std;

    auto prefix = "TacotronModel::AddSingleSpeakerFslTacotronModel[" + name + "]: ";
    const auto& inputs = tparams.getInputs();
    const auto& outputs = tparams.getOutputs();

    size_t inputsCount = 3;
    if (!tparams.trainableSpeakerEmbedding)
    {
        ++inputsCount;
    }
    if (tparams.useResidualEncoder)
    {
        ++inputsCount;
    }
    if (tparams.useLanguageEmbedding)
    {
        ++inputsCount;
    }
    if (tparams.useDurationPrediction)
    {
        inputsCount += 2;
    }

    if (inputs.size() != inputsCount)
    {
        THROW("TacotronModel", name, "incorrect number of inputs");
    }
    if (outputs.size() != 3)
    {
        THROW("TacotronModel", name, "incorrect number of outputs");
    }

    const auto encoderOutput = inputs[0];
    const auto inputLengths = *(inputs.end() - 2);
    const auto melTargets = inputs.back();

    const auto decoderOutput = outputs[0];
    const auto melOutputs = outputs[1];
    const auto stopTokenPredictions = tparams.useDurationPrediction ? Name() : outputs[2];
    const auto durations = tparams.useDurationPrediction ? outputs[2] : Name();

    bool mFrozen = false;
    Name speakerEmbedding = tparams.trainableSpeakerEmbedding ? name / "speaker_embedding" : inputs[1];
    Name finalEncoderOutputName = encoderOutput;
    size_t pos = tparams.trainableSpeakerEmbedding ? 1 : 0;
    // non-trainable
    Name encodedResidual = tparams.useResidualEncoder ? inputs[1 + pos] : Name();
    Name languageEmbedding = tparams.useLanguageEmbedding ? (tparams.useResidualEncoder ? inputs[2 + pos] : inputs[1 + pos]) : Name();
    Name durationTargets = tparams.useDurationPrediction ? *(inputs.end() - 4) : Name();
    Name encoderInputs = tparams.useDurationPrediction ? *(inputs.end() - 3) : Name();

    Names conditions;
    if (tparams.useResidualEncoder)
    {
        conditions.push_back(encodedResidual);
    }

    if (tparams.trainableSpeakerEmbedding)
    {
        work->add<TensorLayer>(speakerEmbedding, TensorParams{ Names{ speakerEmbedding }, WShape{ 1u, 1u, 1u, tparams.speakerEmbeddingSize }, DEC_TRAINABLE });
        if (!mFrozen)
        {
            work->copyDeclaration(speakerEmbedding, speakerEmbedding, speakerEmbedding.grad(), DEC_TRAINABLE_GRAD);
        }
    }

    Name speakerEmbeddingWithBS = speakerEmbedding;
    if (!work->getShape(speakerEmbedding).isBSDependent())
    {
        speakerEmbeddingWithBS = name / "tileable_embed_targets";
        work->add<BatchExpanderLayer>(name / "tileable_embed_targets", ViewParams{ speakerEmbedding, speakerEmbeddingWithBS });
        conditions.push_back(speakerEmbeddingWithBS);
    }

    if (tparams.useLanguageEmbedding)
    {
        conditions.push_back(languageEmbedding);
    }

    // ENCODER: speaker embedding
    if (tparams.concatConditionsWithEncoderOutput)
    {
        if (tparams.useLanguageEmbedding || tparams.useResidualEncoder)
        {
            THROW("TacotronModel", name, "language embedding and residual decoder not supported with concatConditionsWithEncoderOutput");
        }

        work->add<TileLayer>(name / "tiled_embed_targets", TilingParams{ speakerEmbeddingWithBS, name / "tiled_embed_targets", work->getHeight(encoderOutput), "height" });
        work->add<ConcatenationLayer>(name / "encoder_fin_outputs", BasicParamsWithDim{ { encoderOutput, name / "tiled_embed_targets" }, { name / "embedded_encoder_output" }, "width" });
        finalEncoderOutputName = name / "embedded_encoder_output";
    }

    if (tparams.useDurationPrediction)
    {
        work->add<NonZeroMaskLayer>(name / "input_mask", BasicParams{ { encoderInputs }, { name / "input_mask" } });
        tacotron::AddDurationPredictor(work,
                                       name / "duration_predictor",
                                       BasicParams{ { finalEncoderOutputName, inputLengths, durationTargets, name / "input_mask" },
                                                    { name / "durations_output", durations, name / "upsampled_output", name / "positional_embeddings" } },
                                       tparams);
        finalEncoderOutputName = name / "durations_output";
    }

    Names decoderInputs;
    decoderInputs.push_back(finalEncoderOutputName);
    if (!tparams.concatConditionsWithEncoderOutput)
    {
        copy(conditions.begin(), conditions.end(), back_inserter(decoderInputs));
    }
    decoderInputs.push_back(inputLengths);
    decoderInputs.push_back(melTargets);

    Names decoderOutputs =
        tparams.useDurationPrediction ? Names{ name / "decoder_output" } : Names{ name / "frames_prediction", stopTokenPredictions, name / "decoder_output", name / "final_alignments" };

    tacotron::AddDecoder(work, name / "decoder", BasicParams{ decoderInputs, decoderOutputs }, tparams);

    work->add<SplitterLayer>(name / "decoder_splitter", BasicParams{ { name / "decoder_output" }, { name / "decoder_output" / "1", name / "decoder_output" / "2", decoderOutput } });

    // postprocessing

    tacotron::AddPostNet(work, name / "postnet_convolutions", BasicParams{ { name / "decoder_output" / "1" }, { name / "residual" } }, tparams);
    /*if (tparams.useDurationPrediction)
    {
        work->add<LinearLayer>(name / "postnet_projection", LinearParams{ name / "residual", name / "projected_residual2", tparams.numMels, true, tparams.frozen });
        tacotron::AddPostNet(work, name / "postnet_convolutions" / "1", BasicParams{ { name / "projected_residual2" }, { name / "residual2" }, name / "postnet_convolutions" }, tparams);
        work->add<LinearLayer>(name / "postnet_projection" / "1", LinearParams{name / "residual2", name / "projected_residual", name / "postnet_projection", tparams.numMels, true, tparams.frozen });
    }
    else*/
    {
        work->add<LinearLayer>(name / "postnet_projection", LinearParams{ name / "residual", name / "projected_residual", tparams.numMels, tparams.getLayerExecutionTarget(), true, tparams.frozen });
    }
    work->add<ElementWiseSumLayer>(name / "mel_outputs_sum", ElementWiseLayerParams{ { name / "decoder_output" / "2", name / "projected_residual" }, { melOutputs } });
}
}
