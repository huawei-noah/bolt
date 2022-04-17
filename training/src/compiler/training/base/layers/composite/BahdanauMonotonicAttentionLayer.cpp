// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "BahdanauMonotonicAttentionLayer.h"
#include "AttentionMaskCreatorLayer.h"
#include "BahdanauMonotonicAttentionInternalLayers.h"

#include <training/base/layers/activations/SigmoidActivation.h>
#include <training/base/layers/activations/SoftMaxActivation.h>
#include <training/base/layers/activations/TanhActivation.h>
#include <training/base/layers/basic/ArgMaxLayer.h>
#include <training/base/layers/basic/ClampLayer.h>
#include <training/base/layers/basic/ConcatenationLayer.h>
#include <training/base/layers/basic/CumSumLayer.h>
#include <training/base/layers/basic/ElementWiseDivLayer.h>
#include <training/base/layers/basic/ElementWiseMulLayer.h>
#include <training/base/layers/basic/ElementWiseSubLayer.h>
#include <training/base/layers/basic/ElementWiseSumLayer.h>
#include <training/base/layers/basic/ExpLayer.h>
#include <training/base/layers/basic/IndexFillLayer.h>
#include <training/base/layers/basic/LogLayer.h>
#include <training/base/layers/basic/RSqrtLayer.h>
#include <training/base/layers/basic/RandomTensorLayer.h>
#include <training/base/layers/basic/ReduceSumLayer.h>
#include <training/base/layers/basic/RollLayer.h>
#include <training/base/layers/basic/ScaleLayer.h>
#include <training/base/layers/basic/SelectLayer.h>
#include <training/base/layers/basic/SlicerLayer.h>
#include <training/base/layers/basic/SquareLayer.h>
#include <training/base/layers/basic/TransposeLayer.h>
#include <training/base/layers/basic/trainable/LinearLayer.h>

using namespace raul;

namespace
{

const raul::dtype CLAMP_MIN1 = 1.1754943508222875e-38_dt;
const raul::dtype CLAMP_MIN2 = 1.0e-10_dt;
const raul::dtype CLAMP_MAX = 1.0_dt;

} // anonymous namespace

namespace raul
{

BahdanauMonotonicAttentionLayer::BahdanauMonotonicAttentionLayer(const Name& name, const BahdanauAttentionParams& params, raul::NetworkParameters& networkParameters)
    : mNumUnits(params.mNumUnits)
    , mNormalize(params.mNormalize)
    , mSigmoidNoise(params.mSigmoidNoise)
    , mScoreBiasInit(params.mScoreBiasInit)
    , mMode(params.mMode)
    , mStepwise(params.mStepwise)
{
    // Query, State, Memory, [MemorySeqLength], [ScoreMaskValues]
    if (params.getInputs().size() < 3 && params.getInputs().size() > 5)
    {
        THROW("BahdanauMonotonicAttentionLayer", name, "wrong number of input names");
    }

    mHasMask = params.getInputs().size() > 3;

    if (params.getOutputs().size() < 1 || params.getOutputs().size() > 3)
    {
        THROW("BahdanauMonotonicAttentionLayer", name, "wrong number of output names");
    }

    // Input names
    auto [queryName, stateName, memoryName] = std::make_tuple(params.getInputs()[0], params.getInputs()[1], params.getInputs()[2]);

    if (!params.getSharedLayer().empty())
    {
        mAttentionVName = params.getSharedLayer() / "attention_v";
        mScoreBiasName = params.getSharedLayer() / "score_bias";
        if (mNormalize)
        {
            mAttentionGName = params.getSharedLayer() / "attention_g";
            mAttentionBName = params.getSharedLayer() / "attention_b";
        }
    }
    else
    {
        raul::Names trainableNames;
        mAttentionVName = name / "attention_v";
        mScoreBiasName = name / "score_bias";

        trainableNames.push_back(mAttentionVName);
        trainableNames.push_back(mScoreBiasName);

        if (mNormalize)
        {
            mAttentionGName = name / "attention_g";
            mAttentionBName = name / "attention_b";

            trainableNames.push_back(mAttentionGName);
            trainableNames.push_back(mAttentionBName);
        }

        networkParameters.mWorkflow.add<bahdanau::BahdanauTrainableInitializerLayer>(
            name / "trainable_part", TrainableParams{ {}, trainableNames, params.frozen }, mNumUnits, mNormalize, mScoreBiasInit);
    }

    // Declare mask for memory and default scores
    bool hasNoise = mSigmoidNoise > 0.0_dt;
    if (mHasMask && params.getSharedLayer().empty())
    {
        networkParameters.mWorkflow.add<AttentionMaskCreatorLayer>(
            name / "create_mask", BasicParams{ { params.getInputs()[3] }, { name / "Mask", name / "DefaultScores" } }, networkParameters.mWorkflow.getHeight(memoryName));
    }

    // Layers

    // Process query from [batch, 1, 1, decoder_output_size] to [batch, 1, 1, mNumUnits]
    networkParameters.mWorkflow.add<LinearLayer>(
        name / "query_layer",
        LinearParams{ { queryName }, { name / "queryProcessed" }, params.getSharedLayer().empty() ? "" : params.getSharedLayer() / "query_layer", mNumUnits, false, params.frozen });

    // Mask memory if needed
    Name keyName = !params.getSharedLayer().empty() ? params.getSharedLayer() / "keys" : name / "keys";
    if (mHasMask)
    {
        if (params.getSharedLayer().empty())
        {
            networkParameters.mWorkflow.add<ElementWiseMulLayer>(name / "mask_memory", ElementWiseLayerParams{ { memoryName, name / "Mask" }, { params.getOutputs()[1] } });
        }
    }

    if (params.getSharedLayer().empty())
    {
        // Process memory from [batch, 1, max_time, encoder_output_size] to key with size [batch, 1, max_time, mNumUnits]
        networkParameters.mWorkflow.add<LinearLayer>(name / "memory_layer", LinearParams{ { mHasMask ? params.getOutputs()[1] : memoryName }, { keyName }, mNumUnits, false, params.frozen });
    }

    // Sum queryProcessed + key
    networkParameters.mWorkflow.add<ElementWiseSumLayer>(name / "sumQK", ElementWiseLayerParams{ { name / "queryProcessed", keyName }, { name / "sum" } });

    if (mNormalize)
    {
        // Add AttentionB
        networkParameters.mWorkflow.add<ElementWiseSumLayer>(name / "sumQKN", ElementWiseLayerParams{ { name / "sum", mAttentionBName }, { name / "sum_b" } });
    }

    // Tanh activation
    networkParameters.mWorkflow.add<TanhActivation>(name / "tanhQK", BasicParams{ { mNormalize ? name / "sum_b" : name / "sum" }, { name / "tanh" } });

    // Multiply mAttentionV/Normed mAttentionV and output of tanh
    if (mNormalize)
    {
        networkParameters.mWorkflow.add<SquareLayer>(name / "squareAttentionV", BasicParams{ { mAttentionVName }, { name / "squareAttentionV" } });
        networkParameters.mWorkflow.add<ReduceSumLayer>(name / "rsumAttentionV", BasicParamsWithDim{ { name / "squareAttentionV" }, { name / "rsumAttentionV" }, Dimension::Width });
        networkParameters.mWorkflow.add<RSqrtLayer>(name / "rsqrtAttentionV", BasicParams{ { name / "rsumAttentionV" }, { name / "rsqrtAttentionV" } });
        networkParameters.mWorkflow.add<ElementWiseMulLayer>(name / "normalizeAttentionV",
                                                             ElementWiseLayerParams{ { mAttentionGName, mAttentionVName, name / "rsqrtAttentionV" }, { name / "normalizedAttentionV" } });
    }

    networkParameters.mWorkflow.add<ElementWiseMulLayer>(name / "mulAttntanhQK",
                                                         ElementWiseLayerParams{ { mNormalize ? name / "normalizedAttentionV" : mAttentionVName, name / "tanh" }, { name / "mul" } });

    // Reduction sum
    networkParameters.mWorkflow.add<ReduceSumLayer>(name / "rSumQKV", BasicParamsWithDim{ { name / "mul" }, { name / "scores" }, Dimension::Width });

    // Add bias to score
    networkParameters.mWorkflow.add<ElementWiseSumLayer>(name / "sumScoreBias", ElementWiseLayerParams{ { name / "scores", mScoreBiasName }, { name / "biasedScores" } });

    auto currName = name / "biasedScores";
    if (mHasMask)
    {
        auto scoresName = params.getInputs().size() == 5 ? params.getInputs()[4] : !params.getSharedLayer().empty() ? params.getSharedLayer() / "DefaultScores" : name / "DefaultScores";
        networkParameters.mWorkflow.add<SelectLayer>(
            name / "selectScores", ElementWiseLayerParams{ { !params.getSharedLayer().empty() ? params.getSharedLayer() / "Mask" : name / "Mask", currName, scoresName }, { name / "maskedScores" } });
        currName = name / "maskedScores";
    }

    // Add noise
    if (hasNoise)
    {
        networkParameters.mWorkflow.add<RandomTensorLayer>(
            name / "createNoise",
            RandomTensorLayerParams{
                { name / "initialNoise" }, networkParameters.mWorkflow.getDepth(currName), networkParameters.mWorkflow.getHeight(currName), networkParameters.mWorkflow.getWidth(currName) });
        networkParameters.mWorkflow.add<ScaleLayer>(name / "increaseNoise", ScaleParams{ { name / "initialNoise" }, { name / "Noise" }, mSigmoidNoise });
        networkParameters.mWorkflow.add<ElementWiseSumLayer>(name / "addNoise", ElementWiseLayerParams{ { name / "Noise", currName }, { name / "noisyScore" } });
        currName = name / "noisyScore";
    }

    // Transpose
    networkParameters.mWorkflow.add<TransposeLayer>(name / "transpose", TransposingParams{ { currName }, { name / "scoresT" }, Dimension::Width, Dimension::Height });

    // Get weights
    networkParameters.mWorkflow.add<SigmoidActivation>(name / "sigmoid", BasicParamsWithDim{ { name / "scoresT" }, { name / "sigmWeights" }, raul::Dimension::Width });

    // Subtract one
    networkParameters.mWorkflow.add<bahdanau::BahdanauConstantsInitializerLayer>(
        name / "createOnesAndZeroes", BasicParams{ {}, { name / "Ones", name / "Zeroes1", name / "Zeroes2" } }, networkParameters.mWorkflow.getHeight(memoryName));
    networkParameters.mWorkflow.add<ElementWiseSubLayer>(name / "decreaseSigmWeights", ElementWiseLayerParams{ { name / "Ones", name / "sigmWeights" }, { name / "reversedWeights" } });

    // Final calculations
    if (mStepwise)
    {
        if (params.mOldSMA)
        {
            networkParameters.mWorkflow.add<ElementWiseMulLayer>(name / "mulStateAndRWeights", ElementWiseLayerParams{ { stateName, name / "reversedWeights" }, { name / "newStateToRoll" } });
            // Shift weights
            networkParameters.mWorkflow.add<RollLayer>(name / "shift", RollLayerParams{ { name / "newStateToRoll" }, { name / "shiftedWeights" }, Dimension::Width, 1 });
            // Change first values
            networkParameters.mWorkflow.add<IndexFillLayer>(name / "fill",
                                                            IndexFillLayerParams{ { name / "shiftedWeights" }, { name / "filledWeights" }, Dimension::Width, { 0 }, mStepwise ? 0.0_dt : 1.0_dt });
            networkParameters.mWorkflow.add<ElementWiseMulLayer>(name / "mulStateAndWeights", ElementWiseLayerParams{ { stateName, name / "sigmWeights" }, { name / "prefinalState" } });
            networkParameters.mWorkflow.add<ElementWiseSumLayer>(name / "calcNewAttn", ElementWiseLayerParams{ { name / "prefinalState", name / "filledWeights" }, { params.getOutputs()[0] } });
        }
        else
        {
            // Calculate stay part
            networkParameters.mWorkflow.add<ElementWiseMulLayer>(name / "calcStayPart", ElementWiseLayerParams{ { stateName, name / "sigmWeights" }, { name / "stayPart" } });
            // Calculate go part
            networkParameters.mWorkflow.add<ElementWiseMulLayer>(name / "calcGoPart", ElementWiseLayerParams{ { stateName, name / "reversedWeights" }, { name / "goPart" } });
            // Slice go part
            networkParameters.mWorkflow.add<SlicerLayer>(
                name / "sliceGoPart",
                SlicingParams{ { name / "goPart" }, { name / "preShifted", name / "preEos" }, Dimension::Width, { static_cast<int>(networkParameters.mWorkflow.getWidth(name / "goPart")) - 1, 1 } });
            networkParameters.mWorkflow.add<ConcatenationLayer>(name / "computeShifted", BasicParamsWithDim{ { name / "Zeroes1", name / "preShifted" }, { name / "shifted" }, Dimension::Width });
            networkParameters.mWorkflow.add<ConcatenationLayer>(name / "computeEos", BasicParamsWithDim{ { name / "Zeroes2", name / "preEos" }, { name / "eos" }, Dimension::Width });
            // Some these parts
            networkParameters.mWorkflow.add<ElementWiseSumLayer>(name / "calcFinalGoPart", ElementWiseLayerParams{ { name / "shifted", name / "eos" }, { name / "finalGoPart" } });
            // Calculate final attention as sum of stay and go parts
            networkParameters.mWorkflow.add<ElementWiseSumLayer>(name / "calcNewAttn", ElementWiseLayerParams{ { name / "stayPart", name / "finalGoPart" }, { params.getOutputs()[0] } });
        }

        // Indices if needed
        if ((params.getSharedLayer().empty() && params.getOutputs().size() == 3) || (!params.getSharedLayer().empty() && params.getOutputs().size() == 2))
        {
            // Don't need to declare temporary gradients for params.getOutputs().back()
            // because ArgMaxLayer doesn't create nabla-tensor and doesn't calculate gradient
            // if only indices are required
            networkParameters.mWorkflow.add<ArgMaxLayer>(name / "maxAttn", BasicParamsWithDim{ { params.getOutputs()[0] }, { params.getOutputs().back() }, Dimension::Width });
        }
    }
    else
    {
        // Shift weights
        networkParameters.mWorkflow.add<RollLayer>(name / "shift", RollLayerParams{ { name / "reversedWeights" }, { name / "shiftedWeights" }, Dimension::Width, 1 });

        // Change first values
        networkParameters.mWorkflow.add<IndexFillLayer>(name / "fill", IndexFillLayerParams{ { name / "shiftedWeights" }, { name / "filledWeights" }, Dimension::Width, { 0 }, 1.0_dt });

        // Clamp current result
        networkParameters.mWorkflow.add<ClampLayer>(name / "clamp", ClampLayerParams{ { name / "filledWeights" }, { name / "clampedResult" }, CLAMP_MIN1, CLAMP_MAX });

        // Log
        networkParameters.mWorkflow.add<LogLayer>(name / "log", BasicParams{ { name / "clampedResult" }, { name / "logedResult" } });

        // Cumulative sum
        networkParameters.mWorkflow.add<CumSumLayer>(name / "csum1", BasicParamsWithDim{ { name / "logedResult" }, { name / "cumulativeResult" }, raul::Dimension::Width });

        // Exp
        networkParameters.mWorkflow.add<ExpLayer>(name / "exp", BasicParams{ { name / "cumulativeResult" }, { name / "expResult" } });

        // Clamp current result
        networkParameters.mWorkflow.add<ClampLayer>(name / "clamp2", ClampLayerParams{ { name / "expResult" }, { name / "clampedExpResult" }, CLAMP_MIN2, CLAMP_MAX });

        // Divide prev state
        networkParameters.mWorkflow.add<ElementWiseDivLayer>(name / "div", ElementWiseLayerParams{ { stateName, name / "clampedExpResult" }, { name / "normalizedOldState" } });

        // Cumulative sum of the result
        networkParameters.mWorkflow.add<CumSumLayer>(name / "csum2", BasicParamsWithDim{ { name / "normalizedOldState" }, { name / "summedOldStateResult" }, raul::Dimension::Width });

        // Final result
        networkParameters.mWorkflow.add<ElementWiseMulLayer>(name / "mulFinal",
                                                             ElementWiseLayerParams{ { name / "sigmWeights", name / "expResult", name / "summedOldStateResult" }, { params.getOutputs()[0] } });
    }
}

}