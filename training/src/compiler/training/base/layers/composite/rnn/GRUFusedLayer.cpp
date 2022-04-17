// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "GRUFusedLayer.h"

#include <training/base/impl/composite/rnn/GRUFusedLayerCPU.h>

namespace raul
{

GRUFusedLayer::GRUFusedLayer(const Name& name, const GRUParams& params, const Name& basicName, const Name& nameHiddenStateIn, NetworkParameters& networkParameters)
    : TrainableLayer(name, "GRUFusedLayer", params, networkParameters, { true, false })
    , mIsExternalState(!params.mHiddenFeatures)
    , mLengthSequence(0)
    , mSequenceDimension("depth")
    , mUseBiasForInput(params.mUseBiasForInput)
    , mUseBiasForHidden(params.mUseBiasForHidden)
    , mCurrentInputMaxSize(0)
{
    auto prefix = mTypeName + "[" + mName + "::ctor]: ";

    const auto nameInput = params.getInputs()[0];

    if (networkParameters.mWorkflow.getDepth(nameInput) != 1u && networkParameters.mWorkflow.getHeight(nameInput) != 1u)
    {
        throw std::runtime_error("GRUFusedLayer[" + name + "::ctor]: length of sequence should be placed in one dimension: depth or height");
    }

    if (networkParameters.mWorkflow.getDepth(nameInput) == 1u)
    {
        mLengthSequence = networkParameters.mWorkflow.getHeight(nameInput);
        mSequenceDimension = "height";
    }
    else
    {
        mLengthSequence = networkParameters.mWorkflow.getDepth(nameInput);
    }

    if (mLengthSequence == 0)
    {
        throw std::runtime_error("GRUFusedLayer[" + name + "::ctor]: length of sequence cannot be zero");
    }

    if (mIsExternalState)
    {
        THROW("GRUFusedLayer", name, " external state not supported");
    }

    mDirection = BasicParamsWithDim({}, {}, mSequenceDimension).dim;

    if (!mNetworkParams.mWorkflow.isCompilerEnabled())
    {
        DECLARE_IMPL(GRUFusedLayer, GRUFusedLayerCPU<MemoryManager>, GRUFusedLayerCPU<MemoryManagerFP16>)
    }

    size_t cnt = 0;

    Names nameInputSlices(mLengthSequence);
    Names nameOutputSlices(mIsExternalState ? mLengthSequence - 1 : mLengthSequence);
    std::generate_n(nameInputSlices.begin(), mLengthSequence, [&, cnt]() mutable { return name / nameInput + "[" + Conversions::toString(cnt++) + "]"; });

    std::generate_n(nameOutputSlices.begin(), mLengthSequence, [&, cnt]() mutable { return nameHiddenStateIn + "[" + Conversions::toString(cnt++) + "]"; });

    auto nameUnrolledCellZero = basicName / "cell";
    const Names nameUnrolledCellWeights = {
        nameUnrolledCellZero / "linear_ih::Weights", nameUnrolledCellZero / "linear_ih::Biases", nameUnrolledCellZero / "linear_hh::Weights", nameUnrolledCellZero / "linear_hh::Biases"
    };

    mWeightsNameIH = nameUnrolledCellWeights[0];
    mBiasesNameIH = nameUnrolledCellWeights[1];
    mWeightsNameHH = nameUnrolledCellWeights[2];
    mBiasesNameHH = nameUnrolledCellWeights[3];

    Name nameHiddenStateInCopy = nameHiddenStateIn;

    mLinearIHTmp.resize(mLengthSequence);
    mLinearHHTmp.resize(mLengthSequence);

    mInputsLocal.resize(mLengthSequence);
    mOutputsLocal.resize(mLengthSequence);

    mNetworkParams.mWorkflow.copyDeclaration(mName, nameInput, Workflow::Usage::ForwardAndBackward, Workflow::Mode::Read);
    mNetworkParams.mWorkflow.copyDeclaration(mName, nameInput, nameInput.grad(), DEC_BACK_WRIT_ZERO);

    for (size_t i = 0; i < mLengthSequence; ++i)
    {
        const auto idx = Conversions::toString(i);
        std::string nameHiddenStateOut = nameHiddenStateIn + "[" + idx + "]";
        auto nameUnrolledCell = i == 0 ? nameUnrolledCellZero : (basicName / "unrolled_cell[" + idx + "]");

        Name input;
        Name inputHidden;
        Name outputHidden;

        auto inputIdx = i;
        if (i == mLengthSequence - 1)
        {
            input = nameInputSlices[inputIdx];
            inputHidden = nameHiddenStateInCopy;
            outputHidden = nameHiddenStateOut;
        }
        else
        {
            input = nameInputSlices[inputIdx];
            inputHidden = nameHiddenStateInCopy;
            outputHidden = nameHiddenStateOut;

            nameHiddenStateInCopy = nameHiddenStateOut;
        }

        mLinearIHTmp[i] = nameUnrolledCell / "linearIH";
        mLinearHHTmp[i] = nameUnrolledCell / "linearHH";

        mInputsLocal[i] = { input, inputHidden };
        mOutputsLocal[i] = outputHidden;

        const auto parts = 3U;

        const auto sizeHidden = networkParameters.mWorkflow.getWidth(inputHidden);

        mOutputsCount = sizeHidden * parts;

        // Declare inputs
        if (i == 0) // avoid conflicts with outputs
        {
            mNetworkParams.mWorkflow.copyDeclaration(name, inputHidden, raul::Workflow::Usage::Forward, raul::Workflow::Mode::Write);
            mNetworkParams.mWorkflow.copyDeclaration(name, inputHidden, raul::Workflow::Usage::Backward, raul::Workflow::Mode::Read);
            mNetworkParams.mWorkflow.copyDeclaration(name, inputHidden, inputHidden.grad(), DEC_BACK_WRIT_ZERO);
        }

        // Declare temporal storages
        mNetworkParams.mWorkflow.tensorNeeded(mName, mLinearIHTmp[i], WShape{ BS(), 1u, 1u, mOutputsCount }, DEC_FORW_WRIT);
        mNetworkParams.mWorkflow.copyDeclaration(name, mLinearIHTmp[i], mLinearIHTmp[i], DEC_BACK_READ);
        mNetworkParams.mWorkflow.copyDeclaration(name, mLinearIHTmp[i], mLinearHHTmp[i], DEC_FORW_WRIT);
        mNetworkParams.mWorkflow.copyDeclaration(name, mLinearHHTmp[i], mLinearHHTmp[i], DEC_BACK_READ);
        // And their grads
        mNetworkParams.mWorkflow.copyDeclaration(name, mLinearIHTmp[i], mLinearIHTmp[i].grad(), DEC_BACK_WRIT_ZERO);
        mNetworkParams.mWorkflow.copyDeclaration(name, mLinearHHTmp[i], mLinearHHTmp[i].grad(), DEC_BACK_WRIT_ZERO);

        // Declare output
        mNetworkParams.mWorkflow.copyDeclaration(name, inputHidden, outputHidden, DEC_FORW_WRIT);
        mNetworkParams.mWorkflow.copyDeclaration(name, outputHidden, outputHidden, DEC_BACK_READ);
        mNetworkParams.mWorkflow.copyDeclaration(name, outputHidden, outputHidden.grad(), DEC_BACK_READ);
    }

    // Declare trainable params
    if (mSharedLayer.empty() && mSharedWeights.empty())
    {
        mNetworkParams.mWorkflow.tensorNeeded(mName, mWeightsNameIH, WShape{ 1u, 1u, mOutputsCount, mNetworkParams.mWorkflow.getWidth(nameInput) }, DEC_TRAINABLE);
        mNetworkParams.mWorkflow.tensorNeeded(mName, mWeightsNameHH, WShape{ 1u, 1u, mOutputsCount, mNetworkParams.mWorkflow.getWidth(mInputsLocal[0][1]) }, DEC_TRAINABLE);
        if (mUseBiasForInput)
        {
            mNetworkParams.mWorkflow.tensorNeeded(mName, mBiasesNameIH, WShape{ 1u, 1u, 1u, mOutputsCount }, DEC_TRAINABLE);
        }
        if (mUseBiasForHidden)
        {
            mNetworkParams.mWorkflow.tensorNeeded(mName, mBiasesNameHH, WShape{ 1u, 1u, 1u, mOutputsCount }, DEC_TRAINABLE);
        }
    }
    else
    {
        if (!mSharedWeights.empty())
        {
            mWeightsNameIH = mSharedWeights[0];
            mBiasesNameIH = mSharedWeights[1];
            mWeightsNameHH = mSharedWeights[2];
            mBiasesNameHH = mSharedWeights[3];
        }
        else
        {
            mWeightsNameIH = Name(mSharedLayer + "_ih") / "Weights";
            mWeightsNameHH = Name(mSharedLayer + "_hh") / "Weights";
            mBiasesNameIH = Name(mSharedLayer + "_ih") / "Biases";
            mBiasesNameHH = Name(mSharedLayer + "_hh") / "Biases";
        }
    }

    if (!mFrozen)
    {
        mNetworkParams.mWorkflow.copyDeclaration(mName, mWeightsNameIH, mWeightsNameIH.grad(), DEC_TRAINABLE_GRAD);
        mNetworkParams.mWorkflow.copyDeclaration(mName, mWeightsNameHH, mWeightsNameHH.grad(), DEC_TRAINABLE_GRAD);

        if (mUseBiasForInput)
        {
            mNetworkParams.mWorkflow.copyDeclaration(mName, mBiasesNameIH, mBiasesNameIH.grad(), DEC_TRAINABLE_GRAD);
        }
        if (mUseBiasForHidden)
        {
            mNetworkParams.mWorkflow.copyDeclaration(mName, mBiasesNameHH, mBiasesNameHH.grad(), DEC_TRAINABLE_GRAD);
        }
    }

    // concat output
    yato::dimensionality<3U, size_t> shape(
        mNetworkParams.mWorkflow.getDepth(mOutputsLocal[0]), mNetworkParams.mWorkflow.getHeight(mOutputsLocal[0]), mNetworkParams.mWorkflow.getWidth(mOutputsLocal[0]));

    switch (mDirection)
    {
        case Dimension::Depth:
            mDimIndex = 0;
            break;
        case Dimension::Height:
            mDimIndex = 1;
            break;
        case Dimension::Width:
            mDimIndex = 2;
            break;
        default:
            THROW(mTypeName, mName, "unsupported dim");
    }

    for (size_t i = 1; i < mOutputsLocal.size(); ++i)
    {
        yato::dimensionality<3U, size_t> inputShape(
            mNetworkParams.mWorkflow.getDepth(mOutputsLocal[i]), mNetworkParams.mWorkflow.getHeight(mOutputsLocal[i]), mNetworkParams.mWorkflow.getWidth(mOutputsLocal[i]));
        mCurrentInputMaxSize = std::max(mCurrentInputMaxSize, inputShape[0] * inputShape[1] * inputShape[2]);

        for (size_t k = 0; k < 3; ++k)
        {
            if (k == mDimIndex)
            {
                continue;
            }
            if (shape[k] != inputShape[k])
            {
                THROW(mTypeName, mName, "inconsistent input shapes (" + mOutputs[0] + " " + Conversions::toString(shape) + " vs " + mOutputsLocal[i] + " " + Conversions::toString(inputShape) + ")");
            }
        }
        shape[mDimIndex] += inputShape[mDimIndex];
    }

    mNetworkParams.mWorkflow.tensorNeeded(mName, mOutputs[0], WShape{ BS(), shape[0], shape[1], shape[2] }, DEC_FORW_WRIT);
    mNetworkParams.mWorkflow.copyDeclaration(mName, mOutputs[0], mOutputs[0].grad(), DEC_BACK_READ);
}

} // namespace raul