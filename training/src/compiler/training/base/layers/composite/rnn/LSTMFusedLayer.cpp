// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "LSTMFusedLayer.h"

#include <training/base/impl/composite/rnn/LSTMFusedLayerCPU.h>

namespace raul
{

LSTMFusedLayer::LSTMFusedLayer(const Name& name, const LSTMParams& params, const Name& basicName, const Name& nameHiddenStateIn, const Name& nameCellStateIn, NetworkParameters& networkParameters)
    : TrainableLayer(name, "LSTMFusedLayer", params, networkParameters, { true, false })
    , mIsExternalState(!params.mHiddenFeatures)
    , mLengthSequence(0)
    , mSequenceDimension("depth")
    , mUseBias(params.mBias)
    , mZoneout(params.mZoneout)
    , mUseSingleParamTensor(params.mUseSingleParamTensor)
    , mForgetBias(params.mForgetBias)
    , mCurrentInputMaxSize(0)
{
    auto prefix = mTypeName + "[" + mName + "::ctor]: ";

    const auto nameInput = params.getInputs()[0];

    if (networkParameters.mWorkflow.getDepth(nameInput) != 1u && networkParameters.mWorkflow.getHeight(nameInput) != 1u)
    {
        THROW("LSTMFusedLayer", name, " length of sequence should be placed in one dimension: depth or height");
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
        THROW("LSTMFusedLayer", name, " length of sequence cannot be zero");
    }

    if (mIsExternalState)
    {
        THROW("LSTMFusedLayer", name, " external state not supported");
    }

    mDirection = BasicParamsWithDim({}, {}, mSequenceDimension).dim;

    if (!mNetworkParams.mWorkflow.isCompilerEnabled())
    {
        DECLARE_IMPL(LSTMFusedLayer, LSTMFusedLayerCPU<MemoryManager>, LSTMFusedLayerCPU<MemoryManagerFP16>)
    }

    size_t cnt = 0;

    Names nameInputSlices(mLengthSequence);
    Names nameOutputSlices(mIsExternalState ? mLengthSequence - 1 : mLengthSequence);
    std::generate_n(nameInputSlices.begin(), mLengthSequence, [&, cnt]() mutable { return basicName / nameInput + "[" + Conversions::toString(cnt++) + "]"; });

    std::generate_n(nameOutputSlices.begin(), mLengthSequence, [&, cnt]() mutable { return nameHiddenStateIn + "[" + Conversions::toString(cnt++) + "]"; });

    auto nameUnrolledCellZero = basicName / "cell";

    mTmpCalculationsName.resize(mLengthSequence);
    mGatesName.resize(mLengthSequence);
    mRandomNameHidden.resize(mLengthSequence);
    mRandomNameCell.resize(mLengthSequence);
    mNoZoneoutNewCellName.resize(mLengthSequence);
    mNoZoneoutNewHiddenGradName.resize(mLengthSequence);
    mNoZoneoutNewCellGradName.resize(mLengthSequence);

    mInputsLocal.resize(mLengthSequence);
    mOutputsLocal.resize(mLengthSequence);

    Name nameHiddenStateInCopy = nameHiddenStateIn;
    Name nameCellStateInCopy = nameCellStateIn;

    mNetworkParams.mWorkflow.copyDeclaration(mName, nameInput, Workflow::Usage::ForwardAndBackward, Workflow::Mode::Read);
    mNetworkParams.mWorkflow.copyDeclaration(mName, nameInput, nameInput.grad(), DEC_BACK_WRIT_ZERO);

    for (size_t q = 0; q < mLengthSequence; ++q)
    {
        const auto idx = Conversions::toString(q);
        auto nameUnrolledCell = q == 0 ? nameUnrolledCellZero : (basicName / "unrolled_cell[" + idx + "]");
        std::string nameHiddenStateOut = nameHiddenStateIn + "[" + idx + "]";
        std::string nameCellStateOut = nameCellStateIn + "[" + idx + "]";

        Name input;
        Name inputHidden;
        Name inputCell;
        Name outputHidden;
        Name outputCell;

        auto inputIdx = q;
        if (q == mLengthSequence - 1)
        {
            input = nameInputSlices[inputIdx];
            inputHidden = nameHiddenStateInCopy;
            inputCell = nameCellStateInCopy;
            outputHidden = nameHiddenStateOut;
            outputCell = nameCellStateOut;
        }
        else
        {
            input = nameInputSlices[inputIdx];
            inputHidden = nameHiddenStateInCopy;
            inputCell = nameCellStateInCopy;
            outputHidden = nameHiddenStateOut;
            outputCell = nameCellStateOut;

            nameHiddenStateInCopy = nameHiddenStateOut;
            nameCellStateInCopy = nameCellStateOut;
        }

        mTmpCalculationsName[q] = nameUnrolledCell / "tmp";
        mGatesName[q] = nameUnrolledCell / "gates";
        mRandomNameHidden[q] = nameUnrolledCell / "randomH";
        mRandomNameCell[q] = nameUnrolledCell / "randomC";
        mNoZoneoutNewCellName[q] = nameUnrolledCell / "noZoneoutNewCell";
        mNoZoneoutNewHiddenGradName[q] = nameUnrolledCell / "noZoneoutNewHiddenGrad";
        mNoZoneoutNewCellGradName[q] = nameUnrolledCell / "noZoneoutNewCellGrad";

        const auto parts = 4U;
        mInputsLocal[q] = { input, inputHidden, inputCell };
        mOutputsLocal[q] = { outputHidden, outputCell };

        const auto sizeHidden = networkParameters.mWorkflow.getWidth(mInputsLocal[q][1]);

        mOutputsCount = sizeHidden * parts;

        // Declare inputs
        for (size_t i = 1; i < mInputsLocal[q].size(); ++i) // skip data
        {
            if (q == 0 || i == 0) // avoid conflicts with outputs
                mNetworkParams.mWorkflow.copyDeclaration(mName, mInputsLocal[q][i], Workflow::Usage::Forward, Workflow::Mode::Write);

            if (q == 0 || mUseZoneout || i == 0 || i == 1) // avoid conflicts with outputs
                mNetworkParams.mWorkflow.copyDeclaration(mName, mInputsLocal[q][i], Workflow::Usage::Backward, Workflow::Mode::Read);

            if (q == 0 || i == 0) // avoid conflicts with outputs
                mNetworkParams.mWorkflow.copyDeclaration(mName, mInputsLocal[q][i], mInputsLocal[q][i].grad(), DEC_BACK_WRIT_ZERO);
        }

        // Declare temporal storages
        if (!mUseSingleParamTensor)
        {
            mNetworkParams.mWorkflow.tensorNeeded(mName, mTmpCalculationsName[q], WShape{ BS(), 1u, 1u, mOutputsCount }, DEC_FORW_WRIT);
        }
        else
        {
            mNetworkParams.mWorkflow.tensorNeeded(
                mName, mTmpCalculationsName[q], WShape{ BS(), 1u, 1u, mNetworkParams.mWorkflow.getWidth(nameInput) + mNetworkParams.mWorkflow.getWidth(mInputsLocal[q][1]) }, DEC_FORW_WRIT);
            mNetworkParams.mWorkflow.copyDeclaration(mName, mTmpCalculationsName[q], mTmpCalculationsName[q], DEC_BACK_READ);
            mNetworkParams.mWorkflow.copyDeclaration(mName, mTmpCalculationsName[q], mTmpCalculationsName[q].grad(), DEC_BACK_WRIT_ZERO);
        }

        mNetworkParams.mWorkflow.tensorNeeded(mName, mGatesName[q], WShape{ BS(), 1u, 1u, mOutputsCount }, DEC_FORW_WRIT);
        mNetworkParams.mWorkflow.copyDeclaration(mName, mGatesName[q], mGatesName[q], DEC_BACK_READ);
        // And their grads
        mNetworkParams.mWorkflow.copyDeclaration(mName, mGatesName[q], mGatesName[q].grad(), DEC_BACK_WRIT_ZERO);

        mUseZoneout = mZoneout != 0.0_dt;

        // Declare outputs
        for (size_t i = 0; i < mOutputsLocal[q].size(); ++i)
        {
            mNetworkParams.mWorkflow.copyDeclaration(mName, mInputsLocal[q][i + 1], mOutputsLocal[q][i], DEC_FORW_WRIT);
            if (i == 1)
            {
                if (mUseZoneout)
                {
                    mNetworkParams.mWorkflow.copyDeclaration(mName, mOutputsLocal[q][i], mNoZoneoutNewCellName[q], DEC_FORW_WRIT);
                    mNetworkParams.mWorkflow.copyDeclaration(mName, mNoZoneoutNewCellName[q], mNoZoneoutNewCellName[q], DEC_BACK_READ);
                }
                else
                {
                    mNetworkParams.mWorkflow.copyDeclaration(mName, mOutputsLocal[q][i], mOutputsLocal[q][i], DEC_BACK_READ);
                }
            }
            mNetworkParams.mWorkflow.copyDeclaration(mName, mOutputsLocal[q][i], mOutputsLocal[q][i].grad(), DEC_BACK_READ);
        }

        // Zoneout part
        if (mUseZoneout)
        {
            mNetworkParams.mWorkflow.copyDeclaration(mName, mOutputsLocal[q][0], mRandomNameHidden[q], DEC_FORW_WRIT);
            mNetworkParams.mWorkflow.copyDeclaration(mName, mOutputsLocal[q][1], mRandomNameCell[q], DEC_FORW_WRIT);
            mNetworkParams.mWorkflow.copyDeclaration(mName, mOutputsLocal[q][0], mRandomNameHidden[q], DEC_BACK_READ);
            mNetworkParams.mWorkflow.copyDeclaration(mName, mOutputsLocal[q][1], mRandomNameCell[q], DEC_BACK_READ);

            // Declare additional grads storages
            mNetworkParams.mWorkflow.copyDeclaration(mName, mOutputsLocal[q][0].grad(), mNoZoneoutNewHiddenGradName[q], DEC_BACK_WRIT_ZERO);
            mNetworkParams.mWorkflow.copyDeclaration(mName, mOutputsLocal[q][1].grad(), mNoZoneoutNewCellGradName[q], DEC_BACK_WRIT_ZERO);
        }
    }

    // Declare trainable params
    const auto baseName = !mSharedLayer.empty() ? mSharedLayer : nameUnrolledCellZero;
    if (mUseSingleParamTensor)
    {
        mTrainableParamsNames = { baseName / "linear" / "Weights", baseName / "linear" / "Biases" };
    }
    else
    {
        mTrainableParamsNames = { baseName / "linear_ih" / "Weights", baseName / "linear_ih" / "Biases", baseName / "linear_hh" / "Weights", baseName / "linear_hh" / "Biases" };
    }

    if (mSharedLayer.empty())
    {
        if (mUseSingleParamTensor)
        {
            mNetworkParams.mWorkflow.tensorNeeded(
                mName, mTrainableParamsNames[0], WShape{ 1u, 1u, mOutputsCount, mNetworkParams.mWorkflow.getWidth(nameInput) + mNetworkParams.mWorkflow.getWidth(mInputsLocal[0][1]) }, DEC_TRAINABLE);
            if (mUseBias)
            {
                mNetworkParams.mWorkflow.tensorNeeded(mName, mTrainableParamsNames[1], WShape{ 1u, 1u, 1u, mOutputsCount }, DEC_TRAINABLE);
            }
        }
        else
        {
            mNetworkParams.mWorkflow.tensorNeeded(mName, mTrainableParamsNames[0], WShape{ 1u, 1u, mOutputsCount, mNetworkParams.mWorkflow.getWidth(nameInput) }, DEC_TRAINABLE);
            mNetworkParams.mWorkflow.tensorNeeded(mName, mTrainableParamsNames[2], WShape{ 1u, 1u, mOutputsCount, mNetworkParams.mWorkflow.getWidth(mInputsLocal[0][1]) }, DEC_TRAINABLE);
            if (mUseBias)
            {
                mNetworkParams.mWorkflow.tensorNeeded(mName, mTrainableParamsNames[1], WShape{ 1u, 1u, 1u, mOutputsCount }, DEC_TRAINABLE);
                mNetworkParams.mWorkflow.tensorNeeded(mName, mTrainableParamsNames[3], WShape{ 1u, 1u, 1u, mOutputsCount }, DEC_TRAINABLE);
            }
        }
    }

    if (!mFrozen)
    {
        const size_t numOfParams = mUseSingleParamTensor ? (mUseBias ? 2 : 1) : (mUseBias ? 4 : 2);
        for (size_t i = 0; i < numOfParams; ++i)
        {
            mNetworkParams.mWorkflow.copyDeclaration(mName, mTrainableParamsNames[i], mTrainableParamsNames[i].grad(), DEC_TRAINABLE_GRAD);
        }
    }

    // concat output
    yato::dimensionality<3U, size_t> shape(
        mNetworkParams.mWorkflow.getDepth(mOutputsLocal[0][0]), mNetworkParams.mWorkflow.getHeight(mOutputsLocal[0][0]), mNetworkParams.mWorkflow.getWidth(mOutputsLocal[0][0]));

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
            mNetworkParams.mWorkflow.getDepth(mOutputsLocal[i][0]), mNetworkParams.mWorkflow.getHeight(mOutputsLocal[i][0]), mNetworkParams.mWorkflow.getWidth(mOutputsLocal[i][0]));
        mCurrentInputMaxSize = std::max(mCurrentInputMaxSize, inputShape[0] * inputShape[1] * inputShape[2]);

        for (size_t k = 0; k < 3; ++k)
        {
            if (k == mDimIndex)
            {
                continue;
            }
            if (shape[k] != inputShape[k])
            {
                THROW(
                    mTypeName, mName, "inconsistent input shapes (" + mOutputs[0] + " " + Conversions::toString(shape) + " vs " + mOutputsLocal[i][i] + " " + Conversions::toString(inputShape) + ")");
            }
        }
        shape[mDimIndex] += inputShape[mDimIndex];
    }

    mNetworkParams.mWorkflow.tensorNeeded(mName, mOutputs[0], WShape{ BS(), shape[0], shape[1], shape[2] }, DEC_FORW_WRIT);
    mNetworkParams.mWorkflow.copyDeclaration(mName, mOutputs[0], mOutputs[0].grad(), DEC_BACK_READ);
}

} // namespace raul