// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "LSTMFusedGatesCalcLayer.h"

#include "impl/LSTMFusedGatesCalcLayerCPU.h"

#include <training/base/common/MemoryManager.h>

namespace raul
{

LSTMFusedGatesCalcLayer::LSTMFusedGatesCalcLayer(const Name& name, const LSTMFusedGatesCalcParams& params, NetworkParameters& networkParameters)
    : TrainableLayer(name, "LSTMFusedGatesCalc", params, networkParameters, { true, true })
    , mOutputsCount(params.outputsCount)
    , mUseBias(params.mUseBias)
    , mZoneout(params.mZoneout)
    , mUseSingleParamTensor(params.mUseSingleParamTensor)
    , mForgetBias(params.mForgetBias)
    , mTmpCalculationsName(mName / "tmp")
    , mGatesName(mName / "gates")
    , mRandomNameHidden(mName / "randomH")
    , mRandomNameCell(mName / "randomC")
    , mNoZoneoutNewCellName(mName / "noZoneoutNewCell")
    , mNoZoneoutNewHiddenGradName(mName / "noZoneoutNewHiddenGrad")
    , mNoZoneoutNewCellGradName(mName / "noZoneoutNewCellGrad")
{
    auto prefix = mTypeName + "[" + mName + "::ctor]: ";

    if (mInputs.size() != 3)
    {
        throw std::runtime_error(prefix + "wrong number of input names");
    }
    if (mOutputs.size() != 2)
    {
        throw std::runtime_error(prefix + "wrong number of output names");
    }

    DECLARE_IMPL(LSTMFusedGatesCalcLayer, LSTMFusedGatesCalcLayerCPU<MemoryManager>, LSTMFusedGatesCalcLayerCPU<MemoryManagerFP16>)

    // Declare inputs
    for (size_t i = 0; i < mInputs.size(); ++i)
    {
        mNetworkParams.mWorkflow.copyDeclaration(name, mInputs[i], raul::Workflow::Usage::ForwardAndBackward, raul::Workflow::Mode::Read);
        mNetworkParams.mWorkflow.copyDeclaration(name, mInputs[i], mInputs[i].grad(), DEC_BACK_WRIT_ZERO);
    }

    // Declare temporal storages
    if (!mUseSingleParamTensor)
    {
        mNetworkParams.mWorkflow.tensorNeeded(mName, mTmpCalculationsName, WShape{ BS(), 1u, 1u, mOutputsCount }, DEC_FORW_WRIT);
    }
    else
    {
        mNetworkParams.mWorkflow.tensorNeeded(
            mName, mTmpCalculationsName, WShape{ BS(), 1u, 1u, mNetworkParams.mWorkflow.getWidth(mInputs[0]) + mNetworkParams.mWorkflow.getWidth(mInputs[1]) }, DEC_FORW_WRIT);
        mNetworkParams.mWorkflow.copyDeclaration(name, mTmpCalculationsName, mTmpCalculationsName, DEC_BACK_READ);
        mNetworkParams.mWorkflow.copyDeclaration(name, mTmpCalculationsName, mTmpCalculationsName.grad(), DEC_BACK_WRIT_ZERO);
    }
    mNetworkParams.mWorkflow.tensorNeeded(mName, mGatesName, WShape{ BS(), 1u, 1u, mOutputsCount }, DEC_FORW_WRIT);
    mNetworkParams.mWorkflow.copyDeclaration(name, mGatesName, mGatesName, DEC_BACK_READ);
    // And their grads
    mNetworkParams.mWorkflow.copyDeclaration(name, mGatesName, mGatesName.grad(), DEC_BACK_WRIT_ZERO);

    mUseZoneout = mZoneout != 0.0_dt;

    // Declare outputs
    for (size_t i = 0; i < mOutputs.size(); ++i)
    {
        mNetworkParams.mWorkflow.copyDeclaration(name, mInputs[i + 1], mOutputs[i], DEC_FORW_WRIT);
        if (i == 1)
        {
            if (mUseZoneout)
            {
                mNetworkParams.mWorkflow.copyDeclaration(name, mOutputs[i], mNoZoneoutNewCellName, DEC_FORW_WRIT);
                mNetworkParams.mWorkflow.copyDeclaration(name, mNoZoneoutNewCellName, mNoZoneoutNewCellName, DEC_BACK_READ);
            }
            else
            {
                mNetworkParams.mWorkflow.copyDeclaration(name, mOutputs[i], mOutputs[i], DEC_BACK_READ);
            }
        }
        mNetworkParams.mWorkflow.copyDeclaration(name, mOutputs[i], mOutputs[i].grad(), DEC_BACK_READ);
    }

    // Declare trainable params
    if (!mSharedWeights.empty())
    {
        mTrainableParamsNames = mSharedWeights;
    }
    else
    {
        const auto baseName = !mSharedLayer.empty() ? mSharedLayer : mName;
        if (mUseSingleParamTensor)
        {
            mTrainableParamsNames = { baseName / "Weights", baseName / "Biases" };
        }
        else
        {
            mTrainableParamsNames = { Name(baseName + "_ih") / "Weights", Name(baseName + "_ih") / "Biases", Name(baseName + "_hh") / "Weights", Name(baseName + "_hh") / "Biases" };
        }
    }

    if (mUseSingleParamTensor)
    {
        mNetworkParams.mWorkflow.tensorNeeded(
            mName, mTrainableParamsNames[0], WShape{ 1u, 1u, mOutputsCount, mNetworkParams.mWorkflow.getWidth(mInputs[0]) + mNetworkParams.mWorkflow.getWidth(mInputs[1]) }, DEC_TRAINABLE);
        if (mUseBias)
        {
            mNetworkParams.mWorkflow.tensorNeeded(mName, mTrainableParamsNames[1], WShape{ 1u, 1u, 1u, mOutputsCount }, DEC_TRAINABLE);
        }
    }
    else
    {
        mNetworkParams.mWorkflow.tensorNeeded(mName, mTrainableParamsNames[0], WShape{ 1u, 1u, mOutputsCount, mNetworkParams.mWorkflow.getWidth(mInputs[0]) }, DEC_TRAINABLE);
        mNetworkParams.mWorkflow.tensorNeeded(mName, mTrainableParamsNames[2], WShape{ 1u, 1u, mOutputsCount, mNetworkParams.mWorkflow.getWidth(mInputs[1]) }, DEC_TRAINABLE);
        if (mUseBias)
        {
            mNetworkParams.mWorkflow.tensorNeeded(mName, mTrainableParamsNames[1], WShape{ 1u, 1u, 1u, mOutputsCount }, DEC_TRAINABLE);
            mNetworkParams.mWorkflow.tensorNeeded(mName, mTrainableParamsNames[3], WShape{ 1u, 1u, 1u, mOutputsCount }, DEC_TRAINABLE);
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

    // Zoneout part
    if (mUseZoneout)
    {
        mNetworkParams.mWorkflow.copyDeclaration(mName, mOutputs[0], mRandomNameHidden, DEC_FORW_WRIT);
        mNetworkParams.mWorkflow.copyDeclaration(mName, mOutputs[1], mRandomNameCell, DEC_FORW_WRIT);
        mNetworkParams.mWorkflow.copyDeclaration(mName, mOutputs[0], mRandomNameHidden, DEC_BACK_READ);
        mNetworkParams.mWorkflow.copyDeclaration(mName, mOutputs[1], mRandomNameCell, DEC_BACK_READ);

        // Declare additional grads storages
        mNetworkParams.mWorkflow.copyDeclaration(mName, mOutputs[0].grad(), mNoZoneoutNewHiddenGradName, DEC_BACK_WRIT_ZERO);
        mNetworkParams.mWorkflow.copyDeclaration(mName, mOutputs[1].grad(), mNoZoneoutNewCellGradName, DEC_BACK_WRIT_ZERO);
    }
}

} // namespace raul
