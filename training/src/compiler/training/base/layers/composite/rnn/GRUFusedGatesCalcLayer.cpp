// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "GRUFusedGatesCalcLayer.h"

#include "impl/GRUFusedGatesCalcLayerCPU.h"

#include <training/base/common/MemoryManager.h>

namespace raul
{

GRUFusedGatesCalcLayer::GRUFusedGatesCalcLayer(const Name& name, const GRUFusedGatesCalcParams& params, NetworkParameters& networkParameters)
    : TrainableLayer(name, "GRUFusedGatesCalc", params, networkParameters, { true, true })
    , mOutputsCount(params.outputsCount)
    , mUseBiasForInput(params.mUseBiasForInput)
    , mUseBiasForHidden(params.mUseBiasForHidden)
    , mLinearIHTmp(mName / "linearIH")
    , mLinearHHTmp(mName / "linearHH")
    , mWeightsNameIH(Name(mName + "_ih") / "Weights")
    , mBiasesNameIH(Name(mName + "_ih") / "Biases")
    , mWeightsNameHH(Name(mName + "_hh") / "Weights")
    , mBiasesNameHH(Name(mName + "_hh") / "Biases")
{
    auto prefix = mTypeName + "[" + mName + "::ctor]: ";

    if (mInputs.size() != 2)
    {
        THROW("GRUFusedGatesCalcLayer", name, "wrong number of input names");
    }
    if (mOutputs.size() != 1)
    {
        THROW("GRUFusedGatesCalcLayer", name, "wrong number of output names");
    }

    DECLARE_IMPL(GRUFusedGatesCalcLayer, GRUFusedGatesCalcLayerCPU<MemoryManager>, GRUFusedGatesCalcLayerCPU<MemoryManagerFP16>)

    // Declare inputs
    for (size_t i = 0; i < mInputs.size(); ++i)
    {
        mNetworkParams.mWorkflow.copyDeclaration(name, mInputs[i], raul::Workflow::Usage::ForwardAndBackward, raul::Workflow::Mode::Read);
        mNetworkParams.mWorkflow.copyDeclaration(name, mInputs[i], mInputs[i].grad(), DEC_BACK_WRIT_ZERO);
    }

    // Declare temporal storages
    mNetworkParams.mWorkflow.tensorNeeded(mName, mLinearIHTmp, WShape{ BS(), 1u, 1u, mOutputsCount }, DEC_FORW_WRIT);
    mNetworkParams.mWorkflow.copyDeclaration(name, mLinearIHTmp, mLinearIHTmp, DEC_BACK_READ);
    mNetworkParams.mWorkflow.copyDeclaration(name, mLinearIHTmp, mLinearHHTmp, DEC_FORW_WRIT);
    mNetworkParams.mWorkflow.copyDeclaration(name, mLinearHHTmp, mLinearHHTmp, DEC_BACK_READ);
    // And their grads
    mNetworkParams.mWorkflow.copyDeclaration(name, mLinearIHTmp, mLinearIHTmp.grad(), DEC_BACK_WRIT_ZERO);
    mNetworkParams.mWorkflow.copyDeclaration(name, mLinearHHTmp, mLinearHHTmp.grad(), DEC_BACK_WRIT_ZERO);

    // Declare output
    mNetworkParams.mWorkflow.copyDeclaration(name, mInputs[1], mOutputs[0], DEC_FORW_WRIT);
    mNetworkParams.mWorkflow.copyDeclaration(name, mOutputs[0], mOutputs[0].grad(), DEC_BACK_READ);

    // Declare trainable params
    if (mSharedLayer.empty() && mSharedWeights.empty())
    {
        mNetworkParams.mWorkflow.tensorNeeded(mName, mWeightsNameIH, WShape{ 1u, 1u, mOutputsCount, mNetworkParams.mWorkflow.getWidth(mInputs[0]) }, DEC_TRAINABLE);
        mNetworkParams.mWorkflow.tensorNeeded(mName, mWeightsNameHH, WShape{ 1u, 1u, mOutputsCount, mNetworkParams.mWorkflow.getWidth(mInputs[1]) }, DEC_TRAINABLE);
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
}

} // namespace raul