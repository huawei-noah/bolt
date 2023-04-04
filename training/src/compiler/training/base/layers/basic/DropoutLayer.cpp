// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "DropoutLayer.h"

#include "impl/DropoutLayerCPU.h"

namespace raul
{

DropoutLayer::DropoutLayer(const Name& name, const DropoutParams& params, NetworkParameters& networkParameters)
    : BasicLayer(name, "Dropout", params, networkParameters)
    , mProbability(params.probability)
    , mState(random::getGenerator())
    , mTmpBufferName("TempStorageForIntermediateCalculationsCPU")
{
    auto prefix = mTypeName + "[" + mName + "::ctor]: ";
    if (mInputs.size() != 1)
    {
        THROW(mTypeName, mName, "wrong number of input names");
    }
    if (mOutputs.size() != 1)
    {
        THROW(mTypeName, mName, "wrong number of output names");
    }

    if (mProbability < 0.0_dt || mProbability >= 1.0_dt)
    {
        THROW(mTypeName, mName, ", must be in [0,1)");
    }

    DECLARE_IMPL(DropoutLayer, DropoutLayerCPU<MemoryManager>, DropoutLayerCPU<MemoryManagerFP16>)

    mScale = 1.0_dt / (1.0_dt - mProbability);

    mInputName = mInputs[0];
    mOutputName = mOutputs[0];

    mNetworkParams.mWorkflow.copyDeclaration(mName, mInputName, raul::Workflow::Usage::Forward, raul::Workflow::Mode::Read);

    mNetworkParams.mWorkflow.copyDeclaration(mName, mInputName, mInputName.grad(), DEC_BACK_WRIT_ZERO);

    mNetworkParams.mWorkflow.copyDeclaration(mName, mInputName, mOutputName, DEC_FORW_WRIT);

    mNetworkParams.mWorkflow.copyDeclaration(mName, mInputName, mOutputName.grad(), DEC_BACK_READ);

    mDepth = mNetworkParams.mWorkflow.getDepth(mInputName);
    mHeight = mNetworkParams.mWorkflow.getHeight(mInputName);
    mWidth = mNetworkParams.mWorkflow.getWidth(mInputName);

    mNetworkParams.mWorkflow.tensorNeeded(
        mName, mName / "random", WShape{ BS(), mDepth, mHeight, mWidth }, raul::Workflow::Usage::Forward, raul::Workflow::Mode::Write, true, true, false, false, false);
    mNetworkParams.mWorkflow.copyDeclaration(mName, mName / "random", DEC_BACK_READ);
}
} // namespace raul
