// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "SelectLayer.h"

#include "impl/SelectLayerCPU.h"

#include <training/base/common/MemoryManager.h>

namespace raul
{

SelectLayer::SelectLayer(const Name& name, const ElementWiseLayerParams& params, NetworkParameters& networkParameters)
    : BroadcastingLayer(name, "Select", params, networkParameters)
    , mBroadcast(params.mBroadcast)
{
    if (mInputs.size() != 3)
    {
        THROW("SelectLayer", name, "wrong number of input names");
    }

    DECLARE_IMPL(SelectLayer, SelectLayerCPU<MemoryManager>, SelectLayerCPU<MemoryManagerFP16>)

    for (size_t i = 0; i < mInputs.size(); ++i)
    {
        if (i == 0)
        {
            mNetworkParams.mWorkflow.copyDeclaration(name, mInputs[i], raul::Workflow::Usage::ForwardAndBackward, raul::Workflow::Mode::Read);
        }
        else
        {
            mNetworkParams.mWorkflow.copyDeclaration(name, mInputs[i], raul::Workflow::Usage::Forward, raul::Workflow::Mode::Read);
            mNetworkParams.mWorkflow.copyDeclaration(name, mInputs[i], mInputs[i].grad(), DEC_BACK_WRIT_ZERO);
        }
    }

    shape outputShape{ 1u, mNetworkParams.mWorkflow.getDepth(mInputs[0]), mNetworkParams.mWorkflow.getHeight(mInputs[0]), mNetworkParams.mWorkflow.getWidth(mInputs[0]) };
    bool isOutputBSDependent = mNetworkParams.mWorkflow.getShape(mInputs[0]).isBSDependent();
    if (mBroadcast)
    {
        for (size_t i = 1; i < mInputs.size(); ++i)
        {
            isOutputBSDependent |= mNetworkParams.mWorkflow.getShape(mInputs[i]).isBSDependent();
            shape inputShape{ 1u, mNetworkParams.mWorkflow.getDepth(mInputs[i]), mNetworkParams.mWorkflow.getHeight(mInputs[i]), mNetworkParams.mWorkflow.getWidth(mInputs[i]) };
            std::transform(inputShape.begin(), inputShape.end(), outputShape.begin(), outputShape.begin(), [](auto a, auto b) { return std::max(a, b); });
        }
    }
    WShape outputWShape{ raul::BS(), outputShape[1], outputShape[2], outputShape[3] };
    if (!isOutputBSDependent)
    {
        outputWShape = { 1u, outputShape[1], outputShape[2], outputShape[3] };
    }
    mNetworkParams.mWorkflow.tensorNeeded(name, mOutputs[0], outputWShape, DEC_FORW_WRIT);
    mNetworkParams.mWorkflow.copyDeclaration(name, mOutputs[0], mOutputs[0].grad(), DEC_BACK_READ);
}

}