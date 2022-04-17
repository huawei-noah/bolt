// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "ElementWiseSumLayer.h"

#include "impl/ElementWiseSumLayerCPU.h"

#include <algorithm>

#include <training/base/common/MemoryManager.h>

namespace raul
{

ElementWiseSumLayer::ElementWiseSumLayer(const Name& name, const ElementWiseLayerParams& params, NetworkParameters& networkParameters)
    : BroadcastingLayer(name, "ElementWiseSum", params, networkParameters)
    , mBroadcast(params.mBroadcast)
{
    bool hasBatch = false;

    DECLARE_IMPL(ElementWiseSumLayer, ElementWiseSumLayerCPU<MemoryManager>, ElementWiseSumLayerCPU<MemoryManagerFP16>)

    for (const auto& input : mInputs)
    {
        mNetworkParams.mWorkflow.copyDeclaration(name, input, raul::Workflow::Usage::Forward, raul::Workflow::Mode::Read);

        if (mNetworkParams.mWorkflow.isTensorTrainable(input))
        {
            mNetworkParams.mWorkflow.copyDeclaration(mName, input, input.grad(), DEC_TRAINABLE_GRAD);
        }
        else
        {
            mNetworkParams.mWorkflow.copyDeclaration(mName, input, input.grad(), DEC_BACK_WRIT_ZERO);
        }

        hasBatch |= mNetworkParams.mWorkflow.getShape(input).isBSDependent();
    }

    shape outputShape{ hasBatch ? 1u : mNetworkParams.mWorkflow.getBatch(mInputs[0]),
                       mNetworkParams.mWorkflow.getDepth(mInputs[0]),
                       mNetworkParams.mWorkflow.getHeight(mInputs[0]),
                       mNetworkParams.mWorkflow.getWidth(mInputs[0]) };
    if (mBroadcast)
    {
        for (const auto& input_name : mInputs)
        {
            shape inputShape{ hasBatch ? 1u : mNetworkParams.mWorkflow.getBatch(input_name),
                              mNetworkParams.mWorkflow.getDepth(input_name),
                              mNetworkParams.mWorkflow.getHeight(input_name),
                              mNetworkParams.mWorkflow.getWidth(input_name) };
            std::transform(inputShape.begin(), inputShape.end(), outputShape.begin(), outputShape.begin(), [](auto a, auto b) { return std::max(a, b); });
        }
    }

    WShape outputWShape = hasBatch ? WShape{ BS(), outputShape[1], outputShape[2], outputShape[3] } : WShape{ outputShape[0], outputShape[1], outputShape[2], outputShape[3] };
    if (!hasBatch && outputShape.total_size() == 1)
    {
        mNetworkParams.mWorkflow.tensorNeeded(name, mOutputs[0], outputWShape, DEC_FORW_WRIT_NOMEMOPT);
    }
    else
    {
        mNetworkParams.mWorkflow.tensorNeeded(name, mOutputs[0], outputWShape, DEC_FORW_WRIT);
    }
    mNetworkParams.mWorkflow.copyDec(name, mOutputs[0], mOutputs[0].grad(), DEC_BACK_READ);

    mDepth = outputShape[1];
    mHeight = outputShape[2];
    mWidth = outputShape[3];
}

} // namespace raul