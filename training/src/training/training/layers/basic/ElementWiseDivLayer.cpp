// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "ElementWiseDivLayer.h"

#include "impl/ElementWiseDivLayerCPU.h"
#include "impl/ElementWiseDivLayerGPU.h"

#include <algorithm>

#include <training/common/MemoryManager.h>

namespace raul
{

ElementWiseDivLayer::ElementWiseDivLayer(const Name& name, const ElementWiseLayerParams& params, NetworkParameters& networkParameters)
    : BroadcastingLayer(name, "ElementWiseDiv", params, networkParameters)
    , mBroadcast(params.mBroadcast)
    , mBackwardTmpBufferName("TempStorageForIntermediateCalculations")
{
    if (mInputs.size() != 2)
    {
        THROW("ElementWiseDivLayer", mName, "wrong number of input names");
    }

    if (mNetworkParams.mWorkflow.getOverrideLayerExecutionTarget() == LayerExecutionTarget::Default)
    {
        DECLARE_IMPL(ElementWiseDivLayer, ElementWiseDivLayerCPU<MemoryManager>, ElementWiseDivLayerGPU, ElementWiseDivLayerCPU<MemoryManagerFP16>)
    }
    else if (mNetworkParams.mWorkflow.getOverrideLayerExecutionTarget() == LayerExecutionTarget::CPU)
    {
        DECLARE_IMPL(ElementWiseDivLayer, ElementWiseDivLayerCPU<MemoryManager>, NotImplemented, ElementWiseDivLayerCPU<MemoryManager>)
    }
    else if (mNetworkParams.mWorkflow.getOverrideLayerExecutionTarget() == LayerExecutionTarget::CPUFP16)
    {
        DECLARE_IMPL(ElementWiseDivLayer, ElementWiseDivLayerCPU<MemoryManagerFP16>, NotImplemented, ElementWiseDivLayerCPU<MemoryManagerFP16>)
    }
    else
    {
        THROW(mTypeName, mName, "unsupported layer execution target");
    }

    for (const auto& input : mInputs)
    {
        mNetworkParams.mWorkflow.copyDeclaration(mName, input, raul::Workflow::Usage::ForwardAndBackward, raul::Workflow::Mode::Read);

        mNetworkParams.mWorkflow.copyDeclaration(mName, input, input.grad(), DEC_BACK_WRIT_ZERO);
    }

    if (!mNetworkParams.mWorkflow.isBatchPlaceholded(mInputs[0]) && !mNetworkParams.mWorkflow.isBatchPlaceholded(mInputs[1]))
    {
        shape dividend_shape = shape{
            mNetworkParams.mWorkflow.getBatch(mInputs[0]), mNetworkParams.mWorkflow.getDepth(mInputs[0]), mNetworkParams.mWorkflow.getHeight(mInputs[0]), mNetworkParams.mWorkflow.getWidth(mInputs[0])
        };

        if (mBroadcast)
        {
            const shape divisor_shape = shape{ mNetworkParams.mWorkflow.getBatch(mInputs[1]),
                                               mNetworkParams.mWorkflow.getDepth(mInputs[1]),
                                               mNetworkParams.mWorkflow.getHeight(mInputs[1]),
                                               mNetworkParams.mWorkflow.getWidth(mInputs[1]) };

            std::transform(divisor_shape.cbegin(), divisor_shape.cend(), dividend_shape.begin(), dividend_shape.begin(), [](auto a, auto b) { return std::max(a, b); });
        }

        mNetworkParams.mWorkflow.tensorNeeded(mName, mOutputs[0], WShape(dividend_shape), DEC_FORW_WRIT);
    }
    else
    {
        if (mBroadcast)
        {
            shape dividend_shape = shape{ 1u, mNetworkParams.mWorkflow.getDepth(mInputs[0]), mNetworkParams.mWorkflow.getHeight(mInputs[0]), mNetworkParams.mWorkflow.getWidth(mInputs[0]) };

            const shape divisor_shape = shape{ 1u, mNetworkParams.mWorkflow.getDepth(mInputs[1]), mNetworkParams.mWorkflow.getHeight(mInputs[1]), mNetworkParams.mWorkflow.getWidth(mInputs[1]) };

            std::transform(divisor_shape.cbegin(), divisor_shape.cend(), dividend_shape.begin(), dividend_shape.begin(), [](auto a, auto b) { return std::max(a, b); });

            mNetworkParams.mWorkflow.tensorNeeded(mName, mOutputs[0], WShape{ BS(), dividend_shape[1], dividend_shape[2], dividend_shape[3] }, DEC_FORW_WRIT);
        }
        else
        {
            mNetworkParams.mWorkflow.copyDeclaration(mName, mInputs[0], mOutputs[0], DEC_FORW_WRIT);
        }
    }

    mNetworkParams.mWorkflow.copyDeclaration(mName, mOutputs[0], mOutputs[0].grad(), DEC_BACK_READ);

    mDepth = mNetworkParams.mWorkflow.getDepth(mOutputs[0]);
    mHeight = mNetworkParams.mWorkflow.getHeight(mOutputs[0]);
    mWidth = mNetworkParams.mWorkflow.getWidth(mOutputs[0]);

    if ((mNetworkParams.mWorkflow.getExecutionTarget() == ExecutionTarget::GPU && params.getLayerExecutionTarget() == LayerExecutionTarget::Default) ||
        params.getLayerExecutionTarget() == LayerExecutionTarget::GPU)
    {
        mNetworkParams.mWorkflow.tensorNeededMaxShape(
            mName, mBackwardTmpBufferName, WShape{ BS(), mDepth, mHeight, mWidth }, Workflow::Usage::Backward, Workflow::Mode::Write, true, true, false, false, false, LayerExecutionTarget::GPU);
    }
}

} // namespace raul