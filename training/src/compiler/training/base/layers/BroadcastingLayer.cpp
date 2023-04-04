// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "BroadcastingLayer.h"

namespace raul
{

BroadcastingLayer::BroadcastingLayer(const raul::Name& name, const std::string& typeName, const BasicParams& params, NetworkParameters& networkParams, std::pair<bool, bool> doChecks)
    : BasicLayer(name, typeName, params, networkParams, doChecks)
    , mBroadcastQuery(std::vector<bool>(mInputs.size(), false))
{
    auto prefix = mTypeName + "[" + mName + "::ctor]: ";

    if (mOutputs.size() != 1)
    {
        THROW(mTypeName, mName, "no output");
    }

    mLayerTarget = mNetworkParams.mWorkflow.getOverrideLayerExecutionTarget();
}

void BroadcastingLayer::determineBroadcastFlags()
{
    auto& work = mNetworkParams.mWorkflow;

    if (work.getExecutionTarget() == ExecutionTarget::CPU || mLayerTarget == LayerExecutionTarget::CPU)
    {
        if (!mNetworkParams.mMemoryManager.tensorExists(mOutputs[0]))
        {
            THROW("BroadcastingLayer", mName, "all tensors should exist at this moment");
        }

        // Get output shape
        const auto outputShape = mNetworkParams.mMemoryManager[mOutputs[0]].getShape();

        // Check the need of broadcasting for each input
        for (size_t i = 0; i < mInputs.size(); ++i)
        {
            if (!mNetworkParams.mMemoryManager.tensorExists(mInputs[i]))
            {
                THROW("BroadcastingLayer", mName, "all tensors should exist at this moment");
            }

            const Tensor& input = mNetworkParams.mMemoryManager[mInputs[i]];
            if (!input.isBroadcastableTo(outputShape))
            {
                THROW("BroadcastingLayer", mName, "input tensor[" + mInputs[i] + "] is not broadcastable to output tensor[" + mOutputs[0] + "]");
            }
            mBroadcastQuery[i] = (input.getShape() != outputShape);
        }
    }
    else if (work.getExecutionTarget() == ExecutionTarget::CPUFP16 || mLayerTarget == LayerExecutionTarget::CPUFP16)
    {
        if (!work.getMemoryManager<MemoryManagerFP16>().tensorExists(mOutputs[0]))
        {
            THROW("BroadcastingLayer", mName, "all tensors should exist at this moment");
        }

        // Get output shape
        const auto outputShape = work.getMemoryManager<MemoryManagerFP16>()[mOutputs[0]].getShape();

        // Check the need of broadcasting for each input
        for (size_t i = 0; i < mInputs.size(); ++i)
        {
            if (!work.getMemoryManager<MemoryManagerFP16>().tensorExists(mInputs[i]))
            {
                THROW("BroadcastingLayer", mName, "all tensors should exist at this moment");
            }

            const auto& input = work.getMemoryManager<MemoryManagerFP16>()[mInputs[i]];
            if (!input.isBroadcastableTo(outputShape))
            {
                THROW("BroadcastingLayer", mName, "input tensor[" + mInputs[i] + "] is not broadcastable to output tensor[" + mOutputs[0] + "]");
            }
            mBroadcastQuery[i] = (input.getShape() != outputShape);
        }
    }
    else
    {
        THROW_NONAME("BroadcastingLayer", "unsupported execution target");
    }
}
} // raul namespace
