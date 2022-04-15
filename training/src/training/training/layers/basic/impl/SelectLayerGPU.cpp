// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "SelectLayerGPU.h"
#include "../SelectLayer.h"

#include <training/opencl/GPUCommon.h>

namespace raul
{

SelectLayerGPU::SelectLayerGPU(SelectLayer& layer)
    : mLayer(layer)
{
}

void SelectLayerGPU::forwardComputeImpl(NetworkMode)
{
    mLayer.determineBroadcastFlags();

    if (!mLayer.mBroadcast && std::any_of(mLayer.mBroadcastQuery.begin(), mLayer.mBroadcastQuery.end(), [](const auto& needToBroadcast) { return needToBroadcast; }))
    {
        THROW("SelectLayer", mLayer.mName, "input size mismatch");
    }

    auto& work = mLayer.mNetworkParams.mWorkflow;

    auto& output = work.getMemoryManager<MemoryManagerGPU>()(mLayer.mOutputs[0]);
    const auto& cond = work.getMemoryManager<MemoryManagerGPU>()(mLayer.mInputs[0]);
    const auto& x = work.getMemoryManager<MemoryManagerGPU>()(mLayer.mInputs[1]);
    const auto& y = work.getMemoryManager<MemoryManagerGPU>()(mLayer.mInputs[2]);

    if (!mLayer.mBroadcast || (!mLayer.mBroadcastQuery[0] && !mLayer.mBroadcastQuery[1] && !mLayer.mBroadcastQuery[2]))
    {
        const auto depth = work.getDepth(mLayer.mInputs[0]);
        const auto height = work.getHeight(mLayer.mInputs[0]);
        const auto width = work.getWidth(mLayer.mInputs[0]);
        Common::selectForward(&work.getKernelManager(), mLayer.mTypeName + "[" + mLayer.mName + "::forwardComputeImpl]", work.getBatchSize(), depth, height, width, cond, x, y, output);
    }
    else
    {
        THROW(mLayer.mTypeName, mLayer.mName, "no broadcast support on GPU");
    }
}

void SelectLayerGPU::backwardComputeImpl()
{
    auto& work = mLayer.mNetworkParams.mWorkflow;

    const auto& deltas = work.getMemoryManager<MemoryManagerGPU>()(mLayer.mOutputs[0].grad());
    const auto& condition = work.getMemoryManager<MemoryManagerGPU>()(mLayer.mInputs[0]);
    for (size_t q = 1; q < mLayer.mInputs.size(); ++q)
    {
        // if (mNetworkParams.isGradNeeded(mLayer.mInputs[q]))
        {
            const auto depth = work.getDepth(mLayer.mInputs[q].grad());
            const auto height = work.getHeight(mLayer.mInputs[q].grad());
            const auto width = work.getWidth(mLayer.mInputs[q].grad());
            auto& prevLayerDelta = work.getMemoryManager<MemoryManagerGPU>()(mLayer.mInputs[q].grad());
            Common::selectBackward(
                &work.getKernelManager(), mLayer.mTypeName + "[" + mLayer.mName + "::backwardComputeImpl]", q - 1, work.getBatchSize(), depth, height, width, condition, deltas, prevLayerDelta);
        }
    }
}

} // namespace raul
