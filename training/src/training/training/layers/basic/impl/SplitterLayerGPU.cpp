// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "SplitterLayerGPU.h"
#include "../SplitterLayer.h"

namespace raul
{

SplitterLayerGPU::SplitterLayerGPU(SplitterLayer& layer)
    : mLayer(layer)
{
}

void SplitterLayerGPU::forwardComputeImpl(NetworkMode)
{
    auto& work = mLayer.mNetworkParams.mWorkflow;

    const auto& input = work.getMemoryManager<MemoryManagerGPU>()(mLayer.mInputs[0]);
    for (size_t q = 0; q < mLayer.mOutputs.size(); ++q)
    {
        auto& output = work.getMemoryManager<MemoryManagerGPU>()(mLayer.mOutputs[q]);
        Common::splitterForward(
            &work.getKernelManager(), mLayer.mTypeName + "[" + mLayer.mName + "::forwardComputeImpl]", work.getBatchSize(), mLayer.mDepth, mLayer.mHeight, mLayer.mWidth, input, output);
    }
}

void SplitterLayerGPU::backwardComputeImpl()
{
    auto& work = mLayer.mNetworkParams.mWorkflow;
    // if (mNetworkParams.isGradNeeded(mInputs[0]))
    {
        auto& prevLayerDelta = work.getMemoryManager<MemoryManagerGPU>()(mLayer.mInputs[0].grad());
        for (size_t q = 0; q < mLayer.mOutputs.size(); ++q)
        {
            const auto& deltas = work.getMemoryManager<MemoryManagerGPU>()(mLayer.mOutputs[q].grad());
            Common::splitterBackward(
                &work.getKernelManager(), mLayer.mTypeName + "[" + mLayer.mName + "::bakwardComputeImpl]", work.getBatchSize(), mLayer.mDepth, mLayer.mHeight, mLayer.mWidth, deltas, prevLayerDelta);
        }
    }
}

} // namespace raul