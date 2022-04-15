// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "ReverseLayerGPU.h"
#include "../ReverseLayer.h"

namespace raul
{

void ReverseLayerGPU::forwardComputeImpl(NetworkMode)
{
    Workflow& work = mLayer.mNetworkParams.mWorkflow;

    const auto& input = work.getMemoryManager<MemoryManagerGPU>()(mLayer.mInputName);
    auto& output = work.getMemoryManager<MemoryManagerGPU>()(mLayer.mOutputName);

    const auto batch = work.getBatchSize();
    const auto depth = input.getDepth();
    const auto height = input.getHeight();
    const auto width = input.getWidth();

    work.getKernelManager().fillBuffer(output.getBuffer(), 0.0_dt, mLayer.mTypeName + "[" + mLayer.mName + "::forwardComputeImpl]");
    if (mLayer.mReverseOnly)
    {
        Common::reverse(&work.getKernelManager(), mLayer.mTypeName + "[" + mLayer.mName + "::forwardComputeImpl]", batch, depth, height, width, input, output);
    }
    else
    {
        Common::reverse(&work.getKernelManager(),
                        mLayer.mTypeName + "[" + mLayer.mName + "::forwardComputeImpl]",
                        batch,
                        depth,
                        height,
                        width,
                        input,
                        work.getMemoryManager<MemoryManagerGPU>()(mLayer.mInputs[1]),
                        output);
    }
}

void ReverseLayerGPU::backwardComputeImpl()
{
    Workflow& work = mLayer.mNetworkParams.mWorkflow;

    const auto& deltas = work.getMemoryManager<MemoryManagerGPU>()(mLayer.mOutputName.grad());
    auto& prevLayerDelta = work.getMemoryManager<MemoryManagerGPU>()(mLayer.mInputName.grad());

    const auto batch = work.getBatchSize();
    const auto depth = deltas.getDepth();
    const auto height = deltas.getHeight();
    const auto width = deltas.getWidth();

    if (mLayer.mReverseOnly)
    {
        Common::reverse(&work.getKernelManager(), mLayer.mTypeName + "[" + mLayer.mName + "::forwardComputeImpl]", batch, depth, height, width, deltas, prevLayerDelta);
    }
    else
    {
        Common::reverse(&work.getKernelManager(),
                        mLayer.mTypeName + "[" + mLayer.mName + "::forwardComputeImpl]",
                        batch,
                        depth,
                        height,
                        width,
                        deltas,
                        work.getMemoryManager<MemoryManagerGPU>()(mLayer.mInputs[1]),
                        prevLayerDelta);
    }
}

} // namespace raul
