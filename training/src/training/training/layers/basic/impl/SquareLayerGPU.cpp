// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "SquareLayerGPU.h"
#include "../SquareLayer.h"

namespace raul
{

SquareLayerGPU::SquareLayerGPU(SquareLayer& layer)
    : mLayer(layer)
{
}

void SquareLayerGPU::forwardComputeImpl(NetworkMode)
{
    auto& work = mLayer.mNetworkParams.mWorkflow;

    auto& output = work.getMemoryManager<MemoryManagerGPU>()(mLayer.mOutputs[0]);
    const auto& input = work.getMemoryManager<MemoryManagerGPU>()(mLayer.mInputs[0]);
    Common::squareForward(&work.getKernelManager(), mLayer.mTypeName + "[" + mLayer.mName + "::forwardComputeImpl]", work.getBatchSize(), mLayer.mDepth, mLayer.mHeight, mLayer.mWidth, input, output);
}

void SquareLayerGPU::backwardComputeImpl()
{
    auto& work = mLayer.mNetworkParams.mWorkflow;

    const auto& input = work.getMemoryManager<MemoryManagerGPU>()(mLayer.mInputs[0]);
    const auto& deltas = work.getMemoryManager<MemoryManagerGPU>()(mLayer.mOutputs[0].grad());
    // if (mNetworkParams.isGradNeeded(mInputs[0]))
    {
        auto& prevLayerDelta = work.getMemoryManager<MemoryManagerGPU>()(mLayer.mInputs[0].grad());
        Common::squareBackward(&work.getKernelManager(),
                               mLayer.mTypeName + "[" + mLayer.mName + "::bakwardComputeImpl]",
                               work.getBatchSize(),
                               mLayer.mDepth,
                               mLayer.mHeight,
                               mLayer.mWidth,
                               input,
                               deltas,
                               prevLayerDelta);
    }
}

} // namespace raul