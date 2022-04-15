// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "ClampLayerGPU.h"
#include "../ClampLayer.h"

#include <training/opencl/GPUCommon.h>

namespace raul
{

ClampLayerGPU::ClampLayerGPU(ClampLayer& layer)
    : mLayer(layer)
{
}

void ClampLayerGPU::forwardComputeImpl(NetworkMode)
{
    auto& work = mLayer.mNetworkParams.mWorkflow;

    auto& output = work.getMemoryManager<MemoryManagerGPU>()(mLayer.mOutputs[0]);
    const auto& input = work.getMemoryManager<MemoryManagerGPU>()(mLayer.mInputs[0]);
    const auto depth = work.getDepth(mLayer.mInputs[0]);
    const auto height = work.getHeight(mLayer.mInputs[0]);
    const auto width = work.getWidth(mLayer.mInputs[0]);
    gpu::clampForward(work.getKernelManager(),
                      mLayer.mTypeName + "[" + mLayer.mName + "::forwardComputeImpl]",
                      work.getBatchSize(),
                      depth,
                      height,
                      width,
                      mLayer.mMin,
                      mLayer.mMax,
                      input.getBuffer(),
                      output.getBuffer());
}

void ClampLayerGPU::backwardComputeImpl()
{
    auto& work = mLayer.mNetworkParams.mWorkflow;

    const auto& deltas = work.getMemoryManager<MemoryManagerGPU>()(mLayer.mOutputs[0].grad());
    // if (mLayer.mNetworkParams.isGradNeeded(mLayer.mInputs[0]))
    {
        const auto& input = work.getMemoryManager<MemoryManagerGPU>()(mLayer.mInputs[0]);
        auto& prevLayerDelta = work.getMemoryManager<MemoryManagerGPU>()(mLayer.mInputs[0].grad());
        const auto depth = work.getDepth(mLayer.mInputs[0]);
        const auto height = work.getHeight(mLayer.mInputs[0]);
        const auto width = work.getWidth(mLayer.mInputs[0]);
        gpu::clampBackward(work.getKernelManager(),
                           mLayer.mTypeName + "[" + mLayer.mName + "::forwardComputeImpl]",
                           work.getBatchSize(),
                           depth,
                           height,
                           width,
                           mLayer.mMin,
                           mLayer.mMax,
                           input.getBuffer(),
                           deltas.getBuffer(),
                           prevLayerDelta.getBuffer());
    }
}

} // namespace raul