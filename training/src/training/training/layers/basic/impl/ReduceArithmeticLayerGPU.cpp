// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "ReduceArithmeticLayerGPU.h"
#include "../ReduceArithmeticLayer.h"

#include <training/opencl/GPUCommon.h>

namespace raul
{

void ReduceArithmeticLayerGPU::forwardComputeImpl(NetworkMode)
{
    Workflow& work = mLayer.mNetworkParams.mWorkflow;

    auto& output = work.getMemoryManager<MemoryManagerGPU>()(mLayer.mOutputs[0]);
    const auto& input = work.getMemoryManager<MemoryManagerGPU>()(mLayer.mInputs[0]);
    work.getKernelManager().fillBuffer(output.getBuffer(), 0.0_dt, mLayer.mTypeName + "[" + mLayer.mName + "::forwardComputeImpl]");
    if (mLayer.mDim == Dimension::Default)
    {
        mLayer.mDiv = mLayer.mOperation == "sum" || mLayer.mOperation == "count_non_zero_elems"
                          ? 1.0_dt
                          : mLayer.mOperation == "batch_mean" ? TODTYPE(input.getBatchSize()) : TODTYPE(input.getShape().total_size());
        gpu::reduceDefaultForward(work.getKernelManager(),
                                  mLayer.mTypeName + "[" + mLayer.mName + "::forwardComputeImpl]",
                                  work.getBatchSize() * input.getDepth() * input.getHeight() * input.getWidth(),
                                  mLayer.mDiv,
                                  static_cast<size_t>(mLayer.mOperation == "count_non_zero_elems"),
                                  input.getBuffer(),
                                  output.getBuffer());
    }
    else
    {
        const size_t dimension = output.getShape().dimensions_num() - static_cast<size_t>(mLayer.mDim) - 1;
        gpu::reduceDimForward(work.getKernelManager(),
                              mLayer.mTypeName + "[" + mLayer.mName + "::forwardComputeImpl]",
                              work.getBatchSize(),
                              input.getDepth(),
                              input.getHeight(),
                              input.getWidth(),
                              output.getDepth(),
                              output.getHeight(),
                              output.getWidth(),
                              dimension,
                              mLayer.mOperation,
                              input.getBuffer(),
                              output.getBuffer());
    }
}

void ReduceArithmeticLayerGPU::backwardComputeImpl()
{
    Workflow& work = mLayer.mNetworkParams.mWorkflow;

    const auto& deltas = work.getMemoryManager<MemoryManagerGPU>()(mLayer.mOutputs[0].grad());

    // if (mNetworkParams.isGradNeeded(mLayer.mInputs[0]))
    {
        auto& inNablaTensor = work.getMemoryManager<MemoryManagerGPU>()(mLayer.mInputs[0].grad());
        if (mLayer.mDim == Dimension::Default)
        {
            gpu::reduceDefaultBackward(work.getKernelManager(),
                                       mLayer.mTypeName + "[" + mLayer.mName + "::backwardComputeImpl]",
                                       work.getBatchSize(),
                                       inNablaTensor.getDepth(),
                                       inNablaTensor.getHeight(),
                                       inNablaTensor.getWidth(),
                                       mLayer.mDiv,
                                       deltas.getBuffer(),
                                       inNablaTensor.getBuffer());
        }
        else
        {
            auto inputShape = inNablaTensor.getShape();
            mLayer.mDiv = mLayer.mOperation == "sum" || mLayer.mOperation == "count_non_zero_elems" ? 1.0_dt : TODTYPE(inputShape[static_cast<size_t>(mLayer.mDim)]);
            const size_t dimension = deltas.getShape().dimensions_num() - static_cast<size_t>(mLayer.mDim) - 1;
            gpu::reduceDimBackward(work.getKernelManager(),
                                   mLayer.mTypeName + "[" + mLayer.mName + "::backwardComputeImpl]",
                                   work.getBatchSize(),
                                   deltas.getDepth(),
                                   deltas.getHeight(),
                                   deltas.getWidth(),
                                   inNablaTensor.getDepth(),
                                   inNablaTensor.getHeight(),
                                   inNablaTensor.getWidth(),
                                   dimension,
                                   mLayer.mDiv,
                                   deltas.getBuffer(),
                                   inNablaTensor.getBuffer());
        }
    }
}

} // namespace raul