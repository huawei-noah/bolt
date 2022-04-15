// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "DropoutLayerGPU.h"
#include "../DropoutLayer.h"

#include <training/opencl/GPUCommon.h>

namespace raul
{

DropoutLayerGPU::DropoutLayerGPU(DropoutLayer& layer)
    : mLayer(layer)
{
}

void DropoutLayerGPU::forwardComputeImpl(NetworkMode mode)
{
    auto& work = mLayer.mNetworkParams.mWorkflow;

    if (mode == NetworkMode::Train || mode == NetworkMode::TrainCheckpointed)
    {
        auto& mRandom = work.getMemoryManager<MemoryManagerGPU>()(mLayer.mName / "random");
        auto& tmp = mLayer.mNetworkParams.mMemoryManager[mLayer.mTmpBufferName];
        if (mode == NetworkMode::Train)
        {
            if (mLayer.mNetworkParams.mCalculationMode == CalculationMode::DETERMINISTIC)
            {
                for (size_t i = 0; i < mRandom.size(); ++i)
                {
                    tmp[i] = static_cast<float>(random::bernoulli::randBool(mLayer.mProbability, mLayer.mState));
                }
            }
            else
            {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
                for (size_t i = 0; i < mRandom.size(); ++i)
                {
                    tmp[i] = static_cast<float>(random::bernoulli::randBool(mLayer.mProbability));
                }
            }
        }

        auto caller = mLayer.mTypeName + "[" + mLayer.mName + "forwardComputeImpl]";

        work.getKernelManager().writeBuffer(mRandom.getBuffer(), mRandom.size() * sizeof(dtype), &tmp[0], caller);

        auto& output = work.getMemoryManager<MemoryManagerGPU>()(mLayer.mOutputName);
        const auto& input = work.getMemoryManager<MemoryManagerGPU>()(mLayer.mInputName);
        // Get kernel
        gpu::dropoutForward(work.getKernelManager(),
                            mLayer.mTypeName + "[" + mLayer.mName + "forwardComputeImpl]",
                            work.getBatchSize(),
                            mLayer.mDepth,
                            mLayer.mHeight,
                            mLayer.mWidth,
                            mLayer.mScale,
                            input.getBuffer(),
                            mRandom.getBuffer(),
                            output.getBuffer());
    }
    else
    {
        work.getMemoryManager<MemoryManagerGPU>()[mLayer.mOutputName] = work.getMemoryManager<MemoryManagerGPU>()[mLayer.mInputName];
    }
}

void DropoutLayerGPU::backwardComputeImpl()
{
    auto& work = mLayer.mNetworkParams.mWorkflow;
    // if (mNetworkParams.isGradNeeded(mLayer.mInputName) && mNetworkParams.mMemoryManager.tensorExists(mLayer.mOutputName.grad()))
    {
        const auto& mRandom = work.getMemoryManager<MemoryManagerGPU>()(mLayer.mName / "random");
        const auto& deltas = work.getMemoryManager<MemoryManagerGPU>()(mLayer.mOutputName.grad());
        auto& prevLayerDelta = work.getMemoryManager<MemoryManagerGPU>()(mLayer.mInputName.grad());
        gpu::dropoutBackward(work.getKernelManager(),
                             mLayer.mTypeName + "[" + mLayer.mName + "backwardComputeImpl]",
                             work.getBatchSize(),
                             mLayer.mDepth,
                             mLayer.mHeight,
                             mLayer.mWidth,
                             mLayer.mScale,
                             deltas.getBuffer(),
                             mRandom.getBuffer(),
                             prevLayerDelta.getBuffer());
    }
}

} // namespace raul
