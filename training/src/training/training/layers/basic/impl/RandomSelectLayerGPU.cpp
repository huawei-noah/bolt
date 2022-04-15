// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "RandomSelectLayerGPU.h"
#include "../RandomSelectLayer.h"

#include <training/common/Random.h>

namespace raul
{

RandomSelectLayerGPU::RandomSelectLayerGPU(RandomSelectLayer& layer)
    : mLayer(layer)
{
}

void RandomSelectLayerGPU::forwardComputeImpl(NetworkMode)
{
    mLayer.determineBroadcastFlags();

    if (!mLayer.mBroadcast && (mLayer.mBroadcastQuery[0] || mLayer.mBroadcastQuery[1]))
    {
        THROW("RandomSelectLayer", mLayer.mName, "input size mismatch");
    }

    auto& work = mLayer.mNetworkParams.mWorkflow;

    auto& mRandomCPU = work.getMemoryManager<MemoryManager>()[mLayer.mName / "randomCPU"];
    for (size_t q = 0; q < mRandomCPU.size(); ++q)
    {
        mRandomCPU[q] = random::bernoulli::randBool(mLayer.mProbability) ? 1_dt : 0_dt;
    }
    auto& output = work.getMemoryManager<MemoryManagerGPU>()(mLayer.mOutputs[0]);
    const auto& cond = work.getMemoryManager<MemoryManagerGPU>()(mLayer.mRandomName);
    const auto& x = work.getMemoryManager<MemoryManagerGPU>()(mLayer.mInputs[0]);
    const auto& y = work.getMemoryManager<MemoryManagerGPU>()(mLayer.mInputs[1]);

    auto caller = mLayer.mTypeName + "[" + mLayer.mName + "forwardComputeImpl]";

    work.getKernelManager().writeBuffer(cond.getBuffer(), mRandomCPU.size() * sizeof(dtype), &mRandomCPU[0], caller);

    const auto depth = work.getDepth(mLayer.mInputs[0]);
    const auto height = work.getHeight(mLayer.mInputs[0]);
    const auto width = work.getWidth(mLayer.mInputs[0]);
    Common::selectForward(&work.getKernelManager(), caller, work.getBatchSize(), depth, height, width, cond, x, y, output);
}

void RandomSelectLayerGPU::backwardComputeImpl()
{
    auto& work = mLayer.mNetworkParams.mWorkflow;
    const auto& deltas = work.getMemoryManager<MemoryManagerGPU>()(mLayer.mOutputs[0].grad());
    const auto& condition = work.getMemoryManager<MemoryManagerGPU>()(mLayer.mRandomName);
    for (size_t q = 0; q < mLayer.mInputs.size(); ++q)
    {
        // if (mLayer.mNetworkParams.isGradNeeded(mLayer.mInputs[q]))
        {
            const auto depth = work.getDepth(mLayer.mInputs[q].grad());
            const auto height = work.getHeight(mLayer.mInputs[q].grad());
            const auto width = work.getWidth(mLayer.mInputs[q].grad());
            auto& prevLayerDelta = work.getMemoryManager<MemoryManagerGPU>()(mLayer.mInputs[q].grad());
            Common::selectBackward(
                &work.getKernelManager(), mLayer.mTypeName + "[" + mLayer.mName + "::backwardComputeImpl]", q, work.getBatchSize(), depth, height, width, condition, deltas, prevLayerDelta);
        }
    }
}

} // namespace raul
