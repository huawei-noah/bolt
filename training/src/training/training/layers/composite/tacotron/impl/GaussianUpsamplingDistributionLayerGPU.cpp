// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "GaussianUpsamplingDistributionLayerGPU.h"
#include "../GaussianUpsamplingDistributionLayer.h"

namespace raul
{

void GaussianUpsamplingDistributionLayerGPU::forwardComputeImpl(NetworkMode)
{
    Workflow& work = mLayer.mNetworkParams.mWorkflow;

    auto& output = work.getMemoryManager<MemoryManagerGPU>()(mLayer.mOutputName);
    const auto& values = work.getMemoryManager<MemoryManagerGPU>()(mLayer.mValuesName);
    const auto& loc = work.getMemoryManager<MemoryManagerGPU>()(mLayer.mLocName);
    const auto& scale = work.getMemoryManager<MemoryManagerGPU>()(mLayer.mScaleName);

    Common::gaussianUpsamplingDistributionForward(&work.getKernelManager(),
                                                  mLayer.mTypeName + "[" + mLayer.mName + "forwardComputeImpl]",
                                                  work.getBatchSize(),
                                                  output.getDepth(),
                                                  output.getHeight(),
                                                  output.getWidth(),
                                                  values,
                                                  loc,
                                                  scale,
                                                  output);
}

void GaussianUpsamplingDistributionLayerGPU::backwardComputeImpl()
{
    Workflow& work = mLayer.mNetworkParams.mWorkflow;

    // if (mNetworkParams.isGradNeeded(mInputs[1]) || mNetworkParams.isGradNeeded(mInputs[2]))
    {
        const auto& deltas = work.getMemoryManager<MemoryManagerGPU>()(mLayer.mOutputName.grad());
        const auto& values = work.getMemoryManager<MemoryManagerGPU>()(mLayer.mValuesName);
        const auto& loc = work.getMemoryManager<MemoryManagerGPU>()(mLayer.mLocName);
        const auto& scale = work.getMemoryManager<MemoryManagerGPU>()(mLayer.mScaleName);

        // if (mNetworkParams.isGradNeeded(mInputs[1]))
        {
            auto& locGrad = work.getMemoryManager<MemoryManagerGPU>()(mLayer.mLocName.grad());
            Common::gaussianUpsamplingDistributionBackward(&work.getKernelManager(),
                                                           mLayer.mTypeName + "[" + mLayer.mName + "backwardComputeImpl]",
                                                           work.getBatchSize(),
                                                           deltas.getDepth(),
                                                           deltas.getHeight(),
                                                           deltas.getWidth(),
                                                           true,
                                                           values,
                                                           loc,
                                                           scale,
                                                           deltas,
                                                           locGrad);
        }
        // if (mNetworkParams.isGradNeeded(mInputs[2])
        {
            auto& scaleGrad = work.getMemoryManager<MemoryManagerGPU>()(mLayer.mScaleName.grad());
            Common::gaussianUpsamplingDistributionBackward(&work.getKernelManager(),
                                                           mLayer.mTypeName + "[" + mLayer.mName + "backwardComputeImpl]",
                                                           work.getBatchSize(),
                                                           deltas.getDepth(),
                                                           deltas.getHeight(),
                                                           deltas.getWidth(),
                                                           false,
                                                           values,
                                                           loc,
                                                           scale,
                                                           deltas,
                                                           scaleGrad);
        }
    }
}

} // namespace raul