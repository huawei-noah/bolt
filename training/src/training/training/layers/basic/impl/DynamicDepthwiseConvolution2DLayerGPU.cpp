// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "DynamicDepthwiseConvolution2DLayerGPU.h"
#include "../DynamicDepthwiseConvolution2DLayer.h"

namespace raul
{

DynamicDepthwiseConvolution2DLayerGPU::DynamicDepthwiseConvolution2DLayerGPU(DynamicDepthwiseConvolution2DLayer& layer)
    : mLayer(layer)
{
}

void DynamicDepthwiseConvolution2DLayerGPU::forwardComputeImpl(NetworkMode)
{
    auto& memoryManager = mLayer.mNetworkParams.mWorkflow.getMemoryManager<MemoryManagerGPU>();
    auto& output = memoryManager(mLayer.mOutputName);
    const auto& input = memoryManager(mLayer.mInputName);
    const auto& filter = memoryManager(mLayer.mFiltersName);

    const size_t batchSize = mLayer.mNetworkParams.mWorkflow.getBatch(mLayer.mInputName);

    if (mLayer.mInputDepth == 0u)
    {
        mLayer.mInputDepth = input.getWidth();
        mLayer.mInChannels = filter.getHeight();
    }

    mLayer.mNetworkParams.mWorkflow.getKernelManager().fillBuffer(output.getBuffer(), 0.0_dt, mLayer.mTypeName + "[" + mLayer.mName + "::forwardComputeImpl]");
    Common::dynamicDepthwiseConv2DForward(&mLayer.mNetworkParams.mWorkflow.getKernelManager(),
                                          mLayer.mTypeName + "[" + mLayer.mName + "::forwardComputeImpl]",
                                          batchSize,
                                          mLayer.mInChannels,
                                          mLayer.mOutputHeight,
                                          mLayer.mOutputWidth,
                                          mLayer.mChannelMultiplier,
                                          mLayer.mFilterHeight,
                                          mLayer.mFilterWidth,
                                          input,
                                          filter,
                                          output);
}

void DynamicDepthwiseConvolution2DLayerGPU::backwardComputeImpl()
{
    auto& memoryManager = mLayer.mNetworkParams.mWorkflow.getMemoryManager<MemoryManagerGPU>();

    const size_t batchSize = mLayer.mNetworkParams.mWorkflow.getBatch(mLayer.mInputName);

    const auto& deltas = memoryManager(mLayer.mOutputName.grad());
    const auto& input = memoryManager(mLayer.mInputName);
    const auto& filter = memoryManager(mLayer.mFiltersName);

    // if (mLayer.mNetworkParams.isGradNeeded(mLayer.mInputName))
    // {
    auto& inputGrad = memoryManager(mLayer.mInputName.grad());
    Common::dynamicDepthwiseConv2DBackward(&mLayer.mNetworkParams.mWorkflow.getKernelManager(),
                                           mLayer.mTypeName + "[" + mLayer.mName + "::backwardComputeImpl]",
                                           batchSize,
                                           mLayer.mInChannels,
                                           mLayer.mOutputHeight,
                                           mLayer.mOutputWidth,
                                           mLayer.mChannelMultiplier,
                                           mLayer.mFilterHeight,
                                           mLayer.mFilterWidth,
                                           true,
                                           deltas,
                                           filter,
                                           inputGrad);
    // }

    // if (mLayer.mNetworkParams.isGradNeeded(mLayer.mFiltersName))
    // {
    auto& filtersGrad = memoryManager(mLayer.mFiltersName.grad());
    Common::dynamicDepthwiseConv2DBackward(&mLayer.mNetworkParams.mWorkflow.getKernelManager(),
                                           mLayer.mTypeName + "[" + mLayer.mName + "::backwardComputeImpl]",
                                           batchSize,
                                           mLayer.mInChannels,
                                           mLayer.mOutputHeight,
                                           mLayer.mOutputWidth,
                                           mLayer.mChannelMultiplier,
                                           mLayer.mFilterHeight,
                                           mLayer.mFilterWidth,
                                           false,
                                           deltas,
                                           input,
                                           filtersGrad);
    // }
}

} // namespace raul