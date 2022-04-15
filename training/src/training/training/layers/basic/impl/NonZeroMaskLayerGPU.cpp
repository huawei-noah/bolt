// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "NonZeroMaskLayerGPU.h"
#include "../NonZeroMaskLayer.h"

namespace raul
{

NonZeroMaskLayerGPU::NonZeroMaskLayerGPU(NonZeroMaskLayer& layer)
    : mLayer(layer)
{
}

void NonZeroMaskLayerGPU::forwardComputeImpl(NetworkMode)
{
    auto& memoryManager = mLayer.mNetworkParams.mWorkflow.getMemoryManager<MemoryManagerGPU>();
    const auto& input = memoryManager(mLayer.mInputs[0]);
    auto& mask = memoryManager(mLayer.mOutputs[0]);

    Common::nonZeroMask(&mLayer.mNetworkParams.mWorkflow.getKernelManager(),
                        mLayer.mTypeName + "[" + mLayer.mName + "forwardComputeImpl]",
                        mLayer.mNetworkParams.mWorkflow.getBatchSize(),
                        input.getDepth(),
                        input.getHeight(),
                        input.getWidth(),
                        input,
                        mask);
}

void NonZeroMaskLayerGPU::backwardComputeImpl() {}

} // namespace raul