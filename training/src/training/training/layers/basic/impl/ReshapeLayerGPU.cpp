// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "ReshapeLayerGPU.h"
#include "../ReshapeLayer.h"

#include <training/opencl/GemmGPU.h>

namespace raul
{

void ReshapeLayerGPU::forwardComputeImpl(NetworkMode)
{
    mLayer.mNetworkParams.mMemoryManagerGPU[mLayer.mOutputName] = mLayer.mNetworkParams.mMemoryManagerGPU[mLayer.mInputName];
}

void ReshapeLayerGPU::backwardComputeImpl()
{
    TensorGPU& prevLayerDelta = mLayer.mNetworkParams.mWorkflow.getMemoryManager<MemoryManagerGPU>()(mLayer.mInputName.grad());
    const TensorGPU& delta = mLayer.mNetworkParams.mWorkflow.getMemoryManager<MemoryManagerGPU>()(mLayer.mOutputName.grad());

    gpu::axpy(mLayer.mNetworkParams.mWorkflow.getKernelManager(), mLayer.mName / "backward", delta.size(), 1.0_dt, delta.getBuffer(), 1, prevLayerDelta.getBuffer(), 1, 0, 0);
}

} // namespace raul
