// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "TransposeLayerGPU.h"
#include "../TransposeLayer.h"

#include <training/common/Common.h>
#include <training/opencl/GPUCommon.h>

namespace raul
{

TransposeLayerGPU::TransposeLayerGPU(TransposeLayer& layer)
    : mLayer(layer)
{
}

void TransposeLayerGPU::forwardComputeImpl(NetworkMode)
{
    auto& work = mLayer.mNetworkParams.mWorkflow;

    auto& output = work.getMemoryManager<MemoryManagerGPU>()(mLayer.mOutputName);
    const auto& input = work.getMemoryManager<MemoryManagerGPU>()(mLayer.mInputName);

    const auto iDepth = work.getDepth(mLayer.mInputName);
    const auto iHeight = work.getHeight(mLayer.mInputName);
    const auto iWidth = work.getWidth(mLayer.mInputName);

    const auto oDepth = work.getDepth(mLayer.mOutputName);
    const auto oHeight = work.getHeight(mLayer.mOutputName);
    const auto oWidth = work.getWidth(mLayer.mOutputName);

    const auto batch = work.getBatch(mLayer.mInputName);

    const auto dimsNum = 4;
    size_t dimTran[dimsNum] = { 0, 1, 2, 3 };
    dimTran[dimsNum - 1 - mLayer.mDim1] = (cl_int)(dimsNum - 1 - mLayer.mDim2);
    dimTran[dimsNum - 1 - mLayer.mDim2] = (cl_int)(dimsNum - 1 - mLayer.mDim1);

    gpu::transpose(work.getKernelManager(),
                   mLayer.mTypeName + "[" + mLayer.mName + "::forwardComputeImpl]",
                   batch,
                   iDepth,
                   iHeight,
                   iWidth,
                   oDepth,
                   oHeight,
                   oWidth,
                   dimTran[0],
                   dimTran[1],
                   dimTran[2],
                   dimTran[3],
                   input.getBuffer(),
                   output.getBuffer());
}

void TransposeLayerGPU::backwardComputeImpl()
{
    auto& work = mLayer.mNetworkParams.mWorkflow;

    // if (mNetworkParams.isGradNeeded(mInputs[0]))
    {
        auto& prevLayerDelta = work.getMemoryManager<MemoryManagerGPU>()(mLayer.mInputName.grad());
        const auto& deltas = work.getMemoryManager<MemoryManagerGPU>()(mLayer.mOutputName.grad());

        const auto iDepth = work.getDepth(mLayer.mOutputName);
        const auto iHeight = work.getHeight(mLayer.mOutputName);
        const auto iWidth = work.getWidth(mLayer.mOutputName);

        const auto oDepth = work.getDepth(mLayer.mInputName);
        const auto oHeight = work.getHeight(mLayer.mInputName);
        const auto oWidth = work.getWidth(mLayer.mInputName);

        const auto batch = work.getBatch(mLayer.mOutputName);

        const auto dimsNum = 4;
        size_t dimTran[dimsNum] = { 0, 1, 2, 3 };
        dimTran[dimsNum - 1 - mLayer.mDim1] = (cl_int)(dimsNum - 1 - mLayer.mDim2);
        dimTran[dimsNum - 1 - mLayer.mDim2] = (cl_int)(dimsNum - 1 - mLayer.mDim1);

        gpu::transpose(work.getKernelManager(),
                       mLayer.mTypeName + "[" + mLayer.mName + "::forwardComputeImpl]",
                       batch,
                       iDepth,
                       iHeight,
                       iWidth,
                       oDepth,
                       oHeight,
                       oWidth,
                       dimTran[0],
                       dimTran[1],
                       dimTran[2],
                       dimTran[3],
                       deltas.getBuffer(),
                       prevLayerDelta.getBuffer());
    }
}

} // namespace raul
