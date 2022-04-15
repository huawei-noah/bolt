// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "ElementWiseSubLayerGPU.h"
#include "../ElementWiseSubLayer.h"

#include <training/opencl/GPUCommon.h>

namespace raul
{

ElementWiseSubLayerGPU::ElementWiseSubLayerGPU(ElementWiseSubLayer& layer)
    : mLayer(layer)
{
}

void ElementWiseSubLayerGPU::forwardComputeImpl(NetworkMode)
{
    mLayer.determineBroadcastFlags();

    if (!mLayer.mBroadcast && (mLayer.mBroadcastQuery[0] || mLayer.mBroadcastQuery[1]))
    {
        THROW(mLayer.mTypeName, mLayer.mName, "input size mismatch");
    }

    auto& work = mLayer.mNetworkParams.mWorkflow;

    auto& output = work.getMemoryManager<MemoryManagerGPU>()(mLayer.mOutputs[0]);
    const auto& minuend = work.getMemoryManager<MemoryManagerGPU>()(mLayer.mInputs[0]);
    const auto& subtrahend = work.getMemoryManager<MemoryManagerGPU>()(mLayer.mInputs[1]);
    gpu::eltwiseSubOp(work.getKernelManager(),
                      mLayer.mTypeName + "[" + mLayer.mName + "::forwardComputeImpl]",
                      output.getBatchSize(),
                      mLayer.mDepth,
                      mLayer.mHeight,
                      mLayer.mWidth,
                      minuend.getBatchSize(),
                      minuend.getDepth(),
                      minuend.getHeight(),
                      minuend.getWidth(),
                      minuend.getBuffer(),
                      subtrahend.getBatchSize(),
                      subtrahend.getDepth(),
                      subtrahend.getHeight(),
                      subtrahend.getWidth(),
                      subtrahend.getBuffer(),
                      output.getBuffer());
}

void ElementWiseSubLayerGPU::backwardComputeImpl()
{
    auto& work = mLayer.mNetworkParams.mWorkflow;

    const auto& deltas = work.getMemoryManager<MemoryManagerGPU>()(mLayer.mOutputs[0].grad());
    // if (mLayer.mNetworkParams.isGradNeeded(mLayer.mInputs[0]))
    {
        auto& minuendDelta = work.getMemoryManager<MemoryManagerGPU>()(mLayer.mInputs[0].grad());
        if (mLayer.mBroadcastQuery[0])
        {
            for (size_t k = 0; k < deltas.getBatchSize(); ++k)
            {
                const size_t inOff = k * mLayer.mDepth * mLayer.mHeight * mLayer.mWidth;
                const size_t outOff = minuendDelta.getBatchSize() == 1 ? 0 : k * minuendDelta.getDepth() * minuendDelta.getHeight() * minuendDelta.getWidth();
                gpu::tile(work.getKernelManager(),
                          mLayer.mTypeName + "[" + mLayer.mName + "backwardComputeImpl]",
                          mLayer.mDepth,
                          mLayer.mHeight,
                          mLayer.mWidth,
                          minuendDelta.getDepth(),
                          minuendDelta.getHeight(),
                          minuendDelta.getWidth(),
                          inOff,
                          outOff,
                          false,
                          deltas.getBuffer(),
                          minuendDelta.getBuffer());
            }
        }
        else
        {
            Common::axpy(&work.getKernelManager(),
                         mLayer.mTypeName + "[" + mLayer.mName + "::backwardComputeImpl]",
                         minuendDelta.getShape().total_size(),
                         1.0_dt,
                         deltas.getBuffer(),
                         1U,
                         minuendDelta.getBuffer(),
                         1U,
                         0U,
                         0U);
        }
    }
    // if (mLayer.mNetworkParams.isGradNeeded(mLayer.mInputs[1]))
    {
        auto& subtrahendDelta = work.getMemoryManager<MemoryManagerGPU>()(mLayer.mInputs[1].grad());
        if (mLayer.mBroadcastQuery[1])
        {
            auto& tmp = work.getMemoryManager<MemoryManagerGPU>()(mLayer.mBackwardTmpBufferName);
            Common::axpby(&work.getKernelManager(),
                          mLayer.mTypeName + "[" + mLayer.mName + "::backwardComputeImpl]",
                          deltas.getShape().total_size(),
                          -1.0_dt,
                          deltas.getBuffer(),
                          1U,
                          0.0_dt,
                          tmp.getBuffer(),
                          1U,
                          0U,
                          0U);
            for (size_t k = 0; k < deltas.getBatchSize(); ++k)
            {
                const size_t inOff = k * mLayer.mDepth * mLayer.mHeight * mLayer.mWidth;
                const size_t outOff = subtrahendDelta.getBatchSize() == 1 ? 0 : k * subtrahendDelta.getDepth() * subtrahendDelta.getHeight() * subtrahendDelta.getWidth();
                gpu::tile(work.getKernelManager(),
                          mLayer.mTypeName + "[" + mLayer.mName + "backwardComputeImpl]",
                          mLayer.mDepth,
                          mLayer.mHeight,
                          mLayer.mWidth,
                          subtrahendDelta.getDepth(),
                          subtrahendDelta.getHeight(),
                          subtrahendDelta.getWidth(),
                          inOff,
                          outOff,
                          false,
                          tmp.getBuffer(),
                          subtrahendDelta.getBuffer());
            }
        }
        else
        {
            Common::axpy(&work.getKernelManager(),
                         mLayer.mTypeName + "[" + mLayer.mName + "::backwardComputeImpl]",
                         subtrahendDelta.getShape().total_size(),
                         -1.0_dt,
                         deltas.getBuffer(),
                         1U,
                         subtrahendDelta.getBuffer(),
                         1U,
                         0U,
                         0U);
        }
    }
}

} // namespace raul
