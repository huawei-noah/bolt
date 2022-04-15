// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "ElementWiseDivLayerGPU.h"
#include "../ElementWiseDivLayer.h"

#include <training/opencl/GPUCommon.h>

namespace raul
{

ElementWiseDivLayerGPU::ElementWiseDivLayerGPU(ElementWiseDivLayer& layer)
    : mLayer(layer)
{
}

void ElementWiseDivLayerGPU::forwardComputeImpl(NetworkMode)
{
    mLayer.determineBroadcastFlags();

    if (!mLayer.mBroadcast && (mLayer.mBroadcastQuery[0] || mLayer.mBroadcastQuery[1]))
    {
        THROW(mLayer.mTypeName, mLayer.mName, "input size mismatch");
    }

    auto& work = mLayer.mNetworkParams.mWorkflow;

    auto& output = work.getMemoryManager<MemoryManagerGPU>()(mLayer.mOutputs[0]);
    const auto& dividend = work.getMemoryManager<MemoryManagerGPU>()(mLayer.mInputs[0]);
    const auto& divisor = work.getMemoryManager<MemoryManagerGPU>()(mLayer.mInputs[1]);
    gpu::eltwiseDivOp(work.getKernelManager(),
                      mLayer.mTypeName + "[" + mLayer.mName + "::forwardComputeImpl]",
                      output.getBatchSize(),
                      mLayer.mDepth,
                      mLayer.mHeight,
                      mLayer.mWidth,
                      dividend.getBatchSize(),
                      dividend.getDepth(),
                      dividend.getHeight(),
                      dividend.getWidth(),
                      dividend.getBuffer(),
                      divisor.getBatchSize(),
                      divisor.getDepth(),
                      divisor.getHeight(),
                      divisor.getWidth(),
                      divisor.getBuffer(),
                      output.getBuffer());
}

void ElementWiseDivLayerGPU::backwardComputeImpl()
{
    auto& work = mLayer.mNetworkParams.mWorkflow;

    const auto& deltas = work.getMemoryManager<MemoryManagerGPU>()(mLayer.mOutputs[0].grad());
    auto& tmp = work.getMemoryManager<MemoryManagerGPU>()(mLayer.mBackwardTmpBufferName);
    // if (mLayer.mNetworkParams.isGradNeeded(mLayer.mInputs[0]))
    {
        auto& dividendNabla = work.getMemoryManager<MemoryManagerGPU>()(mLayer.mInputs[0].grad());
        const auto& divisor = work.getMemoryManager<MemoryManagerGPU>()(mLayer.mInputs[1]);
        gpu::eltwiseDivOp(work.getKernelManager(),
                          mLayer.mTypeName + "[" + mLayer.mName + "::backwardComputeImpl]",
                          deltas.getBatchSize(),
                          mLayer.mDepth,
                          mLayer.mHeight,
                          mLayer.mWidth,
                          deltas.getBatchSize(),
                          mLayer.mDepth,
                          mLayer.mHeight,
                          mLayer.mWidth,
                          deltas.getBuffer(),
                          divisor.getBatchSize(),
                          divisor.getDepth(),
                          divisor.getHeight(),
                          divisor.getWidth(),
                          divisor.getBuffer(),
                          tmp.getBuffer());

        if (mLayer.mBroadcastQuery[0])
        {
            for (size_t k = 0; k < deltas.getBatchSize(); ++k)
            {
                const size_t inOff = k * mLayer.mDepth * mLayer.mHeight * mLayer.mWidth;
                const size_t outOff = dividendNabla.getBatchSize() == 1 ? 0 : k * dividendNabla.getDepth() * dividendNabla.getHeight() * dividendNabla.getWidth();
                gpu::tile(work.getKernelManager(),
                          mLayer.mTypeName + "[" + mLayer.mName + "backwardComputeImpl]",
                          mLayer.mDepth,
                          mLayer.mHeight,
                          mLayer.mWidth,
                          dividendNabla.getDepth(),
                          dividendNabla.getHeight(),
                          dividendNabla.getWidth(),
                          inOff,
                          outOff,
                          false,
                          tmp.getBuffer(),
                          dividendNabla.getBuffer());
            }
        }
        else
        {
            Common::axpy(&work.getKernelManager(),
                         mLayer.mTypeName + "[" + mLayer.mName + "::backwardComputeImpl]",
                         dividendNabla.getShape().total_size(),
                         1.0_dt,
                         tmp.getBuffer(),
                         1U,
                         dividendNabla.getBuffer(),
                         1U,
                         0U,
                         0U);
        }
    }

    // if (mLayer.mNetworkParams.isGradNeeded(mLayer.mInputs[1]))
    {
        auto& divisorNabla = work.getMemoryManager<MemoryManagerGPU>()(mLayer.mInputs[1].grad());
        const auto& dividend = work.getMemoryManager<MemoryManagerGPU>()(mLayer.mInputs[0]);
        const auto& divisor = work.getMemoryManager<MemoryManagerGPU>()(mLayer.mInputs[1]);
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
        gpu::eltwiseDivOp(work.getKernelManager(),
                          mLayer.mTypeName + "[" + mLayer.mName + "::backwardComputeImpl]",
                          deltas.getBatchSize(),
                          mLayer.mDepth,
                          mLayer.mHeight,
                          mLayer.mWidth,
                          deltas.getBatchSize(),
                          mLayer.mDepth,
                          mLayer.mHeight,
                          mLayer.mWidth,
                          tmp.getBuffer(),
                          divisor.getBatchSize(),
                          divisor.getDepth(),
                          divisor.getHeight(),
                          divisor.getWidth(),
                          divisor.getBuffer(),
                          tmp.getBuffer());
        gpu::eltwiseDivOp(work.getKernelManager(),
                          mLayer.mTypeName + "[" + mLayer.mName + "::backwardComputeImpl]",
                          deltas.getBatchSize(),
                          mLayer.mDepth,
                          mLayer.mHeight,
                          mLayer.mWidth,
                          deltas.getBatchSize(),
                          mLayer.mDepth,
                          mLayer.mHeight,
                          mLayer.mWidth,
                          tmp.getBuffer(),
                          divisor.getBatchSize(),
                          divisor.getDepth(),
                          divisor.getHeight(),
                          divisor.getWidth(),
                          divisor.getBuffer(),
                          tmp.getBuffer());
        gpu::eltwiseMulOp(work.getKernelManager(),
                          mLayer.mTypeName + "[" + mLayer.mName + "::backwardComputeImpl]",
                          deltas.getBatchSize(),
                          mLayer.mDepth,
                          mLayer.mHeight,
                          mLayer.mWidth,
                          2,
                          { deltas.getBatchSize(), dividend.getBatchSize() },
                          { mLayer.mDepth, dividend.getDepth() },
                          { mLayer.mHeight, dividend.getHeight() },
                          { mLayer.mWidth, dividend.getWidth() },
                          { tmp.getBuffer(), dividend.getBuffer() },
                          tmp.getBuffer());

        if (mLayer.mBroadcastQuery[1])
        {
            for (size_t k = 0; k < deltas.getBatchSize(); ++k)
            {
                const size_t inOff = k * mLayer.mDepth * mLayer.mHeight * mLayer.mWidth;
                const size_t outOff = divisorNabla.getBatchSize() == 1 ? 0 : k * divisorNabla.getDepth() * divisorNabla.getHeight() * divisorNabla.getWidth();
                gpu::tile(work.getKernelManager(),
                          mLayer.mTypeName + "[" + mLayer.mName + "backwardComputeImpl]",
                          mLayer.mDepth,
                          mLayer.mHeight,
                          mLayer.mWidth,
                          divisorNabla.getDepth(),
                          divisorNabla.getHeight(),
                          divisorNabla.getWidth(),
                          inOff,
                          outOff,
                          false,
                          tmp.getBuffer(),
                          divisorNabla.getBuffer());
            }
        }
        else
        {
            Common::axpy(&work.getKernelManager(),
                         mLayer.mTypeName + "[" + mLayer.mName + "::backwardComputeImpl]",
                         divisorNabla.getShape().total_size(),
                         1.0_dt,
                         tmp.getBuffer(),
                         1U,
                         divisorNabla.getBuffer(),
                         1U,
                         0U,
                         0U);
        }
    }
}

} // namespace raul
