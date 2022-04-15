// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "ElementWiseMulLayerGPU.h"
#include "../ElementWiseMulLayer.h"

#include <training/opencl/GPUCommon.h>

namespace raul
{

ElementWiseMulLayerGPU::ElementWiseMulLayerGPU(ElementWiseMulLayer& layer)
    : mLayer(layer)
{
}

void ElementWiseMulLayerGPU::forwardComputeImpl(NetworkMode)
{
    mLayer.determineBroadcastFlags();

    if (!mLayer.mBroadcast && std::any_of(mLayer.mBroadcastQuery.begin(), mLayer.mBroadcastQuery.end(), [](const auto& needToBroadcast) { return needToBroadcast; }))
    {
        THROW(mLayer.mTypeName, mLayer.mName, "input size mismatch");
    }

    auto& work = mLayer.mNetworkParams.mWorkflow;

    auto& output = work.getMemoryManager<MemoryManagerGPU>()(mLayer.mOutputs[0]);
    const auto& input0 = work.getMemoryManager<MemoryManagerGPU>()(mLayer.mInputs[0]);
    std::array<size_t, 4> iBatches{ input0.getBatchSize(), 1, 1, 1 };
    std::array<size_t, 4> iDepths{ input0.getDepth(), 1, 1, 1 };
    std::array<size_t, 4> iHeights{ input0.getHeight(), 1, 1, 1 };
    std::array<size_t, 4> iWidths{ input0.getWidth(), 1, 1, 1 };
    std::array<cl::Buffer, 4> inBuffs;
    inBuffs[0] = input0.getBuffer();
    const size_t inputNum = mLayer.mInputs.size();
    size_t sliceNum = 1;
    for (size_t i = 1; i < inputNum; ++i)
    {
        const auto& input = work.getMemoryManager<MemoryManagerGPU>()(mLayer.mInputs[i]);
        iBatches[sliceNum] = input.getBatchSize();
        iDepths[sliceNum] = input.getDepth();
        iHeights[sliceNum] = input.getHeight();
        iWidths[sliceNum] = input.getWidth();
        inBuffs[sliceNum++] = input.getBuffer();
        if (sliceNum == 4 || i == inputNum - 1)
        {
            gpu::eltwiseMulOp(work.getKernelManager(),
                              mLayer.mTypeName + "[" + mLayer.mName + "::forwardComputeImpl]",
                              output.getBatchSize(),
                              mLayer.mDepth,
                              mLayer.mHeight,
                              mLayer.mWidth,
                              sliceNum,
                              iBatches,
                              iDepths,
                              iHeights,
                              iWidths,
                              inBuffs,
                              output.getBuffer());
            inBuffs[0] = output.getBuffer();
            sliceNum = 1;
        }
    }
}

void ElementWiseMulLayerGPU::backwardComputeImpl()
{
    auto& work = mLayer.mNetworkParams.mWorkflow;

    const auto& deltas = work.getMemoryManager<MemoryManagerGPU>()(mLayer.mOutputs[0].grad());
    auto& tmp = work.getMemoryManager<MemoryManagerGPU>()(mLayer.mBackwardTmpBufferName);
    std::array<size_t, 4> iBatches{ work.getBatchSize(), 1, 1, 1 };
    std::array<size_t, 4> iDepths{ mLayer.mDepth, 1, 1, 1 };
    std::array<size_t, 4> iHeights{ mLayer.mHeight, 1, 1, 1 };
    std::array<size_t, 4> iWidths{ mLayer.mWidth, 1, 1, 1 };
    std::array<cl::Buffer, 4> inBuffs;
    const size_t inputNum = mLayer.mInputs.size();
    for (size_t i = 0; i < inputNum; ++i)
    {
        // if (mLayer.mNetworkParams.isGradNeeded(mLayer.mInputs[i]))
        {
            auto& prevLayerDelta = work.getMemoryManager<MemoryManagerGPU>()(mLayer.mInputs[i].grad());
            inBuffs[0] = deltas.getBuffer();
            size_t slicesNum = 1;
            for (size_t j = 0; j < inputNum; ++j)
            {
                if (i != j)
                {
                    const auto& input = work.getMemoryManager<MemoryManagerGPU>()(mLayer.mInputs[j]);
                    iBatches[slicesNum] = input.getBatchSize();
                    iDepths[slicesNum] = input.getDepth();
                    iHeights[slicesNum] = input.getHeight();
                    iWidths[slicesNum] = input.getWidth();
                    inBuffs[slicesNum++] = input.getBuffer();
                }

                if (slicesNum == 4 || j == inputNum - 1)
                {
                    gpu::eltwiseMulOp(work.getKernelManager(),
                                      mLayer.mTypeName + "[" + mLayer.mName + "::backwardComputeImpl]",
                                      deltas.getBatchSize(),
                                      mLayer.mDepth,
                                      mLayer.mHeight,
                                      mLayer.mWidth,
                                      slicesNum,
                                      iBatches,
                                      iDepths,
                                      iHeights,
                                      iWidths,
                                      inBuffs,
                                      tmp.getBuffer());
                    inBuffs[0] = tmp.getBuffer();
                    slicesNum = 1;
                }
            }
            if (mLayer.mBroadcastQuery[i])
            {
                for (size_t k = 0; k < deltas.getBatchSize(); ++k)
                {
                    const size_t inOff = k * mLayer.mDepth * mLayer.mHeight * mLayer.mWidth;
                    const size_t outOff = prevLayerDelta.getBatchSize() == 1 ? 0 : k * prevLayerDelta.getDepth() * prevLayerDelta.getHeight() * prevLayerDelta.getWidth();
                    gpu::tile(work.getKernelManager(),
                              mLayer.mTypeName + "[" + mLayer.mName + "backwardComputeImpl]",
                              mLayer.mDepth,
                              mLayer.mHeight,
                              mLayer.mWidth,
                              prevLayerDelta.getDepth(),
                              prevLayerDelta.getHeight(),
                              prevLayerDelta.getWidth(),
                              inOff,
                              outOff,
                              false,
                              tmp.getBuffer(),
                              prevLayerDelta.getBuffer());
                }
            }
            else
            {
                Common::axpy(&work.getKernelManager(),
                             mLayer.mTypeName + "[" + mLayer.mName + "::backwardComputeImpl]",
                             prevLayerDelta.getShape().total_size(),
                             1.0_dt,
                             tmp.getBuffer(),
                             1U,
                             prevLayerDelta.getBuffer(),
                             1U,
                             0U,
                             0U);
            }
        }
    }
}

} // namespace raul
