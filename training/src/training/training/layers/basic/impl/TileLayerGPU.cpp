// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "TileLayerGPU.h"
#include "../TileLayer.h"

#include <training/common/Common.h>
#include <training/opencl/GPUCommon.h>

namespace raul
{

TileLayerGPU::TileLayerGPU(TileLayer& layer)
    : mLayer(layer)
{
}

void TileLayerGPU::forwardComputeImpl(NetworkMode)
{
    auto& work = mLayer.mNetworkParams.mWorkflow;

    auto& output = work.getMemoryManager<MemoryManagerGPU>()(mLayer.mOutputs[0]);
    const auto& input = work.getMemoryManager<MemoryManagerGPU>()(mLayer.mInputs[0]);
    const size_t batchSize = work.getBatchSize();
    const size_t oDepth = output.getDepth();
    const size_t oHeight = output.getHeight();
    const size_t oWidth = output.getWidth();
    for (size_t i = 0; i < batchSize; ++i)
    {
        const size_t outOff = i * oDepth * oHeight * oWidth;
        const size_t inOff = outOff / mLayer.mRepeatDepth / mLayer.mRepeatHeight / mLayer.mRepeatWidth;
        gpu::tile(work.getKernelManager(),
                  mLayer.mTypeName + "[" + mLayer.mName + "forwardComputeImpl]",
                  oDepth / mLayer.mRepeatDepth,
                  oHeight / mLayer.mRepeatHeight,
                  oWidth / mLayer.mRepeatWidth,
                  oDepth,
                  oHeight,
                  oWidth,
                  inOff,
                  outOff,
                  true,
                  input.getBuffer(),
                  output.getBuffer());
    }
}

void TileLayerGPU::backwardComputeImpl()
{
    auto& work = mLayer.mNetworkParams.mWorkflow;

    const auto& deltas = work.getMemoryManager<MemoryManagerGPU>()(mLayer.mOutputs[0].grad());

    // if (mNetworkParams.isGradNeeded(mInputs[0]))
    {
        auto& prevLayerNabla = work.getMemoryManager<MemoryManagerGPU>()(mLayer.mInputs[0].grad());
        const size_t batchSize = work.getBatchSize();
        const size_t iDepth = deltas.getDepth();
        const size_t iHeight = deltas.getHeight();
        const size_t iWidth = deltas.getWidth();
        for (size_t i = 0; i < batchSize; ++i)
        {
            const size_t inOff = i * iDepth * iHeight * iWidth;
            const size_t outOff = inOff / mLayer.mRepeatDepth / mLayer.mRepeatHeight / mLayer.mRepeatWidth;
            gpu::tile(work.getKernelManager(),
                      mLayer.mTypeName + "[" + mLayer.mName + "backwardComputeImpl]",
                      iDepth,
                      iHeight,
                      iWidth,
                      iDepth / mLayer.mRepeatDepth,
                      iHeight / mLayer.mRepeatHeight,
                      iWidth / mLayer.mRepeatWidth,
                      inOff,
                      outOff,
                      false,
                      deltas.getBuffer(),
                      prevLayerNabla.getBuffer());
        }
    }
}

} // namespace raul
