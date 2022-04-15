// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "ConcatenationLayerGPU.h"
#include "../ConcatenationLayer.h"

#include <training/opencl/GPUCommon.h>

namespace raul
{

ConcatenationLayerGPU::ConcatenationLayerGPU(ConcatenationLayer& layer)
    : mLayer(layer)
{
}

void ConcatenationLayerGPU::forwardComputeImpl(NetworkMode)
{
    auto& work = mLayer.mNetworkParams.mWorkflow;

    auto& output = work.getMemoryManager<MemoryManagerGPU>()(mLayer.mOutputs[0]);
    const size_t oDepth = work.getDepth(mLayer.mOutputs[0]);
    const size_t oHeight = work.getHeight(mLayer.mOutputs[0]);
    const size_t oWidth = work.getWidth(mLayer.mOutputs[0]);
    const size_t concatDim = 2 - mLayer.mDimIndex;
    std::array<cl::Buffer, 4> inBuffs;
    std::array<size_t, 4> iWidth;
    std::array<size_t, 4> iHeight;
    std::array<size_t, 4> iOffset;
    std::array<size_t, 4> axisLen;
    const size_t num = mLayer.mInputs.size();
    const size_t bn = (num + 3) / 4;
    size_t batchOff = 0;
    for (size_t b = 0; b < work.getBatchSize(); ++b)
    {
        size_t outSize = 0;
        for (size_t i = 0; i < bn; ++i)
        {
            const size_t en = (i * 4 + 4 <= num) ? 4 : (num & 3);
            size_t axisMax = 0;
            for (size_t j = 0; j < en; ++j)
            {
                const auto& input = work.getMemoryManager<MemoryManagerGPU>()(mLayer.mInputs[i * 4 + j]);
                inBuffs[j] = input.getBuffer();
                iWidth[j] = work.getWidth(mLayer.mInputs[i * 4 + j]);
                iHeight[j] = work.getHeight(mLayer.mInputs[i * 4 + j]);
                iOffset[j] = b * work.getDepth(mLayer.mInputs[i * 4 + j]) * iHeight[j] * iWidth[j];
                axisLen[j] = input.getShape()[mLayer.mDimIndex + 1];

                if (concatDim == 0)
                {
                    axisLen[j] = (axisLen[j] + 3) / 4;
                }
                axisMax += axisLen[j];
            }
            axisMax -= axisLen[en - 1];
            gpu::concat(work.getKernelManager(),
                        mLayer.mTypeName + "[" + mLayer.mName + "forwardComputeImpl]",
                        concatDim,
                        oDepth,
                        oHeight,
                        oWidth,
                        axisMax,
                        en,
                        outSize,
                        batchOff,
                        iWidth,
                        iHeight,
                        iOffset,
                        axisLen,
                        inBuffs,
                        output.getBuffer());

            switch (mLayer.mDirection)
            {
                case Dimension::Width:
                    outSize += std::accumulate(iWidth.begin(), iWidth.begin() + en, static_cast<size_t>(0));
                    break;
                case Dimension::Height:
                    outSize += oWidth * (axisMax + axisLen[en - 1]);
                    break;
                case Dimension::Depth:
                    outSize += oHeight * oWidth * (axisMax + axisLen[en - 1]);
                    break;
                default:
                    throw std::runtime_error("ConcatenationLayer[forwardCompute]: unknown dim");
            }
        }
        batchOff += oDepth * oHeight * oWidth;
    }
}

void ConcatenationLayerGPU::backwardComputeImpl()
{
    auto& work = mLayer.mNetworkParams.mWorkflow;

    const auto& delta = work.getMemoryManager<MemoryManagerGPU>()(mLayer.mOutputs[0].grad());
    const auto iDepth = work.getDepth(mLayer.mOutputs[0].grad());
    const auto iHeight = work.getHeight(mLayer.mOutputs[0].grad());
    const auto iWidth = work.getWidth(mLayer.mOutputs[0].grad());
    const auto numOfSlices = mLayer.mInputs.size();
    const auto axisNum = 2 - mLayer.mDimIndex;
    std::array<size_t, 4> oWidth;
    std::array<size_t, 4> oHeight;
    std::array<size_t, 4> oOffset;
    std::array<size_t, 4> axisLen;
    size_t inSize = 0;
    for (size_t i = 0; i < numOfSlices; i += 4)
    {
        const size_t sliceNum = ((i + 4) <= numOfSlices) ? 4 : (numOfSlices & 3);
        size_t axisMax = 0;
        size_t axisTotal = 0;
        size_t sliceLen = 0;
        for (size_t j = 0; j < sliceNum; ++j)
        {
            auto& prevLayerDelta = work.getMemoryManager<MemoryManagerGPU>()(mLayer.mInputs[j + i].grad());
            oWidth[j] = work.getWidth(mLayer.mInputs[j + i].grad());
            oHeight[j] = work.getHeight(mLayer.mInputs[j + i].grad());

            axisLen[j] = prevLayerDelta.getShape()[mLayer.mDimIndex + 1];
            sliceLen += axisLen[j];
            if (mLayer.mDirection == Dimension::Width)
            {
                axisLen[j] = (axisLen[j] + 3) / 4;
            }
            axisTotal += axisLen[j];
        }
        axisMax = axisTotal - axisLen[sliceNum - 1];

        size_t batchOff = 0;
        for (size_t b = 0; b < work.getBatchSize(); ++b)
        {
            for (size_t j = 0; j < sliceNum; ++j)
            {
                oOffset[j] = b * work.getDepth(mLayer.mInputs[j + i].grad()) * oHeight[j] * oWidth[j];
            }
            gpu::slice(work.getKernelManager(),
                       mLayer.mTypeName + "[" + mLayer.mName + "backwardComputeImpl]",
                       axisNum,
                       iDepth,
                       iHeight,
                       iWidth,
                       axisMax,
                       sliceNum,
                       inSize,
                       batchOff,
                       axisTotal,
                       oWidth,
                       oHeight,
                       oOffset,
                       axisLen,
                       delta.getBuffer(),
                       mLayer.mTmps);
            batchOff += iDepth * iHeight * iWidth;
        }

        for (size_t j = 0; j < sliceNum; ++j)
        {
            auto& prevLayerDelta = work.getMemoryManager<MemoryManagerGPU>()(mLayer.mInputs[j + i].grad());
            Common::axpy(&work.getKernelManager(),
                         mLayer.mTypeName + "[" + mLayer.mName + "::backwardComputeImpl]",
                         prevLayerDelta.getShape().total_size(),
                         1.0_dt,
                         mLayer.mTmps[j],
                         1U,
                         prevLayerDelta.getBuffer(),
                         1U,
                         0U,
                         0U);
        }

        switch (mLayer.mDirection)
        {
            case Dimension::Depth:
                inSize += sliceLen * iWidth * iHeight;
                break;
            case Dimension::Height:
                inSize += sliceLen * iWidth;
                break;
            case Dimension::Width:
                inSize += sliceLen;
                break;
            default:
                throw std::runtime_error("SlicerLayer[forwardCompute]: unknown dim");
        }
    }
}

} // namespace raul