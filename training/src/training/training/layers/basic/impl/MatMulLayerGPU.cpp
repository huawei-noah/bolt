// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "MatMulLayerGPU.h"
#include "../MatMulLayer.h"

#include <GemmGPU.h>

namespace raul
{

MatMulLayerGPU::MatMulLayerGPU(MatMulLayer& layer)
    : mLayer(layer)
{
}

void MatMulLayerGPU::onBatchSizeChanged(size_t)
{
    auto& work = mLayer.mNetworkParams.mWorkflow;

    auto caller = mLayer.mTypeName + "[" + mLayer.mName + "::onBatchSizeChanged]";

    auto& input1 = work.getMemoryManager<MemoryManagerGPU>()(mLayer.mInputs[0]);
    auto& input2 = work.getMemoryManager<MemoryManagerGPU>()(mLayer.mInputs[1]);
    auto& output = work.getMemoryManager<MemoryManagerGPU>()(mLayer.mOutputs[0]);

    size_t tmpSize1 = gpu::gemm_temp_buffer_size(CblasNoTrans, CblasTrans, input1.getHeight(), input2.getWidth(), input1.getWidth());
    size_t tmpSize2 = gpu::gemm_temp_buffer_size(CblasNoTrans, CblasTrans, output.getHeight(), input2.getHeight(), output.getWidth());
    size_t tmpSize3 = gpu::gemm_temp_buffer_size(CblasTrans, CblasNoTrans, input1.getWidth(), output.getWidth(), input1.getHeight());

    size_t tmpSize = std::max(tmpSize1, std::max(tmpSize2, tmpSize3));
    cl_int status = CL_SUCCESS;
    size_t currentSize = !mLayer.mTmpBuffer() ? 0 : mLayer.mTmpBuffer.getInfo<CL_MEM_SIZE>(&status);
    Common::checkOpenCLStatus(status, caller, "error quering buffer size");

    if (currentSize < tmpSize)
    {
        mLayer.mTmpBuffer = mLayer.mNetworkParams.mWorkflow.getKernelManager().createBuffer(tmpSize, caller);
    }
}

void MatMulLayerGPU::forwardComputeImpl(NetworkMode)
{
    auto& work = mLayer.mNetworkParams.mWorkflow;

    const size_t batchSize = work.getBatchSize();

    auto caller = mLayer.mTypeName + "[" + mLayer.mName + "::forwardComputeImpl]";
    auto& kernelManager = work.getKernelManager();
    auto& output = work.getMemoryManager<MemoryManagerGPU>()(mLayer.mOutputs[0]);
    kernelManager.fillBuffer(output.getBuffer(), 0_dt, caller);

    auto& input1 = work.getMemoryManager<MemoryManagerGPU>()(mLayer.mInputs[0]);
    auto& input2 = work.getMemoryManager<MemoryManagerGPU>()(mLayer.mInputs[1]);

    size_t size = mLayer.mDepth * batchSize;

    size_t input1Stride = input1.getWidth() * input1.getHeight();
    size_t input2Stride = input2.getWidth() * input2.getHeight();
    size_t outputStride = output.getWidth() * output.getHeight();

    for (size_t q = 0; q < size; ++q)
    {
        Common::gemm(&kernelManager,
                     caller / "gemm" / to_string(q),
                     CblasNoTrans,
                     CblasNoTrans,
                     input1.getHeight(),
                     input2.getWidth(),
                     input1.getWidth(),
                     mLayer.mCoeff,
                     input1.getBuffer(),
                     input2.getBuffer(),
                     0.0_dt,
                     output.getBuffer(),
                     mLayer.mTmpBuffer,
                     q * input1Stride,
                     q * input2Stride,
                     q * outputStride);
    }
}

void MatMulLayerGPU::backwardComputeImpl()
{
    auto& work = mLayer.mNetworkParams.mWorkflow;

    auto caller = mLayer.mTypeName + "[" + mLayer.mName + "::backwardComputeImpl]";

    const size_t batchSize = work.getBatchSize();

    auto& kernelManager = work.getKernelManager();

    auto& deltas = work.getMemoryManager<MemoryManagerGPU>()(mLayer.mOutputs[0].grad());
    auto& input1 = work.getMemoryManager<MemoryManagerGPU>()(mLayer.mInputs[0]);
    auto& input2 = work.getMemoryManager<MemoryManagerGPU>()(mLayer.mInputs[1]);
    auto& grad1 = work.getMemoryManager<MemoryManagerGPU>()(mLayer.mInputs[0].grad());
    auto& grad2 = work.getMemoryManager<MemoryManagerGPU>()(mLayer.mInputs[1].grad());
    size_t size = mLayer.mDepth * batchSize;

    size_t deltasStride = input1.getHeight() * input2.getWidth();
    size_t input1Stride = input1.getHeight() * input1.getWidth();
    size_t input2Stride = input2.getHeight() * input2.getWidth();

    // if (mLayer.mNetworkParams.isGradNeeded(mLayer.mInputs[0].grad()))
    {

        for (size_t q = 0; q < size; ++q)
        {
            Common::gemm(&kernelManager,
                         caller / "input1" / to_string(q),
                         CblasNoTrans,
                         CblasTrans,
                         deltas.getHeight(),
                         input2.getHeight(),
                         deltas.getWidth(),
                         mLayer.mCoeff,
                         deltas.getBuffer(),
                         input2.getBuffer(),
                         1.0_dt,
                         grad1.getBuffer(),
                         mLayer.mTmpBuffer,
                         q * deltasStride,
                         q * input2Stride,
                         q * input1Stride);
        }
    }

    // if (mLayer.mNetworkParams.isGradNeeded(mLayer.mInputs[1].grad()))
    {
        for (size_t q = 0; q < size; ++q)
        {
            Common::gemm(&kernelManager,
                         caller / "input2" / to_string(q),
                         CblasTrans,
                         CblasNoTrans,
                         input1.getWidth(),
                         deltas.getWidth(),
                         input1.getHeight(),
                         mLayer.mCoeff,
                         input1.getBuffer(),
                         deltas.getBuffer(),
                         1.0_dt,
                         grad2.getBuffer(),
                         mLayer.mTmpBuffer,
                         q * input1Stride,
                         q * deltasStride,
                         q * input2Stride);
        }
    }
}

} // namespace raul
