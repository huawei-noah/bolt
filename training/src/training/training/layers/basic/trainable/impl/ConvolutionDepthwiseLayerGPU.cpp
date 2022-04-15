// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "ConvolutionDepthwiseLayerGPU.h"
#include "../ConvolutionDepthwiseLayer.h"

#include <training/opencl/GemmGPU.h>

namespace raul
{

cl::Buffer ConvolutionDepthwiseLayerGPU::mTmpBuffer;
cl::Buffer ConvolutionDepthwiseLayerGPU::mIm2ColBackwardBuffer;

void ConvolutionDepthwiseLayerGPU::initNotBSTensors()
{
    auto caller = mLayer.mTypeName + "[" + mLayer.mName + "::initNotBSTensors]";

    size_t tmpSize1 = gpu::gemm_temp_buffer_size(CblasNoTrans, CblasNoTrans, 1, mLayer.mOutputWidth * mLayer.mOutputHeight, mLayer.mKernelWidth * mLayer.mKernelHeight);
    size_t tmpSize2 = gpu::gemm_temp_buffer_size(CblasTrans, CblasNoTrans, mLayer.mKernelWidth * mLayer.mKernelHeight, mLayer.mOutputWidth * mLayer.mOutputHeight, 1);
    size_t tmpSize3 = gpu::gemm_temp_buffer_size(CblasNoTrans, CblasTrans, 1, mLayer.mKernelWidth * mLayer.mKernelHeight, mLayer.mOutputWidth * mLayer.mOutputHeight);

    size_t tmpSize = std::max(tmpSize1, std::max(tmpSize2, tmpSize3));
    cl_int status = CL_SUCCESS;
    size_t currentSize = !mTmpBuffer() ? 0 : mTmpBuffer.getInfo<CL_MEM_SIZE>(&status);
    Common::checkOpenCLStatus(status, caller, "error quering buffer size");

    if (currentSize < tmpSize)
    {
        mTmpBuffer = mLayer.mNetworkParams.mWorkflow.getKernelManager().createBuffer(tmpSize, caller);
    }

    currentSize = !mIm2ColBackwardBuffer() ? 0 : mIm2ColBackwardBuffer.getInfo<CL_MEM_SIZE>(&status);
    tmpSize1 = mLayer.mKernelWidth * mLayer.mKernelHeight * mLayer.mOutputWidth * mLayer.mOutputHeight * sizeof(dtype);
    const size_t widthCol = (mLayer.mInputWidth + 2 * mLayer.mPaddingW - mLayer.mKernelWidth) / mLayer.mStrideW + 1;
    const size_t heightCol = (mLayer.mInputHeight + 2 * mLayer.mPaddingH - mLayer.mKernelHeight) / mLayer.mStrideH + 1;
    tmpSize2 = widthCol * heightCol * mLayer.mKernelHeight * mLayer.mKernelWidth;
    tmpSize = std::max(tmpSize1, tmpSize2);
    Common::checkOpenCLStatus(status, caller, "error quering buffer size");

    if (currentSize < tmpSize)
    {
        mIm2ColBackwardBuffer = mLayer.mNetworkParams.mWorkflow.getKernelManager().createBuffer(tmpSize, caller);
    }
}

cl::Kernel ConvolutionDepthwiseLayerGPU::getForwardBiasKernel()
{
    Workflow& work = mLayer.mNetworkParams.mWorkflow;
    string name = mLayer.mTypeName / "forward";
    string kernelName = "bias_forward";

    auto& manager = work.getKernelManager();

    if (!manager.hasKernel(name, kernelName))
    {
        string source =
            R"(
                // bias has size [h]
                // data has size [N x h x w], work size is N x h
                __kernel void )" +
            kernelName + R"((
                    const int N,
                    const int h,
                    const int w,
                    __global const T *bias,
                    __global T *data)
                {
                    const int idy = get_global_id(0);
                    const int idx = get_global_id(1);

                    if (idx >= h || idy >= N)
                    {
                        return;
                    }
            
                    const int offset = idy * w * h;
                    const T b = bias[idx];
                    for (int i = 0; i < w; ++i)
	                {
                        data[offset + idx * w + i] += b;
                    }
                }
            )";
        manager.registerProgram(name, source);
    }

    return manager.getKernel(name, kernelName, mLayer.mTypeName + "[" + mLayer.mName + "::getForwardBiasKernel]");
}

cl::Kernel ConvolutionDepthwiseLayerGPU::getBackwardBiasKernel()
{
    Workflow& work = mLayer.mNetworkParams.mWorkflow;
    string name = mLayer.mTypeName / "backward";
    string kernelName = "bias_backward";

    auto& manager = work.getKernelManager();

    if (!manager.hasKernel(name, kernelName))
    {
        string source =
            R"(
                // bias has size [h]
                // data has size [N x h x w], work size is h
                __kernel void )" +
            kernelName + R"((
                    const int N,
                    const int h,
                    const int w,
                    __global const T *data,
                    __global T *bias)
                {
                    const int idx = get_global_id(0);
                    if (idx >= h )
                    {
                        return;
                    }
            
                    T sum = 0;
                    int stride = h * w;
                    int offset = idx * w;
                    for (int q = 0; q < N; ++q)
                    {
                        for (int i = 0; i < w; ++i)
	                    {
                            sum += data[offset + i];
                        }
                        offset += stride;
                    }
                    bias[idx] += sum;
                }
            )";
        manager.registerProgram(name, source);
    }

    return manager.getKernel(name, kernelName, mLayer.mTypeName + "[" + mLayer.mName + "::getBackwardBiasKernel]");
}

void ConvolutionDepthwiseLayerGPU::forwardComputeImpl(NetworkMode)
{
    Workflow& work = mLayer.mNetworkParams.mWorkflow;

    const size_t batchSize = work.getBatchSize();

    auto& output = work.getMemoryManager<MemoryManagerGPU>()(mLayer.mOutputName);
    const auto& inputs = work.getMemoryManager<MemoryManagerGPU>()(mLayer.mInputName);
    const auto& weights = work.getMemoryManager<MemoryManagerGPU>()(mLayer.mWeightsName);

    for (size_t q = 0; q < batchSize; ++q)
    {
        for (size_t w = 0; w < mLayer.mInputDepth; ++w)
        {
            auto& im2ColBuffer = work.getMemoryManager<MemoryManagerGPU>()(mLayer.mIm2ColForward[0]);

            size_t inputOffset = q * mLayer.mInputDepth * mLayer.mInputHeight * mLayer.mInputWidth + w * mLayer.mInputHeight * mLayer.mInputWidth;

            gpu::im2col(work.getKernelManager(),
                        mLayer.mName / "forward",
                        inputs.getBuffer(),
                        mLayer.mInputWidth,
                        mLayer.mInputHeight,
                        1,
                        mLayer.mKernelWidth,
                        mLayer.mKernelHeight,
                        mLayer.mStrideW,
                        mLayer.mStrideH,
                        mLayer.mPaddingW,
                        mLayer.mPaddingH,
                        im2ColBuffer.getBuffer(),
                        false,
                        inputOffset);

            gpu::gemm(work.getKernelManager(),
                      mLayer.mName / "forward",
                      CblasNoTrans,
                      CblasNoTrans,
                      1,
                      mLayer.mOutputWidth * mLayer.mOutputHeight,
                      mLayer.mKernelWidth * mLayer.mKernelHeight,
                      1.0_dt,
                      weights.getBuffer(),
                      im2ColBuffer.getBuffer(),
                      0_dt,
                      output.getBuffer(),
                      mTmpBuffer,
                      w * mLayer.mKernelHeight * mLayer.mKernelWidth,
                      0,
                      q * mLayer.mKernelsCount * mLayer.mOutputHeight * mLayer.mOutputWidth + w * mLayer.mOutputHeight * mLayer.mOutputWidth);
        }
    }

    if (mLayer.mUseBias)
    {
        const auto& biases = work.getMemoryManager<MemoryManagerGPU>()(mLayer.mBiasesName);

        cl::NDRange workSize(batchSize, mLayer.mKernelsCount, 1);
        work.getKernelManager().callKernel(getForwardBiasKernel(),
                                           workSize,
                                           mLayer.mName / "forward",
                                           (cl_int)batchSize,
                                           (cl_int)mLayer.mKernelsCount,
                                           (cl_int)(mLayer.mOutputHeight * mLayer.mOutputWidth),
                                           biases.getBuffer(),
                                           output.getBuffer());
    }
}

void ConvolutionDepthwiseLayerGPU::backwardComputeImpl()
{
    Workflow& work = mLayer.mNetworkParams.mWorkflow;

    const size_t batchSize = work.getBatchSize();

    const auto& deltas = work.getMemoryManager<MemoryManagerGPU>()(mLayer.mOutputName.grad());
    const auto& weights = work.getMemoryManager<MemoryManagerGPU>()(mLayer.mWeightsName);

    // prevDelta
    // if (mNetworkParams.isGradNeeded(mLayer.mInputName))
    {
        auto& prevLayerDelta = work.getMemoryManager<MemoryManagerGPU>()(mLayer.mInputName.grad());

        for (size_t i = 0; i < batchSize; ++i)
        {
            for (size_t w = 0; w < mLayer.mInputDepth; ++w)
            {
                gpu::gemm(work.getKernelManager(),
                          mLayer.mName / "backward",
                          CblasTrans,
                          CblasNoTrans,
                          mLayer.mKernelWidth * mLayer.mKernelHeight,
                          mLayer.mOutputWidth * mLayer.mOutputHeight,
                          1,
                          1.0_dt,
                          weights.getBuffer(),
                          deltas.getBuffer(),
                          0_dt,
                          mIm2ColBackwardBuffer,
                          mTmpBuffer,
                          w * mLayer.mKernelHeight * mLayer.mKernelWidth,
                          i * mLayer.mKernelsCount * mLayer.mOutputHeight * mLayer.mOutputWidth + w * mLayer.mOutputHeight * mLayer.mOutputWidth,
                          0);

                size_t outputOffset = i * mLayer.mInputDepth * mLayer.mInputHeight * mLayer.mInputWidth + w * mLayer.mInputHeight * mLayer.mInputWidth;

                gpu::col2im(work.getKernelManager(),
                            mLayer.mName / "backward",
                            mIm2ColBackwardBuffer,
                            mLayer.mInputWidth,
                            mLayer.mInputHeight,
                            1,
                            mLayer.mKernelWidth,
                            mLayer.mKernelHeight,
                            mLayer.mStrideW,
                            mLayer.mStrideH,
                            mLayer.mPaddingW,
                            mLayer.mPaddingH,
                            prevLayerDelta.getBuffer(),
                            false,
                            false,
                            0,
                            outputOffset);
            }
        }
    }

    // gradients and biases weights
    if (!mLayer.mFrozen)
    {
        // gradients
        const auto& inputs = work.getMemoryManager<MemoryManagerGPU>()(mLayer.mInputName);
        auto& gradWeights = work.getMemoryManager<MemoryManagerGPU>()(mLayer.mWeightsName.grad());

        for (size_t q = 0; q < batchSize; ++q)
        {
            for (size_t w = 0; w < mLayer.mInputDepth; ++w)
            {
                size_t inputOffset = q * mLayer.mInputDepth * mLayer.mInputHeight * mLayer.mInputWidth + w * mLayer.mInputHeight * mLayer.mInputWidth;
                gpu::im2col(work.getKernelManager(),
                            mLayer.mName / "backward",
                            inputs.getBuffer(),
                            mLayer.mInputWidth,
                            mLayer.mInputHeight,
                            1,
                            mLayer.mKernelWidth,
                            mLayer.mKernelHeight,
                            mLayer.mStrideW,
                            mLayer.mStrideH,
                            mLayer.mPaddingW,
                            mLayer.mPaddingH,
                            mIm2ColBackwardBuffer,
                            false,
                            inputOffset);

                gpu::gemm(work.getKernelManager(),
                          mLayer.mName / "backward",
                          CblasNoTrans,
                          CblasTrans,
                          1,
                          mLayer.mKernelWidth * mLayer.mKernelHeight,
                          mLayer.mOutputWidth * mLayer.mOutputHeight,
                          1.0_dt,
                          deltas.getBuffer(),
                          mIm2ColBackwardBuffer,
                          1_dt,
                          gradWeights.getBuffer(),
                          mTmpBuffer,
                          inputOffset,
                          0,
                          w * mLayer.mKernelHeight * mLayer.mKernelWidth);
            }
        }

        // biases
        if (mLayer.mUseBias)
        {
            auto& gradBiases = work.getMemoryManager<MemoryManagerGPU>()(mLayer.mBiasesName.grad());

            cl::NDRange workSize(mLayer.mKernelsCount, 1, 1);
            work.getKernelManager().callKernel(getBackwardBiasKernel(),
                                               workSize,
                                               mLayer.mName / "backward",
                                               (cl_int)batchSize,
                                               (cl_int)mLayer.mKernelsCount,
                                               (cl_int)(mLayer.mOutputHeight * mLayer.mOutputWidth),
                                               deltas.getBuffer(),
                                               gradBiases.getBuffer());
        }
    }
}

} // namespace raul