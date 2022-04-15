// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "Convolution1DLayerGPU.h"
#include "../Convolution1DLayer.h"

#include <GPUCommon.h>
#include <GemmGPU.h>

namespace raul
{

cl::Buffer Convolution1DLayerGPU::mTmpBuffer;

Convolution1DLayerGPU::Convolution1DLayerGPU(Convolution1DLayer& layer)
    : mLayer(layer)
{
}

void Convolution1DLayerGPU::initNotBSTensors()
{
    Workflow& work = mLayer.mNetworkParams.mWorkflow;
    string kernelName = "bias_forward";
    auto& manager = work.getKernelManager();

    if (!manager.hasKernel(mLayer.mTypeName, kernelName))
    {
        string source =
#include "kernels/conv1d.cl"
            ;
        string opt = mLayer.mTFStyle ? "-DTF_STYLE" : "";
        manager.registerProgram(mLayer.mTypeName, source, opt);
    }

    size_t tmpSize1 = 0;
    size_t tmpSize2 = 0;
    size_t tmpSize3 = 0;

    if (mLayer.mTFStyle)
    {
        tmpSize1 =
            gpu::gemm_temp_buffer_size(CblasTrans, CblasTrans, mLayer.mOutputSize, mLayer.mOutputChannels / mLayer.mGroups, mLayer.mEffectiveReceptiveField * mLayer.mInputChannels / mLayer.mGroups);
        tmpSize2 =
            gpu::gemm_temp_buffer_size(CblasTrans, CblasTrans, mLayer.mEffectiveReceptiveField * mLayer.mInputChannels / mLayer.mGroups, mLayer.mOutputSize, mLayer.mOutputChannels / mLayer.mGroups);
        tmpSize3 =
            gpu::gemm_temp_buffer_size(CblasTrans, CblasTrans, mLayer.mOutputChannels / mLayer.mGroups, mLayer.mEffectiveReceptiveField * mLayer.mInputChannels / mLayer.mGroups, mLayer.mOutputSize);
    }
    else
    {
        tmpSize1 = gpu::gemm_temp_buffer_size(
            CblasNoTrans, CblasNoTrans, mLayer.mOutputChannels / mLayer.mGroups, mLayer.mOutputSize, mLayer.mEffectiveReceptiveField * mLayer.mInputChannels / mLayer.mGroups);
        tmpSize2 =
            gpu::gemm_temp_buffer_size(CblasTrans, CblasNoTrans, mLayer.mEffectiveReceptiveField * mLayer.mInputChannels / mLayer.mGroups, mLayer.mOutputSize, mLayer.mOutputChannels / mLayer.mGroups);
        tmpSize3 =
            gpu::gemm_temp_buffer_size(CblasNoTrans, CblasTrans, mLayer.mOutputChannels / mLayer.mGroups, mLayer.mEffectiveReceptiveField * mLayer.mInputChannels / mLayer.mGroups, mLayer.mOutputSize);
    }

    auto caller = mLayer.mTypeName + "[" + mLayer.mName + "::initNotBSTensors]";
    size_t tmpSize = std::max(tmpSize1, std::max(tmpSize2, tmpSize3));
    cl_int status = CL_SUCCESS;
    size_t currentSize = !mTmpBuffer() ? 0 : mTmpBuffer.getInfo<CL_MEM_SIZE>(&status);
    Common::checkOpenCLStatus(status, caller, "error quering buffer size");

    if (currentSize < tmpSize)
    {
        mTmpBuffer = mLayer.mNetworkParams.mWorkflow.getKernelManager().createBuffer(tmpSize, caller);
    }
}

cl::Kernel Convolution1DLayerGPU::getKernel(Name name)
{
    return mLayer.mNetworkParams.mWorkflow.getKernelManager().getKernel(mLayer.mTypeName, name, mLayer.mTypeName + "[" + mLayer.mName + "::getKernel]");
}

void Convolution1DLayerGPU::forwardComputeImpl(NetworkMode)
{
    Workflow& work = mLayer.mNetworkParams.mWorkflow;

    const size_t batchSize = work.getBatchSize();
    auto& memory_manager = mLayer.mNetworkParams.mMemoryManagerGPU;
    auto& output = memory_manager(mLayer.mOutputName);
    const auto& inputs = memory_manager(mLayer.mInputName);
    const auto& weights = memory_manager(mLayer.mWeightsName);

    if (mLayer.mDilationEnabled)
    {
        auto& dilatedKernelsWeights = memory_manager(mLayer.mDilationTensor);
        work.getKernelManager().fillBuffer(dilatedKernelsWeights.getBuffer(), 0_dt, mLayer.mName / "forward");
        cl::NDRange workSize(mLayer.mOutputChannels, mLayer.mInputChannels / mLayer.mGroups, 1);
        work.getKernelManager().callKernel(getKernel("dilation_forward"),
                                           workSize,
                                           mLayer.mName / "forward",
                                           (cl_int)mLayer.mOutputChannels,
                                           (cl_int)(mLayer.mInputChannels / mLayer.mGroups),
                                           (cl_int)mLayer.mKernelSize,
                                           (cl_int)mLayer.mEffectiveReceptiveField,
                                           mLayer.mTFStyle ? (cl_int)(mLayer.mOutputChannels * mLayer.mInputChannels / mLayer.mGroups) : (cl_int)1,
                                           (cl_int)mLayer.mDilation,
                                           weights.getBuffer(),
                                           dilatedKernelsWeights.getBuffer());

        for (size_t q = 0; q < batchSize; ++q)
        {
            auto& im2ColFor = memory_manager(mLayer.mIm2ColForward[0]);
            size_t inputOffset = q * mLayer.mInputSize * mLayer.mInputChannels;

            gpu::im2col(work.getKernelManager(),
                        mLayer.mName / "forward" / to_string(q),
                        inputs.getBuffer(),
                        mLayer.mInputSize,
                        1u,
                        mLayer.mInputChannels,
                        mLayer.mEffectiveReceptiveField,
                        1u,
                        mLayer.mStride,
                        1u,
                        mLayer.mPadding,
                        0,
                        im2ColFor.getBuffer(),
                        mLayer.mTFStyle,
                        inputOffset);

            for (size_t group = 0; group < mLayer.mGroups; ++group)
            {
                if (mLayer.mTFStyle)
                {
                    Common::gemm(&work.getKernelManager(),
                                 mLayer.mName / "forward" / to_string(q) / to_string(group),
                                 CblasTrans,
                                 CblasTrans,
                                 mLayer.mOutputSize,
                                 mLayer.mOutputChannels / mLayer.mGroups,
                                 mLayer.mEffectiveReceptiveField * mLayer.mInputChannels / mLayer.mGroups,
                                 1.0_dt,
                                 im2ColFor.getBuffer(),
                                 dilatedKernelsWeights.getBuffer(),
                                 0_dt,
                                 output.getBuffer(),
                                 mTmpBuffer,
                                 group * mLayer.mInputChannels / mLayer.mGroups * mLayer.mEffectiveReceptiveField * mLayer.mOutputSize,
                                 group * mLayer.mOutputChannels / mLayer.mGroups * mLayer.mInputChannels / mLayer.mGroups * mLayer.mEffectiveReceptiveField,
                                 q * mLayer.mOutputSize * mLayer.mOutputChannels + group * mLayer.mOutputChannels / mLayer.mGroups);
                }
                else
                {
                    Common::gemm(&work.getKernelManager(),
                                 mLayer.mName / "forward" / to_string(q) / to_string(group),
                                 CblasNoTrans,
                                 CblasNoTrans,
                                 mLayer.mOutputChannels / mLayer.mGroups,
                                 mLayer.mOutputSize,
                                 mLayer.mEffectiveReceptiveField * mLayer.mInputChannels / mLayer.mGroups,
                                 1.0_dt,
                                 dilatedKernelsWeights.getBuffer(),
                                 im2ColFor.getBuffer(),
                                 0_dt,
                                 output.getBuffer(),
                                 mTmpBuffer,
                                 group * mLayer.mOutputChannels / mLayer.mGroups * mLayer.mInputChannels / mLayer.mGroups * mLayer.mEffectiveReceptiveField,
                                 group * mLayer.mInputChannels / mLayer.mGroups * mLayer.mEffectiveReceptiveField * mLayer.mOutputSize,
                                 q * mLayer.mOutputSize * mLayer.mOutputChannels + group * mLayer.mOutputChannels / mLayer.mGroups * mLayer.mOutputSize);
                }
            }
        }
    }
    else if (mLayer.mTFStyle)
    {
        auto& tempWeights = memory_manager(mLayer.mTempWeights);
        gpu::swapIndices3d(
            work.getKernelManager(), mLayer.mName / "forward", mLayer.mOutputChannels, mLayer.mInputChannels / mLayer.mGroups, mLayer.mKernelSize, weights.getBuffer(), tempWeights.getBuffer());

        for (size_t q = 0; q < batchSize; ++q)
        {
            auto& im2ColFor = memory_manager(mLayer.mIm2ColForward[0]);

            size_t inputOffset = q * mLayer.mInputSize * mLayer.mInputChannels;

            gpu::im2col(work.getKernelManager(),
                        mLayer.mName / "forward" / to_string(q),
                        inputs.getBuffer(),
                        mLayer.mInputSize,
                        1u,
                        mLayer.mInputChannels,
                        mLayer.mEffectiveReceptiveField,
                        1u,
                        mLayer.mStride,
                        1u,
                        mLayer.mPadding,
                        0,
                        im2ColFor.getBuffer(),
                        mLayer.mTFStyle,
                        inputOffset);

            for (size_t group = 0; group < mLayer.mGroups; ++group)
            {
                Common::gemm(&work.getKernelManager(),
                             mLayer.mName / "forward" / to_string(q) / to_string(group),
                             CblasTrans,
                             CblasTrans,
                             mLayer.mOutputSize,
                             mLayer.mOutputChannels / mLayer.mGroups,
                             mLayer.mEffectiveReceptiveField * mLayer.mInputChannels / mLayer.mGroups,
                             1.0_dt,
                             im2ColFor.getBuffer(),
                             tempWeights.getBuffer(),
                             0_dt,
                             output.getBuffer(),
                             mTmpBuffer,
                             group * mLayer.mInputChannels / mLayer.mGroups * mLayer.mEffectiveReceptiveField * mLayer.mOutputSize,
                             group * mLayer.mOutputChannels / mLayer.mGroups * mLayer.mInputChannels / mLayer.mGroups * mLayer.mKernelSize,
                             q * mLayer.mOutputSize * mLayer.mOutputChannels + group * mLayer.mOutputChannels / mLayer.mGroups);
            }
        }
    }
    else
    {
        for (size_t q = 0; q < batchSize; ++q)
        {
            auto& im2ColFor = memory_manager(mLayer.mIm2ColForward[0]);

            size_t inputOffset = q * mLayer.mInputSize * mLayer.mInputChannels;

            gpu::im2col(work.getKernelManager(),
                        mLayer.mName / "forward" / to_string(q),
                        inputs.getBuffer(),
                        mLayer.mInputSize,
                        1u,
                        mLayer.mInputChannels,
                        mLayer.mEffectiveReceptiveField,
                        1u,
                        mLayer.mStride,
                        1u,
                        mLayer.mPadding,
                        0,
                        im2ColFor.getBuffer(),
                        mLayer.mTFStyle,
                        inputOffset);

            for (size_t group = 0; group < mLayer.mGroups; ++group)
            {
                Common::gemm(&work.getKernelManager(),
                             mLayer.mName / "forward" / to_string(q) / to_string(group),
                             CblasNoTrans,
                             CblasNoTrans,
                             mLayer.mOutputChannels / mLayer.mGroups,
                             mLayer.mOutputSize,
                             mLayer.mEffectiveReceptiveField * mLayer.mInputChannels / mLayer.mGroups,
                             1.0_dt,
                             weights.getBuffer(),
                             im2ColFor.getBuffer(),
                             0_dt,
                             output.getBuffer(),
                             mTmpBuffer,
                             group * mLayer.mOutputChannels / mLayer.mGroups * mLayer.mInputChannels * mLayer.mGroups * mLayer.mKernelSize,
                             group * mLayer.mInputChannels / mLayer.mGroups * mLayer.mEffectiveReceptiveField * mLayer.mOutputSize,
                             q * mLayer.mOutputSize * mLayer.mOutputChannels + group * mLayer.mOutputChannels / mLayer.mGroups * mLayer.mOutputSize);
            }
        }
    }

    if (mLayer.mUseBias)
    {
        const auto& biases = work.getMemoryManager<MemoryManagerGPU>()(mLayer.mBiasesName);

        if (mLayer.mTFStyle)
        {
            for (size_t q = 0; q < batchSize; ++q)
            {
                for (size_t i = 0; i < mLayer.mOutputSize; ++i)
                {
                    gpu::axpy(work.getKernelManager(),
                              mLayer.mName / "forward",
                              mLayer.mOutputChannels,
                              1_dt,
                              biases.getBuffer(),
                              0,
                              output.getBuffer(),
                              0,
                              0,
                              q * mLayer.mOutputSize * mLayer.mOutputChannels + i * mLayer.mOutputChannels);
                }
            }
        }
        else
        {
            cl::NDRange workSize(batchSize, mLayer.mOutputChannels, 1);
            work.getKernelManager().callKernel(
                getKernel("bias_forward"), workSize, mLayer.mName / "forward", (cl_int)batchSize, (cl_int)mLayer.mOutputChannels, (cl_int)mLayer.mOutputSize, biases.getBuffer(), output.getBuffer());
        }
    }
}

void Convolution1DLayerGPU::backwardComputeImpl()
{
    Workflow& work = mLayer.mNetworkParams.mWorkflow;
    auto& memory_manager = work.getMemoryManager<MemoryManagerGPU>();

    const size_t batchSize = work.getBatchSize();

    const auto& deltas = memory_manager(mLayer.mOutputName.grad());
    const Name weightsName = mLayer.mDilationEnabled ? mLayer.mDilationTensor : mLayer.mTFStyle ? mLayer.mTempWeights : mLayer.mWeightsName;
    const auto& weights = memory_manager(weightsName);

    auto weightsChannelStride = mLayer.mDilationEnabled ? mLayer.mInputChannels / mLayer.mGroups * mLayer.mEffectiveReceptiveField : mLayer.mInputChannels / mLayer.mGroups * mLayer.mKernelSize;

    // prevDelta
    // if (mNetworkParams.isGradNeeded(mLayer.mInputName))
    {
        auto& prevLayerDelta = memory_manager(mLayer.mInputName.grad());
        for (size_t q = 0; q < batchSize; ++q)
        {
            auto& im2ColBack = memory_manager(mLayer.mIm2ColBackward[0]);
            for (size_t group = 0; group < mLayer.mGroups; ++group)
            {
                if (mLayer.mTFStyle)
                {
                    Common::gemm(&work.getKernelManager(),
                                 mLayer.mName / "backward" / to_string(q) / to_string(group),
                                 CblasTrans,
                                 CblasTrans,
                                 mLayer.mEffectiveReceptiveField * mLayer.mInputChannels / mLayer.mGroups,
                                 mLayer.mOutputSize,
                                 mLayer.mOutputChannels / mLayer.mGroups,
                                 1.0_dt,
                                 weights.getBuffer(),
                                 deltas.getBuffer(),
                                 0_dt,
                                 im2ColBack.getBuffer(),
                                 mTmpBuffer,
                                 group * mLayer.mOutputChannels / mLayer.mGroups * weightsChannelStride,
                                 q * mLayer.mOutputSize * mLayer.mOutputChannels + group * mLayer.mOutputChannels / mLayer.mGroups,
                                 group * mLayer.mEffectiveReceptiveField * mLayer.mInputChannels * mLayer.mOutputSize / mLayer.mGroups);
                }
                else
                {
                    Common::gemm(&work.getKernelManager(),
                                 mLayer.mName / "backward" / to_string(q) / to_string(group),
                                 CblasTrans,
                                 CblasNoTrans,
                                 mLayer.mEffectiveReceptiveField * mLayer.mInputChannels / mLayer.mGroups,
                                 mLayer.mOutputSize,
                                 mLayer.mOutputChannels / mLayer.mGroups,
                                 1.0_dt,
                                 weights.getBuffer(),
                                 deltas.getBuffer(),
                                 0_dt,
                                 im2ColBack.getBuffer(),
                                 mTmpBuffer,
                                 group * mLayer.mOutputChannels / mLayer.mGroups * weightsChannelStride,
                                 q * mLayer.mOutputSize * mLayer.mOutputChannels + group * mLayer.mOutputChannels / mLayer.mGroups * mLayer.mOutputSize,
                                 group * mLayer.mEffectiveReceptiveField * mLayer.mInputChannels * mLayer.mOutputSize / mLayer.mGroups);
                }
            }

            gpu::col2im(work.getKernelManager(),
                        mLayer.mName / "backward" / to_string(q),
                        im2ColBack.getBuffer(),
                        mLayer.mInputSize,
                        1u,
                        mLayer.mInputChannels,
                        mLayer.mEffectiveReceptiveField,
                        1u,
                        mLayer.mStride,
                        1u,
                        mLayer.mPadding,
                        0,
                        prevLayerDelta.getBuffer(),
                        mLayer.mTFStyle,
                        false,
                        0,
                        q * mLayer.mInputChannels * mLayer.mInputSize);
        }
    }

    // gradients weights
    if (!mLayer.mFrozen)
    {
        const auto& inputs = memory_manager(mLayer.mInputName);

        auto& im2colMatrix = memory_manager(mLayer.mTmpIm2ColName);

        // auto& gradWeights = memory_manager(mLayer.mWeightsName.grad());

        Name tmpWeightsName = mLayer.mDilationEnabled ? mLayer.mDilationTensor : mLayer.mTFStyle ? Name(mLayer.mTempWeights.grad()) : Name(mLayer.mWeightsName.grad());
        auto& tmpWeightsGrad = memory_manager(tmpWeightsName);

        if (mLayer.mDilationEnabled || mLayer.mTFStyle)
        {
            work.getKernelManager().fillBuffer(tmpWeightsGrad.getBuffer(), 0_dt, mLayer.mName / "backward");
        }

        auto gradWeightsStride = mLayer.mDilationEnabled ? mLayer.mInputChannels / mLayer.mGroups * mLayer.mEffectiveReceptiveField : mLayer.mInputChannels / mLayer.mGroups * mLayer.mKernelSize;

        for (size_t q = 0; q < batchSize; ++q)
        {
            gpu::im2col(work.getKernelManager(),
                        mLayer.mName / "backward",
                        inputs.getBuffer(),
                        mLayer.mInputSize,
                        1u,
                        mLayer.mInputChannels,
                        mLayer.mEffectiveReceptiveField,
                        1u,
                        mLayer.mStride,
                        1u,
                        mLayer.mPadding,
                        0,
                        im2colMatrix.getBuffer(),
                        mLayer.mTFStyle,
                        q * mLayer.mInputSize * mLayer.mInputChannels,
                        q * mLayer.mOutputSize * mLayer.mInputChannels * mLayer.mEffectiveReceptiveField);
        }

        for (size_t q = 0; q < batchSize; ++q)
        {
            for (size_t group = 0; group < mLayer.mGroups; ++group)
            {
                if (mLayer.mTFStyle)
                {
                    Common::gemm(&work.getKernelManager(),
                                 mLayer.mName / "backward" / to_string(q) / to_string(group),
                                 CblasTrans,
                                 CblasTrans,
                                 mLayer.mOutputChannels / mLayer.mGroups,
                                 mLayer.mEffectiveReceptiveField * mLayer.mInputChannels / mLayer.mGroups,
                                 mLayer.mOutputSize,
                                 1.0_dt,
                                 deltas.getBuffer(),
                                 im2colMatrix.getBuffer(),
                                 1_dt,
                                 tmpWeightsGrad.getBuffer(),
                                 mTmpBuffer,
                                 q * mLayer.mOutputSize * mLayer.mOutputChannels + group * mLayer.mOutputChannels / mLayer.mGroups,
                                 q * mLayer.mOutputSize * mLayer.mInputChannels * mLayer.mEffectiveReceptiveField +
                                     group * mLayer.mOutputSize * mLayer.mInputChannels * mLayer.mEffectiveReceptiveField / mLayer.mGroups,
                                 group * mLayer.mOutputChannels / mLayer.mGroups * gradWeightsStride);
                }
                else
                {
                    Common::gemm(&work.getKernelManager(),
                                 mLayer.mName / "backward" / to_string(q) / to_string(group),
                                 CblasNoTrans,
                                 CblasTrans,
                                 mLayer.mOutputChannels / mLayer.mGroups,
                                 mLayer.mEffectiveReceptiveField * mLayer.mInputChannels / mLayer.mGroups,
                                 mLayer.mOutputSize,
                                 1.0_dt,
                                 deltas.getBuffer(),
                                 im2colMatrix.getBuffer(),
                                 1_dt,
                                 tmpWeightsGrad.getBuffer(),
                                 mTmpBuffer,
                                 q * mLayer.mOutputSize * mLayer.mOutputChannels + group * mLayer.mOutputChannels / mLayer.mGroups * mLayer.mOutputSize,
                                 q * mLayer.mOutputSize * mLayer.mInputChannels * mLayer.mEffectiveReceptiveField +
                                     group * mLayer.mOutputSize * mLayer.mInputChannels * mLayer.mEffectiveReceptiveField / mLayer.mGroups,
                                 group * mLayer.mOutputChannels / mLayer.mGroups * gradWeightsStride);
                }
            }
        }

        if (mLayer.mTFStyle || mLayer.mDilationEnabled)
        {
            const auto& weightsGrad = memory_manager(mLayer.mWeightsName.grad());
            cl::NDRange workSize(mLayer.mOutputChannels, mLayer.mInputChannels / mLayer.mGroups, 1);
            work.getKernelManager().callKernel(getKernel("weights_backward"),
                                               workSize,
                                               mLayer.mName / "backward",
                                               (cl_int)mLayer.mOutputChannels,
                                               (cl_int)(mLayer.mInputChannels / mLayer.mGroups),
                                               mLayer.mDilationEnabled ? (cl_int)mLayer.mEffectiveReceptiveField : (cl_int)mLayer.mKernelSize,
                                               (cl_int)mLayer.mKernelSize,
                                               (cl_int)mLayer.mDilation,
                                               tmpWeightsGrad.getBuffer(),
                                               weightsGrad.getBuffer());
        }

        if (mLayer.mUseBias)
        {
            cl::NDRange workSize(mLayer.mOutputChannels, 1, 1);
            const auto& biasesGrad = memory_manager(mLayer.mBiasesName.grad());

            work.getKernelManager().callKernel(getKernel("bias_backward"),
                                               workSize,
                                               mLayer.mName / "backward",
                                               (cl_int)batchSize,
                                               (cl_int)mLayer.mOutputChannels,
                                               (cl_int)mLayer.mOutputSize,
                                               deltas.getBuffer(),
                                               biasesGrad.getBuffer());
        }
    }
}

} // namespace raul
