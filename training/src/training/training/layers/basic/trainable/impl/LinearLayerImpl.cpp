// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "LinearLayerImpl.h"
#include "../LinearLayer.h"

#include <training/opencl/GemmGPU.h>

namespace raul
{

template<>
MemoryManager::tensor::type LinearLayerImpl<MemoryManager>::mTmpBuffer = 0;
template<>
MemoryManagerFP16::tensor::type LinearLayerImpl<MemoryManagerFP16>::mTmpBuffer = 0;
template<>
cl::Buffer LinearLayerImpl<MemoryManagerGPU>::mTmpBuffer = cl::Buffer();

template<typename MM>
void LinearLayerImpl<MM>::onBatchSizeChanged(size_t)
{
}

template<typename MM>
void LinearLayerImpl<MM>::initNotBSTensors()
{
}

template<>
void LinearLayerImpl<MemoryManagerGPU>::onBatchSizeChanged(size_t newBatchSize)
{
    auto caller = mLayer.mTypeName + "[" + mLayer.mName + "::onBatchSizeChanged]";
    size_t N = newBatchSize * mLayer.mDepth * mLayer.mHeight;
    size_t tmpSize1 = gpu::gemm_temp_buffer_size(CblasNoTrans, CblasTrans, N, mLayer.mOutputsCount, mLayer.mInputsCount);
    size_t tmpSize2 = gpu::gemm_temp_buffer_size(CblasNoTrans, CblasNoTrans, N, mLayer.mInputsCount, mLayer.mOutputsCount);
    size_t tmpSize3 = gpu::gemm_temp_buffer_size(CblasTrans, CblasNoTrans, mLayer.mOutputsCount, mLayer.mInputsCount, N);

    size_t tmpSize = std::max(tmpSize1, std::max(tmpSize2, tmpSize3));
    cl_int status = CL_SUCCESS;
    size_t currentSize = !mTmpBuffer() ? 0 : mTmpBuffer.getInfo<CL_MEM_SIZE>(&status);
    Common::checkOpenCLStatus(status, caller, "error quering buffer size");
    if (currentSize < tmpSize)
    {
        mTmpBuffer = mLayer.mNetworkParams.mWorkflow.getKernelManager().createBuffer(tmpSize, caller);
    }
}

template<>
void LinearLayerImpl<MemoryManagerGPU>::initNotBSTensors()
{
    auto& manager = mLayer.mNetworkParams.mWorkflow.getKernelManager();
    if (!manager.hasKernel("transpose_nchw"))
    {
        string source =
#include "kernels/transpose_nchw.cl"
            ;
        manager.registerProgram("transpose_nchw", source);
    }
    if (!manager.hasKernel("padding_nchw_constant"))
    {
        string source =
#include "kernels/padding_nchw.cl"
            ;
        manager.registerProgram("padding_nchw_constant", source, "-DUSE_CONSTANT");
    }
}

template<typename MM>
void LinearLayerImpl<MM>::forwardComputeImpl(NetworkMode)
{
    Workflow& work = mLayer.mNetworkParams.mWorkflow;

    auto& output = work.getMemoryManager<MM>().getTensor(mLayer.mOutputName);
    const size_t batchSize = mLayer.mNetworkParams.mWorkflow.getBatchSize();
    size_t N = batchSize * mLayer.mDepth * mLayer.mHeight;

    auto& inputs = work.getMemoryManager<MM>().getTensor(mLayer.mInputName);

    const auto& weights = work.getMemoryManager<MM>().getTensor(mLayer.mWeightsName);

    Common::gemm(&mLayer.mNetworkParams.mWorkflow.getKernelManager(),
                 mLayer.mName / "forward_weights",
                 CblasNoTrans,
                 CblasTrans,
                 N,
                 mLayer.mOutputsCount,
                 mLayer.mInputsCount,
                 1.0_dt,
                 inputs.getBuffer(),
                 weights.getBuffer(),
                 0.0_dt,
                 output.getBuffer(),
                 mTmpBuffer);

    if (mLayer.mUseBias)
    {
        const auto& biases = work.getMemoryManager<MM>().getTensor(mLayer.mBiasesName);

        for (size_t index = 0; index < N; ++index)
        {
            Common::axpy(&mLayer.mNetworkParams.mWorkflow.getKernelManager(),
                         mLayer.mName / "forward_biases",
                         mLayer.mOutputsCount,
                         1.0_dt,
                         biases.getBuffer(),
                         1,
                         output.getBuffer(),
                         1,
                         0,
                         index * mLayer.mOutputsCount);
        }
    }
}

template<>
void LinearLayerImpl<MemoryManagerGPU>::forwardComputeImpl(NetworkMode mode)
{
    Workflow& work = mLayer.mNetworkParams.mWorkflow;
    auto& memory_manager = work.getMemoryManager<MemoryManagerGPU>();
    auto& output = memory_manager(mLayer.mOutputName);
    auto& kernelManager = mLayer.mNetworkParams.mWorkflow.getKernelManager();

    const size_t batchSize = mLayer.mNetworkParams.mWorkflow.getBatchSize();
    size_t N = batchSize * mLayer.mDepth * mLayer.mHeight;

    auto& inputs = memory_manager(mLayer.mInputName);

    const auto& weights = memory_manager(mLayer.mWeightsName);

    if (mLayer.mSharedLayer.empty() && mLayer.mSharedWeights.empty())
    {

        {
            auto kernelTranspose = kernelManager.getKernel("transpose_nchw", mLayer.mName / "transpose_weights");
            size_t n = mLayer.mOutputsCount;
            size_t k = mLayer.mInputsCount;
            // prepare weights
            cl::NDRange workSize{ (k + 3) / 4, n, 1 };
            cl_int dimTran[3] = { 1, 0, 2 };
            auto& transposed = memory_manager(mLayer.mTransposedWeightsName).getBuffer();
            kernelManager.fillBuffer(transposed, 0_dt, mLayer.mName / "transpose_weights");
            kernelManager.callKernel(kernelTranspose,
                                     workSize,
                                     mLayer.mName / "transpose_weights",
                                     (cl_int)k,
                                     (cl_int)n,
                                     0,
                                     0,
                                     (cl_int)n,
                                     (cl_int)k,
                                     0,
                                     0,
                                     dimTran[0],
                                     dimTran[1],
                                     dimTran[2],
                                     (cl_int)k,
                                     (cl_int)workSize[0],
                                     (cl_int)workSize[1],
                                     weights.getBuffer(),
                                     transposed);

            if (mLayer.mForwardWeightsUsage == LinearLayer::WeightsUsage::Padded)
            {
                auto kernelPad = kernelManager.getKernel("padding_nchw_constant", mLayer.mName / "pad_transposed_weights");
                workSize = cl::NDRange{ (mLayer.mForwardAlignedWeightsSize + 3) / 4, k, 1 };
                auto& transposedPadded = memory_manager(mLayer.mTransposedPaddedWeightsName).getBuffer();
                kernelManager.fillBuffer(transposedPadded, 0_dt, mLayer.mName / "pad_transposed_weights");
                kernelManager.callKernel(kernelPad,
                                         workSize,
                                         mLayer.mName / "pad_transposed_weights",
                                         (cl_int)n,
                                         (cl_int)k,
                                         0,
                                         0,
                                         (cl_int)mLayer.mForwardAlignedWeightsSize,
                                         (cl_int)k,
                                         0,
                                         0,
                                         (cl_int)n,
                                         (cl_int)k,
                                         (cl_int)mLayer.mForwardAlignedWeightsSize,
                                         (cl_int)k,
                                         0,
                                         0,
                                         0,
                                         0,
                                         0,
                                         0,
                                         (cl_int)workSize[0],
                                         (cl_int)workSize[1],
                                         transposed,
                                         transposedPadded);
            }
        }

        if (mode == NetworkMode::Train || mode == NetworkMode::TrainCheckpointed)
        {
            if (mLayer.mBackwardWeightsUsage == LinearLayer::WeightsUsage::Padded)
            {
                auto kernelPad = kernelManager.getKernel("padding_nchw_constant", mLayer.mName / "pad_weights");
                auto& padded = memory_manager(mLayer.mPaddedWeightsName).getBuffer();
                size_t n = mLayer.mInputsCount;
                size_t k = mLayer.mOutputsCount;
                cl::NDRange workSize{ (mLayer.mBackwardAlignedWeightsSize + 3) / 4, k, 1 };
                kernelManager.fillBuffer(padded, 0_dt, mLayer.mName / "pad_weights");
                kernelManager.callKernel(kernelPad,
                                         workSize,
                                         mLayer.mName / "pad_weights",
                                         (cl_int)n,
                                         (cl_int)k,
                                         0,
                                         0,
                                         (cl_int)mLayer.mBackwardAlignedWeightsSize,
                                         (cl_int)k,
                                         0,
                                         0,
                                         (cl_int)n,
                                         (cl_int)k,
                                         (cl_int)mLayer.mBackwardAlignedWeightsSize,
                                         (cl_int)k,
                                         0,
                                         0,
                                         0,
                                         0,
                                         0,
                                         0,
                                         (cl_int)workSize[0],
                                         (cl_int)workSize[1],
                                         weights.getBuffer(),
                                         padded);
            }
        }
    }

    gpu::gemm_padded_b(kernelManager,
                       mLayer.mName / "forward_weights",
                       CblasNoTrans,
                       N,
                       mLayer.mOutputsCount,
                       mLayer.mInputsCount,
                       1.0_dt,
                       inputs.getBuffer(),
                       memory_manager(mLayer.mForwardWeightsNameGpu).getBuffer(),
                       0.0_dt,
                       output.getBuffer(),
                       mTmpBuffer);

    if (mLayer.mUseBias)
    {
        const auto& biases = memory_manager(mLayer.mBiasesName);

        for (size_t index = 0; index < N; ++index)
        {
            Common::axpy(&kernelManager, mLayer.mName / "forward_biases", mLayer.mOutputsCount, 1.0_dt, biases.getBuffer(), 1, output.getBuffer(), 1, 0, index * mLayer.mOutputsCount);
        }
    }
}

template<typename MM>
void LinearLayerImpl<MM>::backwardComputeImpl()
{
    Workflow& work = mLayer.mNetworkParams.mWorkflow;

    const auto& deltas = work.getMemoryManager<MM>().getTensor(mLayer.mOutputName.grad());

    const size_t batchSize = mLayer.mNetworkParams.mWorkflow.getBatchSize();
    size_t N = batchSize * mLayer.mDepth * mLayer.mHeight;

    const auto& weights = work.getMemoryManager<MM>().getTensor(mLayer.mWeightsName);

    ////if (mNetworkParams.isGradNeeded(mInputName))
    {
        auto& prevLayerDelta = work.getMemoryManager<MM>().getTensor(mLayer.mInputName.grad());

        Common::gemm(&mLayer.mNetworkParams.mWorkflow.getKernelManager(),
                     mLayer.mName / "backward_deltas",
                     CblasNoTrans,
                     CblasNoTrans,
                     N,
                     mLayer.mInputsCount,
                     mLayer.mOutputsCount,
                     1.0_dt,
                     deltas.getBuffer(),
                     weights.getBuffer(),
                     1.0_dt,
                     prevLayerDelta.getBuffer(),
                     mTmpBuffer);
    }

    if (!mLayer.mFrozen)
    {
        const auto& inputs = work.getMemoryManager<MM>().getTensor(mLayer.mInputName);
        auto& gradWeights = work.getMemoryManager<MM>().getTensor(mLayer.mWeightsName.grad());

        Common::gemm(&mLayer.mNetworkParams.mWorkflow.getKernelManager(),
                     mLayer.mName / "backward_weights",
                     CblasTrans,
                     CblasNoTrans,
                     mLayer.mOutputsCount,
                     mLayer.mInputsCount,
                     N,
                     1.0_dt,
                     deltas.getBuffer(),
                     inputs.getBuffer(),
                     1.0_dt,
                     gradWeights.getBuffer(),
                     mTmpBuffer);

        if (mLayer.mUseBias)
        {
            auto& gradBiases = work.getMemoryManager<MM>().getTensor(mLayer.mBiasesName.grad());
            for (size_t index = 0; index < N; ++index)
            {
                Common::axpy(&mLayer.mNetworkParams.mWorkflow.getKernelManager(),
                             mLayer.mName / "backward_biases",
                             mLayer.mOutputsCount,
                             1.0_dt,
                             deltas.getBuffer(),
                             1,
                             gradBiases.getBuffer(),
                             1,
                             index * mLayer.mOutputsCount,
                             0);
            }
        }
    }
}

template<>
void LinearLayerImpl<MemoryManagerGPU>::backwardComputeImpl()
{
    Workflow& work = mLayer.mNetworkParams.mWorkflow;
    auto& memory_manager = work.getMemoryManager<MemoryManagerGPU>();
    const auto& deltas = memory_manager.getTensor(mLayer.mOutputName.grad());

    const size_t batchSize = mLayer.mNetworkParams.mWorkflow.getBatchSize();
    size_t N = batchSize * mLayer.mDepth * mLayer.mHeight;

    ////if (mNetworkParams.isGradNeeded(mInputName))
    {
        auto& prevLayerDelta = memory_manager(mLayer.mInputName.grad());

        gpu::gemm_padded_b(mLayer.mNetworkParams.mWorkflow.getKernelManager(),
                           mLayer.mName / "backward_deltas",
                           CblasNoTrans,
                           N,
                           mLayer.mInputsCount,
                           mLayer.mOutputsCount,
                           1.0_dt,
                           deltas.getBuffer(),
                           memory_manager(mLayer.mBackwardWeightsNameGpu).getBuffer(),
                           1.0_dt,
                           prevLayerDelta.getBuffer(),
                           mTmpBuffer);
    }

    if (!mLayer.mFrozen)
    {
        const auto& inputs = memory_manager(mLayer.mInputName);
        auto& gradWeights = memory_manager(mLayer.mWeightsName.grad());

        Common::gemm(&mLayer.mNetworkParams.mWorkflow.getKernelManager(),
                     mLayer.mName / "backward_weights",
                     CblasTrans,
                     CblasNoTrans,
                     mLayer.mOutputsCount,
                     mLayer.mInputsCount,
                     N,
                     1.0_dt,
                     deltas.getBuffer(),
                     inputs.getBuffer(),
                     1.0_dt,
                     gradWeights.getBuffer(),
                     mTmpBuffer);

        if (mLayer.mUseBias)
        {
            auto& gradBiases = memory_manager(mLayer.mBiasesName.grad());
            for (size_t index = 0; index < N; ++index)
            {
                Common::axpy(&mLayer.mNetworkParams.mWorkflow.getKernelManager(),
                             mLayer.mName / "backward_biases",
                             mLayer.mOutputsCount,
                             1.0_dt,
                             deltas.getBuffer(),
                             1,
                             gradBiases.getBuffer(),
                             1,
                             index * mLayer.mOutputsCount,
                             0);
            }
        }
    }
}

INSTANTIATE_IMPL(LinearLayerImpl)

} // namespace raul
