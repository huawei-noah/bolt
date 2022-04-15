// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "LinearLayerCPUFP32Master.h"
#include "../LinearLayer.h"

namespace raul
{

LinearLayerCPUFP32Master::LinearLayerCPUFP32Master(LinearLayer& layer)
    : mLayer(layer)
    , mInitMasterWeights(false)
{
    mLayer.mNetworkParams.mWorkflow.tensorNeeded(mLayer.mName,
                                                 mLayer.mWeightsName + "_fp32",
                                                 WShape{ 1u, 1u, mLayer.mOutputsCount, mLayer.mInputsCount },
                                                 raul::Workflow::Usage::ForwardAndBackward,
                                                 raul::Workflow::Mode::Read,
                                                 false,
                                                 false,
                                                 !mLayer.mFrozen,
                                                 false,
                                                 false,
                                                 LayerExecutionTarget::CPU);

    if (mLayer.mUseBias)
    {
        mLayer.mNetworkParams.mWorkflow.tensorNeeded(mLayer.mName,
                                                     mLayer.mBiasesName + "_fp32",
                                                     WShape{ 1u, 1u, 1u, mLayer.mOutputsCount },
                                                     raul::Workflow::Usage::ForwardAndBackward,
                                                     raul::Workflow::Mode::Read,
                                                     false,
                                                     false,
                                                     !mLayer.mFrozen,
                                                     false,
                                                     false,
                                                     LayerExecutionTarget::CPU);
    }

    if (!mLayer.mFrozen)
    {
        mLayer.mNetworkParams.mWorkflow.copyDeclaration(mLayer.mName, mLayer.mWeightsName + "_fp32", Name(mLayer.mWeightsName + "_fp32").grad(), DEC_TRAINABLE_GRAD, LayerExecutionTarget::CPU);

        if (mLayer.mUseBias)
        {
            mLayer.mNetworkParams.mWorkflow.copyDeclaration(mLayer.mName, mLayer.mBiasesName + "_fp32", Name(mLayer.mBiasesName + "_fp32").grad(), DEC_TRAINABLE_GRAD, LayerExecutionTarget::CPU);
        }
    }
}

void LinearLayerCPUFP32Master::forwardComputeImpl(NetworkMode)
{
    Workflow& work = mLayer.mNetworkParams.mWorkflow;
    auto& weightsFP32 = work.getMemoryManager<MemoryManager>().getTensor(mLayer.mWeightsName + "_fp32");
    auto& weightsFP16 = work.getMemoryManager<MemoryManagerFP16>()[mLayer.mWeightsName];
    auto& biasesFP32 = work.getMemoryManager<MemoryManager>().getTensor(mLayer.mBiasesName + "_fp32");
    auto& biasesFP16 = work.getMemoryManager<MemoryManagerFP16>()[mLayer.mBiasesName];

    const auto& inputs = work.getMemoryManager<MemoryManagerFP16>()[mLayer.mInputName];
    auto& output = work.getMemoryManager<MemoryManagerFP16>()[mLayer.mOutputName];

    const size_t batchSize = mLayer.mNetworkParams.mWorkflow.getBatchSize();
    size_t N = batchSize * mLayer.mDepth * mLayer.mHeight;

    // initialize master weights from FP16
    if (!mInitMasterWeights)
    {
        std::transform(weightsFP16.begin(), weightsFP16.end(), weightsFP32.begin(), [](const half& val) { return toFloat32(val); });

        if (mLayer.mUseBias)
        {
            std::transform(biasesFP16.begin(), biasesFP16.end(), biasesFP32.begin(), [](const half& val) { return toFloat32(val); });
        }

        mInitMasterWeights = true;
    }
    else
    {
        std::transform(weightsFP32.begin(), weightsFP32.end(), weightsFP16.begin(), [](const dtype& val) { return toFloat16(val); });

        if (mLayer.mUseBias)
        {
            std::transform(biasesFP32.begin(), biasesFP32.end(), biasesFP16.begin(), [](const dtype& val) { return toFloat16(val); });
        }
    }

    Common::gemm(&mLayer.mNetworkParams.mWorkflow.getKernelManager(),
                 "",
                 CblasNoTrans,
                 CblasTrans,
                 N,
                 mLayer.mOutputsCount,
                 mLayer.mInputsCount,
                 1.0_dt,
                 inputs.getBuffer(),
                 weightsFP16.getBuffer(),
                 0.0_dt,
                 output.getBuffer());

    if (mLayer.mUseBias)
    {
        for (size_t index = 0; index < N; ++index)
        {
            Common::axpy(&mLayer.mNetworkParams.mWorkflow.getKernelManager(), "", mLayer.mOutputsCount, 1.0_dt, biasesFP16.getBuffer(), 1, output.getBuffer(), 1, 0, index * mLayer.mOutputsCount);
        }
    }
}

void LinearLayerCPUFP32Master::backwardComputeImpl()
{
    Workflow& work = mLayer.mNetworkParams.mWorkflow;

    const auto& deltas = work.getMemoryManager<MemoryManagerFP16>()[mLayer.mOutputName.grad()];

    const size_t batchSize = mLayer.mNetworkParams.mWorkflow.getBatchSize();
    size_t N = batchSize * mLayer.mDepth * mLayer.mHeight;

    const auto& weightsFP32 = work.getMemoryManager<MemoryManager>().getTensor(mLayer.mWeightsName + "_fp32");
    auto& weightsFP16 = work.getMemoryManager<MemoryManagerFP16>()[mLayer.mWeightsName];
    std::transform(weightsFP32.begin(), weightsFP32.end(), weightsFP16.begin(), [](const dtype& val) { return toFloat16(val); });

    ////if (mNetworkParams.isGradNeeded(mInputName))
    {
        auto& prevLayerDelta = work.getMemoryManager<MemoryManagerFP16>()[mLayer.mInputName.grad()];

        Common::gemm(&mLayer.mNetworkParams.mWorkflow.getKernelManager(),
                     "",
                     CblasNoTrans,
                     CblasNoTrans,
                     N,
                     mLayer.mInputsCount,
                     mLayer.mOutputsCount,
                     1.0_dt,
                     deltas.getBuffer(),
                     weightsFP16.getBuffer(),
                     1.0_dt,
                     prevLayerDelta.getBuffer());
    }

    if (!mLayer.mFrozen)
    {
        const auto& inputs = work.getMemoryManager<MemoryManagerFP16>()[mLayer.mInputName];

        auto& gradWeightsFP32 = work.getMemoryManager<MemoryManager>().getTensor(Name(mLayer.mWeightsName + "_fp32").grad());
        auto& gradWeightsFP16 = work.getMemoryManager<MemoryManagerFP16>()[mLayer.mWeightsName.grad()];

        Common::gemm(&mLayer.mNetworkParams.mWorkflow.getKernelManager(),
                     "",
                     CblasTrans,
                     CblasNoTrans,
                     mLayer.mOutputsCount,
                     mLayer.mInputsCount,
                     N,
                     1.0_dt,
                     deltas.getBuffer(),
                     inputs.getBuffer(),
                     1.0_dt,
                     gradWeightsFP16.getBuffer());

        std::transform(gradWeightsFP16.begin(), gradWeightsFP16.end(), gradWeightsFP32.begin(), [](const half& val) { return toFloat32(val); });

        if (mLayer.mUseBias)
        {
            auto& gradBiasesFP32 = work.getMemoryManager<MemoryManager>().getTensor(Name(mLayer.mBiasesName + "_fp32").grad());
            auto& gradBiasesFP16 = work.getMemoryManager<MemoryManagerFP16>()[mLayer.mBiasesName.grad()];

            for (size_t index = 0; index < N; ++index)
            {
                Common::axpy(&mLayer.mNetworkParams.mWorkflow.getKernelManager(), "", mLayer.mOutputsCount, 1.0_dt, deltas.getBuffer(), 1, gradBiasesFP16.getBuffer(), 1, index * mLayer.mOutputsCount);
            }

            std::transform(gradBiasesFP16.begin(), gradBiasesFP16.end(), gradBiasesFP32.begin(), [](const half& val) { return toFloat32(val); });
        }
    }
}

} // namespace raul
