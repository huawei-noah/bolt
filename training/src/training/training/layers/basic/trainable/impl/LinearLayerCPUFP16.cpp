// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "LinearLayerCPUFP16.h"
#include "../LinearLayer.h"

namespace raul
{

LinearLayerCPUFP16::LinearLayerCPUFP16(LinearLayer& layer)
    : mLayer(layer)
{
    auto inputShape = mLayer.mNetworkParams.mWorkflow.getShape(mLayer.mInputName);

    mLayer.mNetworkParams.mWorkflow.tensorNeededMaxShape(
        mLayer.mName, "LinearLayerBufferInputFP16", inputShape, Workflow::Usage::ForwardAndBackward, Workflow::Mode::Write, true, true, false, false, false, LayerExecutionTarget::CPUFP16);

    mLayer.mNetworkParams.mWorkflow.tensorNeededMaxShape(
        mLayer.mName, "LinearLayerBufferInputGradFP16", inputShape, Workflow::Usage::Backward, Workflow::Mode::Write, true, true, false, false, false, LayerExecutionTarget::CPUFP16);

    mLayer.mNetworkParams.mWorkflow.tensorNeededMaxShape(mLayer.mName,
                                                         "LinearLayerBufferOutputFP16",
                                                         WShape{ BS(), mLayer.mDepth, mLayer.mHeight, mLayer.mOutputsCount },
                                                         raul::Workflow::Usage::Forward,
                                                         raul::Workflow::Mode::Write,
                                                         true,
                                                         true,
                                                         false,
                                                         false,
                                                         false,
                                                         LayerExecutionTarget::CPUFP16);

    mLayer.mNetworkParams.mWorkflow.tensorNeededMaxShape(mLayer.mName,
                                                         "LinearLayerBufferOutputGradFP16",
                                                         WShape{ BS(), mLayer.mDepth, mLayer.mHeight, mLayer.mOutputsCount },
                                                         Workflow::Usage::Backward,
                                                         Workflow::Mode::Write,
                                                         true,
                                                         true,
                                                         false,
                                                         false,
                                                         false,
                                                         LayerExecutionTarget::CPUFP16);

    mLayer.mNetworkParams.mWorkflow.tensorNeededMaxShape(mLayer.mName,
                                                         "LinearLayerBufferWeightsFP16",
                                                         WShape{ 1u, 1u, mLayer.mOutputsCount, mLayer.mInputsCount },
                                                         Workflow::Usage::ForwardAndBackward,
                                                         Workflow::Mode::Write,
                                                         true,
                                                         true,
                                                         false,
                                                         false,
                                                         false,
                                                         LayerExecutionTarget::CPUFP16);

    if (mLayer.mUseBias)
    {
        mLayer.mNetworkParams.mWorkflow.tensorNeededMaxShape(mLayer.mName,
                                                             "LinearLayerBufferBiasFP16",
                                                             WShape{ 1u, 1u, 1u, mLayer.mOutputsCount },
                                                             Workflow::Usage::Forward,
                                                             Workflow::Mode::Write,
                                                             true,
                                                             true,
                                                             false,
                                                             false,
                                                             false,
                                                             LayerExecutionTarget::CPUFP16);
    }

    if (!mLayer.mFrozen)
    {
        mLayer.mNetworkParams.mWorkflow.tensorNeededMaxShape(mLayer.mName,
                                                             "LinearLayerBufferWeightsGradFP16",
                                                             WShape{ 1u, 1u, mLayer.mOutputsCount, mLayer.mInputsCount },
                                                             Workflow::Usage::Backward,
                                                             Workflow::Mode::Write,
                                                             true,
                                                             true,
                                                             false,
                                                             false,
                                                             false,
                                                             LayerExecutionTarget::CPUFP16);

        if (mLayer.mUseBias)
        {
            mLayer.mNetworkParams.mWorkflow.tensorNeededMaxShape(mLayer.mName,
                                                                 "LinearLayerBufferBiasGradFP16",
                                                                 WShape{ 1u, 1u, 1u, mLayer.mOutputsCount },
                                                                 Workflow::Usage::Backward,
                                                                 Workflow::Mode::Write,
                                                                 true,
                                                                 true,
                                                                 false,
                                                                 false,
                                                                 false,
                                                                 LayerExecutionTarget::CPUFP16);
        }
    }
}

void LinearLayerCPUFP16::forwardComputeImpl(NetworkMode)
{
    auto& output = mLayer.mNetworkParams.mMemoryManager[mLayer.mOutputName];
    auto& outputFP16 = mLayer.mNetworkParams.mWorkflow.getMemoryManager<MemoryManagerFP16>()["LinearLayerBufferOutputFP16"];

    std::transform(output.begin(), output.end(), outputFP16.begin(), [](dtype val) { return toFloat16(val); });

    const size_t batchSize = mLayer.mNetworkParams.mWorkflow.getBatchSize();
    size_t N = batchSize * mLayer.mDepth * mLayer.mHeight;

    const auto& inputs = mLayer.mNetworkParams.mMemoryManager[mLayer.mInputName];
    auto& inputsFP16 = mLayer.mNetworkParams.mWorkflow.getMemoryManager<MemoryManagerFP16>()["LinearLayerBufferInputFP16"];
    std::transform(inputs.begin(), inputs.end(), inputsFP16.begin(), [](dtype val) { return toFloat16(val); });

    const auto& weights = mLayer.mNetworkParams.mWorkflow.getMemoryManager()[mLayer.mWeightsName];
    auto& weightsFP16 = mLayer.mNetworkParams.mWorkflow.getMemoryManager<MemoryManagerFP16>()["LinearLayerBufferWeightsFP16"];
    std::transform(weights.begin(), weights.end(), weightsFP16.begin(), [](dtype val) { return toFloat16(val); });

    Common::gemm(&mLayer.mNetworkParams.mWorkflow.getKernelManager(),
                 "",
                 CblasNoTrans,
                 CblasTrans,
                 N,
                 mLayer.mOutputsCount,
                 mLayer.mInputsCount,
                 1.0_dt,
                 &inputsFP16[0],
                 &weightsFP16[0],
                 0.0_dt,
                 &outputFP16[0]);

    if (mLayer.mUseBias)
    {
        const auto& biases = mLayer.mNetworkParams.mWorkflow.getMemoryManager()[mLayer.mBiasesName];
        auto& biasesFP16 = mLayer.mNetworkParams.mWorkflow.getMemoryManager<MemoryManagerFP16>()["LinearLayerBufferBiasFP16"];
        std::transform(biases.begin(), biases.end(), biasesFP16.begin(), [](dtype val) { return toFloat16(val); });

        for (size_t index = 0; index < N; ++index)
        {
            Common::axpy(&mLayer.mNetworkParams.mWorkflow.getKernelManager(), "", mLayer.mOutputsCount, 1.0_dt, &biasesFP16[0], 1, &outputFP16[0], 1, 0, index * mLayer.mOutputsCount);
        }
    }

    std::transform(outputFP16.begin(), outputFP16.begin() + output.size(), output.begin(), [](half val) { return toFloat32(val); });
}

void LinearLayerCPUFP16::backwardComputeImpl()
{
    const auto& deltas = mLayer.mNetworkParams.mMemoryManager[mLayer.mOutputName.grad()];
    auto& deltasFP16 = mLayer.mNetworkParams.mWorkflow.getMemoryManager<MemoryManagerFP16>()["LinearLayerBufferOutputGradFP16"];
    std::transform(deltas.begin(), deltas.end(), deltasFP16.begin(), [](dtype val) { return toFloat16(val); });

    const size_t batchSize = mLayer.mNetworkParams.mWorkflow.getBatchSize();
    size_t N = batchSize * mLayer.mDepth * mLayer.mHeight;

    const auto& weights = mLayer.mNetworkParams.mWorkflow.getMemoryManager()[mLayer.mWeightsName];
    auto& weightsFP16 = mLayer.mNetworkParams.mWorkflow.getMemoryManager<MemoryManagerFP16>()["LinearLayerBufferWeightsFP16"];
    std::transform(weights.begin(), weights.end(), weightsFP16.begin(), [](dtype val) { return toFloat16(val); });

    ////if (mNetworkParams.isGradNeeded(mInputName))
    {
        auto& prevLayerDelta = mLayer.mNetworkParams.mMemoryManager[mLayer.mInputName.grad()];
        auto& prevLayerDeltaFP16 = mLayer.mNetworkParams.mWorkflow.getMemoryManager<MemoryManagerFP16>()["LinearLayerBufferInputGradFP16"];
        std::transform(prevLayerDelta.begin(), prevLayerDelta.end(), prevLayerDeltaFP16.begin(), [](dtype val) { return toFloat16(val); });

        Common::gemm(&mLayer.mNetworkParams.mWorkflow.getKernelManager(),
                     "",
                     CblasNoTrans,
                     CblasNoTrans,
                     N,
                     mLayer.mInputsCount,
                     mLayer.mOutputsCount,
                     1.0_dt,
                     &deltasFP16[0],
                     &weightsFP16[0],
                     1.0_dt,
                     &prevLayerDeltaFP16[0]);

        std::transform(prevLayerDeltaFP16.begin(), prevLayerDeltaFP16.begin() + prevLayerDelta.size(), prevLayerDelta.begin(), [](half val) { return toFloat32(val); });
    }

    if (!mLayer.mFrozen)
    {
        const Tensor& inputs = mLayer.mNetworkParams.mMemoryManager[mLayer.mInputName];
        auto& inputsFP16 = mLayer.mNetworkParams.mWorkflow.getMemoryManager<MemoryManagerFP16>()["LinearLayerBufferInputFP16"];
        std::transform(inputs.begin(), inputs.end(), inputsFP16.begin(), [](dtype val) { return toFloat16(val); });

        auto& gradWeights = mLayer.mNetworkParams.mWorkflow.getMemoryManager()[mLayer.mWeightsName.grad()];
        auto& gradWeightsFP16 = mLayer.mNetworkParams.mWorkflow.getMemoryManager<MemoryManagerFP16>()["LinearLayerBufferWeightsGradFP16"];
        std::transform(gradWeights.begin(), gradWeights.end(), gradWeightsFP16.begin(), [](dtype val) { return toFloat16(val); });

        Common::gemm(&mLayer.mNetworkParams.mWorkflow.getKernelManager(),
                     "",
                     CblasTrans,
                     CblasNoTrans,
                     mLayer.mOutputsCount,
                     mLayer.mInputsCount,
                     N,
                     1.0_dt,
                     &deltasFP16[0],
                     &inputsFP16[0],
                     1.0_dt,
                     &gradWeightsFP16[0]);

        std::transform(gradWeightsFP16.begin(), gradWeightsFP16.begin() + gradWeights.size(), gradWeights.begin(), [](half val) { return toFloat32(val); });

        if (mLayer.mUseBias)
        {
            auto& gradBiases = mLayer.mNetworkParams.mWorkflow.getMemoryManager()[mLayer.mBiasesName.grad()];
            auto& gradBiasesFP16 = mLayer.mNetworkParams.mWorkflow.getMemoryManager<MemoryManagerFP16>()["LinearLayerBufferBiasGradFP16"];

            std::transform(gradBiases.begin(), gradBiases.end(), gradBiasesFP16.begin(), [](dtype val) { return toFloat16(val); });

            for (size_t index = 0; index < N; ++index)
            {
                Common::axpy(&mLayer.mNetworkParams.mWorkflow.getKernelManager(), "", mLayer.mOutputsCount, 1.0_dt, &deltasFP16[0], 1, &gradBiasesFP16[0], 1, index * mLayer.mOutputsCount);
            }

            std::transform(gradBiasesFP16.begin(), gradBiasesFP16.begin() + gradBiases.size(), gradBiases.begin(), [](half val) { return toFloat32(val); });
        }
    }
}

} // namespace raul
