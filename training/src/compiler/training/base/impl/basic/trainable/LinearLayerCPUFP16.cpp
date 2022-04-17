// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "LinearLayerCPUFP16.h"
#include <training/base/layers/basic/trainable/LinearLayer.h>
#include <training/base/impl/ImplFactory.h>

namespace
{
bool reg = raul::TheImplFactory::Instance().regCPUFP32FP16MixedLocal<raul::LinearLayer, raul::LinearLayerCPUFP16>();
} // anonymous namespace

namespace raul
{

LinearLayerCPUFP16::LinearLayerCPUFP16(LinearLayer& layer)
    : mLayer(layer)
{
    auto inputShape = mLayer.getNetworkParams().mWorkflow.getShape(mLayer.getInputName());

    mLayer.getNetworkParams().mWorkflow.tensorNeededMaxShape(
        mLayer.getName(), "LinearLayerBufferInputFP16", inputShape, Workflow::Usage::ForwardAndBackward, Workflow::Mode::Write, true, true, false, false, false, LayerExecutionTarget::CPUFP16);

    mLayer.getNetworkParams().mWorkflow.tensorNeededMaxShape(
        mLayer.getName(), "LinearLayerBufferInputGradFP16", inputShape, Workflow::Usage::Backward, Workflow::Mode::Write, true, true, false, false, false, LayerExecutionTarget::CPUFP16);

    mLayer.getNetworkParams().mWorkflow.tensorNeededMaxShape(mLayer.getName(),
                                                             "LinearLayerBufferOutputFP16",
                                                             WShape{ BS(), mLayer.getDepth(), mLayer.getHeight(), mLayer.getOutputsCount() },
                                                             raul::Workflow::Usage::Forward,
                                                             raul::Workflow::Mode::Write,
                                                             true,
                                                             true,
                                                             false,
                                                             false,
                                                             false,
                                                             LayerExecutionTarget::CPUFP16);

    mLayer.getNetworkParams().mWorkflow.tensorNeededMaxShape(mLayer.getName(),
                                                             "LinearLayerBufferOutputGradFP16",
                                                             WShape{ BS(), mLayer.getDepth(), mLayer.getHeight(), mLayer.getOutputsCount() },
                                                             Workflow::Usage::Backward,
                                                             Workflow::Mode::Write,
                                                             true,
                                                             true,
                                                             false,
                                                             false,
                                                             false,
                                                             LayerExecutionTarget::CPUFP16);

    mLayer.getNetworkParams().mWorkflow.tensorNeededMaxShape(mLayer.getName(),
                                                             "LinearLayerBufferWeightsFP16",
                                                             WShape{ 1u, 1u, mLayer.getOutputsCount(), mLayer.getInputsCount() },
                                                             Workflow::Usage::ForwardAndBackward,
                                                             Workflow::Mode::Write,
                                                             true,
                                                             true,
                                                             false,
                                                             false,
                                                             false,
                                                             LayerExecutionTarget::CPUFP16);

    if (mLayer.isUseBias())
    {
        mLayer.getNetworkParams().mWorkflow.tensorNeededMaxShape(mLayer.getName(),
                                                                 "LinearLayerBufferBiasFP16",
                                                                 WShape{ 1u, 1u, 1u, mLayer.getOutputsCount() },
                                                                 Workflow::Usage::Forward,
                                                                 Workflow::Mode::Write,
                                                                 true,
                                                                 true,
                                                                 false,
                                                                 false,
                                                                 false,
                                                                 LayerExecutionTarget::CPUFP16);
    }

    if (!mLayer.isFrozen())
    {
        mLayer.getNetworkParams().mWorkflow.tensorNeededMaxShape(mLayer.getName(),
                                                                 "LinearLayerBufferWeightsGradFP16",
                                                                 WShape{ 1u, 1u, mLayer.getOutputsCount(), mLayer.getInputsCount() },
                                                                 Workflow::Usage::Backward,
                                                                 Workflow::Mode::Write,
                                                                 true,
                                                                 true,
                                                                 false,
                                                                 false,
                                                                 false,
                                                                 LayerExecutionTarget::CPUFP16);

        if (mLayer.isUseBias())
        {
            mLayer.getNetworkParams().mWorkflow.tensorNeededMaxShape(mLayer.getName(),
                                                                     "LinearLayerBufferBiasGradFP16",
                                                                     WShape{ 1u, 1u, 1u, mLayer.getOutputsCount() },
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
    auto& output = mLayer.getNetworkParams().mMemoryManager[mLayer.getOutputName()];
    auto& outputFP16 = mLayer.getNetworkParams().mWorkflow.getMemoryManager<MemoryManagerFP16>()["LinearLayerBufferOutputFP16"];

    std::transform(output.begin(), output.end(), outputFP16.begin(), [](dtype val) { return toFloat16(val); });

    const size_t batchSize = mLayer.getNetworkParams().mWorkflow.getBatchSize();
    size_t N = batchSize * mLayer.getDepth() * mLayer.getHeight();

    const auto& inputs = mLayer.getNetworkParams().mMemoryManager[mLayer.getInputName()];
    auto& inputsFP16 = mLayer.getNetworkParams().mWorkflow.getMemoryManager<MemoryManagerFP16>()["LinearLayerBufferInputFP16"];
    std::transform(inputs.begin(), inputs.end(), inputsFP16.begin(), [](dtype val) { return toFloat16(val); });

    const auto& weights = mLayer.getNetworkParams().mWorkflow.getMemoryManager()[mLayer.getWeightsName()];
    auto& weightsFP16 = mLayer.getNetworkParams().mWorkflow.getMemoryManager<MemoryManagerFP16>()["LinearLayerBufferWeightsFP16"];
    std::transform(weights.begin(), weights.end(), weightsFP16.begin(), [](dtype val) { return toFloat16(val); });

    Common::gemm(CblasNoTrans,
                 CblasTrans,
                 N,
                 mLayer.getOutputsCount(),
                 mLayer.getInputsCount(),
                 1.0_dt,
                 &inputsFP16[0],
                 &weightsFP16[0],
                 0.0_dt,
                 &outputFP16[0]);

    if (mLayer.isUseBias())
    {
        const auto& biases = mLayer.getNetworkParams().mWorkflow.getMemoryManager()[mLayer.getBiasesName()];
        auto& biasesFP16 = mLayer.getNetworkParams().mWorkflow.getMemoryManager<MemoryManagerFP16>()["LinearLayerBufferBiasFP16"];
        std::transform(biases.begin(), biases.end(), biasesFP16.begin(), [](dtype val) { return toFloat16(val); });

        for (size_t index = 0; index < N; ++index)
        {
            Common::axpy(mLayer.getOutputsCount(), 1.0_dt, &biasesFP16[0], 1, &outputFP16[0], 1, 0, index * mLayer.getOutputsCount());
        }
    }

    std::transform(outputFP16.begin(), outputFP16.begin() + output.size(), output.begin(), [](half val) { return toFloat32(val); });
}

void LinearLayerCPUFP16::backwardComputeImpl()
{
    const auto& deltas = mLayer.getNetworkParams().mMemoryManager[mLayer.getOutputName().grad()];
    auto& deltasFP16 = mLayer.getNetworkParams().mWorkflow.getMemoryManager<MemoryManagerFP16>()["LinearLayerBufferOutputGradFP16"];
    std::transform(deltas.begin(), deltas.end(), deltasFP16.begin(), [](dtype val) { return toFloat16(val); });

    const size_t batchSize = mLayer.getNetworkParams().mWorkflow.getBatchSize();
    size_t N = batchSize * mLayer.getDepth() * mLayer.getHeight();

    const auto& weights = mLayer.getNetworkParams().mWorkflow.getMemoryManager()[mLayer.getWeightsName()];
    auto& weightsFP16 = mLayer.getNetworkParams().mWorkflow.getMemoryManager<MemoryManagerFP16>()["LinearLayerBufferWeightsFP16"];
    std::transform(weights.begin(), weights.end(), weightsFP16.begin(), [](dtype val) { return toFloat16(val); });

    ////if (mNetworkParams.isGradNeeded(mInputName))
    {
        auto& prevLayerDelta = mLayer.getNetworkParams().mMemoryManager[mLayer.getInputName().grad()];
        auto& prevLayerDeltaFP16 = mLayer.getNetworkParams().mWorkflow.getMemoryManager<MemoryManagerFP16>()["LinearLayerBufferInputGradFP16"];
        std::transform(prevLayerDelta.begin(), prevLayerDelta.end(), prevLayerDeltaFP16.begin(), [](dtype val) { return toFloat16(val); });

        Common::gemm(CblasNoTrans,
                     CblasNoTrans,
                     N,
                     mLayer.getInputsCount(),
                     mLayer.getOutputsCount(),
                     1.0_dt,
                     &deltasFP16[0],
                     &weightsFP16[0],
                     1.0_dt,
                     &prevLayerDeltaFP16[0]);

        std::transform(prevLayerDeltaFP16.begin(), prevLayerDeltaFP16.begin() + prevLayerDelta.size(), prevLayerDelta.begin(), [](half val) { return toFloat32(val); });
    }

    if (!mLayer.isFrozen())
    {
        const Tensor& inputs = mLayer.getNetworkParams().mMemoryManager[mLayer.getInputName()];
        auto& inputsFP16 = mLayer.getNetworkParams().mWorkflow.getMemoryManager<MemoryManagerFP16>()["LinearLayerBufferInputFP16"];
        std::transform(inputs.begin(), inputs.end(), inputsFP16.begin(), [](dtype val) { return toFloat16(val); });

        auto& gradWeights = mLayer.getNetworkParams().mWorkflow.getMemoryManager()[mLayer.getWeightsName().grad()];
        auto& gradWeightsFP16 = mLayer.getNetworkParams().mWorkflow.getMemoryManager<MemoryManagerFP16>()["LinearLayerBufferWeightsGradFP16"];
        std::transform(gradWeights.begin(), gradWeights.end(), gradWeightsFP16.begin(), [](dtype val) { return toFloat16(val); });

        Common::gemm(CblasTrans,
                     CblasNoTrans,
                     mLayer.getOutputsCount(),
                     mLayer.getInputsCount(),
                     N,
                     1.0_dt,
                     &deltasFP16[0],
                     &inputsFP16[0],
                     1.0_dt,
                     &gradWeightsFP16[0]);

        std::transform(gradWeightsFP16.begin(), gradWeightsFP16.begin() + gradWeights.size(), gradWeights.begin(), [](half val) { return toFloat32(val); });

        if (mLayer.isUseBias())
        {
            auto& gradBiases = mLayer.getNetworkParams().mWorkflow.getMemoryManager()[mLayer.getBiasesName().grad()];
            auto& gradBiasesFP16 = mLayer.getNetworkParams().mWorkflow.getMemoryManager<MemoryManagerFP16>()["LinearLayerBufferBiasGradFP16"];

            std::transform(gradBiases.begin(), gradBiases.end(), gradBiasesFP16.begin(), [](dtype val) { return toFloat16(val); });

            for (size_t index = 0; index < N; ++index)
            {
                Common::axpy(mLayer.getOutputsCount(), 1.0_dt, &deltasFP16[0], 1, &gradBiasesFP16[0], 1, index * mLayer.getOutputsCount());
            }

            std::transform(gradBiasesFP16.begin(), gradBiasesFP16.begin() + gradBiases.size(), gradBiases.begin(), [](half val) { return toFloat32(val); });
        }
    }
}

} // namespace raul
