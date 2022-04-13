// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "LinearLayerCPUFP32.h"
#include <training/base/layers/basic/trainable/LinearLayer.h>
#include <training/base/impl/ImplFactory.h>

namespace raul
{

std::shared_ptr<Tensor> LinearLayerCPUFP32::mInput;
std::shared_ptr<Tensor> LinearLayerCPUFP32::mOutput;
std::shared_ptr<Tensor> LinearLayerCPUFP32::mDeltas;
std::shared_ptr<Tensor> LinearLayerCPUFP32::mPrevLayerDeltas;
std::shared_ptr<Tensor> LinearLayerCPUFP32::mWeights;
std::shared_ptr<Tensor> LinearLayerCPUFP32::mGradWeights;
std::shared_ptr<Tensor> LinearLayerCPUFP32::mBiases;
std::shared_ptr<Tensor> LinearLayerCPUFP32::mGradBiases;

void LinearLayerCPUFP32::forwardComputeImpl(NetworkMode)
{
    Workflow& work = mLayer.getNetworkParams().mWorkflow;

    auto& output = work.getMemoryManager<MemoryManagerFP16>()[mLayer.getOutputName()];
    if (!mOutput)
    {
        mOutput = std::make_shared<Tensor>(output.size());
    }
    if (mOutput->size() < output.size())
    {
        mOutput = std::make_shared<Tensor>(output.size());
    }
    std::transform(output.begin(), output.end(), mOutput->begin(), [](half val) { return toFloat32(val); });
    //*mOutput = TORANGE(output);

    const size_t batchSize = mLayer.getNetworkParams().mWorkflow.getBatchSize();
    size_t N = batchSize * mLayer.getDepth() * mLayer.getHeight();

    const auto& inputs = work.getMemoryManager<MemoryManagerFP16>()[mLayer.getInputName()];
    if (!mInput)
    {
        mInput = std::make_shared<Tensor>(inputs.size());
    }
    if (mInput->size() < inputs.size())
    {
        mInput = std::make_shared<Tensor>(inputs.size());
    }
    std::transform(inputs.begin(), inputs.end(), mInput->begin(), [](half val) { return toFloat32(val); });
    //*mInput = TORANGE(inputs);

    const auto& weights = work.getMemoryManager<MemoryManagerFP16>()[mLayer.getWeightsName()];
    if (!mWeights)
    {
        mWeights = std::make_shared<Tensor>(weights.size());
    }
    if (mWeights->size() < weights.size())
    {
        mWeights = std::make_shared<Tensor>(weights.size());
    }
    std::transform(weights.begin(), weights.end(), mWeights->begin(), [](half val) { return toFloat32(val); });
    //*mWeights = TORANGE(weights);

    Common::gemm(CblasNoTrans,
                 CblasTrans,
                 N,
                 mLayer.getOutputsCount(),
                 mLayer.getInputsCount(),
                 1.0_dt,
                 &(*mInput)[0],
                 &(*mWeights)[0],
                 0.0_dt,
                 &(*mOutput)[0]);

    if (mLayer.isUseBias())
    {
        const auto& biases = work.getMemoryManager<MemoryManagerFP16>()[mLayer.getBiasesName()];
        if (!mBiases)
        {
            mBiases = std::make_shared<Tensor>(biases.size());
        }
        if (mBiases->size() < biases.size())
        {
            mBiases = std::make_shared<Tensor>(biases.size());
        }
        //*mBiases = TORANGE(biases);
        std::transform(biases.begin(), biases.end(), mBiases->begin(), [](half val) { return toFloat32(val); });

        for (size_t index = 0; index < N; ++index)
        {
            Common::axpy(mLayer.getOutputsCount(), 1.0_dt, &(*mBiases)[0], 1, &(*mOutput)[0], 1, 0, index * mLayer.getOutputsCount());
        }
    }

    std::transform(mOutput->begin(), mOutput->begin() + output.size(), output.begin(), [](dtype val) { return toFloat16(val); });
    // output = TORANGE_FP16((*mOutput));
}

void LinearLayerCPUFP32::backwardComputeImpl()
{
    Workflow& work = mLayer.getNetworkParams().mWorkflow;

    const auto& deltas = work.getMemoryManager<MemoryManagerFP16>()[mLayer.getOutputName().grad()];
    if (!mDeltas)
    {
        mDeltas = std::make_shared<Tensor>(deltas.size());
    }
    if (mDeltas->size() < deltas.size())
    {
        mDeltas = std::make_shared<Tensor>(deltas.size());
    }
    //*mDeltas = TORANGE(deltas);
    std::transform(deltas.begin(), deltas.end(), mDeltas->begin(), [](half val) { return toFloat32(val); });

    const size_t batchSize = mLayer.getNetworkParams().mWorkflow.getBatchSize();
    size_t N = batchSize * mLayer.getDepth() * mLayer.getHeight();

    const auto& weights = work.getMemoryManager<MemoryManagerFP16>()[mLayer.getWeightsName()];
    //*mWeights = TORANGE(weights);
    std::transform(weights.begin(), weights.end(), mWeights->begin(), [](half val) { return toFloat32(val); });

    ////if (mNetworkParams.isGradNeeded(mInputName))
    {
        auto& prevLayerDelta = work.getMemoryManager<MemoryManagerFP16>()[mLayer.getInputName().grad()];
        if (!mPrevLayerDeltas)
        {
            mPrevLayerDeltas = std::make_shared<Tensor>(prevLayerDelta.size());
        }
        if (mPrevLayerDeltas->size() < prevLayerDelta.size())
        {
            mPrevLayerDeltas = std::make_shared<Tensor>(prevLayerDelta.size());
        }
        //*mPrevLayerDeltas = TORANGE(prevLayerDelta);
        std::transform(prevLayerDelta.begin(), prevLayerDelta.end(), mPrevLayerDeltas->begin(), [](half val) { return toFloat32(val); });

        Common::gemm(CblasNoTrans,
                     CblasNoTrans,
                     N,
                     mLayer.getInputsCount(),
                     mLayer.getOutputsCount(),
                     1.0_dt,
                     &(*mDeltas)[0],
                     &(*mWeights)[0],
                     1.0_dt,
                     &(*mPrevLayerDeltas)[0]);

        // prevLayerDelta = TORANGE_FP16(*mPrevLayerDeltas);
        std::transform(mPrevLayerDeltas->begin(), mPrevLayerDeltas->begin() + prevLayerDelta.size(), prevLayerDelta.begin(), [](dtype val) { return toFloat16(val); });
    }

    if (!mLayer.isFrozen())
    {
        const auto& inputs = work.getMemoryManager<MemoryManagerFP16>()[mLayer.getInputName()];
        //*mInput = TORANGE(inputs);
        std::transform(inputs.begin(), inputs.end(), mInput->begin(), [](half val) { return toFloat32(val); });

        auto& gradWeights = work.getMemoryManager<MemoryManagerFP16>()[mLayer.getWeightsName().grad()];
        if (!mGradWeights)
        {
            mGradWeights = std::make_shared<Tensor>(gradWeights.size());
        }
        if (mGradWeights->size() < gradWeights.size())
        {
            mGradWeights = std::make_shared<Tensor>(gradWeights.size());
        }
        //*mGradWeights = TORANGE(gradWeights);
        std::transform(gradWeights.begin(), gradWeights.end(), mGradWeights->begin(), [](half val) { return toFloat32(val); });

        Common::gemm(CblasTrans,
                     CblasNoTrans,
                     mLayer.getOutputsCount(),
                     mLayer.getInputsCount(),
                     N,
                     1.0_dt,
                     &(*mDeltas)[0],
                     &(*mInput)[0],
                     1.0_dt,
                     &(*mGradWeights)[0]);

        // gradWeights = TORANGE_FP16(*mGradWeights);
        std::transform(mGradWeights->begin(), mGradWeights->begin() + gradWeights.size(), gradWeights.begin(), [](dtype val) { return toFloat16(val); });

        if (mLayer.isUseBias())
        {
            auto& gradBiases = work.getMemoryManager<MemoryManagerFP16>()[mLayer.getBiasesName().grad()];
            if (!mGradBiases)
            {
                mGradBiases = std::make_shared<Tensor>(gradBiases.size());
            }
            if (mGradBiases->size() < gradBiases.size())
            {
                mGradBiases = std::make_shared<Tensor>(gradBiases.size());
            }
            //*mGradBiases = TORANGE(gradBiases);
            std::transform(gradBiases.begin(), gradBiases.end(), mGradBiases->begin(), [](half val) { return toFloat32(val); });

            for (size_t index = 0; index < N; ++index)
            {
                Common::axpy(mLayer.getOutputsCount(), 1.0_dt, &(*mDeltas)[0], 1, &(*mGradBiases)[0], 1, index * mLayer.getOutputsCount());
            }

            // gradBiases = TORANGE_FP16(*mGradBiases);
            std::transform(mGradBiases->begin(), mGradBiases->begin() + gradBiases.size(), gradBiases.begin(), [](dtype val) { return toFloat16(val); });
        }
    }
}

} // namespace raul