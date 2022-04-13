// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "LinearLayerImpl.h"
#include <training/base/layers/basic/trainable/LinearLayer.h>

#include <training/base/impl/ImplFactory.h>

namespace
{
bool reg1 = raul::TheImplFactory::Instance().regCPUFP32<raul::LinearLayer, raul::LinearLayerImpl<raul::MemoryManager>>();
bool reg2 = raul::TheImplFactory::Instance().regCPUFP16<raul::LinearLayer, raul::LinearLayerImpl<raul::MemoryManagerFP16>>();
} // anonymous namespace

namespace raul
{

template<typename MM>
void LinearLayerImpl<MM>::onBatchSizeChanged(size_t)
{
}

template<typename MM>
void LinearLayerImpl<MM>::initNotBSTensors()
{
}

template<typename MM>
void LinearLayerImpl<MM>::forwardComputeImpl(NetworkMode)
{
    Workflow& work = mLayer.getNetworkParams().mWorkflow;

    auto& output = work.getMemoryManager<MM>().getTensor(mLayer.getOutputName());
    const size_t batchSize = mLayer.getNetworkParams().mWorkflow.getBatchSize();
    size_t N = batchSize * mLayer.getDepth() * mLayer.getHeight();

    auto& inputs = work.getMemoryManager<MM>().getTensor(mLayer.getInputName());

    const auto& weights = work.getMemoryManager<MM>().getTensor(mLayer.getWeightsName());

    auto beta = 0.0_dt;
    if (mLayer.isUseBias())
    {
        const auto& biases = work.getMemoryManager<MM>().getTensor(mLayer.getBiasesName());
        for (size_t i = 0; i < N; i++)
        {
            std::copy(biases.cbegin(), biases.cend(), output.begin() + i * mLayer.getOutputsCount());
        }

        beta = 1.0_dt;
    }

    Common::gemm(CblasNoTrans,
                 CblasTrans,
                 N,
                 mLayer.getOutputsCount(),
                 mLayer.getInputsCount(),
                 1.0_dt,
                 inputs.getBuffer(),
                 weights.getBuffer(),
                 beta,
                 output.getBuffer());
}

template<typename MM>
void LinearLayerImpl<MM>::backwardComputeImpl()
{
    Workflow& work = mLayer.getNetworkParams().mWorkflow;

    const auto& deltas = work.getMemoryManager<MM>().getTensor(mLayer.getOutputName().grad());

    const size_t batchSize = mLayer.getNetworkParams().mWorkflow.getBatchSize();
    size_t N = batchSize * mLayer.getDepth() * mLayer.getHeight();

    const auto& weights = work.getMemoryManager<MM>().getTensor(mLayer.getWeightsName());

    ////if (mNetworkParams.isGradNeeded(mInputName))
    {
        auto& prevLayerDelta = work.getMemoryManager<MM>().getTensor(mLayer.getInputName().grad());

        Common::gemm(CblasNoTrans,
                     CblasNoTrans,
                     N,
                     mLayer.getInputsCount(),
                     mLayer.getOutputsCount(),
                     1.0_dt,
                     deltas.getBuffer(),
                     weights.getBuffer(),
                     1.0_dt,
                     prevLayerDelta.getBuffer());
    }

    if (!mLayer.isFrozen())
    {
        const auto& inputs = work.getMemoryManager<MM>().getTensor(mLayer.getInputName());
        auto& gradWeights = work.getMemoryManager<MM>().getTensor(mLayer.getWeightsName().grad());

        Common::gemm(CblasTrans,
                     CblasNoTrans,
                     mLayer.getOutputsCount(),
                     mLayer.getInputsCount(),
                     N,
                     1.0_dt,
                     deltas.getBuffer(),
                     inputs.getBuffer(),
                     1.0_dt,
                     gradWeights.getBuffer());

        if (mLayer.isUseBias())
        {
            auto& gradBiases = work.getMemoryManager<MM>().getTensor(mLayer.getBiasesName().grad());
            for (size_t i = 0; i < N; i++)
            {
                std::transform(deltas.cbegin() + i * mLayer.getOutputsCount(),
                               deltas.cbegin() + i * mLayer.getOutputsCount() + mLayer.getOutputsCount(),
                               gradBiases.cbegin(),
                               gradBiases.begin(),
                               std::plus<typename MM::type>());
            }
        }
    }
}

INSTANTIATE_IMPL(LinearLayerImpl)

} // namespace raul