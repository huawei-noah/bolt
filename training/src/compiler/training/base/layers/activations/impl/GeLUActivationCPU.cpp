// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "GeLUActivationCPU.h"
#include "../GeLUActivation.h"

namespace raul
{

template<typename MM>
void GeLUErfCPU<MM>::forwardComputeImpl(NetworkMode)
{
    Workflow& work = mLayer.mNetworkParams.mWorkflow;

    auto& output = work.getMemoryManager<MM>().getTensor(mLayer.mOutputName);
    const auto& inputs = work.getMemoryManager<MM>().getTensor(mLayer.mInputName);

    std::transform(inputs.begin(), inputs.end(), output.begin(), [&](dtype val) -> dtype { return Common::GeLU_Erf(val); });
}

template<typename MM>
void GeLUErfCPU<MM>::backwardComputeImpl()
{
    Workflow& work = mLayer.mNetworkParams.mWorkflow;

    ////if (mLayer.mNetworkParams.isGradNeeded(mLayer.mInputName))
    {
        const auto& inputs = work.getMemoryManager<MM>().getTensor(mLayer.mInputName);
        const auto& deltas = work.getMemoryManager<MM>().getTensor(mLayer.mOutputName.grad());
        auto& prevLayerDelta = work.getMemoryManager<MM>().getTensor(mLayer.mInputName.grad());

#if defined(_OPENMP)
#pragma omp parallel for
#endif
        for (size_t q = 0; q < prevLayerDelta.size(); ++q)
        {
            auto x = inputs[q];
            prevLayerDelta[q] += static_cast<dtype>(deltas[q] * 0.5_dt * (1.0_dt + std::erf(x * RAUL_SQRT1_2) + x * RAUL_SQRT2_PI * exp(-0.5 * x * x)));
        }
    }
}

template<typename MM>
void GeLUTanhCPU<MM>::forwardComputeImpl(NetworkMode)
{
    Workflow& work = mLayer.mNetworkParams.mWorkflow;

    auto& output = work.getMemoryManager<MM>().getTensor(mLayer.mOutputName);
    const auto& inputs = work.getMemoryManager<MM>().getTensor(mLayer.mInputName);

    std::transform(inputs.begin(), inputs.end(), output.begin(), [&](dtype val) -> dtype { return Common::GeLU_Tanh(val); });
}

template<typename MM>
void GeLUTanhCPU<MM>::backwardComputeImpl()
{
    Workflow& work = mLayer.mNetworkParams.mWorkflow;

    ////if (mLayer.mNetworkParams.isGradNeeded(mLayer.mInputName))
    {
        const auto& inputs = work.getMemoryManager<MM>().getTensor(mLayer.mInputName);
        const auto& deltas = work.getMemoryManager<MM>().getTensor(mLayer.mOutputName.grad());
        auto& prevLayerDelta = work.getMemoryManager<MM>().getTensor(mLayer.mInputName.grad());

#if defined(_OPENMP)
#pragma omp parallel for
#endif
        for (size_t q = 0; q < prevLayerDelta.size(); ++q)
        {
            auto x = inputs[q];
            auto th = std::tanh(RAUL_SQRT2_PI * (x + GELU_CONST * std::pow(x, 3)));
            prevLayerDelta[q] += static_cast<dtype>(deltas[q] * 0.5_dt * (1.0_dt + th + x * RAUL_SQRT2_PI * (1.0_dt + 3 * GELU_CONST * x * x) * (1.0_dt - th * th)));
        }
    }
}

template class GeLUErfCPU<MemoryManager>;
template class GeLUTanhCPU<MemoryManager>;

} // namespace raul