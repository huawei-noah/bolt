// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "SqrtLayerCPU.h"
#include "../SqrtLayer.h"

#include <atomic>
#include <cmath>

namespace raul
{

template<typename MM>
SqrtLayerCPU<MM>::SqrtLayerCPU(SqrtLayer& layer)
    : mLayer(layer)
{
}

template<typename MM>
void SqrtLayerCPU<MM>::forwardComputeImpl(NetworkMode)
{
    auto& work = mLayer.mNetworkParams.mWorkflow;

    auto& output = work.getMemoryManager<MM>()[mLayer.mOutputs[0]];
    const auto& input = work.getMemoryManager<MM>()[mLayer.mInputs[0]];

    std::atomic<bool> NegativeNumberDetected = false;

#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (size_t q = 0; q < output.size(); ++q)
    {
        if (TODTYPE(input[q]) < 0.0_dt)
        {
            NegativeNumberDetected = true;
        }
        output[q] = static_cast<std::remove_reference_t<decltype(output[q])>>(std::sqrt(TODTYPE(input[q])));
    }

    mLayer.mNegativeNumberDetected = NegativeNumberDetected;
    if (mLayer.mNegativeNumberDetected)
    {
        throw std::runtime_error(mLayer.mTypeName + "[" + mLayer.mName + "::forwardCompute]: negative input for sqrt()");
    }
}

template<typename MM>
void SqrtLayerCPU<MM>::backwardComputeImpl()
{
    if (mLayer.mNegativeNumberDetected)
    {
        throw std::runtime_error(mLayer.mTypeName + "[" + mLayer.mName + "::backwardCompute]: negative input for sqrt()");
    }

    auto& work = mLayer.mNetworkParams.mWorkflow;

    const auto& output = work.getMemoryManager<MM>()[mLayer.mOutputs[0]];
    const auto& deltas = work.getMemoryManager<MM>()[mLayer.mOutputs[0].grad()];

    // if (mLayer.mNetworkParams.isGradNeeded(mLayer.mInputs[0]))
    {
        auto& nabla_tensor = work.getMemoryManager<MM>()[mLayer.mInputs[0].grad()];

#if defined(_OPENMP)
#pragma omp parallel for
#endif
        for (size_t q = 0; q < output.size(); ++q)
        {
            nabla_tensor[q] += static_cast<std::remove_reference_t<decltype(output[q])>>(0.5_dt * deltas[q] / output[q]);
        }
    }
}

template class SqrtLayerCPU<MemoryManager>;
template class SqrtLayerCPU<MemoryManagerFP16>;

} // namespace raul