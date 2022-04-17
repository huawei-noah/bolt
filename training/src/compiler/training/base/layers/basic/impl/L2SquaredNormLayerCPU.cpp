// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "L2SquaredNormLayerCPU.h"
#include "../L2SquaredNormLayer.h"

namespace raul
{

template<typename MM>
void L2SquaredNormLayerCPU<MM>::forwardComputeImpl(NetworkMode)
{
    Workflow& work = mLayer.mNetworkParams.mWorkflow;

    auto& output = work.getMemoryManager<MM>()[mLayer.mOutputs[0]];
    const auto& input = work.getMemoryManager<MM>()[mLayer.mInputs[0]];

    dtype sum = 0.0_dt;

#if defined(_OPENMP)
#pragma omp parallel for reduction(+ : sum)
#endif
    for (size_t q = 0; q < input.size(); ++q)
    {
        sum += TODTYPE(input[q]) * TODTYPE(input[q]);
    }
    output[0] = TOMMTYPE(sum / 2.0_dt);
}

template<typename MM>
void L2SquaredNormLayerCPU<MM>::backwardComputeImpl()
{
    Workflow& work = mLayer.mNetworkParams.mWorkflow;

    const auto& input = work.getMemoryManager<MM>()[mLayer.mInputs[0]];
    const auto& deltas = work.getMemoryManager<MM>()[mLayer.mOutputs[0].grad()];

    // if (mNetworkParams.isGradNeeded(mLayer.mInputs[0]))
    {
        auto& prevLayerDelta = work.getMemoryManager<MM>()[mLayer.mInputs[0].grad()];

#if defined(_OPENMP)
#pragma omp parallel for
#endif
        for (size_t q = 0; q < prevLayerDelta.size(); ++q)
        {
            prevLayerDelta[q] += deltas[0] * input[q];
        }
    }
}

template class L2SquaredNormLayerCPU<MemoryManager>;
template class L2SquaredNormLayerCPU<MemoryManagerFP16>;
} // namespace raul