// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "L2NormLayerCPU.h"
#include "../L2NormLayer.h"

namespace raul
{

template<typename MM>
void L2NormLayerCPU<MM>::forwardComputeImpl(NetworkMode)
{
    Workflow& work = mLayer.mNetworkParams.mWorkflow;

    auto& output = work.getMemoryManager<MM>()[mLayer.mOutputs[0]];
    const auto& input = work.getMemoryManager<MM>()[mLayer.mInputs[0]];

    size_t size = input.getBatchSize() * input.getDepth() * input.getHeight();
    auto input2D = input.reshape(yato::dims(size, input.getWidth()));
    auto output2D = output.reshape(yato::dims(size, input.getWidth()));

#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (size_t q = 0; q < size; ++q)
    {
        dtype sum = 0.0_dt;
        for (size_t i = 0; i < input.getWidth(); ++i)
        {
            sum += TODTYPE(input2D[q][i]) * TODTYPE(input2D[q][i]);
        }

        for (size_t i = 0; i < input.getWidth(); ++i)
        {
            output2D[q][i] = input2D[q][i] / TOMMTYPE(std::sqrt(sum));
        }
    }
}

template<typename MM>
void L2NormLayerCPU<MM>::backwardComputeImpl()
{
    Workflow& work = mLayer.mNetworkParams.mWorkflow;

    const auto& output = work.getMemoryManager<MM>()[mLayer.mOutputs[0]];
    const auto& input = work.getMemoryManager<MM>()[mLayer.mInputs[0]];
    const auto& deltas = work.getMemoryManager<MM>()[mLayer.mOutputs[0].grad()];

    // if (mNetworkParams.isGradNeeded(mLayer.mInputs[0]))
    {
        auto& prevLayerDelta = work.getMemoryManager<MM>()[mLayer.mInputs[0].grad()];

        size_t size = deltas.getBatchSize() * deltas.getDepth() * deltas.getHeight();
        auto deltas2D = deltas.reshape(yato::dims(size, deltas.getWidth()));
        auto prevLayerDelta2D = prevLayerDelta.reshape(yato::dims(size, deltas.getWidth()));
        auto output2D = output.reshape(yato::dims(size, deltas.getWidth()));
        auto input2D = input.reshape(yato::dims(size, deltas.getWidth()));
#if defined(_OPENMP)
#pragma omp parallel for
#endif
        for (size_t q = 0; q < size; ++q)
        {
            for (size_t i = 0; i < deltas.getWidth(); ++i)
            {
                dtype sum = 0.0_dt;
                for (size_t j = 0; j < deltas.getWidth(); ++j)
                {
                    if (i == j)
                    {
                        sum += output2D[q][j] / input2D[q][j];
                    }
                    sum -= deltas2D[q][j] * input2D[q][i] / TODTYPE(std::pow(TODTYPE(input2D[q][j]), 2) / std::pow(TODTYPE(output2D[q][j]), 3));
                }
                prevLayerDelta2D[q][i] += TOMMTYPE(sum);
            }
        }
    }
}

template class L2NormLayerCPU<MemoryManager>;
template class L2NormLayerCPU<MemoryManagerFP16>;
} // namespace raul