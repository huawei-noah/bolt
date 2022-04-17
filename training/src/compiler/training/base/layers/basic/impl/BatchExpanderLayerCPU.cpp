// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "BatchExpanderLayerCPU.h"
#include "../BatchExpanderLayer.h"

namespace raul
{

template<typename MM>
BatchExpanderLayerCPU<MM>::BatchExpanderLayerCPU(BatchExpanderLayer& layer)
    : mLayer(layer)
{
}

template<typename MM>
void BatchExpanderLayerCPU<MM>::forwardComputeImpl(NetworkMode)
{
    auto& work = mLayer.mNetworkParams.mWorkflow;

    // Size check
    if (work.getMemoryManager<MM>()[mLayer.mInputs[0]].getBatchSize() != 1)
    {
        THROW(mLayer.mTypeName, mLayer.mName, "input tensor should have batch_size = 1");
    }

    const auto& input = work.getMemoryManager<MM>()[mLayer.mInputs[0]];
    auto& output = work.getMemoryManager<MM>()[mLayer.mOutputs[0]];

    const size_t inputSize = input.size();
#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (size_t q = 0; q < output.size(); ++q)
    {
        output[q] = input[q % inputSize];
    }
}

template<typename MM>
void BatchExpanderLayerCPU<MM>::backwardComputeImpl()
{
    auto& work = mLayer.mNetworkParams.mWorkflow;

    const auto& deltas = work.getMemoryManager<MM>()[mLayer.mOutputs[0].grad()];

    // if (mNetworkParams.isGradNeeded(mInputs[0]))
    {
        const auto batchSize = deltas.getBatchSize();
        const auto size = deltas.getDepth() * deltas.getHeight() * deltas.getWidth();
        auto deltas2D = deltas.reshape(yato::dims(batchSize, size));

        auto& prevLayerNabla = work.getMemoryManager<MM>()[mLayer.mInputs[0].grad()];
#if defined(_OPENMP)
#pragma omp parallel for
#endif
        for (size_t i = 0; i < size; ++i)
        {
            for (size_t j = 0; j < batchSize; ++j)
            {
                prevLayerNabla[i] += deltas2D[j][i];
            }
        }
    }
}

template class BatchExpanderLayerCPU<MemoryManager>;
template class BatchExpanderLayerCPU<MemoryManagerFP16>;
} // namespace raul
