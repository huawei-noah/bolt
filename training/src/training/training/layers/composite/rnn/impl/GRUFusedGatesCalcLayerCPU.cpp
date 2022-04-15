// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "GRUFusedGatesCalcLayerCPU.h"
#include "../GRUFusedGatesCalcLayer.h"

namespace raul
{

template<typename MM>
void GRUFusedGatesCalcLayerCPU<MM>::forwardComputeImpl(NetworkMode)
{
    Workflow& work = mLayer.mNetworkParams.mWorkflow;

    const auto& linearIH = work.getMemoryManager<MM>()[mLayer.mInputs[0]];
    const auto& linearHH = work.getMemoryManager<MM>()[mLayer.mInputs[1]];
    const auto& hiddenState = work.getMemoryManager<MM>()[mLayer.mInputs[2]];

    auto& newHiddenState = work.getMemoryManager<MM>()[mLayer.mOutputs[0]];

    const auto batchSize = work.getBatchSize();
    const auto sliceSize = linearIH.getWidth() / 3;

    const auto linearIH2D = linearIH.reshape(yato::dims(batchSize, sliceSize * 3));
    const auto linearHH2D = linearHH.reshape(yato::dims(batchSize, sliceSize * 3));
    const auto hiddenState2D = hiddenState.reshape(yato::dims(batchSize, sliceSize));
    const auto newHiddenState2D = newHiddenState.reshape(yato::dims(batchSize, sliceSize));

#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (size_t i = 0; i < batchSize; ++i)
    {
        for (size_t j = 0; j < sliceSize; ++j)
        {
            auto sigmoidGates0 = 1.0_dt / (1.0_dt + std::exp(-TODTYPE(linearIH2D[i][j]) - TODTYPE(linearHH2D[i][j])));
            auto sigmoidGates1 = 1.0_dt / (1.0_dt + std::exp(-TODTYPE(linearIH2D[i][sliceSize + j]) - TODTYPE(linearHH2D[i][sliceSize + j])));
            auto tanhGates2 = std::tanh(sigmoidGates0 * TODTYPE(linearHH2D[i][sliceSize * 2 + j]) + TODTYPE(linearIH2D[i][sliceSize * 2 + j]));
            newHiddenState2D[i][j] = TOMMTYPE(sigmoidGates1 * TODTYPE(hiddenState2D[i][j]) + tanhGates2 * (1.0_dt - sigmoidGates1));
        }
    }
}

template<typename MM>
void GRUFusedGatesCalcLayerCPU<MM>::backwardComputeImpl()
{
    Workflow& work = mLayer.mNetworkParams.mWorkflow;

    const auto& deltasHidden = work.getMemoryManager<MM>()[mLayer.mOutputs[0].grad()];
    const auto& linearIH = work.getMemoryManager<MM>()[mLayer.mInputs[0]];
    const auto& linearHH = work.getMemoryManager<MM>()[mLayer.mInputs[1]];
    const auto& hiddenState = work.getMemoryManager<MM>()[mLayer.mInputs[2]];

    const auto batchSize = work.getBatchSize();
    const auto sliceSize = linearIH.getWidth() / 3;

    const auto deltasHidden2D = deltasHidden.reshape(yato::dims(batchSize, sliceSize));
    const auto linearIH2D = linearIH.reshape(yato::dims(batchSize, sliceSize * 3));
    const auto linearHH2D = linearHH.reshape(yato::dims(batchSize, sliceSize * 3));
    const auto hiddenState2D = hiddenState.reshape(yato::dims(batchSize, sliceSize));

    // if (mLayer.mNetworkParams.isGradNeeded(mLayer.mInputs[0]))
    {
        auto& linearIHGrad = work.getMemoryManager<MM>()[mLayer.mInputs[0].grad()];
        auto linearIHGrad2D = linearIHGrad.reshape(yato::dims(batchSize, sliceSize * 3));

#if defined(_OPENMP)
#pragma omp parallel for
#endif
        for (size_t i = 0; i < batchSize; ++i)
        {
            for (size_t j = 0; j < sliceSize; ++j)
            {
                auto sigmoidGates0 = 1.0_dt / (1.0_dt + std::exp(-TODTYPE(linearIH2D[i][j]) - TODTYPE(linearHH2D[i][j])));
                auto sigmoidGates1 = 1.0_dt / (1.0_dt + std::exp(-TODTYPE(linearIH2D[i][sliceSize + j]) - TODTYPE(linearHH2D[i][sliceSize + j])));
                auto tanhGates2 = std::tanh(sigmoidGates0 * TODTYPE(linearHH2D[i][sliceSize * 2 + j]) + TODTYPE(linearIH2D[i][sliceSize * 2 + j]));
                linearIHGrad2D[i][j] += TOMMTYPE(sigmoidGates0 * (1.0_dt - sigmoidGates0) * TODTYPE(linearHH2D[i][sliceSize * 2 + j]) * (1.0_dt - tanhGates2 * tanhGates2) * (1.0_dt - sigmoidGates1) *
                                                 TODTYPE(deltasHidden2D[i][j]));
                linearIHGrad2D[i][sliceSize + j] += TOMMTYPE(sigmoidGates1 * (1.0_dt - sigmoidGates1) * (TODTYPE(hiddenState2D[i][j]) - tanhGates2) * TODTYPE(deltasHidden2D[i][j]));
                linearIHGrad2D[i][sliceSize * 2 + j] += TOMMTYPE((1.0_dt - tanhGates2 * tanhGates2) * (1.0_dt - sigmoidGates1) * TODTYPE(deltasHidden2D[i][j]));
            }
        }
    }

    // if (mLayer.mNetworkParams.isGradNeeded(mLayer.mInputs[1]))
    {
        auto& linearHHGrad = work.getMemoryManager<MM>()[mLayer.mInputs[1].grad()];
        auto linearHHGrad2D = linearHHGrad.reshape(yato::dims(batchSize, sliceSize * 3));

#if defined(_OPENMP)
#pragma omp parallel for
#endif
        for (size_t i = 0; i < batchSize; ++i)
        {
            for (size_t j = 0; j < sliceSize; ++j)
            {
                auto sigmoidGates0 = 1.0_dt / (1.0_dt + std::exp(-TODTYPE(linearIH2D[i][j]) - TODTYPE(linearHH2D[i][j])));
                auto sigmoidGates1 = 1.0_dt / (1.0_dt + std::exp(-TODTYPE(linearIH2D[i][sliceSize + j]) - TODTYPE(linearHH2D[i][sliceSize + j])));
                auto tanhGates2 = std::tanh(sigmoidGates0 * TODTYPE(linearHH2D[i][sliceSize * 2 + j]) + TODTYPE(linearIH2D[i][sliceSize * 2 + j]));
                linearHHGrad2D[i][j] += TOMMTYPE(sigmoidGates0 * (1.0_dt - sigmoidGates0) * TODTYPE(linearHH2D[i][sliceSize * 2 + j]) * (1.0_dt - tanhGates2 * tanhGates2) * (1.0_dt - sigmoidGates1) *
                                                 TODTYPE(deltasHidden2D[i][j]));
                linearHHGrad2D[i][sliceSize + j] += TOMMTYPE(sigmoidGates1 * (1.0_dt - sigmoidGates1) * (TODTYPE(hiddenState2D[i][j]) - tanhGates2) * TODTYPE(deltasHidden2D[i][j]));
                linearHHGrad2D[i][sliceSize * 2 + j] += TOMMTYPE(sigmoidGates0 * (1.0_dt - tanhGates2 * tanhGates2) * (1.0_dt - sigmoidGates1) * TODTYPE(deltasHidden2D[i][j]));
            }
        }
    }

    // if (mLayer.mNetworkParams.isGradNeeded(mLayer.mInputs[2]))
    {
        auto& hiddenStateGrad = work.getMemoryManager<MM>()[mLayer.mInputs[2].grad()];
        auto hiddenStateGrad2D = hiddenStateGrad.reshape(yato::dims(batchSize, sliceSize));

#if defined(_OPENMP)
#pragma omp parallel for
#endif
        for (size_t i = 0; i < batchSize; ++i)
        {
            for (size_t j = 0; j < sliceSize; ++j)
            {
                auto sigmoidGates1 = 1.0_dt / (1.0_dt + std::exp(-TODTYPE(linearIH2D[i][sliceSize + j]) - TODTYPE(linearHH2D[i][sliceSize + j])));
                hiddenStateGrad2D[i][j] += TOMMTYPE(sigmoidGates1 * TODTYPE(deltasHidden2D[i][j]));
            }
        }
    }
}

template class GRUFusedGatesCalcLayerCPU<MemoryManager>;
template class GRUFusedGatesCalcLayerCPU<MemoryManagerFP16>;

} // namespace raul