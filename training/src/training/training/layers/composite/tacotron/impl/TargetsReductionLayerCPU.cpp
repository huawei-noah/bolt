// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "TargetsReductionLayerCPU.h"
#include "../TargetsReductionLayer.h"

namespace raul
{
namespace tacotron
{

template<typename MM>
void TargetsReductionLayerCPU<MM>::forwardComputeImpl(NetworkMode)
{
    // reduce mel_targets
    const auto& targets = mLayer.mNetworkParams.mWorkflow.getMemoryManager<MM>()[mLayer.mMelTargetsName];
    auto& reducedTargets = mLayer.mNetworkParams.mWorkflow.getMemoryManager<MM>()[mLayer.mReducedMelTargetsName];

    auto targets3D = targets.reshape(yato::dims(targets.getBatchSize(), targets.getDepth() * targets.getHeight(), targets.getWidth()));
    auto reducedTargets3D = reducedTargets.reshape(yato::dims(targets.getBatchSize(), reducedTargets.getDepth() * reducedTargets.getHeight(), targets.getWidth()));

#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (size_t q = 0; q < targets.getBatchSize(); ++q)
    {
        for (size_t i = 0; i < reducedTargets.getDepth() * reducedTargets.getHeight(); ++i)
        {
            auto src = targets3D[q][(mLayer.mReductionFactor - 1) + i * mLayer.mReductionFactor];
            auto tgt = reducedTargets3D[q][i];
            std::copy(src.begin(), src.end(), tgt.begin());
        }
    }
}

template<typename MM>
void TargetsReductionLayerCPU<MM>::backwardComputeImpl()
{
}

template class TargetsReductionLayerCPU<MemoryManager>;
template class TargetsReductionLayerCPU<MemoryManagerFP16>;

}
} // namespace raul
