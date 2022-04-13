// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "RoundLayerCPU.h"
#include "../RoundLayer.h"

#include <cfenv>

namespace raul
{

#if defined(_MSC_VER)
#pragma fenv_access(on)
#endif

template<typename MM>
RoundLayerCPU<MM>::RoundLayerCPU(RoundLayer& layer)
    : mLayer(layer)
{
}

template<typename MM>
void RoundLayerCPU<MM>::forwardComputeImpl(NetworkMode)
{
    auto mode = std::fegetround();
    std::fesetround(FE_TONEAREST);

    auto& memoryManager = mLayer.mNetworkParams.mWorkflow.getMemoryManager<MM>();
    const auto& input = memoryManager[mLayer.mInputs[0]];
    auto& output = memoryManager[mLayer.mOutputs[0]];

#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (size_t e = 0; e < input.size(); ++e)
    {
        output[e] = TODTYPE(lrint(input[e]));
    }

    std::fesetround(mode);
}

template<>
void RoundLayerCPU<MemoryManagerFP16>::forwardComputeImpl(NetworkMode)
{
    auto mode = std::fegetround();
    std::fesetround(FE_TONEAREST);

    auto& memoryManager = mLayer.mNetworkParams.mWorkflow.getMemoryManager<MemoryManagerFP16>();
    const auto& input = memoryManager[mLayer.mInputs[0]];
    auto& output = memoryManager[mLayer.mOutputs[0]];

#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (size_t e = 0; e < input.size(); ++e)
    {
        output[e] = TOHTYPE(lrint(toFloat32(input[e])));
    }

    std::fesetround(mode);
}

template<typename MM>
void RoundLayerCPU<MM>::backwardComputeImpl()
{
}

template class RoundLayerCPU<MemoryManager>;
template class RoundLayerCPU<MemoryManagerFP16>;
} // namespace raul
