// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "NonZeroMaskLayerCPU.h"
#include <training/base/layers/basic/NonZeroMaskLayer.h>

namespace raul
{

template<typename MM>
NonZeroMaskLayerCPU<MM>::NonZeroMaskLayerCPU(NonZeroMaskLayer& layer)
    : mLayer(layer)
{
}

template<typename MM>
void NonZeroMaskLayerCPU<MM>::forwardComputeImpl(NetworkMode)
{
    auto& memoryManager = mLayer.mNetworkParams.mWorkflow.getMemoryManager<MM>();
    const auto& input = memoryManager[mLayer.mInputs[0]];
    auto& mask = memoryManager[mLayer.mOutputs[0]];

#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (size_t e = 0; e < input.size(); ++e)
    {
        mask[e] = TOMMTYPE(input[e] == 0 ? 0 : 1);
    }
}

template<typename MM>
void NonZeroMaskLayerCPU<MM>::backwardComputeImpl()
{
}

template class NonZeroMaskLayerCPU<MemoryManager>;
template class NonZeroMaskLayerCPU<MemoryManagerFP16>;
} // namespace raul
