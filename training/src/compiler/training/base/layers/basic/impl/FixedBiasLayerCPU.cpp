// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "FixedBiasLayerCPU.h"
#include "../FixedBiasLayer.h"

#include <training/base/impl/ImplFactory.h>
namespace
{
bool reg1 = raul::TheImplFactory::Instance().regCPUFP32<raul::FixedBiasLayer, raul::FixedBiasLayerCPU<raul::MemoryManager>>();
bool reg2 = raul::TheImplFactory::Instance().regCPUFP16<raul::FixedBiasLayer, raul::FixedBiasLayerCPU<raul::MemoryManagerFP16>>();
} // anonymous namespace

namespace raul
{

template<typename MM>
FixedBiasLayerCPU<MM>::FixedBiasLayerCPU(FixedBiasLayer& layer)
    : mLayer(layer)
{
}

template<typename MM>
void FixedBiasLayerCPU<MM>::forwardComputeImpl(NetworkMode)
{
    auto& work = mLayer.mNetworkParams.mWorkflow;

    auto& output = work.getMemoryManager<MM>()[mLayer.mOutputs[0]];
    const auto& input = work.getMemoryManager<MM>()[mLayer.mInputs[0]];

#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (size_t q = 0; q < output.size(); ++q)
    {
        output[q] = input[q] + static_cast<std::remove_reference_t<decltype(output[q])>>(mLayer.mBias);
    }
}

template<typename MM>
void FixedBiasLayerCPU<MM>::backwardComputeImpl()
{
    auto& work = mLayer.mNetworkParams.mWorkflow;

    const auto& deltas = work.getMemoryManager<MM>()[mLayer.mOutputs[0].grad()];

    // if (mLayer.mNetworkParams.isGradNeeded(mLayer.mInputs[0]))
    {
        auto& nabla_tensor = work.getMemoryManager<MM>()[mLayer.mInputs[0].grad()];
#if defined(_OPENMP)
#pragma omp parallel for
#endif
        for (size_t q = 0; q < deltas.size(); ++q)
        {
            nabla_tensor[q] += deltas[q];
        }
    }
}

template class FixedBiasLayerCPU<MemoryManager>;
template class FixedBiasLayerCPU<MemoryManagerFP16>;

} // namespace raul