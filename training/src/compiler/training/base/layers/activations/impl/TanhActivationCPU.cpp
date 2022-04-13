// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "TanhActivationCPU.h"
#include "../TanhActivation.h"

#include <algorithm>

#include <training/base/impl/ImplFactory.h>
namespace
{
bool reg1 = raul::TheImplFactory::Instance().regCPUFP32<raul::TanhActivation, raul::TanhActivationCPU<raul::MemoryManager>>();
bool reg2 = raul::TheImplFactory::Instance().regCPUFP16<raul::TanhActivation, raul::TanhActivationCPU<raul::MemoryManagerFP16>>();
} // anonymous namespace

namespace raul
{

template<typename MM>
void TanhActivationCPU<MM>::forwardComputeImpl(NetworkMode)
{
    Workflow& work = mLayer.mNetworkParams.mWorkflow;

    const auto& input = work.getMemoryManager<MM>()[mLayer.mInputName];
    auto& output = work.getMemoryManager<MM>()[mLayer.mOutputName];

#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (size_t q = 0; q < output.size(); ++q)
    {
        output[q] = static_cast<typename MM::type>(std::tanh(TODTYPE(input[q])));
    }
}

template<typename MM>
void TanhActivationCPU<MM>::backwardComputeImpl()
{
    Workflow& work = mLayer.mNetworkParams.mWorkflow;

    const auto& deltas = work.getMemoryManager<MM>()[mLayer.mOutputName.grad()];
    const auto& output = work.getMemoryManager<MM>()[mLayer.mOutputName];

    ////if (mLayer.mNetworkParams.isGradNeeded(mLayer.mInputName))
    {
        auto& prevLayerDelta = work.getMemoryManager<MM>()[mLayer.mInputName.grad()];
#if defined(_OPENMP)
#pragma omp parallel for
#endif
        for (size_t q = 0; q < prevLayerDelta.size(); ++q)
        {
            prevLayerDelta[q] += deltas[q] * (static_cast<typename MM::type>(1.0_dt) - output[q] * output[q]);
        }
    }
}

template class TanhActivationCPU<MemoryManager>;
template class TanhActivationCPU<MemoryManagerFP16>;

} // namespace raul