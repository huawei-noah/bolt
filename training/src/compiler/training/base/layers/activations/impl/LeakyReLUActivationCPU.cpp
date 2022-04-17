// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "LeakyReLUActivationCPU.h"
#include "../LeakyReLUActivation.h"

namespace raul
{

template<typename MM>
void LeakyReLUActivationCPU<MM>::forwardComputeImpl(NetworkMode)
{
    Workflow& work = mLayer.mNetworkParams.mWorkflow;

    auto& output = work.getMemoryManager<MM>().getTensor(mLayer.mOutputName);
    const auto& input = work.getMemoryManager<MM>().getTensor(mLayer.mInputName);

#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (size_t q = 0; q < output.size(); ++q)
    {
        output[q] = static_cast<typename MM::type>(std::max(0.0_dt, TODTYPE(input[q])) + mLayer.mNegativeSlope * std::min(0.0_dt, TODTYPE(input[q])));
    }
}

template<typename MM>
void LeakyReLUActivationCPU<MM>::backwardComputeImpl()
{
    Workflow& work = mLayer.mNetworkParams.mWorkflow;

    const auto& output = work.getMemoryManager<MM>().getTensor(mLayer.mOutputName);
    const auto& delta = work.getMemoryManager<MM>().getTensor(mLayer.mOutputName.grad());
    auto& prevLayerDelta = work.getMemoryManager<MM>().getTensor(mLayer.mInputName.grad());

#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (size_t q = 0; q < prevLayerDelta.size(); ++q)
    {
        prevLayerDelta[q] += static_cast<typename MM::type>(TODTYPE(output[q]) > 0.0_dt ? delta[q] : delta[q] * mLayer.mNegativeSlope);
    }
}

template class LeakyReLUActivationCPU<MemoryManager>;
template class LeakyReLUActivationCPU<MemoryManagerFP16>;

} // namespace raul