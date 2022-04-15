// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "HSwishActivationCPU.h"
#include "../HSwishActivation.h"

#include <algorithm>

namespace raul
{

template<typename MM>
void HSwishActivationCPU<MM>::forwardComputeImpl(NetworkMode)
{
    Workflow& work = mLayer.mNetworkParams.mWorkflow;

    auto& output = work.getMemoryManager<MM>()[mLayer.mOutputName];

    const auto& inputs = work.getMemoryManager<MM>()[mLayer.mInputName];

    std::transform(inputs.begin(), inputs.end(), output.begin(), [&](typename MM::type val) -> typename MM::type { return Common::HSwish(val); });
}

template<typename MM>
void HSwishActivationCPU<MM>::backwardComputeImpl()
{
    // if (mLayer.mNetworkParams.isGradNeeded(mLayer.mInputName))
    {
        Workflow& work = mLayer.mNetworkParams.mWorkflow;

        const auto& deltas = work.getMemoryManager<MM>()[mLayer.mOutputName.grad()];
        auto& prevLayerDelta = work.getMemoryManager<MM>()[mLayer.mInputName.grad()];

        const auto& inputs = work.getMemoryManager<MM>()[mLayer.mInputName];

        for (size_t q = 0; q < prevLayerDelta.size(); ++q)
        {
            const auto grad_middle = [](const auto x, const auto grad) { return TOMMTYPE(TODTYPE(grad) * (TODTYPE(x) / 3.0_dt + 0.5_dt)); };

            if (TODTYPE(inputs[q]) > 3.0_dt)
            {
                prevLayerDelta[q] += deltas[q];
            }
            else if (TODTYPE(inputs[q]) > -3.0_dt && TODTYPE(inputs[q]) < 3.0_dt)
            {
                prevLayerDelta[q] += grad_middle(inputs[q], deltas[q]);
            }
            else if (TODTYPE(inputs[q]) == 3.0_dt)
            {
                switch (mLayer.p3PointVal)
                {
                    case Limit::Left:
                        prevLayerDelta[q] += grad_middle(inputs[q], deltas[q]);
                        break;
                    case Limit::Middle:
                        prevLayerDelta[q] += (grad_middle(inputs[q], deltas[q]) + deltas[q]) / TOMMTYPE(2.0f);
                        break;
                    case Limit::Right:
                        prevLayerDelta[q] += deltas[q];
                        break;
                        // default: Do nothing
                }
            }
            else if (TODTYPE(inputs[q]) == -3.0_dt)
            {
                switch (mLayer.m3PointVal)
                {
                    case Limit::Left:
                        break;
                    case Limit::Middle:
                        prevLayerDelta[q] += grad_middle(inputs[q], deltas[q]) / TOMMTYPE(2.0f);
                        break;
                    case Limit::Right:
                        prevLayerDelta[q] += grad_middle(inputs[q], deltas[q]);
                        break;
                        // default: Do nothing
                }
            }
        }
    }
}

template class HSwishActivationCPU<MemoryManager>;
template class HSwishActivationCPU<MemoryManagerFP16>;

} // namespace raul
