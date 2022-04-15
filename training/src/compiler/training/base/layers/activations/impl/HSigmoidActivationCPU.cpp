// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "HSigmoidActivationCPU.h"
#include "../HSigmoidActivation.h"

#include <algorithm>

namespace raul
{

template<typename MM>
void HSigmoidActivationCPU<MM>::forwardComputeImpl(NetworkMode)
{
    Workflow& work = mLayer.mNetworkParams.mWorkflow;

    auto& output = work.getMemoryManager<MM>()[mLayer.mOutputName];

    const auto& inputs = work.getMemoryManager<MM>()[mLayer.mInputName];

    std::transform(inputs.begin(), inputs.end(), output.begin(), [&](typename MM::type val) -> typename MM::type { return Common::HSigmoid(val); });
}

template<typename MM>
void HSigmoidActivationCPU<MM>::backwardComputeImpl()
{
    // if (mLayer.mNetworkParams.isGradNeeded(mLayer.mInputName))
    {
        Workflow& work = mLayer.mNetworkParams.mWorkflow;

        const auto& deltas = work.getMemoryManager<MM>()[mLayer.mOutputName.grad()];
        auto& prevLayerDelta = work.getMemoryManager<MM>()[mLayer.mInputName.grad()];

        const auto& inputs = work.getMemoryManager<MM>()[mLayer.mInputName];

        for (size_t q = 0; q < prevLayerDelta.size(); ++q)
        {
            if (TODTYPE(inputs[q]) > -3.0_dt && TODTYPE(inputs[q]) < 3.0_dt)
            {
                prevLayerDelta[q] += TOMMTYPE(TODTYPE(deltas[q]) / 6.0_dt);
            }
            else if (TODTYPE(inputs[q]) == 3.0_dt)
            {
                switch (mLayer.p3PointVal)
                {
                    case Limit::Left:
                        prevLayerDelta[q] += TOMMTYPE(TODTYPE(deltas[q]) / 6.0_dt);
                        break;
                    case Limit::Middle:
                        prevLayerDelta[q] += TOMMTYPE(TODTYPE(deltas[q]) / 12.0_dt);
                        break;
                    case Limit::Right:
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
                        prevLayerDelta[q] += TOMMTYPE(TODTYPE(deltas[q]) / 12.0_dt);
                        break;
                    case Limit::Right:
                        prevLayerDelta[q] += TOMMTYPE(TODTYPE(deltas[q]) / 6.0_dt);
                        break;
                        // default: Do nothing
                }
            }
        }
    }
}

template class HSigmoidActivationCPU<MemoryManager>;
template class HSigmoidActivationCPU<MemoryManagerFP16>;

} // namespace raul