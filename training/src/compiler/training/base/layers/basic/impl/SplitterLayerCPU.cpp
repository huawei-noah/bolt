// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "SplitterLayerCPU.h"
#include "../SplitterLayer.h"

namespace raul
{

template<typename MM>
SplitterLayerCPU<MM>::SplitterLayerCPU(SplitterLayer& layer)
    : mLayer(layer)
{
}

template<typename MM>
void SplitterLayerCPU<MM>::forwardComputeImpl(NetworkMode)
{
    auto& work = mLayer.mNetworkParams.mWorkflow;

    const auto& inputBlob = work.getMemoryManager<MM>()[mLayer.mInputs[0]];

    for (auto& output : mLayer.mOutputs)
    {
        auto& out = work.getMemoryManager<MM>()[output];
        out = TORANGE_MM(inputBlob);
    }
}

template<typename MM>
void SplitterLayerCPU<MM>::backwardComputeImpl()
{
    auto& work = mLayer.mNetworkParams.mWorkflow;

    // if (mLayer.mNetworkParams.isGradNeeded(mLayer.mInputs[0]))
    {
        auto& prevLayerDelta = work.getMemoryManager<MM>()[mLayer.mInputs[0].grad()];

        for (size_t q = 0; q < mLayer.mOutputs.size(); ++q)
        {
            if (work.getMemoryManager<MM>().tensorExists(mLayer.mOutputs[q].grad()))
            {
                const auto& deltas = work.getMemoryManager<MM>()[mLayer.mOutputs[q].grad()];

                if (deltas.size() != prevLayerDelta.size())
                {
                    throw std::runtime_error(mLayer.mTypeName + "[" + mLayer.mName + "::backwardCompute]: gradient size mismatch");
                }

                std::transform(deltas.begin(), deltas.end(), prevLayerDelta.begin(), prevLayerDelta.begin(), std::plus<typename MM::type>());
            }
        }
    }
}

template class SplitterLayerCPU<MemoryManager>;
template class SplitterLayerCPU<MemoryManagerFP16>;

} // namespace raul