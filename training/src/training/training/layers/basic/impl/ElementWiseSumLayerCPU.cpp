// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "ElementWiseSumLayerCPU.h"
#include "../ElementWiseSumLayer.h"

namespace raul
{

template<typename MM>
ElementWiseSumLayerCPU<MM>::ElementWiseSumLayerCPU(ElementWiseSumLayer& layer)
    : mLayer(layer)
{
}

template<typename MM>
void ElementWiseSumLayerCPU<MM>::forwardComputeImpl(NetworkMode)
{
    mLayer.determineBroadcastFlags();

    if (!mLayer.mBroadcast && std::any_of(mLayer.mBroadcastQuery.begin(), mLayer.mBroadcastQuery.end(), [](const auto& needToBroadcast) { return needToBroadcast; }))
    {
        THROW(mLayer.mTypeName, mLayer.mName, "input size mismatch");
    }

    auto& work = mLayer.mNetworkParams.mWorkflow;

    auto& output = work.getMemoryManager<MM>()[mLayer.mOutputs[0]];

    size_t exactFit = mLayer.mInputs.size();
    for (size_t q = 0; q < mLayer.mInputs.size(); ++q)
    {
        if (!mLayer.mBroadcastQuery[q])
        {
            exactFit = q;
            break;
        }
    }
    if (exactFit < mLayer.mInputs.size())
    {
        output = TORANGE_MM(work.getMemoryManager<MM>()[mLayer.mInputs[exactFit]]);
    }
    else
    {
        std::fill(output.begin(), output.end(), TOMMTYPE(0.0_dt));
    }

    for (size_t q = 0; q < mLayer.mInputs.size(); ++q)
    {
        if (q == exactFit)
        {
            continue;
        }
        const auto& input = work.getMemoryManager<MM>()[mLayer.mInputs[q]];
        if (!mLayer.mBroadcastQuery[q])
        {
            output += input;
        }
        else
        {
            auto input_viewer = input.getBroadcastedViewer(output.getShape());
#if defined(_OPENMP)
#pragma omp parallel for
#endif
            for (size_t i = 0; i < output.size(); ++i)
            {
                output[i] += input_viewer[i];
            }
        }
    }
}

template<typename MM>
void ElementWiseSumLayerCPU<MM>::backwardComputeImpl()
{
    auto& work = mLayer.mNetworkParams.mWorkflow;

    const auto& deltas = work.getMemoryManager<MM>()[mLayer.mOutputs[0].grad()];

    for (size_t i = 0; i < mLayer.mInputs.size(); ++i)
    {
        // if (mNetworkParams.isGradNeeded(input))
        {
            auto& in_nabla_tensor = work.getMemoryManager<MM>()[mLayer.mInputs[i].grad()];

            if (!mLayer.mBroadcastQuery[i])
            {
                in_nabla_tensor += TORANGE_MM(deltas);
            }
            else
            {
                auto in_nabla = in_nabla_tensor.getBroadcastedViewer(deltas.getShape());
                for (size_t q = 0; q < in_nabla.size(); ++q)
                {
                    in_nabla[q] += deltas[q];
                }
            }
        }
    }
}

template class ElementWiseSumLayerCPU<MemoryManager>;
template class ElementWiseSumLayerCPU<MemoryManagerFP16>;

} // namespace raul