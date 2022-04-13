// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "ElementWiseSubLayerCPU.h"
#include "../ElementWiseSubLayer.h"

namespace raul
{

template<typename MM>
ElementWiseSubLayerCPU<MM>::ElementWiseSubLayerCPU(ElementWiseSubLayer& layer)
    : mLayer(layer)
{
}

template<typename MM>
void ElementWiseSubLayerCPU<MM>::forwardComputeImpl(NetworkMode)
{
    mLayer.determineBroadcastFlags();

    if (!mLayer.mBroadcast && (mLayer.mBroadcastQuery[0] || mLayer.mBroadcastQuery[1]))
    {
        THROW(mLayer.mTypeName, mLayer.mName, "input size mismatch");
    }

    auto& work = mLayer.mNetworkParams.mWorkflow;

    auto& output = work.getMemoryManager<MM>()[mLayer.mOutputs[0]];

    const auto& minuend = work.getMemoryManager<MM>()[mLayer.mInputs[0]];
    const auto& subtrahend = work.getMemoryManager<MM>()[mLayer.mInputs[1]];

    if (!mLayer.mBroadcastQuery[0] && !mLayer.mBroadcastQuery[1])
    {
        output = TORANGE_MM(minuend);
        output -= subtrahend;
    }
    else
    {
        std::fill(output.begin(), output.end(), TOMMTYPE(0.0_dt));
        for (size_t i = 0; i < mLayer.mInputs.size(); i++)
        {
            auto input = (i == 0 ? minuend : subtrahend);
            if (i == 1)
            {
                input *= TOMMTYPE(-1.0_dt);
            }
            if (!mLayer.mBroadcastQuery[i])
            {
                output += input;
            }
            else
            {
                auto input_viewer = input.getBroadcastedViewer(output.getShape());
#if defined(_OPENMP)
#pragma omp parallel for
#endif
                for (size_t j = 0; j < output.size(); ++j)
                {
                    output[j] += input_viewer[j];
                }
            }
        }
    }
}

template<typename MM>
void ElementWiseSubLayerCPU<MM>::backwardComputeImpl()
{
    auto& work = mLayer.mNetworkParams.mWorkflow;

    const auto& deltas = work.getMemoryManager<MM>()[mLayer.mOutputs[0].grad()];

    // if (mLayer.mNetworkParams.isGradNeeded(mLayer.mInputs[0]))
    {
        auto& minuend_nabla_tensor = work.getMemoryManager<MM>()[mLayer.mInputs[0].grad()];
        if (!mLayer.mBroadcastQuery[0])
        {
            work.getMemoryManager<MM>()[mLayer.mInputs[0].grad()] += deltas;
        }
        else
        {
            auto minuend_nabla = minuend_nabla_tensor.getBroadcastedViewer(deltas.getShape());
            for (size_t q = 0; q < minuend_nabla.size(); ++q)
            {
                minuend_nabla[q] += deltas[q];
            }
        }
    }

    // if (mLayer.mNetworkParams.isGradNeeded(mLayer.mInputs[1]))
    {
        auto& subtrahend_nabla_tensor = work.getMemoryManager<MM>()[mLayer.mInputs[1].grad()];
        if (!mLayer.mBroadcastQuery[1])
        {
            subtrahend_nabla_tensor -= deltas;
        }
        else
        {
            auto subtrahend_nabla = subtrahend_nabla_tensor.getBroadcastedViewer(deltas.getShape());
            for (size_t q = 0; q < subtrahend_nabla.size(); ++q)
            {
                subtrahend_nabla[q] -= deltas[q];
            }
        }
    }
}

template class ElementWiseSubLayerCPU<MemoryManager>;
template class ElementWiseSubLayerCPU<MemoryManagerFP16>;

} // namespace raul