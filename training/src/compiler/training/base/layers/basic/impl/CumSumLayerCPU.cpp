// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "CumSumLayerCPU.h"
#include "../CumSumLayer.h"

#include <training/base/impl/ImplFactory.h>
namespace
{
bool reg1 = raul::TheImplFactory::Instance().regCPUFP32<raul::CumSumLayer, raul::CumSumLayerCPU<raul::MemoryManager>>();
bool reg2 = raul::TheImplFactory::Instance().regCPUFP16<raul::CumSumLayer, raul::CumSumLayerCPU<raul::MemoryManagerFP16>>();
} // anonymous namespace

namespace raul
{

template<typename MM>
CumSumLayerCPU<MM>::CumSumLayerCPU(CumSumLayer& layer)
    : mLayer(layer)
{
}

template<typename MM>
void CumSumLayerCPU<MM>::forwardComputeImpl(NetworkMode)
{
    auto& work = mLayer.mNetworkParams.mWorkflow;

    auto& output = work.getMemoryManager<MM>()[mLayer.mOutputs[0]];
    const auto& input = work.getMemoryManager<MM>()[mLayer.mInputs[0]];

    const auto outputShape = output.getShape();
    const auto outputStrides = Common::getStrides(outputShape);

    for (size_t q = 0; q < output.size(); ++q)
    {
        auto indexes = Common::offsetToIndexes(q, outputStrides);
        if (indexes[static_cast<size_t>(mLayer.mDimension)] == 0)
        {
            output[q] = input[q];
        }
        else
        {
            indexes[static_cast<size_t>(mLayer.mDimension)] -= 1;
            output[q] = input[q] + output[Common::indexesToOffset(indexes, outputStrides)];
        }
    }
}

template<typename MM>
void CumSumLayerCPU<MM>::backwardComputeImpl()
{
    auto& work = mLayer.mNetworkParams.mWorkflow;

    const auto& deltas = work.getMemoryManager<MM>()[mLayer.mOutputs[0].grad()];

    // if (mLayer.mNetworkParams.isGradNeeded(mLayer.mInputs[0]))
    {
        auto& prevLayerDelta = work.getMemoryManager<MM>()[mLayer.mInputs[0].grad()];

        const auto outputShape = deltas.getShape();
        const auto outputStrides = Common::getStrides(outputShape);
        for (size_t q = prevLayerDelta.size(); q > 0; --q)
        {
            auto indexes = Common::offsetToIndexes(q - 1, outputStrides);
            if (indexes[static_cast<size_t>(mLayer.mDimension)] == outputShape[static_cast<size_t>(mLayer.mDimension)] - 1)
            {
                prevLayerDelta[q - 1] += deltas[q - 1];
            }
            else
            {
                indexes[static_cast<size_t>(mLayer.mDimension)] += 1;
                prevLayerDelta[q - 1] += deltas[q - 1] + prevLayerDelta[Common::indexesToOffset(indexes, outputStrides)];
            }
        }
    }
}

template class CumSumLayerCPU<MemoryManager>;
template class CumSumLayerCPU<MemoryManagerFP16>;

} // namespace raul