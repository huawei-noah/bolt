// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "ReverseLayerCPU.h"
#include "../ReverseLayer.h"

#include <algorithm>

#include <training/base/impl/ImplFactory.h>
namespace
{
bool reg1 = raul::TheImplFactory::Instance().regCPUFP32<raul::ReverseLayer, raul::ReverseLayerCPU<raul::MemoryManager>>();
bool reg2 = raul::TheImplFactory::Instance().regCPUFP16<raul::ReverseLayer, raul::ReverseLayerCPU<raul::MemoryManagerFP16>>();
} // anonymous namespace

namespace raul
{

template<typename MM>
void ReverseLayerCPU<MM>::forwardComputeImpl(NetworkMode)
{
    Workflow& work = mLayer.mNetworkParams.mWorkflow;

    const auto& input = work.getMemoryManager<MM>()[mLayer.mInputName];
    auto& output = work.getMemoryManager<MM>()[mLayer.mOutputName];

    const auto batch = work.getBatchSize();
    const auto depth = input.getDepth();
    const auto height = input.getHeight();
    const auto width = input.getWidth();
    auto input3D = input.reshape(yato::dims(batch, depth * height, width));
    auto output3D = output.reshape(yato::dims(batch, depth * height, width));

#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (size_t i = 0; i < batch; ++i)
    {
        auto realLength = mLayer.mReverseOnly ? depth * height : static_cast<size_t>(work.getMemoryManager<MM>()[mLayer.mInputs[1]][i]);
        for (size_t w = 0; w < width; ++w)
        {
            for (size_t start = 0; start < depth * height; ++start)
            {
                if (start < realLength)
                {
                    output3D[i][realLength - 1 - start][w] = input3D[i][start][w];
                }
                else
                {
                    output3D[i][start][w] = input3D[i][start][w];
                }
            }
        }
    }
}

template<typename MM>
void ReverseLayerCPU<MM>::backwardComputeImpl()
{
    Workflow& work = mLayer.mNetworkParams.mWorkflow;

    const auto& deltas = work.getMemoryManager<MM>()[mLayer.mOutputName.grad()];
    auto& prevLayerDelta = work.getMemoryManager<MM>()[mLayer.mInputName.grad()];

    const auto batch = work.getBatchSize();
    const auto depth = deltas.getDepth();
    const auto height = deltas.getHeight();
    const auto width = deltas.getWidth();
    auto prevLayerDelta3D = prevLayerDelta.reshape(yato::dims(batch, depth * height, width));
    auto deltas3D = deltas.reshape(yato::dims(batch, depth * height, width));

#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (size_t i = 0; i < batch; ++i)
    {
        auto realLength = mLayer.mReverseOnly ? depth * height : static_cast<size_t>(work.getMemoryManager<MM>()[mLayer.mInputs[1]][i]);
        for (size_t w = 0; w < width; ++w)
        {
            for (size_t start = 0; start < depth * height; ++start)
            {
                if (start < realLength)
                {
                    prevLayerDelta3D[i][realLength - 1 - start][w] += deltas3D[i][start][w];
                }
                else
                {
                    prevLayerDelta3D[i][start][w] += deltas3D[i][start][w];
                }
            }
        }
    }
}

template class ReverseLayerCPU<MemoryManager>;
template class ReverseLayerCPU<MemoryManagerFP16>;

} // namespace raul