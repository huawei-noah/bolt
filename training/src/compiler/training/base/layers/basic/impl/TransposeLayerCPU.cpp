// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "TransposeLayerCPU.h"
#include "../TransposeLayer.h"

#include <training/base/impl/ImplFactory.h>

using namespace raul;

namespace
{

template<typename T>
void CopyWithSwappedIndices(const T& from, T& to, size_t ind1, size_t ind2, bool overwrite = true)
{
    if (overwrite)
    {
        std::fill(to.begin(), to.end(), static_cast<typename T::type>(0_dt));
    }
    if (ind1 == ind2)
    {
        std::transform(from.begin(), from.end(), to.begin(), to.begin(), std::plus<typename T::type>());
    }
    else
    {
        auto inData4D = from.get4DView();
        auto outData4D = to.get4DView();

        size_t N = from.getBatchSize();
        size_t C = from.getDepth();
        size_t H = from.getHeight();
        size_t W = from.getWidth();

        size_t inputIndices[4] = { 0 };
        size_t* outputIndices[4] = { &inputIndices[0], &inputIndices[1], &inputIndices[2], &inputIndices[3] };

        outputIndices[ind1] = &inputIndices[ind2];
        outputIndices[ind2] = &inputIndices[ind1];

        for (inputIndices[0] = 0; inputIndices[0] < N; ++inputIndices[0])
        {
            for (inputIndices[1] = 0; inputIndices[1] < C; ++inputIndices[1])
            {
                for (inputIndices[2] = 0; inputIndices[2] < H; ++inputIndices[2])
                {
                    for (inputIndices[3] = 0; inputIndices[3] < W; ++inputIndices[3])
                    {
                        outData4D[*outputIndices[0]][*outputIndices[1]][*outputIndices[2]][*outputIndices[3]] += inData4D[inputIndices[0]][inputIndices[1]][inputIndices[2]][inputIndices[3]];
                    }
                }
            }
        }
    }
}

bool reg1 = raul::TheImplFactory::Instance().regCPUFP32<raul::TransposeLayer, raul::TransposeLayerCPU<raul::MemoryManager>>();
bool reg2 = raul::TheImplFactory::Instance().regCPUFP16<raul::TransposeLayer, raul::TransposeLayerCPU<raul::MemoryManagerFP16>>();

} // anonymous namespace

namespace raul
{

template<typename MM>
TransposeLayerCPU<MM>::TransposeLayerCPU(TransposeLayer& layer)
    : mLayer(layer)
{
}

template<typename MM>
void TransposeLayerCPU<MM>::forwardComputeImpl(NetworkMode)
{
    auto& work = mLayer.mNetworkParams.mWorkflow;

    auto& output = work.getMemoryManager<MM>()[mLayer.mOutputName];
    const auto& inputs = work.getMemoryManager<MM>()[mLayer.mInputName];
    CopyWithSwappedIndices(inputs, output, mLayer.mDim1, mLayer.mDim2);
}

template<typename MM>
void TransposeLayerCPU<MM>::backwardComputeImpl()
{
    auto& work = mLayer.mNetworkParams.mWorkflow;

    // if (mLayer.mNetworkParams.isGradNeeded(mLayer.mInputName))
    {
        const auto& deltas = work.getMemoryManager<MM>()[mLayer.mOutputName.grad()];
        auto& prevLayerDelta = work.getMemoryManager<MM>()[mLayer.mInputName.grad()];
        CopyWithSwappedIndices(deltas, prevLayerDelta, mLayer.mDim1, mLayer.mDim2, false);
    }
}

template class TransposeLayerCPU<MemoryManager>;
template class TransposeLayerCPU<MemoryManagerFP16>;

} // namespace raul