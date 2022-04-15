// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "ConcatenationLayerCPU.h"
#include "../ConcatenationLayer.h"

#include <training/base/impl/ImplFactory.h>
namespace
{
bool reg1 = raul::TheImplFactory::Instance().regCPUFP32<raul::ConcatenationLayer, raul::ConcatenationLayerCPU<raul::MemoryManager>>();
bool reg2 = raul::TheImplFactory::Instance().regCPUFP16<raul::ConcatenationLayer, raul::ConcatenationLayerCPU<raul::MemoryManagerFP16>>();
} // anonymous namespace

namespace raul
{

template<typename MM>
ConcatenationLayerCPU<MM>::ConcatenationLayerCPU(ConcatenationLayer& layer)
    : mLayer(layer)
{
}

template<typename MM>
void ConcatenationLayerCPU<MM>::forwardComputeImpl(NetworkMode)
{
    auto& work = mLayer.mNetworkParams.mWorkflow;

    auto& output = work.getMemoryManager<MM>()[mLayer.mOutputs[0]];

    auto output4d = output.get4DView();

    yato::dimensionality<3U, size_t> concatDims(output.getDepth(), output.getHeight(), output.getWidth());

    size_t accumulatedSize = 0;
    for (size_t q = 0; q < mLayer.mInputs.size(); ++q)
    {
        const auto& input = work.getMemoryManager<MM>()[mLayer.mInputs[q]];

        auto inputView = input.get4DView();

        typename MM::type* startEl = nullptr;
        switch (mLayer.mDirection)
        {
            case Dimension::Depth:
                startEl = &output4d[0][accumulatedSize][0][0];
                break;
            case Dimension::Height:
                startEl = &output4d[0][0][accumulatedSize][0];
                break;
            case Dimension::Width:
                startEl = &output4d[0][0][0][accumulatedSize];
                break;
            default:
                throw std::runtime_error(mLayer.mTypeName + "[" + mLayer.mName + "forwardCompute]: unknown dim");
        }
        accumulatedSize += input.getShape()[mLayer.mDimIndex + 1];
        auto sliceView = yato::array_view_4d<typename MM::type>(startEl, input.getShape(), concatDims);
        Common::copyView(inputView, sliceView, true);
    }
}

template<typename MM>
void ConcatenationLayerCPU<MM>::backwardComputeImpl()
{
    auto& work = mLayer.mNetworkParams.mWorkflow;

    const auto& delta = work.getMemoryManager<MM>()[mLayer.mOutputs[0].grad()];
    auto delta4d = delta.get4DView();

    yato::dimensionality<3U, size_t> concatDims(delta.getDepth(), delta.getHeight(), delta.getWidth());

    size_t accumulatedSize = 0;
    for (size_t q = 0; q < mLayer.mInputs.size(); ++q)
    {
        // if (mNetworkParams.isGradNeeded(mInputs[q]))
        {
            auto& prevLayerDelta = work.getMemoryManager<MM>()[mLayer.mInputs[q].grad()];

            auto prevDelta4d = prevLayerDelta.get4DView();

            const typename MM::type* startEl = nullptr;
            switch (mLayer.mDirection)
            {
                case Dimension::Depth:
                    startEl = &delta4d[0][accumulatedSize][0][0];
                    break;
                case Dimension::Height:
                    startEl = &delta4d[0][0][accumulatedSize][0];
                    break;
                case Dimension::Width:
                    startEl = &delta4d[0][0][0][accumulatedSize];
                    break;
                default:
                    throw std::runtime_error(mLayer.mTypeName + "[" + mLayer.mName + "backwardCompute]: unknown dim");
            }
            auto sliceView = yato::array_view_4d<const typename MM::type>(startEl, prevLayerDelta.getShape(), concatDims);

            Common::copyView(sliceView, prevDelta4d);
        }
        accumulatedSize += work.getMemoryManager<MM>()[mLayer.mInputs[q]].getShape()[mLayer.mDimIndex + 1];
    }
}

template class ConcatenationLayerCPU<MemoryManager>;
template class ConcatenationLayerCPU<MemoryManagerFP16>;

} // namespace raul
