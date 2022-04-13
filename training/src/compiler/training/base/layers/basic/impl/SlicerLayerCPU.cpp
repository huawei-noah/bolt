// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "SlicerLayerCPU.h"
#include "../SlicerLayer.h"

#include <training/base/impl/ImplFactory.h>
namespace
{
bool reg1 = raul::TheImplFactory::Instance().regCPUFP32<raul::SlicerLayer, raul::SlicerLayerCPU<raul::MemoryManager>>();
bool reg2 = raul::TheImplFactory::Instance().regCPUFP16<raul::SlicerLayer, raul::SlicerLayerCPU<raul::MemoryManagerFP16>>();
} // anonymous namespace

namespace raul
{

template<typename MM>
SlicerLayerCPU<MM>::SlicerLayerCPU(SlicerLayer& layer)
    : mLayer(layer)
{
}

template<typename MM>
void SlicerLayerCPU<MM>::forwardComputeImpl(NetworkMode)
{
    auto& work = mLayer.mNetworkParams.mWorkflow;

    const auto& input = work.getMemoryManager<MM>()[mLayer.mInputs[0]];

    auto input4d = input.get4DView();
    auto inputDims = yato::dims(input.getDepth(), input.getHeight(), input.getWidth());

    size_t accumulatedSize = 0;
    for (size_t q = 0; q < mLayer.mSlices.size(); ++q)
    {
        auto& output = work.getMemoryManager<MM>()[mLayer.mOutputs[q]];
        auto outputDims = output.getShape();
        const typename MM::type* startEl = nullptr;
        switch (mLayer.mDirection)
        {
            case Dimension::Depth:
                startEl = &input4d[0][accumulatedSize][0][0];
                break;
            case Dimension::Height:
                startEl = &input4d[0][0][accumulatedSize][0];
                break;
            case Dimension::Width:
                startEl = &input4d[0][0][0][accumulatedSize];
                break;
            default:
                throw std::runtime_error("SlicerLayer[forwardCompute]: unknown dim");
        }
        accumulatedSize += mLayer.mSlices[q];
        auto sliceView = yato::array_view_4d<const typename MM::type>(startEl, outputDims, inputDims);
        auto outputView = output.get4DView();
        Common::copyView(sliceView, outputView, true);
    }
}

template<typename MM>
void SlicerLayerCPU<MM>::backwardComputeImpl()
{
    auto& work = mLayer.mNetworkParams.mWorkflow;

    // if (mLayer.mNetworkParams.isGradNeeded(mLayer.mInputs[0]))
    {
        auto& prevLayerDelta = work.getMemoryManager<MM>()[mLayer.mInputs[0].grad()];

        auto prevDelta4d = prevLayerDelta.get4DView();
        auto inputDims = yato::dims(prevLayerDelta.getDepth(), prevLayerDelta.getHeight(), prevLayerDelta.getWidth());
        size_t accumulatedSize = 0;
        for (size_t q = 0; q < mLayer.mOutputs.size(); ++q)
        {
            if (work.getMemoryManager<MM>().tensorExists(mLayer.mOutputs[q].grad()))
            {
                typename MM::type* startEl = nullptr;
                switch (mLayer.mDirection)
                {
                    case Dimension::Depth:
                        startEl = &prevDelta4d[0][accumulatedSize][0][0];
                        break;
                    case Dimension::Height:
                        startEl = &prevDelta4d[0][0][accumulatedSize][0];
                        break;
                    case Dimension::Width:
                        startEl = &prevDelta4d[0][0][0][accumulatedSize];
                        break;
                    default:
                        throw std::runtime_error("SlicerLayer[backwardCompute]: unknown dim");
                }

                const auto& delta = work.getMemoryManager<MM>()[mLayer.mOutputs[q].grad()];
                auto outputDims = delta.getShape();
                auto deltaView = delta.get4DView();
                auto sliceView = yato::array_view_4d<typename MM::type>(startEl, outputDims, inputDims);
                Common::copyView(deltaView, sliceView);
            }
            accumulatedSize += mLayer.mSlices[q];
        }
    }
}

template class SlicerLayerCPU<MemoryManager>;
template class SlicerLayerCPU<MemoryManagerFP16>;

} // namespace raul
