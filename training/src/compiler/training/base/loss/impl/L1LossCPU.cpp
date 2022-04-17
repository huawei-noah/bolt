// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "L1LossCPU.h"
#include "../L1Loss.h"

#include <training/base/impl/ImplFactory.h>
namespace
{
bool reg1 = raul::TheImplFactory::Instance().regCPUFP32<raul::L1Loss, raul::L1LossLayerCPU<raul::MemoryManager>>();
bool reg2 = raul::TheImplFactory::Instance().regCPUFP16<raul::L1Loss, raul::L1LossLayerCPU<raul::MemoryManagerFP16>>();
} // anonymous namespace

namespace raul
{

template<typename MM>
void L1LossLayerCPU<MM>::forwardComputeImpl(NetworkMode mode)
{
    auto& work = mLayer.mNetworkParams.mWorkflow;

    if (mLayer.wrapper)
    {
        auto& output = work.getMemoryManager<MM>()[mLayer.mOutputName];
        const auto& input = work.getMemoryManager<MM>()[mLayer.mInputName];
        output = TORANGE_MM(input);
    }
    else
    {
        if (mode == NetworkMode::Test)
        {
            return;
        }

        auto& output = work.getMemoryManager<MM>()[mLayer.mOutputName];
        const auto& input = work.getMemoryManager<MM>()[mLayer.mInputName];

        const auto& targets = work.getMemoryManager<MM>()[mLayer.mLabelName];

#if defined(_OPENMP)
#pragma omp parallel for
#endif
        for (size_t q = 0; q < output.size(); ++q)
        {
            output[q] = TOMMTYPE(std::abs(TODTYPE(input[q] - targets[q])));
        }
    }
}

template<typename MM>
void L1LossLayerCPU<MM>::backwardComputeImpl()
{
    auto& work = mLayer.mNetworkParams.mWorkflow;

    if (!mLayer.wrapper)
    {
        auto& prevLayerDelta = work.getMemoryManager<MM>()[mLayer.mInputName.grad()];

        const auto& targets = work.getMemoryManager<MM>()[mLayer.mLabelName];
        const auto& inputs = work.getMemoryManager<MM>()[mLayer.mInputName];

        const auto& deltas = work.getMemoryManager<MM>()[mLayer.mOutputName.grad()];
        if (deltas.getShape() != prevLayerDelta.getShape())
        {
            if (!deltas.isBroadcastableTo(prevLayerDelta.getShape()))
            {
                THROW(mLayer.mTypeName, mLayer.mName, "bad incoming deltas shape");
            }
            auto deltas_viewer = deltas.getBroadcastedViewer(prevLayerDelta.getShape());
#if defined(_OPENMP)
#pragma omp parallel for
#endif
            for (size_t q = 0; q < prevLayerDelta.size(); ++q)
            {
                dtype val = (inputs[q] == targets[q] ? 0_dt : (inputs[q] > targets[q] ? 1_dt : -1_dt));
                prevLayerDelta[q] += TOMMTYPE(val) * deltas_viewer[q];
            }
        }
        else
        {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
            for (size_t q = 0; q < prevLayerDelta.size(); ++q)
            {
                dtype val = (inputs[q] == targets[q] ? 0_dt : (inputs[q] > targets[q] ? 1_dt : -1_dt));
                prevLayerDelta[q] += TOMMTYPE(val) * deltas[q];
            }
        }
    }
    else if (!mLayer.mIsFinal)
    {
        const auto& deltas = work.getMemoryManager<MM>()[mLayer.mOutputName.grad()];
        auto& prevLayerDelta = work.getMemoryManager<MM>()[mLayer.mInputName.grad()];

        prevLayerDelta += deltas;
    }
}

template class L1LossLayerCPU<MemoryManager>;
template class L1LossLayerCPU<MemoryManagerFP16>;

} // namespace raul
