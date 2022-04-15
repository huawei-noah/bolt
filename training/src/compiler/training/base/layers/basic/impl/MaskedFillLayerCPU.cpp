// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "MaskedFillLayerCPU.h"
#include "../MaskedFillLayer.h"

namespace raul
{
template<typename MM>
void MaskedFillLayerCPU<MM>::forwardComputeImpl(NetworkMode)
{
    Workflow& work = mLayer.mNetworkParams.mWorkflow;

    auto& output = work.getMemoryManager<MM>()[mLayer.mOutputName];

    const auto& mask = work.getMemoryManager<MM>()[mLayer.mMaskName];
    const auto& inputs = work.getMemoryManager<MM>()[mLayer.mInputName];

    size_t size = inputs.size();

    std::copy(inputs.begin(), inputs.end(), output.begin());

    if (mask.getShape() != output.getShape())
    {
        const auto maskViewer = mask.getBroadcastedViewer(inputs.getShape());
#if defined(_OPENMP)
#pragma omp parallel for
#endif
        for (size_t q = 0; q < size; ++q)
        {
            if ((maskViewer[q] == TOMMTYPE(0.0_dt)) == mLayer.mInverted)
            {
                output[q] = TOMMTYPE(mLayer.mFillValue);
            }
        }
    }
    else
    {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
        for (size_t q = 0; q < size; ++q)
        {
            if ((mask[q] == TOMMTYPE(0.0_dt)) == mLayer.mInverted)
            {
                output[q] = TOMMTYPE(mLayer.mFillValue);
            }
        }
    }
}

template<typename MM>
void MaskedFillLayerCPU<MM>::backwardComputeImpl()
{
    Workflow& work = mLayer.mNetworkParams.mWorkflow;

    // if (mNetworkParams.isGradNeeded(mLayer.mInputName))
    {
        auto& prevLayerDelta = work.getMemoryManager<MM>()[mLayer.mInputName.grad()];
        const auto& deltas = work.getMemoryManager<MM>()[mLayer.mOutputName.grad()];

        const auto& mask = work.getMemoryManager<MM>()[mLayer.mMaskName];

        size_t size = deltas.size();

        if (mask.getShape() != deltas.getShape())
        {
            const auto maskViewer = mask.getBroadcastedViewer(deltas.getShape());
#if defined(_OPENMP)
#pragma omp parallel for
#endif
            for (size_t q = 0; q < size; ++q)
            {
                if ((maskViewer[q] == 0.0_dt) != mLayer.mInverted)
                {
                    prevLayerDelta[q] += deltas[q];
                }
            }
        }
        else
        {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
            for (size_t q = 0; q < size; ++q)
            {
                if ((mask[q] == 0.0_dt) != mLayer.mInverted)
                {
                    prevLayerDelta[q] += deltas[q];
                }
            }
        }
    }
}

template class MaskedFillLayerCPU<MemoryManager>;
template class MaskedFillLayerCPU<MemoryManagerFP16>;

} // namespace raul