// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "DropoutLayerCPU.h"
#include "../DropoutLayer.h"

namespace raul
{

template<typename MM>
DropoutLayerCPU<MM>::DropoutLayerCPU(DropoutLayer& layer)
    : mLayer(layer)
{
}

template<typename MM>
void DropoutLayerCPU<MM>::forwardComputeImpl(NetworkMode mode)
{
    auto& work = mLayer.mNetworkParams.mWorkflow;

    auto& output = work.getMemoryManager<MM>()[mLayer.mOutputName];
    auto& mRandom = work.getMemoryManager<MM>()[mLayer.mName / "random"];
    const auto& inputs = work.getMemoryManager<MM>()[mLayer.mInputName];

    if (mode == NetworkMode::Train || mode == NetworkMode::TrainCheckpointed)
    {
        if (mode == NetworkMode::Train)
        {
            if (mLayer.mNetworkParams.mCalculationMode == CalculationMode::DETERMINISTIC)
            {
                for (size_t i = 0; i < output.size(); ++i)
                {
                    mRandom[i] = TOMMTYPE(random::bernoulli::randBool(mLayer.mProbability, mLayer.mState));
                }
            }
            else
            {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
                for (size_t i = 0; i < output.size(); ++i)
                {
                    mRandom[i] = TOMMTYPE(random::bernoulli::randBool(mLayer.mProbability));
                }
            }
        }

#if defined(_OPENMP)
#pragma omp parallel for
#endif
        for (size_t i = 0; i < output.size(); ++i)
        {
            if (mRandom[i] == TOMMTYPE(1.0f))
            {
                output[i] = TOMMTYPE(0.0_hf);
            }
            else
            {
                output[i] = TOMMTYPE(TODTYPE(inputs[i]) * mLayer.mScale);
            }
        }
    }
    else
    {
        std::copy(inputs.begin(), inputs.end(), output.begin());
    }
}

template<typename MM>
void DropoutLayerCPU<MM>::backwardComputeImpl()
{
    auto& work = mLayer.mNetworkParams.mWorkflow;

    // if (mLayer.mNetworkParams.isGradNeeded(mLayer.mInputName) && mNetworkParams.mMemoryManager.tensorExists(mLayer.mOutputName.grad()))
    {
        auto& prevLayerDelta = work.getMemoryManager<MM>()[mLayer.mInputName.grad()];
        const auto& mRandom = work.getMemoryManager<MM>()[mLayer.mName / "random"];
        const auto& deltas = work.getMemoryManager<MM>()[mLayer.mOutputName.grad()];

        if (!deltas.empty())
        {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
            for (size_t i = 0; i < deltas.size(); ++i)
            {
                if (mRandom[i] == TOMMTYPE(0.0f))
                {
                    prevLayerDelta[i] += deltas[i] * TOMMTYPE(mLayer.mScale);
                }
            }
        }
    }
}

template class DropoutLayerCPU<MemoryManager>;
template class DropoutLayerCPU<MemoryManagerFP16>;

} // namespace raul
