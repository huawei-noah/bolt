// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "RandomSelectLayerCPU.h"
#include "../RandomSelectLayer.h"

#include <training/base/common/Random.h>

#include <training/base/impl/ImplFactory.h>
namespace
{
bool reg1 = raul::TheImplFactory::Instance().regCPUFP32<raul::RandomSelectLayer, raul::RandomSelectLayerCPU<raul::MemoryManager>>();
bool reg2 = raul::TheImplFactory::Instance().regCPUFP16<raul::RandomSelectLayer, raul::RandomSelectLayerCPU<raul::MemoryManagerFP16>>();
} // anonymous namespace

namespace raul
{

template<typename MM>
RandomSelectLayerCPU<MM>::RandomSelectLayerCPU(RandomSelectLayer& layer)
    : mLayer(layer)
{
}

template<typename MM>
void RandomSelectLayerCPU<MM>::forwardComputeImpl(NetworkMode)
{
    mLayer.determineBroadcastFlags();

    if (!mLayer.mBroadcast && (mLayer.mBroadcastQuery[0] || mLayer.mBroadcastQuery[1]))
    {
        THROW("RandomSelectLayer", mLayer.mName, "input size mismatch");
    }

    auto& work = mLayer.mNetworkParams.mWorkflow;

    auto& memoryManager = work.getMemoryManager<MM>();
    auto& mRandomCPU = memoryManager[mLayer.mRandomName];

    auto& output = memoryManager[mLayer.mOutputs[0]];

    const auto& x = memoryManager[mLayer.mInputs[0]];
    const auto& y = memoryManager[mLayer.mInputs[1]];

    if (mLayer.mBroadcastQuery[0] && mLayer.mBroadcastQuery[1])
    {
        const auto x_viewer = x.getBroadcastedViewer(output.getShape());
        const auto y_viewer = y.getBroadcastedViewer(output.getShape());

        if (mLayer.mNetworkParams.mCalculationMode == CalculationMode::DETERMINISTIC)
        {
            for (size_t q = 0; q < output.size(); ++q)
            {
                mRandomCPU[q] = TOMMTYPE(random::bernoulli::randBool(mLayer.mProbability) ? 1_dt : 0_dt);
                output[q] = TOMMTYPE(x_viewer[q] * mRandomCPU[q] + (1_dt - mRandomCPU[q]) * y_viewer[q]);
            }
        }
        else
        {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
            for (size_t q = 0; q < output.size(); ++q)
            {
                mRandomCPU[q] = TOMMTYPE(random::bernoulli::randBool(mLayer.mProbability) ? 1_dt : 0_dt);
                output[q] = TOMMTYPE(x_viewer[q] * mRandomCPU[q] + (1_dt - mRandomCPU[q]) * y_viewer[q]);
            }
        }
    }
    else if (mLayer.mBroadcastQuery[0])
    {
        const auto x_viewer = x.getBroadcastedViewer(output.getShape());

        if (mLayer.mNetworkParams.mCalculationMode == CalculationMode::DETERMINISTIC)
        {
            for (size_t q = 0; q < output.size(); ++q)
            {
                mRandomCPU[q] = TOMMTYPE(random::bernoulli::randBool(mLayer.mProbability) ? 1_dt : 0_dt);
                output[q] = TOMMTYPE(x_viewer[q] * mRandomCPU[q] + (1_dt - mRandomCPU[q]) * y[q]);
            }
        }
        else
        {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
            for (size_t q = 0; q < output.size(); ++q)
            {
                mRandomCPU[q] = TOMMTYPE(random::bernoulli::randBool(mLayer.mProbability) ? 1_dt : 0_dt);
                output[q] = TOMMTYPE(x_viewer[q] * mRandomCPU[q] + (1_dt - mRandomCPU[q]) * y[q]);
            }
        }
    }
    else if (mLayer.mBroadcastQuery[1])
    {
        const auto y_viewer = y.getBroadcastedViewer(output.getShape());

        if (mLayer.mNetworkParams.mCalculationMode == CalculationMode::DETERMINISTIC)
        {
            for (size_t q = 0; q < output.size(); ++q)
            {
                mRandomCPU[q] = TOMMTYPE(random::bernoulli::randBool(mLayer.mProbability) ? 1_dt : 0_dt);
                output[q] = TOMMTYPE(x[q] * mRandomCPU[q] + (1_dt - mRandomCPU[q]) * y_viewer[q]);
            }
        }
        else
        {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
            for (size_t q = 0; q < output.size(); ++q)
            {
                mRandomCPU[q] = TOMMTYPE(random::bernoulli::randBool(mLayer.mProbability) ? 1_dt : 0_dt);
                output[q] = TOMMTYPE(x[q] * mRandomCPU[q] + (1_dt - mRandomCPU[q]) * y_viewer[q]);
            }
        }
    }
    else
    {
        if (mLayer.mNetworkParams.mCalculationMode == CalculationMode::DETERMINISTIC)
        {
            for (size_t q = 0; q < output.size(); ++q)
            {
                mRandomCPU[q] = TOMMTYPE(random::bernoulli::randBool(mLayer.mProbability) ? 1_dt : 0_dt);
                output[q] = TOMMTYPE(x[q] * mRandomCPU[q] + (1_dt - mRandomCPU[q]) * y[q]);
            }
        }
        else
        {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
            for (size_t q = 0; q < output.size(); ++q)
            {
                mRandomCPU[q] = TOMMTYPE(random::bernoulli::randBool(mLayer.mProbability) ? 1_dt : 0_dt);
                output[q] = TOMMTYPE(x[q] * mRandomCPU[q] + (1_dt - mRandomCPU[q]) * y[q]);
            }
        }
    }
}

template<typename MM>
void RandomSelectLayerCPU<MM>::backwardComputeImpl()
{
    auto& work = mLayer.mNetworkParams.mWorkflow;

    auto& memoryManager = work.getMemoryManager<MM>();
    auto& mRandomCPU = memoryManager[mLayer.mRandomName];
    const auto& delta = memoryManager[mLayer.mOutputs[0].grad()];

    for (size_t q = 0; q < mLayer.mInputs.size(); ++q)
    {
        // if (mLayer.mNetworkParams.isGradNeeded(mLayer.mInputs[q]))
        {
            auto& in_nabla_tensor = memoryManager[mLayer.mInputs[q].grad()];

            if (mLayer.mBroadcastQuery[q])
            {
                auto in_nabla = in_nabla_tensor.getBroadcastedViewer(delta.getShape());
                for (size_t i = 0; i < in_nabla.size(); ++i)
                {
                    if ((q == 0 && mRandomCPU[i] == TOMMTYPE(1_dt)) || (q == 1 && mRandomCPU[i] == TOMMTYPE(0_dt)))
                    {
                        in_nabla[i] += delta[i];
                    }
                }
            }
            else
            {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
                for (size_t i = 0; i < in_nabla_tensor.size(); ++i)
                {
                    if ((q == 0 && mRandomCPU[i] == TOMMTYPE(1_dt)) || (q == 1 && mRandomCPU[i] == TOMMTYPE(0_dt)))
                    {
                        in_nabla_tensor[i] += delta[i];
                    }
                }
            }
        }
    }
}

template class RandomSelectLayerCPU<MemoryManager>;
template class RandomSelectLayerCPU<MemoryManagerFP16>;

} // namespace raul
