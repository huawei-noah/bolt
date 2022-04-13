// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "LayerNormCPU.h"
#include "../LayerNorm.h"

namespace raul
{
template<typename MM>
void LayerNormLayerCPU<MM>::initNotBSTensors()
{
    Workflow& work = mLayer.mNetworkParams.mWorkflow;

    work.getMemoryManager<MM>()[mLayer.mWeightsName] = TOMMTYPE(1_dt);
}

template<typename MM>
void LayerNormLayerCPU<MM>::forwardComputeImpl(NetworkMode mode)
{
    Workflow& work = mLayer.mNetworkParams.mWorkflow;

    const size_t batchSize = mLayer.mNetworkParams.mWorkflow.getBatchSize();

    auto& output = work.getMemoryManager<MM>()[mLayer.mOutputName];
    auto& inputs = work.getMemoryManager<MM>()[mLayer.mInputName];
    auto& gamma = work.getMemoryManager<MM>()[mLayer.mWeightsName];
    auto& beta = work.getMemoryManager<MM>()[mLayer.mBiasesName];

    size_t N = batchSize * mLayer.mInputDepth * mLayer.mInputHeight;

    auto outputs2D = output.reshape(yato::dims(N, mLayer.mInputWidth));
    auto inputs2D = inputs.reshape(yato::dims(N, mLayer.mInputWidth));

    if (mode == NetworkMode::Train || mode == NetworkMode::TrainCheckpointed)
    {
        auto& xHat = work.getMemoryManager<MM>()[mLayer.mXHatName];
        auto& Var = work.getMemoryManager<MM>()[mLayer.mVarianceName];

        auto xhat2D = xHat.reshape(yato::dims(N, mLayer.mInputWidth));

#if defined(_OPENMP)
#pragma omp parallel for
#endif
        for (size_t n = 0; n < N; ++n)
        {
            raul::dtype mean = 0.0_dt;
            raul::dtype var = 0.0_dt;

            for (size_t i = 0; i < mLayer.mInputWidth; ++i)
            {
                mean += inputs2D[n][i];
            }
            mean /= static_cast<dtype>(mLayer.mInputWidth);
            for (size_t i = 0; i < mLayer.mInputWidth; ++i)
            {
                var += (inputs2D[n][i] - mean) * (inputs2D[n][i] - mean);
            }
            if (mLayer.mTFStyle)
            {
                var = TODTYPE(sqrt(var / TODTYPE(mLayer.mInputWidth) + mLayer.mEps));
            }
            else
            {
                if (mLayer.mUseBesselCorrection)
                {
                    var = TODTYPE(sqrt(var / TODTYPE(mLayer.mInputWidth - 1)) + mLayer.mEps); // we use Bessel's correction to mimic torch default behaviour
                }
                else
                {
                    var = TODTYPE(sqrt(var / TODTYPE(mLayer.mInputWidth)) + mLayer.mEps);
                }
            }

            auto var_1 = 1.0_dt / var;

            Var[n] = TOMMTYPE(var_1);

            for (size_t i = 0; i < mLayer.mInputWidth; ++i)
            {
                xhat2D[n][i] = (inputs2D[n][i] - TOMMTYPE(mean)) * TOMMTYPE(var_1);
                outputs2D[n][i] = gamma[i] * xhat2D[n][i] + beta[i];
            }
        }
    }
    else
    {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
        for (size_t n = 0; n < N; ++n)
        {
            dtype mean = 0_dt;
            dtype var = 0_dt;

            for (size_t i = 0; i < mLayer.mInputWidth; ++i)
            {
                mean += inputs2D[n][i];
            }
            mean /= static_cast<dtype>(mLayer.mInputWidth);
            for (size_t i = 0; i < mLayer.mInputWidth; ++i)
            {
                var += (inputs2D[n][i] - mean) * (inputs2D[n][i] - mean);
            }
            if (mLayer.mTFStyle)
            {
                var = TODTYPE(sqrt(var / TODTYPE(mLayer.mInputWidth) + mLayer.mEps));
            }
            else
            {
                if (mLayer.mUseBesselCorrection)
                {
                    var = TODTYPE(sqrt(var / TODTYPE(mLayer.mInputWidth - 1)) + mLayer.mEps); // we use Bessel's correction to mimic torch default behaviour
                }
                else
                {
                    var = TODTYPE(sqrt(var / TODTYPE(mLayer.mInputWidth)) + mLayer.mEps);
                }
            }
            auto var_1 = 1.0_dt / var;
            for (size_t i = 0; i < mLayer.mInputWidth; ++i)
            {
                auto xhat = (inputs2D[n][i] - TOMMTYPE(mean)) * TOMMTYPE(var_1);
                outputs2D[n][i] = gamma[i] * TOMMTYPE(xhat) + beta[i];
            }
        }
    }
}

template<typename MM>
void LayerNormLayerCPU<MM>::backwardComputeImpl()
{
    Workflow& work = mLayer.mNetworkParams.mWorkflow;

    const size_t batchSize = mLayer.mNetworkParams.mWorkflow.getBatchSize();

    auto& xHat = work.getMemoryManager<MM>()[mLayer.mXHatName];
    auto& Var = work.getMemoryManager<MM>()[mLayer.mVarianceName];

    auto& deltas = work.getMemoryManager<MM>()[mLayer.mOutputName.grad()];

    size_t N = batchSize * mLayer.mInputDepth * mLayer.mInputHeight;

    auto deltas2D = deltas.reshape(yato::dims(N, mLayer.mInputWidth));

    auto xhat2D = xHat.reshape(yato::dims(N, mLayer.mInputWidth));

    if (!mLayer.mFrozen)
    {
        auto& gradWeights = work.getMemoryManager<MM>()[mLayer.mWeightsName.grad()];
        auto& gradBiases = work.getMemoryManager<MM>()[mLayer.mBiasesName.grad()];
#if defined(_OPENMP)
#pragma omp parallel for
#endif
        for (size_t i = 0; i < mLayer.mInputWidth; ++i)
        {
            for (size_t n = 0; n < N; ++n)
            {
                gradBiases[i] += deltas2D[n][i];
                gradWeights[i] += deltas2D[n][i] * xhat2D[n][i];
            }
        }
    }

    // if (mLayer.mNetworkParams.isGradNeeded(mLayer.mInputName))
    {
        auto& nablaXhat = work.getMemoryManager<MM>()[mLayer.mXHatNablaName];
        auto nablaXhat2D = nablaXhat.reshape(yato::dims(N, mLayer.mInputWidth));
        auto& gamma = work.getMemoryManager<MM>()[mLayer.mWeightsName];
#if defined(_OPENMP)
#pragma omp parallel for
#endif
        for (size_t i = 0; i < mLayer.mInputWidth; ++i)
        {
            for (size_t n = 0; n < N; ++n)
            {
                nablaXhat2D[n][i] = deltas2D[n][i] * gamma[i];
            }
        }

        auto& prevLayerDelta = work.getMemoryManager<MM>()[mLayer.mInputName.grad()];
        auto prevDeltas2D = prevLayerDelta.reshape(yato::dims(N, mLayer.mInputWidth));
        auto v_1 = 1_dt / TODTYPE(mLayer.mInputWidth);
        auto scale = mLayer.mTFStyle || !mLayer.mUseBesselCorrection ? 1_dt / TODTYPE(mLayer.mInputWidth) : 1_dt / (TODTYPE(mLayer.mInputWidth) - 1_dt);
#if defined(_OPENMP)
#pragma omp parallel for
#endif
        for (size_t n = 0; n < N; ++n)
        {
            for (size_t j = 0; j < mLayer.mInputWidth; ++j)
            {
                auto coeff = xhat2D[n][j] * scale;
                dtype val = 0_dt;
                for (size_t i = 0; i < mLayer.mInputWidth; ++i)
                {
                    dtype V = v_1 + xhat2D[n][i] * coeff;
                    val -= V * nablaXhat2D[n][i];
                }
                val += nablaXhat2D[n][j];
                prevDeltas2D[n][j] += TOMMTYPE(val) * Var[n];
            }
        }
    }
}

template class LayerNormLayerCPU<MemoryManager>;
template class LayerNormLayerCPU<MemoryManagerFP16>;
} // namespace raul