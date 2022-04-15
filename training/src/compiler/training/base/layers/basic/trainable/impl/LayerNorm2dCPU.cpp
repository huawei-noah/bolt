// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "LayerNorm2dCPU.h"
#include "../LayerNorm2D.h"

namespace raul
{

template<typename MM>
LayerNorm2DLayerCPU<MM>::LayerNorm2DLayerCPU(LayerNorm2DLayer& layer)
    : mLayer(layer)
{
    Workflow& work = mLayer.mNetworkParams.mWorkflow;

    work.tensorNeeded(mLayer.mName, mLayer.mName / "cache1", WShape{ BS(), mLayer.mInputDepth, 1u, 1u }, DEC_BACK_WRIT);

    work.tensorNeeded(mLayer.mName, mLayer.mName / "cache2", WShape{ BS(), mLayer.mInputDepth, 1u, 1u }, DEC_BACK_WRIT);
}

template<typename MM>
void LayerNorm2DLayerCPU<MM>::initNotBSTensors()
{
    Workflow& work = mLayer.mNetworkParams.mWorkflow;
    work.getMemoryManager<MM>()[mLayer.mWeightsName] = TOMMTYPE(1_dt);
}

template<typename MM>
void LayerNorm2DLayerCPU<MM>::forwardComputeImpl(NetworkMode mode)
{
    Workflow& work = mLayer.mNetworkParams.mWorkflow;

    const size_t batchSize = mLayer.mNetworkParams.mWorkflow.getBatchSize();

    auto& output = work.getMemoryManager<MM>()[mLayer.mOutputName];
    auto& inputs = work.getMemoryManager<MM>()[mLayer.mInputName];
    auto& gamma = work.getMemoryManager<MM>()[mLayer.mWeightsName];
    auto& beta = work.getMemoryManager<MM>()[mLayer.mBiasesName];

    size_t N = batchSize * mLayer.mInputDepth;

    size_t len = mLayer.mInputWidth * mLayer.mInputHeight;

    auto outputs2D = output.reshape(yato::dims(N, len));
    auto inputs2D = inputs.reshape(yato::dims(N, len));

    if (mode == NetworkMode::Train || mode == NetworkMode::TrainCheckpointed)
    {
        auto& xHat = work.getMemoryManager<MM>()[mLayer.mXHatName];
        auto& Var = work.getMemoryManager<MM>()[mLayer.mVarianceName];

        auto xhat2D = xHat.reshape(yato::dims(N, len));

#if defined(_OPENMP)
#pragma omp parallel for
#endif
        for (size_t n = 0; n < N; ++n)
        {
            raul::dtype mean = 0.0_dt;
            raul::dtype var = 0.0_dt;

            for (size_t i = 0; i < len; ++i)
            {
                mean += TODTYPE(inputs2D[n][i]);
            }
            mean /= TODTYPE(len);
            for (size_t i = 0; i < len; ++i)
            {
                var += (TODTYPE(inputs2D[n][i]) - mean) * (TODTYPE(inputs2D[n][i]) - mean);
            }

            if (mLayer.mTFStyle)
            {
                var = TODTYPE(sqrt(var / TODTYPE(len) + mLayer.mEps));
            }
            else
            {
                if (mLayer.mUseBesselCorrection)
                {
                    var = TODTYPE(sqrt(var / TODTYPE(len - 1)) + mLayer.mEps); // we use Bessel's correction to mimic torch default behaviour
                }
                else
                {
                    var = TODTYPE(sqrt(var / TODTYPE(len)) + mLayer.mEps);
                }
            }

            auto var_1 = 1.0_dt / var;

            Var[n] = TOMMTYPE(var_1);
            size_t ind = 0;
            for (size_t j = 0; j < mLayer.mInputHeight; ++j)
            {
                for (size_t i = 0; i < mLayer.mInputWidth; ++i, ++ind)
                {
                    dtype xhat = (TODTYPE(inputs2D[n][ind]) - mean) * var_1;
                    xhat2D[n][ind] = TOMMTYPE(xhat);
                    outputs2D[n][ind] = TOMMTYPE(TODTYPE(gamma[i]) * xhat + TODTYPE(beta[i]));
                }
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

            for (size_t i = 0; i < len; ++i)
            {
                mean += inputs2D[n][i];
            }
            mean /= static_cast<dtype>(len);
            for (size_t i = 0; i < len; ++i)
            {
                var += (inputs2D[n][i] - mean) * (inputs2D[n][i] - mean);
            }

            if (mLayer.mTFStyle)
            {
                var = TODTYPE(sqrt(var / TODTYPE(len) + mLayer.mEps));
            }
            else
            {
                if (mLayer.mUseBesselCorrection)
                {
                    var = TODTYPE(sqrt(var / TODTYPE(len - 1)) + mLayer.mEps); // we use Bessel's correction to mimic torch default behaviour
                }
                else
                {
                    var = TODTYPE(sqrt(var / TODTYPE(len)) + mLayer.mEps);
                }
            }

            auto var_1 = 1.0_dt / var;
            size_t ind = 0;
            for (size_t j = 0; j < mLayer.mInputHeight; ++j)
            {
                for (size_t i = 0; i < mLayer.mInputWidth; ++i, ++ind)
                {
                    auto xhat = (inputs2D[n][ind] - mean) * var_1;
                    outputs2D[n][ind] = TOMMTYPE(gamma[i] * xhat + beta[i]);
                }
            }
        }
    }
}

template<typename MM>
void LayerNorm2DLayerCPU<MM>::backwardComputeImpl()
{
    Workflow& work = mLayer.mNetworkParams.mWorkflow;

    const size_t batchSize = mLayer.mNetworkParams.mWorkflow.getBatchSize();

    auto& xHat = work.getMemoryManager<MM>()[mLayer.mXHatName];
    auto& Var = work.getMemoryManager<MM>()[mLayer.mVarianceName];

    auto& deltas = work.getMemoryManager<MM>()[mLayer.mOutputName.grad()];

    size_t N = batchSize * mLayer.mInputDepth;

    size_t len = mLayer.mInputHeight * mLayer.mInputWidth;

    auto deltas2D = deltas.reshape(yato::dims(N, len));

    auto xhat2D = xHat.reshape(yato::dims(N, len));

    if (!mLayer.mFrozen)
    {
        auto& gradWeights = work.getMemoryManager<MM>()[mLayer.mWeightsName.grad()];
        auto& gradBiases = work.getMemoryManager<MM>()[mLayer.mBiasesName.grad()];
#if defined(_OPENMP)
#pragma omp parallel for
#endif
        for (size_t i = 0; i < mLayer.mInputWidth; ++i)
        {
            dtype gBias = 0;
            dtype gWeights = 0;
            for (size_t j = 0; j < mLayer.mInputHeight; ++j)
            {
                size_t ind = i + j * mLayer.mInputWidth;
                for (size_t n = 0; n < N; ++n)
                {
                    gBias += TODTYPE(deltas2D[n][ind]);
                    gWeights += TODTYPE(deltas2D[n][ind]) * TODTYPE(xhat2D[n][ind]);
                }
            }

            gradBiases[i] = TOMMTYPE(TODTYPE(gradBiases[i]) + gBias);
            gradWeights[i] = TOMMTYPE(TODTYPE(gradWeights[i]) + gWeights);
        }
    }

    // if (mLayer.mNetworkParams.isGradNeeded(mLayer.mInputName))
    {
        auto& nablaXhat = work.getMemoryManager<MM>()[mLayer.mXHatNablaName];
        auto nablaXhat2D = nablaXhat.reshape(yato::dims(N, len));
        auto& gamma = work.getMemoryManager<MM>()[mLayer.mWeightsName];
#if defined(_OPENMP)
#pragma omp parallel for
#endif
        for (size_t n = 0; n < N; ++n)
        {
            size_t ind = 0;
            for (size_t j = 0; j < mLayer.mInputHeight; ++j)
            {
                for (size_t i = 0; i < mLayer.mInputWidth; ++i, ++ind)
                {
                    nablaXhat2D[n][ind] = deltas2D[n][ind] * gamma[i];
                }
            }
        }

        auto& prevLayerDelta = work.getMemoryManager<MM>()[mLayer.mInputName.grad()];
        auto prevDeltas2D = prevLayerDelta.reshape(yato::dims(N, len));

        auto v_1 = 1_dt / TODTYPE(len);
        auto scale = mLayer.mTFStyle || !mLayer.mUseBesselCorrection ? 1_dt / TODTYPE(len) : 1_dt / (TODTYPE(len) - 1_dt);

        auto& cacheTensor1 = work.getMemoryManager<MM>().getTensor(mLayer.mName / "cache1");
        auto& cacheTensor2 = work.getMemoryManager<MM>().getTensor(mLayer.mName / "cache2");
        auto* cache1 = cacheTensor1.getBuffer();
        auto* cache2 = cacheTensor2.getBuffer();

        auto v_1_mm = TOMMTYPE(v_1);
        for (size_t n = 0; n < N; ++n)
        {
            cache1[n] = 0;
            cache2[n] = 0;
            for (size_t i = 0; i < len; ++i)
            {
                cache1[n] += nablaXhat2D[n][i] * v_1_mm;
                cache2[n] += nablaXhat2D[n][i] * xhat2D[n][i];
            }

            // cache1[n] *= v_1_mm;
        }

#if defined(_OPENMP)
#pragma omp parallel for
#endif
        for (size_t n = 0; n < N; ++n)
        {
            for (size_t j = 0; j < len; ++j)
            {
                auto coeff = TODTYPE(xhat2D[n][j]) * scale;
                auto val = TODTYPE(nablaXhat2D[n][j]) - TODTYPE(cache1[n]) - coeff * TODTYPE(cache2[n]);
                prevDeltas2D[n][j] = TOMMTYPE(TODTYPE(prevDeltas2D[n][j]) + val * TODTYPE(Var[n]));
            }
        }
    }
}

template class LayerNorm2DLayerCPU<MemoryManager>;
template class LayerNorm2DLayerCPU<MemoryManagerFP16>;
} // namespace raul