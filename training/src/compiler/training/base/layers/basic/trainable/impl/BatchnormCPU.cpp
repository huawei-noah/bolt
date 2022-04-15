// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "BatchnormCPU.h"
#include "../Batchnorm.h"

#include <training/base/impl/ImplFactory.h>

namespace
{

std::tuple<size_t, size_t, size_t> reassign(raul::Dimension dim, size_t i, size_t k, size_t q)
{
    if (dim == raul::Dimension::Depth)
    {
        return std::make_tuple(i, k, q);
    }
    if (dim == raul::Dimension::Height)
    {
        return std::make_tuple(k, i, q);
    }
    return std::make_tuple(k, q, i);
}

bool reg1 = raul::TheImplFactory::Instance().regCPUFP32<raul::BatchNormLayer, raul::BatchNormLayerCPU<raul::MemoryManager>>();
bool reg2 = raul::TheImplFactory::Instance().regCPUFP16<raul::BatchNormLayer, raul::BatchNormLayerCPU<raul::MemoryManagerFP16>>();
} // anonymous namespace

namespace raul
{
template<typename MM>
void BatchNormLayerCPU<MM>::initNotBSTensors()
{
    Workflow& work = mLayer.mNetworkParams.mWorkflow;

    work.getMemoryManager<MM>()[mLayer.mWeightsName] = TOMMTYPE(1_dt);
    work.getMemoryManager<MM>()[mLayer.mName / "VarianceEval"] = TOMMTYPE(1_dt);
}

template<typename MM>
void BatchNormLayerCPU<MM>::forwardComputeImpl(NetworkMode mode)
{
    Workflow& work = mLayer.mNetworkParams.mWorkflow;

    const size_t batchSize = mLayer.mNetworkParams.mWorkflow.getBatchSize();

    auto& outputs = work.getMemoryManager<MM>()[mLayer.mOutputName];

    const auto& gamma = work.getMemoryManager<MM>()[mLayer.mWeightsName];
    const auto& beta = work.getMemoryManager<MM>()[mLayer.mBiasesName];

    auto& xHat = work.getMemoryManager<MM>()[mLayer.mName / "XHat"];
    auto& varSqrt = work.getMemoryManager<MM>()[mLayer.mName / "VarianceSqrt"];

    auto& mean = work.getMemoryManager<MM>()[mLayer.mName / "Mean"];
    auto& var = work.getMemoryManager<MM>()[mLayer.mName / "Variance"];
    auto& meanEval = work.getMemoryManager<MM>()[mLayer.mName / "MeanEval"];
    auto& varEval = work.getMemoryManager<MM>()[mLayer.mName / "VarianceEval"];

    const auto& inputs = work.getMemoryManager<MM>()[mLayer.mInputName];

    if (mLayer.mDimension == raul::Dimension::Depth)
    {
        auto outputs3D = outputs.reshape(yato::dims(batchSize, mLayer.mInputDepth, mLayer.mInputHeight * mLayer.mInputWidth));
        auto inputs3D = inputs.reshape(yato::dims(batchSize, mLayer.mInputDepth, mLayer.mInputHeight * mLayer.mInputWidth));

        if (mode == NetworkMode::Train || mode == NetworkMode::TrainCheckpointed)
        {
            auto xhat3D = xHat.reshape(yato::dims(batchSize, mLayer.mInputDepth, mLayer.mInputHeight * mLayer.mInputWidth));

            const dtype reciprocalN = 1.0_dt / static_cast<dtype>(batchSize * mLayer.mInputHeight * mLayer.mInputWidth);
#if defined(_OPENMP)
#pragma omp parallel for
#endif
            for (size_t c = 0; c < mLayer.mInputDepth; ++c)
            {
                dtype sum = 0_dt;
                for (size_t batch = 0; batch < batchSize; ++batch)
                {
                    for (size_t i = 0; i < mLayer.mInputHeight * mLayer.mInputWidth; ++i)
                    {
                        sum += inputs3D[batch][c][i];
                    }
                }
                mean[c] = TOMMTYPE(sum * reciprocalN);
                if (mode == NetworkMode::Train)
                {
                    meanEval[c] = TOMMTYPE((1.0_dt - mLayer.mMomentum) * TODTYPE(meanEval[c]) + mLayer.mMomentum * mean[c]);
                }
            }
#if defined(_OPENMP)
#pragma omp parallel for
#endif
            for (size_t c = 0; c < mLayer.mInputDepth; ++c)
            {
                dtype varT = 0_dt;
                dtype vv = 0;
                for (size_t batch = 0; batch < batchSize; ++batch)
                {
                    for (size_t i = 0; i < mLayer.mInputHeight * mLayer.mInputWidth; ++i)
                    {
                        vv = TODTYPE(inputs3D[batch][c][i] - mean[c]);
                        varT += vv * vv * reciprocalN;
                    }
                }
                var[c] = TOMMTYPE(varT);
                varSqrt[c] = TOMMTYPE(std::sqrt(varT + mLayer.mEps));
                if (mode == NetworkMode::Train)
                {
                    varEval[c] = TOMMTYPE((1.0_dt - mLayer.mMomentum) * TODTYPE(varEval[c]) + mLayer.mMomentum * var[c]);
                }
            }
#if defined(_OPENMP)
#pragma omp parallel for
#endif
            for (size_t c = 0; c < mLayer.mInputDepth; ++c)
            {
                auto g = gamma[c];
                auto b = beta[c];
                dtype v = 1_dt / TODTYPE(varSqrt[c]);
                auto m = mean[c];
                for (size_t batch = 0; batch < batchSize; ++batch)
                {
                    for (size_t i = 0; i < mLayer.mInputHeight * mLayer.mInputWidth; ++i)
                    {
                        dtype val = (inputs3D[batch][c][i] - m) * v;
                        xhat3D[batch][c][i] = TOMMTYPE(val);
                        outputs3D[batch][c][i] = TOMMTYPE(g * val + b);
                    }
                }
            }
        }
        else
        {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
            for (size_t c = 0; c < mLayer.mInputDepth; ++c)
            {
                for (size_t batch = 0; batch < batchSize; ++batch)
                {
                    for (size_t i = 0; i < mLayer.mInputHeight * mLayer.mInputWidth; ++i)
                    {
                        varSqrt[c] = TOMMTYPE(std::sqrt(TODTYPE(varEval[c] + mLayer.mEps)));
                        outputs3D[batch][c][i] = gamma[c] * ((inputs3D[batch][c][i] - meanEval[c]) / varSqrt[c]) + beta[c];
                    }
                }
            }
        }
    }
    else
    {
        auto outputs4D = outputs.get4DView();
        auto inputs4D = inputs.get4DView();

        if (mode == NetworkMode::Train || mode == NetworkMode::TrainCheckpointed)
        {
            auto xhat4D = xHat.get4DView();

            Tensor sum(mLayer.mChosenDimSize, 0.0_dt);
            Tensor varT(mLayer.mChosenDimSize, 0.0_dt);

            const dtype reciprocalN = 1.0_dt / static_cast<dtype>(batchSize * mLayer.mOtherDims[0] * mLayer.mOtherDims[1]);
#if defined(_OPENMP)
#pragma omp parallel for
#endif
            for (size_t i = 0; i < mLayer.mChosenDimSize; ++i)
            {
                for (size_t j = 0; j < batchSize; ++j)
                {
                    for (size_t k = 0; k < mLayer.mOtherDims[0]; ++k)
                    {
                        for (size_t q = 0; q < mLayer.mOtherDims[1]; ++q)
                        {
                            // Rearrange indices in proper way
                            auto [depth, height, width] = reassign(mLayer.mDimension, i, k, q);
                            sum[i] += inputs4D[j][depth][height][width];
                        }
                    }
                }
                mean[i] = TOMMTYPE(sum[i] * reciprocalN);
                if (mode == NetworkMode::Train)
                {
                    meanEval[i] = TOMMTYPE((1.0_dt - mLayer.mMomentum) * TODTYPE(meanEval[i]) + mLayer.mMomentum * mean[i]);
                }
            }
#if defined(_OPENMP)
#pragma omp parallel for
#endif
            for (size_t i = 0; i < mLayer.mChosenDimSize; ++i)
            {
                for (size_t j = 0; j < batchSize; ++j)
                {
                    for (size_t k = 0; k < mLayer.mOtherDims[0]; ++k)
                    {
                        for (size_t q = 0; q < mLayer.mOtherDims[1]; ++q)
                        {
                            // Rearrange indices in proper way
                            auto [depth, height, width] = reassign(mLayer.mDimension, i, k, q);
                            varT[i] += TODTYPE(pow(inputs4D[j][depth][height][width] - mean[i], 2.0_dt)) * reciprocalN;
                        }
                    }
                }
                var[i] = TOMMTYPE(varT[i]);
                if (mode == NetworkMode::Train)
                {
                    varEval[i] = TOMMTYPE((1.0_dt - mLayer.mMomentum) * TODTYPE(varEval[i]) + mLayer.mMomentum * var[i]);
                }
            }
#if defined(_OPENMP)
#pragma omp parallel for
#endif
            for (size_t i = 0; i < mLayer.mChosenDimSize; ++i)
            {
                for (size_t j = 0; j < batchSize; ++j)
                {
                    for (size_t k = 0; k < mLayer.mOtherDims[0]; ++k)
                    {
                        for (size_t q = 0; q < mLayer.mOtherDims[1]; ++q)
                        {
                            // Rearrange indices in proper way
                            auto [depth, height, width] = reassign(mLayer.mDimension, i, k, q);
                            varSqrt[i] = TOMMTYPE(std::sqrt(TODTYPE(var[i] + mLayer.mEps)));
                            xhat4D[j][depth][height][width] = (inputs4D[j][depth][height][width] - mean[i]) / varSqrt[i];
                            outputs4D[j][depth][height][width] = gamma[i] * xhat4D[j][depth][height][width] + beta[i];
                        }
                    }
                }
            }
        }
        else
        {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
            for (size_t i = 0; i < mLayer.mChosenDimSize; ++i)
            {
                for (size_t j = 0; j < batchSize; ++j)
                {
                    for (size_t k = 0; k < mLayer.mOtherDims[0]; ++k)
                    {
                        for (size_t q = 0; q < mLayer.mOtherDims[1]; ++q)
                        {
                            // Rearrange indices in proper way
                            auto [depth, height, width] = reassign(mLayer.mDimension, i, k, q);
                            varSqrt[i] = TOMMTYPE(std::sqrt(TODTYPE(varEval[i] + mLayer.mEps)));
                            outputs4D[j][depth][height][width] = gamma[i] * ((inputs4D[j][depth][height][width] - meanEval[i]) / varSqrt[i]) + beta[i];
                        }
                    }
                }
            }
        }
    }
}

template<typename MM>
void BatchNormLayerCPU<MM>::backwardComputeImpl()
{
    Workflow& work = mLayer.mNetworkParams.mWorkflow;

    const size_t batchSize = mLayer.mNetworkParams.mWorkflow.getBatchSize();

    const auto& xHat = work.getMemoryManager<MM>()[mLayer.mName / "XHat"];
    const auto& varSqrt = work.getMemoryManager<MM>()[mLayer.mName / "VarianceSqrt"];

    const auto& deltas = work.getMemoryManager<MM>()[mLayer.mOutputName.grad()];

    const auto& gamma = work.getMemoryManager<MM>()[mLayer.mWeightsName];

    if (mLayer.mDimension == raul::Dimension::Depth)
    {
        auto deltas3D = deltas.reshape(yato::dims(batchSize, mLayer.mInputDepth, mLayer.mInputHeight * mLayer.mInputWidth));
        auto xhat3D = xHat.reshape(yato::dims(batchSize, mLayer.mInputDepth, mLayer.mInputHeight * mLayer.mInputWidth));

        if (!mLayer.mFrozen)
        {
            auto& nablaBeta = work.getMemoryManager<MM>()[mLayer.mBiasesName.grad()];
            auto& nablaGamma = work.getMemoryManager<MM>()[mLayer.mWeightsName.grad()];
#if defined(_OPENMP)
#pragma omp parallel for
#endif
            for (size_t c = 0; c < mLayer.mInputDepth; ++c)
            {
                for (size_t batch = 0; batch < batchSize; ++batch)
                {
                    for (size_t i = 0; i < mLayer.mInputHeight * mLayer.mInputWidth; ++i)
                    {
                        nablaBeta[c] += deltas3D[batch][c][i];
                        nablaGamma[c] += deltas3D[batch][c][i] * xhat3D[batch][c][i];
                    }
                }
            }
        }

        ////if (mLayer.mNetworkParams.isGradNeeded(mLayer.mInputName))
        {
            auto& prevLayerDelta = work.getMemoryManager<MM>()[mLayer.mInputName.grad()];
            auto prevDeltas3D = prevLayerDelta.reshape(yato::dims(batchSize, mLayer.mInputDepth, mLayer.mInputHeight * mLayer.mInputWidth));
            const size_t N = batchSize * mLayer.mInputHeight * mLayer.mInputWidth;

#if defined(_OPENMP)
#pragma omp parallel for
#endif
            for (size_t c = 0; c < mLayer.mInputDepth; ++c)
            {
                dtype g = TODTYPE(gamma[c]);
                dtype dvar = 0;
                dtype dvar2 = 0;
                dtype val = 0;
                dtype coeff = 1_dt / (TODTYPE(N) * TODTYPE(varSqrt[c]));
                for (size_t batch = 0; batch < batchSize; ++batch)
                {
                    for (size_t i = 0; i < mLayer.mInputHeight * mLayer.mInputWidth; ++i)
                    {
                        val = deltas3D[batch][c][i] * g;
                        dvar += val;
                        dvar2 += val * TODTYPE(xhat3D[batch][c][i]);
                    }
                }

                for (size_t batch = 0; batch < batchSize; ++batch)
                {
                    for (size_t i = 0; i < mLayer.mInputHeight * mLayer.mInputWidth; ++i)
                    {
                        prevDeltas3D[batch][c][i] += TOMMTYPE((TODTYPE(N) * deltas3D[batch][c][i] * g - dvar - xhat3D[batch][c][i] * dvar2) * coeff);
                    }
                }
            }
        }
    }
    else
    {
        auto deltas4D = deltas.get4DView();
        auto xhat4D = xHat.get4DView();

        if (!mLayer.mFrozen)
        {
            auto& nablaBeta = work.getMemoryManager<MM>()[mLayer.mBiasesName.grad()];
            auto& nablaGamma = work.getMemoryManager<MM>()[mLayer.mWeightsName.grad()];
#if defined(_OPENMP)
#pragma omp parallel for
#endif
            for (size_t i = 0; i < mLayer.mChosenDimSize; ++i)
            {
                for (size_t j = 0; j < batchSize; ++j)
                {
                    for (size_t k = 0; k < mLayer.mOtherDims[0]; ++k)
                    {
                        for (size_t q = 0; q < mLayer.mOtherDims[1]; ++q)
                        {
                            // Rearrange indices in proper way
                            auto [depth, height, width] = reassign(mLayer.mDimension, i, k, q);
                            nablaBeta[i] += deltas4D[j][depth][height][width];
                            nablaGamma[i] += deltas4D[j][depth][height][width] * xhat4D[j][depth][height][width];
                        }
                    }
                }
            }
        }

        ////if (mLayer.mNetworkParams.isGradNeeded(mLayer.mInputName))
        {
            Tensor nablaXhat(batchSize, mLayer.mInputDepth, mLayer.mInputHeight, mLayer.mInputWidth);
            auto nablaXhat4D = nablaXhat.get4DView();

            Tensor dvar(mLayer.mChosenDimSize, 0.0_dt);
            Tensor dvar2(mLayer.mChosenDimSize, 0.0_dt);
#if defined(_OPENMP)
#pragma omp parallel for
#endif
            for (size_t i = 0; i < mLayer.mChosenDimSize; ++i)
            {
                for (size_t j = 0; j < batchSize; ++j)
                {
                    for (size_t k = 0; k < mLayer.mOtherDims[0]; ++k)
                    {
                        for (size_t q = 0; q < mLayer.mOtherDims[1]; ++q)
                        {
                            // Rearrange indices in proper way
                            auto [depth, height, width] = reassign(mLayer.mDimension, i, k, q);
                            nablaXhat4D[j][depth][height][width] = TODTYPE(deltas4D[j][depth][height][width]) * TODTYPE(gamma[i]);
                            dvar[i] += nablaXhat4D[j][depth][height][width];
                            dvar2[i] += nablaXhat4D[j][depth][height][width] * TODTYPE(xhat4D[j][depth][height][width]);
                        }
                    }
                }
            }

            auto& prevLayerDelta = work.getMemoryManager<MM>()[mLayer.mInputName.grad()];
            auto prevDeltas4D = prevLayerDelta.get4DView();
            const size_t N = batchSize * mLayer.mOtherDims[0] * mLayer.mOtherDims[1];
#if defined(_OPENMP)
#pragma omp parallel for
#endif
            for (size_t i = 0; i < mLayer.mChosenDimSize; ++i)
            {
                for (size_t j = 0; j < batchSize; ++j)
                {
                    for (size_t k = 0; k < mLayer.mOtherDims[0]; ++k)
                    {
                        for (size_t q = 0; q < mLayer.mOtherDims[1]; ++q)
                        {
                            // Rearrange indices in proper way
                            auto [depth, height, width] = reassign(mLayer.mDimension, i, k, q);
                            prevDeltas4D[j][depth][height][width] += TOMMTYPE((TODTYPE(N) * nablaXhat4D[j][depth][height][width] - dvar[i] - TODTYPE(xhat4D[j][depth][height][width]) * dvar2[i]) /
                                                                              (static_cast<raul::dtype>(N) * TODTYPE(varSqrt[i])));
                        }
                    }
                }
            }
        }
    }
}

template class BatchNormLayerCPU<MemoryManager>;
template class BatchNormLayerCPU<MemoryManagerFP16>;
} // namespace raul