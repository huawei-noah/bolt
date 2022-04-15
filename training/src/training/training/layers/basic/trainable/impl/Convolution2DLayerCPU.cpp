// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "Convolution2DLayerCPU.h"
#include "../Convolution2DLayer.h"

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace raul
{

template<typename MM>
void Convolution2DLayerCPU<MM>::forwardComputeImpl(NetworkMode)
{
    Workflow& work = mLayer.mNetworkParams.mWorkflow;

    auto& weights = work.getMemoryManager<MM>()[mLayer.mWeightsName];

    if (mLayer.mQuantizeWeights)
    {
        work.getMemoryManager<MM>()[mLayer.mWeightsBackup] = TORANGE_MM(weights);
        mLayer.mNetworkParams.mQuantizerPtr->quantize(weights.begin(), weights.end());
        mLayer.mNetworkParams.mQuantizerPtr->dequantize(weights.begin(), weights.end());
    }

    const size_t batchSize = mLayer.mNetworkParams.mWorkflow.getBatchSize();

    auto& output = work.getMemoryManager<MM>()[mLayer.mOutputName];

    auto& inputs = work.getMemoryManager<MM>()[mLayer.mInputName];

    auto inputs3D = inputs.reshape(yato::dims(batchSize, mLayer.mInputDepth, mLayer.mInputHeight * mLayer.mInputWidth));
    auto outputs3D = output.reshape(yato::dims(batchSize, mLayer.mKernelsCount, mLayer.mOutputHeight * mLayer.mOutputWidth));

    // Fill dilated weights if needed
    if (mLayer.mDilationEnabled)
    {
        auto& dilationWeights = work.getMemoryManager<MM>()[mLayer.mDilationTensor];

        auto kernelsWeights4D = weights.reshape(yato::dims(mLayer.mKernelsCount, mLayer.mInputDepth / mLayer.mGroups, mLayer.mKernelHeight, mLayer.mKernelWidth));
        auto dilatedKernelsWeights4D =
            dilationWeights.reshape(yato::dims(mLayer.mKernelsCount, mLayer.mInputDepth / mLayer.mGroups, mLayer.mEffectiveReceptiveFieldH, mLayer.mEffectiveReceptiveFieldW));
#if defined(_OPENMP)
#pragma omp parallel for
#endif
        for (size_t kernelIndex = 0; kernelIndex < mLayer.mKernelsCount; ++kernelIndex)
        {
            for (size_t d = 0; d < mLayer.mInputDepth / mLayer.mGroups; ++d)
            {
                for (size_t ky = 0; ky < mLayer.mKernelHeight; ++ky)
                {
                    for (size_t kx = 0; kx < mLayer.mKernelWidth; ++kx)
                    {
                        dilatedKernelsWeights4D[kernelIndex][d][ky * mLayer.mDilationH][kx * mLayer.mDilationW] = kernelsWeights4D[kernelIndex][d][ky][kx];
                    }
                }
            }
        }
    }

#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (size_t q = 0; q < batchSize; ++q)
    {
        size_t index = 0;
#if defined(_OPENMP)
        index = omp_get_thread_num();
#endif

        auto& im2ColFor = work.getMemoryManager<MM>()[mLayer.mIm2ColForward[index]];

        Common::im2col(&inputs3D[q][0][0],
                       mLayer.mInputWidth,
                       mLayer.mInputHeight,
                       mLayer.mInputDepth,
                       mLayer.mEffectiveReceptiveFieldW,
                       mLayer.mEffectiveReceptiveFieldH,
                       mLayer.mStrideW,
                       mLayer.mStrideH,
                       mLayer.mPaddingW,
                       mLayer.mPaddingH,
                       &im2ColFor[0]);

        auto& wT = mLayer.mDilationEnabled ? work.getMemoryManager<MM>()[mLayer.mDilationTensor] : weights;

        for (size_t group = 0; group < mLayer.mGroups; ++group)
        {
            Common::gemm(&mLayer.mNetworkParams.mWorkflow.getKernelManager(),
                         "",
                         CblasNoTrans,
                         CblasNoTrans,
                         mLayer.mKernelsCount / mLayer.mGroups,
                         mLayer.mOutputWidth * mLayer.mOutputHeight,
                         mLayer.mEffectiveReceptiveFieldW * mLayer.mEffectiveReceptiveFieldH * mLayer.mInputDepth / mLayer.mGroups,
                         1.0_dt,
                         &wT[0] + group * mLayer.mKernelsCount / mLayer.mGroups * mLayer.mEffectiveReceptiveFieldW * mLayer.mEffectiveReceptiveFieldH * mLayer.mInputDepth / mLayer.mGroups,
                         &im2ColFor[0] + group * mLayer.mInputDepth / mLayer.mGroups * mLayer.mEffectiveReceptiveFieldW * mLayer.mEffectiveReceptiveFieldH * mLayer.mOutputWidth * mLayer.mOutputHeight,
                         0.0_dt,
                         &outputs3D[q][group * mLayer.mKernelsCount / mLayer.mGroups][0]);
        }
    }

    if (mLayer.mUseBias)
    {
        const auto& biases = work.getMemoryManager<MM>()[mLayer.mBiasesName];

#if defined(_OPENMP)
#pragma omp parallel for
#endif
        for (size_t q = 0; q < batchSize; ++q)
        {
            for (size_t kernelIndex = 0; kernelIndex < mLayer.mKernelsCount; ++kernelIndex)
            {
                const auto bias = biases[kernelIndex];
                std::transform(
                    outputs3D[q][kernelIndex].begin(), outputs3D[q][kernelIndex].end(), outputs3D[q][kernelIndex].begin(), [bias](typename MM::type& val) -> typename MM::type { return val + bias; });
            }
        }
    }

    if (mLayer.mQuantizeWeights)
    {
        weights = TORANGE_MM(work.getMemoryManager<MM>()[mLayer.mWeightsBackup]);
    }
}

template<typename MM>
void Convolution2DLayerCPU<MM>::backwardComputeImpl()
{
    Workflow& work = mLayer.mNetworkParams.mWorkflow;

    const size_t batchSize = mLayer.mNetworkParams.mWorkflow.getBatchSize();

    auto& deltas = work.getMemoryManager<MM>()[mLayer.mOutputName.grad()];

    auto deltas3D = deltas.reshape(yato::dims(batchSize, mLayer.mKernelsCount, mLayer.mOutputHeight * mLayer.mOutputWidth));

    const auto& weights = mLayer.mDilationEnabled ? work.getMemoryManager<MM>()[mLayer.mDilationTensor] : work.getMemoryManager<MM>()[mLayer.mWeightsName];

    // prevDelta
    ////if (mLayer.mNetworkParams.isGradNeeded(mInputName))
    {
        auto& prevLayerDelta = work.getMemoryManager<MM>()[mLayer.mInputName.grad()];
        auto prevDeltas3D = prevLayerDelta.reshape(yato::dims(batchSize, mLayer.mInputDepth, mLayer.mInputHeight * mLayer.mInputWidth));
#if defined(_OPENMP)
#pragma omp parallel for
#endif
        for (size_t i = 0; i < batchSize; ++i)
        {
            size_t index = 0;
#if defined(_OPENMP)
            index = omp_get_thread_num();
#endif
            auto& im2ColBack = work.getMemoryManager<MM>()[mLayer.mIm2ColBackward[index]];
            for (size_t group = 0; group < mLayer.mGroups; ++group)
            {
                Common::gemm(&mLayer.mNetworkParams.mWorkflow.getKernelManager(),
                             "",
                             CblasTrans,
                             CblasNoTrans,
                             mLayer.mEffectiveReceptiveFieldW * mLayer.mEffectiveReceptiveFieldH * mLayer.mInputDepth / mLayer.mGroups,
                             mLayer.mOutputWidth * mLayer.mOutputHeight,
                             mLayer.mKernelsCount / mLayer.mGroups,
                             1.0_dt,
                             &weights[0] + group * mLayer.mKernelsCount / mLayer.mGroups * mLayer.mEffectiveReceptiveFieldW * mLayer.mEffectiveReceptiveFieldH * mLayer.mInputDepth / mLayer.mGroups,
                             &deltas3D[i][0][0],
                             0.0_dt,
                             &im2ColBack[0] +
                                 group * mLayer.mEffectiveReceptiveFieldW * mLayer.mEffectiveReceptiveFieldH * mLayer.mInputDepth * mLayer.mOutputWidth * mLayer.mOutputHeight / mLayer.mGroups);
            }

            Common::col2im(&im2ColBack[0],
                           mLayer.mInputWidth,
                           mLayer.mInputHeight,
                           mLayer.mInputDepth,
                           mLayer.mEffectiveReceptiveFieldW,
                           mLayer.mEffectiveReceptiveFieldH,
                           mLayer.mStrideW,
                           mLayer.mStrideH,
                           mLayer.mPaddingW,
                           mLayer.mPaddingH,
                           &prevDeltas3D[i][0][0],
                           false,
                           false);
        }
    }

    // gradients weights
    if (!mLayer.mFrozen)
    {
        auto& inputs = work.getMemoryManager<MM>()[mLayer.mInputName];

        auto inputs3D = inputs.reshape(yato::dims(batchSize, mLayer.mInputDepth, mLayer.mInputHeight * mLayer.mInputWidth));

        auto& gradWeights = work.getMemoryManager<MM>()[mLayer.mWeightsName.grad()];

        if (mLayer.mDilationEnabled)
        {
            work.getMemoryManager<MM>()[mLayer.mDilationTensor] = TOMMTYPE(0);
        }

        if (mLayer.mNetworkParams.mCalculationMode == CalculationMode::DETERMINISTIC)
        {
            auto& im2ColBack = work.getMemoryManager<MM>()[mLayer.mIm2ColBackward[0]];

            auto& tG = mLayer.mDilationEnabled ? work.getMemoryManager<MM>()[mLayer.mDilationTensor] : gradWeights;

            for (size_t q = 0; q < batchSize; ++q)
            {
                Common::im2col(&inputs3D[q][0][0],
                               mLayer.mInputWidth,
                               mLayer.mInputHeight,
                               mLayer.mInputDepth,
                               mLayer.mEffectiveReceptiveFieldW,
                               mLayer.mEffectiveReceptiveFieldH,
                               mLayer.mStrideW,
                               mLayer.mStrideH,
                               mLayer.mPaddingW,
                               mLayer.mPaddingH,
                               &im2ColBack[0]);
                for (size_t group = 0; group < mLayer.mGroups; ++group)
                {
                    Common::gemm(&mLayer.mNetworkParams.mWorkflow.getKernelManager(),
                                 "",
                                 CblasNoTrans,
                                 CblasTrans,
                                 mLayer.mKernelsCount / mLayer.mGroups,
                                 mLayer.mEffectiveReceptiveFieldW * mLayer.mEffectiveReceptiveFieldH * mLayer.mInputDepth / mLayer.mGroups,
                                 mLayer.mOutputWidth * mLayer.mOutputHeight,
                                 1.0_dt,
                                 &deltas3D[q][group * mLayer.mKernelsCount / mLayer.mGroups][0],
                                 &im2ColBack[0] +
                                     group * mLayer.mEffectiveReceptiveFieldW * mLayer.mEffectiveReceptiveFieldH * mLayer.mInputDepth * mLayer.mOutputWidth * mLayer.mOutputHeight / mLayer.mGroups,
                                 1.0_dt,
                                 &tG[0] + group * mLayer.mKernelsCount / mLayer.mGroups * mLayer.mEffectiveReceptiveFieldW * mLayer.mEffectiveReceptiveFieldH * mLayer.mInputDepth / mLayer.mGroups);
                }
            }
        }
#if defined(_OPENMP)
        else if (mLayer.mNetworkParams.mCalculationMode == CalculationMode::FAST)
        {
            auto& tG = mLayer.mDilationEnabled ? work.getMemoryManager<MM>()[mLayer.mDilationTensor] : gradWeights;

#pragma omp parallel for
            for (size_t q = 0; q < batchSize; ++q)
            {
                size_t index = omp_get_thread_num();

                auto& im2ColBack = work.getMemoryManager<MM>()[mLayer.mIm2ColBackward[index]];

                Common::im2col(&inputs3D[q][0][0],
                               mLayer.mInputWidth,
                               mLayer.mInputHeight,
                               mLayer.mInputDepth,
                               mLayer.mEffectiveReceptiveFieldW,
                               mLayer.mEffectiveReceptiveFieldH,
                               mLayer.mStrideW,
                               mLayer.mStrideH,
                               mLayer.mPaddingW,
                               mLayer.mPaddingH,
                               &im2ColBack[0]);
#pragma omp critical
                for (size_t group = 0; group < mLayer.mGroups; ++group)
                {
                    Common::gemm(&mLayer.mNetworkParams.mWorkflow.getKernelManager(),
                                 "",
                                 CblasNoTrans,
                                 CblasTrans,
                                 mLayer.mKernelsCount / mLayer.mGroups,
                                 mLayer.mEffectiveReceptiveFieldW * mLayer.mEffectiveReceptiveFieldH * mLayer.mInputDepth / mLayer.mGroups,
                                 mLayer.mOutputWidth * mLayer.mOutputHeight,
                                 1.0_dt,
                                 &deltas3D[q][group * mLayer.mKernelsCount / mLayer.mGroups][0],
                                 &im2ColBack[0] +
                                     group * mLayer.mEffectiveReceptiveFieldW * mLayer.mEffectiveReceptiveFieldH * mLayer.mInputDepth * mLayer.mOutputWidth * mLayer.mOutputHeight / mLayer.mGroups,
                                 1.0_dt,
                                 &tG[0] + group * mLayer.mKernelsCount / mLayer.mGroups * mLayer.mEffectiveReceptiveFieldW * mLayer.mEffectiveReceptiveFieldH * mLayer.mInputDepth / mLayer.mGroups);
                }
            }
        }
#endif
        else
        {
            THROW("Convolution2DLayer", mLayer.mName, "unexpected calculation mode");
        }

        if (mLayer.mDilationEnabled)
        {
            const auto& dilationWeightsGrad = work.getMemoryManager<MM>()[mLayer.mDilationTensor];
            auto gradWeights4D = gradWeights.reshape(yato::dims(mLayer.mKernelsCount, mLayer.mInputDepth / mLayer.mGroups, mLayer.mKernelHeight, mLayer.mKernelWidth));
            const auto dilatedGradWeights4D =
                dilationWeightsGrad.reshape(yato::dims(mLayer.mKernelsCount, mLayer.mInputDepth / mLayer.mGroups, mLayer.mEffectiveReceptiveFieldH, mLayer.mEffectiveReceptiveFieldW));
#if defined(_OPENMP)
#pragma omp parallel for
#endif
            for (size_t kernelIndex = 0; kernelIndex < mLayer.mKernelsCount; ++kernelIndex)
            {
                for (size_t d = 0; d < mLayer.mInputDepth / mLayer.mGroups; ++d)
                {
                    for (size_t ky = 0; ky < mLayer.mKernelHeight; ++ky)
                    {
                        for (size_t kx = 0; kx < mLayer.mKernelWidth; ++kx)
                        {
                            gradWeights4D[kernelIndex][d][ky][kx] += dilatedGradWeights4D[kernelIndex][d][ky * mLayer.mDilationH][kx * mLayer.mDilationW];
                        }
                    }
                }
            }
        }
    }

    // gradients biases
    if (!mLayer.mFrozen && mLayer.mUseBias)
    {
        auto& gradBiases = work.getMemoryManager<MM>()[mLayer.mBiasesName.grad()];
        for (size_t i = 0; i < batchSize; ++i)
        {
            for (size_t kernelIndex = 0; kernelIndex < mLayer.mKernelsCount; ++kernelIndex)
            {
                gradBiases[kernelIndex] += std::accumulate(deltas3D[i][kernelIndex].begin(), deltas3D[i][kernelIndex].end(), TOMMTYPE(0), std::plus<typename MM::type>());
            }
        }
    }
}

template class Convolution2DLayerCPU<MemoryManager>;
template class Convolution2DLayerCPU<MemoryManagerFP16>;

} // namespace raul
