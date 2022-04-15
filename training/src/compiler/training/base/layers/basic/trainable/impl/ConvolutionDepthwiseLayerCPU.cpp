// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "ConvolutionDepthwiseLayerCPU.h"
#include "../ConvolutionDepthwiseLayer.h"

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace raul
{

template<typename MM>
void ConvolutionDepthwiseLayerCPU<MM>::forwardComputeImpl(NetworkMode)
{
    Workflow& work = mLayer.mNetworkParams.mWorkflow;

    const size_t batchSize = mLayer.mNetworkParams.mWorkflow.getBatchSize();

    auto& output = work.getMemoryManager<MM>()[mLayer.mOutputName];

    const auto& inputs = work.getMemoryManager<MM>()[mLayer.mInputName];

    auto inputs3D = inputs.reshape(yato::dims(batchSize, mLayer.mInputDepth, mLayer.mInputHeight * mLayer.mInputWidth));
    auto outputs3D = output.reshape(yato::dims(batchSize, mLayer.mKernelsCount, mLayer.mOutputHeight * mLayer.mOutputWidth));

    const auto& weights = work.getMemoryManager<MM>()[mLayer.mWeightsName];
    auto kernelsWeights2D = weights.reshape(yato::dims(mLayer.mKernelsCount, mLayer.mKernelHeight * mLayer.mKernelWidth));

#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (size_t q = 0; q < batchSize; ++q)
    {
        for (size_t w = 0; w < mLayer.mInputDepth; ++w)
        {
            size_t index = 0;
#if defined(_OPENMP)
            index = omp_get_thread_num();
#endif

            auto& im2ColFor = work.getMemoryManager<MM>()[mLayer.mIm2ColForward[index]];

            Common::im2col(&inputs3D[q][w][0],
                           mLayer.mInputWidth,
                           mLayer.mInputHeight,
                           1,
                           mLayer.mKernelWidth,
                           mLayer.mKernelHeight,
                           mLayer.mStrideW,
                           mLayer.mStrideH,
                           mLayer.mPaddingW,
                           mLayer.mPaddingH,
                           &im2ColFor[0]);

            Common::gemm(CblasNoTrans,
                         CblasNoTrans,
                         1,
                         mLayer.mOutputWidth * mLayer.mOutputHeight,
                         mLayer.mKernelWidth * mLayer.mKernelHeight * 1,
                         1.0_dt,
                         &kernelsWeights2D[w][0],
                         &im2ColFor[0],
                         0.0_dt,
                         &outputs3D[q][w][0]);
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
                const typename MM::type bias = biases[kernelIndex];
                std::transform(
                    outputs3D[q][kernelIndex].begin(), outputs3D[q][kernelIndex].end(), outputs3D[q][kernelIndex].begin(), [bias](typename MM::type& val) -> typename MM::type { return val + bias; });
            }
        }
    }
}

template<typename MM>
void ConvolutionDepthwiseLayerCPU<MM>::backwardComputeImpl()
{
    Workflow& work = mLayer.mNetworkParams.mWorkflow;

    const size_t batchSize = mLayer.mNetworkParams.mWorkflow.getBatchSize();

    const auto& deltas = work.getMemoryManager<MM>()[mLayer.mOutputName.grad()];

    auto deltas3D = deltas.reshape(yato::dims(batchSize, mLayer.mKernelsCount, mLayer.mOutputHeight * mLayer.mOutputWidth));

    const auto& weights = work.getMemoryManager<MM>()[mLayer.mWeightsName];
    auto kernelsWeights2D = weights.reshape(yato::dims(mLayer.mKernelsCount, mLayer.mKernelHeight * mLayer.mKernelWidth));

    // prevDelta
    // if (mNetworkParams.isGradNeeded(mLayer.mInputName))
    {
        auto& prevLayerDelta = work.getMemoryManager<MM>()[mLayer.mInputName.grad()];
        auto prevDeltas3D = prevLayerDelta.reshape(yato::dims(batchSize, mLayer.mInputDepth, mLayer.mInputHeight * mLayer.mInputWidth));
#if defined(_OPENMP)
#pragma omp parallel for
#endif
        for (size_t i = 0; i < batchSize; ++i)
        {
            for (size_t w = 0; w < mLayer.mInputDepth; ++w)
            {
                typename MM::tensor prevDeltaGemmTmp(mLayer.mKernelWidth * mLayer.mKernelHeight * 1 * mLayer.mOutputWidth * mLayer.mOutputHeight, TOMMTYPE(0));

                Common::gemm(CblasTrans,
                             CblasNoTrans,
                             mLayer.mKernelWidth * mLayer.mKernelHeight * 1,
                             mLayer.mOutputWidth * mLayer.mOutputHeight,
                             1,
                             1.0_dt,
                             &kernelsWeights2D[w][0],
                             &deltas3D[i][w][0],
                             0.0_dt,
                             prevDeltaGemmTmp.data());

                Common::col2im(prevDeltaGemmTmp.data(),
                               mLayer.mInputWidth,
                               mLayer.mInputHeight,
                               1,
                               mLayer.mKernelWidth,
                               mLayer.mKernelHeight,
                               mLayer.mStrideW,
                               mLayer.mStrideH,
                               mLayer.mPaddingW,
                               mLayer.mPaddingH,
                               &prevDeltas3D[i][w][0],
                               false,
                               false);
            }
        }
    }

    // gradients weights
    if (!mLayer.mFrozen)
    {
        const auto& inputs = work.getMemoryManager<MM>()[mLayer.mInputName];

        const size_t widthCol = (mLayer.mInputWidth + 2 * mLayer.mPaddingW - mLayer.mKernelWidth) / mLayer.mStrideW + 1;
        const size_t heightCol = (mLayer.mInputHeight + 2 * mLayer.mPaddingH - mLayer.mKernelHeight) / mLayer.mStrideH + 1;

        auto inputs3D = inputs.reshape(yato::dims(batchSize, mLayer.mInputDepth, mLayer.mInputHeight * mLayer.mInputWidth));

        auto& gradWeights = work.getMemoryManager<MM>()[mLayer.mWeightsName.grad()];
        auto gradWeights2D = gradWeights.reshape(yato::dims(mLayer.mKernelsCount, mLayer.mKernelHeight * mLayer.mKernelWidth));

        for (size_t q = 0; q < batchSize; ++q)
        {
            for (size_t w = 0; w < mLayer.mInputDepth; ++w)
            {
                typename MM::tensor im2colMatrix(widthCol * heightCol * 1 * mLayer.mKernelHeight * mLayer.mKernelWidth);

                Common::im2col(&inputs3D[q][w][0],
                               mLayer.mInputWidth,
                               mLayer.mInputHeight,
                               1,
                               mLayer.mKernelWidth,
                               mLayer.mKernelHeight,
                               mLayer.mStrideW,
                               mLayer.mStrideH,
                               mLayer.mPaddingW,
                               mLayer.mPaddingH,
                               im2colMatrix.data());

                Common::gemm(CblasNoTrans,
                             CblasTrans,
                             1,
                             mLayer.mKernelWidth * mLayer.mKernelHeight * 1,
                             mLayer.mOutputWidth * mLayer.mOutputHeight,
                             1.0_dt,
                             &deltas3D[q][w][0],
                             im2colMatrix.data(),
                             1.0_dt,
                             &gradWeights2D[w][0]);
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

template class ConvolutionDepthwiseLayerCPU<MemoryManager>;
template class ConvolutionDepthwiseLayerCPU<MemoryManagerFP16>;

} // namespace raul