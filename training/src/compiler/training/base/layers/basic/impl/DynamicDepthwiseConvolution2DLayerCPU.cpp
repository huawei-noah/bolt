// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "DynamicDepthwiseConvolution2DLayerCPU.h"
#include "../DynamicDepthwiseConvolution2DLayer.h"

namespace raul
{

template<typename MM>
DynamicDepthwiseConvolution2DLayerCPU<MM>::DynamicDepthwiseConvolution2DLayerCPU(DynamicDepthwiseConvolution2DLayer& layer)
    : mLayer(layer)
{
}

template<typename MM>
void DynamicDepthwiseConvolution2DLayerCPU<MM>::forwardComputeImpl(NetworkMode)
{
    auto& memoryManager = mLayer.mNetworkParams.mWorkflow.getMemoryManager<MM>();
    auto& output = memoryManager[mLayer.mOutputName];
    const auto& input = memoryManager[mLayer.mInputName];
    const auto& filter = memoryManager[mLayer.mFiltersName];

    const size_t batchSize = mLayer.mNetworkParams.mWorkflow.getBatch(mLayer.mInputName);

    if (mLayer.mInputDepth == 0u)
    {
        mLayer.mInputDepth = input.getWidth();
        mLayer.mInChannels = filter.getHeight();
    }

    auto output4D = output.reshape(yato::dims(batchSize, mLayer.mOutputHeight, mLayer.mOutputWidth, mLayer.mInChannels * mLayer.mChannelMultiplier));
    auto input4D = input.reshape(yato::dims(batchSize, mLayer.mInputHeight, mLayer.mInputWidth, mLayer.mInChannels));
    auto filter4D = filter.reshape(yato::dims(mLayer.mFilterHeight, mLayer.mFilterWidth, mLayer.mInChannels, mLayer.mChannelMultiplier));

#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (size_t b = 0; b < batchSize; ++b)
    {
        for (size_t i = 0; i < mLayer.mOutputHeight; ++i)
        {
            for (size_t j = 0; j < mLayer.mOutputWidth; ++j)
            {
                for (size_t k = 0; k < mLayer.mInChannels; ++k)
                {
                    for (size_t q = 0; q < mLayer.mChannelMultiplier; ++q)
                    {
                        for (size_t di = 0; di < mLayer.mFilterHeight; ++di)
                        {
                            for (size_t dj = 0; dj < mLayer.mFilterWidth; ++dj)
                            {
                                output4D[b][i][j][k * mLayer.mChannelMultiplier + q] += filter4D[di][dj][k][q] * input4D[b][i + di][j + dj][k];
                            }
                        }
                    }
                }
            }
        }
    }
}

template<typename MM>
void DynamicDepthwiseConvolution2DLayerCPU<MM>::backwardComputeImpl()
{
    auto& memoryManager = mLayer.mNetworkParams.mWorkflow.getMemoryManager<MM>();

    const size_t batchSize = mLayer.mNetworkParams.mWorkflow.getBatch(mLayer.mInputName);

    const auto& deltas = memoryManager[mLayer.mOutputName.grad()];
    const auto& filters = memoryManager[mLayer.mFiltersName];
    const auto& input = memoryManager[mLayer.mInputName];

    const auto deltas4D = deltas.reshape(yato::dims(batchSize, mLayer.mOutputHeight, mLayer.mOutputWidth, mLayer.mInChannels * mLayer.mChannelMultiplier));
    // if (mLayer.mNetworkParams.isGradNeeded(mLayer.mInputName))
    // {
    auto& inputDelta = memoryManager[mLayer.mInputName.grad()];

    auto inputDelta4D = inputDelta.reshape(yato::dims(batchSize, mLayer.mInputHeight, mLayer.mInputWidth, mLayer.mInChannels));
    const auto filter4D = filters.reshape(yato::dims(mLayer.mFilterHeight, mLayer.mFilterWidth, mLayer.mInChannels, mLayer.mChannelMultiplier));

#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (size_t b = 0; b < batchSize; ++b)
    {
        for (size_t i = 0; i < mLayer.mOutputHeight; ++i)
        {
            for (size_t j = 0; j < mLayer.mOutputWidth; ++j)
            {
                for (size_t k = 0; k < mLayer.mInChannels; ++k)
                {
                    for (size_t q = 0; q < mLayer.mChannelMultiplier; ++q)
                    {
                        for (size_t di = 0; di < mLayer.mFilterHeight; ++di)
                        {
                            for (size_t dj = 0; dj < mLayer.mFilterWidth; ++dj)
                            {
                                inputDelta4D[b][i + di][j + dj][k] += filter4D[di][dj][k][q] * deltas4D[b][i][j][k * mLayer.mChannelMultiplier + q];
                            }
                        }
                    }
                }
            }
        }
    }
    //}

    // if (mLayer.mNetworkParams.isGradNeeded(mLayer.mFiltersName))
    // {
    auto& filtersDelta = memoryManager[mLayer.mFiltersName.grad()];

    auto filtersDelta4D = filtersDelta.reshape(yato::dims(mLayer.mFilterHeight, mLayer.mFilterWidth, mLayer.mInChannels, mLayer.mChannelMultiplier));
    const auto input4D = input.reshape(yato::dims(batchSize, mLayer.mInputHeight, mLayer.mInputWidth, mLayer.mInChannels));

    for (size_t b = 0; b < batchSize; ++b)
    {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
        for (size_t k = 0; k < mLayer.mInChannels; ++k)
        {
            for (size_t q = 0; q < mLayer.mChannelMultiplier; ++q)
            {
                for (size_t di = 0; di < mLayer.mFilterHeight; ++di)
                {
                    for (size_t dj = 0; dj < mLayer.mFilterWidth; ++dj)
                    {
                        for (size_t i = 0; i < mLayer.mOutputHeight; ++i)
                        {
                            for (size_t j = 0; j < mLayer.mOutputWidth; ++j)
                            {
                                filtersDelta4D[di][dj][k][q] += input4D[b][i + di][j + dj][k] * deltas4D[b][i][j][k * mLayer.mChannelMultiplier + q];
                            }
                        }
                    }
                }
            }
        }
    }
    // }
}

template class DynamicDepthwiseConvolution2DLayerCPU<MemoryManager>;
template class DynamicDepthwiseConvolution2DLayerCPU<MemoryManagerFP16>;

} // namespace raul