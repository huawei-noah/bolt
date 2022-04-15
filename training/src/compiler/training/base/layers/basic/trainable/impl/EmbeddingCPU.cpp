// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "EmbeddingCPU.h"
#include "../Embedding.h"

namespace raul
{

template<typename MM>
void EmbeddingCPU<MM>::forwardComputeImpl(NetworkMode mode)
{
    Workflow& work = mLayer.mNetworkParams.mWorkflow;

    auto& output = work.getMemoryManager<MM>()[mLayer.mOutputName];

    // input tensor will come as indices in float
    mLayer.mIndices.clear();

    const auto& inputs = work.getMemoryManager<MM>()[mLayer.mInputName];

    auto batchSize = inputs.getBatchSize();
    auto depth = inputs.getDepth();
    auto height = inputs.getHeight() * inputs.getWidth();

    auto inputs3D = inputs.reshape(yato::dims(batchSize, depth, height));
    auto outputs3D = output.reshape(yato::dims(batchSize, depth, height * mLayer.mEmbeddingSize));

    const auto& lut = work.getMemoryManager<MM>()[mLayer.mWeightsName];
    const auto lut2D = lut.reshape(yato::dims(mLayer.mDictionarySize, mLayer.mEmbeddingSize));

    if (mode == NetworkMode::Train || mode == NetworkMode::TrainCheckpointed)
    {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
        for (size_t q = 0; q < batchSize; ++q)
        {
            for (size_t c = 0; c < depth; ++c)
            {
                for (size_t i = 0; i < height; ++i)
                {
                    size_t index = static_cast<size_t>(inputs3D[q][c][i] + 0.5);
                    size_t offset = i * mLayer.mEmbeddingSize;
                    if (index == mLayer.mPaddingIdx)
                    {
                        std::fill_n(outputs3D[q][c].begin() + offset, mLayer.mEmbeddingSize, TOMMTYPE(0.0));
                        continue;
                    }
                    if (mLayer.mScaleGradByFreq)
                    {
#if defined(_OPENMP)
#pragma omp critical
#endif
                        {
                            if (mLayer.mIndices.find(index) == mLayer.mIndices.end())
                            {
                                mLayer.mIndices[index] = 1;
                            }
                            else
                            {
                                ++mLayer.mIndices[index];
                            }
                        }
                    }

                    for (size_t e = 0; e < mLayer.mEmbeddingSize; ++e)
                    {
                        outputs3D[q][c][offset + e] = lut2D[index][e] * TOMMTYPE(mLayer.mOutputScale);
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
        for (size_t q = 0; q < batchSize; ++q)
        {
            for (size_t c = 0; c < depth; ++c)
            {
                for (size_t i = 0; i < height; ++i)
                {
                    size_t offset = i * mLayer.mEmbeddingSize;
                    size_t index = static_cast<size_t>(inputs3D[q][c][i] + 0.5);
                    if (index == mLayer.mPaddingIdx)
                    {
                        std::fill_n(outputs3D[q][c].begin() + offset, mLayer.mEmbeddingSize, TOMMTYPE(0.0));
                        continue;
                    }
                    for (size_t e = 0; e < mLayer.mEmbeddingSize; ++e)
                    {
                        outputs3D[q][c][offset + e] = lut2D[index][e] * TOMMTYPE(mLayer.mOutputScale);
                    }
                }
            }
        }
    }
}

template<typename MM>
void EmbeddingCPU<MM>::backwardComputeImpl()
{
    Workflow& work = mLayer.mNetworkParams.mWorkflow;

    const auto& deltas = work.getMemoryManager<MM>()[mLayer.mOutputName.grad()]; // gradient from next layer

    // if (mNetworkParams.mTensorDeclarator.hasGradient(mLayer.mWeightsName))
    if (!mLayer.mFrozen)
    {
        auto& gradWeights = work.getMemoryManager<MM>()[mLayer.mWeightsName.grad()];
        const auto& inputs = work.getMemoryManager<MM>()[mLayer.mInputName];
        auto deltas2D = deltas.reshape(yato::dims(deltas.size() / mLayer.mEmbeddingSize, mLayer.mEmbeddingSize));
        auto grad2D = gradWeights.reshape(yato::dims(mLayer.mDictionarySize, mLayer.mEmbeddingSize));
        size_t el = 0;
        for (auto v = inputs.begin(); v != inputs.end(); ++v, ++el)
        {
            auto index = static_cast<size_t>(*v + 0.5);
            if (index == mLayer.mPaddingIdx)
            {
                continue;
            }

            auto scale = mLayer.mOutputScale;
            if (mLayer.mScaleGradByFreq)
            {
                scale /= static_cast<dtype>(mLayer.mIndices.find(index)->second);
            }

            for (size_t i = 0; i < mLayer.mEmbeddingSize; ++i)
            {
                grad2D[index][i] += TOMMTYPE(scale) * deltas2D[el][i];
            }
        }
    }
}

template class EmbeddingCPU<MemoryManager>;
template class EmbeddingCPU<MemoryManagerFP16>;
} // namespace raul