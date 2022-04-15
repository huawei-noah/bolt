// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "GaussianUpsamplingDistributionLayerCPU.h"
#include "../GaussianUpsamplingDistributionLayer.h"

namespace raul
{

template<typename MM>
void GaussianUpsamplingDistributionLayerCPU<MM>::forwardComputeImpl(NetworkMode)
{
    Workflow& work = mLayer.mNetworkParams.mWorkflow;

    auto& output = work.getMemoryManager<MM>()[mLayer.mOutputName];
    const auto& values = work.getMemoryManager<MM>()[mLayer.mValuesName];
    const auto& loc = work.getMemoryManager<MM>()[mLayer.mLocName];
    const auto& scale = work.getMemoryManager<MM>()[mLayer.mScaleName];

    // Reshape
    const size_t batchSize = work.getBatchSize();
    const size_t valuesLen = values.getBatchSize();
    const size_t height = loc.getHeight();
    auto output3D = output.reshape(yato::dims(batchSize, valuesLen, height));
    auto loc2D = loc.reshape(yato::dims(batchSize, height));
    auto scale2D = scale.reshape(yato::dims(batchSize, height));
#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (size_t i = 0; i < valuesLen; ++i)
    {
        for (size_t j = 0; j < batchSize; ++j)
        {
            for (size_t k = 0; k < height; ++k)
            {
                output3D[j][i][k] = TOMMTYPE(std::exp(
                    std::log(std::exp(-0.5_dt * (values[i] - loc2D[j][k]) * (values[i] - loc2D[j][k]) / scale2D[j][k] / scale2D[j][k]) / std::sqrt(2.0_dt * RAUL_PI * scale2D[j][k] * scale2D[j][k]))));
            }
        }
    }
}

template<typename MM>
void GaussianUpsamplingDistributionLayerCPU<MM>::backwardComputeImpl()
{
    Workflow& work = mLayer.mNetworkParams.mWorkflow;

    // if (mNetworkParams.isGradNeeded(mInputs[1]) || mNetworkParams.isGradNeeded(mInputs[2]))
    {
        const auto& deltas = work.getMemoryManager<MM>()[mLayer.mOutputName.grad()];
        const auto& values = work.getMemoryManager<MM>()[mLayer.mValuesName];
        const auto& loc = work.getMemoryManager<MM>()[mLayer.mLocName];
        const auto& scale = work.getMemoryManager<MM>()[mLayer.mScaleName];

        const size_t batchSize = work.getBatchSize();
        const size_t valuesLen = values.getBatchSize();
        const size_t height = loc.getHeight();
        auto deltas3D = deltas.reshape(yato::dims(batchSize, valuesLen, height));
        auto loc2D = loc.reshape(yato::dims(batchSize, height));
        auto scale2D = scale.reshape(yato::dims(batchSize, height));
        // if (mNetworkParams.isGradNeeded(mInputs[1]))
        {
            auto& locGrad = work.getMemoryManager<MM>()[mLayer.mLocName.grad()];
            auto locGrad2D = locGrad.reshape(yato::dims(batchSize, height));

            for (size_t i = 0; i < valuesLen; ++i)
            {
                for (size_t j = 0; j < batchSize; ++j)
                {
                    for (size_t k = 0; k < height; ++k)
                    {
                        locGrad2D[j][k] +=
                            TOMMTYPE(deltas3D[j][i][k] * 0.398942_dt * (values[i] - loc2D[j][k]) *
                                     std::exp(-0.5_dt * (values[i] - loc2D[j][k]) * (values[i] - loc2D[j][k]) / scale2D[j][k] / scale2D[j][k]) / scale2D[j][k] / scale2D[j][k] / scale2D[j][k]);
                    }
                }
            }
        }
        // if (mNetworkParams.isGradNeeded(mInputs[2])
        {
            auto& scaleGrad = work.getMemoryManager<MM>()[mLayer.mScaleName.grad()];
            auto scaleGrad2D = scaleGrad.reshape(yato::dims(batchSize, height));

            for (size_t i = 0; i < valuesLen; ++i)
            {
                for (size_t j = 0; j < batchSize; ++j)
                {
                    for (size_t k = 0; k < height; ++k)
                    {
                        scaleGrad2D[j][k] +=
                            TOMMTYPE(deltas3D[j][i][k] * (0.398942_dt * (values[i] - loc2D[j][k]) * (values[i] - loc2D[j][k]) / scale2D[j][k] / scale2D[j][k] - 1.0_dt / std::sqrt(2.0_dt * RAUL_PI)) /
                                     scale2D[j][k] / scale2D[j][k] * std::exp(-0.5_dt * (values[i] - loc2D[j][k]) * (values[i] - loc2D[j][k]) / scale2D[j][k] / scale2D[j][k]));
                    }
                }
            }
        }
    }
}

template class GaussianUpsamplingDistributionLayerCPU<MemoryManager>;
template class GaussianUpsamplingDistributionLayerCPU<MemoryManagerFP16>;
} // namespace raul