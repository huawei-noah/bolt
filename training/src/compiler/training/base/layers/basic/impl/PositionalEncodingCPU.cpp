// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "PositionalEncodingCPU.h"
#include "../PositionalEncoding.h"

namespace raul
{

template<typename MM>
PositionalEncodingCPU<MM>::PositionalEncodingCPU(PositionalEncoding& layer)
    : mLayer(layer)
{
    mLayer.mNetworkParams.mWorkflow.tensorNeeded(mLayer.mName, mLayer.mName / "ranges", WShape{ BS(), 1u, 1u, mLayer.mMaxMelLength }, DEC_FORW_WRIT);
}

template<typename MM>
void PositionalEncodingCPU<MM>::initNotBSTensors()
{
    Workflow& work = mLayer.mNetworkParams.mWorkflow;
    auto& memoryManager = work.getMemoryManager<MM>();

    auto& pe = memoryManager[mLayer.mName / "pe"];

    auto data = pe.reshape(yato::dims(mLayer.mMaxLength, mLayer.mModelSize));
    double w = 1;
    double kw = pow(0.0001, 2. / static_cast<dtype>(mLayer.mModelSize));
    for (size_t i = 0; i < mLayer.mModelSize; i += 2)
    {
        for (size_t t = 0; t < mLayer.mMaxLength; ++t)
        {
            data[t][i] = TOMMTYPE(sin(w * TODTYPE(t)));
            data[t][i + 1] = TOMMTYPE(cos(w * TODTYPE(t)));
        }
        w *= kw;
    }
}

template<typename MM>
void durations_range(const typename MM::tensor& durations, typename MM::tensor& ranges, typename MM::type fillValue)
{
    size_t maxMelLength = ranges.getWidth();
    size_t w = durations.getWidth();
    auto dData = durations.reshape(yato::dims(durations.getBatchSize(), w));
    auto rData = ranges.reshape(yato::dims(durations.getBatchSize(), maxMelLength));
    for (size_t q = 0; q < durations.getBatchSize(); ++q)
    {
        size_t index = 0;
        for (size_t i = 0; i < w; ++i)
        {
            auto el = static_cast<size_t>(dData[q][i]);
            for (size_t j = 0; j < el; ++j)
            {
                if (index >= maxMelLength)
                {
                    break;
                }
                rData[q][index++] = TOMMTYPE(j);
            }
            if (index >= maxMelLength)
            {
                break;
            }
        }
        if (index < maxMelLength - 1)
        {
            std::fill(rData[q].begin() + index, rData[q].end(), fillValue);
        }
    }
}

template<typename MM>
void PositionalEncodingCPU<MM>::forwardComputeImpl(NetworkMode)
{
    auto& memoryManager = mLayer.mNetworkParams.mWorkflow.getMemoryManager<MM>();
    auto& output = memoryManager[mLayer.mOutputName];

    const auto& inputs = memoryManager[mLayer.mInputName];

    if (!mLayer.mDurationEncoding)
    {
        auto inData3D = inputs.reshape(yato::dims(inputs.getBatchSize() * inputs.getDepth(), inputs.getHeight(), mLayer.mModelSize));
        auto outData3D = output.reshape(inData3D.dimensions());

        const auto& pe = memoryManager[mLayer.mName / "pe"];
        const auto data = pe.reshape(yato::dims(pe.getHeight(), pe.getWidth()));

        size_t N = inData3D.dimensions()[0];
        size_t height = inputs.getHeight();

#if defined(_OPENMP)
#pragma omp parallel for
#endif
        for (size_t e = 0; e < N; ++e)
        {
            for (size_t t = 0; t < height; ++t)
            {
                for (size_t i = 0; i < mLayer.mModelSize; ++i)
                {
                    outData3D[e][t][i] = inData3D[e][t][i] + data[t][i];
                }
            }
        }
    }
    else
    {
        auto& ranges = memoryManager[mLayer.mName / "ranges"];
        auto batchSize = inputs.getBatchSize();
        auto width = mLayer.mMaxMelLength;
        auto ranges2D = ranges.reshape(batchSize, width);
        auto outputs3D = output.reshape(batchSize, mLayer.mMaxMelLength, mLayer.mModelSize);
        auto lut2D = memoryManager[mLayer.mName / "pe"].reshape(mLayer.mMaxLength, mLayer.mModelSize);
        durations_range<MM>(inputs, ranges, TOMMTYPE(mLayer.mMaxLength - 1));

        for (size_t q = 0; q < batchSize; ++q)
        {
            for (size_t i = 0; i < width; ++i)
            {
                size_t index = static_cast<size_t>(ranges2D[q][i] + 0.5);
                std::copy(lut2D[index].begin(), lut2D[index].end(), outputs3D[q][i].begin());
            }
        }
    }
}

template<typename MM>
void PositionalEncodingCPU<MM>::backwardComputeImpl()
{
    // if (mNetworkParams.isGradNeeded(mInputName))
    {
        if (!mLayer.mDurationEncoding)
        {
            auto& memoryManager = mLayer.mNetworkParams.mWorkflow.getMemoryManager<MM>();
            auto& prevLayerDelta = memoryManager[mLayer.mInputName.grad()];
            // just copy
            prevLayerDelta += memoryManager[mLayer.mOutputName.grad()];
        }
    }
}

template class PositionalEncodingCPU<MemoryManager>;
template class PositionalEncodingCPU<MemoryManagerFP16>;
} // namespace raul
