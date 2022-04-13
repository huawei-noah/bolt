// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "AveragePoolLayerCPU.h"
#include "../AveragePoolLayer.h"

#include <training/base/impl/ImplFactory.h>
namespace
{
bool reg1 = raul::TheImplFactory::Instance().regCPUFP32<raul::AveragePoolLayer, raul::AveragePoolLayerCPU<raul::MemoryManager>>();
bool reg2 = raul::TheImplFactory::Instance().regCPUFP16<raul::AveragePoolLayer, raul::AveragePoolLayerCPU<raul::MemoryManagerFP16>>();
} // anonymous namespace

namespace raul
{

template<typename MM>
AveragePoolLayerCPU<MM>::AveragePoolLayerCPU(AveragePoolLayer& layer)
    : mLayer(layer)
{
}

template<typename MM>
void AveragePoolLayerCPU<MM>::forwardComputeImpl(NetworkMode)
{
    auto& work = mLayer.mNetworkParams.mWorkflow;

    const size_t batchSize = work.getBatchSize();

    auto& output = work.getMemoryManager<MM>()[mLayer.mOutputName];

    const auto& inputs = work.getMemoryManager<MM>()[mLayer.mInputName];

    auto inputs3D = inputs.reshape(yato::dims(batchSize, mLayer.mInputDepth, mLayer.mInputHeight * mLayer.mInputWidth));
    auto outputs3D = output.reshape(yato::dims(batchSize, mLayer.mInputDepth, mLayer.mOutputHeight * mLayer.mOutputWidth));

    const dtype reciprocalKernelSize = 1.0_dt / static_cast<dtype>(mLayer.mKernelHeight * mLayer.mKernelWidth);

    for (size_t b = 0; b < batchSize; ++b)
    {
        for (size_t k = 0; k < mLayer.mInputDepth; ++k)
        {
            for (size_t i = 0; i < mLayer.mOutputHeight; ++i)
            {
                for (size_t j = 0; j < mLayer.mOutputWidth; ++j)
                {
                    auto out_index = j + mLayer.mOutputWidth * i;
                    auto sum = TOMMTYPE(0.0_dt);
                    for (size_t n = 0; n < mLayer.mKernelHeight; ++n)
                    {
                        for (size_t m = 0; m < mLayer.mKernelWidth; ++m)
                        {
                            auto cur_h = i * mLayer.mStrideH + n - mLayer.mPaddingH;
                            auto cur_w = j * mLayer.mStrideW + m - mLayer.mPaddingW;
                            if (cur_h < mLayer.mInputHeight && cur_w < mLayer.mInputWidth)
                            {
                                auto index = cur_w + mLayer.mInputWidth * (cur_h);
                                sum += inputs3D[b][k][index];
                            }
                        }
                    }
                    outputs3D[b][k][out_index] = sum * TOMMTYPE(reciprocalKernelSize);
                }
            }
        }
    }
}

template<typename MM>
void AveragePoolLayerCPU<MM>::backwardComputeImpl()
{
    auto& work = mLayer.mNetworkParams.mWorkflow;

    // if (mLayer.mNetworkParams.isGradNeeded(mLayer.mLayer.mInputName))
    {
        const size_t batchSize = work.getBatchSize();

        auto& prevLayerDelta = work.getMemoryManager<MM>()[mLayer.mInputName.grad()];

        const auto& deltas = work.getMemoryManager<MM>()[mLayer.mOutputName.grad()];

        auto deltas3D = deltas.reshape(yato::dims(batchSize, mLayer.mInputDepth, mLayer.mOutputHeight * mLayer.mOutputWidth));
        auto prevDeltas3D = prevLayerDelta.reshape(yato::dims(batchSize, mLayer.mInputDepth, mLayer.mInputHeight * mLayer.mInputWidth));

        const dtype reciprocalKernelSize = 1.0_dt / static_cast<dtype>(mLayer.mKernelHeight * mLayer.mKernelWidth);

        for (size_t batch = 0; batch < batchSize; ++batch)
        {
            for (size_t c = 0; c < mLayer.mInputDepth; ++c)
            {
                for (size_t i = 0; i < mLayer.mOutputHeight; ++i)
                {
                    for (size_t j = 0; j < mLayer.mOutputWidth; ++j)
                    {
                        auto out_index = j + mLayer.mOutputWidth * i;
                        for (size_t n = 0; n < mLayer.mKernelHeight; ++n)
                        {
                            for (size_t m = 0; m < mLayer.mKernelWidth; ++m)
                            {
                                auto cur_h = i * mLayer.mStrideH + n - mLayer.mPaddingH;
                                auto cur_w = j * mLayer.mStrideW + m - mLayer.mPaddingW;
                                if (cur_h < mLayer.mInputHeight && cur_w < mLayer.mInputWidth)
                                {
                                    auto index = cur_w + mLayer.mInputWidth * (cur_h);
                                    prevDeltas3D[batch][c][index] += deltas3D[batch][c][out_index] * TOMMTYPE(reciprocalKernelSize);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

template class AveragePoolLayerCPU<MemoryManager>;
template class AveragePoolLayerCPU<MemoryManagerFP16>;

} // namespace raul
