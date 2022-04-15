// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "SoftMaxActivationCPU.h"
#include "../SoftMaxActivation.h"

#include <algorithm>

#include <training/base/impl/ImplFactory.h>

namespace
{

std::tuple<size_t, size_t, size_t, size_t> reassign(raul::Dimension dim, size_t i, size_t j, size_t k, size_t q)
{
    if (dim == raul::Dimension::Depth)
    {
        return std::make_tuple(j, i, k, q);
    }
    if (dim == raul::Dimension::Height)
    {
        return std::make_tuple(j, k, i, q);
    }
    return std::make_tuple(i, j, k, q);
}

bool reg1 = raul::TheImplFactory::Instance().regCPUFP32<raul::SoftMaxActivation, raul::SoftMaxActivationCPU<raul::MemoryManager>>();
bool reg2 = raul::TheImplFactory::Instance().regCPUFP16<raul::SoftMaxActivation, raul::SoftMaxActivationCPU<raul::MemoryManagerFP16>>();
} // anonymous namespace

namespace raul
{

template<typename MM>
void SoftMaxActivationCPU<MM>::forwardComputeImpl(NetworkMode)
{
    Workflow& work = mLayer.mNetworkParams.mWorkflow;

    const auto& inputs = work.getMemoryManager<MM>()[mLayer.mInputName];
    auto& output = work.getMemoryManager<MM>()[mLayer.mOutputName];

    const size_t batchSize = inputs.getBatchSize();

    if (mLayer.mDimension == Dimension::Default)
    {
        for (size_t q = 0; q < batchSize; ++q)
        {
            size_t batchOffset = q * mLayer.mInputsCount;

            auto sum = static_cast<typename MM::type>(0.0_dt);
            auto max = (*std::max_element(inputs.begin() + batchOffset, inputs.begin() + batchOffset + mLayer.mInputsCount));

            for (size_t i = 0; i < mLayer.mInputsCount; ++i)
            {
                output[batchOffset + i] = static_cast<typename MM::type>(std::exp(TODTYPE(inputs[batchOffset + i] - max)));
                sum += output[batchOffset + i];
            }

            for (size_t i = 0; i < mLayer.mInputsCount; ++i)
            {
                output[batchOffset + i] /= sum;
            }
        }
    }
    else if (mLayer.mDimension == Dimension::Width)
    {
        size_t size = batchSize * inputs.getDepth() * inputs.getHeight();
        auto input2D = inputs.reshape(yato::dims(size, inputs.getWidth()));
        auto output2D = output.reshape(yato::dims(size, inputs.getWidth()));
#if defined(_OPENMP)
#pragma omp parallel for
#endif
        for (size_t q = 0; q < size; ++q)
        {
            auto sum = 0.0_dt;
            auto max = (*std::max_element(inputs.begin() + q * inputs.getWidth(), inputs.begin() + (q + 1) * inputs.getWidth()));

            for (size_t i = 0; i < inputs.getWidth(); ++i)
            {
                output2D[q][i] = static_cast<typename MM::type>(std::exp(TODTYPE(input2D[q][i]) - max));
                sum += output2D[q][i];
            }

            for (size_t i = 0; i < inputs.getWidth(); ++i)
            {
                output2D[q][i] = static_cast<typename MM::type>(TODTYPE(output2D[q][i]) / sum);
            }
        }
    }
    else
    {
        // Output Strides
        auto outputShape = output.getShape();
        const auto outputStrides = Common::getStrides(outputShape);
        // Divisor strides
        outputShape[static_cast<size_t>(mLayer.mDimension)] = 1;
        const auto divStrides = Common::getStrides(outputShape);
        // Reshape intput in 4D view
        const auto input4D = inputs.get4DView();
        auto inputShape = inputs.getShape();
        // Pick chosen dimension
        const auto chosenDimSize = inputShape[static_cast<size_t>(mLayer.mDimension)];
        // Delete it
        std::vector<size_t> otherDims;
        size_t otherDimsSize = 1u;
        for (size_t i = 0; i < inputShape.dimensions_num(); ++i)
        {
            if (i != static_cast<size_t>(mLayer.mDimension))
            {
                otherDims.push_back(inputShape[i]);
                otherDimsSize *= inputShape[i];
            }
        }
        // Store divisors
        std::vector<raul::dtype> divisors(otherDimsSize, 0.0_dt);
        // Store indices mapping
        std::vector<size_t> divIndices(output.size(), 0u);
        // Main loop
#if defined(_OPENMP)
#pragma omp parallel for
#endif
        for (size_t j = 0; j < otherDims[0]; ++j)
        {
            for (size_t k = 0; k < otherDims[1]; ++k)
            {
                for (size_t q = 0; q < otherDims[2]; ++q)
                {
                    auto max = 0.0_dt;
                    for (size_t i = 0; i < chosenDimSize; ++i)
                    {
                        auto [realI, realJ, realK, realQ] = reassign(mLayer.mDimension, i, j, k, q);
                        max = std::max(max, TODTYPE(input4D[realI][realJ][realK][realQ]));
                    }
                    for (size_t i = 0; i < chosenDimSize; ++i)
                    {
                        // Rearrange indices in proper way
                        auto [realI, realJ, realK, realQ] = reassign(mLayer.mDimension, i, j, k, q);
                        // Find offset in output and divisors
                        raul::shape outputIndices{ realI, realJ, realK, realQ };
                        const auto outputOffset = Common::indexesToOffset(outputIndices, outputStrides);
                        // Fill output with exp(input)
                        auto val = std::exp(TODTYPE(input4D[realI][realJ][realK][realQ]) - max);
                        output[outputOffset] = TOMMTYPE(val);
                        // Calculate reduce_sum(exp(input), axis=mDimension)
                        outputIndices[static_cast<size_t>(mLayer.mDimension)] = 1;
                        const auto divOffset = Common::indexesToOffset(outputIndices, divStrides);
                        divisors[divOffset] += val;
                        divIndices[outputOffset] = divOffset;
                    }
                }
            }
        }
        // Normalize values
#if defined(_OPENMP)
#pragma omp parallel for
#endif
        for (size_t n = 0; n < output.size(); ++n)
        {
            output[n] = TOMMTYPE(TODTYPE(output[n]) / divisors[divIndices[n]]);
        }
    }
}

template<typename MM>
void SoftMaxActivationCPU<MM>::backwardComputeImpl()
{
    Workflow& work = mLayer.mNetworkParams.mWorkflow;

    const auto& output = work.getMemoryManager<MM>()[mLayer.mOutputName];
    const size_t batchSize = output.getBatchSize();

    ////if (mLayer.mNetworkParams.isGradNeeded(mLayer.mInputName))
    {
        const auto& deltas = work.getMemoryManager<MM>()[mLayer.mOutputName.grad()];
        auto& prevLayerDelta = work.getMemoryManager<MM>()[mLayer.mInputName.grad()];

        if (mLayer.mDimension == Dimension::Default)
        {
            for (size_t q = 0; q < batchSize; ++q)
            {
                size_t batchOffset = q * mLayer.mInputsCount;

                for (size_t i = 0; i < mLayer.mInputsCount; ++i)
                {
                    auto sum = static_cast<typename MM::type>(0.0_dt);

                    for (size_t j = 0; j < mLayer.mInputsCount; ++j)
                    {
                        sum += deltas[batchOffset + j] *
                               ((j == i) ? output[batchOffset + j] * (static_cast<typename MM::type>(1.0_dt) - output[batchOffset + j]) : -output[batchOffset + i] * output[batchOffset + j]);
                    }

                    prevLayerDelta[batchOffset + i] += sum;
                }
            }
        }
        else if (mLayer.mDimension == Dimension::Width)
        {
            size_t size = batchSize * deltas.getDepth() * deltas.getHeight();

            auto deltas2D = deltas.reshape(yato::dims(size, deltas.getWidth()));
            auto prevLayerDelta2D = prevLayerDelta.reshape(yato::dims(size, deltas.getWidth()));
            auto output2D = output.reshape(yato::dims(size, deltas.getWidth()));
#if defined(_OPENMP)
#pragma omp parallel for
#endif
            for (size_t q = 0; q < size; ++q)
            {
                for (size_t i = 0; i < deltas.getWidth(); ++i)
                {
                    dtype sum = 0.0_dt;

                    for (size_t j = 0; j < deltas.getWidth(); ++j)
                    {
                        sum += TODTYPE(deltas2D[q][j]) * ((j == i) ? TODTYPE(output2D[q][j]) * (1.0_dt - TODTYPE(output2D[q][j])) : -TODTYPE(output2D[q][i]) * TODTYPE(output2D[q][j]));
                    }

                    prevLayerDelta2D[q][i] += static_cast<typename MM::type>(sum);
                }
            }
        }
        else
        {
            auto deltas4D = deltas.get4DView();
            auto output4D = output.get4DView();
            auto prevLayerDelta4D = prevLayerDelta.get4DView();
            const auto outputShape = output.getShape();
            const auto chosenDimSize = outputShape[static_cast<size_t>(mLayer.mDimension)];
            std::vector<size_t> otherDims;
            for (size_t i = 0; i < outputShape.dimensions_num(); ++i)
            {
                if (i != static_cast<size_t>(mLayer.mDimension))
                {
                    otherDims.push_back(outputShape[i]);
                }
            }
#if defined(_OPENMP)
#pragma omp parallel for
#endif
            for (size_t j = 0; j < otherDims[0]; ++j)
            {
                for (size_t k = 0; k < otherDims[1]; ++k)
                {
                    for (size_t q = 0; q < otherDims[2]; ++q)
                    {
                        for (size_t i = 0; i < chosenDimSize; ++i)
                        {
                            raul::dtype sum = 0.0_dt;
                            auto [realI1, realJ1, realK1, realQ1] = reassign(mLayer.mDimension, i, j, k, q);
                            for (size_t n = 0; n < chosenDimSize; ++n)
                            {
                                auto [realI2, realJ2, realK2, realQ2] = reassign(mLayer.mDimension, n, j, k, q);
                                sum += TODTYPE(deltas4D[realI2][realJ2][realK2][realQ2]) *
                                       ((n == i) ? TODTYPE(output4D[realI1][realJ1][realK1][realQ1]) * (1.0_dt - TODTYPE(output4D[realI1][realJ1][realK1][realQ1]))
                                                 : -TODTYPE(output4D[realI1][realJ1][realK1][realQ1]) * TODTYPE(output4D[realI2][realJ2][realK2][realQ2]));
                            }
                            prevLayerDelta4D[realI1][realJ1][realK1][realQ1] += static_cast<typename MM::type>(sum);
                        }
                    }
                }
            }
        }
    }
}

template class SoftMaxActivationCPU<MemoryManager>;
template class SoftMaxActivationCPU<MemoryManagerFP16>;

} // namespace raul