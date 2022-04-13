// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "ReduceArithmeticLayerCPU.h"
#include "../ReduceArithmeticLayer.h"
#include "../ReduceBatchMeanLayer.h"
#include "../ReduceMeanLayer.h"
#include "../ReduceNonZeroLayer.h"
#include "../ReduceStdLayer.h"
#include "../ReduceSumLayer.h"

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
    if (dim == raul::Dimension::Width)
    {
        return std::make_tuple(j, k, q, i);
    }
    return std::make_tuple(i, j, k, q);
}

bool reg1 = raul::TheImplFactory::Instance().regCPUFP32<raul::ReduceArithmeticLayer, raul::ReduceArithmeticLayerCPU<raul::MemoryManager>>();
bool reg2 = raul::TheImplFactory::Instance().regCPUFP16<raul::ReduceArithmeticLayer, raul::ReduceArithmeticLayerCPU<raul::MemoryManagerFP16>>();

bool reg3 = raul::TheImplFactory::Instance().regCPUFP32<raul::ReduceSumLayer, raul::ReduceArithmeticLayerCPU<raul::MemoryManager>>();
bool reg4 = raul::TheImplFactory::Instance().regCPUFP16<raul::ReduceSumLayer, raul::ReduceArithmeticLayerCPU<raul::MemoryManagerFP16>>();

bool reg5 = raul::TheImplFactory::Instance().regCPUFP32<raul::ReduceBatchMeanLayer, raul::ReduceArithmeticLayerCPU<raul::MemoryManager>>();
bool reg6 = raul::TheImplFactory::Instance().regCPUFP16<raul::ReduceBatchMeanLayer, raul::ReduceArithmeticLayerCPU<raul::MemoryManagerFP16>>();

bool reg7 = raul::TheImplFactory::Instance().regCPUFP32<raul::ReduceNonZeroLayer, raul::ReduceArithmeticLayerCPU<raul::MemoryManager>>();
bool reg8 = raul::TheImplFactory::Instance().regCPUFP16<raul::ReduceNonZeroLayer, raul::ReduceArithmeticLayerCPU<raul::MemoryManagerFP16>>();

bool reg9 = raul::TheImplFactory::Instance().regCPUFP32<raul::ReduceMeanLayer, raul::ReduceArithmeticLayerCPU<raul::MemoryManager>>();
bool reg10 = raul::TheImplFactory::Instance().regCPUFP16<raul::ReduceMeanLayer, raul::ReduceArithmeticLayerCPU<raul::MemoryManagerFP16>>();

bool reg11 = raul::TheImplFactory::Instance().regCPUFP32<raul::ReduceStdLayer, raul::ReduceArithmeticLayerCPU<raul::MemoryManager>>();
bool reg12 = raul::TheImplFactory::Instance().regCPUFP16<raul::ReduceStdLayer, raul::ReduceArithmeticLayerCPU<raul::MemoryManagerFP16>>();
} // anonymous namespace

namespace raul
{

template<typename MM>
void ReduceArithmeticLayerCPU<MM>::forwardComputeImpl(NetworkMode)
{
    Workflow& work = mLayer.mNetworkParams.mWorkflow;

    auto& output = work.getMemoryManager<MM>()[mLayer.mOutputs[0]];
    const auto& input = work.getMemoryManager<MM>()[mLayer.mInputs[0]];

    output = TOMMTYPE(0.0);

    if (mLayer.mDim == Dimension::Default)
    {
        dtype sum = 0.0_dt;
        dtype nonZeroElems = 0.0_dt;
        //#if defined(_OPENMP)
        //#pragma omp parallel for reduction(+ : sum, nonZeroElems)
        //#endif
        for (size_t q = 0; q < input.size(); ++q)
        {
            sum += input[q];
            if (input[q] != 0.0_dt)
            {
                nonZeroElems += 1.0_dt;
            }
        }
        mLayer.mDiv = mLayer.mOperation == "sum" || mLayer.mOperation == "count_non_zero_elems" ? 1.0_dt : mLayer.mOperation == "batch_mean" ? TODTYPE(input.getBatchSize()) : TODTYPE(input.size());
        if (mLayer.mOperation == "count_non_zero_elems")
        {
            output[0] = TOMMTYPE(nonZeroElems);
        }
        else
        {
            output[0] = TOMMTYPE(sum / mLayer.mDiv);
        }
        if (mLayer.mOperation == "std")
        {
            mLayer.mMeanValues.resize(1);
            dtype std = 0.0_dt;
#if defined(_OPENMP)
#pragma omp parallel for reduction(+ : std)
#endif
            for (size_t q = 0; q < input.size(); ++q)
            {
                std += TODTYPE((input[q] - output[0]) * (input[q] - output[0]));
            }
            // Remember mean value
            mLayer.mMeanValues[0] = TODTYPE(output[0]);
            output[0] = TOMMTYPE(std::sqrt(std / (mLayer.mDiv - 1)));
        }
    }
    else
    {
        const auto outputStrides = Common::getStrides(output.getShape());
        const auto input4D = input.get4DView();
        auto inputShape = input.getShape();
        // Pick chosen dimension
        const auto chosenDimSize = inputShape[static_cast<size_t>(mLayer.mDim)];
        // Delete it
        std::vector<size_t> otherDims;
        for (size_t i = 0; i < inputShape.dimensions_num(); ++i)
        {
            if (i != static_cast<size_t>(mLayer.mDim))
            {
                otherDims.push_back(inputShape[i]);
            }
        }
        // Divisor if operation == "mean" or "std"
        mLayer.mDiv = mLayer.mOperation == "sum" || mLayer.mOperation == "count_non_zero_elems" ? 1.0_dt : TODTYPE(chosenDimSize);
        bool needToCountNonZeroElems = mLayer.mOperation == "count_non_zero_elems";
        // Main loop
        for (size_t i = 0; i < chosenDimSize; ++i)
        {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
            for (size_t j = 0; j < otherDims[0]; ++j)
            {
                for (size_t k = 0; k < otherDims[1]; ++k)
                {
                    for (size_t q = 0; q < otherDims[2]; ++q)
                    {
                        // Rearrange indices in proper way
                        auto [realI, realJ, realK, realQ] = reassign(mLayer.mDim, i, j, k, q);
                        // Find offset in output
                        shape outputIndices{ realI, realJ, realK, realQ };
                        outputIndices[static_cast<size_t>(mLayer.mDim)] = 1;
                        const auto offset = Common::indexesToOffset(outputIndices, outputStrides);
                        if (needToCountNonZeroElems)
                        {
                            if (input4D[realI][realJ][realK][realQ] != TOMMTYPE(0.0_dt))
                            {
                                output[offset] += TOMMTYPE(1.0_dt);
                            }
                        }
                        else
                        {
                            output[offset] += TOMMTYPE(TODTYPE(input4D[realI][realJ][realK][realQ]) / mLayer.mDiv);
                        }
                    }
                }
            }
        }
        if (mLayer.mOperation == "std")
        {
            // Copy mean values to another vector
            mLayer.mMeanValues.resize(output.size());
            std::copy(output.begin(), output.end(), mLayer.mMeanValues.begin());
            output = TOMMTYPE(0.0_dt);
            for (size_t i = 0; i < chosenDimSize; ++i)
            {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
                for (size_t j = 0; j < otherDims[0]; ++j)
                {
                    for (size_t k = 0; k < otherDims[1]; ++k)
                    {
                        for (size_t q = 0; q < otherDims[2]; ++q)
                        {
                            // Rearrange indices in proper way
                            auto [realI, realJ, realK, realQ] = reassign(mLayer.mDim, i, j, k, q);
                            // Find offset in output
                            shape outputIndices{ realI, realJ, realK, realQ };
                            outputIndices[static_cast<size_t>(mLayer.mDim)] = 1;
                            const auto offset = Common::indexesToOffset(outputIndices, outputStrides);
                            output[offset] +=
                                (TOMMTYPE(mLayer.mMeanValues[offset]) - input4D[realI][realJ][realK][realQ]) * (TOMMTYPE(mLayer.mMeanValues[offset]) - input4D[realI][realJ][realK][realQ]);
                        }
                    }
                }
            }
#if defined(_OPENMP)
#pragma omp parallel for
#endif
            for (size_t n = 0; n < output.size(); ++n)
            {
                output[n] = TOMMTYPE(std::sqrt(TODTYPE(output[n]) / (mLayer.mDiv - 1)));
            }
        }
    }
}

template<typename MM>
void ReduceArithmeticLayerCPU<MM>::backwardComputeImpl()
{
    Workflow& work = mLayer.mNetworkParams.mWorkflow;

    const auto& output = work.getMemoryManager<MM>()[mLayer.mOutputs[0]];
    auto& delta = work.getMemoryManager<MM>()[mLayer.mOutputs[0].grad()];

    // if (mNetworkParams.isGradNeeded(mLayer.mInputs[0]))
    {
        auto& in_nabla_tensor = work.getMemoryManager<MM>()[mLayer.mInputs[0].grad()];
        const auto inputStrides = Common::getStrides(in_nabla_tensor.getShape());
        const auto outputStrides = Common::getStrides(output.getShape());

        if (mLayer.mOperation == "std")
        {
            const auto& input = work.getMemoryManager<MM>()[mLayer.mInputs[0]];
            for (size_t q = 0; q < in_nabla_tensor.size(); q++)
            {
                auto inputIndexes = Common::offsetToIndexes(q, inputStrides);
                // Calculate index in output
                if (mLayer.mDim == Dimension::Default)
                {
                    std::fill(inputIndexes.begin(), inputIndexes.end(), 1);
                }
                else
                {
                    inputIndexes[static_cast<size_t>(mLayer.mDim)] = 1;
                }
                const auto outputOffset = Common::indexesToOffset(inputIndexes, outputStrides);
                in_nabla_tensor[q] += TOMMTYPE(TODTYPE(delta[outputOffset]) * (TODTYPE(input[q]) - mLayer.mMeanValues[outputOffset]) / TODTYPE(mLayer.mDiv - 1) / TODTYPE(output[outputOffset]));
            }
        }
        else
        {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
            for (size_t q = 0; q < in_nabla_tensor.size(); q++)
            {
                auto inputIndexes = Common::offsetToIndexes(q, inputStrides);
                // Calculate index in output
                if (mLayer.mDim == Dimension::Default)
                {
                    std::fill(inputIndexes.begin(), inputIndexes.end(), 1);
                }
                else
                {
                    inputIndexes[static_cast<size_t>(mLayer.mDim)] = 1;
                }
                const auto outputOffset = Common::indexesToOffset(inputIndexes, outputStrides);
                in_nabla_tensor[q] += TOMMTYPE(TODTYPE(delta[outputOffset]) / mLayer.mDiv);
            }
        }
    }
}

template class ReduceArithmeticLayerCPU<MemoryManager>;
template class ReduceArithmeticLayerCPU<MemoryManagerFP16>;
} // namespace raul