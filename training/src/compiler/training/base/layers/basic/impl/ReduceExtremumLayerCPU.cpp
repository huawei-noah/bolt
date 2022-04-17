// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "ReduceExtremumLayerCPU.h"
#include "../ReduceExtremumLayer.h"

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

} // anonymous

namespace raul
{

template<template<typename> typename Comparator, typename MM>
void ReduceExtremumLayerCPU<Comparator, MM>::forwardComputeImpl(NetworkMode)
{
    Workflow& work = mLayer.mNetworkParams.mWorkflow;

    auto& output = work.getMemoryManager<MM>()[mLayer.mOutputs[0]];
    const auto& input = work.getMemoryManager<MM>()[mLayer.mInputs[0]];
    std::fill(output.begin(), output.end(), mComparator.getBound());

    if (mLayer.mDim == raul::Dimension::Default)
    {
        mLayer.mCountExtremums.resize(1);
        for (size_t q = 0; q < input.size(); ++q)
        {
            if (mComparator.compare(output[0], input[q]))
            {
                if (output[0] == input[q])
                {
                    mLayer.mCountExtremums[0]++;
                }
                else
                {
                    output[0] = input[q];
                    mLayer.mCountExtremums[0] = 1;
                }
            }
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
        size_t otherDimsSize = 1;
        for (size_t i = 0; i < inputShape.dimensions_num(); ++i)
        {
            if (i != static_cast<size_t>(mLayer.mDim))
            {
                otherDims.push_back(inputShape[i]);
                otherDimsSize *= inputShape[i];
            }
        }
        mLayer.mCountExtremums.resize(otherDimsSize);
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
                        raul::shape outputIndices{ realI, realJ, realK, realQ };
                        outputIndices[static_cast<size_t>(mLayer.mDim)] = 1;
                        const auto offset = Common::indexesToOffset(outputIndices, outputStrides);
                        if (mComparator.compare(output[offset], input4D[realI][realJ][realK][realQ]))
                        {
                            if (output[offset] == input4D[realI][realJ][realK][realQ])
                            {
                                mLayer.mCountExtremums[offset]++;
                            }
                            else
                            {
                                output[offset] = input4D[realI][realJ][realK][realQ];
                                mLayer.mCountExtremums[offset] = 1;
                            }
                        }
                    }
                }
            }
        }
    }
}

template<template<typename> typename Comparator, typename MM>
void ReduceExtremumLayerCPU<Comparator, MM>::backwardComputeImpl()
{
    Workflow& work = mLayer.mNetworkParams.mWorkflow;

    const auto& delta = work.getMemoryManager<MM>()[mLayer.mOutputs[0].grad()];
    const auto& output = work.getMemoryManager<MM>()[mLayer.mOutputs[0]];
    const auto& input = work.getMemoryManager<MM>()[mLayer.mInputs[0]];

    // if (mNetworkParams.isGradNeeded(mLayer.mInputs[0]))
    {
        auto& in_nabla_tensor = work.getMemoryManager<MM>()[mLayer.mInputs[0].grad()];

        if (mLayer.mDim == raul::Dimension::Default)
        {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
            for (size_t j = 0; j < in_nabla_tensor.size(); ++j)
            {
                in_nabla_tensor[j] = TOMMTYPE(input[j] == output[0]) * delta[0] / TOMMTYPE(mLayer.mCountExtremums[0]);
            }
        }
        else
        {
            const auto inputStrides = Common::getStrides(in_nabla_tensor.getShape());
            const auto outputStrides = Common::getStrides(output.getShape());
#if defined(_OPENMP)
#pragma omp parallel for
#endif
            for (size_t j = 0; j < in_nabla_tensor.size(); ++j)
            {
                auto inputIndexes = Common::offsetToIndexes(j, inputStrides);
                inputIndexes[static_cast<size_t>(mLayer.mDim)] = 1;
                const auto outputOffset = Common::indexesToOffset(inputIndexes, outputStrides);
                in_nabla_tensor[j] = TOMMTYPE(input[j] == output[outputOffset]) * delta[outputOffset] / TOMMTYPE(mLayer.mCountExtremums[outputOffset]);
            }
        }
    }
}

template class ReduceExtremumLayerCPU<Max, MemoryManager>;
template class ReduceExtremumLayerCPU<Min, MemoryManager>;
template class ReduceExtremumLayerCPU<Max, MemoryManagerFP16>;
template class ReduceExtremumLayerCPU<Min, MemoryManagerFP16>;
} // namespace raul