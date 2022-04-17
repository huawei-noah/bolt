// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "RepeatInterleaveLayer.h"

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
    return std::make_tuple(j, k, q, i);
}

} // anonymous namespace

namespace raul
{

RepeatInterleaveLayer::RepeatInterleaveLayer(const Name& name, const RepeatInterleaveParams& params, NetworkParameters& networkParameters)
    : BasicLayer(name, "RepeatInterleave", params, networkParameters)
    , mDimension(params.dim)
    , mRepeats(params.mRepeats)
{
    auto prefix = mTypeName + "[" + mName + "::ctor]: ";

    if (mDimension == raul::Dimension::Batch)
    {
        THROW(mTypeName, mName, "only depth, height and width dimensions are supported");
    }

    mNetworkParams.mWorkflow.copyDeclaration(name, mInputs[0], raul::Workflow::Usage::Forward, raul::Workflow::Mode::Read);

    mNetworkParams.mWorkflow.copyDeclaration(mName, mInputs[0], mInputs[0].grad(), DEC_BACK_WRIT_ZERO);

    shape outputShape{ 1u, mNetworkParams.mWorkflow.getDepth(mInputs[0]), mNetworkParams.mWorkflow.getHeight(mInputs[0]), mNetworkParams.mWorkflow.getWidth(mInputs[0]) };
    if (mRepeats.size() == 1)
    {
        if (mDimension == raul::Dimension::Default)
        {
            outputShape[1] = mRepeats[0] * std::accumulate(outputShape.begin(), outputShape.end(), static_cast<size_t>(1), std::multiplies<size_t>());
            outputShape[2] = 1;
            outputShape[3] = 1;
        }
        else
        {
            outputShape[static_cast<size_t>(mDimension)] *= mRepeats[0];
            mRepeats.resize(outputShape[static_cast<size_t>(mDimension)], mRepeats[0]);
        }
    }
    else
    {
        if (outputShape[static_cast<size_t>(mDimension)] != mRepeats.size())
        {
            THROW("RepeatInterleaveLayer", name, "number of repetitions should match dimension size or be equal to 1");
        }
        outputShape[static_cast<size_t>(mDimension)] = std::accumulate(mRepeats.begin(), mRepeats.end(), static_cast<size_t>(0));
    }

    mNetworkParams.mWorkflow.tensorNeeded(mName, mOutputs[0], raul::WShape{ raul::BS(), outputShape[1], outputShape[2], outputShape[3] }, DEC_FORW_WRIT);

    mNetworkParams.mWorkflow.copyDeclaration(name, mOutputs[0], mOutputs[0].grad(), DEC_BACK_READ);
}

void RepeatInterleaveLayer::forwardComputeImpl(NetworkMode)
{

    Tensor& output = mNetworkParams.mMemoryManager[mOutputs[0]];
    const Tensor& input = mNetworkParams.mMemoryManager[mInputs[0]];

    if (mDimension == raul::Dimension::Default)
    {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
        for (size_t i = 0; i < input.size(); ++i)
        {
            for (size_t j = mRepeats[0] * i; j < mRepeats[0] * i + mRepeats[0]; ++j)
            {
                output[j] = input[i];
            }
        }
    }
    else
    {
        // Get 4D view
        const auto input4D = input.get4DView();
        auto output4D = output.get4DView();
        // Need shapes to get indices
        const auto outputShape = output.getShape();

        // Pick chosen dimension
        const auto chosenDimSize = outputShape[static_cast<size_t>(mDimension)];
        // Other dimensions
        std::vector<size_t> otherDims;
        for (size_t i = 0; i < outputShape.dimensions_num(); ++i)
        {
            if (i != static_cast<size_t>(mDimension))
            {
                otherDims.push_back(outputShape[i]);
            }
        }
        size_t indexToRepeat = 0u;
        size_t cumSum = 0u;
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
                        auto [realIO, realJO, realKO, realQO] = reassign(mDimension, i, j, k, q);
                        auto [realII, realJI, realKI, realQI] = reassign(mDimension, indexToRepeat, j, k, q);
                        output4D[realIO][realJO][realKO][realQO] = input4D[realII][realJI][realKI][realQI];
                    }
                }
            }
            if (i - cumSum == mRepeats[indexToRepeat] - 1)
            {
                cumSum += mRepeats[indexToRepeat];
                indexToRepeat += 1;
            }
        }
    }
}

void RepeatInterleaveLayer::backwardComputeImpl()
{

    const Tensor& delta = mNetworkParams.mMemoryManager[mOutputs[0].grad()];

    // if (mNetworkParams.isGradNeeded(mInputs[0]))
    {
        auto& prevLayerNabla = mNetworkParams.mMemoryManager[mInputs[0].grad()];

        if (mDimension == raul::Dimension::Default)
        {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
            for (size_t i = 0; i < prevLayerNabla.size(); ++i)
            {
                for (size_t j = mRepeats[0] * i; j < mRepeats[0] * i + mRepeats[0]; ++j)
                {
                    prevLayerNabla[i] += delta[j];
                }
            }
        }
        else
        {
            // Get 4D view
            const auto delta4D = delta.get4DView();
            auto prevLayerNabla4D = prevLayerNabla.get4DView();
            // Need shapes to get indices
            const auto deltaShape = delta.getShape();

            // Pick chosen dimension
            const auto chosenDimSize = deltaShape[static_cast<size_t>(mDimension)];
            // Other dimensions
            std::vector<size_t> otherDims;
            for (size_t i = 0; i < deltaShape.dimensions_num(); ++i)
            {
                if (i != static_cast<size_t>(mDimension))
                {
                    otherDims.push_back(deltaShape[i]);
                }
            }
            size_t indexToRepeat = 0u;
            size_t cumSum = 0u;
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
                            auto [realIO, realJO, realKO, realQO] = reassign(mDimension, i, j, k, q);
                            auto [realII, realJI, realKI, realQI] = reassign(mDimension, indexToRepeat, j, k, q);
                            prevLayerNabla4D[realII][realJI][realKI][realQI] += delta4D[realIO][realJO][realKO][realQO];
                        }
                    }
                }
                if (i - cumSum == mRepeats[indexToRepeat] - 1)
                {
                    cumSum += mRepeats[indexToRepeat];
                    indexToRepeat += 1;
                }
            }
        }
    }
}

}