// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "RollLayer.h"

namespace raul
{

RollLayer::RollLayer(const Name& name, const RollLayerParams& params, NetworkParameters& networkParameters)
    : BasicLayer(name, "Roll", params, networkParameters)
    , mDimension(params.dim)
    , mShift(params.mShift)
    , mCycled(params.mCycled)
    , mValueToFill(params.mValueToFill)
{
    auto prefix = mTypeName + "[" + mName + "::ctor]: ";

    if (mDimension == raul::Dimension::Default)
    {
        THROW(mTypeName, mName, "default dimension not supported");
    }

    if (mInputs.size() != 1)
    {
        THROW(mTypeName, mName, "wrong number of input names");
    }
    if (mOutputs.size() != 1)
    {
        THROW(mTypeName, mName, "wrong number of output names");
    }

    mNetworkParams.mWorkflow.copyDeclaration(name, mInputs[0], raul::Workflow::Usage::Forward, raul::Workflow::Mode::Read);

    mNetworkParams.mWorkflow.copyDeclaration(name, mInputs[0], mInputs[0].grad(), DEC_BACK_WRIT_ZERO);

    mNetworkParams.mWorkflow.copyDeclaration(name, mInputs[0], mOutputs[0], DEC_FORW_WRIT);

    mNetworkParams.mWorkflow.copyDeclaration(name, mOutputs[0], mOutputs[0].grad(), DEC_BACK_READ);
}

void RollLayer::forwardComputeImpl(NetworkMode)
{
    auto& work = mNetworkParams.mWorkflow;

    if (work.getExecutionTarget() == ExecutionTarget::CPU)
    {
        Tensor& output = mNetworkParams.mMemoryManager[mOutputs[0]];
        const Tensor& input = mNetworkParams.mMemoryManager[mInputs[0]];

        const auto outputShape = output.getShape();
        const auto outputStrides = Common::getStrides(outputShape);

        const auto dimIndex = static_cast<size_t>(mDimension);
        size_t realShift = mShift % outputShape[dimIndex];

#if defined(_OPENMP)
#pragma omp parallel for
#endif
        for (size_t q = 0; q < output.size(); ++q)
        {
            if (realShift == 0)
            {
                output[q] = input[q];
            }
            else
            {
                auto indexes = Common::offsetToIndexes(q, outputStrides);
                size_t oldElIndex = indexes[dimIndex];
                size_t newElIndex = (oldElIndex + realShift) % outputShape[dimIndex];
                indexes[dimIndex] = newElIndex;
                auto newQ = Common::indexesToOffset(indexes, outputStrides);
                if (newElIndex < oldElIndex && !mCycled)
                {
                    output[newQ] = mValueToFill;
                }
                else
                {
                    output[newQ] = input[q];
                }
            }
        }
    }
    else if (work.getExecutionTarget() == ExecutionTarget::CPUFP16)
    {
        auto& output = work.getMemoryManager<MemoryManagerFP16>()[mOutputs[0]];
        const auto& input = work.getMemoryManager<MemoryManagerFP16>()[mInputs[0]];

        const auto outputShape = output.getShape();
        const auto outputStrides = Common::getStrides(outputShape);

        const auto dimIndex = static_cast<size_t>(mDimension);
        size_t realShift = mShift % outputShape[dimIndex];

#if defined(_OPENMP)
#pragma omp parallel for
#endif
        for (size_t q = 0; q < output.size(); ++q)
        {
            if (realShift == 0)
            {
                output[q] = input[q];
            }
            else
            {
                auto indexes = Common::offsetToIndexes(q, outputStrides);
                size_t oldElIndex = indexes[dimIndex];
                size_t newElIndex = (oldElIndex + realShift) % outputShape[dimIndex];
                indexes[dimIndex] = newElIndex;
                auto newQ = Common::indexesToOffset(indexes, outputStrides);
                if (newElIndex < oldElIndex && !mCycled)
                {
                    output[newQ] = TOHTYPE(mValueToFill);
                }
                else
                {
                    output[newQ] = input[q];
                }
            }
        }
    }
    else
    {
        THROW_NONAME("RollLayer", "unsupported execution target");
    }
}

void RollLayer::backwardComputeImpl()
{
    auto& work = mNetworkParams.mWorkflow;

    if (work.getExecutionTarget() == ExecutionTarget::CPU)
    {
        const Tensor& deltas = mNetworkParams.mMemoryManager[mOutputs[0].grad()];

        // if (mNetworkParams.isGradNeeded(mInputs[0]))
        {
            const auto outputShape = deltas.getShape();
            const auto outputStrides = Common::getStrides(outputShape);
            const size_t dimIndex = static_cast<size_t>(mDimension);
            size_t realShift = mShift % outputShape[dimIndex];

            Tensor& prevLayerDelta = mNetworkParams.mMemoryManager[mInputs[0].grad()];

#if defined(_OPENMP)
#pragma omp parallel for
#endif
            for (size_t q = 0; q < prevLayerDelta.size(); ++q)
            {
                if (realShift == 0)
                {
                    prevLayerDelta[q] += deltas[q];
                }
                else
                {
                    auto indexes = Common::offsetToIndexes(q, outputStrides);
                    size_t oldElIndex = indexes[dimIndex];
                    size_t newElIndex = (oldElIndex - realShift + outputShape[dimIndex]) % outputShape[dimIndex];
                    indexes[dimIndex] = newElIndex;
                    auto newQ = Common::indexesToOffset(indexes, outputStrides);
                    prevLayerDelta[newQ] += deltas[q];
                }
            }
        }
    }
    else if (work.getExecutionTarget() == ExecutionTarget::CPUFP16)
    {
        const auto& deltas = work.getMemoryManager<MemoryManagerFP16>()[mOutputs[0].grad()];

        // if (mNetworkParams.isGradNeeded(mInputs[0]))
        {
            const auto outputShape = deltas.getShape();
            const auto outputStrides = Common::getStrides(outputShape);
            const size_t dimIndex = static_cast<size_t>(mDimension);
            size_t realShift = mShift % outputShape[dimIndex];

            auto& prevLayerDelta = work.getMemoryManager<MemoryManagerFP16>()[mInputs[0].grad()];

#if defined(_OPENMP)
#pragma omp parallel for
#endif
            for (size_t q = 0; q < prevLayerDelta.size(); ++q)
            {
                if (realShift == 0)
                {
                    prevLayerDelta[q] += deltas[q];
                }
                else
                {
                    auto indexes = Common::offsetToIndexes(q, outputStrides);
                    size_t oldElIndex = indexes[dimIndex];
                    size_t newElIndex = (oldElIndex - realShift + outputShape[dimIndex]) % outputShape[dimIndex];
                    indexes[dimIndex] = newElIndex;
                    auto newQ = Common::indexesToOffset(indexes, outputStrides);
                    prevLayerDelta[newQ] += deltas[q];
                }
            }
        }
    }
    else
    {
        THROW_NONAME("RollLayer", "unsupported execution target");
    }
}

}