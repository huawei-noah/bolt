// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef ARG_EXTREMUM_LAYER_H
#define ARG_EXTREMUM_LAYER_H

#include <training/base/common/Common.h>
#include <training/base/layers/BasicLayer.h>

namespace
{

struct ArgMax
{
    static bool compare(const raul::dtype x, const raul::dtype y) { return y >= x; }
    static raul::dtype getBound() { return -std::numeric_limits<raul::dtype>::infinity(); }
};

struct ArgMin
{
    static bool compare(const raul::dtype x, const raul::dtype y) { return y <= x; }
    static raul::dtype getBound() { return std::numeric_limits<raul::dtype>::infinity(); }
};

}

namespace raul
{

/**
 * @brief ArgExtremum (ArgMax and ArgMin) Layer
 *
 * First output is indices, second (if needed) - values.
 * No gradient if only indices required (copy tf and torch behavior).
 *
 */
template<typename T>
class ArgExtremumLayer : public BasicLayer
{
  public:
    ArgExtremumLayer(const Name& name, const BasicParamsWithDim& params, NetworkParameters& networkParameters);

    ArgExtremumLayer(ArgExtremumLayer&&) = default;
    ArgExtremumLayer(const ArgExtremumLayer&) = delete;
    ArgExtremumLayer& operator=(const ArgExtremumLayer&) = delete;

    void forwardComputeImpl(NetworkMode mode) override;
    void backwardComputeImpl() override;

  private:
    Dimension mDimension;
    T mComparator;
    bool mValuesRequired;
    std::vector<raul::dtype> mValues;
};

template<typename T>
ArgExtremumLayer<T>::ArgExtremumLayer(const Name& name, const BasicParamsWithDim& params, NetworkParameters& networkParameters)
    : BasicLayer(name, "ArgExtremum", params, networkParameters)
    , mDimension(params.dim)
{
    auto prefix = mTypeName + "[" + name + "::ctor]: ";
    if (mInputs.size() != 1)
    {
        THROW(mTypeName, mName, "wrong number of input names");
    }

    if (mOutputs.size() > 2)
    {
        THROW(mTypeName, mName, "wrong number of output names");
    }

    if (mDimension == raul::Dimension::Default)
    {
        THROW(mTypeName, mName, "wrong dimension specified");
    }

    const auto& inputName = mInputs[0];
    mValuesRequired = mOutputs.size() == 2;

    mNetworkParams.mWorkflow.copyDeclaration(name, inputName, raul::Workflow::Usage::Forward, raul::Workflow::Mode::Read);

    shape outputShape{ 1u, mNetworkParams.mWorkflow.getDepth(mInputs[0]), mNetworkParams.mWorkflow.getHeight(mInputs[0]), mNetworkParams.mWorkflow.getWidth(mInputs[0]) };
    if (mDimension == raul::Dimension::Batch)
    {
        mNetworkParams.mWorkflow.tensorNeeded(name, mOutputs[0], raul::WShape{ 1u, outputShape[1], outputShape[2], outputShape[3] }, DEC_FORW_WRIT);
    }
    else
    {
        for (size_t i = 1; i < outputShape.dimensions_num(); i++)
        {
            if (static_cast<size_t>(mDimension) == i)
            {
                outputShape[i] = 1u;
            }
        }
        mNetworkParams.mWorkflow.tensorNeeded(name, mOutputs[0], raul::WShape{ raul::BS(), outputShape[1], outputShape[2], outputShape[3] }, DEC_FORW_WRIT);
    }

    if (mValuesRequired)
    {
        mNetworkParams.mWorkflow.copyDeclaration(name, inputName, inputName.grad(), DEC_BACK_WRIT_ZERO);
        mNetworkParams.mWorkflow.copyDeclaration(name, mOutputs[0], mOutputs[0], DEC_BACK_READ);
        mNetworkParams.mWorkflow.copyDeclaration(name, mOutputs[0], mOutputs[1], DEC_FORW_WRIT);
        mNetworkParams.mWorkflow.copyDeclaration(name, mOutputs[1], mOutputs[1].grad(), DEC_BACK_READ);
    }
}

template<typename T>
void ArgExtremumLayer<T>::forwardComputeImpl(NetworkMode)
{
    auto& work = mNetworkParams.mWorkflow;

    if (work.getExecutionTarget() == ExecutionTarget::CPU)
    {
        Tensor& indices = mNetworkParams.mMemoryManager[mOutputs[0]];

        const Tensor& input = mNetworkParams.mMemoryManager[mInputs[0]];

        // Storage for values
        mValues.resize(indices.size());
        std::fill(mValues.begin(), mValues.end(), mComparator.getBound());

        // Need for dimensionality hints
        const auto inputStrides = Common::getStrides(input.getShape());
        const auto outputStrides = Common::getStrides(indices.getShape());

        for (size_t q = 0; q < input.size(); ++q)
        {
            auto inputIndexes = Common::offsetToIndexes(q, inputStrides);
            const auto index = inputIndexes[static_cast<size_t>(mDimension)];
            inputIndexes[static_cast<size_t>(mDimension)] = 1;

            // Calculate offset in output tensor
            const auto outputOffset = Common::indexesToOffset(inputIndexes, outputStrides);
            if (mComparator.compare(mValues[outputOffset], input[q]))
            {
                mValues[outputOffset] = input[q];
                indices[outputOffset] = TODTYPE(index);
            }
        }

        // If values are required
        if (mValuesRequired)
        {
            Tensor& values = mNetworkParams.mMemoryManager[mOutputs[1]];
#if defined(_OPENMP)
#pragma omp parallel for
#endif
            for (size_t q = 0; q < indices.size(); ++q)
            {
                values[q] = mValues[q];
            }
        }
    }
    else if (work.getExecutionTarget() == ExecutionTarget::CPUFP16)
    {
        auto& indices = work.template getMemoryManager<MemoryManagerFP16>()[mOutputs[0]];

        const auto& input = work.template getMemoryManager<MemoryManagerFP16>()[mInputs[0]];

        // Storage for values
        mValues.resize(indices.size());
        std::fill(mValues.begin(), mValues.end(), mComparator.getBound());

        // Need for dimensionality hints
        const auto inputStrides = Common::getStrides(input.getShape());
        const auto outputStrides = Common::getStrides(indices.getShape());

        for (size_t q = 0; q < input.size(); ++q)
        {
            auto inputIndexes = Common::offsetToIndexes(q, inputStrides);
            const auto index = inputIndexes[static_cast<size_t>(mDimension)];
            inputIndexes[static_cast<size_t>(mDimension)] = 1;

            // Calculate offset in output tensor
            const auto outputOffset = Common::indexesToOffset(inputIndexes, outputStrides);
            if (mComparator.compare(mValues[outputOffset], TODTYPE(input[q])))
            {
                mValues[outputOffset] = TODTYPE(input[q]);
                indices[outputOffset] = TOHTYPE(index);
            }
        }

        // If values are required
        if (mValuesRequired)
        {
            auto& values = work.template getMemoryManager<MemoryManagerFP16>()[mOutputs[1]];
#if defined(_OPENMP)
#pragma omp parallel for
#endif
            for (size_t q = 0; q < indices.size(); ++q)
            {
                values[q] = TOHTYPE(mValues[q]);
            }
        }
    }
    else
    {
        THROW_NONAME("ArgExtremumLayer", "unsupported execution target");
    }
}

template<typename T>
void ArgExtremumLayer<T>::backwardComputeImpl()
{
    if (!mValuesRequired)
    {
        // Tensorflow and pytorch behavior.
        // No gradient if only indices required.
        return;
    }

    auto& work = mNetworkParams.mWorkflow;

    if (work.getExecutionTarget() == ExecutionTarget::CPU)
    {
        const Tensor& delta = mNetworkParams.mMemoryManager[mOutputs[1].grad()];
        const Tensor& indices = mNetworkParams.mMemoryManager[mOutputs[0]];

        // if (mNetworkParams.isGradNeeded(mInputs[0]))
        {
            auto& in_nabla_tensor = mNetworkParams.mMemoryManager[mInputs[0].grad()];

            const auto inputStrides = Common::getStrides(in_nabla_tensor.getShape());
            const auto outputStrides = Common::getStrides(indices.getShape());
#if defined(_OPENMP)
#pragma omp parallel for
#endif
            for (size_t j = 0; j < in_nabla_tensor.size(); ++j)
            {
                auto inputIndexes = Common::offsetToIndexes(j, inputStrides);
                const auto index = inputIndexes[static_cast<size_t>(mDimension)];
                inputIndexes[static_cast<size_t>(mDimension)] = 1;

                const auto outputOffset = Common::indexesToOffset(inputIndexes, outputStrides);
                in_nabla_tensor[j] += TODTYPE((index == static_cast<size_t>(indices[outputOffset]))) * delta[outputOffset];
            }
        }
    }
    else if (work.getExecutionTarget() == ExecutionTarget::CPUFP16)
    {
        const auto& delta = work.template getMemoryManager<MemoryManagerFP16>()[mOutputs[1].grad()];
        const auto& indices = work.template getMemoryManager<MemoryManagerFP16>()[mOutputs[0]];

        // if (mNetworkParams.isGradNeeded(mInputs[0]))
        {
            auto& in_nabla_tensor = work.template getMemoryManager<MemoryManagerFP16>()[mInputs[0].grad()];

            const auto inputStrides = Common::getStrides(in_nabla_tensor.getShape());
            const auto outputStrides = Common::getStrides(indices.getShape());
#if defined(_OPENMP)
#pragma omp parallel for
#endif
            for (size_t j = 0; j < in_nabla_tensor.size(); ++j)
            {
                auto inputIndexes = Common::offsetToIndexes(j, inputStrides);
                const auto index = inputIndexes[static_cast<size_t>(mDimension)];
                inputIndexes[static_cast<size_t>(mDimension)] = 1;

                const auto outputOffset = Common::indexesToOffset(inputIndexes, outputStrides);
                in_nabla_tensor[j] += TOHTYPE((index == static_cast<size_t>(indices[outputOffset]))) * delta[outputOffset];
            }
        }
    }
    else
    {
        THROW_NONAME("ArgExtremumLayer", "unsupported execution target");
    }
}

}

#endif