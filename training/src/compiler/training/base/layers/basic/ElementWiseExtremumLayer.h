// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef ELEMENT_WISE_EXTREMUM_LAYER_H
#define ELEMENT_WISE_EXTREMUM_LAYER_H

#include <training/base/common/Common.h>
#include <training/base/layers/BroadcastingLayer.h>

#include <type_traits>

namespace
{

struct ElementWiseMaxHelper
{
    static bool compare(const raul::dtype x, const raul::dtype y) { return y >= x; }
    static raul::dtype getBound() { return -std::numeric_limits<raul::dtype>::infinity(); }
};

struct ElementWiseMinHelper
{
    static bool compare(const raul::dtype x, const raul::dtype y) { return y <= x; }
    static raul::dtype getBound() { return std::numeric_limits<raul::dtype>::infinity(); }
};

}

namespace raul
{

/**
 * @brief Template for ElementWiseMax and ElementWiseMin layers
 * @tparam T
 */
template<typename T>
class ElementWiseExtremumLayer : public BroadcastingLayer
{
  public:
    ElementWiseExtremumLayer(const Name& name, const ElementWiseLayerParams& params, NetworkParameters& networkParameters);

    ElementWiseExtremumLayer(ElementWiseExtremumLayer&&) = default;
    ElementWiseExtremumLayer(const ElementWiseExtremumLayer&) = delete;
    ElementWiseExtremumLayer& operator=(const ElementWiseExtremumLayer&) = delete;

    void forwardComputeImpl(NetworkMode mode) override;
    void backwardComputeImpl() override;

  private:
    bool mBroadcast;
    std::vector<size_t> mIndexes;
    T mComparator;

    Name mTmpBufferName;

    size_t mDepth;
    size_t mHeight;
    size_t mWidth;
};

template<typename T>
ElementWiseExtremumLayer<T>::ElementWiseExtremumLayer(const Name& name, const ElementWiseLayerParams& params, NetworkParameters& networkParameters)
    : BroadcastingLayer(name, "ElementWiseExtremum", params, networkParameters)
    , mBroadcast(params.mBroadcast)
    , mTmpBufferName("TempStorageForIntermediateCalculations")
{
    for (const auto& input : mInputs)
    {
        mNetworkParams.mWorkflow.copyDeclaration(name, input, raul::Workflow::Usage::Forward, raul::Workflow::Mode::Read);

        mNetworkParams.mWorkflow.copyDeclaration(name, input, input.grad(), DEC_BACK_WRIT_ZERO);
    }

    shape outputShape{ 1u, mNetworkParams.mWorkflow.getDepth(mInputs[0]), mNetworkParams.mWorkflow.getHeight(mInputs[0]), mNetworkParams.mWorkflow.getWidth(mInputs[0]) };
    if (mBroadcast)
    {
        for (size_t j = 1; j < mInputs.size(); ++j)
        {
            shape inputShape{ 1u, mNetworkParams.mWorkflow.getDepth(mInputs[j]), mNetworkParams.mWorkflow.getHeight(mInputs[j]), mNetworkParams.mWorkflow.getWidth(mInputs[j]) };
            std::transform(inputShape.begin(), inputShape.end(), outputShape.begin(), outputShape.begin(), [](auto a, auto b) { return std::max(a, b); });
        }
    }

    mNetworkParams.mWorkflow.tensorNeeded(name, mOutputs[0], raul::WShape{ BS(), outputShape[1], outputShape[2], outputShape[3] }, DEC_FORW_WRIT);
    mNetworkParams.mWorkflow.copyDeclaration(name, mOutputs[0], mOutputs[0].grad(), DEC_BACK_READ);

    mDepth = mNetworkParams.mWorkflow.getDepth(mOutputs[0]);
    mHeight = mNetworkParams.mWorkflow.getHeight(mOutputs[0]);
    mWidth = mNetworkParams.mWorkflow.getWidth(mOutputs[0]);
}

template<typename T>
void ElementWiseExtremumLayer<T>::forwardComputeImpl(NetworkMode)
{
    determineBroadcastFlags();

    if (!mBroadcast && std::any_of(mBroadcastQuery.begin(), mBroadcastQuery.end(), [](const auto& needToBroadcast) { return needToBroadcast; }))
    {
        THROW("ElementWiseExtremumLayer", mName, "input size mismatch");
    }

    if (mNetworkParams.mWorkflow.getExecutionTarget() == ExecutionTarget::CPU)
    {
        Tensor& output = mNetworkParams.mMemoryManager[mOutputs[0]];
        std::fill(output.begin(), output.end(), mComparator.getBound());
        mIndexes.resize(output.size());

        // Comparison
        for (size_t q = 0; q < mInputs.size(); ++q)
        {
            const Tensor& input = mNetworkParams.mMemoryManager[mInputs[q]];

            if (mBroadcastQuery[q])
            {
                auto input_viewer = input.getBroadcastedViewer(output.getShape());
#if defined(_OPENMP)
#pragma omp parallel for
#endif
                for (size_t i = 0; i < output.size(); ++i)
                {
                    if (mComparator.compare(output[i], input_viewer[i]))
                    {
                        output[i] = input_viewer[i];
                        mIndexes[i] = q;
                    }
                }
            }
            else
            {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
                for (size_t i = 0; i < output.size(); ++i)
                {
                    if (mComparator.compare(output[i], input[i]))
                    {
                        output[i] = input[i];
                        mIndexes[i] = q;
                    }
                }
            }
        }
    }
    else
    {
        THROW(mTypeName, mName, "unsupported execution target");
    }
}

template<typename T>
void ElementWiseExtremumLayer<T>::backwardComputeImpl()
{
    if (mNetworkParams.mWorkflow.getExecutionTarget() == ExecutionTarget::CPU)
    {
        const Tensor& delta = mNetworkParams.mMemoryManager[mOutputs[0].grad()];

        for (size_t i = 0; i < mInputs.size(); i++)
        {
            const auto input_name = mInputs[i];
            // if (mNetworkParams.isGradNeeded(input_name))
            {
                auto& in_nabla_tensor = mNetworkParams.mMemoryManager[input_name.grad()];
                if (mBroadcastQuery[i])
                {
                    auto in_nabla = in_nabla_tensor.getBroadcastedViewer(delta.getShape());
                    for (size_t j = 0; j < mIndexes.size(); j++)
                    {
                        if (mIndexes[j] == i)
                        {
                            in_nabla[j] += delta[j];
                        }
                    }
                }
                else
                {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
                    for (size_t j = 0; j < mIndexes.size(); j++)
                    {
                        if (mIndexes[j] == i)
                        {
                            in_nabla_tensor[j] += delta[j];
                        }
                    }
                }
            }
        }
    }
    else
    {
        THROW(mTypeName, mName, "unsupported execution target");
    }
}

} // raul namespace

#endif
