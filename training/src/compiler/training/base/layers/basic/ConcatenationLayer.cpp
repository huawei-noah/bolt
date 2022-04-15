// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "ConcatenationLayer.h"

#include "impl/ConcatenationLayerCPU.h"

#include <algorithm>

#include <training/base/common/MemoryManager.h>

namespace raul
{

size_t ConcatenationLayer::mGlobalInputMaxSize = 0;

ConcatenationLayer::ConcatenationLayer(const Name& name, const BasicParamsWithDim& params, NetworkParameters& networkParameters)
    : BasicLayer(name, "Concatenation", params, networkParameters)
    , mDirection(params.dim)
    , mCurrentInputMaxSize(0)
{
    auto prefix = mTypeName + "[" + mName + "::ctor]: ";
    if (mOutputs.size() != 1)
    {
        THROW(mTypeName, mName, "wrong number of output names");
    }

    DECLARE_IMPL(ConcatenationLayer, ConcatenationLayerCPU<MemoryManager>, ConcatenationLayerCPU<MemoryManagerFP16>)

    switch (mDirection)
    {
        case Dimension::Depth:
            mDimIndex = 0;
            break;
        case Dimension::Height:
            mDimIndex = 1;
            break;
        case Dimension::Width:
            mDimIndex = 2;
            break;
        default:
            THROW(mTypeName, mName, "unsupported dim");
    }

    yato::dimensionality<3U, size_t> shape(mNetworkParams.mWorkflow.getDepth(mInputs[0]), mNetworkParams.mWorkflow.getHeight(mInputs[0]), mNetworkParams.mWorkflow.getWidth(mInputs[0]));
    mCurrentInputMaxSize = shape[0] * shape[1] * shape[2];
    for (size_t i = 1; i < mInputs.size(); ++i)
    {
        yato::dimensionality<3U, size_t> inputShape(mNetworkParams.mWorkflow.getDepth(mInputs[i]), mNetworkParams.mWorkflow.getHeight(mInputs[i]), mNetworkParams.mWorkflow.getWidth(mInputs[i]));
        mCurrentInputMaxSize = std::max(mCurrentInputMaxSize, inputShape[0] * inputShape[1] * inputShape[2]);

        for (size_t k = 0; k < 3; ++k)
        {
            if (k == mDimIndex)
            {
                continue;
            }
            if (shape[k] != inputShape[k])
            {
                THROW(mTypeName, mName, "inconsistent input shapes (" + mInputs[0] + " " + Conversions::toString(shape) + " vs " + mInputs[i] + " " + Conversions::toString(inputShape) + ")");
            }
        }
        shape[mDimIndex] += inputShape[mDimIndex];
    }

    for (const auto& input : mInputs)
    {
        mNetworkParams.mWorkflow.copyDeclaration(mName, input, Workflow::Usage::Forward, Workflow::Mode::Read);
        mNetworkParams.mWorkflow.copyDeclaration(mName, input, input.grad(), DEC_BACK_WRIT_ZERO);
    }

    mNetworkParams.mWorkflow.tensorNeeded(mName, mOutputs[0], raul::WShape{ raul::BS(), shape[0], shape[1], shape[2] }, DEC_FORW_WRIT);
    mNetworkParams.mWorkflow.copyDeclaration(mName, mOutputs[0], mOutputs[0].grad(), DEC_BACK_READ);
}

} // namespace raul