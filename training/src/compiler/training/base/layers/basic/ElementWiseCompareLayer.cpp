// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "ElementWiseCompareLayer.h"

#include "impl/ElementWiseCompareLayerCPU.h"

namespace raul
{

ElementWiseCompareLayer::ElementWiseCompareLayer(const Name& name, const ElementWiseComparisonLayerParams& params, NetworkParameters& networkParameters)
    : BroadcastingLayer(name, "ElementWiseCompare", params, networkParameters)
    , mBroadcast(params.mBroadcast)
    , mTolerance(params.mTolerance)
    , mCompName(params.mComparator)
{
    if (mInputs.size() != 2)
    {
        THROW("ElementWiseCompareLayer", name, "wrong number of input names");
    }

    if (comparators<dtype>.find(mCompName) == comparators<dtype>.end())
    {
        THROW("ElementWiseCompareLayer", name, "Unknown comparator: " + mCompName);
    }

    DECLARE_IMPL(ElementWiseCompareLayer, ElementWiseCompareLayerCPU<MemoryManager>, ElementWiseCompareLayerCPU<MemoryManagerFP16>)

    mNetworkParams.mWorkflow.copyDeclaration(name, mInputs[0], raul::Workflow::Usage::Forward, raul::Workflow::Mode::Read);

    mNetworkParams.mWorkflow.copyDeclaration(name, mInputs[1], raul::Workflow::Usage::Forward, raul::Workflow::Mode::Read);

    shape outputShape{ 1u, mNetworkParams.mWorkflow.getDepth(mInputs[0]), mNetworkParams.mWorkflow.getHeight(mInputs[0]), mNetworkParams.mWorkflow.getWidth(mInputs[0]) };
    if (mBroadcast)
    {
        shape inputShape{ 1u, mNetworkParams.mWorkflow.getDepth(mInputs[1]), mNetworkParams.mWorkflow.getHeight(mInputs[1]), mNetworkParams.mWorkflow.getWidth(mInputs[1]) };
        std::transform(inputShape.begin(), inputShape.end(), outputShape.begin(), outputShape.begin(), [](auto a, auto b) { return std::max(a, b); });
    }
    mNetworkParams.mWorkflow.tensorNeeded(name, mOutputs[0], raul::WShape{ raul::BS(), outputShape[1], outputShape[2], outputShape[3] }, DEC_FORW_WRIT);

    mEqual = !mCompName.compare(std::string("exact_equal")) || (!mCompName.compare(std::string("equal")) && mTolerance == 0_dt);
    mLess = !mCompName.compare(std::string("exact_less")) || (!mCompName.compare(std::string("less")) && mTolerance == 0_dt);
}

}