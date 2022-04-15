// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "SlicerLayer.h"

#include "impl/SlicerLayerCPU.h"

#include <algorithm>

#include <training/base/common/MemoryManager.h>

namespace raul
{

SlicerLayer::SlicerLayer(const Name& name, const SlicingParams& params, NetworkParameters& networkParameters)
    : BasicLayer(name, "Slicer", params, networkParameters)
    , mDirection(params.dim)
    , mBackwardTmpBufferName("TempStorageForIntermediateCalculations")
{
    auto prefix = mTypeName + "[" + mName + "::ctor]: ";

    if (mInputs.size() != 1)
    {
        THROW(mTypeName, mName, "wrong number of input names");
    }

    size_t sliceCount = params.sliceSize.size();
    if (sliceCount == 0)
    {
        sliceCount = mOutputs.size();
    }

    if (mOutputs.size() != sliceCount)
    {
        THROW(mTypeName, mName, "wrong number of output names");
    }

    DECLARE_IMPL(SlicerLayer, SlicerLayerCPU<MemoryManager>, SlicerLayerCPU<MemoryManagerFP16>)

    mDimIndex = 0;
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

    yato::dimensionality<3U, size_t> inputShape(mNetworkParams.mWorkflow.getDepth(mInputs[0]), mNetworkParams.mWorkflow.getHeight(mInputs[0]), mNetworkParams.mWorkflow.getWidth(mInputs[0]));

    if (params.sliceSize.empty())
    {
        if (inputShape[mDimIndex] % sliceCount != 0)
        {
            THROW(mTypeName, mName, "wrong number of slices");
        }
        mSlices = std::vector<size_t>(sliceCount, inputShape[mDimIndex] / sliceCount);
    }
    else
    {
        auto cnt_1 = std::count_if(params.sliceSize.begin(), params.sliceSize.end(), [](int a) { return a == -1; });
        if (cnt_1 > 1)
        {
            THROW(mTypeName, mName, "only one slice with -1 is allowed");
        }

        size_t size = inputShape[mDimIndex];
        auto acc = std::accumulate(params.sliceSize.begin(), params.sliceSize.end(), size_t(0), [](size_t a, int b) { return b > 0 ? a + b : a; });
        if (acc > size)
        {
            THROW(mTypeName, mName, "sum of slices (" + std::to_string(acc) + ") is greater then input tensor size (" + std::to_string(size) + ")");
        }
        size_t remainder = size - acc;
        if (remainder == 0 && cnt_1 != 0)
        {
            THROW(mTypeName, mName, "nothing left for slice with -1");
        }

        for (auto& s : params.sliceSize)
        {
            if (s == 0)
            {
                THROW(mTypeName, mName, "slice of size 0 is not allowed");
            }
            mSlices.push_back(s == -1 ? remainder : size_t(s));
        }
    }

    mNetworkParams.mWorkflow.copyDeclaration(mName, mInputs[0], Workflow::Usage::Forward, Workflow::Mode::Read);
    mNetworkParams.mWorkflow.copyDeclaration(mName, mInputs[0], mInputs[0].grad(), DEC_BACK_WRIT_ZERO);

    auto shape = inputShape;
    for (size_t i = 0; i < mOutputs.size(); ++i)
    {
        shape[mDimIndex] = mSlices[i];

        mNetworkParams.mWorkflow.tensorNeeded(mName, mOutputs[i], raul::WShape{ raul::BS(), shape[0], shape[1], shape[2] }, DEC_FORW_WRIT);
        mNetworkParams.mWorkflow.copyDeclaration(mName, mOutputs[i], mOutputs[i].grad(), DEC_BACK_READ);
    }
}

} // namespace raul