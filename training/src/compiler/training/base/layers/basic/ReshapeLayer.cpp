// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "ReshapeLayer.h"

#include <algorithm>

#include <training/base/common/MemoryManager.h>

#include "impl/ReshapeLayerCPU.h"

namespace raul
{
ReshapeLayer::ReshapeLayer(const Name& name, const ViewParams& params, NetworkParameters& networkParameters)
    : BasicLayer(name, "Reshape", params, networkParameters)
{
    MEASURE_BLOCK(mTypeName + "[" + mName + "::ctor]")

    auto prefix = mTypeName + "[" + mName + "::ctor]: ";

    DECLARE_IMPL(ReshapeLayer, ReshapeLayerCPU<MemoryManager>, ReshapeLayerCPU<MemoryManagerFP16>)

    if (mInputs.size() != 1)
    {
        THROW(mTypeName, mName, "wrong number of input names");
    }
    if (mOutputs.size() != 1)
    {
        THROW(mTypeName, mName, "wrong number of output names");
    }

    mInputName = mInputs[0];
    mOutputName = mOutputs[0];

    const size_t dimensions = 3;

    shape inputShape = shape{ 1u, mNetworkParams.mWorkflow.getDepth(mInputName), mNetworkParams.mWorkflow.getHeight(mInputName), mNetworkParams.mWorkflow.getWidth(mInputName) };

    size_t totalElInInput = inputShape.total_size();

    std::vector<int> sizes = { params.depth, params.height, params.width };
    size_t totalElInParams = 1;
    for (size_t i = 0; i < dimensions; ++i)
    {
        if (sizes[i] == 0)
        {
            THROW(mTypeName, mName, "new sizes must be positve or -1");
        }
        if (sizes[i] > 0)
        {
            totalElInParams *= sizes[i];
        }
    }

    if (totalElInInput % totalElInParams != 0)
    {
        THROW(mTypeName, mName, "bad shape");
    }

    size_t newSize = totalElInParams;
    for (size_t i = 0; i < dimensions; ++i)
    {
        if (sizes[i] < 0)
        {
            sizes[i] = static_cast<int>(totalElInInput / totalElInParams);
            newSize *= sizes[i];
        }
    }

    if (newSize != totalElInInput)
    {
        THROW(mTypeName, mName, "bad shape");
    }

    mNetworkParams.mWorkflow.copyDeclaration(name, mInputName, raul::Workflow::Usage::Forward, raul::Workflow::Mode::Read);

    mNetworkParams.mWorkflow.copyDeclaration(mName, mInputName, mInputName.grad(), DEC_BACK_WRIT_ZERO);

    mNetworkParams.mWorkflow.tensorNeeded(mName, mOutputName, raul::WShape{ raul::BS(), static_cast<size_t>(sizes[0]), static_cast<size_t>(sizes[1]), static_cast<size_t>(sizes[2]) }, DEC_FORW_WRIT);

    mNetworkParams.mWorkflow.tensorNeeded(
        mName, mOutputName.grad(), raul::WShape{ raul::BS(), static_cast<size_t>(sizes[0]), static_cast<size_t>(sizes[1]), static_cast<size_t>(sizes[2]) }, DEC_BACK_READ);
}

} // namespace raul