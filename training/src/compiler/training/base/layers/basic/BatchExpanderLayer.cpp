// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "BatchExpanderLayer.h"

#include "impl/BatchExpanderLayerCPU.h"

#include <algorithm>

#include <training/base/common/MemoryManager.h>

namespace raul
{

BatchExpanderLayer::BatchExpanderLayer(const Name& name, const ViewParams& params, NetworkParameters& networkParameters)
    : BasicLayer(name, "BatchExpander", params, networkParameters)
{
    auto prefix = "BatchExpanderLayer[" + name + "::ctor]: ";
    if (mInputs.size() != 1)
    {
        THROW(mTypeName, mName, "wrong number of input names");
    }

    if (mInputs[0].empty())
    {
        THROW(mTypeName, mName, "empty output name");
    }

    if (mOutputs.size() != 1)
    {
        THROW(mTypeName, mName, "wrong number of input names");
    }

    if (mOutputs[0].empty())
    {
        THROW(mTypeName, mName, "empty output name");
    }

    if (mNetworkParams.mWorkflow.getShape(mInputs[0]).isBSDependent())
    {
        THROW(mTypeName, mName, "input tensor already has batch dimension");
    }

    DECLARE_IMPL(BatchExpanderLayer, BatchExpanderLayerCPU<MemoryManager>, BatchExpanderLayerCPU<MemoryManagerFP16>)

    size_t width = mNetworkParams.mWorkflow.getWidth(mInputs[0]);
    size_t height = mNetworkParams.mWorkflow.getHeight(mInputs[0]);
    size_t depth = mNetworkParams.mWorkflow.getDepth(mInputs[0]);

    mNetworkParams.mWorkflow.copyDeclaration(mName, mInputs[0], Workflow::Usage::Forward, Workflow::Mode::Read);
    mNetworkParams.mWorkflow.tensorNeeded(name, mOutputs[0], WShape{ BS(), depth, height, width }, DEC_FORW_WRIT);
    mNetworkParams.mWorkflow.copyDeclaration(mName, mOutputs[0], mOutputs[0].grad(), DEC_BACK_READ);

    if (mNetworkParams.mWorkflow.isTensorTrainable(mInputs[0]))
    {
        mNetworkParams.mWorkflow.copyDeclaration(mName, mInputs[0], mInputs[0].grad(), DEC_TRAINABLE_GRAD);
    }
    else
    {
        mNetworkParams.mWorkflow.copyDeclaration(mName, mInputs[0], mInputs[0].grad(), DEC_BACK_WRIT_ZERO);
    }
}

} // namespace raul