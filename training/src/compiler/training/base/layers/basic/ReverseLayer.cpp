// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "ReverseLayer.h"

#include "impl/ReverseLayerCPU.h"

namespace raul
{

ReverseLayer::ReverseLayer(const Name& name, const BasicParams& params, NetworkParameters& networkParameters)
    : BasicLayer(name, "Reverse", params, networkParameters)
{
    auto prefix = mTypeName + "[" + mName + "::ctor]: ";

    DECLARE_IMPL(ReverseLayer, ReverseLayerCPU<MemoryManager>, ReverseLayerCPU<MemoryManagerFP16>)

    if (mInputs.size() != 1 && mInputs.size() != 2)
    {
        THROW(mTypeName, mName, "wrong number of input names");
    }
    if (mOutputs.size() != 1)
    {
        THROW(mTypeName, mName, "wrong number of output names");
    }
    if (mNetworkParams.mWorkflow.getDepth(mInputs[0]) != 1u && mNetworkParams.mWorkflow.getHeight(mInputs[0]) != 1u)
    {
        THROW(mTypeName, mName, "unsupported mode; depth or height should have size = 1");
    }

    mReverseOnly = mInputs.size() == 1;

    mInputName = mInputs[0];
    mOutputName = mOutputs[0];

    mNetworkParams.mWorkflow.copyDeclaration(mName, mInputName, raul::Workflow::Usage::Forward, raul::Workflow::Mode::Read);
    mNetworkParams.mWorkflow.copyDeclaration(mName, mInputName, mInputName.grad(), DEC_BACK_WRIT_ZERO);
    if (!mReverseOnly)
    {
        mNetworkParams.mWorkflow.copyDeclaration(mName, mInputs[1], raul::Workflow::Usage::ForwardAndBackward, raul::Workflow::Mode::Read);
    }
    mNetworkParams.mWorkflow.copyDeclaration(mName, mInputName, mOutputName, DEC_FORW_WRIT_NOMEMOPT);
    mNetworkParams.mWorkflow.copyDeclaration(mName, mInputName, mOutputName.grad(), DEC_BACK_READ);
}

} // namespace raul