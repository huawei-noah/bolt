// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "ReLUActivation.h"

#include <training/base/common/Common.h>
#include <training/base/common/MemoryManager.h>

#include "impl/ReLUActivationImpl.h"

namespace raul
{

ReLUActivation::ReLUActivation(const Name& name, const BasicParams& params, NetworkParameters& networkParameters)
    : ReLUActivation(name, "ReLUActivation", params, networkParameters)
{
}

ReLUActivation::ReLUActivation(const Name& name, const std::string& typeName, const BasicParams& params, NetworkParameters& networkParameters)
    : BasicLayer(name, typeName, params, networkParameters)
{
    auto prefix = typeName + "[" + mName + "::ctor]: ";
    if (mInputs.size() != 1)
    {
        THROW(typeName, mName, "wrong number of input names");
    }
    if (mOutputs.size() != 1)
    {
        THROW(typeName, mName, "wrong number of output names");
    }

    DECLARE_IMPL(ReLUActivation, ReLUActivationImpl<MemoryManager>, ReLUActivationImpl<MemoryManagerFP16>)

    mInputName = mInputs[0];
    mOutputName = mOutputs[0];

    mNetworkParams.mWorkflow.copyDeclaration(mName, mInputName, raul::Workflow::Usage::Forward, raul::Workflow::Mode::Read);

    mNetworkParams.mWorkflow.copyDeclaration(mName, mInputName, mInputName.grad(), DEC_BACK_WRIT_ZERO);

    mNetworkParams.mWorkflow.copyDeclaration(mName, mInputName, mOutputName, DEC_FORW_WRIT);
    mNetworkParams.mWorkflow.copyDeclaration(mName, mInputName, mOutputName, DEC_BACK_READ);

    mNetworkParams.mWorkflow.copyDeclaration(mName, mInputName, mOutputName.grad(), DEC_BACK_READ);

    mDepth = mNetworkParams.mWorkflow.getDepth(mInputName);
    mHeight = mNetworkParams.mWorkflow.getHeight(mInputName);
    mWidth = mNetworkParams.mWorkflow.getWidth(mInputName);
}

ReLU6Activation::ReLU6Activation(const Name& name, const BasicParams& params, NetworkParameters& networkParameters)
    : ReLUActivation(name, "ReLU6Activation", params, networkParameters)
{
    DECLARE_IMPL(ReLU6Activation, ReLU6ActivationImpl<MemoryManager>, ReLU6ActivationImpl<MemoryManagerFP16>)
}

} // namespace raul