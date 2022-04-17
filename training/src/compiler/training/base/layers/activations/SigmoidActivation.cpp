// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "SigmoidActivation.h"

#include "impl/SigmoidActivationCPU.h"

#include <algorithm>

#include <training/base/common/Common.h>
#include <training/base/common/MemoryManager.h>

namespace raul
{
SigmoidActivation::SigmoidActivation(const Name& name, const BasicParams& params, NetworkParameters& networkParameters)
    : BasicLayer(name, "Sigmoid", params, networkParameters)
{

    // d.polubotko: keep registration to pass TestWorkflowOverrideLayerExecutionTarget.TrainingOverrideTarget test
    if (mNetworkParams.mWorkflow.getOverrideLayerExecutionTarget() == LayerExecutionTarget::Default)
    {
        DECLARE_IMPL(SigmoidActivation, SigmoidActivationCPU<MemoryManager>, SigmoidActivationCPU<MemoryManagerFP16>)
    }
    else if (mNetworkParams.mWorkflow.getOverrideLayerExecutionTarget() == LayerExecutionTarget::CPU)
    {
        DECLARE_IMPL(SigmoidActivation, SigmoidActivationCPU<MemoryManager>, SigmoidActivationCPU<MemoryManager>)
    }
    else if (mNetworkParams.mWorkflow.getOverrideLayerExecutionTarget() == LayerExecutionTarget::CPUFP16)
    {
        DECLARE_IMPL(SigmoidActivation, SigmoidActivationCPU<MemoryManagerFP16>, SigmoidActivationCPU<MemoryManagerFP16>)
    }
    else
    {
        THROW(mTypeName, mName, "unsupported layer execution target");
    }

    mInputName = mInputs[0];
    mOutputName = mOutputs[0];

    mNetworkParams.mWorkflow.copyDeclaration(mName, mInputName, raul::Workflow::Usage::Forward, raul::Workflow::Mode::Read);

    mNetworkParams.mWorkflow.copyDeclaration(mName, mInputName, mInputName.grad(), DEC_BACK_WRIT_ZERO);

    mNetworkParams.mWorkflow.copyDeclaration(mName, mInputName, mOutputName, DEC_FORW_WRIT_COMP);
    mNetworkParams.mWorkflow.copyDeclaration(mName, mInputName, mOutputName, DEC_BACK_READ_COMP);

    mNetworkParams.mWorkflow.copyDeclaration(mName, mInputName, mOutputName.grad(), DEC_BACK_READ);

    mDepth = mNetworkParams.mWorkflow.getDepth(mInputName);
    mHeight = mNetworkParams.mWorkflow.getHeight(mInputName);
    mWidth = mNetworkParams.mWorkflow.getWidth(mInputName);
}
} // namespace raul