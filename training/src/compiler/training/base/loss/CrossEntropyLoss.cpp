// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "CrossEntropyLoss.h"

#include "impl/CrossEntropyLossCPU.h"

namespace raul
{

CrossEntropyLoss::CrossEntropyLoss(const Name& name, const LossParams& params, NetworkParameters& networkParameters)
    : BasicLayer(name, "CrossEntropyLoss", params, networkParameters)
    , mIsFinal(params.mIsFinal)
{
    if (mInputs.size() != 2 && mInputs.size() != 3)
    {
        THROW(mTypeName, mName, "wrong number of input names");
    }
    if (mOutputs.size() != 1)
    {
        THROW(mTypeName, mName, "wrong number of output names");
    }

    DECLARE_IMPL(CrossEntropyLoss, CrossEntropyLossCPU<MemoryManager>, CrossEntropyLossCPU<MemoryManagerFP16>)

    mInputName = mInputs[0];
    mLabelName = mInputs[1];
    mOutputName = mOutputs[0];

    if (params.reduction != LossParams::Reduction::None || mInputs.size() == 3)
    {
        LossParams paramsWrap = params;
        paramsWrap.getOutputs()[0] = mOutputName / "Wrap";
        mInputName = paramsWrap.getOutputs()[0];

        wrapper = std::make_unique<LossWrapper<CrossEntropyLoss>>(mName, paramsWrap, networkParameters);

        mNetworkParams.mWorkflow.copyDeclaration(mName, mInputName, raul::Workflow::Usage::Forward, raul::Workflow::Mode::Read);
        mNetworkParams.mWorkflow.copyDeclaration(mName, mInputName, mOutputName, DEC_FORW_WRIT_NOMEMOPT);
        if (!mIsFinal)
        {
            mNetworkParams.mWorkflow.copyDeclaration(mName, mOutputName, mOutputName.grad(), DEC_BACK_READ);
            mNetworkParams.mWorkflow.copyDeclaration(mName, mInputName, mInputName.grad(), DEC_BACK_WRIT_ZERO);
        }
    }
    else
    {
        mNetworkParams.mWorkflow.copyDeclaration(mName, mInputName, raul::Workflow::Usage::ForwardAndBackward, raul::Workflow::Mode::Read);

        mNetworkParams.mWorkflow.copyDeclaration(mName, mInputName, mInputName.grad(), DEC_BACK_WRIT_ZERO);

        mNetworkParams.mWorkflow.copyDeclaration(mName, mInputName, mOutputName, DEC_FORW_WRIT_NOMEMOPT);

        mNetworkParams.mWorkflow.copyDeclaration(mName, mInputName, mOutputName.grad(), DEC_BACK_READ);

        mNetworkParams.mWorkflow.copyDeclaration(mName, mLabelName, raul::Workflow::Usage::ForwardAndBackward, raul::Workflow::Mode::Read);
    }
    mDepth = mNetworkParams.mWorkflow.getDepth(mInputName);
    mHeight = mNetworkParams.mWorkflow.getHeight(mInputName);
    mWidth = mNetworkParams.mWorkflow.getWidth(mInputName);
}

} // namespace raul