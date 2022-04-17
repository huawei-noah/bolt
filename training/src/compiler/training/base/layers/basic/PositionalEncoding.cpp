// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "PositionalEncoding.h"
#include "impl/PositionalEncodingCPU.h"

#include <algorithm>

#include <training/base/common/MemoryManager.h>

namespace raul
{

PositionalEncoding::PositionalEncoding(const Name& name, const PositionalEncodingParams& params, NetworkParameters& networkParameters)
    : BasicLayer(name, "PositionalEncoding", params, networkParameters)
    , mModelSize(params.modelSize)
    , mMaxLength(params.maxLength)
    , mDurationEncoding(params.durationEncoding)
    , mMaxMelLength(params.maxMelLength)
{
    DECLARE_IMPL(PositionalEncoding, PositionalEncodingCPU<MemoryManager>, PositionalEncodingCPU<MemoryManagerFP16>)

    auto prefix = mTypeName + "[" + mName + "::ctor]: ";

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

    if (mModelSize % 2 != 0)
    {
        THROW(mTypeName, mName, "input vector length must be even");
    }

    mNetworkParams.mWorkflow.copyDeclaration(mName, mInputName, raul::Workflow::Usage::Forward, raul::Workflow::Mode::Read);

    if (!mDurationEncoding)
    {
        if (mNetworkParams.mWorkflow.getWidth(mInputName) != mModelSize)
        {
            THROW(mTypeName, mName, "bad input vector length");
        }
        mNetworkParams.mWorkflow.copyDeclaration(mName, mInputName, mInputName.grad(), DEC_BACK_WRIT_ZERO);
        mNetworkParams.mWorkflow.copyDeclaration(mName, mInputName, mOutputName, DEC_FORW_WRIT);
        mNetworkParams.mWorkflow.copyDeclaration(mName, mOutputName, mOutputName.grad(), DEC_BACK_READ);
    }
    else
    {
        if (mNetworkParams.mWorkflow.getDepth(mInputName) != 1)
        {
            THROW(mTypeName, mName, "input depth must be 1 for duration encoding mode");
        }
        if (mNetworkParams.mWorkflow.getHeight(mInputName) != 1)
        {
            THROW(mTypeName, mName, "input height must be 1 for duration encoding mode");
        }
        mNetworkParams.mWorkflow.tensorNeeded(mName, mOutputName, WShape{ BS(), 1u, mMaxMelLength, mModelSize }, DEC_FORW_WRIT);
    }
    mNetworkParams.mWorkflow.tensorNeeded(mName, mName / "pe", WShape{ 1u, 1u, mMaxLength, mModelSize }, DEC_FORW_READ_NOMEMOPT);
}

} // namespace raul