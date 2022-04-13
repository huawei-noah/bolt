// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "LabelSmoothing.h"

#include <algorithm>
#include <limits>

#include <training/base/common/MemoryManager.h>

#include "impl/LabelSmoothingCPU.h"

namespace
{

const size_t NoPadding = std::numeric_limits<std::size_t>::max();

} // anonymous namespace

namespace raul
{

LabelSmoothing::LabelSmoothing(const Name& name, const LabelSmoothingParams& params, NetworkParameters& networkParameters)
    : BasicLayer(name, "LabelSmoothing", params, networkParameters)
    , mSmoothing(TODTYPE(params.smoothing))
    , mPaddingIdx(params.paddingClass >= 0 ? params.paddingClass : NoPadding)
{
    auto prefix = mTypeName + "[" + mName + "::ctor]: ";

    if (mInputs.size() != 1)
    {
        THROW(mTypeName, mName, "wrong number of input names");
    }
    if (mOutputs.size() != 1)
    {
        THROW(mTypeName, mName, "wrong number of output names");
    }

    DECLARE_IMPL(LabelSmoothing, LabelSmoothingCPU<MemoryManager>, LabelSmoothingCPU<MemoryManagerFP16>)

    mInputName = mInputs[0];
    mOutputName = mOutputs[0];

    if (mNetworkParams.mWorkflow.getWidth(mInputName) < 2)
    {
        THROW(mTypeName, mName, "input width must be at least 2");
    }
    if (mNetworkParams.mWorkflow.getWidth(mInputName) == 2 && mPaddingIdx != NoPadding)
    {
        THROW(mTypeName, mName, "when using padding, input width must be at least 3");
    }

    mNetworkParams.mWorkflow.copyDeclaration(mName, mInputName, raul::Workflow::Usage::Forward, raul::Workflow::Mode::Read);

    mNetworkParams.mWorkflow.copyDeclaration(mName, mInputName, mOutputName, DEC_FORW_WRIT);
}

} // namespace raul