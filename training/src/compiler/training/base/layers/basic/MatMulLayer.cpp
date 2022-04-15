// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "MatMulLayer.h"

#include "impl/MatMulLayerCPU.h"

#include <algorithm>

namespace raul
{

MatMulLayer::MatMulLayer(const Name& name, const MatMulParams& params, NetworkParameters& networkParameters)
    : BasicLayer(name, "MatMul", params, networkParameters)
    , mCoeff(params.scale)
{
    auto prefix = mTypeName + "[" + mName + "::ctor]: ";

    if (mInputs.size() != 2)
    {
        THROW(mTypeName, mName, "wrong number of input names");
    }
    if (mOutputs.size() != 1)
    {
        THROW(mTypeName, mName, "wrong number of output names");
    }

    if (networkParameters.mWorkflow.getDepth(mInputs[0]) != networkParameters.mWorkflow.getDepth(mInputs[1]))
    {
        THROW(mTypeName, mName, "wrong depth");
    }
    if (networkParameters.mWorkflow.getWidth(mInputs[0]) != networkParameters.mWorkflow.getHeight(mInputs[1]))
    {
        THROW(mTypeName, mName, "wrong tensor sizes");
    }

    DECLARE_IMPL(MatMulLayer, MatMulLayerCPU<MemoryManager>, MatMulLayerCPU<MemoryManagerFP16>)

    for (const auto& input : mInputs)
    {
        mNetworkParams.mWorkflow.copyDeclaration(name, input, raul::Workflow::Usage::ForwardAndBackward, raul::Workflow::Mode::Read);

        if (mNetworkParams.mWorkflow.isTensorTrainable(input))
        {
            mNetworkParams.mWorkflow.copyDeclaration(mName, input, input.grad(), DEC_TRAINABLE_GRAD);
        }
        else
        {
            mNetworkParams.mWorkflow.copyDeclaration(mName, input, input.grad(), DEC_BACK_WRIT_ZERO);
        }
    }

    mDepth = networkParameters.mWorkflow.getDepth(mInputs[0]);

    mNetworkParams.mWorkflow.tensorNeeded(
        name, mOutputs[0], raul::WShape{ raul::BS(), mDepth, networkParameters.mWorkflow.getHeight(mInputs[0]), networkParameters.mWorkflow.getWidth(mInputs[1]) }, DEC_FORW_WRIT);

    mNetworkParams.mWorkflow.copyDeclaration(name, mOutputs[0], mOutputs[0].grad(), DEC_BACK_READ);
}

} // namespace raul