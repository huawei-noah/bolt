// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "AveragePoolLayer.h"

#include "impl/AveragePoolLayerCPU.h"

namespace raul
{

AveragePoolLayer::AveragePoolLayer(const Name& name, const Pool2DParams& params, NetworkParameters& networkParameters)
    : BasicLayer(name, "AveragePooling", params, networkParameters)
    , mKernelWidth(params.kernelWidth)
    , mKernelHeight(params.kernelHeight)
    , mPaddingW(params.paddingW)
    , mPaddingH(params.paddingH)
    , mStrideW(params.strideW)
    , mStrideH(params.strideH)
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

    mInputName = mInputs[0];
    mOutputName = mOutputs[0];

    mInputDepth = mNetworkParams.mWorkflow.getDepth(mInputName);
    mInputHeight = mNetworkParams.mWorkflow.getHeight(mInputName);
    mInputWidth = mNetworkParams.mWorkflow.getWidth(mInputName);

    if ((mPaddingH > mKernelHeight / 2) || (mPaddingW > mKernelWidth / 2))
    {
        THROW(mTypeName, mName, "Padding should be smaller than half of kernel size");
    }
    if (mKernelHeight == 0 || mKernelWidth == 0)
    {
        THROW(mTypeName, mName, "Kernel size can't be null");
    }
    if ((mInputWidth + mPaddingW * 2 < mKernelWidth) || (mInputHeight + mPaddingW * 2 < mKernelHeight))
    {
        THROW(mTypeName, mName, "ImageSize + 2*Padding can't be less than KernelSize");
    }

    DECLARE_IMPL(AveragePoolLayer, AveragePoolLayerCPU<MemoryManager>, AveragePoolLayerCPU<MemoryManagerFP16>)

    mOutputWidth = (mInputWidth + mPaddingW * 2 - mKernelWidth) / mStrideW + 1;
    mOutputHeight = (mInputHeight + mPaddingH * 2 - mKernelHeight) / mStrideH + 1;

    mNetworkParams.mWorkflow.copyDeclaration(mName, mInputName, raul::Workflow::Usage::Forward, raul::Workflow::Mode::Read);

    mNetworkParams.mWorkflow.copyDeclaration(mName, mInputName, mInputName.grad(), DEC_BACK_WRIT_ZERO);

    mNetworkParams.mWorkflow.tensorNeeded(mName, mOutputName, raul::WShape{ BS(), mInputDepth, mOutputHeight, mOutputWidth }, DEC_FORW_WRIT);

    mNetworkParams.mWorkflow.copyDeclaration(mName, mOutputName, mOutputName.grad(), DEC_BACK_READ);
}

} // namespace raul