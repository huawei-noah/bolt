// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "DynamicDepthwiseConvolution2DLayer.h"

#include "impl/DynamicDepthwiseConvolution2DLayerCPU.h"

namespace raul
{

DynamicDepthwiseConvolution2DLayer::DynamicDepthwiseConvolution2DLayer(const Name& name, const BasicParams& params, NetworkParameters& networkParameters)
    : BasicLayer(name, "DynamicDepthwiseConvolution2D", params, networkParameters)
{
    auto prefix = "DynamicDepthwiseConvolution2D[" + mName + "::ctor]: ";
    if (mInputs.size() != 2)
    {
        THROW(mTypeName, mName, "wrong number of input names");
    }
    if (mOutputs.size() != 1)
    {
        THROW(mTypeName, mName, "wrong number of output names");
    }

    DECLARE_IMPL(DynamicDepthwiseConvolution2DLayer, DynamicDepthwiseConvolution2DLayerCPU<MemoryManager>, DynamicDepthwiseConvolution2DLayerCPU<MemoryManagerFP16>)

    mInputName = mInputs[0];
    mFiltersName = mInputs[1];
    mOutputName = mOutputs[0];

    // TF input format - NHWC
    mInputHeight = mNetworkParams.mWorkflow.getDepth(mInputName);
    mInputWidth = mNetworkParams.mWorkflow.getHeight(mInputName);

    // TF kernel format - [filter_height, filter_width, in_channels, channel_multiplier]
    if (mNetworkParams.mWorkflow.isBatchPlaceholded(mFiltersName) || mNetworkParams.mWorkflow.isDepthPlaceholded(mFiltersName))
    {
        THROW(mTypeName, mName, "should know these filters dimensions");
    }

    mFilterHeight = mNetworkParams.mWorkflow.getBatch(mFiltersName);
    mFilterWidth = mNetworkParams.mWorkflow.getDepth(mFiltersName);

    mOutputHeight = mInputHeight - mFilterHeight + 1;
    mOutputWidth = mInputWidth - mFilterWidth + 1;

    mNetworkParams.mWorkflow.copyDeclaration(mName, mInputName, raul::Workflow::Usage::ForwardAndBackward, raul::Workflow::Mode::Read);

    mNetworkParams.mWorkflow.copyDeclaration(mName, mInputName, mInputName.grad(), DEC_BACK_WRIT_ZERO);

    mNetworkParams.mWorkflow.copyDeclaration(mName, mFiltersName, raul::Workflow::Usage::ForwardAndBackward, raul::Workflow::Mode::Read);

    mNetworkParams.mWorkflow.copyDeclaration(mName, mFiltersName, mFiltersName.grad(), DEC_BACK_WRIT_ZERO);

    if (!mNetworkParams.mWorkflow.isWidthPlaceholded(mInputName))
    {
        if (mNetworkParams.mWorkflow.isHeightPlaceholded(mFiltersName))
        {
            THROW("DynamicDepthwiseConvolution2DLayer", mName, "filters height should be known");
        }

        mInputDepth = mNetworkParams.mWorkflow.getWidth(mInputName);
        mInChannels = mNetworkParams.mWorkflow.getHeight(mFiltersName);
        mChannelMultiplier = mNetworkParams.mWorkflow.getWidth(mFiltersName);

        if (mInChannels != mInputDepth)
        {
            THROW("DynamicDepthwiseConvolution2DLayer", mName, "input channels != kernels amount");
        }

        if (mNetworkParams.mWorkflow.isBatchPlaceholded(mInputName))
        {
            mNetworkParams.mWorkflow.tensorNeeded(mName, mOutputName, raul::WShape{ raul::BS(), mOutputHeight, mOutputWidth, mInChannels * mChannelMultiplier }, DEC_FORW_WRIT);
        }
        else
        {
            mNetworkParams.mWorkflow.tensorNeeded(
                mName, mOutputName, raul::WShape{ mNetworkParams.mWorkflow.getBatch(mInputName), mOutputHeight, mOutputWidth, mInChannels * mChannelMultiplier }, DEC_FORW_WRIT);
        }
    }
    else
    {
        if (!mNetworkParams.mWorkflow.isHeightPlaceholded(mFiltersName))
        {
            THROW("DynamicDepthwiseConvolution2DLayer", mName, "input channels != kernels amount");
        }

        mChannelMultiplier = mNetworkParams.mWorkflow.getWidth(mFiltersName);

        mNetworkParams.mWorkflow.tensorNeeded(mName, mOutputName, raul::WShape{ 1u, mOutputHeight, mOutputWidth, BS(mChannelMultiplier) }, DEC_FORW_WRIT);
    }

    mNetworkParams.mWorkflow.copyDeclaration(mName, mOutputName, mOutputName.grad(), DEC_BACK_READ);
}

}