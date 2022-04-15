// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "ConvolutionDepthwiseLayer.h"

#include <algorithm>
#include <functional>

#if defined(_OPENMP)
#include <omp.h>
#endif

#include <training/system/NameGenerator.h>

#include "impl/ConvolutionDepthwiseLayerCPU.h"

namespace raul
{

ConvolutionDepthwiseLayer::ConvolutionDepthwiseLayer(const Name& name, const Convolution2DParams& params, NetworkParameters& networkParameters)
    : ConvolutionDepthwiseLayer(name, "ConvolutionDepthwise2D", params, networkParameters){ MEASURE_BLOCK("ConvolutionDepthwise2D[" + mName + "::ctor]") }

    ConvolutionDepthwiseLayer::ConvolutionDepthwiseLayer(const Name& name, const std::string& typeName, const Convolution2DParams& params, NetworkParameters& networkParameters)
    : TrainableLayer(name, typeName, params, networkParameters)
    , mKernelWidth(params.kernelWidth)
    , mKernelHeight(params.kernelHeight)
    , mKernelsCount(params.kernelsCount)
    , mStrideW(params.strideW)
    , mStrideH(params.strideH)
    , mPaddingW(params.paddingW)
    , mPaddingH(params.paddingH)
    , mUseBias(params.bias)
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

    DECLARE_IMPL(ConvolutionDepthwiseLayer, ConvolutionDepthwiseLayerCPU<MemoryManager>, ConvolutionDepthwiseLayerCPU<MemoryManagerFP16>)

    mInputName = mInputs[0];
    mOutputName = mOutputs[0];

    mInputDepth = mNetworkParams.mWorkflow.getDepth(mInputName);
    mInputHeight = mNetworkParams.mWorkflow.getHeight(mInputName);
    mInputWidth = mNetworkParams.mWorkflow.getWidth(mInputName);

    if (mKernelsCount != mInputDepth)
    {
        THROW("ConvolutionDepthwiseLayer", mName, "input channels != kernels amount");
    }

    mOutputWidth = (mInputWidth + 2 * mPaddingW - mKernelWidth) / mStrideW + 1;
    mOutputHeight = (mInputHeight + 2 * mPaddingH - mKernelHeight) / mStrideH + 1;

    mNetworkParams.mWorkflow.copyDeclaration(mName, mInputName, raul::Workflow::Usage::ForwardAndBackward, raul::Workflow::Mode::Read);

    mNetworkParams.mWorkflow.copyDeclaration(mName, mInputName, mInputName.grad(), DEC_BACK_WRIT_ZERO);

    mNetworkParams.mWorkflow.tensorNeeded(mName, mOutputName, raul::WShape{ raul::BS(), mKernelsCount, mOutputHeight, mOutputWidth }, DEC_FORW_WRIT);

    mNetworkParams.mWorkflow.copyDeclaration(mName, mOutputName, mOutputName.grad(), DEC_BACK_READ);

    size_t numThreads = 1;
#if defined(_OPENMP)
#pragma omp parallel
    {
        numThreads = omp_get_num_threads();
    }
#endif
    for (size_t i = 0; i < numThreads; ++i)
    {
        Name im2ColF = mName / "Im2ColFor" / Conversions::toString(i);
        mIm2ColForward.push_back(im2ColF);

        mNetworkParams.mWorkflow.tensorNeeded(mName, im2ColF, raul::WShape{ 1u, 1u, 1u, mOutputHeight * mOutputWidth * mInputDepth * mKernelHeight * mKernelWidth }, DEC_FORW_WRIT);
    }

    mNetworkParams.mWorkflow.tensorNeeded(mName, mWeightsName, raul::WShape{ mKernelsCount, 1u, mKernelHeight, mKernelWidth }, DEC_TRAINABLE);

    if (mUseBias)
    {
        mNetworkParams.mWorkflow.tensorNeeded(mName, mBiasesName, raul::WShape{ 1u, mKernelsCount, 1u, 1u }, DEC_TRAINABLE);
    }

    if (!mFrozen)
    {
        mNetworkParams.mWorkflow.copyDeclaration(mName, mWeightsName, mWeightsName.grad(), DEC_TRAINABLE_GRAD);

        if (mUseBias)
        {
            mNetworkParams.mWorkflow.copyDeclaration(mName, mBiasesName, mBiasesName.grad(), DEC_TRAINABLE_GRAD);
        }
    }
}

void ConvolutionDepthwiseLayer::forwardComputeImpl(NetworkMode mode)
{
    mImpl->forwardComputeImpl(mode);
}

void ConvolutionDepthwiseLayer::backwardComputeImpl()
{
    mImpl->backwardComputeImpl();
}
} // namespace raul
