// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "LayerNorm2D.h"

#include "impl/LayerNorm2dCPU.h"

namespace raul
{
LayerNorm2DLayer::LayerNorm2DLayer(const Name& name, const LayerNormParams& params, NetworkParameters& networkParameters)
    : TrainableLayer(name, "LayerNorm2D", params, networkParameters)
    , mVarianceName(name / "Variance")
    , mXHatName(name / "XHat")
    , mXHatNablaName(name / "XHatNabla")
    , mEps(params.eps)
    , mTFStyle(params.useTFstyle)
    , mUseBesselCorrection(params.useBesselCorrection)
{
    MEASURE_BLOCK(mTypeName + "[" + mName + "::ctor]")
    auto prefix = mTypeName + "[" + mName + "::ctor]: ";

    if (mInputs.size() != 1)
    {
        THROW(mTypeName, mName, "wrong number of input names");
    }
    if (mOutputs.size() != 1)
    {
        THROW(mTypeName, mName, "wrong number of output names");
    }

    auto target = params.getLayerExecutionTarget();

    if (target == LayerExecutionTarget::Default)
    {
        DECLARE_IMPL(LayerNormLayer, LayerNorm2DLayerCPU<MemoryManager>, NotImplemented, LayerNorm2DLayerCPU<MemoryManagerFP16>)
    }
    else if (target == LayerExecutionTarget::CPU)
    {
        DECLARE_IMPL(LayerNormLayer, LayerNorm2DLayerCPU<MemoryManager>, NotImplemented, NotImplemented)
    }
    else
    {
        THROW(mTypeName, mName, "unsupported layer execution target");
    }

    mInputName = mInputs[0];
    mOutputName = mOutputs[0];

    mInputDepth = mNetworkParams.mWorkflow.getDepth(mInputName);
    mInputHeight = mNetworkParams.mWorkflow.getHeight(mInputName);
    mInputWidth = mNetworkParams.mWorkflow.getWidth(mInputName);

    mOutputWidth = mInputWidth;
    mOutputHeight = mInputHeight;
    mOutputSize = mOutputHeight * mOutputWidth;

    mNetworkParams.mWorkflow.copyDeclaration(mName, mInputName, raul::Workflow::Usage::ForwardAndBackward, raul::Workflow::Mode::Read, target);
    mNetworkParams.mWorkflow.copyDeclaration(mName, mInputName, mInputName.grad(), DEC_BACK_WRIT_ZERO, target);

    // mNetworkParams.mWorkflow.tensorNeeded(mName, mOutputName, raul::WShape{ BS(), mInputDepth, mOutputHeight, mInputWidth }, DEC_FORW_WRIT, target);
    mNetworkParams.mWorkflow.copyDeclaration(mName, mInputName, mOutputName, DEC_FORW_WRIT, target);
    mNetworkParams.mWorkflow.copyDeclaration(mName, mOutputName, mOutputName.grad(), DEC_BACK_READ, target);

    mNetworkParams.mWorkflow.tensorNeeded(mName, mWeightsName, raul::WShape{ 1, 1, 1, mInputWidth }, DEC_TRAINABLE, target);
    mNetworkParams.mWorkflow.tensorNeeded(mName, mBiasesName, raul::WShape{ 1, 1, 1, mInputWidth }, DEC_TRAINABLE, target);

    mNetworkParams.mWorkflow.tensorNeeded(mName, mVarianceName, raul::WShape{ BS(), mInputDepth, 1u, 1u }, DEC_FRBC_WRIT, target);
    mNetworkParams.mWorkflow.tensorNeeded(mName, mXHatName, raul::WShape{ BS(), mInputDepth, mInputHeight, mInputWidth }, DEC_FRBC_WRIT, target);

    if (!mFrozen)
    {
        mNetworkParams.mWorkflow.copyDeclaration(mName, mWeightsName, mWeightsName.grad(), DEC_TRAINABLE_GRAD, target);
        mNetworkParams.mWorkflow.copyDeclaration(mName, mBiasesName, mBiasesName.grad(), DEC_TRAINABLE_GRAD, target);
        mNetworkParams.mWorkflow.copyDeclaration(mName, mXHatName, mXHatNablaName, DEC_BACK_WRIT_ZERO, target);
    }
}

} // namespace raul
