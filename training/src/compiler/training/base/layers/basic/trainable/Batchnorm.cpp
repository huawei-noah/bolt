// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "Batchnorm.h"

#include <algorithm>

#include "impl/BatchnormCPU.h"

namespace raul
{

BatchNormLayer::BatchNormLayer(const Name& name, const BatchnormParams& params, NetworkParameters& networkParameters)
    : TrainableLayer(name, "BatchNorm", params, networkParameters)
    , mMomentum(params.momentum)
    , mEps(params.eps)
    , mDimension(params.dim)
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

    if (mMomentum < 0.0_dt || mMomentum > 1.0_dt)
    {
        THROW(mTypeName, mName, "incorrect momentum value");
    }

    if (mDimension == raul::Dimension::Batch || mDimension == raul::Dimension::Default)
    {
        THROW(mTypeName, mName, "incorrect dimension specified");
    }

    DECLARE_IMPL(BatchNormLayer, BatchNormLayerCPU<MemoryManager>, BatchNormLayerCPU<MemoryManagerFP16>)

    mInputName = mInputs[0];
    mOutputName = mOutputs[0];

    using shape3D = yato::dimensionality<3U, size_t>;

    const auto inputShape = shape3D{ mNetworkParams.mWorkflow.getDepth(mInputName), mNetworkParams.mWorkflow.getHeight(mInputName), mNetworkParams.mWorkflow.getWidth(mInputName) };

    mInputWidth = inputShape[2];
    mInputHeight = inputShape[1];
    mInputDepth = inputShape[0];

    mInputDepth = mNetworkParams.mWorkflow.getDepth(mInputName);
    mInputHeight = mNetworkParams.mWorkflow.getHeight(mInputName);
    mInputWidth = mNetworkParams.mWorkflow.getWidth(mInputName);

    mChosenDimSize = inputShape[static_cast<size_t>(mDimension) - 1];
    for (size_t k = 0, i = 0; i < inputShape.dimensions_num(); ++i)
    {
        if (i + 1 != static_cast<size_t>(mDimension))
        {
            mOtherDims[k] = inputShape[i];
            k++;
        }
    }

    mNetworkParams.mWorkflow.copyDeclaration(mName, mInputName, raul::Workflow::Usage::Forward, raul::Workflow::Mode::Read);

    mNetworkParams.mWorkflow.copyDeclaration(mName, mInputName, mInputName.grad(), DEC_BACK_WRIT_ZERO);

    mNetworkParams.mWorkflow.tensorNeeded(mName, mOutputName, raul::WShape{ raul::BS(), mInputDepth, mInputHeight, mInputWidth }, DEC_FORW_WRIT);

    mNetworkParams.mWorkflow.copyDeclaration(mName, mOutputName, mOutputName.grad(), DEC_BACK_READ);

    shape internalTensorsShape{ 1u, 1u, 1u, 1u };
    internalTensorsShape[static_cast<size_t>(mDimension)] = inputShape[static_cast<size_t>(mDimension) - 1];

    mNetworkParams.mWorkflow.tensorNeeded(mName, mName / "Mean", raul::WShape(internalTensorsShape), DEC_FORW_WRIT);

    mNetworkParams.mWorkflow.tensorNeeded(mName, mName / "Variance", raul::WShape(internalTensorsShape), DEC_FORW_WRIT);

    mNetworkParams.mWorkflow.tensorNeeded(mName, mName / "MeanEval", raul::WShape(internalTensorsShape), DEC_FORW_WRIT_NOMEMOPT);

    mNetworkParams.mWorkflow.tensorNeeded(mName, mName / "VarianceEval", raul::WShape(internalTensorsShape), DEC_FORW_WRIT_NOMEMOPT);

    mNetworkParams.mWorkflow.tensorNeeded(mName, mName / "XHat", raul::WShape{ raul::BS(), mInputDepth, mInputHeight, mInputWidth }, DEC_FORW_WRIT);
    mNetworkParams.mWorkflow.copyDeclaration(mName, mName / "XHat", DEC_BACK_READ);

    mNetworkParams.mWorkflow.tensorNeeded(mName, mName / "VarianceSqrt", raul::WShape(internalTensorsShape), DEC_FORW_WRIT);
    mNetworkParams.mWorkflow.copyDeclaration(mName, mName / "VarianceSqrt", DEC_BACK_READ);

    // beta
    mNetworkParams.mWorkflow.tensorNeeded(mName, mBiasesName, raul::WShape(internalTensorsShape), DEC_TRAINABLE);

    // gamma
    mNetworkParams.mWorkflow.tensorNeeded(mName, mWeightsName, raul::WShape(internalTensorsShape), DEC_TRAINABLE);

    if (!mFrozen)
    {
        mNetworkParams.mWorkflow.copyDeclaration(mName, mBiasesName, mBiasesName.grad(), DEC_TRAINABLE_GRAD);

        mNetworkParams.mWorkflow.copyDeclaration(mName, mWeightsName, mWeightsName.grad(), DEC_TRAINABLE_GRAD);
    }
}

} // namespace raul