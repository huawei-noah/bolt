// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "GaussianUpsamplingDistributionLayer.h"

#include <training/api/API.h>

#include "impl/GaussianUpsamplingDistributionLayerCPU.h"
#include "impl/GaussianUpsamplingDistributionLayerGPU.h"

namespace raul
{

GaussianUpsamplingDistributionLayer::GaussianUpsamplingDistributionLayer(const Name& name, const BasicParams& params, NetworkParameters& networkParameters)
    : BasicLayer(name, "GaussianUpsamplingDistributionLayer", params, networkParameters, { false, false })
{
    if (mNetworkParams.mWorkflow.getOverrideLayerExecutionTarget() == LayerExecutionTarget::Default)
    {
        DECLARE_IMPL(GaussianUpsamplingDistributionLayer,
                     GaussianUpsamplingDistributionLayerCPU<MemoryManager>,
                     GaussianUpsamplingDistributionLayerGPU,
                     GaussianUpsamplingDistributionLayerCPU<MemoryManagerFP16>)
    }
    else if (mNetworkParams.mWorkflow.getOverrideLayerExecutionTarget() == LayerExecutionTarget::CPU)
    {
        DECLARE_IMPL(GaussianUpsamplingDistributionLayer, GaussianUpsamplingDistributionLayerCPU<MemoryManager>, NotImplemented, GaussianUpsamplingDistributionLayerCPU<MemoryManager>)
    }
    else if (mNetworkParams.mWorkflow.getOverrideLayerExecutionTarget() == LayerExecutionTarget::CPUFP16)
    {
        DECLARE_IMPL(GaussianUpsamplingDistributionLayer, GaussianUpsamplingDistributionLayerCPU<MemoryManagerFP16>, NotImplemented, GaussianUpsamplingDistributionLayerCPU<MemoryManagerFP16>)
    }
    else
    {
        THROW(mTypeName, mName, "unsupported layer execution target");
    }

    auto prefix = "GaussianUpsamplingDistributionLayer[" + mName + "::ctor]: ";

    if (mInputs.size() != 3)
    {
        THROW(mTypeName, mName, "wrong number of input names");
    }
    if (mOutputs.size() != 1)
    {
        THROW(mTypeName, mName, "wrong number of output names");
    }
    if (mOutputs[0].empty())
    {
        THROW(mTypeName, mName, "empty output name");
    }
    for (const auto& input : mInputs)
    {
        if (input.empty())
        {
            THROW(mTypeName, mName, "empty input name");
        }
        mNetworkParams.mWorkflow.copyDeclaration(mName, input, raul::Workflow::Usage::ForwardAndBackward, raul::Workflow::Mode::Read);
    }

    mValuesName = mInputs[0];
    mLocName = mInputs[1];
    mScaleName = mInputs[2];
    mOutputName = mOutputs[0];

    if (!mNetworkParams.mWorkflow.isBatchPlaceholded(mValuesName))
    {
        mNetworkParams.mWorkflow.tensorNeeded(mName, mOutputName, WShape{ BS(), 1u, mNetworkParams.mWorkflow.getBatch(mValuesName), mNetworkParams.mWorkflow.getHeight(mLocName) }, DEC_FORW_WRIT);
        mNetworkParams.mWorkflow.copyDeclaration(mName, mOutputName, mOutputName.grad(), DEC_BACK_READ);

        mNetworkParams.mWorkflow.copyDeclaration(mName, mLocName, mLocName.grad(), DEC_BACK_WRIT_ZERO);
        mNetworkParams.mWorkflow.copyDeclaration(mName, mScaleName, mScaleName.grad(), DEC_BACK_WRIT_ZERO);
    }
    else
    {
        THROW(mTypeName, mName, "values tensor should have known batch size");
    }
}

} // namespace raul
