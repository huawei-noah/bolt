// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "RandomSelectLayer.h"

#include "impl/RandomSelectLayerCPU.h"
#include "impl/RandomSelectLayerGPU.h"

namespace raul
{
RandomSelectLayer::RandomSelectLayer(const Name& name, const RandomSelectParams& params, NetworkParameters& networkParameters)
    : BroadcastingLayer(name, "RandomSelect", params, networkParameters)
    , mProbability(params.probability)
    , mBroadcast(params.broadcast)
    , mRandomName(name / "random")
{
    if (mInputs.size() != 2)
    {
        THROW("RandomSelectLayer", name, "wrong number of input names");
    }

    if (mNetworkParams.mWorkflow.getOverrideLayerExecutionTarget() == LayerExecutionTarget::Default)
    {
        DECLARE_IMPL(RandomSelectLayer, RandomSelectLayerCPU<MemoryManager>, RandomSelectLayerGPU, RandomSelectLayerCPU<MemoryManagerFP16>)
    }
    else if (mNetworkParams.mWorkflow.getOverrideLayerExecutionTarget() == LayerExecutionTarget::CPU)
    {
        DECLARE_IMPL(RandomSelectLayer, RandomSelectLayerCPU<MemoryManager>, NotImplemented, RandomSelectLayerCPU<MemoryManager>)
    }
    else if (mNetworkParams.mWorkflow.getOverrideLayerExecutionTarget() == LayerExecutionTarget::CPUFP16)
    {
        DECLARE_IMPL(RandomSelectLayer, RandomSelectLayerCPU<MemoryManagerFP16>, NotImplemented, RandomSelectLayerCPU<MemoryManagerFP16>)
    }
    else
    {
        THROW(mTypeName, mName, "unsupported layer execution target");
    }

    for (const auto& input : mInputs)
    {
        mNetworkParams.mWorkflow.copyDeclaration(name, input, raul::Workflow::Usage::Forward, raul::Workflow::Mode::Read);
        mNetworkParams.mWorkflow.copyDeclaration(name, input, input.grad(), DEC_BACK_WRIT_ZERO);
    }

    if (params.getLayerExecutionTarget() == LayerExecutionTarget::GPU ||
        (mNetworkParams.mWorkflow.getExecutionTarget() == ExecutionTarget::GPU && params.getLayerExecutionTarget() == LayerExecutionTarget::Default))
    {
        if (mNetworkParams.mWorkflow.getShape(mInputs[0]) != mNetworkParams.mWorkflow.getShape(mInputs[1]))
        {
            THROW("RandomSelectLayer", name, "broadcasting not supported on GPU");
        }
    }

    shape outputShape{ 1u, mNetworkParams.mWorkflow.getDepth(mInputs[0]), mNetworkParams.mWorkflow.getHeight(mInputs[0]), mNetworkParams.mWorkflow.getWidth(mInputs[0]) };
    if (mBroadcast)
    {
        shape inputShape{ 1u, mNetworkParams.mWorkflow.getDepth(mInputs[1]), mNetworkParams.mWorkflow.getHeight(mInputs[1]), mNetworkParams.mWorkflow.getWidth(mInputs[1]) };
        std::transform(inputShape.begin(), inputShape.end(), outputShape.begin(), outputShape.begin(), [](auto a, auto b) { return std::max(a, b); });
    }
    mNetworkParams.mWorkflow.tensorNeeded(name, mOutputs[0], raul::WShape{ raul::BS(), outputShape[1], outputShape[2], outputShape[3] }, DEC_FORW_WRIT);
    mNetworkParams.mWorkflow.copyDeclaration(name, mOutputs[0], mOutputs[0].grad(), DEC_BACK_READ);

    mNetworkParams.mWorkflow.copyDeclaration(mName, mOutputs[0], mRandomName, DEC_FORW_WRIT);
    mNetworkParams.mWorkflow.copyDeclaration(mName, mOutputs[0], mRandomName, DEC_BACK_READ);

    if (params.getLayerExecutionTarget() == LayerExecutionTarget::GPU ||
        (mNetworkParams.mWorkflow.getExecutionTarget() == ExecutionTarget::GPU && params.getLayerExecutionTarget() == LayerExecutionTarget::Default))
    {
        mNetworkParams.mWorkflow.tensorNeeded(mName,
                                              mName / "randomCPU",
                                              WShape{ BS(), outputShape[1], outputShape[2], outputShape[3] },
                                              raul::Workflow::Usage::Forward,
                                              raul::Workflow::Mode::Write,
                                              true,
                                              true,
                                              false,
                                              false,
                                              false,
                                              LayerExecutionTarget::CPU);

        if (!mNetworkParams.mWorkflow.getKernelManager().hasKernel(mTypeName, "selectForward"))
        {
            string source =
#include "kernels/select.cl"
                ;
            mNetworkParams.mWorkflow.getKernelManager().registerProgram(mTypeName, source);
        }
    }
}
}
