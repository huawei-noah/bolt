// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "GlobalAveragePoolLayer.h"

namespace raul
{

GlobAveragePoolLayer::GlobAveragePoolLayer(const Name& name, const BasicParams& params, NetworkParameters& networkParameters)
    : BasicLayer(name, "GlobalAveragePooling", params, networkParameters)
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

    mNetworkParams.mWorkflow.copyDeclaration(mName, mInputName, raul::Workflow::Usage::Forward, raul::Workflow::Mode::Read);

    mNetworkParams.mWorkflow.copyDeclaration(mName, mInputName, mInputName.grad(), DEC_BACK_WRIT_ZERO);

    mNetworkParams.mWorkflow.tensorNeeded(mName, mOutputName, raul::WShape{ BS(), mInputDepth, 1u, 1u }, DEC_FORW_WRIT);

    mNetworkParams.mWorkflow.copyDeclaration(mName, mOutputName, mOutputName.grad(), DEC_BACK_READ);
}

void GlobAveragePoolLayer::forwardComputeImpl(NetworkMode)
{
    const size_t batchSize = mNetworkParams.mWorkflow.getBatchSize();
    const dtype reciprocalKernelSize = 1.0_dt / static_cast<dtype>(mInputWidth * mInputHeight);

    if (mNetworkParams.mWorkflow.getExecutionTarget() == ExecutionTarget::CPU)
    {
        auto& output = mNetworkParams.mMemoryManager[mOutputName];

        const auto& inputs = mNetworkParams.mMemoryManager[mInputName];

        auto inputs3D = inputs.reshape(yato::dims(batchSize, mInputDepth, mInputHeight * mInputWidth));
        auto outputs2D = output.reshape(yato::dims(batchSize, mInputDepth));

        for (size_t b = 0; b < batchSize; ++b)
        {

            for (size_t k = 0; k < mInputDepth; ++k)
            {
                dtype sum = 0.0_dt;
                for (size_t i = 0; i < mInputHeight * mInputWidth; ++i)
                {
                    sum += inputs3D[b][k][i];
                }
                outputs2D[b][k] = sum * reciprocalKernelSize;
            }
        }
    }
    else if (mNetworkParams.mWorkflow.getExecutionTarget() == ExecutionTarget::CPUFP16)
    {
        auto& output = mNetworkParams.mMemoryManagerFP16[mOutputName];

        const auto& inputs = mNetworkParams.mMemoryManagerFP16[mInputName];

        auto inputs3D = inputs.reshape(yato::dims(batchSize, mInputDepth, mInputHeight * mInputWidth));
        auto outputs2D = output.reshape(yato::dims(batchSize, mInputDepth));

        for (size_t b = 0; b < batchSize; ++b)
        {
            for (size_t k = 0; k < mInputDepth; ++k)
            {
                dtype sum = 0.0_dt;
                for (size_t i = 0; i < mInputHeight * mInputWidth; ++i)
                {
                    sum += TODTYPE(inputs3D[b][k][i]);
                }
                outputs2D[b][k] = TOHTYPE(sum * reciprocalKernelSize);
            }
        }
    }
}

void GlobAveragePoolLayer::backwardComputeImpl()
{
    // if (mNetworkParams.isGradNeeded(mInputName))
    {
        const size_t batchSize = mNetworkParams.mWorkflow.getBatchSize();
        const dtype reciprocalKernelSize = 1.0_dt / static_cast<dtype>(mInputHeight * mInputWidth);
        if (mNetworkParams.mWorkflow.getExecutionTarget() == ExecutionTarget::CPU)
        {
            auto& prevLayerDelta = mNetworkParams.mMemoryManager[mInputName.grad()];

            const auto& deltas = mNetworkParams.mMemoryManager[mOutputName.grad()];

            auto deltas2D = deltas.reshape(yato::dims(batchSize, mInputDepth));
            auto prevDeltas3D = prevLayerDelta.reshape(yato::dims(batchSize, mInputDepth, mInputHeight * mInputWidth));

            for (size_t batch = 0; batch < batchSize; ++batch)
            {
                for (size_t c = 0; c < mInputDepth; ++c)
                {
                    for (size_t i = 0; i < mInputHeight * mInputWidth; ++i)
                    {
                        prevDeltas3D[batch][c][i] += deltas2D[batch][c] * reciprocalKernelSize;
                    }
                }
            }
        }
        else if (mNetworkParams.mWorkflow.getExecutionTarget() == ExecutionTarget::CPUFP16)
        {
            auto& prevLayerDelta = mNetworkParams.mMemoryManagerFP16[mInputName.grad()];

            const auto& deltas = mNetworkParams.mMemoryManagerFP16[mOutputName.grad()];

            auto deltas2D = deltas.reshape(yato::dims(batchSize, mInputDepth));
            auto prevDeltas3D = prevLayerDelta.reshape(yato::dims(batchSize, mInputDepth, mInputHeight * mInputWidth));

            for (size_t batch = 0; batch < batchSize; ++batch)
            {
                for (size_t c = 0; c < mInputDepth; ++c)
                {
                    for (size_t i = 0; i < mInputHeight * mInputWidth; ++i)
                    {
                        prevDeltas3D[batch][c][i] += TOHTYPE(deltas2D[batch][c] * reciprocalKernelSize);
                    }
                }
            }
        }
    }
}

} // namespace raul