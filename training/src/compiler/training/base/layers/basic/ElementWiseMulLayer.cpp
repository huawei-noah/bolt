// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "ElementWiseMulLayer.h"

#include "impl/ElementWiseMulLayerCPU.h"

#include <algorithm>

namespace raul
{

ElementWiseMulLayer::ElementWiseMulLayer(const Name& name, const ElementWiseLayerParams& params, NetworkParameters& networkParameters)
    : BroadcastingLayer(name, "ElementWiseMul", params, networkParameters)
    , mBroadcast(params.mBroadcast)
    , mBackwardTmpBufferName("TempStorageForIntermediateCalculations")
{
    auto prefix = "ElementWiseMulLayer[" + name + "::ctor]: ";

    DECLARE_IMPL(ElementWiseDivLayer, ElementWiseMulLayerCPU<MemoryManager>, ElementWiseMulLayerCPU<MemoryManagerFP16>)

    for (const auto& input : mInputs)
    {
        mNetworkParams.mWorkflow.copyDeclaration(name, input, raul::Workflow::Usage::ForwardAndBackward, raul::Workflow::Mode::Read);

        if (mNetworkParams.mWorkflow.isTensorTrainable(input))
        {
            mNetworkParams.mWorkflow.copyDeclaration(name, input, input.grad(), DEC_TRAINABLE_GRAD);
        }
        else
        {
            mNetworkParams.mWorkflow.copyDeclaration(name, input, input.grad(), DEC_BACK_WRIT_ZERO);
        }
    }

    shape outputShape{ 1u, mNetworkParams.mWorkflow.getDepth(mInputs[0]), mNetworkParams.mWorkflow.getHeight(mInputs[0]), mNetworkParams.mWorkflow.getWidth(mInputs[0]) };
    if (mBroadcast)
    {
        for (size_t i = 1; i < mInputs.size(); ++i)
        {
            shape inputShape{ 1u, mNetworkParams.mWorkflow.getDepth(mInputs[i]), mNetworkParams.mWorkflow.getHeight(mInputs[i]), mNetworkParams.mWorkflow.getWidth(mInputs[i]) };
            std::transform(inputShape.begin(), inputShape.end(), outputShape.begin(), outputShape.begin(), [](auto a, auto b) { return std::max(a, b); });
        }

        for (const auto& mInput : mInputs)
        {
            shape inputShape{ 1u, mNetworkParams.mWorkflow.getDepth(mInput), mNetworkParams.mWorkflow.getHeight(mInput), mNetworkParams.mWorkflow.getWidth(mInput) };
            if (!Common::shapeIsBroadcastable(inputShape, outputShape))
            {
                THROW("ElementWiseMulLayer", name, "tensor '" + mInput + "'" + Conversions::toString(inputShape) + " is not broadcastable to " + Conversions::toString(outputShape));
            }
        }
    }
    else
    {
        for (size_t i = 1; i < mInputs.size(); ++i)
        {
            shape inputShape{ 1u, mNetworkParams.mWorkflow.getDepth(mInputs[i]), mNetworkParams.mWorkflow.getHeight(mInputs[i]), mNetworkParams.mWorkflow.getWidth(mInputs[i]) };
            if (inputShape != outputShape)
            {
                THROW("ElementWiseMulLayer",
                      name,
                      "broadcasting is off while tensor '" + mInputs[i] + "'" + Conversions::toString(inputShape) + " and tensor '" + mInputs[0] + "'" + Conversions::toString(outputShape) +
                          " shapes differ");
            }
        }
    }

    mNetworkParams.mWorkflow.tensorNeeded(name, mOutputs[0], raul::WShape{ BS(), outputShape[1], outputShape[2], outputShape[3] }, DEC_FORW_WRIT_COMP);
    mNetworkParams.mWorkflow.copyDeclaration(name, mOutputs[0], mOutputs[0], DEC_BACK_READ_COMP);
    mNetworkParams.mWorkflow.copyDeclaration(name, mOutputs[0], mOutputs[0].grad(), DEC_BACK_READ);

    mDepth = outputShape[1];
    mHeight = outputShape[2];
    mWidth = outputShape[3];
}

} // namespace raul