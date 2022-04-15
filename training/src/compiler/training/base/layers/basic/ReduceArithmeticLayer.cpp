// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "ReduceArithmeticLayer.h"

#include "impl/ReduceArithmeticLayerCPU.h"

namespace raul
{

std::unordered_set<std::string> ReduceArithmeticLayer::mAvailableOps = { "sum", "mean", "batch_mean", "std", "count_non_zero_elems" };

ReduceArithmeticLayer::ReduceArithmeticLayer(const Name& name, const BasicParamsWithDim& params, NetworkParameters& networkParameters, const std::string& operation)
    : BasicLayer(name, "ReduceArithmetic", params, networkParameters)
    , mDim(params.dim)
    , mOperation(operation)
{
    if (mAvailableOps.find(mOperation) == mAvailableOps.end())
    {
        THROW("ReduceArithmeticLayer", mName, "unavailabe operation");
    }

    if (mInputs.size() != 1)
    {
        THROW("Reduce" + mOperation + "Layer", mName, "wrong number of input names");
    }

    if (mOutputs.size() != 1)
    {
        THROW("Reduce" + mOperation + "Layer", mName, "wrong number of output names");
    }

    if (mInputs[0].empty())
    {
        THROW("Reduce" + mOperation + "Layer", mName, "empty first input name");
    }

    if (mOutputs[0].empty())
    {
        THROW("Reduce" + mOperation + "Layer", mName, "empty output name");
    }

    DECLARE_IMPL(ReduceArithmeticLayer, ReduceArithmeticLayerCPU<MemoryManager>, ReduceArithmeticLayerCPU<MemoryManagerFP16>)

    mNetworkParams.mWorkflow.copyDeclaration(mName, mInputs[0], raul::Workflow::Usage::ForwardAndBackward, raul::Workflow::Mode::Read);

    mNetworkParams.mWorkflow.copyDeclaration(mName, mInputs[0], mInputs[0].grad(), DEC_BACK_WRIT_ZERO);

    if (mDim == Dimension::Default)
    {
        mNetworkParams.mWorkflow.tensorNeeded(mName, mOutputs[0], WShape{ 1u, 1u, 1u, 1u }, DEC_FRBC_WRIT_NOMEMOPT);

        mNetworkParams.mWorkflow.tensorNeeded(mName, mOutputs[0].grad(), WShape{ 1u, 1u, 1u, 1u }, DEC_BACK_READ);
    }
    else
    {
        shape inputShape = shape{ 1u, mNetworkParams.mWorkflow.getDepth(mInputs[0]), mNetworkParams.mWorkflow.getHeight(mInputs[0]), mNetworkParams.mWorkflow.getWidth(mInputs[0]) };

        if (mDim == Dimension::Batch)
        {
            mNetworkParams.mWorkflow.tensorNeeded(mName, mOutputs[0], WShape{ 1u, inputShape[1], inputShape[2], inputShape[3] }, DEC_FRBC_WRIT_NOMEMOPT);

            mNetworkParams.mWorkflow.tensorNeeded(mName, mOutputs[0].grad(), WShape{ 1u, inputShape[1], inputShape[2], inputShape[3] }, DEC_BACK_READ);
        }
        else
        {
            for (size_t i = 1; i < inputShape.dimensions_num(); ++i)
            {
                if (static_cast<size_t>(mDim) == i)
                {
                    inputShape[i] = 1;
                    break;
                }
            }
            if (mNetworkParams.mWorkflow.getShape(mInputs[0]).isBSDependent())
            {
                mNetworkParams.mWorkflow.tensorNeeded(mName, mOutputs[0], WShape(BS(), inputShape[1], inputShape[2], inputShape[3]), DEC_FRBC_WRIT_NOMEMOPT);

                mNetworkParams.mWorkflow.tensorNeeded(mName, mOutputs[0].grad(), WShape(BS(), inputShape[1], inputShape[2], inputShape[3]), DEC_BACK_READ);
            }
            else
            {
                mNetworkParams.mWorkflow.tensorNeeded(mName, mOutputs[0], WShape(mNetworkParams.mWorkflow.getBatch(mInputs[0]), inputShape[1], inputShape[2], inputShape[3]), DEC_FRBC_WRIT_NOMEMOPT);

                mNetworkParams.mWorkflow.tensorNeeded(mName, mOutputs[0].grad(), WShape(mNetworkParams.mWorkflow.getBatch(mInputs[0]), inputShape[1], inputShape[2], inputShape[3]), DEC_BACK_READ);
            }
        }
    }
}

}