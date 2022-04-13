// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "ReduceExtremumLayer.h"

#include "impl/ReduceExtremumLayerCPU.h"

namespace raul
{

template<template<typename> typename Comparator>
ReduceExtremumLayer<Comparator>::ReduceExtremumLayer(const Name& name, const BasicParamsWithDim& params, NetworkParameters& networkParameters)
    : BasicLayer(name, "ReduceExtremum", params, networkParameters)
    , mDim(params.dim)
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

    typedef ReduceExtremumLayerCPU<Comparator, MemoryManager> ReduceExtremumLayerCPUFP32;
    typedef ReduceExtremumLayerCPU<Comparator, MemoryManagerFP16> ReduceExtremumLayerCPUFP16;

    DECLARE_IMPL(ReduceExtremumLayer, ReduceExtremumLayerCPUFP32, ReduceExtremumLayerCPUFP16)

    mNetworkParams.mWorkflow.copyDeclaration(mName, mInputs[0], raul::Workflow::Usage::ForwardAndBackward, raul::Workflow::Mode::Read);

    mNetworkParams.mWorkflow.copyDeclaration(mName, mInputs[0], mInputs[0].grad(), DEC_BACK_WRIT_ZERO);

    shape outputShape = shape{ 1u, mNetworkParams.mWorkflow.getDepth(mInputs[0]), mNetworkParams.mWorkflow.getHeight(mInputs[0]), mNetworkParams.mWorkflow.getWidth(mInputs[0]) };

    bool changeBS = false;
    if (mDim == raul::Dimension::Default)
    {
        outputShape = shape{ 1u, 1u, 1u, 1u };
        changeBS = true;
    }
    else
    {
        if (mDim == raul::Dimension::Batch)
        {
            changeBS = true;
        }
        else
        {
            for (size_t i = 1; i < 4; i++)
            {
                if (static_cast<size_t>(mDim) == i)
                {
                    outputShape[i] = 1;
                    break;
                }
            }
        }
    }
    if (changeBS)
    {
        mNetworkParams.mWorkflow.tensorNeeded(mName, mOutputs[0], raul::WShape{ outputShape }, DEC_FORW_WRIT);
    }
    else
    {
        if (mNetworkParams.mWorkflow.getShape(mInputs[0]).isBSDependent())
        {
            mNetworkParams.mWorkflow.tensorNeeded(mName, mOutputs[0], raul::WShape{ raul::BS(), outputShape[1], outputShape[2], outputShape[3] }, DEC_FORW_WRIT);
        }
        else
        {
            mNetworkParams.mWorkflow.tensorNeeded(mName, mOutputs[0], raul::WShape{ mNetworkParams.mWorkflow.getBatch(mInputs[0]), outputShape[1], outputShape[2], outputShape[3] }, DEC_FORW_WRIT);
        }
    }
    mNetworkParams.mWorkflow.copyDeclaration(name, mOutputs[0], mOutputs[0], DEC_BACK_READ);
    mNetworkParams.mWorkflow.copyDeclaration(mName, mOutputs[0], mOutputs[0].grad(), DEC_BACK_READ);
}

template class ReduceExtremumLayer<Max>;
template class ReduceExtremumLayer<Min>;

}