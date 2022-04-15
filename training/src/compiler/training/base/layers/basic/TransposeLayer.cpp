// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "TransposeLayer.h"

#include "impl/TransposeLayerCPU.h"

#include <algorithm>

#include <training/base/common/MemoryManager.h>

using namespace raul;

namespace
{

size_t DimensionToIndex(raul::Dimension dim)
{
    switch (dim)
    {
        case raul::Dimension::Batch:
            return 0;
        case raul::Dimension::Depth:
            return 1;
        case raul::Dimension::Height:
            return 2;
        case raul::Dimension::Width:
            return 3;
        default:
            THROW_NONAME("TransposeLayer", "Bad dimension " + std::to_string(static_cast<int>(dim)));
    }
}

raul::WShape getProperOutShape(size_t dim, const raul::shape& inputShape)
{
    switch (dim)
    {
        case 1:
            return raul::WShape{ inputShape[1], raul::BS(), inputShape[2], inputShape[3] };
        case 2:
            return raul::WShape{ inputShape[2], inputShape[1], raul::BS(), inputShape[3] };
        case 3:
            return raul::WShape{ inputShape[3], inputShape[1], inputShape[2], raul::BS() };
        default:
            THROW_NONAME("TransposeLayer", "Incorrect dimension number: " + std::to_string(static_cast<int>(dim)));
    }
}

} // anonymous namespace

namespace raul
{

TransposeLayer::TransposeLayer(const Name& name, const TransposingParams& params, NetworkParameters& networkParameters)
    : BasicLayer(name, "Transpose", params, networkParameters)
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

    DECLARE_IMPL(TransposeLayer, TransposeLayerCPU<MemoryManager>, TransposeLayerCPU<MemoryManagerFP16>)

    mInputName = mInputs[0];
    mOutputName = mOutputs[0];

    const shape inputShape = shape{ 1u, mNetworkParams.mWorkflow.getDepth(mInputName), mNetworkParams.mWorkflow.getHeight(mInputName), mNetworkParams.mWorkflow.getWidth(mInputName) };

    auto dim1 = params.dim1;
    auto dim2 = params.dim2;

    if (dim1 == Dimension::Default)
    {
        dim1 = Dimension::Width;
    }
    if (dim2 == Dimension::Default)
    {
        dim2 = Dimension::Height;
    }

    mNetworkParams.mWorkflow.copyDeclaration(mName, mInputName, raul::Workflow::Usage::ForwardAndBackward, raul::Workflow::Mode::Read);

    mNetworkParams.mWorkflow.copyDeclaration(mName, mInputName, mInputName.grad(), DEC_BACK_WRIT_ZERO);

    mDim1 = DimensionToIndex(dim1);
    mDim2 = DimensionToIndex(dim2);

    // Needed for specific case, when both input dimensions are BATCH (not restricted)
    if ((mDim1 == 0) ^ (mDim2 == 0))
    {
        mNetworkParams.mWorkflow.tensorNeeded(mName, mOutputName, mDim1 == 0 ? getProperOutShape(mDim2, inputShape) : getProperOutShape(mDim1, inputShape), DEC_FORW_WRIT);
    }
    else
    {
        auto outputShape = inputShape;
        std::swap(outputShape[mDim1], outputShape[mDim2]);

        mNetworkParams.mWorkflow.tensorNeeded(mName, mOutputName, raul::WShape(BS(), outputShape[1], outputShape[2], outputShape[3]), DEC_FORW_WRIT);
    }

    mNetworkParams.mWorkflow.copyDeclaration(mName, mOutputName, mOutputName.grad(), DEC_BACK_READ);
}

} // namespace raul