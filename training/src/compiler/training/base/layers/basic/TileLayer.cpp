// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "TileLayer.h"

#include "impl/TileLayerCPU.h"

namespace raul
{

TileLayer::TileLayer(const Name& name, const TilingParams& params, NetworkParameters& networkParameters)
    : BasicLayer(name, "Tile", params, networkParameters)
    , mDimension(params.dim)
    , mRepeatDepth(params.mRepeatDepth)
    , mRepeatHeight(params.mRepeatHeight)
    , mRepeatWidth(params.mRepeatWidth)
{
    auto prefix = mTypeName + "[" + mName + "::ctor]: ";

    if (mDimension == raul::Dimension::Batch)
    {
        THROW(mTypeName, mName, "batch dimension not supported");
    }

    DECLARE_IMPL(TileLayer, TileLayerCPU<MemoryManager>, TileLayerCPU<MemoryManagerFP16>)

    mNetworkParams.mWorkflow.copyDeclaration(name, mInputs[0], raul::Workflow::Usage::Forward, raul::Workflow::Mode::Read);

    mNetworkParams.mWorkflow.copyDeclaration(name, mInputs[0], mInputs[0].grad(), DEC_BACK_WRIT_ZERO);

    mNetworkParams.mWorkflow.tensorNeeded(name,
                                          mOutputs[0],
                                          raul::WShape{ BS(),
                                                        mNetworkParams.mWorkflow.getDepth(mInputs[0]) * static_cast<size_t>(mRepeatDepth),
                                                        mNetworkParams.mWorkflow.getHeight(mInputs[0]) * static_cast<size_t>(mRepeatHeight),
                                                        mNetworkParams.mWorkflow.getWidth(mInputs[0]) * static_cast<size_t>(mRepeatWidth) },
                                          DEC_FORW_WRIT);

    mNetworkParams.mWorkflow.copyDeclaration(name, mOutputs[0], mOutputs[0].grad(), DEC_BACK_READ);
}

}