// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "TensorLayer.h"

#include "impl/TensorLayerCPU.h"

#include <algorithm>

namespace raul
{
TensorLayer::TensorLayer(const Name& name, const TensorParams& params, NetworkParameters& networkParameters)
    : BasicLayer(name, "Tensor", params, networkParameters, { false, true })
{
    if (!mInputs.empty())
    {
        THROW("TensorLayer", name, "no inputs expected");
    }

    if (std::any_of(mOutputs.begin(), mOutputs.end(), [](const auto& s) { return s.empty(); }))
    {
        THROW("TensorLayer", name, "empty output name");
    }

    DECLARE_IMPL(TensorLayer, TensorLayerCPU<MemoryManager>, TensorLayerCPU<MemoryManagerFP16>)

    mInit = params.init;
    mInitValue = params.initValue;

    for (const auto& out : mOutputs)
    {
        mNetworkParams.mWorkflow.tensorNeeded(mName, out, params.shape, params.usage, params.mode, params.isOptimizeGraph, params.isOptimizeMem, params.isTrainable, params.isZero, params.isCompress);
        if (params.isTrainable)
        {
            mNetworkParams.mWorkflow.tensorNeeded(mName, out.grad(), params.shape, DEC_TRAINABLE_GRAD);
        }
    }
}

} // namespace raul