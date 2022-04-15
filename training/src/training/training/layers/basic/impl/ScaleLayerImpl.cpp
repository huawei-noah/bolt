// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "ScaleLayerImpl.h"
#include "../ScaleLayer.h"

#include <training/opencl/GemmGPU.h>

namespace raul
{
template<typename MM>
void ScaleLayerImpl<MM>::forwardComputeImpl(NetworkMode)
{
    Workflow& work = mLayer.mNetworkParams.mWorkflow;

    auto& output = work.getMemoryManager<MM>().getTensor(mLayer.mOutputs[0]);
    const auto& input = work.getMemoryManager<MM>().getTensor(mLayer.mInputs[0]);

    size_t n = output.size();
    std::fill(output.begin(), output.end(), TOMMTYPE(0));
    OPENBLAS_CONST dtype sa = mLayer.mScale;
    Common::axpy(&mLayer.mNetworkParams.mWorkflow.getKernelManager(), "", n, sa, input.getBuffer(), 1, output.getBuffer(), 1, 0, 0);
}

template<>
void ScaleLayerImpl<MemoryManagerGPU>::forwardComputeImpl(NetworkMode)
{
    Workflow& work = mLayer.mNetworkParams.mWorkflow;

    auto& output = work.getMemoryManager<MemoryManagerGPU>()(mLayer.mOutputs[0]);
    const auto& input = work.getMemoryManager<MemoryManagerGPU>()(mLayer.mInputs[0]);

    auto caller = mLayer.mTypeName + "[" + mLayer.mName + "::forwardComputeImpl]";

    work.getKernelManager().fillBuffer(output.getBuffer(), 0_dt, caller);
    size_t n = output.size();
    OPENBLAS_CONST dtype sa = mLayer.mScale;
    Common::axpy(&mLayer.mNetworkParams.mWorkflow.getKernelManager(), caller, n, sa, input.getBuffer(), 1, output.getBuffer(), 1, 0, 0);
}

template<typename MM>
void ScaleLayerImpl<MM>::backwardComputeImpl()
{
    Workflow& work = mLayer.mNetworkParams.mWorkflow;

    const auto& deltas = work.getMemoryManager<MM>().getTensor(mLayer.mOutputs[0].grad());

    // if (mNetworkParams.isGradNeeded(mInputs[0]))
    {
        auto& nabla_tensor = work.getMemoryManager<MM>().getTensor(mLayer.mInputs[0].grad());

        {
            size_t n = nabla_tensor.size();
            OPENBLAS_CONST dtype sa = mLayer.mScale;
            Common::axpy(&mLayer.mNetworkParams.mWorkflow.getKernelManager(), "", n, sa, deltas.getBuffer(), 1, nabla_tensor.getBuffer(), 1, 0, 0);
        }
    }
}

INSTANTIATE_IMPL(ScaleLayerImpl)

} // namespace raul
