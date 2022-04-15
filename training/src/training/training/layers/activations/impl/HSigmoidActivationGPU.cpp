// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "HSigmoidActivationGPU.h"
#include "../HSigmoidActivation.h"

#include <algorithm>

namespace raul
{

void HSigmoidActivationGPU::forwardComputeImpl(NetworkMode)
{
    Workflow& work = mLayer.mNetworkParams.mWorkflow;
    auto& output = mLayer.mNetworkParams.mMemoryManagerGPU(mLayer.mOutputName);
    const auto& input = mLayer.mNetworkParams.mMemoryManagerGPU(mLayer.mInputName);
    Common::hsigmoidForward(
        &work.getKernelManager(), mLayer.mTypeName + "[" + mLayer.mName + "::forwardComputeImpl]", work.getBatchSize(), mLayer.mDepth, mLayer.mHeight, mLayer.mWidth, input, output);
}

void HSigmoidActivationGPU::backwardComputeImpl()
{
    ////if (mLayer.mNetworkParams.isGradNeeded(mLayer.mInputName))
    {

        Workflow& work = mLayer.mNetworkParams.mWorkflow;
        const auto& input = mLayer.mNetworkParams.mMemoryManagerGPU(mLayer.mInputName);
        const auto& deltas = mLayer.mNetworkParams.mMemoryManagerGPU(mLayer.mOutputName.grad());
        auto& prevLayerDelta = mLayer.mNetworkParams.mMemoryManagerGPU(mLayer.mInputName.grad());

        dtype leftDivisor = 0.0_dt;
        dtype rightDivisor = 0.0_dt;
        switch (mLayer.p3PointVal)
        {
            case Limit::Left:
                leftDivisor = 1.0_dt / 6.0_dt;
                break;
            case Limit::Middle:
                leftDivisor = 1.0_dt / 12.0_dt;
                break;
            case Limit::Right:
                break;
        }
        switch (mLayer.m3PointVal)
        {
            case Limit::Left:
                break;
            case Limit::Middle:
                rightDivisor = 1.0_dt / 12.0_dt;
                break;
            case Limit::Right:
                rightDivisor = 1.0_dt / 6.0_dt;
                break;
        }
        Common::hsigmoidBackward(&work.getKernelManager(),
                                 mLayer.mTypeName + "[" + mLayer.mName + "::bakwardComputeImpl]",
                                 work.getBatchSize() * mLayer.mDepth * mLayer.mHeight * mLayer.mWidth,
                                 leftDivisor,
                                 rightDivisor,
                                 input,
                                 deltas,
                                 prevLayerDelta);
    }
}

} // namespace raul