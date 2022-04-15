// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "L1LossGPU.h"
#include "../L1Loss.h"

#include <training/opencl/GPUCommon.h>
#include <training/opencl/GemmGPU.h>

namespace raul
{

void L1LossLayerGPU::initNotBSTensors()
{
    if (!mLayer.mNetworkParams.mWorkflow.getKernelManager().hasKernel(mLayer.mTypeName, "l1Forward"))
    {
        const std::string source =
#include <training/opencl/kernels/l1_loss.cl>
            ;
        mLayer.mNetworkParams.mWorkflow.getKernelManager().registerProgram(mLayer.mTypeName, source);
    }
}

void L1LossLayerGPU::forwardComputeImpl(NetworkMode mode)
{
    auto& work = mLayer.mNetworkParams.mWorkflow;

    if (mLayer.wrapper)
    {
        mLayer.mNetworkParams.mMemoryManagerGPU[mLayer.mOutputName] = mLayer.mNetworkParams.mMemoryManagerGPU[mLayer.mInputName];
    }
    else
    {
        if (mode == NetworkMode::Test)
        {
            return;
        }

        raul::TensorGPU& output = mLayer.mNetworkParams.mMemoryManagerGPU(mLayer.mOutputName);
        const raul::TensorGPU& input = mLayer.mNetworkParams.mMemoryManagerGPU(mLayer.mInputName);
        const raul::TensorGPU& target = mLayer.mNetworkParams.mMemoryManagerGPU(mLayer.mLabelName);
        // Get kernel
        auto mseForwardKernel = work.getKernelManager().getKernel(mLayer.mTypeName, "l1Forward", mLayer.mTypeName + "[" + mLayer.mName + "::forwardComputeImpl]");
        work.getKernelManager().callKernel(mseForwardKernel,
                                           cl::NDRange{ (mLayer.mWidth + 3) / 4, mLayer.mHeight, work.getBatchSize() * mLayer.mDepth },
                                           mLayer.mTypeName + "[" + mLayer.mName + "::forwardComputeImpl]",
                                           (cl_int)mLayer.mHeight,
                                           (cl_int)mLayer.mWidth,
                                           (cl_int)mLayer.mHeight,
                                           (cl_int)mLayer.mWidth,
                                           0,
                                           0,
                                           (cl_int)mLayer.mHeight,
                                           (cl_int)mLayer.mWidth,
                                           0,
                                           0,
                                           input.getBuffer(),
                                           target.getBuffer(),
                                           output.getBuffer());
    }
}

void L1LossLayerGPU::backwardComputeImpl()
{
    auto& work = mLayer.mNetworkParams.mWorkflow;

    if (!mLayer.wrapper)
    {
        const raul::TensorGPU& input = mLayer.mNetworkParams.mMemoryManagerGPU(mLayer.mInputName);
        const raul::TensorGPU& target = mLayer.mNetworkParams.mMemoryManagerGPU(mLayer.mLabelName);
        const raul::TensorGPU& deltas = mLayer.mNetworkParams.mMemoryManagerGPU(mLayer.mOutputName.grad());
        raul::TensorGPU& prevLayerDelta = mLayer.mNetworkParams.mMemoryManagerGPU(mLayer.mInputName.grad());
        // Get kernel
        auto l1BackwardKernel = work.getKernelManager().getKernel(mLayer.mTypeName, "l1Backward", mLayer.mTypeName + "[" + mLayer.mName + "::BackwardComputeImpl]");
        work.getKernelManager().callKernel(l1BackwardKernel,
                                           cl::NDRange{ (mLayer.mWidth + 3) / 4, mLayer.mHeight, work.getBatchSize() * mLayer.mDepth },
                                           mLayer.mTypeName + "[" + mLayer.mName + "::backwardComputeImpl]",
                                           (cl_int)mLayer.mHeight,
                                           (cl_int)mLayer.mWidth,
                                           (cl_int)mLayer.mHeight,
                                           (cl_int)mLayer.mWidth,
                                           0,
                                           0,
                                           (cl_int)mLayer.mHeight,
                                           (cl_int)mLayer.mWidth,
                                           0,
                                           0,
                                           input.getBuffer(),
                                           target.getBuffer(),
                                           deltas.getBuffer(),
                                           prevLayerDelta.getBuffer());
    }
    else if (!mLayer.mIsFinal)
    {
        auto& prevLayerDelta = mLayer.mNetworkParams.mWorkflow.getMemoryManager<MemoryManagerGPU>()(mLayer.mInputName.grad());
        const auto& delta = mLayer.mNetworkParams.mWorkflow.getMemoryManager<MemoryManagerGPU>()(mLayer.mOutputName.grad());

        gpu::axpy(mLayer.mNetworkParams.mWorkflow.getKernelManager(), mLayer.mName / "backward", delta.size(), 1.0_dt, delta.getBuffer(), 1, prevLayerDelta.getBuffer(), 1, 0, 0);
    }
}

} // namespace raul
