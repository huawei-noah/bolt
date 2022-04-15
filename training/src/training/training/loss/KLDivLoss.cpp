// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "KLDivLoss.h"

#include <algorithm>

#include <training/api/API.h>
#include <training/common/MemoryManager.h>
#include <training/opencl/GemmGPU.h>

namespace raul
{

KLDivLoss::KLDivLoss(const Name& name, const LossParams& params, NetworkParameters& networkParameters)
    : BasicLayer(name, "KLDivLoss", params, networkParameters)
    , mIsFinal(params.mIsFinal)
{
    auto prefix = mTypeName + "[" + mName + "::ctor]: ";

    if (mInputs.size() != 2 && mInputs.size() != 3)
    {
        THROW("KLDivLoss", name, "wrong number of input names");
    }
    if (mOutputs.size() != 1)
    {
        THROW("KLDivLoss", name, "wrong number of output names");
    }

    mInputName = mInputs[0];
    mLabelName = mInputs[1];
    mOutputName = mOutputs[0];

    if (params.reduction != LossParams::Reduction::None || mInputs.size() == 3)
    {
        LossParams paramsWrap = params;
        paramsWrap.getOutputs()[0] = mOutputName / "Wrap";
        mInputName = paramsWrap.getOutputs()[0];

        wrapper = std::make_unique<LossWrapper<KLDivLoss>>(mName, paramsWrap, networkParameters);

        mNetworkParams.mWorkflow.copyDeclaration(mName, mInputName, raul::Workflow::Usage::Forward, raul::Workflow::Mode::Read);

        mNetworkParams.mWorkflow.copyDeclaration(mName, mInputName, mOutputName, DEC_FORW_WRIT_NOMEMOPT);
    }
    else
    {
        mNetworkParams.mWorkflow.copyDeclaration(mName, mInputName, raul::Workflow::Usage::Forward, raul::Workflow::Mode::Read);

        mNetworkParams.mWorkflow.copyDeclaration(mName, mInputName, mInputName.grad(), DEC_BACK_WRIT_ZERO);

        mNetworkParams.mWorkflow.copyDeclaration(mName, mInputName, mOutputName, DEC_FORW_WRIT_NOMEMOPT);

        mNetworkParams.mWorkflow.copyDeclaration(mName, mInputName, mOutputName.grad(), DEC_BACK_READ);

        mNetworkParams.mWorkflow.copyDeclaration(mName, mLabelName, raul::Workflow::Usage::ForwardAndBackward, raul::Workflow::Mode::Read);
    }
    mDepth = mNetworkParams.mWorkflow.getDepth(mInputName);
    mHeight = mNetworkParams.mWorkflow.getHeight(mInputName);
    mWidth = mNetworkParams.mWorkflow.getWidth(mInputName);
}

void KLDivLoss::initNotBSTensors()
{
    if (mNetworkParams.mWorkflow.getExecutionTarget() == ExecutionTarget::GPU)
    {
        if (!mNetworkParams.mWorkflow.getKernelManager().hasKernel(mTypeName, "klDivLossForward"))
        {
            const std::string source =
#include <training/opencl/kernels/kl_div_loss.cl>
                ;
            mNetworkParams.mWorkflow.getKernelManager().registerProgram(mTypeName, source);
        }
    }
}

void KLDivLoss::forwardComputeImpl(NetworkMode mode)
{
    auto& work = mNetworkParams.mWorkflow;

    if (wrapper)
    {
        if (work.getExecutionTarget() == ExecutionTarget::CPU)
        {
            Tensor& output = mNetworkParams.mMemoryManager[mOutputName];
            const Tensor& input = mNetworkParams.mMemoryManager[mInputName];
            output = TORANGE(input);
        }
        else if (work.getExecutionTarget() == ExecutionTarget::GPU)
        {
            mNetworkParams.mMemoryManagerGPU[mOutputName] = mNetworkParams.mMemoryManagerGPU[mInputName];
        }
        else if (work.getExecutionTarget() == ExecutionTarget::CPUFP16)
        {
            auto& output = work.getMemoryManager<MemoryManagerFP16>()[mOutputName];
            const auto& input = work.getMemoryManager<MemoryManagerFP16>()[mInputName];
            output = TORANGE_FP16(input);
        }
        else
        {
            THROW(mTypeName, mName, "unsupported execution target");
        }
    }
    else
    {
        if (mode == raul::NetworkMode::Test)
        {
            return;
        }

        if (work.getExecutionTarget() == ExecutionTarget::CPU)
        {
            Tensor& output = mNetworkParams.mMemoryManager[mOutputName];
            output = 0.0_dt;

            const Tensor& inputs = mNetworkParams.mMemoryManager[mInputName];
            const Tensor& targets = mNetworkParams.mMemoryManager[mLabelName];
#if defined(_OPENMP)
#pragma omp parallel for
#endif
            for (size_t q = 0; q < inputs.size(); ++q)
            {
                if (targets[q] > 0)
                {
                    output[q] = targets[q] * (log(targets[q]) - inputs[q]);
                }
            }
        }
        else if (work.getExecutionTarget() == ExecutionTarget::GPU)
        {
            raul::TensorGPU& output = mNetworkParams.mMemoryManagerGPU(mOutputName);
            const raul::TensorGPU& input = mNetworkParams.mMemoryManagerGPU(mInputName);
            const raul::TensorGPU& target = mNetworkParams.mMemoryManagerGPU(mLabelName);
            // Get kernel
            auto kldivForwardKernel = work.getKernelManager().getKernel(mTypeName, "klDivLossForward", mTypeName + "[" + mName + "::forwardComputeImpl]");
            work.getKernelManager().callKernel(kldivForwardKernel,
                                               cl::NDRange{ (mWidth + 3) / 4, mHeight, work.getBatchSize() * mDepth },
                                               mTypeName + "[" + mName + "::forwardComputeImpl]",
                                               (cl_int)mHeight,
                                               (cl_int)mWidth,
                                               (cl_int)mHeight,
                                               (cl_int)mWidth,
                                               0,
                                               0,
                                               (cl_int)mHeight,
                                               (cl_int)mWidth,
                                               0,
                                               0,
                                               input.getBuffer(),
                                               target.getBuffer(),
                                               output.getBuffer());
        }
        else if (work.getExecutionTarget() == ExecutionTarget::CPUFP16)
        {
            auto& output = work.getMemoryManager<MemoryManagerFP16>()[mOutputName];
            output = 0.0_hf;

            const auto& inputs = work.getMemoryManager<MemoryManagerFP16>()[mInputName];
            const auto& targets = work.getMemoryManager<MemoryManagerFP16>()[mLabelName];
#if defined(_OPENMP)
#pragma omp parallel for
#endif
            for (size_t q = 0; q < inputs.size(); ++q)
            {
                if (targets[q] > 0)
                {
                    output[q] = targets[q] * (TOHTYPE(log(TODTYPE(targets[q]))) - inputs[q]);
                }
            }
        }
        else
        {
            THROW(mTypeName, mName, "unsupported execution target");
        }
    }
}

void KLDivLoss::backwardComputeImpl()
{
    auto& work = mNetworkParams.mWorkflow;

    if (!wrapper)
    {
        if (work.getExecutionTarget() == ExecutionTarget::CPU)
        {
            Tensor& prevLayerDelta = mNetworkParams.mMemoryManager[mInputName.grad()];

            const Tensor& targets = mNetworkParams.mMemoryManager[mLabelName];

            const Tensor& deltas = mNetworkParams.mMemoryManager[mOutputName.grad()];
            if (deltas.getShape() != prevLayerDelta.getShape())
            {
                if (!deltas.isBroadcastableTo(prevLayerDelta.getShape()))
                {
                    THROW(mTypeName, mName, "bad incoming deltas shape");
                }
                auto deltas_viewer = deltas.getBroadcastedViewer(prevLayerDelta.getShape());
#if defined(_OPENMP)
#pragma omp parallel for
#endif
                for (size_t q = 0; q < prevLayerDelta.size(); ++q)
                {
                    prevLayerDelta[q] -= targets[q] * deltas_viewer[q];
                }
            }
            else
            {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
                for (size_t q = 0; q < prevLayerDelta.size(); ++q)
                {
                    prevLayerDelta[q] -= targets[q] * deltas[q];
                }
            }
        }
        else if (work.getExecutionTarget() == ExecutionTarget::GPU)
        {
            const raul::TensorGPU& target = mNetworkParams.mMemoryManagerGPU(mLabelName);
            const raul::TensorGPU& deltas = mNetworkParams.mMemoryManagerGPU(mOutputName.grad());
            raul::TensorGPU& prevLayerDelta = mNetworkParams.mMemoryManagerGPU(mInputName.grad());
            // Get kernel
            auto kldivBackwardKernel = work.getKernelManager().getKernel(mTypeName, "klDivLossBackward", mTypeName + "[" + mName + "::BackwardComputeImpl]");
            work.getKernelManager().callKernel(kldivBackwardKernel,
                                               cl::NDRange{ (mWidth + 3) / 4, mHeight, work.getBatchSize() * mDepth },
                                               mTypeName + "[" + mName + "::backwardComputeImpl]",
                                               (cl_int)mHeight,
                                               (cl_int)mWidth,
                                               (cl_int)mHeight,
                                               (cl_int)mWidth,
                                               0,
                                               0,
                                               (cl_int)mHeight,
                                               (cl_int)mWidth,
                                               0,
                                               0,
                                               target.getBuffer(),
                                               deltas.getBuffer(),
                                               prevLayerDelta.getBuffer());
        }
        else if (work.getExecutionTarget() == ExecutionTarget::CPUFP16)
        {
            auto& prevLayerDelta = work.getMemoryManager<MemoryManagerFP16>()[mInputName.grad()];

            const auto& targets = work.getMemoryManager<MemoryManagerFP16>()[mLabelName];

            const auto& deltas = work.getMemoryManager<MemoryManagerFP16>()[mOutputName.grad()];
            if (deltas.getShape() != prevLayerDelta.getShape())
            {
                if (!deltas.isBroadcastableTo(prevLayerDelta.getShape()))
                {
                    THROW(mTypeName, mName, "bad incoming deltas shape");
                }
                auto deltas_viewer = deltas.getBroadcastedViewer(prevLayerDelta.getShape());
#if defined(_OPENMP)
#pragma omp parallel for
#endif
                for (size_t q = 0; q < prevLayerDelta.size(); ++q)
                {
                    prevLayerDelta[q] -= targets[q] * deltas_viewer[q];
                }
            }
            else
            {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
                for (size_t q = 0; q < prevLayerDelta.size(); ++q)
                {
                    prevLayerDelta[q] -= targets[q] * deltas[q];
                }
            }
        }
        else
        {
            THROW(mTypeName, mName, "unsupported execution target");
        }
    }
    else if (!mIsFinal)
    {
        if (work.getExecutionTarget() == ExecutionTarget::CPU)
        {
            const Tensor& deltas = mNetworkParams.mMemoryManager[mOutputName.grad()];
            Tensor& prevLayerDelta = mNetworkParams.mMemoryManager[mInputName.grad()];

            prevLayerDelta += deltas;
        }
        else if (work.getExecutionTarget() == ExecutionTarget::GPU)
        {
            auto& prevLayerDelta = mNetworkParams.mWorkflow.getMemoryManager<MemoryManagerGPU>()(mInputName.grad());
            const auto& delta = mNetworkParams.mWorkflow.getMemoryManager<MemoryManagerGPU>()(mOutputName.grad());

            gpu::axpy(mNetworkParams.mWorkflow.getKernelManager(), mName / "backward", delta.size(), 1.0_dt, delta.getBuffer(), 1, prevLayerDelta.getBuffer(), 1, 0, 0);
        }
        else if (work.getExecutionTarget() == ExecutionTarget::CPUFP16)
        {
            const auto& deltas = work.getMemoryManager<MemoryManagerFP16>()[mOutputName.grad()];
            auto& prevLayerDelta = work.getMemoryManager<MemoryManagerFP16>()[mInputName.grad()];

            prevLayerDelta += deltas;
        }
        else
        {
            THROW(mTypeName, mName, "unsupported execution target");
        }
    }
}

} // namespace raul
