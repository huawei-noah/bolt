// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "NegativeLogLikelihoodLoss.h"

#include <algorithm>

#include <training/api/API.h>
#include <training/opencl/GemmGPU.h>

namespace raul
{

NLLLoss::NLLLoss(const Name& name, const LossParams& params, NetworkParameters& networkParameters)
    : BasicLayer(name, "NLLLoss", params, networkParameters)
    , mIsFinal(params.mIsFinal)
{
    MEASURE_BLOCK(mTypeName + "[" + name + "::ctor]")
    auto prefix = mTypeName + "[" + mName + "::ctor]: ";

    if (mInputs.size() != 2 && mInputs.size() != 3)
    {
        THROW(mTypeName, mName, "wrong number of input names");
    }
    if (mOutputs.size() != 1)
    {
        THROW(mTypeName, mName, "wrong number of output names");
    }

    mInputName = mInputs[0];
    mLabelName = mInputs[1];
    mOutputName = mOutputs[0];

    if (params.reduction != LossParams::Reduction::None || mInputs.size() == 3)
    {
        LossParams paramsWrap = params;
        paramsWrap.getOutputs()[0] = mOutputName / "Wrap";
        mInputName = paramsWrap.getOutputs()[0];

        wrapper = std::make_shared<LossWrapper<NLLLoss>>(name, paramsWrap, networkParameters);

        mNetworkParams.mWorkflow.copyDeclaration(mName, mInputName, raul::Workflow::Usage::Forward, raul::Workflow::Mode::Read);

        mNetworkParams.mWorkflow.copyDeclaration(mName, mInputName, mOutputName, DEC_FORW_WRIT_NOMEMOPT);
    }
    else
    {
        mNetworkParams.mWorkflow.copyDeclaration(mName, mInputName, raul::Workflow::Usage::Forward, raul::Workflow::Mode::Read);

        mNetworkParams.mWorkflow.copyDeclaration(mName, mInputName, mInputName.grad(), DEC_BACK_WRIT_ZERO);

        mNetworkParams.mWorkflow.copyDeclaration(mName, mInputName, mOutputName, DEC_FORW_WRIT_NOMEMOPT);

        mNetworkParams.mWorkflow.copyDeclaration(mName, mOutputName, mOutputName.grad(), DEC_BACK_READ);

        mNetworkParams.mWorkflow.copyDeclaration(mName, mLabelName, raul::Workflow::Usage::ForwardAndBackward, raul::Workflow::Mode::Read);
    }
    mDepth = mNetworkParams.mWorkflow.getDepth(mInputName);
    mHeight = mNetworkParams.mWorkflow.getHeight(mInputName);
    mWidth = mNetworkParams.mWorkflow.getWidth(mInputName);
}

void NLLLoss::initNotBSTensors()
{
    if (mNetworkParams.mWorkflow.getExecutionTarget() == ExecutionTarget::GPU)
    {
        if (!mNetworkParams.mWorkflow.getKernelManager().hasKernel(mTypeName / "forward"))
        {
            std::string source = "__kernel void nllForward(const int h, "
                                 "    const int w, "
                                 "    const int ih_str, "
                                 "    const int iw_str, "
                                 "    const int ih_off, "
                                 "    const int iw_off, "
                                 "    const int oh_str, "
                                 "    const int ow_str, "
                                 "    const int oh_off, "
                                 "    const int ow_off, "
                                 "    __global T *input, "
                                 "    __global T *target, "
                                 "    __global T *output) "
                                 "{ "
                                 "    int idx = get_global_id(0); "
                                 "    int idy = get_global_id(1); "
                                 "    int idz = get_global_id(2); "
                                 "    if (idx >= h || idy >= w) { "
                                 "        return; "
                                 "    } "
                                 "    T4 data; "
                                 "    T4 labels; "
                                 "    int in_off = (idz * iw_str + idy + iw_off) * ih_str + idx + ih_off; "
                                 "    data = vload4(in_off, input); "
                                 "    labels = vload4(in_off, target); "
                                 "    data.x = -labels.x * data.x; "
                                 "    data.y = -labels.y * data.y; "
                                 "    data.z = -labels.z * data.z; "
                                 "    data.w = -labels.w * data.w; "
                                 "    int out_off = (idz * ow_str + idy + ow_off) * oh_str + idx + oh_off; "
                                 "    vstore4(data, out_off, output); "
                                 "} ";
            mNetworkParams.mWorkflow.getKernelManager().registerProgram(mTypeName / "forward", source);
        }
        if (!mNetworkParams.mWorkflow.getKernelManager().hasKernel(mTypeName / "backward"))
        {
            std::string source = "__kernel void nllBackward(const int h, "
                                 "    const int w, "
                                 "    const int ih_str, "
                                 "    const int iw_str, "
                                 "    const int ih_off, "
                                 "    const int iw_off, "
                                 "    const int oh_str, "
                                 "    const int ow_str, "
                                 "    const int oh_off, "
                                 "    const int ow_off, "
                                 "    __global T *target, "
                                 "    __global T *deltas, "
                                 "    __global T *prevLayerDelta) "
                                 "{ "
                                 "    int idx = get_global_id(0); "
                                 "    int idy = get_global_id(1); "
                                 "    int idz = get_global_id(2); "
                                 "    if (idx >= h || idy >= w) { "
                                 "        return; "
                                 "    } "
                                 "    T4 labels; "
                                 "    T4 del; "
                                 "    int in_off = (idz * iw_str + idy + iw_off) * ih_str + idx + ih_off; "
                                 "    labels = vload4(in_off, target); "
                                 "    del = vload4(in_off, deltas); "
                                 "    labels.x = -labels.x * del.x; "
                                 "    labels.y = -labels.y * del.y; "
                                 "    labels.z = -labels.z * del.z; "
                                 "    labels.w = -labels.w * del.w; "
                                 "    T4 prev; "
                                 "    prev = vload4(in_off, prevLayerDelta); "
                                 "    labels.x += prev.x; "
                                 "    labels.y += prev.y; "
                                 "    labels.z += prev.z; "
                                 "    labels.w += prev.w; "
                                 "    int out_off = (idz * ow_str + idy + ow_off) * oh_str + idx + oh_off; "
                                 "    vstore4(labels, out_off, prevLayerDelta); "
                                 "} ";
            mNetworkParams.mWorkflow.getKernelManager().registerProgram(mTypeName / "backward", source);
        }
    }
}

void NLLLoss::forwardComputeImpl(NetworkMode mode)
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
        if (mode == NetworkMode::Test)
        {
            return;
        }

        if (work.getExecutionTarget() == ExecutionTarget::CPU)
        {
            Tensor& output = mNetworkParams.mMemoryManager[mOutputName];

            Tensor& inputs = mNetworkParams.mMemoryManager[mInputName];
            const Tensor& targets = mNetworkParams.mMemoryManager[mLabelName];

            for (size_t q = 0; q < output.size(); ++q)
            {
                output[q] = -targets[q] * inputs[q];
            }
        }
        else if (work.getExecutionTarget() == ExecutionTarget::GPU)
        {
            raul::TensorGPU& output = mNetworkParams.mMemoryManagerGPU(mOutputName);
            const raul::TensorGPU& input = mNetworkParams.mMemoryManagerGPU(mInputName);
            const raul::TensorGPU& target = mNetworkParams.mMemoryManagerGPU(mLabelName);
            // Get kernel
            auto nllForwardKernel = work.getKernelManager().getKernel(mTypeName / "forward", "nllForward", mTypeName + "[" + mName + "::forwardComputeImpl]");
            work.getKernelManager().callKernel(nllForwardKernel,
                                               cl::NDRange{ mHeight, mWidth, (work.getBatchSize() * mDepth + 3) / 4 },
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

            auto& inputs = work.getMemoryManager<MemoryManagerFP16>()[mInputName];
            const auto& targets = work.getMemoryManager<MemoryManagerFP16>()[mLabelName];

            for (size_t q = 0; q < output.size(); ++q)
            {
                output[q] = -targets[q] * inputs[q];
            }
        }
        else
        {
            THROW(mTypeName, mName, "unsupported execution target");
        }
    }
}

void NLLLoss::backwardComputeImpl()
{
    auto& work = mNetworkParams.mWorkflow;

    if (!wrapper)
    {
        if (work.getExecutionTarget() == ExecutionTarget::CPU)
        {
            Tensor& prevLayerDelta = mNetworkParams.mMemoryManager[mInputName.grad()];
            const Tensor& targets = mNetworkParams.mMemoryManager[mLabelName];

            // Get global gradient if exists
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
                for (size_t q = 0; q < targets.size(); ++q)
                {
                    prevLayerDelta[q] -= targets[q] * deltas_viewer[q];
                }
            }
            else
            {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
                for (size_t q = 0; q < targets.size(); ++q)
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
            auto nllBackwardKernel = work.getKernelManager().getKernel(mTypeName / "backward", "nllBackward", mTypeName + "[" + mName + "::BackwardComputeImpl]");
            work.getKernelManager().callKernel(nllBackwardKernel,
                                               cl::NDRange{ mHeight, mWidth, (work.getBatchSize() * mDepth + 3) / 4 },
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

            // Get global gradient if exists
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
                for (size_t q = 0; q < targets.size(); ++q)
                {
                    prevLayerDelta[q] -= targets[q] * deltas_viewer[q];
                }
            }
            else
            {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
                for (size_t q = 0; q < targets.size(); ++q)
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
