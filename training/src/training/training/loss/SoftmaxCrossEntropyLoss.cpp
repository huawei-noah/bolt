// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "SoftmaxCrossEntropyLoss.h"

#include <training/api/API.h>
#include <training/common/MemoryManager.h>
#include <training/opencl/GemmGPU.h>

namespace raul
{

SoftmaxCrossEntropyLoss::SoftmaxCrossEntropyLoss(const Name& name, const LossParams& params, NetworkParameters& networkParameters)
    : BasicLayer(name, "SoftmaxCrossEntropyLoss", params, networkParameters)
    , mIsFinal(params.mIsFinal)
{
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

        wrapper = std::make_unique<LossWrapper<SoftmaxCrossEntropyLoss>>(mName, paramsWrap, networkParameters);

        mNetworkParams.mWorkflow.copyDeclaration(mName, mInputName, raul::Workflow::Usage::Forward, raul::Workflow::Mode::Read);
        mNetworkParams.mWorkflow.copyDeclaration(mName, mInputName, mOutputName, DEC_FORW_WRIT_NOMEMOPT);
        if (!mIsFinal)
        {
            mNetworkParams.mWorkflow.copyDeclaration(mName, mOutputName, mOutputName.grad(), DEC_BACK_READ);
            mNetworkParams.mWorkflow.copyDeclaration(mName, mInputName, mInputName.grad(), DEC_BACK_WRIT_ZERO);
        }
    }
    else
    {
        if (mNetworkParams.mWorkflow.getExecutionTarget() == ExecutionTarget::GPU)
        {
            mNetworkParams.mWorkflow.copyDeclaration(mName, mInputName, raul::Workflow::Usage::ForwardAndBackward, raul::Workflow::Mode::Read);
        }
        else
        {
            mNetworkParams.mWorkflow.copyDeclaration(mName, mInputName, raul::Workflow::Usage::Forward, raul::Workflow::Mode::Read);
        }
        mNetworkParams.mWorkflow.copyDeclaration(mName, mInputName, mInputName.grad(), DEC_BACK_WRIT_ZERO);
        mNetworkParams.mWorkflow.copyDeclaration(mName, mInputName, mOutputName, DEC_FORW_WRIT_NOMEMOPT);
        mNetworkParams.mWorkflow.copyDeclaration(mName, mOutputName, mOutputName.grad(), DEC_BACK_READ);
        mNetworkParams.mWorkflow.copyDeclaration(mName, mLabelName, raul::Workflow::Usage::ForwardAndBackward, raul::Workflow::Mode::Read);

        // Temporal tensor to reduce calculations
        mNetworkParams.mWorkflow.copyDeclaration(mName, mInputName, mInputName / "AFTER_SOFTMAX", DEC_FRBC_WRIT_NOMEMOPT);
    }
    mDepth = mNetworkParams.mWorkflow.getDepth(mInputName);
    mHeight = mNetworkParams.mWorkflow.getHeight(mInputName);
    mWidth = mNetworkParams.mWorkflow.getWidth(mInputName);
}

void SoftmaxCrossEntropyLoss::initNotBSTensors()
{
    if (mNetworkParams.mWorkflow.getExecutionTarget() == ExecutionTarget::GPU)
    {
        if (!mNetworkParams.mWorkflow.getKernelManager().hasKernel(mTypeName / "forward"))
        {
            std::string source = "__kernel void softmaxCEForward(const int x, "
                                 "    const int y, "
                                 "    const int z, "
                                 "    const int externalDimSize, "
                                 "    const int internalDimSize, "
                                 "    __global T *input, "
                                 "    __global T *targets, "
                                 "    __global T *output) "
                                 "{ "
                                 "    int idx = get_global_id(0); "
                                 "    int idy = get_global_id(1); "
                                 "    int idz = get_global_id(2); "
                                 "    if (x == idx && y == idy && z == idz) "
                                 "    { "
                                 "        for (int q = 0; q < externalDimSize; ++q) "
                                 "        { "
                                 "            T sum = 0.0f; "
                                 "            for (int i = 0; i < internalDimSize; ++i) "
                                 "            { "
                                 "                sum += exp(input[q * internalDimSize + i]); "
                                 "            } "
                                 "            for (int i = 0; i < internalDimSize; ++i) "
                                 "            { "
                                 "                output[q * internalDimSize + i] = -targets[q * internalDimSize + i] * log(exp(input[q * internalDimSize + i]) / sum); "
                                 "            } "
                                 "        } "
                                 "    } "
                                 "} ";
            mNetworkParams.mWorkflow.getKernelManager().registerProgram(mTypeName / "forward", source);
        }
        if (!mNetworkParams.mWorkflow.getKernelManager().hasKernel(mTypeName / "backward"))
        {
            std::string source =
                "__kernel void softmaxCEBackward(const int x, "
                "    const int y, "
                "    const int z, "
                "    const int externalDimSize, "
                "    const int internalDimSize, "
                "    __global T *input, "
                "    __global T *targets, "
                "    __global T *deltas, "
                "    __global T *prevLayerDelta) "
                "{ "
                "    int idx = get_global_id(0); "
                "    int idy = get_global_id(1); "
                "    int idz = get_global_id(2); "
                "    if (x == idx && y == idy && z == idz) "
                "    { "
                "        for (int q = 0; q < externalDimSize; ++q) "
                "        { "
                "            T sum = 0.0f; "
                "            for (int i = 0; i < internalDimSize; ++i) "
                "            { "
                "                sum += exp(input[q * internalDimSize + i]); "
                "            } "
                "            for (int i = 0; i < internalDimSize; ++i) "
                "            { "
                "                prevLayerDelta[q * internalDimSize + i] += (exp(input[q * internalDimSize + i]) / sum - targets[q * internalDimSize + i]) * deltas[q * internalDimSize + i]; "
                "            } "
                "        } "
                "    } "
                "} ";
            mNetworkParams.mWorkflow.getKernelManager().registerProgram(mTypeName / "backward", source);
        }
    }
}

void SoftmaxCrossEntropyLoss::forwardComputeImpl(NetworkMode mode)
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
        else if (mNetworkParams.mWorkflow.getExecutionTarget() == ExecutionTarget::GPU)
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
            const Tensor& input = mNetworkParams.mMemoryManager[mInputName];

            const Tensor& target = mNetworkParams.mMemoryManager[mLabelName];

            size_t size = input.getBatchSize() * input.getDepth() * input.getHeight();
            auto input2D = input.reshape(yato::dims(size, input.getWidth()));
            auto output2D = output.reshape(yato::dims(size, input.getWidth()));
            auto target2D = target.reshape(yato::dims(size, input.getWidth()));

            Tensor& inputTemp = mNetworkParams.mMemoryManager[mInputName / "AFTER_SOFTMAX"];
            auto inputTemp2D = inputTemp.reshape(yato::dims(size, input.getWidth()));

#if defined(_OPENMP)
#pragma omp parallel for
#endif
            for (size_t q = 0; q < size; ++q)
            {
                dtype sum = 0.0_dt;
                dtype max = (*std::max_element(input.begin() + q * input.getWidth(), input.begin() + (q + 1) * input.getWidth()));

                for (size_t i = 0; i < input.getWidth(); ++i)
                {
                    inputTemp2D[q][i] = std::exp(input2D[q][i] - max);
                    sum += inputTemp2D[q][i];
                }

                for (size_t i = 0; i < input.getWidth(); ++i)
                {
                    inputTemp2D[q][i] /= sum;
                    output2D[q][i] = -target2D[q][i] * std::log(inputTemp2D[q][i]);
                }
            }
        }
        else if (mNetworkParams.mWorkflow.getExecutionTarget() == ExecutionTarget::GPU)
        {
            raul::TensorGPU& output = mNetworkParams.mMemoryManagerGPU(mOutputName);
            const raul::TensorGPU& input = mNetworkParams.mMemoryManagerGPU(mInputName);
            const raul::TensorGPU& target = mNetworkParams.mMemoryManagerGPU(mLabelName);
            auto softmaxCEForwardKernel = work.getKernelManager().getKernel(mTypeName / "forward", "softmaxCEForward", mTypeName + "[" + mName + "::forwardComputeImpl]");
            size_t externalDimSize = work.getBatchSize() * mDepth * mHeight;
            size_t internalDimSize = mWidth;
            work.getKernelManager().callKernel(softmaxCEForwardKernel,
                                               cl::NDRange{ mHeight, mWidth, work.getBatchSize() * mDepth },
                                               mTypeName + "[" + mName + "::forwardComputeImpl]",
                                               0,
                                               0,
                                               0,
                                               (cl_int)externalDimSize,
                                               (cl_int)internalDimSize,
                                               input.getBuffer(),
                                               target.getBuffer(),
                                               output.getBuffer());
        }
        else if (work.getExecutionTarget() == ExecutionTarget::CPUFP16)
        {
            auto& output = work.getMemoryManager<MemoryManagerFP16>()[mOutputName];
            const auto& input = work.getMemoryManager<MemoryManagerFP16>()[mInputName];

            const auto& target = work.getMemoryManager<MemoryManagerFP16>()[mLabelName];

            size_t size = input.getBatchSize() * input.getDepth() * input.getHeight();
            auto input2D = input.reshape(yato::dims(size, input.getWidth()));
            auto output2D = output.reshape(yato::dims(size, input.getWidth()));
            auto target2D = target.reshape(yato::dims(size, input.getWidth()));

            auto& inputTemp = work.getMemoryManager<MemoryManagerFP16>()[mInputName / "AFTER_SOFTMAX"];
            auto inputTemp2D = inputTemp.reshape(yato::dims(size, input.getWidth()));

#if defined(_OPENMP)
#pragma omp parallel for
#endif
            for (size_t q = 0; q < size; ++q)
            {
                dtype sum = 0.0_dt;
                half max = (*std::max_element(input.begin() + q * input.getWidth(), input.begin() + (q + 1) * input.getWidth()));

                for (size_t i = 0; i < input.getWidth(); ++i)
                {
                    inputTemp2D[q][i] = TOHTYPE(std::exp(TODTYPE(input2D[q][i] - max)));
                    sum += TODTYPE(inputTemp2D[q][i]);
                }

                for (size_t i = 0; i < input.getWidth(); ++i)
                {
                    inputTemp2D[q][i] = TOHTYPE(TODTYPE(inputTemp2D[q][i]) / sum);
                    output2D[q][i] = -target2D[q][i] * TOHTYPE(std::log(TODTYPE(inputTemp2D[q][i])));
                }
            }
        }
        else
        {
            THROW(mTypeName, mName, "unsupported execution target");
        }
    }
}

void SoftmaxCrossEntropyLoss::backwardComputeImpl()
{
    auto& work = mNetworkParams.mWorkflow;

    if (!wrapper)
    {
        // if (!mNetworkParams.isGradNeeded(mInputName))
        {
            // return;
        }

        if (work.getExecutionTarget() == ExecutionTarget::CPU)
        {
            Tensor& prevLayerDelta = mNetworkParams.mMemoryManager[mInputName.grad()];

            const Tensor& targets = mNetworkParams.mMemoryManager[mLabelName];
            const Tensor& deltas = mNetworkParams.mMemoryManager[mOutputName.grad()];
            const Tensor& inputTemp = mNetworkParams.mMemoryManager[mInputName / "AFTER_SOFTMAX"];

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
                    prevLayerDelta[q] += (inputTemp[q] - targets[q]) * deltas_viewer[q];
                }
            }
            else
            {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
                for (size_t q = 0; q < prevLayerDelta.size(); ++q)
                {
                    prevLayerDelta[q] += (inputTemp[q] - targets[q]) * deltas[q];
                }
            }
        }
        else if (mNetworkParams.mWorkflow.getExecutionTarget() == ExecutionTarget::GPU)
        {
            const raul::TensorGPU& input = mNetworkParams.mMemoryManagerGPU(mInputName);
            const raul::TensorGPU& target = mNetworkParams.mMemoryManagerGPU(mLabelName);
            const raul::TensorGPU& deltas = mNetworkParams.mMemoryManagerGPU(mOutputName.grad());
            raul::TensorGPU& prevLayerDelta = mNetworkParams.mMemoryManagerGPU(mInputName.grad());
            auto softmaxCEBackwardKernel = work.getKernelManager().getKernel(mTypeName / "backward", "softmaxCEBackward", mTypeName + "[" + mName + "::backwardComputeImpl]");
            size_t externalDimSize = work.getBatchSize() * mDepth * mHeight;
            size_t internalDimSize = mWidth;
            work.getKernelManager().callKernel(softmaxCEBackwardKernel,
                                               cl::NDRange{ mHeight, mWidth, work.getBatchSize() * mDepth },
                                               mTypeName + "[" + mName + "::backwardComputeImpl]",
                                               0,
                                               0,
                                               0,
                                               (cl_int)externalDimSize,
                                               (cl_int)internalDimSize,
                                               input.getBuffer(),
                                               target.getBuffer(),
                                               deltas.getBuffer(),
                                               prevLayerDelta.getBuffer());
        }
        else if (work.getExecutionTarget() == ExecutionTarget::CPUFP16)
        {
            auto& prevLayerDelta = work.getMemoryManager<MemoryManagerFP16>()[mInputName.grad()];

            const auto& targets = work.getMemoryManager<MemoryManagerFP16>()[mLabelName];
            const auto& deltas = work.getMemoryManager<MemoryManagerFP16>()[mOutputName.grad()];
            const auto& inputTemp = work.getMemoryManager<MemoryManagerFP16>()[mInputName / "AFTER_SOFTMAX"];

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
                    prevLayerDelta[q] += (inputTemp[q] - targets[q]) * deltas_viewer[q];
                }
            }
            else
            {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
                for (size_t q = 0; q < prevLayerDelta.size(); ++q)
                {
                    prevLayerDelta[q] += (inputTemp[q] - targets[q]) * deltas[q];
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
        else if (mNetworkParams.mWorkflow.getExecutionTarget() == ExecutionTarget::GPU)
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

}
