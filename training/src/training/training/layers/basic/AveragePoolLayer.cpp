// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "AveragePoolLayer.h"

namespace raul
{

AveragePoolLayer::AveragePoolLayer(const Name& name, const Pool2DParams& params, NetworkParameters& networkParameters)
    : BasicLayer(name, "AveragePooling", params, networkParameters)
    , mKernelWidth(params.kernelWidth)
    , mKernelHeight(params.kernelHeight)
    , mPaddingW(params.paddingW)
    , mPaddingH(params.paddingH)
    , mStrideW(params.strideW)
    , mStrideH(params.strideH)
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

    mInputName = mInputs[0];
    mOutputName = mOutputs[0];

    mInputDepth = mNetworkParams.mWorkflow.getDepth(mInputName);
    mInputHeight = mNetworkParams.mWorkflow.getHeight(mInputName);
    mInputWidth = mNetworkParams.mWorkflow.getWidth(mInputName);

    if ((mPaddingH > mKernelHeight / 2) || (mPaddingW > mKernelWidth / 2))
    {
        THROW(mTypeName, mName, "Padding should be smaller than half of kernel size");
    }
    if (mKernelHeight == 0 || mKernelWidth == 0)
    {
        THROW(mTypeName, mName, "Kernel size can't be null");
    }
    if ((mInputWidth + mPaddingW * 2 < mKernelWidth) || (mInputHeight + mPaddingW * 2 < mKernelHeight))
    {
        THROW(mTypeName, mName, "ImageSize + 2*Padding can't be less than KernelSize");
    }

    mOutputWidth = (mInputWidth + mPaddingW * 2 - mKernelWidth) / mStrideW + 1;
    mOutputHeight = (mInputHeight + mPaddingH * 2 - mKernelHeight) / mStrideH + 1;

    mNetworkParams.mWorkflow.copyDeclaration(mName, mInputName, raul::Workflow::Usage::Forward, raul::Workflow::Mode::Read);

    mNetworkParams.mWorkflow.copyDeclaration(mName, mInputName, mInputName.grad(), DEC_BACK_WRIT_ZERO);

    mNetworkParams.mWorkflow.tensorNeeded(mName, mOutputName, raul::WShape{ BS(), mInputDepth, mOutputHeight, mOutputWidth }, DEC_FORW_WRIT);

    mNetworkParams.mWorkflow.copyDeclaration(mName, mOutputName, mOutputName.grad(), DEC_BACK_READ);
}

void AveragePoolLayer::forwardComputeImpl(NetworkMode)
{
    auto& work = mNetworkParams.mWorkflow;

    if (work.getExecutionTarget() == ExecutionTarget::CPU)
    {
        const size_t batchSize = mNetworkParams.mWorkflow.getBatchSize();

        Tensor& output = mNetworkParams.mMemoryManager[mOutputName];

        const Tensor& inputs = mNetworkParams.mMemoryManager[mInputName];

        auto inputs3D = inputs.reshape(yato::dims(batchSize, mInputDepth, mInputHeight * mInputWidth));
        auto outputs3D = output.reshape(yato::dims(batchSize, mInputDepth, mOutputHeight * mOutputWidth));

        const dtype reciprocalKernelSize = 1.0_dt / static_cast<dtype>(mKernelHeight * mKernelWidth);

        for (size_t b = 0; b < batchSize; ++b)
        {
            for (size_t k = 0; k < mInputDepth; ++k)
            {
                for (size_t i = 0; i < mOutputHeight; ++i)
                {
                    for (size_t j = 0; j < mOutputWidth; ++j)
                    {
                        auto out_index = j + mOutputWidth * i;
                        dtype sum = 0.0_dt;
                        for (size_t n = 0; n < mKernelHeight; ++n)
                        {
                            for (size_t m = 0; m < mKernelWidth; ++m)
                            {
                                auto cur_h = i * mStrideH + n - mPaddingH;
                                auto cur_w = j * mStrideW + m - mPaddingW;
                                if (cur_h < mInputHeight && cur_w < mInputWidth)
                                {
                                    auto index = cur_w + mInputWidth * (cur_h);
                                    sum += inputs3D[b][k][index];
                                }
                            }
                        }
                        outputs3D[b][k][out_index] = sum * reciprocalKernelSize;
                    }
                }
            }
        }
    }
    else if (work.getExecutionTarget() == ExecutionTarget::CPUFP16)
    {
        const size_t batchSize = mNetworkParams.mWorkflow.getBatchSize();

        auto& output = work.getMemoryManager<MemoryManagerFP16>()[mOutputName];

        const auto& inputs = work.getMemoryManager<MemoryManagerFP16>()[mInputName];

        auto inputs3D = inputs.reshape(yato::dims(batchSize, mInputDepth, mInputHeight * mInputWidth));
        auto outputs3D = output.reshape(yato::dims(batchSize, mInputDepth, mOutputHeight * mOutputWidth));

        const dtype reciprocalKernelSize = 1.0_dt / static_cast<dtype>(mKernelHeight * mKernelWidth);

        for (size_t b = 0; b < batchSize; ++b)
        {
            for (size_t k = 0; k < mInputDepth; ++k)
            {
                for (size_t i = 0; i < mOutputHeight; ++i)
                {
                    for (size_t j = 0; j < mOutputWidth; ++j)
                    {
                        auto out_index = j + mOutputWidth * i;
                        half sum = 0.0_hf;
                        for (size_t n = 0; n < mKernelHeight; ++n)
                        {
                            for (size_t m = 0; m < mKernelWidth; ++m)
                            {
                                auto cur_h = i * mStrideH + n - mPaddingH;
                                auto cur_w = j * mStrideW + m - mPaddingW;
                                if (cur_h < mInputHeight && cur_w < mInputWidth)
                                {
                                    auto index = cur_w + mInputWidth * (cur_h);
                                    sum += inputs3D[b][k][index];
                                }
                            }
                        }
                        outputs3D[b][k][out_index] = sum * TOHTYPE(reciprocalKernelSize);
                    }
                }
            }
        }
    }
    else
    {
        THROW_NONAME("AveragePoolLayer", "unsupported execution target");
    }
}

void AveragePoolLayer::backwardComputeImpl()
{
    auto& work = mNetworkParams.mWorkflow;

    if (work.getExecutionTarget() == ExecutionTarget::CPU)
    {
        // if (mNetworkParams.isGradNeeded(mInputName))
        {
            const size_t batchSize = mNetworkParams.mWorkflow.getBatchSize();

            Tensor& prevLayerDelta = mNetworkParams.mMemoryManager[mInputName.grad()];

            const Tensor& deltas = mNetworkParams.mMemoryManager[mOutputName.grad()];

            auto deltas3D = deltas.reshape(yato::dims(batchSize, mInputDepth, mOutputHeight * mOutputWidth));
            auto prevDeltas3D = prevLayerDelta.reshape(yato::dims(batchSize, mInputDepth, mInputHeight * mInputWidth));

            const dtype reciprocalKernelSize = 1.0_dt / static_cast<dtype>(mKernelHeight * mKernelWidth);

            for (size_t batch = 0; batch < batchSize; ++batch)
            {
                for (size_t c = 0; c < mInputDepth; ++c)
                {
                    for (size_t i = 0; i < mOutputHeight; ++i)
                    {
                        for (size_t j = 0; j < mOutputWidth; ++j)
                        {
                            auto out_index = j + mOutputWidth * i;
                            for (size_t n = 0; n < mKernelHeight; ++n)
                            {
                                for (size_t m = 0; m < mKernelWidth; ++m)
                                {
                                    auto cur_h = i * mStrideH + n - mPaddingH;
                                    auto cur_w = j * mStrideW + m - mPaddingW;
                                    if (cur_h < mInputHeight && cur_w < mInputWidth)
                                    {
                                        auto index = cur_w + mInputWidth * (cur_h);
                                        prevDeltas3D[batch][c][index] += deltas3D[batch][c][out_index] * reciprocalKernelSize;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    else if (work.getExecutionTarget() == ExecutionTarget::CPUFP16)
    {
        // if (mNetworkParams.isGradNeeded(mInputName))
        {
            const size_t batchSize = mNetworkParams.mWorkflow.getBatchSize();

            auto& prevLayerDelta = work.getMemoryManager<MemoryManagerFP16>()[mInputName.grad()];

            const auto& deltas = work.getMemoryManager<MemoryManagerFP16>()[mOutputName.grad()];

            auto deltas3D = deltas.reshape(yato::dims(batchSize, mInputDepth, mOutputHeight * mOutputWidth));
            auto prevDeltas3D = prevLayerDelta.reshape(yato::dims(batchSize, mInputDepth, mInputHeight * mInputWidth));

            const dtype reciprocalKernelSize = 1.0_dt / static_cast<dtype>(mKernelHeight * mKernelWidth);

            for (size_t batch = 0; batch < batchSize; ++batch)
            {
                for (size_t c = 0; c < mInputDepth; ++c)
                {
                    for (size_t i = 0; i < mOutputHeight; ++i)
                    {
                        for (size_t j = 0; j < mOutputWidth; ++j)
                        {
                            auto out_index = j + mOutputWidth * i;
                            for (size_t n = 0; n < mKernelHeight; ++n)
                            {
                                for (size_t m = 0; m < mKernelWidth; ++m)
                                {
                                    auto cur_h = i * mStrideH + n - mPaddingH;
                                    auto cur_w = j * mStrideW + m - mPaddingW;
                                    if (cur_h < mInputHeight && cur_w < mInputWidth)
                                    {
                                        auto index = cur_w + mInputWidth * (cur_h);
                                        prevDeltas3D[batch][c][index] += deltas3D[batch][c][out_index] * TOHTYPE(reciprocalKernelSize);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    else
    {
        THROW_NONAME("AveragePoolLayer", "unsupported execution target");
    }
}
} // namespace raul
