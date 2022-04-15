// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "LinearLayer.h"

#include <algorithm>

#include <training/api/API.h>

#include <training/tools/Profiler.h>

#include "impl/LinearLayerCPUFP16.h"
#include "impl/LinearLayerCPUFP32.h"
#include "impl/LinearLayerCPUFP32Master.h"
#include "impl/LinearLayerImpl.h"

#include <training/opencl/GemmGPU.h>

namespace raul
{

LinearLayer::LinearLayer(const Name& name, const LinearParams& params, NetworkParameters& networkParameters)
    : TrainableLayer(name, "Linear", params, networkParameters, { true, true })
    , mOutputsCount(params.outputsCount)
    , mUseBias(params.bias)
{
    MEASURE_BLOCK(mTypeName + "[" + mName + "::ctor]")
    auto prefix = mTypeName + "[" + mName + "::ctor]: ";

    mInputName = mInputs[0];
    mOutputName = mOutputs[0];

    mDepth = mNetworkParams.mWorkflow.getDepth(mInputName);
    mHeight = mNetworkParams.mWorkflow.getHeight(mInputName);
    mInputsCount = mNetworkParams.mWorkflow.getWidth(mInputName);

    if (params.getLayerExecutionTarget() == LayerExecutionTarget::Default)
    {
        DECLARE_IMPL(LinearLayer, LinearLayerImpl<MemoryManager>, LinearLayerImpl<MemoryManagerGPU>, LinearLayerImpl<MemoryManagerFP16>)
    }
    else if (params.getLayerExecutionTarget() == LayerExecutionTarget::CPU && mNetworkParams.mWorkflow.getOverrideLayerExecutionTarget() == LayerExecutionTarget::Default)
    {
        // DECLARE_IMPL(LinearLayer, LinearLayerImpl<MemoryManager>, NotImplemented, LinearLayerCPUFP32)
        DECLARE_IMPL(LinearLayer, LinearLayerImpl<MemoryManager>, NotImplemented, LinearLayerCPUFP32Master)
    }
    else if (params.getLayerExecutionTarget() == LayerExecutionTarget::CPUFP16 && mNetworkParams.mWorkflow.getOverrideLayerExecutionTarget() == LayerExecutionTarget::Default)
    {
        DECLARE_IMPL(LinearLayer, LinearLayerCPUFP16, NotImplemented, LinearLayerImpl<MemoryManagerFP16>)
    }
    else if (params.getLayerExecutionTarget() == LayerExecutionTarget::CPU && mNetworkParams.mWorkflow.getOverrideLayerExecutionTarget() == LayerExecutionTarget::CPU)
    {
        DECLARE_IMPL(LinearLayer, LinearLayerImpl<MemoryManager>, NotImplemented, LinearLayerImpl<MemoryManager>)
    }
    else if (params.getLayerExecutionTarget() == LayerExecutionTarget::CPUFP16 && mNetworkParams.mWorkflow.getOverrideLayerExecutionTarget() == LayerExecutionTarget::CPUFP16)
    {
        DECLARE_IMPL(LinearLayer, LinearLayerImpl<MemoryManagerFP16>, NotImplemented, LinearLayerImpl<MemoryManagerFP16>)
    }
    else
    {
        THROW(mTypeName, mName, "unsupported layer execution target");
    }

    if (mInputs.size() != 1)
    {
        THROW(mTypeName, mName, "wrong number of input names");
    }
    if (mOutputs.size() != 1)
    {
        THROW(mTypeName, mName, "wrong number of output names");
    }

    mNetworkParams.mWorkflow.copyDeclaration(mName, mInputName, Workflow::Usage::ForwardAndBackward, Workflow::Mode::Read);

    mNetworkParams.mWorkflow.copyDeclaration(mName, mInputName, mInputName.grad(), DEC_BACK_WRIT_ZERO);

    mNetworkParams.mWorkflow.tensorNeeded(mName, mOutputName, WShape{ BS(), mDepth, mHeight, mOutputsCount }, DEC_FORW_WRIT_COMP);

    mNetworkParams.mWorkflow.copyDeclaration(mName, mOutputName, mOutputName.grad(), DEC_BACK_READ);

    bool isGPU = (mNetworkParams.mWorkflow.getExecutionTarget() == ExecutionTarget::GPU && params.getLayerExecutionTarget() == LayerExecutionTarget::Default) ||
                 params.getLayerExecutionTarget() == LayerExecutionTarget::GPU;
    if (!isGPU || (mSharedLayer.empty() && mSharedWeights.empty()))
    {
        mNetworkParams.mWorkflow.tensorNeeded(mName, mWeightsName, WShape{ 1u, 1u, mOutputsCount, mInputsCount }, DEC_TRAINABLE);
    }

    if (isGPU)
    {
        mTransposedWeightsName = mWeightsName / "GpuTransposed";
        mTransposedPaddedWeightsName = mWeightsName / "GpuTransposedPadded";
        mPaddedWeightsName = mWeightsName / "GpuPadded";

        auto caller = mTypeName + "[" + mName + "::ctor]";

        {
            size_t m_align = 0;
            gpu::gemm_bolt_aligned_sizes(1, mOutputsCount, m_align, mForwardAlignedWeightsSize);
            gpu::gemm_bolt_aligned_sizes(1, mInputsCount, m_align, mBackwardAlignedWeightsSize);

            mForwardWeightsUsage = WeightsUsage::Transposed;
            mBackwardWeightsUsage = WeightsUsage::Unchanged;

            mForwardWeightsNameGpu = mTransposedWeightsName;
            mBackwardWeightsNameGpu = mWeightsName;

            if (mForwardAlignedWeightsSize != mOutputsCount)
            {
                mForwardWeightsUsage = WeightsUsage::Padded;
                mForwardWeightsNameGpu = mTransposedPaddedWeightsName;
            }

            if (mBackwardAlignedWeightsSize != mInputsCount)
            {
                mBackwardWeightsUsage = WeightsUsage::Padded;
                mBackwardWeightsNameGpu = mPaddedWeightsName;
            }
        }

        if (mSharedLayer.empty() && mSharedWeights.empty())
        {
            mNetworkParams.mWorkflow.tensorNeeded(mName, mTransposedWeightsName, WShape{ 1u, 1u, mInputsCount, mOutputsCount }, DEC_FORW_WRIT);
            if (mForwardWeightsUsage == WeightsUsage::Padded)
            {
                mNetworkParams.mWorkflow.tensorNeeded(mName, mTransposedPaddedWeightsName, WShape{ 1u, 1u, mInputsCount, mForwardAlignedWeightsSize }, DEC_FORW_WRIT);
            }
            if (mBackwardWeightsUsage == WeightsUsage::Padded)
            {
                mNetworkParams.mWorkflow.tensorNeeded(mName, mPaddedWeightsName, WShape{ 1u, 1u, mOutputsCount, mBackwardAlignedWeightsSize }, DEC_FORW_WRIT);
                mNetworkParams.mWorkflow.copyDeclaration(mName, mPaddedWeightsName, DEC_BACK_READ);
            }
        }
        else
        {
            mNetworkParams.mWorkflow.copyDeclaration(mName, mForwardWeightsNameGpu, Workflow::Usage::Forward, Workflow::Mode::Read);
            mNetworkParams.mWorkflow.copyDeclaration(mName, mBackwardWeightsNameGpu, Workflow::Usage::Backward, Workflow::Mode::Read);
        }
    }

    if (mUseBias)
    {
        mNetworkParams.mWorkflow.tensorNeeded(mName, mBiasesName, WShape{ 1u, 1u, 1u, mOutputsCount }, DEC_TRAINABLE);
    }

    if (!mFrozen)
    {
        mNetworkParams.mWorkflow.copyDeclaration(mName, mWeightsName, mWeightsName.grad(), DEC_TRAINABLE_GRAD);

        if (mUseBias)
        {
            mNetworkParams.mWorkflow.copyDeclaration(mName, mBiasesName, mBiasesName.grad(), DEC_TRAINABLE_GRAD);
        }
    }
}
} // namespace raul
