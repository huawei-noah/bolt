// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "ConvertPrecisionLayer.h"

namespace raul
{

ConvertPrecisionLayer::ConvertPrecisionLayer(const Name& name, const ConvertPrecisionParams& params, NetworkParameters& networkParameters)
    : BasicLayer(name, "Cast", params, networkParameters)
    , mFromTarget(params.mFromTarget)
    , mToTarget(params.mToTarget)
{
    if (mFromTarget == LayerExecutionTarget::Default && mToTarget == LayerExecutionTarget::Default)
    {
        THROW(mTypeName, mName, "Layer execution target should be overriden");
    }

    try
    {
        if (mInputs.size() != 1)
        {
            THROW(mTypeName, mName, "wrong number of input names");
        }
        if (mOutputs.size() != 1)
        {
            THROW(mTypeName, mName, "wrong number of output names");
        }

        const auto& tensorIn = mInputs[0];
        const auto& tensorOut = mOutputs[0];

        if (params.mOptimizeMemory)
        {
            mNetworkParams.mWorkflow.copyDeclaration(mName, tensorIn, tensorOut, DEC_FORW_WRIT, mToTarget);
        }
        else
        {
            mNetworkParams.mWorkflow.copyDeclaration(mName, tensorIn, tensorOut, DEC_FORW_WRIT_NOMEMOPT, mToTarget);
        }

        if (mNetworkParams.mWorkflow.isTensorDeclared(tensorIn.grad()))
        {
            mNetworkParams.mWorkflow.copyDeclaration(mName, tensorOut, tensorOut.grad(), DEC_BACK_READ, mToTarget);
        }

        if (params.mOptimizeMemory)
        {
            mNetworkParams.mWorkflow.copyDeclaration(mName, tensorIn, DEC_FORW_READ, mFromTarget);
        }
        else
        {
            mNetworkParams.mWorkflow.copyDeclaration(mName, tensorIn, DEC_FORW_READ_NOMEMOPT, mFromTarget);
        }

        if (mNetworkParams.mWorkflow.isTensorDeclared(tensorIn.grad()))
        {
            mNetworkParams.mWorkflow.copyDeclaration(mName, tensorIn.grad(), DEC_BACK_WRIT_ZERO, mFromTarget);
        }
    }
    catch (...)
    {
        THROW(mTypeName, mName, "Cannot create CastLayer");
    }
}

void ConvertPrecisionLayer::forwardComputeImpl(NetworkMode)
{
    const auto& tensorIn = mInputs[0];
    const auto& tensorOut = mOutputs[0];

    if (mFromTarget == LayerExecutionTarget::CPU || (mFromTarget == LayerExecutionTarget::Default && mNetworkParams.mWorkflow.getExecutionTarget() == ExecutionTarget::CPU))
    {
        const auto& mmCpuFP32 = mNetworkParams.mWorkflow.getMemoryManager<MemoryManager>();

        if (mToTarget == LayerExecutionTarget::CPUFP16 || (mToTarget == LayerExecutionTarget::Default && mNetworkParams.mWorkflow.getExecutionTarget() == ExecutionTarget::CPUFP16))
        {
            auto& mmCpuFP16 = mNetworkParams.mWorkflow.getMemoryManager<MemoryManagerFP16>();

            const auto& input = mmCpuFP32[tensorIn];
            auto& output = mmCpuFP16[tensorOut];

            std::transform(input.begin(), input.end(), output.begin(), [](dtype val) { return toFloat16(val); });
        }
        else
        {
            THROW(mTypeName, mName, "Layer target not supported");
        }
    }
    else if (mFromTarget == LayerExecutionTarget::CPUFP16 || (mFromTarget == LayerExecutionTarget::Default && mNetworkParams.mWorkflow.getExecutionTarget() == ExecutionTarget::CPUFP16))
    {
        const auto& mmCpuFP16 = mNetworkParams.mWorkflow.getMemoryManager<MemoryManagerFP16>();

        if (mToTarget == LayerExecutionTarget::CPU || (mToTarget == LayerExecutionTarget::Default && mNetworkParams.mWorkflow.getExecutionTarget() == ExecutionTarget::CPU))
        {
            auto& mmCpuFP32 = mNetworkParams.mWorkflow.getMemoryManager<MemoryManager>();

            const auto& input = mmCpuFP16[tensorIn];
            auto& output = mmCpuFP32[tensorOut];

            std::transform(input.begin(), input.end(), output.begin(), [](half val) { return toFloat32(val); });
        }
        else
        {
            THROW(mTypeName, mName, "Layer target not supported");
        }
    }
    else
    {
        THROW(mTypeName, mName, "Target not supported");
    }
}

void ConvertPrecisionLayer::backwardComputeImpl()
{
    const auto& tensorIn = mInputs[0];
    const auto& tensorOut = mOutputs[0];

    if (mFromTarget == LayerExecutionTarget::CPU || (mFromTarget == LayerExecutionTarget::Default && mNetworkParams.mWorkflow.getExecutionTarget() == ExecutionTarget::CPU))
    {
        auto& mmCpuFP32 = mNetworkParams.mWorkflow.getMemoryManager<MemoryManager>();

        if (mToTarget == LayerExecutionTarget::CPUFP16 || (mToTarget == LayerExecutionTarget::Default && mNetworkParams.mWorkflow.getExecutionTarget() == ExecutionTarget::CPUFP16))
        {
            const auto& mmCpuFP16 = mNetworkParams.mWorkflow.getMemoryManager<MemoryManagerFP16>();

            if (mmCpuFP32.tensorExists(tensorIn.grad()))
            {
                const auto& input = mmCpuFP16[tensorOut.grad()];
                auto& output = mmCpuFP32[tensorIn.grad()];

                std::transform(input.begin(), input.end(), output.begin(), output.begin(), [](half val, dtype in) { return in + toFloat32(val); });
            }
        }
        else
        {
            THROW(mTypeName, mName, "Layer target not supported");
        }
    }
    else if (mFromTarget == LayerExecutionTarget::CPUFP16 || (mFromTarget == LayerExecutionTarget::Default && mNetworkParams.mWorkflow.getExecutionTarget() == ExecutionTarget::CPUFP16))
    {
        auto& mmCpuFP16 = mNetworkParams.mWorkflow.getMemoryManager<MemoryManagerFP16>();

        if (mToTarget == LayerExecutionTarget::CPU || (mToTarget == LayerExecutionTarget::Default && mNetworkParams.mWorkflow.getExecutionTarget() == ExecutionTarget::CPU))
        {
            const auto& mmCpuFP32 = mNetworkParams.mWorkflow.getMemoryManager<MemoryManager>();

            if (mmCpuFP16.tensorExists(tensorIn.grad()))
            {
                const auto& input = mmCpuFP32[tensorOut.grad()];
                auto& output = mmCpuFP16[tensorIn.grad()];

                std::transform(input.begin(), input.end(), output.begin(), output.begin(), [](dtype val, half in) -> half { return in + toFloat16(val); });
            }
        }
        else
        {
            THROW(mTypeName, mName, "Layer target not supported");
        }
    }
    else
    {
        THROW(mTypeName, mName, "Target not supported");
    }
}

} // namespace raul