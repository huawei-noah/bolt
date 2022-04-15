// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "ConvertPrecisionLayer.h"

#define STORE_RESET_EXEC_TARGET                                                                                                                                                                        \
    auto execTarget = mNetworkParams.mWorkflow.getOverrideLayerExecutionTarget();                                                                                                                      \
    mNetworkParams.mWorkflow.resetLayerExecutionTargetOverride();

#define RESTORE_EXEC_TARGET mNetworkParams.mWorkflow.overrideLayerExecutionTarget(execTarget);

namespace raul
{

ConvertPrecisionLayer::ConvertPrecisionLayer(const Name& name, const ConvertPrecisionParams& params, NetworkParameters& networkParameters)
    : BasicLayer(name, "Cast", params, networkParameters)
    , mInvertDirection(params.mInvertDirection)
{
    mTarget = mNetworkParams.mWorkflow.getOverrideLayerExecutionTarget();

    if (mTarget == LayerExecutionTarget::Default)
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

        if (!mInvertDirection)
        {
            mNetworkParams.mWorkflow.copyDeclaration(mName, tensorIn, tensorOut, DEC_FORW_WRIT);

            if (mNetworkParams.mWorkflow.isTensorDeclared(tensorIn.grad()))
            {
                mNetworkParams.mWorkflow.copyDeclaration(mName, tensorOut, tensorOut.grad(), DEC_BACK_READ);
            }

            // switch to ExecutionTarget
            STORE_RESET_EXEC_TARGET
            if (params.mOptimizeMemory)
            {
                mNetworkParams.mWorkflow.copyDeclaration(mName, tensorIn, DEC_FORW_READ);
            }
            else
            {
                mNetworkParams.mWorkflow.copyDeclaration(mName, tensorIn, DEC_FORW_READ_NOMEMOPT);
            }
            if (mNetworkParams.mWorkflow.isTensorDeclared(tensorIn.grad()))
            {
                mNetworkParams.mWorkflow.copyDeclaration(mName, tensorIn.grad(), DEC_BACK_WRIT_ZERO);
            }
            // restore override
            RESTORE_EXEC_TARGET
        }
        else
        {
            // switch to ExecutionTarget
            STORE_RESET_EXEC_TARGET

            if (params.mOptimizeMemory)
            {
                mNetworkParams.mWorkflow.copyDeclaration(mName, tensorIn, tensorOut, DEC_FORW_WRIT);
            }
            else
            {
                mNetworkParams.mWorkflow.copyDeclaration(mName, tensorIn, tensorOut, DEC_FORW_WRIT_NOMEMOPT);
            }

            if (mNetworkParams.mWorkflow.isTensorDeclared(tensorIn.grad()))
            {
                mNetworkParams.mWorkflow.copyDeclaration(mName, tensorOut, tensorOut.grad(), DEC_BACK_READ);
            }

            // restore override
            RESTORE_EXEC_TARGET

            if (params.mOptimizeMemory)
            {
                mNetworkParams.mWorkflow.copyDeclaration(mName, tensorIn, DEC_FORW_READ);
            }
            else
            {
                mNetworkParams.mWorkflow.copyDeclaration(mName, tensorIn, DEC_FORW_READ_NOMEMOPT);
            }
            if (mNetworkParams.mWorkflow.isTensorDeclared(tensorIn.grad()))
            {
                mNetworkParams.mWorkflow.copyDeclaration(mName, tensorIn.grad(), DEC_BACK_WRIT_ZERO);
            }
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

    const ExecutionTarget compareETCPU = mInvertDirection ? ExecutionTarget::CPUFP16 : ExecutionTarget::CPU;
    const LayerExecutionTarget compareLETCPUFP16 = mInvertDirection ? LayerExecutionTarget::CPU : LayerExecutionTarget::CPUFP16;

    const ExecutionTarget compareETCPUFP16 = mInvertDirection ? ExecutionTarget::CPU : ExecutionTarget::CPUFP16;
    const LayerExecutionTarget compareLETCPU = mInvertDirection ? LayerExecutionTarget::CPUFP16 : LayerExecutionTarget::CPU;

    if (mNetworkParams.mWorkflow.getExecutionTarget() == compareETCPU)
    {
        const auto& mmCpuFP32 = mNetworkParams.mWorkflow.getMemoryManager<MemoryManager>();

        if (mTarget == compareLETCPUFP16)
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
    else if (mNetworkParams.mWorkflow.getExecutionTarget() == compareETCPUFP16)
    {
        const auto& mmCpuFP16 = mNetworkParams.mWorkflow.getMemoryManager<MemoryManagerFP16>();

        if (mTarget == compareLETCPU)
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

    const ExecutionTarget compareETCPU = mInvertDirection ? ExecutionTarget::CPUFP16 : ExecutionTarget::CPU;
    const LayerExecutionTarget compareLETCPUFP16 = mInvertDirection ? LayerExecutionTarget::CPU : LayerExecutionTarget::CPUFP16;

    const ExecutionTarget compareETCPUFP16 = mInvertDirection ? ExecutionTarget::CPU : ExecutionTarget::CPUFP16;
    const LayerExecutionTarget compareLETCPU = mInvertDirection ? LayerExecutionTarget::CPUFP16 : LayerExecutionTarget::CPU;

    if (mNetworkParams.mWorkflow.getExecutionTarget() == compareETCPU)
    {
        auto& mmCpuFP32 = mNetworkParams.mWorkflow.getMemoryManager<MemoryManager>();

        if (mTarget == compareLETCPUFP16)
        {
            const auto& mmCpuFP16 = mNetworkParams.mWorkflow.getMemoryManager<MemoryManagerFP16>();

            if (mmCpuFP32.tensorExists(tensorIn.grad()))
            {
                const auto& input = mmCpuFP16[tensorOut.grad()];
                auto& output = mmCpuFP32[tensorIn.grad()];

                std::transform(input.begin(), input.end(), output.begin(), [](half val) { return toFloat32(val); });
            }
        }
        else
        {
            THROW(mTypeName, mName, "Layer target not supported");
        }
    }
    else if (mNetworkParams.mWorkflow.getExecutionTarget() == compareETCPUFP16)
    {
        auto& mmCpuFP16 = mNetworkParams.mWorkflow.getMemoryManager<MemoryManagerFP16>();

        if (mTarget == compareLETCPU)
        {
            const auto& mmCpuFP32 = mNetworkParams.mWorkflow.getMemoryManager<MemoryManager>();

            if (mmCpuFP16.tensorExists(tensorIn.grad()))
            {
                const auto& input = mmCpuFP32[tensorOut.grad()];
                auto& output = mmCpuFP16[tensorIn.grad()];

                std::transform(input.begin(), input.end(), output.begin(), [](dtype val) { return toFloat16(val); });
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
