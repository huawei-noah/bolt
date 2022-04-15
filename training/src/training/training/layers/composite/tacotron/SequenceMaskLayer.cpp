// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "SequenceMaskLayer.h"

#include "impl/SequenceMaskLayerCPU.h"
#include "impl/SequenceMaskLayerGPU.h"

namespace raul
{
namespace tacotron
{

SequenceMaskLayer::SequenceMaskLayer(const Name& name, const BasicParams& params, size_t outputsPerStep, NetworkParameters& networkParameters)
    : BasicLayer(name, "SequenceMask", params, networkParameters)
{
    const auto& input = mInputs[0];
    mLengthsName = mInputs[1];
    mOutputMaskName = mOutputs[0];
    mOutputsPerStep = outputsPerStep;

    if (mNetworkParams.mWorkflow.getOverrideLayerExecutionTarget() == LayerExecutionTarget::Default)
    {
        DECLARE_IMPL(SequenceMaskLayer, SequenceMaskLayerCPU<MemoryManager>, SequenceMaskLayerGPU, SequenceMaskLayerCPU<MemoryManagerFP16>)
    }
    else if (mNetworkParams.mWorkflow.getOverrideLayerExecutionTarget() == LayerExecutionTarget::CPU)
    {
        DECLARE_IMPL(SequenceMaskLayer, SequenceMaskLayerCPU<MemoryManager>, NotImplemented, SequenceMaskLayerCPU<MemoryManager>)
    }
    else if (mNetworkParams.mWorkflow.getOverrideLayerExecutionTarget() == LayerExecutionTarget::CPUFP16)
    {
        DECLARE_IMPL(SequenceMaskLayer, SequenceMaskLayerCPU<MemoryManagerFP16>, NotImplemented, SequenceMaskLayerCPU<MemoryManagerFP16>)
    }
    else
    {
        THROW(mTypeName, mName, "unsupported layer execution target");
    }

    mNetworkParams.mWorkflow.copyDeclaration(name, mLengthsName, raul::Workflow::Usage::Forward, raul::Workflow::Mode::Read);
    mNetworkParams.mWorkflow.copyDeclaration(name, input, mOutputMaskName, DEC_FORW_WRIT);
}

}
} // namespace raul::tacotron
