// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "TargetsReductionLayer.h"

#include "impl/TargetsReductionLayerCPU.h"
#include "impl/TargetsReductionLayerGPU.h"

#include <training/api/API.h>

namespace raul
{
namespace tacotron
{

TargetsReductionLayer::TargetsReductionLayer(const Name& name, const BasicParams& params, const TacotronParams& tparams, raul::NetworkParameters& networkParameters)
    : BasicLayer(name, "TargetsReduction", params, networkParameters)
{
    DECLARE_IMPL(TargetsReductionLayer, TargetsReductionLayerCPU<MemoryManager>, TargetsReductionLayerGPU, TargetsReductionLayerCPU<MemoryManagerFP16>)

    auto prefix = mTypeName + "[" + mName + "::ctor]: ";
    if (mInputs.size() != 1)
    {
        THROW(mTypeName, mName, "wrong number of inputs");
    }
    if (mOutputs.size() != 1)
    {
        THROW(mTypeName, mName, "wrong number of outputs");
    }
    if (mNetworkParams.mWorkflow.getDepth(mInputs[0]) != 1u && mNetworkParams.mWorkflow.getHeight(mInputs[0]) != 1u)
    {
        THROW(mTypeName, mName, "unsupported mode; depth or height should have size = 1");
    }
    mMelTargetsName = mInputs[0];
    mReducedMelTargetsName = mOutputs[0];
    mReductionFactor = tparams.outputsPerStep;

    size_t max_decoder_iterations = mNetworkParams.mWorkflow.getHeight(mMelTargetsName) / mReductionFactor;
    if (max_decoder_iterations == 0)
    {
        max_decoder_iterations = 1;
    }
    WShape shape = { BS(), mNetworkParams.mWorkflow.getDepth(mMelTargetsName), max_decoder_iterations, mNetworkParams.mWorkflow.getWidth(mMelTargetsName) };
    mNetworkParams.mWorkflow.tensorNeeded(mName, mReducedMelTargetsName, shape, DEC_FORW_WRIT);
}

}
} // namespace raul