// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "TargetsReductionLayerGPU.h"
#include "../TargetsReductionLayer.h"

namespace raul
{
namespace tacotron
{

void TargetsReductionLayerGPU::forwardComputeImpl(NetworkMode)
{
    // reduce mel_targets
    auto& work = mLayer.mNetworkParams.mWorkflow;
    const auto& targets = work.getMemoryManager<MemoryManagerGPU>()(mLayer.mMelTargetsName);
    auto& reducedTargets = work.getMemoryManager<MemoryManagerGPU>()(mLayer.mReducedMelTargetsName);
    work.getKernelManager().fillBuffer(reducedTargets.getBuffer(), 0.0_dt, mLayer.mTypeName + "[" + mLayer.mName + "::forwardComputeImpl]");
    Common::reduceTargets(&work.getKernelManager(),
                          mLayer.mTypeName + "[" + mLayer.mName + "::forwardComputeImpl]",
                          work.getBatchSize(),
                          targets.getDepth(),
                          reducedTargets.getDepth(),
                          targets.getHeight(),
                          reducedTargets.getHeight(),
                          targets.getWidth(),
                          mLayer.mReductionFactor,
                          targets,
                          reducedTargets);
}

void TargetsReductionLayerGPU::backwardComputeImpl() {}

}
} // namespace raul
