// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "TacotronDataInitializationLayerGPU.h"
#include "../TacotronDataInitializationLayer.h"

namespace raul
{
namespace tacotron
{

TacotronDataInitializationLayerGPU::TacotronDataInitializationLayerGPU(TacotronDataInitializationLayer& layer)
    : mLayer(layer)
{
}

void TacotronDataInitializationLayerGPU::forwardComputeImpl(NetworkMode)
{
    auto& work = mLayer.mNetworkParams.mWorkflow;
    auto caller = mLayer.mTypeName + "[" + mLayer.mName + "::forwardComputeImpl]";
    for (const auto& output : mLayer.getOutputs())
    {
        if (output == mLayer.mInitialAlignmentsName)
        {
            continue;
        }
        work.getKernelManager().fillBuffer(work.getMemoryManager<MemoryManagerGPU>()(output).getBuffer(), 0_dt, caller);
    }
    if (!mLayer.mInitialAlignmentsName.empty())
    {
        auto& initialAlignments = work.getMemoryManager<MemoryManagerGPU>()(mLayer.mInitialAlignmentsName);
        // Dirac distribution, which is needed in order for the monotonic attention
        // distributions to have the correct behavior.
        const size_t N = initialAlignments.getBatchSize();
        const size_t height = initialAlignments.size() / N;
        Common::initAlignment(&work.getKernelManager(), caller, 1.0_dt, N, height, initialAlignments);
    }

    // Dirac distribution for initial_alignments
}

void TacotronDataInitializationLayerGPU::backwardComputeImpl() {}

}
} // namespace raul