// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "SequenceMaskLayerCPU.h"
#include "../SequenceMaskLayer.h"

namespace raul
{
namespace tacotron
{

template<typename MM>
void SequenceMaskLayerCPU<MM>::forwardComputeImpl(NetworkMode)
{
    Workflow& work = mLayer.mNetworkParams.mWorkflow;

    auto& memory_manager = work.getMemoryManager<MM>();
    sequence_mask(memory_manager[mLayer.mLengthsName], mLayer.mOutputsPerStep, memory_manager[mLayer.mOutputMaskName]);
}

template<typename MM>
void SequenceMaskLayerCPU<MM>::backwardComputeImpl()
{
}

template class SequenceMaskLayerCPU<MemoryManager>;
template class SequenceMaskLayerCPU<MemoryManagerFP16>;

}
} // namespace raul