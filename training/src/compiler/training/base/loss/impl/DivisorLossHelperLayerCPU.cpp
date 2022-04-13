// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "DivisorLossHelperLayerCPU.h"
#include "../DivisorLossHelperLayer.h"

#include <training/base/impl/ImplFactory.h>
namespace
{
bool reg1 = raul::TheImplFactory::Instance().regCPUFP32<raul::DivisorLossHelperLayer, raul::DivisorLossHelperLayerCPU<raul::MemoryManager>>();
bool reg2 = raul::TheImplFactory::Instance().regCPUFP16<raul::DivisorLossHelperLayer, raul::DivisorLossHelperLayerCPU<raul::MemoryManagerFP16>>();
} // anonymous namespace

namespace raul
{

template<typename MM>
void DivisorLossHelperLayerCPU<MM>::forwardComputeImpl(NetworkMode)
{
    auto& work = mLayer.mNetworkParams.mWorkflow;

    auto divisor = TOMMTYPE(mLayer.mNetworkParams.mLossReductionCoefficient);
    if (mLayer.mIsCustomMean)
    {
        divisor *= TOMMTYPE(mLayer.mNetworkParams.mWorkflow.getDepth(mLayer.mInputName) * mLayer.mNetworkParams.mWorkflow.getHeight(mLayer.mInputName) *
                            mLayer.mNetworkParams.mWorkflow.getWidth(mLayer.mInputName));
    }

    auto& t = work.getMemoryManager<MM>()[mLayer.mOutputs[0]];
    t[0] = divisor;
}

template class DivisorLossHelperLayerCPU<MemoryManager>;
template class DivisorLossHelperLayerCPU<MemoryManagerFP16>;

} // namespace raul
