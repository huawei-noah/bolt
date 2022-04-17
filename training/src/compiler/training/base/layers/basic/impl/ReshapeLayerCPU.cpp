// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "ReshapeLayerCPU.h"
#include "../ReshapeLayer.h"

#include <training/base/impl/ImplFactory.h>
namespace
{
bool reg1 = raul::TheImplFactory::Instance().regCPUFP32<raul::ReshapeLayer, raul::ReshapeLayerCPU<raul::MemoryManager>>();
bool reg2 = raul::TheImplFactory::Instance().regCPUFP16<raul::ReshapeLayer, raul::ReshapeLayerCPU<raul::MemoryManagerFP16>>();
} // anonymous namespace

namespace raul
{

template<typename MM>
void ReshapeLayerCPU<MM>::forwardComputeImpl(NetworkMode)
{
    auto& mm = mLayer.mNetworkParams.mWorkflow.getMemoryManager<MM>();
    auto& output = mm[mLayer.mOutputName];
    output = TORANGE_MM(mm[mLayer.mInputName]);
}

template<typename MM>
void ReshapeLayerCPU<MM>::backwardComputeImpl()
{
    auto& mm = mLayer.mNetworkParams.mWorkflow.getMemoryManager<MM>();
    auto& prevLayerDelta = mm[mLayer.mInputName.grad()];
    const auto& delta = mm[mLayer.mOutputName.grad()];
    std::transform(delta.begin(), delta.end(), prevLayerDelta.begin(), prevLayerDelta.begin(), [](typename MM::type x, typename MM::type grad) { return TOMMTYPE(x + grad); });
}

template class ReshapeLayerCPU<MemoryManager>;
template class ReshapeLayerCPU<MemoryManagerFP16>;

} // namespace raul
