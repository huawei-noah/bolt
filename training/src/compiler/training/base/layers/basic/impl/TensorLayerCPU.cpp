// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "TensorLayerCPU.h"
#include "../TensorLayer.h"

#include <training/base/impl/ImplFactory.h>
namespace
{
bool reg1 = raul::TheImplFactory::Instance().regCPUFP32<raul::TensorLayer, raul::TensorLayerCPU<raul::MemoryManager>>();
bool reg2 = raul::TheImplFactory::Instance().regCPUFP16<raul::TensorLayer, raul::TensorLayerCPU<raul::MemoryManagerFP16>>();
} // anonymous namespace

namespace raul
{

template<typename MM>
TensorLayerCPU<MM>::TensorLayerCPU(TensorLayer& layer)
    : mLayer(layer)
{
}

template<typename MM>
void TensorLayerCPU<MM>::forwardComputeImpl(NetworkMode)
{
    if (mLayer.mInit)
    {
        auto& work = mLayer.mNetworkParams.mWorkflow;
        for (const auto& out : mLayer.mOutputs)
        {
            work.getMemoryManager<MM>()[out] = static_cast<typename MM::type>(mLayer.mInitValue);
        }
    }
}

template class TensorLayerCPU<MemoryManager>;
template class TensorLayerCPU<MemoryManagerFP16>;

} // namespace raul