// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef NON_ZERO_MASK_LAYER_H
#define NON_ZERO_MASK_LAYER_H

#include "impl/NonZeroMaskLayerCPU.h"
#include <training/base/common/Common.h>
#include <training/base/layers/BasicLayer.h>

namespace raul
{

/**
 * @brief Element-wise non-zero mask
 *
 */
class NonZeroMaskLayer : public BasicLayer
{
  public:
    NonZeroMaskLayer(const Name& name, const BasicParams& params, raul::NetworkParameters& networkParameters)
        : BasicLayer(name, "NonZeroMask", params, networkParameters)
    {
        DECLARE_IMPL(NonZeroMaskLayer, NonZeroMaskLayerCPU<MemoryManager>, NonZeroMaskLayerCPU<MemoryManagerFP16>)
        using namespace std;
        auto prefix = mTypeName + "[" + mName + "::ctor]: ";
        if (mInputs.size() != 1)
        {
            THROW(mTypeName, mName, "wrong number of inputs")
        }
        if (mOutputs.size() != 1)
        {
            THROW(mTypeName, mName, "wrong number of outputs")
        }

        mNetworkParams.mWorkflow.copyDeclaration(mName, mInputs[0], Workflow::Usage::ForwardAndBackward, Workflow::Mode::Read);
        mNetworkParams.mWorkflow.copyDeclaration(name, mInputs[0], mOutputs[0], DEC_FORW_WRIT);
    }

    NonZeroMaskLayer(NonZeroMaskLayer&&) = default;
    NonZeroMaskLayer(const NonZeroMaskLayer&) = delete;
    NonZeroMaskLayer& operator=(const NonZeroMaskLayer&) = delete;

    void forwardComputeImpl(NetworkMode mode) override { mImpl->forwardComputeImpl(mode); }

    void backwardComputeImpl() override { mImpl->backwardComputeImpl(); }

    template<typename MM>
    friend class NonZeroMaskLayerCPU;
};

} // raul namespace

#endif