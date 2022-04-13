// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef ATTENTION_MASK_CREATOR_LAYER_H
#define ATTENTION_MASK_CREATOR_LAYER_H

#include <training/base/layers/BasicLayer.h>

namespace raul
{

class AttentionMaskCreatorLayer : public raul::BasicLayer
{
  public:
    AttentionMaskCreatorLayer(const raul::Name& name, const raul::BasicParams& params, size_t maskLen, raul::NetworkParameters& networkParameters)
        : BasicLayer(name, "AttentionMaskCreator", params, networkParameters, { false, false })
        , mMaskLen(maskLen)
    {
        auto prefix = "AttentionMaskCreatorLayer[" + mName + "::ctor]: ";

        if (mInputs.size() != 1)
        {
            THROW(mTypeName, mName, "wrong number of input names");
        }
        if (mInputs[0].empty())
        {
            THROW(mTypeName, mName, "empty input name");
        }

        if (mOutputs.size() > 2)
        {
            THROW(mTypeName, mName, "wrong number of output names");
        }
        if (std::any_of(mOutputs.begin(), mOutputs.end(), [](const auto& s) { return s.empty(); }))
        {
            THROW(mTypeName, mName, "empty output name");
        }

        // Declare needed constants
        networkParameters.mWorkflow.copyDeclaration(name, mInputs[0], raul::Workflow::Usage::Forward, raul::Workflow::Mode::Read);

        networkParameters.mWorkflow.tensorNeeded(name, mOutputs[0], raul::WShape{ raul::BS(), 1u, mMaskLen, 1u }, DEC_FORW_WRIT);

        if (mOutputs.size() == 2)
        {
            networkParameters.mWorkflow.copyDeclaration(name, mOutputs[0], mOutputs[1], DEC_FORW_WRIT_NOMEMOPT);
        }
    }

    void forwardComputeImpl(raul::NetworkMode) override
    {
        auto& work = mNetworkParams.mWorkflow;

        // Fill mask for memory depending on memorySeqLength
        if (work.getExecutionTarget() == ExecutionTarget::CPU)
        {
            const Tensor& memorySeqLength = mNetworkParams.mMemoryManager[mInputs[0]];
            Tensor& mask = mNetworkParams.mMemoryManager[mOutputs[0]];
            size_t batchSize = memorySeqLength.getBatchSize();

            for (size_t i = 0; i < batchSize; ++i)
            {
                if (memorySeqLength[i] == 0.0_dt)
                {
                    THROW("AttentionMaskCreatorLayer", mName, "all values in memorySeqLength must be greater than zero");
                }
                for (size_t j = 0; j < mMaskLen; ++j)
                {
                    mask[i * mMaskLen + j] = static_cast<dtype>(static_cast<dtype>(j) < memorySeqLength[i]);
                    if (mOutputs.size() == 2)
                    {
                        mNetworkParams.mMemoryManager[mOutputs[1]][i * mMaskLen + j] = -std::numeric_limits<dtype>::infinity();
                    }
                }
            }
        }
        else if (work.getExecutionTarget() == ExecutionTarget::CPUFP16)
        {
            const auto& memorySeqLength = work.getMemoryManager<MemoryManagerFP16>()[mInputs[0]];
            auto& mask = work.getMemoryManager<MemoryManagerFP16>()[mOutputs[0]];
            size_t batchSize = memorySeqLength.getBatchSize();

            for (size_t i = 0; i < batchSize; ++i)
            {
                if (memorySeqLength[i] == 0.0_hf)
                {
                    THROW("AttentionMaskCreatorLayer", mName, "all values in memorySeqLength must be greater than zero");
                }
                for (size_t j = 0; j < mMaskLen; ++j)
                {
                    mask[i * mMaskLen + j] = TOHTYPE(TOHTYPE(j) < memorySeqLength[i]);
                    if (mOutputs.size() == 2)
                    {
                        work.getMemoryManager<MemoryManagerFP16>()[mOutputs[1]][i * mMaskLen + j] = -std::numeric_limits<half>::infinity();
                    }
                }
            }
        }
        else
        {
            THROW(mTypeName, mName, "unsupported execution target");
        }
    }

    void backwardComputeImpl() override {}

  private:
    size_t mMaskLen;
};

}
#endif