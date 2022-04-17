// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef BAHDANAU_MONOTONIC_ATTENTION_INTERNAL_LAYERS_H
#define BAHDANAU_MONOTONIC_ATTENTION_INTERNAL_LAYERS_H

#include <training/base/initializers/RandomUniformInitializer.h>
#include <training/base/layers/TrainableLayer.h>

namespace raul::bahdanau
{

class BahdanauTrainableInitializerLayer : public raul::TrainableLayer
{
  public:
    BahdanauTrainableInitializerLayer(const raul::Name& name,
                                      const raul::TrainableParams& params,
                                      size_t numUnits,
                                      bool normalize,
                                      raul::dtype scoreBiasInit,
                                      raul::NetworkParameters& networkParameters)
        : TrainableLayer(name, "BahdanauTrainableInitializer", params, networkParameters, { false, true })
        , mNumUnits(numUnits)
        , mNormalize(normalize)
        , mScoreBiasInit(scoreBiasInit)
    {
        auto prefix = "BahdanauTrainableInitializerLayer[" + mName + "::ctor]: ";

        if (mInputs.size() != 0)
        {
            THROW(mTypeName, mName, "no input names expected");
        }

        if (mOutputs.size() != 2 && !(mNormalize && mOutputs.size() == 4))
        {
            THROW(mTypeName, mName, "wrong number of output names");
        }
        if (std::any_of(mOutputs.begin(), mOutputs.end(), [](const auto& s) { return s.empty(); }))
        {
            THROW(mTypeName, mName, "empty output name");
        }

        networkParameters.mWorkflow.tensorNeeded(name, mOutputs[0], raul::WShape{ 1u, 1u, 1u, mNumUnits }, DEC_TRAINABLE);

        networkParameters.mWorkflow.tensorNeeded(name, mOutputs[1], raul::WShape{ 1u, 1u, 1u, 1u }, DEC_TRAINABLE);

        if (mNormalize)
        {
            networkParameters.mWorkflow.tensorNeeded(name, mOutputs[2], raul::WShape{ 1u, 1u, 1u, 1u }, DEC_TRAINABLE);

            networkParameters.mWorkflow.tensorNeeded(name, mOutputs[3], raul::WShape{ 1u, 1u, 1u, mNumUnits }, DEC_TRAINABLE);
        }

        if (!mFrozen)
        {
            networkParameters.mWorkflow.copyDeclaration(name, mOutputs[0], mOutputs[0].grad(), DEC_TRAINABLE_GRAD);
            networkParameters.mWorkflow.copyDeclaration(name, mOutputs[1], mOutputs[1].grad(), DEC_TRAINABLE_GRAD);

            if (mNormalize)
            {
                networkParameters.mWorkflow.copyDeclaration(name, mOutputs[2], mOutputs[2].grad(), DEC_TRAINABLE_GRAD);
                networkParameters.mWorkflow.copyDeclaration(name, mOutputs[3], mOutputs[3].grad(), DEC_TRAINABLE_GRAD);
            }
        }
    }

    void initNotBSTensors() override
    {
        auto& work = mNetworkParams.mWorkflow;

        if (work.getExecutionTarget() == ExecutionTarget::CPU)
        {
            // Initialize attentionV
            const raul::dtype limit = std::sqrt(3.0_dt / static_cast<raul::dtype>(mNumUnits));
            raul::initializers::RandomUniformInitializer initializer{ 0.0_dt, 1.0_dt };
            initializer(mNetworkParams.mMemoryManager[mOutputs[0]]);
            mNetworkParams.mMemoryManager[mOutputs[0]] *= 2 * limit;
            mNetworkParams.mMemoryManager[mOutputs[0]] -= limit;

            // Initialize scoreBias
            mNetworkParams.mMemoryManager[mOutputs[1]][0] = mScoreBiasInit;

            // Initialize attentionG
            if (mNormalize)
            {
                mNetworkParams.mMemoryManager[mOutputs[2]][0] = std::sqrt(1.0_dt / static_cast<raul::dtype>(mNumUnits));
            }
        }
        else if (work.getExecutionTarget() == ExecutionTarget::CPUFP16)
        {
            // Initialize attentionV
            const raul::dtype limit = std::sqrt(3.0_dt / static_cast<raul::dtype>(mNumUnits));
            raul::initializers::RandomUniformInitializer initializer{ 0.0_dt, 1.0_dt };
            initializer(work.getMemoryManager<MemoryManagerFP16>()[mOutputs[0]]);
            work.getMemoryManager<MemoryManagerFP16>()[mOutputs[0]] *= TOHTYPE(2 * limit);
            work.getMemoryManager<MemoryManagerFP16>()[mOutputs[0]] -= TOHTYPE(limit);

            // Initialize scoreBias
            work.getMemoryManager<MemoryManagerFP16>()[mOutputs[1]][0] = TOHTYPE(mScoreBiasInit);

            // Initialize attentionG
            if (mNormalize)
            {
                work.getMemoryManager<MemoryManagerFP16>()[mOutputs[2]][0] = TOHTYPE(std::sqrt(1.0_dt / static_cast<raul::dtype>(mNumUnits)));
            }
        }
    }

    void forwardComputeImpl(raul::NetworkMode) override {}

    void backwardComputeImpl() override {}

  private:
    size_t mNumUnits;
    bool mNormalize;
    raul::dtype mScoreBiasInit;
};

class BahdanauConstantsInitializerLayer : public raul::BasicLayer
{
  public:
    BahdanauConstantsInitializerLayer(const raul::Name& name, const raul::BasicParams& params, size_t width, raul::NetworkParameters& networkParameters)
        : BasicLayer(name, "BahdanauConstantsInitializer", params, networkParameters, { false, false })
    {
        auto prefix = "BahdanauConstantsInitializerLayer[" + mName + "::ctor]: ";

        if (mOutputs.size() != 3)
        {
            THROW(mTypeName, mName, "wrong number of output names");
        }
        if (mOutputs[0].empty())
        {
            THROW(mTypeName, mName, "empty output name");
        }

        // Declare needed constant
        networkParameters.mWorkflow.tensorNeeded(name, mOutputs[0], raul::WShape{ 1u, 1u, 1u, width }, DEC_FORW_WRIT);
        networkParameters.mWorkflow.tensorNeeded(name, mOutputs[1], raul::WShape{ raul::BS(), 1u, 1u, 1u }, DEC_FORW_WRIT);
        networkParameters.mWorkflow.tensorNeeded(name, mOutputs[2], raul::WShape{ raul::BS(), 1u, 1u, width - 1u }, DEC_FORW_WRIT);
    }

    void forwardComputeImpl(raul::NetworkMode) override
    {
        auto& work = mNetworkParams.mWorkflow;

        if (work.getExecutionTarget() == ExecutionTarget::CPU)
        {
            auto& out = mNetworkParams.mMemoryManager[mOutputs[0]];
            std::fill(out.begin(), out.end(), 1.0_dt);
        }
        else if (work.getExecutionTarget() == ExecutionTarget::CPUFP16)
        {
            auto& out = work.getMemoryManager<MemoryManagerFP16>()[mOutputs[0]];
            std::fill(out.begin(), out.end(), 1.0_hf);
        }
        else
        {
            THROW_NONAME("BahdanauConstantsInitializerLayer", "unsupported execution target");
        }
    }

    void backwardComputeImpl() override {}
};

} // namespace raul::bahdanau

#endif