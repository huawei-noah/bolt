// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef LOCATION_SENSITIVE_ATTENTION_INTERNAL_LAYERS_H
#define LOCATION_SENSITIVE_ATTENTION_INTERNAL_LAYERS_H

#include <training/base/initializers/XavierInitializer.h>
#include <training/base/layers/TrainableLayer.h>

namespace raul::lsa
{

class LSATrainableInitializerLayer : public raul::TrainableLayer
{
  public:
    LSATrainableInitializerLayer(const raul::Name& name, const raul::TrainableParams& params, size_t numUnits, raul::NetworkParameters& networkParameters)
        : TrainableLayer(name, "LSATrainableInitializer", params, networkParameters, { false, true })
    {
        auto prefix = "LSATrainableInitializerLayer[" + mName + "::ctor]: ";

        if (mInputs.size() != 0)
        {
            THROW(mTypeName, mName, "no input names expected");
        }

        if (mOutputs.size() != 2)
        {
            THROW(mTypeName, mName, "wrong number of output names");
        }
        if (mOutputs[0].empty() || mOutputs[1].empty())
        {
            THROW(mTypeName, mName, "empty output name");
        }

        networkParameters.mWorkflow.tensorNeeded(name, mOutputs[0], raul::WShape{ 1u, 1u, 1u, numUnits }, DEC_TRAINABLE);

        networkParameters.mWorkflow.tensorNeeded(name, mOutputs[1], raul::WShape{ 1u, 1u, 1u, numUnits }, DEC_TRAINABLE);

        if (!mFrozen)
        {
            networkParameters.mWorkflow.copyDeclaration(name, mOutputs[0], mOutputs[0].grad(), DEC_TRAINABLE_GRAD);
            networkParameters.mWorkflow.copyDeclaration(name, mOutputs[1], mOutputs[1].grad(), DEC_TRAINABLE_GRAD);
        }
    }

    void initNotBSTensors() override
    {
        // Initialize trainable params
        raul::initializers::XavierUniformInitializer initializer;
        initializer(mNetworkParams.mMemoryManager[mOutputs[0]]);
    }

    void forwardComputeImpl(raul::NetworkMode) override {}

    void backwardComputeImpl() override {}
};

// Give needed constant
class LSAConstantsInitializerLayer : public raul::BasicLayer
{
  public:
    LSAConstantsInitializerLayer(const raul::Name& name, const raul::BasicParams& params, raul::NetworkParameters& networkParameters)
        : BasicLayer(name, "LSAConstantsInitializer", params, networkParameters, { false, false })
    {
        auto prefix = "LSAConstantsInitializerLayer[" + mName + "::ctor]: ";

        if (mOutputs.size() != 1)
        {
            THROW(mTypeName, mName, "wrong number of output names");
        }
        if (mOutputs[0].empty())
        {
            THROW(mTypeName, mName, "empty output name");
        }

        // Declare needed constants
        networkParameters.mWorkflow.tensorNeeded(name, mOutputs[0], raul::WShape{ 1u, 1u, 1u, 1u }, DEC_FORW_WRIT);
    }

    void forwardComputeImpl(raul::NetworkMode) override
    {
        // Initialize constants
        mNetworkParams.mMemoryManager[mOutputs[0]][0] = 1.0_dt;
    }

    void backwardComputeImpl() override {}
};

}

#endif