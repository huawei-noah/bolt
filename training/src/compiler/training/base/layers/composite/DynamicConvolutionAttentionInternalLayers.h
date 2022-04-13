// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef DYNAMIC_CONVOLUTION_ATTENTION_INTERNAL_LAYERS_H
#define DYNAMIC_CONVOLUTION_ATTENTION_INTERNAL_LAYERS_H

#include <training/base/initializers/XavierInitializer.h>
#include <training/base/layers/TrainableLayer.h>

#include <math.h>

namespace
{

size_t comb(size_t n, size_t k)
{
    if (k > n)
    {
        return 0;
    }
    if (k * 2 > n)
    {
        k = n - k;
    }
    if (k == 0)
    {
        return 1;
    }

    size_t result = n;
    for (size_t i = 2; i <= k; ++i)
    {
        result *= (n - i + 1);
        result /= i;
    }
    return result;
}

} // anonymous namespace

namespace raul::dca
{

const raul::dtype MIN_INPUT = 1.775e-38_dt;
const raul::dtype MIN_OUTPUT = -1.0e6_dt;

class DCATrainableInitializerLayer : public raul::TrainableLayer
{
  public:
    DCATrainableInitializerLayer(const raul::Name& name,
                                 const raul::TrainableParams& params,
                                 size_t numUnits,
                                 raul::dtype priorAlpha,
                                 raul::dtype priorBeta,
                                 size_t priorFilterSize,
                                 raul::NetworkParameters& networkParameters)
        : TrainableLayer(name, "DCATrainableInitializer", params, networkParameters, { false, true })
        , mPriorAlpha(priorAlpha)
        , mPriorBeta(priorBeta)
        , mPriorFilterSize(priorFilterSize)
    {
        auto prefix = "DCATrainableInitializerLayer[" + mName + "::ctor]: ";

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
        initializers::XavierUniformInitializer initializer;
        Name priorFilterName = Name(mName.str().substr(0, mName.str().find_last_of("::") - 1)) / "apply_prior_filters" / "Weights";
        if (mNetworkParams.mWorkflow.getExecutionTarget() == ExecutionTarget::CPU)
        {
            initializer(mNetworkParams.mMemoryManager[mOutputs[0]]);

            Tensor& priorFilter = mNetworkParams.mMemoryManager[priorFilterName];
            const auto divisor = static_cast<dtype>(tgamma(mPriorAlpha + mPriorBeta + TODTYPE(mPriorFilterSize - 1)) * tgamma(mPriorAlpha) * tgamma(mPriorBeta) / tgamma(mPriorAlpha + mPriorBeta));

#if defined(_OPENMP)
#pragma omp parallel for
#endif
            for (size_t i = 0; i < priorFilter.size(); ++i)
            {
                priorFilter[i] = TODTYPE(comb(mPriorFilterSize - 1, mPriorFilterSize - 1 - i)) * tgamma(TODTYPE(mPriorFilterSize - 1 - i) + mPriorAlpha) * tgamma(TODTYPE(i) + mPriorBeta) / divisor;
            }
        }
        else
        {
            THROW(mTypeName, mName, "unsupported execution target");
        }
    }

    void forwardComputeImpl(NetworkMode) override {}
    void backwardComputeImpl() override {}

  private:
    dtype mPriorAlpha;
    dtype mPriorBeta;
    size_t mPriorFilterSize;
};

// Reshape calculated features
class CustomReshapeLayer : public raul::BasicLayer
{
  public:
    CustomReshapeLayer(const raul::Name& name, const raul::BasicParams& params, raul::NetworkParameters& networkParameters)
        : BasicLayer(name, "CustomReshape", params, networkParameters, { false, false })
        , mOutputHeight(networkParameters.mWorkflow.getHeight(mInputs[0]))
    {
        auto prefix = mTypeName + "[" + mName + "::ctor]: ";

        if (mInputs.size() != 1)
        {
            THROW(mTypeName, mName, "wrong number of input names");
        }
        if (mOutputs.size() != 1)
        {
            THROW(mTypeName, mName, "wrong number of output names");
        }

        mNetworkParams.mWorkflow.copyDeclaration(name, mInputs[0], raul::Workflow::Usage::Forward, raul::Workflow::Mode::Read);

        mNetworkParams.mWorkflow.copyDeclaration(mName, mInputs[0], mInputs[0].grad(), DEC_BACK_WRIT_ZERO);

        networkParameters.mWorkflow.tensorNeeded(name, mOutputs[0], raul::WShape{ BS(), mOutputDepth, mOutputHeight, mOutputWidth }, DEC_FORW_WRIT);

        mNetworkParams.mWorkflow.copyDeclaration(mName, mOutputs[0], mOutputs[0].grad(), DEC_BACK_READ);
    }

    void forwardComputeImpl(raul::NetworkMode) override
    {
        if (mNetworkParams.mWorkflow.getExecutionTarget() == ExecutionTarget::CPU)
        {
            Tensor& output = mNetworkParams.mMemoryManager[mOutputs[0]];
            const Tensor& input = mNetworkParams.mMemoryManager[mInputs[0]];

            const size_t realBatchSize = mNetworkParams.mWorkflow.getBatch(mOutputs[0]);

            // Get 4D View
            auto output4D = output.reshape(yato::dims(realBatchSize, mOutputDepth, mOutputHeight, mOutputWidth));
            auto input4D = input.reshape(yato::dims(1u, 1u, mOutputHeight, realBatchSize * mOutputWidth));
#if defined(_OPENMP)
#pragma omp parallel for
#endif
            for (size_t hi = 0; hi < mOutputHeight; ++hi)
            {
                for (size_t wi = 0; wi < realBatchSize * mOutputWidth; ++wi)
                {
                    output4D[wi / mOutputWidth][0][hi][wi % mOutputWidth] = input4D[0][0][hi][wi];
                }
            }
        }
        else
        {
            THROW(mTypeName, mName, "unsupported execution target");
        }
    }

    void backwardComputeImpl() override
    {
        if (mNetworkParams.mWorkflow.getExecutionTarget() == ExecutionTarget::CPU)
        {
            const Tensor& deltas = mNetworkParams.mMemoryManager[mOutputs[0].grad()];
            Tensor& inputNabla = mNetworkParams.mMemoryManager[mInputs[0].grad()];

            const size_t realBatchSize = mNetworkParams.mWorkflow.getBatch(mOutputs[0].grad());

            // Get 4D View
            auto deltas4D = deltas.reshape(yato::dims(realBatchSize, mOutputDepth, mOutputHeight, mOutputWidth));
            auto inputNabla4D = inputNabla.reshape(yato::dims(1u, 1u, mOutputHeight, realBatchSize * mOutputWidth));

#if defined(_OPENMP)
#pragma omp parallel for
#endif
            for (size_t hi = 0; hi < mOutputHeight; ++hi)
            {
                for (size_t wi = 0; wi < realBatchSize * mOutputWidth; ++wi)
                {
                    inputNabla4D[0][0][hi][wi] += deltas4D[wi / mOutputWidth][0][hi][wi % mOutputWidth];
                }
            }
        }
        else
        {
            THROW(mTypeName, mName, "unsupported execution target");
        }
    }

  private:
    // Hardcoded values
    size_t mOutputDepth{ 1u };
    size_t mOutputWidth{ 8u };

    size_t mOutputHeight;
};

// Do not use very small values
class DCAConstantsInitializerLayer : public raul::BasicLayer
{
  public:
    DCAConstantsInitializerLayer(const raul::Name& name, const raul::BasicParams& params, raul::NetworkParameters& networkParameters)
        : BasicLayer(name, "DCAConstantsInitializer", params, networkParameters, { false, false })
    {
        auto prefix = "DCAConstantsInitializerLayer[" + mName + "::ctor]: ";

        if (mOutputs.size() != 2)
        {
            THROW(mTypeName, mName, "wrong number of output names");
        }
        if (mOutputs[0].empty() || mOutputs[1].empty())
        {
            THROW(mTypeName, mName, "empty output name");
        }

        // Declare needed constants
        networkParameters.mWorkflow.tensorNeeded(name, mOutputs[0], raul::WShape{ 1u, 1u, 1u, 1u }, DEC_FORW_WRIT);
        networkParameters.mWorkflow.tensorNeeded(name, mOutputs[1], raul::WShape{ 1u, 1u, 1u, 1u }, DEC_FORW_WRIT);
    }

    void forwardComputeImpl(raul::NetworkMode) override
    {
        // Initialize constants
        if (mNetworkParams.mWorkflow.getExecutionTarget() == ExecutionTarget::CPU)
        {
            mNetworkParams.mMemoryManager[mOutputs[0]][0] = MIN_INPUT;
            mNetworkParams.mMemoryManager[mOutputs[1]][0] = MIN_OUTPUT;
        }
        else
        {
            THROW(mTypeName, mName, "unsupported execution target");
        }
    }

    void backwardComputeImpl() override {}
};

} // raul::dca namespace

#endif