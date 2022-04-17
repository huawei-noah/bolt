// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef BASIC_LAYER_H
#define BASIC_LAYER_H

#include <training/base/common/NetworkParameters.h>
#include <training/base/layers/BasicImpl.h>
#include <training/compiler/Workflow.h>
#include <training/system/Profiler.h>

#include <training/base/layers/parameters/BasicParameters.h>

namespace raul
{

/**
 * @brief Base class for all layers
 * @param parent used for layer internal params sharing
 * Layer that creates new layers with shared params MUST:
 *     1. Aggregate gradients for shared weights in the end of backwardCompute
 *     2. Make sure, that params of layer aliases are not visible via getTrainableParametersNames() and getParameters()
 *
 */
class BasicLayer
{
  public:
    BasicLayer(const Name& name, const Name& typeName, const BasicParams& params, NetworkParameters& networkParams, std::pair<bool, bool> doChecks = { true, true })
        : mName(name)
        , mTypeName(typeName)
        , mInputs(params.getInputs())
        , mOutputs(params.getOutputs())
        , mSharedWeights(params.getSharedWeights())
        , mSharedLayer(params.getSharedLayer())
        , mNetworkParams(networkParams)
    {
        if (mName.empty())
        {
            THROW(mTypeName, mName, "empty layer name");
        }

        if (doChecks.first)
        {
            if (mInputs.empty())
            {
                THROW(mTypeName, mName, "empty inputs");
            }
            if (std::any_of(mInputs.begin(), mInputs.end(), [](const auto& s) { return s.empty(); }))
            {
                THROW(mTypeName, mName, "empty input name");
            }
        }

        if (doChecks.second)
        {
            if (mOutputs.empty())
            {
                THROW(mTypeName, mName, "empty outputs");
            }
            if (std::any_of(mOutputs.begin(), mOutputs.end(), [](const auto& s) { return s.empty(); }))
            {
                THROW(mTypeName, mName, "empty output name");
            }
        }
    }

    virtual ~BasicLayer() = default;

    BasicLayer(BasicLayer&&) = default;
    BasicLayer& operator=(BasicLayer&&) = delete;

    /*
     * @brief Define implementation of layer. Used by Compiler
     */
    void setImpl(std::unique_ptr<BasicImpl> impl) { mImpl = std::move(impl); }

    /*
     * @brief Override to initialize non-zero tensors without batch dimension.
     * Consider to call explicitly when using eager workflow and directly call forward / backward for layers.
     * Executed when Workflow::prepareMemoryForTraining() called
     */
    virtual void initNotBSTensors()
    {
        if (mImpl)
        {
            mImpl->initNotBSTensors();
        }
    }

    virtual void onBatchSizeChanged(size_t newBatchSize)
    {
        if (mImpl)
        {
            mImpl->onBatchSizeChanged(newBatchSize);
        }
    }

    virtual void forwardComputeImpl(NetworkMode mode)
    {
        if (mImpl)
        {
            mImpl->forwardComputeImpl(mode);
        }
        else
        {
            THROW(mTypeName, mName, "Layer has no forward implementation");
        }
    }

    virtual void backwardComputeImpl()
    {
        if (mImpl)
        {
            mImpl->backwardComputeImpl();
        }
        else
        {
            THROW(mTypeName, mName, "Layer has no backward implementation");
        }
    }

    virtual void forwardCompute(NetworkMode mode)
    {
        try
        {
            if (mNetworkParams.mCallback && mNetworkParams.mWorkflow.getExecutionTarget() == ExecutionTarget::CPU)
            {
                mNetworkParams.mCallback(this, mNetworkParams.mMemoryManager, raul::NetworkParameters::CallbackPlace::Before_Forward);
            }
            else if (mNetworkParams.mCallbackFP16 && mNetworkParams.mWorkflow.getExecutionTarget() == ExecutionTarget::CPUFP16)
            {
                mNetworkParams.mCallbackFP16(this, mNetworkParams.mMemoryManagerFP16, raul::NetworkParameters::CallbackPlace::Before_Forward);
            }
        }
        catch (...)
        {
            THROW(mTypeName, mName, "Cannot execute Before_Forward callback");
        }
        try
        {
            MEASURE_BLOCK(mTypeName + "[" + mName + "::forwardCompute]");
            forwardComputeImpl(mode);
        }
        catch (...)
        {
            THROW(mTypeName, mName, "Cannot compute the forward path in the layer");
        }
        try
        {
            if (mNetworkParams.mCallback && mNetworkParams.mWorkflow.getExecutionTarget() == ExecutionTarget::CPU)
            {
                mNetworkParams.mCallback(this, mNetworkParams.mMemoryManager, raul::NetworkParameters::CallbackPlace::After_Forward);
            }
            else if (mNetworkParams.mCallbackFP16 && mNetworkParams.mWorkflow.getExecutionTarget() == ExecutionTarget::CPUFP16)
            {
                mNetworkParams.mCallbackFP16(this, mNetworkParams.mMemoryManagerFP16, raul::NetworkParameters::CallbackPlace::After_Forward);
            }
        }
        catch (...)
        {
            THROW(mTypeName, mName, "Cannot execute After_Forward callback");
        }
    }
    virtual void backwardCompute()
    {
        try
        {

            if (mNetworkParams.mCallback && mNetworkParams.mWorkflow.getExecutionTarget() == ExecutionTarget::CPU)
            {
                mNetworkParams.mCallback(this, mNetworkParams.mMemoryManager, raul::NetworkParameters::CallbackPlace::Before_Backward);
            }
            else if (mNetworkParams.mCallbackFP16 && mNetworkParams.mWorkflow.getExecutionTarget() == ExecutionTarget::CPUFP16)
            {
                mNetworkParams.mCallbackFP16(this, mNetworkParams.mMemoryManagerFP16, raul::NetworkParameters::CallbackPlace::Before_Backward);
            }
        }
        catch (...)
        {
            THROW(mTypeName, mName, "Cannot execute Before_Backward callback");
        }
        try
        {
            MEASURE_BLOCK(mTypeName + "[" + mName + "::backwardCompute]");
            backwardComputeImpl();
        }
        catch (...)
        {
            THROW(mTypeName, mName, "Cannot compute the backward path in the layer");
        }
        try
        {
            if (mNetworkParams.mCallback && mNetworkParams.mWorkflow.getExecutionTarget() == ExecutionTarget::CPU)
            {
                mNetworkParams.mCallback(this, mNetworkParams.mMemoryManager, raul::NetworkParameters::CallbackPlace::After_Backward);
            }
            else if (mNetworkParams.mCallbackFP16 && mNetworkParams.mWorkflow.getExecutionTarget() == ExecutionTarget::CPUFP16)
            {
                mNetworkParams.mCallbackFP16(this, mNetworkParams.mMemoryManagerFP16, raul::NetworkParameters::CallbackPlace::After_Backward);
            }
        }
        catch (...)
        {
            THROW(mTypeName, mName, "Cannot execute After_Backward callback");
        }
    }

    [[nodiscard]] const Name& getName() const { return mName; }
    [[nodiscard]] std::string getTypeName() const { return mTypeName; }

    [[nodiscard]] const Names& getInputs() const { return mInputs; }
    [[nodiscard]] const Names& getOutputs() const { return mOutputs; }
    [[nodiscard]] const Names& getSharedWeights() const { return mSharedWeights; }
    [[nodiscard]] const Name& getSharedLayer() const { return mSharedLayer; }

    [[nodiscard]] virtual bool isTrainable() const { return false; }

    const NetworkParameters& getNetworkParams() const { return mNetworkParams; }
    NetworkParameters& getNetworkParams() { return mNetworkParams; }

    virtual void print(std::ostream& out, std::string prefix = "") const { out << prefix << mTypeName << "[" << mName << "]" << std::endl; }

  protected:
    const Name mName;
    const std::string mTypeName;
    const Names mInputs;
    const Names mOutputs;
    const Names mSharedWeights;
    const Name mSharedLayer;

    NetworkParameters& mNetworkParams;

    std::unique_ptr<BasicImpl> mImpl;
};

} // raul namespace

#endif // BASIC_LAYER_H