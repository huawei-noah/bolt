// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef WORKFLOW_ACTIONS_H
#define WORKFLOW_ACTIONS_H

#include "WorkflowPool.h"
#include <training/base/common/Tensor.h>
#include <training/base/layers/BasicLayer.h>
#include <training/base/loss/scaling/ScalingStrategy.h>

namespace raul
{

template<typename MM>
struct TensorAction : public Workflow::Action
{
    TensorAction(MM& manager, const Name& name)
        : mMemoryManager(manager)
        , mName(name)
    {
    }

    MM& mMemoryManager;
    Name mName;
};

struct LayerAction : public Workflow::Action
{
    LayerAction(BasicLayer* layer)
        : mLayer(layer)
    {
    }

    BasicLayer* mLayer;
};

template<typename MM>
struct CreateTensor : public TensorAction<MM>
{
    CreateTensor(MM& manager, const Name& name, const WShape shapeVar, const Workflow& work)
        : TensorAction<MM>(manager, name)
        , mShape(shapeVar)
        , mWork(work)
    {
    }

    virtual void execute(const Workflow::ActionParam&) override { this->mMemoryManager.createTensor(this->mName, mShape.getShape(mWork)); }

    virtual std::string type() const override { return "CreateTensor"; }

    WShape mShape;
    const Workflow& mWork;
};

template<typename MM>
struct CreateShape : public TensorAction<MM>
{
    CreateShape(MM& manager, const Name& name, const WShape shapeVar, const Workflow& work)
        : TensorAction<MM>(manager, name)
        , mShape(shapeVar)
        , mWork(work)
    {
    }

    virtual void execute(const Workflow::ActionParam&) override { this->mMemoryManager.createShape(this->mName, mShape.getShape(mWork), mWork.getAllocationMode()); }

    virtual std::string type() const override { return "CreateShape"; }

    WShape mShape;
    const Workflow& mWork;
};

template<typename MM>
struct DeleteTensor : public TensorAction<MM>
{
    DeleteTensor(MM& manager, const Name& name)
        : TensorAction<MM>(manager, name)
    {
    }

    virtual void execute(const Workflow::ActionParam&) override
    {
        if (this->mMemoryManager.tensorExists(this->mName)) this->mMemoryManager.deleteTensor(this->mName);
    }

    virtual std::string type() const override { return "DeleteTensor"; }
};

template<typename MM>
struct Allocate : public TensorAction<MM>
{
    Allocate(MM& manager, const Name& name, const Workflow& work, std::shared_ptr<WorkflowPool<MM>>& pool)
        : TensorAction<MM>(manager, name)
        , mWork(work)
        , mPool(pool)
        , mPoolTensorName(name)
    {
    }

    Allocate(MM& manager, const Name& name, const Name& namePool, const Workflow& work, std::shared_ptr<WorkflowPool<MM>>& pool)
        : TensorAction<MM>(manager, name)
        , mWork(work)
        , mPool(pool)
        , mPoolTensorName(namePool)
    {
    }

    virtual void execute(const Workflow::ActionParam&) override
    {
        typename MM::type* offset = nullptr;

        if (mWork.getAllocationMode() == AllocationMode::POOL)
        {
            offset = mPool->getOffset(mPoolTensorName);
        }

        this->mMemoryManager[this->mName].memAllocate(offset);
    }

    virtual std::string type() const override { return "Allocate"; }

    const Workflow& mWork;
    std::shared_ptr<WorkflowPool<MM>>& mPool;
    const Name mPoolTensorName;
};

template<typename MM>
struct Deallocate : public TensorAction<MM>
{
    Deallocate(MM& manager, const Name& name)
        : TensorAction<MM>(manager, name)
    {
    }

    virtual void execute(const Workflow::ActionParam&) override { this->mMemoryManager[this->mName].memClear(); }

    virtual std::string type() const override { return "Deallocate"; }
};

struct Forward : public LayerAction
{
    Forward(BasicLayer* layer, raul::NetworkMode mode, Workflow& work)
        : LayerAction(layer)
        , mMode(mode)
        , mWork(work)
    {
    }

    virtual void execute(const Workflow::ActionParam&) override
    {
        try
        {
            auto listeners = mWork.getListeners(mLayer->getName());

            for (auto listener : listeners)
            {
                listener->BeforeForward(mWork);
            }

            mLayer->forwardCompute(mMode);

            for (auto listener : listeners)
            {
                listener->AfterForward(mWork);
            }
        }
        catch (...)
        {
            THROW_NONAME("Forward", "Cannot execute a layer action");
        }
    }

    virtual std::string type() const override { return "Forward"; }

    raul::NetworkMode mMode;
    Workflow& mWork;
};

struct InitNonBS : public LayerAction
{
    InitNonBS(BasicLayer* layer)
        : LayerAction(layer)
    {
    }

    virtual void execute(const Workflow::ActionParam&) override { mLayer->initNotBSTensors(); }

    virtual std::string type() const override { return "InitNonBS"; }
};

struct UpdateBS : public LayerAction
{
    UpdateBS(BasicLayer* layer, Workflow& work)
        : LayerAction(layer)
        , mWork(work)
    {
    }

    virtual void execute(const Workflow::ActionParam&) override { mLayer->onBatchSizeChanged(mWork.getBatchSize()); }

    virtual std::string type() const override { return "UpdateBS"; }

    Workflow& mWork;
};

struct Backward : public LayerAction
{
    Backward(BasicLayer* layer, Workflow& work)
        : LayerAction(layer)
        , mWork(work)
    {
    }

    Backward setScaling(ScalingStrategy scaling)
    {
        mScaling = scaling;
        return *this;
    }

    void execute(const Workflow::ActionParam&) override
    {
        try
        {
            auto listeners = mWork.getListeners(mLayer->getName());

            try
            {
                for (auto listener : listeners)
                {
                    listener->BeforeBackward(mWork);
                }
            }
            catch (...)
            {
                THROW_NONAME("Backward", "Cannot execute BeforeBackward listeners");
            }

            if (mScaling)
            {
                scaleGrads();
            }

            // expected to be a pure function
            mLayer->backwardCompute();

            /// Currently, it cannot be implemented due to architecture restrictions.
            propagateScale();

            try
            {
                for (auto listener : listeners)
                {
                    listener->AfterBackward(mWork);
                }
            }
            catch (...)
            {
                THROW_NONAME("Backward", "Cannot execute AfterBackward listeners");
            }
        }
        catch (...)
        {
            THROW_NONAME("Backward", "Cannot execute a layer action");
        }
    }

    [[nodiscard]] std::string type() const override { return "Backward"; }

  private:
    void rescaleGrads(Names& grads);
    void applyScale(const Names& grads, dtype scale);
    void scaleGrads();

    /**
     * Copy scale from output gradients of the layer to input
     * gradients and parameter ones if they are presented.
     *
     * This function is a workaround.
     */
    void propagateScale();

  public:
    Workflow& mWork;
    std::optional<ScalingStrategy> mScaling;
};

template<typename MM>
struct Compress : public TensorAction<MM>
{
    Compress(MM& manager, const Name& name, CompressionMode mode)
        : TensorAction<MM>(manager, name)
        , mMode(mode)
    {
    }

    virtual void execute(const Workflow::ActionParam&) override
    {
        auto& t = this->mMemoryManager[this->mName];
        t.compress(mMode);
    }

    virtual std::string type() const override { return "Compress"; }

    CompressionMode mMode;
};

template<typename MM>
struct Decompress : public TensorAction<MM>
{
    Decompress(MM& manager, const Name& name, CompressionMode mode)
        : TensorAction<MM>(manager, name)
        , mMode(mode)
    {
    }

    virtual void execute(const Workflow::ActionParam&) override
    {
        auto& t = this->mMemoryManager[this->mName];
        t.decompress(mMode);
    }

    virtual std::string type() const override { return "Decompress"; }

    CompressionMode mMode;
};

template<typename MM>
struct Zero : public TensorAction<MM>
{
    Zero(MM& manager, const Name& name)
        : TensorAction<MM>(manager, name)
    {
    }

    virtual void execute(const Workflow::ActionParam&) override
    {
        auto& t = this->mMemoryManager[this->mName];
        std::fill(t.begin(), t.end(), TOMMTYPE(0));
    }

    virtual std::string type() const override { return "Zero"; }
};

} // raul namespace

#endif