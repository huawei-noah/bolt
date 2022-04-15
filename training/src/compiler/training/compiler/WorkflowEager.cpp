// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <training/base/layers/BasicLayer.h>
#include <training/compiler/Compiler.h>

#include "WorkflowActions.h"
#include "WorkflowDB.h"
#include "WorkflowEager.h"

namespace raul
{

void WorkflowEager::prepareMemoryForTraining()
{
    Workflow::prepareMemoryForTraining();

    for (auto& layer : mLayers)
    {
        layer->initNotBSTensors();
    }

    executePipeline(Workflow::Pipelines::Zero);
}

void WorkflowEager::preparePipelines(Execution)
{
    try
    {
        if (mUseCompiler)
        {
            if (!mCompiler->isResolved())
            {
                createImplementations();
            }
        }

        if (mIsForwardCalled)
        {
            THROW_NONAME("WorkflowEager", "forward called without leading backward");
        }

        // check same outputs usage globally
        {
            std::unordered_set<Name> allOutputs;
            for (auto& layer : mLayers)
            {
                for (const auto& output : layer->getOutputs())
                {
                    auto it = allOutputs.find(output);
                    if (it != allOutputs.end())
                    {
                        THROW_NONAME("WorkflowEager", "the workflow is not correct, there are same outputs defined: " + output);
                    }
                    else
                    {
                        allOutputs.insert(output);
                    }
                }
            }
        }

        // check unique names (inputs, weights) usage per layer
        {
            for (auto& layer : mLayers)
            {
                if (!isUniqueNames(layer->getInputs()))
                {
                    THROW_NONAME("WorkflowEager", "the workflow is not correct, there are same inputs defined for layer " + layer->getName());
                }

                if (!isUniqueNames(layer->getSharedWeights()))
                {
                    THROW_NONAME("WorkflowEager", "the workflow is not correct, there are same weights defined for layer " + layer->getName());
                }
            }
        }

        clearPipelines();

        createAuxPipelines();

        mIsPipelinesPrepared = true;
    }
    catch (...)
    {
        THROW_NONAME("WorkflowEager", "Cannot prepare pipelines");
    }
}

void WorkflowEager::createAuxPipelines()
{
    try
    {
        // Tensor vs index in mTensorNeeded
        std::unordered_map<Name, size_t> uniqueTensors;

        std::unordered_set<Name> layers;
        for (const auto& layer : mWorkflowDB->getLayersTable())
        {
            layers.insert(layer.first);
        }

        // check inequality, fill uniqueTensors
        for (const Name& lName : layers)
        {
            try
            {
                std::vector<Name> tensors = mWorkflowDB->getSlice(mWorkflowDB->getLayersTable(), lName);
                for (const auto& tName : tensors)
                {
                    auto uniqueIt = uniqueTensors.find(tName);
                    if (uniqueIt != uniqueTensors.end())
                    {
                        auto tensorUsage = mWorkflowDB->getCell(mWorkflowDB->getLayersTable(), lName, tName);

                        if (!mWorkflowDB->isCellElementEmpty(tensorUsage, Usage::Forward))
                        {
                            if (mWorkflowDB->getUsage((*uniqueIt).second).isZero || mWorkflowDB->getUsage(tensorUsage[static_cast<size_t>(Usage::Forward)]).isZero)
                            {
                                mWorkflowDB->getUsage((*uniqueIt).second).isZero = true;
                                mWorkflowDB->getUsage(tensorUsage[static_cast<size_t>(Usage::Forward)]).isZero = true;
                            }

                            checkAttributesInequality((*uniqueIt).second, tensorUsage[static_cast<size_t>(Usage::Forward)], tName);
                        }
                        if (!mWorkflowDB->isCellElementEmpty(tensorUsage, Usage::Backward))
                        {
                            if (mWorkflowDB->getUsage((*uniqueIt).second).isZero || mWorkflowDB->getUsage(tensorUsage[static_cast<size_t>(Usage::Backward)]).isZero)
                            {
                                mWorkflowDB->getUsage((*uniqueIt).second).isZero = true;
                                mWorkflowDB->getUsage(tensorUsage[static_cast<size_t>(Usage::Backward)]).isZero = true;
                            }

                            checkAttributesInequality((*uniqueIt).second, tensorUsage[static_cast<size_t>(Usage::Backward)], tName);
                        }
                    }
                    else
                    {
                        auto tensorUsage = mWorkflowDB->getCell(mWorkflowDB->getLayersTable(), lName, tName);

                        Usage tUsage = Usage::Backward;
                        if (!mWorkflowDB->isCellElementEmpty(tensorUsage, Usage::Forward))
                        {
                            tUsage = Usage::Forward;
                        }

                        if (!mWorkflowDB->isCellElementEmpty(tensorUsage, Usage::Forward) && !mWorkflowDB->isCellElementEmpty(tensorUsage, Usage::Backward))
                        {
                            if (mWorkflowDB->getUsage(tensorUsage[static_cast<size_t>(Usage::Forward)]).isZero || mWorkflowDB->getUsage(tensorUsage[static_cast<size_t>(Usage::Backward)]).isZero)
                            {
                                mWorkflowDB->getUsage(tensorUsage[static_cast<size_t>(Usage::Forward)]).isZero = true;
                                mWorkflowDB->getUsage(tensorUsage[static_cast<size_t>(Usage::Backward)]).isZero = true;
                            }

                            checkAttributesInequality(tensorUsage[static_cast<size_t>(Usage::Forward)], tensorUsage[static_cast<size_t>(Usage::Backward)], tName);
                        }

                        uniqueTensors.insert({ tName, tensorUsage[static_cast<size_t>(tUsage)] });
                    }
                }
            }
            catch (...)
            {
                THROW_NONAME("WorkflowEager", "Cannot process layer `" + lName + "`");
            }
        }

        // create pipelines
        execTargetCreateAuxPipelines(uniqueTensors);

        for (const auto& layer : mLayers)
        {
            mPipelineCreateBatched.push_back(std::make_shared<UpdateBS>(layer.get(), *this));
        }
    }
    catch (...)
    {
        THROW_NONAME("WorkflowEager", "Cannot create auxiliary pipelines");
    }
}

void WorkflowEager::execTargetCreateAuxPipelines(const std::unordered_map<Name, size_t>& uniqueTensors)
{
    for (const auto& uniqueTensor : uniqueTensors)
    {
        const WorkflowDB::TensorUsage& usage = mWorkflowDB->getUsage(uniqueTensor.second);

        if (usage.shape.isBSDependent())
        {
            mPipelineCreateBatched.push_back(newActionCreateTensor(usage.tensorName, usage.layerExecutionTarget, usage.shape));
            mPipelineDeleteBatched.push_back(newActionDeleteTensor(usage.tensorName, usage.layerExecutionTarget));
        }
        else
        {
            mPipelineCreateNotBatched.push_back(newActionCreateTensor(usage.tensorName, usage.layerExecutionTarget, usage.shape));
        }

        if (usage.isZero)
        {
            mPipelineZeroTensors.push_back(newActionZero(usage.tensorName, usage.layerExecutionTarget));
        }
    }
}

void WorkflowEager::forwardPassTesting()
{
    for (auto& layer : mLayers)
    {
        layer->forwardCompute(NetworkMode::Test);
    }
}

void WorkflowEager::forwardPassTraining(bool)
{
    try
    {
        executePipeline(Workflow::Pipelines::Zero);
    }
    catch (...)
    {
        THROW_NONAME("WorkflowEager", "Cannot execute zeroing pipeline");
    }

    try
    {
        for (auto& layer : mLayers)
        {
            layer->forwardCompute(NetworkMode::Train);
        }
    }
    catch (...)
    {
        THROW_NONAME("WorkflowEager", "Cannot execute forward in training mode");
    }
}

void WorkflowEager::backwardPassTraining()
{
    try
    {
        for (auto it = mLayers.rbegin(); it != mLayers.rend(); ++it)
        {
            (*it)->backwardCompute();
        }
    }
    catch (...)
    {
        THROW_NONAME("WorkflowEager", "Cannot execute backward");
    }
}

} // namespace raul