// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "Workflow.h"

#include <training/base/layers/BasicLayer.h>

#include <training/base/tools/Utils.h>

#include <training/compiler/Compiler.h>
#include <training/compiler/WorkflowActions.h>

#include "WorkflowDB.h"

namespace
{
const raul::Name suffixRecalced = "recalc";
const raul::Name suffixForward = "forwardPass";
const raul::Name suffixBack = "back";
} // anonymous

namespace raul
{

WShape::WShape()
{
    for (size_t q = 0; q < shape::dimensions_number; ++q)
    {
        mIsBS[q] = false;
        mMultiplier[q] = 1u;
    }
}

WShape::WShape(const shape& shapeVal)
    : mShape(shapeVal)
{
    for (size_t q = 0; q < shape::dimensions_number; ++q)
    {
        mIsBS[q] = false;
        mMultiplier[q] = 1u;
    }
}

bool WShape::isBSDependent() const
{
    bool ret = false;

    for (size_t q = 0; q < shape::dimensions_number; ++q)
    {
        ret |= mIsBS[q];
    }

    return ret;
}

std::string WShape::toString() const
{
    std::string s;
    s += "[";
    for (size_t i = 0; i < 4; ++i)
    {
        if (mIsBS[i])
        {
            s += "BATCH";
        }
        else
        {
            s += std::to_string(mShape[i]);
        }
        if (i < 3)
        {
            s += ", ";
        }
    }
    s += "]";
    return s;
}

shape WShape::getShape(const Workflow& work) const
{
    shape ret = mShape;

    for (size_t q = 0; q < shape::dimensions_number; ++q)
    {
        if (mIsBS[q])
        {
            size_t bs = work.getBatchSize();
            ret[q] = bs * mMultiplier[q];
        }
    }

    return ret;
}

bool WShape::operator==(const WShape& other) const
{
    bool ret = (mShape == other.mShape);

    for (size_t q = 0; q < shape::dimensions_number; ++q)
    {
        ret &= (mIsBS[q] == other.mIsBS[q]);
        if (mIsBS[q] && other.mIsBS[q])
        {
            ret &= (mMultiplier[q] == other.mMultiplier[q]);
        }
    }

    return ret;
}

Workflow::Workflow(CompressionMode compressionMode,
                   CalculationMode calculationMode,
                   AllocationMode allocationMode,
                   ExecutionTarget executionTarget,
                   bool useCompiler,
                   quantization::IQuantizer* quantizer)
    : mNetworkParameters(mMemoryManager, mMemoryManagerFP16, *this, 0, compressionMode, calculationMode, quantizer)
    , mCompiler(std::make_unique<Compiler>(executionTarget))
    , mAllocationMode(allocationMode)
    , mExecutionTarget(executionTarget)
    , mOverridedLayerExecutionTarget(LayerExecutionTarget::Default)
    , mWorkflowDB(std::make_shared<WorkflowDB>())
    , mUseCompiler(useCompiler)
    , mCompilationStarted(false)
    , mIsPipelinesPrepared(false)
    , mBatchSize(0)
    , mIsBatchSizeInited(false)
    , mIsMemoryPrepared(false)
    , mIsForwardCalled(false)
{
    try
    {
        if (mAllocationMode == AllocationMode::POOL)
        {
            mWorkflowPoolTest = std::make_shared<WorkflowPool<MemoryManager>>();
            mWorkflowPoolTrain = std::make_shared<WorkflowPool<MemoryManager>>();

            mWorkflowPoolTestFP16 = std::make_shared<WorkflowPool<MemoryManagerFP16>>();
            mWorkflowPoolTrainFP16 = std::make_shared<WorkflowPool<MemoryManagerFP16>>();
        }

        if (mExecutionTarget == ExecutionTarget::CPU || mExecutionTarget == ExecutionTarget::CPUFP16)
        {
            if (mAllocationMode == AllocationMode::POOL && mNetworkParameters.mCompressionMode != CompressionMode::NONE)
            {
                if (mAllocationMode == AllocationMode::POOL && mNetworkParameters.mCompressionMode != CompressionMode::NONE)
                {
                    THROW_NONAME("Workflow", "allocation mode POOL not possible for compressions");
                }
            }
        }
        else
        {
            THROW_NONAME("Workflow", "unsupported execution target");
        }
    }
    catch (...)
    {
        THROW_NONAME("Workflow", "cannot create workflow");
    }
}

Workflow::~Workflow() = default;

void Workflow::addLayer(LayerMem layer)
{
    try
    {
        if (mIsPipelinesPrepared)
        {
            THROW_NONAME("Workflow", "pipelines prepared, no addition possible");
        }

        const std::string layerName = layer->getName();

        if (layerName.empty())
        {
            THROW_NONAME("Workflow", "empty layer name");
        }

        if (mLayersDict.find(layerName) != mLayersDict.end())
        {
            THROW_NONAME("Workflow", "layer with the same name [" + layerName + "] already exists");
        }

        if (!checkOutputsNeeded(layer.get()))
        {
            THROW_NONAME("Workflow", "layer [" + layerName + "] does not declare outputs for forward pass");
        }

        mLayersDict.insert({ layerName, layer.get() });
        mLayers.emplace_back(std::move(layer));
    }
    catch (...)
    {
        THROW_NONAME("Workflow", "Cannot add layer");
    }
}

void Workflow::tensorNeeded(const Name& layerName,
                            const Name& tensorName,
                            WShape shape,
                            Workflow::Usage usage,
                            Workflow::Mode mode,
                            bool isOptimizeGraph,
                            bool isOptimizeMem,
                            bool isTrainable,
                            bool isZero,
                            bool isCompress,
                            LayerExecutionTarget layerExecutionTarget)
{
    try
    {

        if (mOverridedLayerExecutionTarget != LayerExecutionTarget::Default)
        {
            layerExecutionTarget = mOverridedLayerExecutionTarget;
        }

        if (mIsPipelinesPrepared)
        {
            THROW_NONAME("Workflow", "pipelines prepared, no declaration possible");
        }

        WorkflowDB::TensorUsage tensorUsage({ layerName, tensorName, shape, usage, mode, isOptimizeGraph, isOptimizeMem, isTrainable, isZero, isCompress, layerExecutionTarget });

        if (mWorkflowDB->isTensorExistsInTable(tensorName, layerName, usage))
        {
            THROW_NONAME("Workflow", "tensor [" + tensorUsage.tensorName + "] has been already declared for layer [" + tensorUsage.layerName + "] with same usage");
        }

        if (isOptimizeMem && isTrainable)
        {
            THROW_NONAME("Workflow", "tensor [" + tensorUsage.tensorName + "] has been declared as trainable and memory optimizable at the same time");
        }

        if (mode == Mode::Read && isZero)
        {
            THROW_NONAME("Workflow", "tensor [" + tensorUsage.tensorName + "] has been declared as read only and zeroed at the same time");
        }

        if (!isOptimizeMem && isCompress)
        {
            THROW_NONAME("Workflow", "tensor [" + tensorUsage.tensorName + "] has been declared as persistent and compressed at the same time");
        }

        if (layerExecutionTarget != LayerExecutionTarget::Default)
        {
            if (mNetworkParameters.mCompressionMode != CompressionMode::NONE)
            {
                THROW_NONAME("Workflow", "compressions not possible for customized layer execution target");
            }
        }

        mWorkflowDB->addUsage(tensorUsage);
    }
    catch (...)
    {
        THROW_NONAME("Workflow", "Cannot declare tensor");
    }
}

void Workflow::tensorNeededMaxShape(const Name& layerName,
                                    const Name& tensorName,
                                    WShape shape,
                                    Workflow::Usage usage,
                                    Workflow::Mode mode,
                                    bool isOptimizeGraph,
                                    bool isOptimizeMem,
                                    bool isTrainable,
                                    bool isZero,
                                    bool isCompress,
                                    LayerExecutionTarget layerExecutionTarget)
{
    try
    {
        if (mOverridedLayerExecutionTarget != LayerExecutionTarget::Default)
        {
            layerExecutionTarget = mOverridedLayerExecutionTarget;
        }

        if (mIsPipelinesPrepared)
        {
            THROW_NONAME("Workflow", "pipelines prepared, no declaration possible");
        }

        mWorkflowDB->chooseMaxShape(tensorName, shape); // shape might be adjusted

        WorkflowDB::TensorUsage tensorUsage({ layerName, tensorName, shape, usage, mode, isOptimizeGraph, isOptimizeMem, isTrainable, isZero, isCompress, layerExecutionTarget });

        if (isOptimizeMem && isTrainable)
        {
            THROW_NONAME("Workflow", "tensor [" + tensorUsage.tensorName + "] has been declared as trainable and memory optimizable at the same time");
        }

        if (mode == Mode::Read && isZero)
        {
            THROW_NONAME("Workflow", "tensor [" + tensorUsage.tensorName + "] has been declared as read only and zeroed at the same time");
        }

        if (!isOptimizeMem && isCompress)
        {
            THROW_NONAME("Workflow", "tensor [" + tensorUsage.tensorName + "] has been declared as persistent and compressed at the same time");
        }

        if (layerExecutionTarget != LayerExecutionTarget::Default)
        {
            // d.polubotko: assume user knows what to do so skip this check
            // if (mNetworkParameters.mCompressionMode != CompressionMode::NONE)
            {
                // THROW_NONAME("Workflow", "compressions not possible for customized layer execution target");
            }
        }

        mWorkflowDB->addUsage(tensorUsage);
    }
    catch (...)
    {
        THROW_NONAME("Workflow", "Cannot declare tensor");
    }
}

void Workflow::copyDeclaration(const Name& layerName,
                               const Name& tensorName,
                               Workflow::Usage usage,
                               Workflow::Mode mode,
                               bool isOptimizeGraph,
                               bool isOptimizeMem,
                               bool isTrainable,
                               bool isZero,
                               bool isCompress,
                               LayerExecutionTarget layerExecutionTarget)
{
    try
    {
        if (mIsPipelinesPrepared)
        {
            THROW_NONAME("Workflow", "pipelines prepared, no declaration possible");
        }

        if (mWorkflowDB->isTensorExistsInTable(tensorName, layerName, usage))
        {
            THROW_NONAME("Workflow", "tensor [" + tensorName + "] has been already declared for layer [" + layerName + "] with same usage");
        }

        if (mOverridedLayerExecutionTarget != LayerExecutionTarget::Default)
        {
            layerExecutionTarget = mOverridedLayerExecutionTarget;
        }

        if (layerExecutionTarget != LayerExecutionTarget::Default)
        {
            if (mNetworkParameters.mCompressionMode != CompressionMode::NONE)
            {
                THROW_NONAME("Workflow", "compressions not possible for customized layer execution target");
            }
        }

        WorkflowDB::TensorUsage usg = mWorkflowDB->findFirstTensor(tensorName);

        usg.layerName = layerName;
        usg.usage = usage;
        usg.mode = mode;
        usg.isOptimizeGraph = isOptimizeGraph;
        usg.isOptimizeMem = isOptimizeMem;
        usg.isTrainable = isTrainable;
        usg.isZero = isZero;
        usg.isCompress = isCompress;
        usg.layerExecutionTarget = layerExecutionTarget;

        if (mode == Mode::Read && isZero)
        {
            THROW_NONAME("Workflow", "tensor [" + tensorName + "] has been declared as read only and zeroed at the same time");
        }

        mWorkflowDB->addUsage(usg);
    }
    catch (...)
    {
        THROW_NONAME("Workflow", "Cannot copy declaration");
    }
}

void Workflow::copyDeclaration(const Name& layerName, const Name& tensorName, Workflow::Usage usage, Workflow::Mode mode)
{
    try
    {
        if (mIsPipelinesPrepared)
        {
            THROW_NONAME("Workflow", "pipelines prepared, no declaration possible");
        }

        if (mWorkflowDB->isTensorExistsInTable(tensorName, layerName, usage))
        {
            THROW_NONAME("Workflow", "tensor [" + tensorName + "] has been already declared for layer [" + layerName + "] with same usage");
        }

        WorkflowDB::TensorUsage usg = mWorkflowDB->findFirstTensor(tensorName);

        usg.layerName = layerName;
        usg.usage = usage;
        usg.mode = mode;

        if (mOverridedLayerExecutionTarget != LayerExecutionTarget::Default)
        {
            usg.layerExecutionTarget = mOverridedLayerExecutionTarget;
        }

        if (mode == Mode::Read && usg.isZero)
        {
            THROW_NONAME("Workflow", "tensor [" + tensorName + "] has been declared as read only and zeroed at the same time");
        }

        mWorkflowDB->addUsage(usg);
    }
    catch (...)
    {
        THROW_NONAME("Workflow", "Cannot copy declaration");
    }
}

void Workflow::copyDeclaration(const Name& layerName,
                               const Name& fromTensorName,
                               const Name& toTensorName,
                               Workflow::Usage usage,
                               Workflow::Mode mode,
                               bool isOptimizeGraph,
                               bool isOptimizeMem,
                               bool isTrainable,
                               bool isZero,
                               bool isCompress,
                               LayerExecutionTarget layerExecutionTarget)
{
    try
    {
        if (mOverridedLayerExecutionTarget != LayerExecutionTarget::Default)
        {
            layerExecutionTarget = mOverridedLayerExecutionTarget;
        }

        if (mIsPipelinesPrepared)
        {
            THROW_NONAME("Workflow", "pipelines prepared, no declaration possible");
        }

        if (mWorkflowDB->isTensorExistsInTable(toTensorName, layerName, usage))
        {
            THROW_NONAME("Workflow", "tensor [" + toTensorName + "] has been already declared for layer [" + layerName + "] with same usage");
        }

        if (layerExecutionTarget != LayerExecutionTarget::Default)
        {
            if (mNetworkParameters.mCompressionMode != CompressionMode::NONE)
            {
                THROW_NONAME("Workflow", "compressions not possible for customized layer execution target");
            }
        }

        WorkflowDB::TensorUsage usg = mWorkflowDB->findFirstTensor(fromTensorName);

        usg.layerName = layerName;
        usg.tensorName = toTensorName;
        usg.usage = usage;
        usg.mode = mode;
        usg.isOptimizeGraph = isOptimizeGraph;
        usg.isOptimizeMem = isOptimizeMem;
        usg.isTrainable = isTrainable;
        usg.isZero = isZero;
        usg.isCompress = isCompress;
        usg.layerExecutionTarget = layerExecutionTarget;

        if (mode == Mode::Read && isZero)
        {
            THROW_NONAME("Workflow", "tensor [" + toTensorName + "] has been declared as read only and zeroed at the same time");
        }

        mWorkflowDB->addUsage(usg);
    }
    catch (...)
    {
        THROW_NONAME("Workflow", "Cannot copy declaration");
    }
}

void Workflow::copyDec(const Name& layerName,
                       const Name& fromTensorName,
                       const Name& toTensorName,
                       Usage usage,
                       Mode mode,
                       bool isOptimizeGraph,
                       bool isOptimizeMem,
                       bool isTrainable,
                       bool isZero,
                       bool isCompress)
{
    if (mIsPipelinesPrepared)
    {
        THROW_NONAME("Workflow", "pipelines prepared, no declaration possible");
    }

    if (mWorkflowDB->isTensorExistsInTable(toTensorName, layerName, usage))
    {
        THROW_NONAME("Workflow", "tensor [" + toTensorName + "] has been already declared for layer [" + layerName + "] with same usage");
    }

    if (isTensorDeclared(toTensorName))
    {
        copyDeclaration(layerName, toTensorName, usage, mode);
    }
    else
    {
        copyDeclaration(layerName, fromTensorName, toTensorName, usage, mode, isOptimizeGraph, isOptimizeMem, isTrainable, isZero, isCompress);
    }
}

void Workflow::createImplementations()
{
    if (mCompiler->isResolved())
    {
        THROW_NONAME("Workflow", "implementations already resolved");
    }

    mCompilationStarted = true;

    auto fronts = mCompiler->resolveImplementation(mBuilders, mNetworkParameters);
    for (auto& front : fronts)
    {
        addLayer(std::move(front));
    }

    mBuilders.clear();
    mBuilders.shrink_to_fit();
}

void Workflow::preparePipelines(Execution execution)
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
            THROW_NONAME("Workflow", "forward called without leading backward");
        }

        if (mOverridedLayerExecutionTarget != LayerExecutionTarget::Default)
        {
            THROW_NONAME("Workflow", "override should be reseted before pipelines preparation");
        }

        if (execution == Execution::Checkpointed)
        {
            if (mExecutionTarget == ExecutionTarget::CPUFP16)
            {
                THROW_NONAME("Workflow", "checkpointed pipeline not supported for CPUFP16 targets");
            }
        }

        fillExternalInputs();

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
                        THROW_NONAME("Workflow", "the workflow is not correct, there are same outputs defined: " + output);
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
                    THROW_NONAME("Workflow", "the workflow is not correct, there are same inputs defined for layer " + layer->getName());
                }

                if (!isUniqueNames(layer->getSharedWeights()))
                {
                    THROW_NONAME("Workflow", "the workflow is not correct, there are same weights defined for layer " + layer->getName());
                }
            }
        }

        if (!isGraphCorrected())
        {
            const auto separator = ", ";
            std::string externalInputs;

            auto begin = mExternalInputs.cbegin();
            const auto end = mExternalInputs.cend();

            if (begin != end)
            {
                externalInputs = *begin;
                ++begin;
            }

            for (; begin != end; ++begin)
            {
                externalInputs += separator;
                externalInputs += *begin;
            }

            THROW_NONAME("Workflow", "the workflow is not correct, there are external inputs: " + externalInputs);
        }

        clearPipelines();

        // d.polubotko: order matter (createAuxPipelines adjust isZero, isCompress flags)
        createAuxPipelines();
        createForwardTestPipeline();

        if (execution == Execution::Normal)
        {
            createTrainPipeline();
        }

        if (execution == Execution::Checkpointed)
        {
            createTrainCheckpointedPipeline();
        }

        mIsPipelinesPrepared = true;
    }
    catch (...)
    {
        THROW_NONAME("Workflow", "Cannot prepare pipelines");
    }
}

void Workflow::fillExternalInputs()
{
    std::unordered_map<std::string, size_t> edgeParent;

    for (size_t q = 0; q < mLayers.size(); ++q)
    {
        for (const raul::Name& output : mLayers[q]->getOutputs())
        {
            if (output.empty())
            {
                THROW_NONAME("Workflow", "output tensor name is empty");
            }
        }

        for (const raul::Name& input : mLayers[q]->getInputs())
        {
            if (input.empty())
            {
                THROW_NONAME("Workflow", "input tensor name is empty");
            }
            auto it = edgeParent.find(input);

            if (it != edgeParent.end())
            {
            }
            else
            {
                mExternalInputs.insert(input);
            }
        }

        for (const raul::Name& output : mLayers[q]->getOutputs())
        {
            edgeParent[output] = q;
        }
    }
}

bool Workflow::checkOutputsNeeded(const BasicLayer* layer) const
{
    bool ret = true;

    const Names& outputs = layer->getOutputs();

    for (const auto& output : outputs)
    {
        auto tensorUsage = mWorkflowDB->getCell(mWorkflowDB->getLayersTable(), layer->getName(), output);

        if (mWorkflowDB->isCellElementEmpty(tensorUsage, Usage::Forward))
        {
            ret = false;
            break;
        }
    }

    return ret;
}

void Workflow::createAuxPipelines()
{
    try
    {
        // Tensor -> index in mTensorNeeded
        std::unordered_map<Name, size_t> uniqueTensors;

        // check inequality, fill uniqueTensors
        for (auto& mLayer : mLayers)
        {
            const Name& lName = mLayer->getName();
            std::vector<Name> tensors = mWorkflowDB->getSlice(mWorkflowDB->getLayersTable(), lName);
            for (const auto& tName : tensors)
            {
                auto uniqueIt = uniqueTensors.find(tName);
                if (uniqueIt != uniqueTensors.end())
                {
                    auto tensorUsage = mWorkflowDB->getCell(mWorkflowDB->getLayersTable(), lName, tName);

                    if (!mWorkflowDB->isCellElementEmpty(tensorUsage, Usage::Forward))
                    {
                        auto& tUsg = mWorkflowDB->getUsage(tensorUsage[static_cast<size_t>(Usage::Forward)]);

                        if (mWorkflowDB->getUsage((*uniqueIt).second).isZero || tUsg.isZero)
                        {
                            mWorkflowDB->getUsage((*uniqueIt).second).isZero = true;
                            tUsg.isZero = true;
                        }

                        if (mWorkflowDB->getUsage((*uniqueIt).second).isCompress || tUsg.isCompress)
                        {
                            mWorkflowDB->getUsage((*uniqueIt).second).isCompress = true;
                            tUsg.isCompress = true;
                        }

                        checkAttributesInequality((*uniqueIt).second, tensorUsage[static_cast<size_t>(Usage::Forward)], tName);
                    }

                    if (!mWorkflowDB->isCellElementEmpty(tensorUsage, Usage::Backward))
                    {
                        auto& tUsg = mWorkflowDB->getUsage(tensorUsage[static_cast<size_t>(Usage::Backward)]);

                        if (mWorkflowDB->getUsage((*uniqueIt).second).isZero || tUsg.isZero)
                        {
                            mWorkflowDB->getUsage((*uniqueIt).second).isZero = true;
                            tUsg.isZero = true;
                        }

                        if (mWorkflowDB->getUsage((*uniqueIt).second).isCompress || tUsg.isCompress)
                        {
                            mWorkflowDB->getUsage((*uniqueIt).second).isCompress = true;
                            tUsg.isCompress = true;
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
                        auto& tUsgF = mWorkflowDB->getUsage(tensorUsage[static_cast<size_t>(Usage::Forward)]);
                        auto& tUsgB = mWorkflowDB->getUsage(tensorUsage[static_cast<size_t>(Usage::Backward)]);

                        if (tUsgF.isZero || tUsgB.isZero)
                        {
                            tUsgF.isZero = true;
                            tUsgB.isZero = true;
                        }

                        if (tUsgF.isCompress || tUsgB.isCompress)
                        {
                            tUsgF.isCompress = true;
                            tUsgB.isCompress = true;
                        }

                        checkAttributesInequality(tensorUsage[static_cast<size_t>(Usage::Forward)], tensorUsage[static_cast<size_t>(Usage::Backward)], tName);
                    }

                    uniqueTensors.insert({ tName, tensorUsage[static_cast<size_t>(tUsage)] });
                }
            }
        }

        // create pipelines
        execTargetCreateAuxPipelines(uniqueTensors);

        for (const auto& layer : mLayers)
        {
            mPipelineCreateNotBatched.push_back(std::make_shared<InitNonBS>(layer.get()));
            mPipelineCreateBatched.push_back(std::make_shared<UpdateBS>(layer.get(), *this));
        }
    }
    catch (...)
    {
        THROW_NONAME("Workflow", "Cannot create auxiliary pipelines");
    }
}

void Workflow::checkAttributesInequality(size_t indexA, size_t indexB, const Name& name) const
{
    const WorkflowDB::TensorUsage& usageA = mWorkflowDB->getUsage(indexA);
    const WorkflowDB::TensorUsage& usageB = mWorkflowDB->getUsage(indexB);

    if (usageA.isOptimizeGraph != usageB.isOptimizeGraph)
    {
        THROW_NONAME("Workflow",
                     "attribute isOptimizeGraph inequality for tensor " + name + ". " + Conversions::toString(usageA.isOptimizeGraph) + " from layer '" + usageA.layerName + "' vs " +
                         Conversions::toString(usageB.isOptimizeGraph) + " from layer '" + usageB.layerName + "'");
    }
    if (usageA.isOptimizeMem != usageB.isOptimizeMem)
    {
        THROW_NONAME("Workflow",
                     "attribute isOptimizeMem inequality for tensor " + name + ". " + Conversions::toString(usageA.isOptimizeMem) + " from layer '" + usageA.layerName + "' vs " +
                         Conversions::toString(usageB.isOptimizeMem) + " from layer '" + usageB.layerName + "'");
    }
    if (usageA.isTrainable != usageB.isTrainable)
    {
        THROW_NONAME("Workflow",
                     "attribute isTrainable inequality for tensor " + name + ". " + Conversions::toString(usageA.isTrainable) + " from layer '" + usageA.layerName + "' vs " +
                         Conversions::toString(usageB.isTrainable) + " from layer '" + usageB.layerName + "'");
    }
    if (usageA.isZero != usageB.isZero)
    {
        THROW_NONAME("Workflow",
                     "attribute isZero inequality for tensor " + name + ". " + Conversions::toString(usageA.isZero) + " from layer '" + usageA.layerName + "' vs " +
                         Conversions::toString(usageB.isZero) + " from layer '" + usageB.layerName + "'");
    }
    if (usageA.isCompress != usageB.isCompress)
    {
        THROW_NONAME("Workflow",
                     "attribute isCompress inequality for tensor " + name + ". " + Conversions::toString(usageA.isCompress) + " from layer '" + usageA.layerName + "' vs " +
                         Conversions::toString(usageB.isCompress) + " from layer '" + usageB.layerName + "'");
    }
    if (usageA.layerExecutionTarget != usageB.layerExecutionTarget)
    {
        THROW_NONAME("Workflow",
                     "attribute layerExecutionTarget inequality for tensor " + name + ". " + Conversions::toString(usageA.layerExecutionTarget) + " from layer '" + usageA.layerName + "' vs " +
                         Conversions::toString(usageB.layerExecutionTarget) + " from layer '" + usageB.layerName + "'");
    }
    if (usageA.shape != usageB.shape)
    {
        THROW_NONAME("Workflow",
                     "attribute shape inequality for tensor " + name + ". " + usageA.shape.toString() + " from layer '" + usageA.layerName + "' vs " + usageB.shape.toString() + " from layer '" +
                         usageB.layerName + "'");
    }
}

void Workflow::createForwardTestPipeline()
{
    try
    {

        Timeline timelineTensors = getTimeline(getLayerNames(), Usage::Forward);

        const std::pair<Appearance, Appearance> appearance = timelineToAppearance(timelineTensors);
        const Appearance& timelineLayersFirst = appearance.first;
        const Appearance& timelineLayersLast = appearance.second;

        // create pipelines

        if (mAllocationMode == AllocationMode::POOL)
        {
            std::tuple<Timeline, Timeline> timelineJoint = appearanceToTimeline(appearance, {});

            mWorkflowPoolTest->createIntervals(getLayerNames(), std::get<0>(timelineJoint));
            mWorkflowPoolTestFP16->createIntervals(getLayerNames(), std::get<1>(timelineJoint));
        }

        execTargetCreateForwardTestPipeline(timelineLayersFirst, timelineLayersLast);
    }
    catch (...)
    {
        THROW_NONAME("Workflow", "Cannot create forward testing pipeline");
    }
}

void Workflow::createTrainPipeline()
{
    try
    {

        Names layers = getLayerNames();

        // Tensor name -> [first, last] layer
        Timeline timelineTensorsForward = getTimeline(layers, Usage::Forward);
        Timeline timelineTensorsBackward = getTimeline(layers, Usage::Backward);

        if (mNetworkParameters.mCompressionMode == CompressionMode::NONE)
        {
            // remove duplication of potential allocations
            for (auto& timeForward : timelineTensorsForward)
            {
                const Name& tName = timeForward.first;

                auto it = timelineTensorsBackward.find(tName);
                if (it != timelineTensorsBackward.end())
                {
                    (*it).second.second = "";
                }
            }

            // remove duplication of potential deallocations
            for (auto& timeBackward : timelineTensorsBackward)
            {
                const Name& tName = timeBackward.first;

                auto it = timelineTensorsForward.find(tName);
                if (it != timelineTensorsForward.end())
                {
                    (*it).second.second = "";
                }
            }
        }

        std::pair<Appearance, Appearance> appearanceForward = timelineToAppearance(timelineTensorsForward);
        const Appearance& timelineLayersFirstForward = appearanceForward.first;
        Appearance& timelineLayersLastForward = appearanceForward.second;

        std::pair<Appearance, Appearance> appearanceBackward = timelineToAppearance(timelineTensorsBackward);
        const Appearance& timelineLayersFirstBackward = appearanceBackward.first;
        Appearance& timelineLayersLastBackward = appearanceBackward.second;

        Appearance forwardCompress;
        Appearance backwardDecompress;

        if (mNetworkParameters.mCompressionMode != CompressionMode::NONE)
        {
            // remove duplication of potential allocations
            for (const auto& appearForward : timelineLayersFirstForward)
            {
                for (const auto& tName : appearForward.second)
                {
                    for (auto& appearBackward : timelineLayersLastBackward)
                    {
                        const Name& lName = appearBackward.first;

                        auto lastBackIt = std::find(appearBackward.second.begin(), appearBackward.second.end(), tName);
                        if (lastBackIt != appearBackward.second.end())
                        {
                            appearBackward.second.erase(lastBackIt);

                            // decompression case
                            auto decompressLayer = backwardDecompress.find(lName);
                            if (decompressLayer != backwardDecompress.end())
                            {
                                (*decompressLayer).second.push_back(tName);
                            }
                            else
                            {
                                backwardDecompress.insert({ lName, { tName } });
                            }
                        }
                    }
                }
            }

            // remove duplication of potential deallocations
            for (const auto& appearBackward : timelineLayersFirstBackward)
            {
                for (const auto& tName : appearBackward.second)
                {
                    for (auto& appearForward : timelineLayersLastForward)
                    {
                        const Name& lName = appearForward.first;

                        auto lastForwardIt = std::find(appearForward.second.begin(), appearForward.second.end(), tName);
                        if (lastForwardIt != appearForward.second.end())
                        {
                            appearForward.second.erase(lastForwardIt);

                            // compression case
                            auto compressLayer = forwardCompress.find(lName);
                            if (compressLayer != forwardCompress.end())
                            {
                                (*compressLayer).second.push_back(tName);
                            }
                            else
                            {
                                forwardCompress.insert({ lName, { tName } });
                            }
                        }
                    }
                }
            }
        }

        if (mAllocationMode == AllocationMode::POOL)
        {
            std::tuple<Timeline, Timeline> timelineJoint = appearanceToTimeline(appearanceForward, appearanceBackward);

            Names lNames = getLayerNames();
            Names lNamesOrig = lNames;

            for (auto it = lNamesOrig.rbegin(); it != lNamesOrig.rend(); ++it)
            {
                lNames.push_back((*it) + suffixBack);
            }

            mWorkflowPoolTrain->createIntervals(lNames, std::get<0>(timelineJoint));
            mWorkflowPoolTrainFP16->createIntervals(lNames, std::get<1>(timelineJoint));
        }

        // create pipelines
        // forward
        for (auto& layer : mLayers)
        {
            fillTrainPipeline(mPipelineForwardTrain, layer.get(), timelineLayersFirstForward, timelineLayersLastForward, Usage::Forward);

            // compress
            if (mNetworkParameters.mCompressionMode != CompressionMode::NONE)
            {
                fillTrainPipelineCompression(mPipelineForwardTrain, layer.get(), forwardCompress, Usage::Forward);
            }
        }

        // backward
        for (auto it = mLayers.rbegin(); it != mLayers.rend(); ++it)
        {
            // decompress
            if (mNetworkParameters.mCompressionMode != CompressionMode::NONE)
            {
                fillTrainPipelineCompression(mPipelineBackwardTrain, (*it).get(), backwardDecompress, Usage::Backward);
            }

            fillTrainPipeline(mPipelineBackwardTrain, (*it).get(), timelineLayersLastBackward, timelineLayersFirstBackward, Usage::Backward);
        }
    }
    catch (...)
    {
        THROW_NONAME("Workflow", "Cannot create training pipelines");
    }
}

Names Workflow::getLayerNames() const
{
    Names ret(mLayers.size());

    for (size_t q = 0; q < mLayers.size(); ++q)
    {
        ret[q] = mLayers[q]->getName();
    }

    return ret;
}

Workflow::Timeline Workflow::getTimeline(const Names& layers, Usage usage) const
{
    Timeline ret;

    if (usage != Usage::Forward && usage != Usage::Backward)
    {
        THROW_NONAME("Workflow", "incorrect parameter");
    }

    for (const auto& lName : layers)
    {
        std::vector<Name> tensors = mWorkflowDB->getSlice(mWorkflowDB->getLayersTable(), lName);
        for (const auto& tName : tensors)
        {
            auto tensorUsage = mWorkflowDB->getCell(mWorkflowDB->getLayersTable(), lName, tName);
            if (!mWorkflowDB->isCellElementEmpty(tensorUsage, usage))
            {
                WorkflowDB::TensorUsage usg = mWorkflowDB->getUsage(tensorUsage[static_cast<size_t>(usage)]);
                if (usg.isOptimizeMem)
                {
                    auto timelinedI = ret.find(tName);
                    if (timelinedI != ret.end())
                    {
                        (*timelinedI).second.second = lName;
                    }
                    else
                    {
                        ret.insert({ tName, { lName, lName } });
                    }
                }
            }
        }
    }

    return ret;
}

std::pair<Workflow::Appearance, Workflow::Appearance> Workflow::timelineToAppearance(const Timeline& timelines) const
{
    std::pair<Workflow::Appearance, Workflow::Appearance> ret;

    for (auto& timelineT : timelines)
    {
        const Name& tName = timelineT.first;
        const Name& lNameF = timelineT.second.first;
        const Name& lNameL = timelineT.second.second;

        if (!lNameF.empty())
        {
            auto lItF = ret.first.find(lNameF);
            if (lItF != ret.first.end())
            {
                (*lItF).second.push_back(tName);
            }
            else
            {
                ret.first.insert({ lNameF, { tName } });
            }
        }

        if (!lNameL.empty())
        {
            auto lItL = ret.second.find(lNameL);
            if (lItL != ret.second.end())
            {
                (*lItL).second.push_back(tName);
            }
            else
            {
                ret.second.insert({ lNameL, { tName } });
            }
        }
    }

    return ret;
}

std::tuple<Workflow::Timeline, Workflow::Timeline>
Workflow::appearanceToTimeline(const std::pair<Appearance, Appearance>& forward, const std::pair<Appearance, Appearance>& backward) const
{
    std::tuple<Timeline, Timeline> ret; // d.polubotko: CPUFP32, CPUFP16

    const Appearance& forwardFirst = forward.first;
    const Appearance& forwardLast = forward.second;

    const Appearance& backwardFirst = backward.first;
    const Appearance& backwardLast = backward.second;

    for (const auto& appear : forwardFirst)
    {
        const Name& lName = appear.first;
        const Names& tNames = appear.second;

        for (const auto& tName : tNames)
        {
            std::vector<size_t> tensorUsage = mWorkflowDB->getCell(mWorkflowDB->getLayersTable(), lName, tName);
            WorkflowDB::TensorUsage usg = mWorkflowDB->getUsage(tensorUsage[static_cast<size_t>(Usage::Forward)]);

            ExecutionTarget target = mExecutionTarget;

            if (usg.layerExecutionTarget != LayerExecutionTarget::Default)
            {
                target = static_cast<ExecutionTarget>(usg.layerExecutionTarget);
            }

            Timeline* dst = &std::get<0>(ret);

            if (target == ExecutionTarget::CPU)
            {
            }
            else if (target == ExecutionTarget::CPUFP16)
            {
                dst = &std::get<1>(ret);
            }
            else
            {
                THROW_NONAME("Workflow", "unsopported execution target");
            }

            auto it = dst->find(tName);
            if (it != dst->end())
            {
                THROW_NONAME("Workflow", "tensor already added for forward");
            }
            else
            {
                dst->insert({ tName, { lName, "" } });
            }
        }
    }

    for (const auto& appear : forwardLast)
    {
        const Name& lName = appear.first;
        const Names& tNames = appear.second;

        for (const auto& tName : tNames)
        {
            std::vector<size_t> tensorUsage = mWorkflowDB->getCell(mWorkflowDB->getLayersTable(), lName, tName);
            WorkflowDB::TensorUsage usg = mWorkflowDB->getUsage(tensorUsage[static_cast<size_t>(Usage::Forward)]);

            ExecutionTarget target = mExecutionTarget;

            if (usg.layerExecutionTarget != LayerExecutionTarget::Default)
            {
                target = static_cast<ExecutionTarget>(usg.layerExecutionTarget);
            }

            Timeline* dst = &std::get<0>(ret);

            if (target == ExecutionTarget::CPU)
            {
            }
            else if (target == ExecutionTarget::CPUFP16)
            {
                dst = &std::get<1>(ret);
            }
            else
            {
                THROW_NONAME("Workflow", "unsopported execution target");
            }

            auto it = dst->find(tName);
            if (it != dst->end())
            {
                (*it).second.second = lName;
            }
            else
            {
                THROW_NONAME("Workflow", "tensor has not been added for forward");
            }
        }
    }

    for (const auto& appear : backwardLast)
    {
        const Name& lName = appear.first;
        const Names& tNames = appear.second;

        for (const auto& tName : tNames)
        {
            std::vector<size_t> tensorUsage = mWorkflowDB->getCell(mWorkflowDB->getLayersTable(), lName, tName);
            WorkflowDB::TensorUsage usg = mWorkflowDB->getUsage(tensorUsage[static_cast<size_t>(Usage::Backward)]);

            ExecutionTarget target = mExecutionTarget;

            if (usg.layerExecutionTarget != LayerExecutionTarget::Default)
            {
                target = static_cast<ExecutionTarget>(usg.layerExecutionTarget);
            }

            Timeline* dst = &std::get<0>(ret);

            if (target == ExecutionTarget::CPU)
            {
            }
            else if (target == ExecutionTarget::CPUFP16)
            {
                dst = &std::get<1>(ret);
            }
            else
            {
                THROW_NONAME("Workflow", "unsopported execution target");
            }

            auto it = dst->find(tName);
            if (it != dst->end())
            {
                THROW_NONAME("Workflow", "tensor already added for backward");
            }
            else
            {
                dst->insert({ tName, { lName + suffixBack, "" } });
            }
        }
    }

    for (const auto& appear : backwardFirst)
    {
        const Name& lName = appear.first;
        const Names& tNames = appear.second;

        for (const auto& tName : tNames)
        {
            std::vector<size_t> tensorUsage = mWorkflowDB->getCell(mWorkflowDB->getLayersTable(), lName, tName);
            WorkflowDB::TensorUsage usg = mWorkflowDB->getUsage(tensorUsage[static_cast<size_t>(Usage::Backward)]);

            ExecutionTarget target = mExecutionTarget;

            if (usg.layerExecutionTarget != LayerExecutionTarget::Default)
            {
                target = static_cast<ExecutionTarget>(usg.layerExecutionTarget);
            }

            Timeline* dst = &std::get<0>(ret);

            if (target == ExecutionTarget::CPU)
            {
            }
            else if (target == ExecutionTarget::CPUFP16)
            {
                dst = &std::get<1>(ret);
            }
            else
            {
                THROW_NONAME("Workflow", "unsopported execution target");
            }

            auto it = dst->find(tName);
            if (it != dst->end())
            {
                (*it).second.second = lName + suffixBack;
            }
            else
            {
                THROW_NONAME("Workflow", "tensor has not been added for backward");
            }
        }
    }

    return ret;
}

void Workflow::clearPipelines()
{
    mPipelineCreateBatched.clear();
    mPipelineCreateNotBatched.clear();
    mPipelineDeleteBatched.clear();
    mPipelineZeroTensors.clear();

    mPipelineForwardTest.clear();
    mPipelineForwardTrain.clear();
    mPipelineBackwardTrain.clear();
}

void Workflow::flush()
{
    // Clear memory manager
    mMemoryManager.clear();

    // Clear all pipelines
    clearPipelines();

    // Set all flags to initial values
    mIsPipelinesPrepared = false;
    mBatchSize = 0;
    mIsBatchSizeInited = false;
    mIsMemoryPrepared = false;
    mIsForwardCalled = false;
}

void Workflow::prepareMemoryForTraining()
{
    executePipeline(Workflow::Pipelines::CreateNotBatched);

    mIsMemoryPrepared = true;
}

template<typename MM>
std::vector<ParamAndGradImpl<typename MM::tensor>> Workflow::getTrainableParameters()
{
    typedef ParamAndGradImpl<typename MM::tensor> PAndG;

    std::vector<PAndG> res;

    const auto& names = getTrainableParameterNames();

    for (auto& name : names)
    {
        if (!getMemoryManager<MM>().tensorExists(name))
        {
            THROW_NONAME("Workflow", "tensor [" + name + "] does not exist");
        }

        if (!getMemoryManager<MM>().tensorExists(name.grad()))
        {
            THROW_NONAME("Workflow", "tensor [" + name.grad() + "] does not exist");
        }

        res.push_back(PAndG{ getMemoryManager<MM>().getTensor(name), getMemoryManager<MM>().getTensor(name.grad()) });
    }

    return res;
}

template<typename MM>
std::vector<ParamAndGradImpl<typename MM::tensor>> Workflow::getTrainableParametersSafe()
{
    typedef ParamAndGradImpl<typename MM::tensor> PAndG;

    std::vector<PAndG> res;

    const auto& names = getTrainableParameterNames();

    for (auto& name : names)
    {
        if (!getMemoryManager<MM>().tensorExists(name) || !getMemoryManager<MM>().tensorExists(name.grad()))
        {
            continue;
        }

        res.push_back(PAndG{ getMemoryManager<MM>().getTensor(name), getMemoryManager<MM>().getTensor(name.grad()) });
    }

    return res;
}

Names Workflow::getTrainableParameterNames() const
{
    Names ret;
    std::unordered_set<Name> trainableTensors;

    for (auto& layer : mLayers)
    {
        const auto layerTrainableParams = getLayerTrainableParameterNames(layer->getName());
        for (size_t i = 0; i < layerTrainableParams.size(); ++i)
        {
            if (trainableTensors.find(layerTrainableParams[i]) == trainableTensors.end())
            {
                trainableTensors.insert(layerTrainableParams[i]);
                ret.push_back(layerTrainableParams[i]);
            }
        }
    }

    return ret;
}

Names Workflow::getLayerTrainableParameterNames(const Name& layerName) const
{
    Names ret;

    auto slice = mWorkflowDB->getSlice(mWorkflowDB->getLayersTable(), layerName);
    std::set<Name> orderedNames(slice.begin(), slice.end());

    for (const auto& tName : orderedNames)
    {
        if (mWorkflowDB->findFirstTensor(tName).isTrainable)
        {
            ret.push_back(tName);
        }
    }

    return ret;
}

bool Workflow::isTensorTrainable(const Name& name) const
{
    return mWorkflowDB->findFirstTensor(name).isTrainable;
}

bool Workflow::isTensorOptimizeMem(const Name& name) const
{
    return mWorkflowDB->findFirstTensor(name).isOptimizeMem;
}

Names Workflow::getLayerParameterNames(const Name& layerName) const
{
    if (mLayersDict.find(layerName) == mLayersDict.end())
    {
        THROW_NONAME("Workflow", "Layer \"" + layerName + "\" not found");
    }
    const auto layer = mLayersDict.find(layerName)->second;
    if (layer->getTypeName() == "Data")
    {
        return Names();
    }

    auto slice = mWorkflowDB->getSlice(mWorkflowDB->getLayersTable(), layerName);
    std::set<Name> orderedNames(slice.begin(), slice.end());
    std::set<Name> trainableTensors;

    for (const auto& tName : orderedNames)
    {
        if (mWorkflowDB->findFirstTensor(tName).isTrainable || (!mWorkflowDB->findFirstTensor(tName).isOptimizeMem && !mWorkflowDB->findFirstTensor(tName).isZero))
        {
            // check if tensor was declared inside the layer
            if (Common::startsWith(tName, layerName))
            {
                trainableTensors.insert(tName);
            }
        }
    }

    Names ret(trainableTensors.begin(), trainableTensors.end());

    return ret;
}

bool Workflow::isTensorDeclared(const Name& tensorName) const
{
    return mWorkflowDB->isTensorDeclared(tensorName);
}

size_t Workflow::getDimension(const Name& tensorName, size_t dim) const
{
    WorkflowDB::TensorUsage usage = mWorkflowDB->findFirstTensor(tensorName);

    size_t ret = 0u;

    if (usage.shape.mIsBS[dim])
    {
        ret = getBatchSize() * usage.shape.mMultiplier[dim];
    }
    else
    {
        ret = usage.shape.mShape[dim];
    }

    return ret;
}

bool Workflow::isDimensionPlaceholded(const Name& tensorName, size_t dim) const
{
    WorkflowDB::TensorUsage usage = mWorkflowDB->findFirstTensor(tensorName);

    bool ret = false;

    if (usage.shape.mIsBS[dim])
    {
        ret = true;
    }

    return ret;
}

size_t Workflow::getBatch(const Name& tensorName) const
{
    return getDimension(tensorName, 0u);
}

size_t Workflow::getDepth(const Name& tensorName) const
{
    return getDimension(tensorName, 1u);
}

size_t Workflow::getHeight(const Name& tensorName) const
{
    return getDimension(tensorName, 2u);
}

size_t Workflow::getWidth(const Name& tensorName) const
{
    return getDimension(tensorName, 3u);
}

WShape Workflow::getShape(const Name& tensorName) const
{
    return mWorkflowDB->findFirstTensor(tensorName).shape;
}

bool Workflow::isBatchPlaceholded(const Name& tensorName) const
{
    return isDimensionPlaceholded(tensorName, 0u);
}

bool Workflow::isDepthPlaceholded(const Name& tensorName) const
{
    return isDimensionPlaceholded(tensorName, 1u);
}

bool Workflow::isHeightPlaceholded(const Name& tensorName) const
{
    return isDimensionPlaceholded(tensorName, 2u);
}

bool Workflow::isWidthPlaceholded(const Name& tensorName) const
{
    return isDimensionPlaceholded(tensorName, 3u);
}

size_t Workflow::getBatchSize() const
{
    if (!mIsBatchSizeInited)
    {
        THROW_NONAME("Workflow", "batch size not inited");
    }

    return mBatchSize;
}

void Workflow::setBatchSize(size_t batchSize)
{
    if (!mIsPipelinesPrepared)
    {
        THROW_NONAME("Workflow", "pipelines not prepared, batch size setup not possible");
    }

    if (batchSize < 1u)
    {
        THROW_NONAME("Workflow", "batch size is 0");
    }

    mBatchSize = batchSize;

    mIsBatchSizeInited = true; // order is important for Workflow::Pipelines::CreateBatched

    executePipeline(Workflow::Pipelines::DeleteBatched);
    executePipeline(Workflow::Pipelines::CreateBatched);

    mNetworkParameters.mLossReductionCoefficient = batchSize;

    if (mAllocationMode == AllocationMode::POOL)
    {
        mWorkflowPoolTest->clearPool();
        mWorkflowPoolTrain->clearPool();

        mWorkflowPoolTestFP16->clearPool();
        mWorkflowPoolTrainFP16->clearPool();
    }
}

void Workflow::forwardPassTesting()
{
    if (!mIsBatchSizeInited)
    {
        THROW_NONAME("Workflow", "tensors not allocated, set batch size");
    }

    if (!mIsMemoryPrepared)
    {
        THROW_NONAME("Workflow", "tensors not allocated, call prepareMemoryForTraining()");
    }

    if (mAllocationMode == AllocationMode::POOL)
    {
        mWorkflowPoolTest->createPool(mMemoryManager);
        mWorkflowPoolTestFP16->createPool(mMemoryManagerFP16);
    }

    try
    {
        executePipeline(Workflow::Pipelines::ForwardTest);
    }
    catch (...)
    {
        THROW_NONAME("Workflow", "Cannot execute forward testing pipeline");
    }
}

void Workflow::forwardPassTraining(bool performZero)
{
    if (!mIsBatchSizeInited)
    {
        THROW_NONAME("Workflow", "tensors not allocated, set batch size");
    }

    if (!mIsMemoryPrepared)
    {
        THROW_NONAME("Workflow", "tensors not allocated, call prepareMemoryForTraining()");
    }

    if (mIsForwardCalled)
    {
        THROW_NONAME("Workflow", "already executed, execute backward");
    }

    if (performZero)
    {
        executePipeline(Workflow::Pipelines::Zero); // fill non optimizable tensors (gradients for weights etc)
    }

    if (mAllocationMode == AllocationMode::POOL)
    {
        mWorkflowPoolTrain->createPool(mMemoryManager);
        mWorkflowPoolTrainFP16->createPool(mMemoryManagerFP16);
    }

    try
    {
        executePipeline(Workflow::Pipelines::ForwardTrain);
    }
    catch (...)
    {
        THROW_NONAME("Workflow", "Cannot execute forward training pipeline");
    }

    mIsForwardCalled = true;
}

void Workflow::backwardPassTraining()
{
    if (!mIsBatchSizeInited)
    {
        THROW_NONAME("Workflow", "tensors not allocated, set batch size");
    }

    if (!mIsMemoryPrepared)
    {
        THROW_NONAME("Workflow", "tensors not allocated, call prepareMemoryForTraining()");
    }

    if (!mIsForwardCalled)
    {
        THROW_NONAME("Workflow", "execute forward first");
    }

    try
    {
        executePipeline(Workflow::Pipelines::BackwardTrain);
    }
    catch (...)
    {
        THROW_NONAME("Workflow", "Cannot execute backward training pipeline");
    }

    mIsForwardCalled = false;
}

void Workflow::setCheckpoints(const Names& checkpoints)
{
    if (!isUniqueNames(checkpoints))
    {
        THROW_NONAME("Workflow", "there are same checkpoints defined");
    }

    for (const auto& tName : checkpoints)
    {
        std::pair<bool, size_t> found = findLayerByOutput(tName);

        if (!found.first)
        {
            THROW_NONAME("Workflow", "tensor " + tName + " not available for checkpointing");
        }
    }

    mCheckpoints = checkpoints;
}

Names Workflow::getPotentialCheckpoints() const
{
    if (mNetworkParameters.mCompressionMode != CompressionMode::NONE)
    {
        THROW_NONAME("Workflow", "compression not possible for checkpointed mode");
    }

    Names layers = getLayerNames();

    // Tensor name vs pair - Layer first, last names in sequence
    const Timeline timelineTensorsForward = getTimeline(layers, Usage::Forward);
    const Timeline timelineTensorsBackward = getTimeline(layers, Usage::Backward);

    const std::pair<Appearance, Appearance> appearanceForward = timelineToAppearance(timelineTensorsForward);
    const Appearance& timelineLayersLastForward = appearanceForward.second;

    const std::pair<Appearance, Appearance> appearanceBackward = timelineToAppearance(timelineTensorsBackward);
    const Appearance& timelineLayersFirstBackward = appearanceBackward.first;

    std::set<Name> recalcActivations;

    // find duplication of potential deallocations - tensors needed in backward (if output)
    for (const auto& appearBackward : timelineLayersFirstBackward)
    {
        for (const auto& tName : appearBackward.second)
        {
            for (auto& appearForward : timelineLayersLastForward)
            {
                auto lastForwardIt = std::find(appearForward.second.begin(), appearForward.second.end(), tName);
                if (lastForwardIt != appearForward.second.end())
                {
                    std::pair<bool, size_t> found = findLayerByOutput(tName);

                    if (found.first) // tensor is in outputs
                    {
                        recalcActivations.insert(tName);
                    }
                }
            }
        }
    }

    return Names(recalcActivations.begin(), recalcActivations.end());
}

template<>
MemoryManager& Workflow::getMemoryManager()
{
    return mMemoryManager;
}

template<>
MemoryManagerFP16& Workflow::getMemoryManager<MemoryManagerFP16>()
{
    return mMemoryManagerFP16;
}

template<>
const MemoryManager& Workflow::getMemoryManager() const
{
    return mMemoryManager;
}

template<>
const MemoryManagerFP16& Workflow::getMemoryManager<MemoryManagerFP16>() const
{
    return mMemoryManagerFP16;
}

Compiler& Workflow::getCompiler()
{
    if (!mUseCompiler)
    {
        THROW_NONAME("Workflow", "Compiler disabled");
    }

    return *mCompiler;
}

const Compiler& Workflow::getCompiler() const
{
    if (!mUseCompiler)
    {
        THROW_NONAME("Workflow", "Compiler disabled");
    }

    return *mCompiler;
}

void Workflow::overrideLayerExecutionTarget(LayerExecutionTarget layerExecutionTarget)
{
    if (layerExecutionTarget == LayerExecutionTarget::Default)
    {
        THROW_NONAME("Workflow", "Default parameter not possible");
    }

    if (mOverridedLayerExecutionTarget != LayerExecutionTarget::Default)
    {
        THROW_NONAME("Workflow", "Reset override first");
    }

    mOverridedLayerExecutionTarget = layerExecutionTarget;
}

const BasicLayer* Workflow::operator[](const raul::Name& name) const
{
    auto i = mLayersDict.find(name);
    if (i == mLayersDict.end())
    {
        THROW_NONAME("Workflow[operator", "layer with name [" + name + "] is not found");
    }
    return (*i).second;
}

BasicLayer* Workflow::operator[](const raul::Name& name)
{
    auto i = mLayersDict.find(name);
    if (i == mLayersDict.end())
    {
        THROW_NONAME("Workflow[operator", "layer with name [" + name + "] is not found");
    }
    return (*i).second;
}

void Workflow::printInfo(std::ostream& stream) const
{
    for (const auto& layer : mLayers)
    {
        stream << layer->getTypeName() << " [" << layer->getName() << "]: ";
        stream << std::endl;
        switch (mExecutionTarget)
        {
            case ExecutionTarget::CPU:
                utils::printTensorNames(stream, "inputs", layer->getInputs(), mMemoryManager);
                utils::printTensorNames(stream, "outputs", layer->getOutputs(), mMemoryManager);
                break;
            case ExecutionTarget::CPUFP16:
                utils::printTensorNames(stream, "inputs", layer->getInputs(), mMemoryManagerFP16);
                utils::printTensorNames(stream, "outputs", layer->getOutputs(), mMemoryManagerFP16);
                break;
        }
    }
}

std::unordered_set<std::string> Workflow::getSetOfLayers() const
{
    std::unordered_set<std::string> layers;
    for (const auto& layer : mLayers)
    {
        layers.insert(layer->getTypeName());
    }
    return layers;
}

const Workflow::Pipeline& Workflow::getPipeline(Workflow::Pipelines pipeline) const
{
    if (pipeline == Workflow::Pipelines::CreateBatched)
    {
        return mPipelineCreateBatched;
    }
    if (pipeline == Workflow::Pipelines::CreateNotBatched)
    {
        return mPipelineCreateNotBatched;
    }
    if (pipeline == Workflow::Pipelines::DeleteBatched)
    {
        return mPipelineDeleteBatched;
    }
    if (pipeline == Workflow::Pipelines::Zero)
    {
        return mPipelineZeroTensors;
    }
    if (pipeline == Workflow::Pipelines::ForwardTrain)
    {
        return mPipelineForwardTrain;
    }
    if (pipeline == Workflow::Pipelines::BackwardTrain)
    {
        return mPipelineBackwardTrain;
    }

    return mPipelineForwardTest;
}

void Workflow::addCallback(const Name& layerName, WorkflowListener& listener)
{
    if (!mIsPipelinesPrepared)
    {
        THROW_NONAME("Workflow", "pipelines not prepared");
    }

    if (mLayersDict.find(layerName) == mLayersDict.end())
    {
        THROW_NONAME("Workflow", "layer " + layerName + " not in topology");
    }

    auto i = mListeners.find(layerName);
    if (i == mListeners.end())
    {
        mListeners.insert({ layerName, { &listener } });
    }
    else
    {
        (*i).second.push_back(&listener);
    }
}

Workflow::Listeners Workflow::getListeners(const Name& layerName) const
{
    Listeners ret;

    auto i = mListeners.find(layerName);
    if (i != mListeners.end())
    {
        ret = (*i).second;
    }

    return ret;
}

void Workflow::executePipeline(Pipelines pipeline, const ActionParam& param) const
{
    if (!mIsPipelinesPrepared)
    {
        THROW_NONAME("Workflow", "pipelines not prepared");
    }

    const Pipeline& pipe = getPipeline(pipeline);
    for (auto& action : pipe)
    {
        action->execute(param);
    }
}

void Workflow::createTrainCheckpointedPipeline()
{

    if (mNetworkParameters.mCompressionMode != CompressionMode::NONE)
    {
        THROW_NONAME("Workflow", "compression not possible for checkpointed mode");
    }

    Names layers = getLayerNames();

    // Tensor name vs pair - Layer first, last names in sequence
    Timeline timelineTensorsForward = getTimeline(layers, Usage::Forward);
    Timeline timelineTensorsBackward = getTimeline(layers, Usage::Backward);

    std::pair<Appearance, Appearance> appearanceForward = timelineToAppearance(timelineTensorsForward);
    const Appearance& timelineLayersFirstForward = appearanceForward.first;
    Appearance& timelineLayersLastForward = appearanceForward.second;

    std::pair<Appearance, Appearance> appearanceBackward = timelineToAppearance(timelineTensorsBackward);
    Appearance& timelineLayersFirstBackward = appearanceBackward.first;
    Appearance& timelineLayersLastBackward = appearanceBackward.second;

    std::unordered_set<Name> recalcActivations;

    // to handle skip connection issue and deletion of already calculated activations
    std::unordered_set<Name> recalcActivationsAllocated;

    std::unordered_set<Name> internalTensors;

    // remove duplication of potential allocations not including checkpoints - internal tensors
    for (const auto& appearForward : timelineLayersFirstForward)
    {
        for (const auto& tName : appearForward.second)
        {
            for (auto& appearBackward : timelineLayersLastBackward)
            {
                auto lastBackIt = std::find(appearBackward.second.begin(), appearBackward.second.end(), tName);
                if (lastBackIt != appearBackward.second.end())
                {
                    if (!isCheckpoint(tName))
                    {
                        std::pair<bool, size_t> found = findLayerByOutput(tName);

                        if (!found.first) // tensor not in outputs
                        {
                            internalTensors.insert(tName);
                            appearBackward.second.erase(lastBackIt);
                        }
                    }
                }
            }
        }
    }

    // find duplication of potential deallocations not including checkpoints - tensors needed in backward (if output) or internal tensors (remove duplication)
    for (const auto& appearBackward : timelineLayersFirstBackward)
    {
        for (const auto& tName : appearBackward.second)
        {
            for (auto& appearForward : timelineLayersLastForward)
            {
                auto lastForwardIt = std::find(appearForward.second.begin(), appearForward.second.end(), tName);
                if (lastForwardIt != appearForward.second.end())
                {
                    if (!isCheckpoint(tName))
                    {
                        std::pair<bool, size_t> found = findLayerByOutput(tName);

                        if (found.first) // tensor is in outputs
                        {
                            recalcActivations.insert(tName);
                        }
                        else
                        {
                            internalTensors.insert(tName);
                            appearForward.second.erase(lastForwardIt);
                        }
                    }
                }
            }
        }
    }

    // remove duplication of potential checkpoints deallocations, allocations
    removeDuplications(mCheckpoints, timelineLayersLastForward, timelineLayersLastBackward);

    // create pipelines
    // forward
    for (auto& mLayer : mLayers)
    {
        fillTrainPipeline(mPipelineForwardTrain, mLayer.get(), timelineLayersFirstForward, timelineLayersLastForward, Usage::Forward);
    }

    // if checkpoint lifetime differs for forward/backward (different layers)
    // then deallocation might happen earlier than expected
    // better to correct
    for (auto& cp : mCheckpoints)
    {
        Name forwardLayerName;
        Name backwardLayerName;

        auto itTF = timelineTensorsForward.find(cp);
        if (itTF != timelineTensorsForward.end())
        {
            forwardLayerName = (*itTF).second.first;
        }

        auto itTB = timelineTensorsBackward.find(cp);
        if (itTB != timelineTensorsBackward.end())
        {
            backwardLayerName = (*itTB).second.first;
        }

        if (!forwardLayerName.empty() && !backwardLayerName.empty() && forwardLayerName != backwardLayerName)
        {
            removeDuplications({ cp }, timelineLayersFirstBackward);

            auto itfB = timelineLayersFirstBackward.find(forwardLayerName);
            if (itfB != timelineLayersFirstBackward.end())
            {
                (*itfB).second.push_back(cp);
            }
            else
            {
                timelineLayersFirstBackward.insert({ forwardLayerName, { cp } });
            }
        }
    }

    size_t recalcAmount = 0;

    Timeline timelineJointPool;
    Names lNamesPool;

    if (mAllocationMode == AllocationMode::POOL)
    {
        mWorkflowPoolTrain->clearTensorNameMapper();

        timelineJointPool = std::get<0>(appearanceToTimeline(appearanceForward, {})); // d.polubotko: use CPUFP32 only

        lNamesPool = getLayerNames();
    }

    // backward
    for (auto it = mLayers.rbegin(); it != mLayers.rend(); ++it)
    {
        const Name& lName = (*it)->getName();
        auto lItL = timelineLayersLastBackward.find(lName);

        // potential allocations
        if (lItL != timelineLayersLastBackward.end())
        {
            // perform forward processing before gradients allocation and zero finished (issues with pool)
            for (const auto& tName : (*lItL).second)
            {
                if (recalcActivations.find(tName) != recalcActivations.end())
                {
                    Layers forwardPathLayers = findPathFromActivationTillCheckpoint(tName, recalcActivationsAllocated);
                    Names forwardPath;

                    for (auto layer : forwardPathLayers)
                    {
                        forwardPath.push_back(layer->getName());
                    }

                    Timeline subTimelineTensorsForward = getTimeline(forwardPath, Usage::Forward);
                    std::pair<Appearance, Appearance> subAppearanceForward = timelineToAppearance(subTimelineTensorsForward);
                    Appearance& subTimelineLayersFirstForward = subAppearanceForward.first;
                    Appearance& subTimelineLayersLastForward = subAppearanceForward.second;

                    // remove duplication of checkpoints allocations, deallocations
                    removeDuplications(mCheckpoints, subTimelineLayersFirstForward, subTimelineLayersLastForward);

                    // remove duplication of potential activation deallocations
                    for (auto& appearForward : subTimelineLayersLastForward)
                    {
                        auto lastForwardIt = std::find(appearForward.second.begin(), appearForward.second.end(), tName);
                        if (lastForwardIt != appearForward.second.end())
                        {
                            appearForward.second.erase(lastForwardIt);
                        }
                    }

                    // remove duplication of internal tensors allocations, deallocations
                    removeDuplications(internalTensors, subTimelineLayersFirstForward, subTimelineLayersLastForward);

                    // remove duplication of already found tensors (skip connection issue)
                    removeDuplications(recalcActivationsAllocated, subTimelineLayersFirstForward, subTimelineLayersLastForward);

                    if (mAllocationMode == AllocationMode::POOL)
                    {
                        Timeline timelineJoint = std::get<0>(appearanceToTimeline(subAppearanceForward, {})); // d.polubotko: use CPUFP32 only

                        const Name suffix = suffixForward + Conversions::toString(recalcAmount);

                        for (const auto& timeline : timelineJoint)
                        {
                            if (timeline.first == tName)
                            {
                                mWorkflowPoolTrain->setTensorNameMapper(tName + suffixRecalced, tName);

                                Name lNameJF = timeline.second.first + suffix;

                                timelineJointPool.insert({ tName + suffixRecalced, { lNameJF, "" } });
                            }
                            else
                            {
                                mWorkflowPoolTrain->setTensorNameMapper(timeline.first + suffix, timeline.first);
                                Name tNameJ = timeline.first + suffix;
                                Name lNameJF = timeline.second.first + suffix;
                                Name lNameJL = timeline.second.second + suffix;

                                timelineJointPool.insert({ tNameJ, { lNameJF, lNameJL } });
                            }
                        }

                        for (auto& lNameForw : forwardPath)
                        {
                            lNamesPool.push_back(lNameForw + suffixForward + Conversions::toString(recalcAmount));
                        }
                    }

                    // create subpipeline
                    for (const auto& lForwardLayer : forwardPathLayers)
                    {
                        fillTrainPipelineCheckpointed(mPipelineBackwardTrain, lForwardLayer, subTimelineLayersFirstForward, subTimelineLayersLastForward, mWorkflowPoolTrain, recalcAmount, tName);
                    }

                    recalcActivationsAllocated.insert(tName);

                    ++recalcAmount;
                }
            }

            // perform gradients allocation and zero after forward pass finished (issues with pool)
            for (const auto& tName : (*lItL).second)
            {
                if (recalcActivations.find(tName) == recalcActivations.end()) // not layer output (gradient)
                {
                    mPipelineBackwardTrain.push_back(std::make_shared<Allocate<MemoryManager>>(mMemoryManager, tName, *this, mWorkflowPoolTrain));

                    auto tensorUsage = mWorkflowDB->getCell(mWorkflowDB->getLayersTable(), lName, tName);
                    WorkflowDB::TensorUsage usage = mWorkflowDB->getUsage(tensorUsage[static_cast<size_t>(Usage::Backward)]);
                    if (usage.isZero)
                    {
                        mPipelineBackwardTrain.push_back(std::make_shared<Zero<MemoryManager>>(mMemoryManager, tName));
                    }

                    if (mAllocationMode == AllocationMode::POOL)
                    {
                        timelineJointPool.insert({ tName, { lName + suffixBack, "" } });
                    }
                }
            }
        }

        auto layer = (*it).get();
        auto backward = std::make_shared<Backward>(layer, *this);
        auto itScale = mScalingStrategies.find(layer->getName());
        if (itScale != mScalingStrategies.end())
        {
            backward->setScaling(itScale->second);
        }
        mPipelineBackwardTrain.push_back(backward);

        auto lItF = timelineLayersFirstBackward.find(lName);

        if (lItF != timelineLayersFirstBackward.end())
        {
            for (const auto& tName : (*lItF).second)
            {
                mPipelineBackwardTrain.push_back(std::make_shared<Deallocate<MemoryManager>>(mMemoryManager, tName));

                if (recalcActivationsAllocated.find(tName) != recalcActivationsAllocated.end())
                {
                    recalcActivationsAllocated.erase(tName);
                }

                if (mAllocationMode == AllocationMode::POOL)
                {
                    auto itTimeJP = timelineJointPool.find(tName);

                    if (recalcActivations.find(tName) != recalcActivations.end())
                    {
                        itTimeJP = timelineJointPool.find(tName + suffixRecalced);
                    }

                    (*itTimeJP).second.second = lName + suffixBack;
                }
            }
        }

        if (mAllocationMode == AllocationMode::POOL)
        {
            lNamesPool.push_back(lName + suffixBack);
        }
    }

    if (mAllocationMode == AllocationMode::POOL)
    {
        mWorkflowPoolTrain->createIntervals(lNamesPool, timelineJointPool);
    }
}

std::pair<bool, size_t> Workflow::findLayerByOutput(const Name& tensor) const
{
    std::pair<bool, size_t> ret(false, 0);

    for (size_t q = 0; q < mLayers.size(); ++q)
    {
        const Names& outputs = mLayers[q]->getOutputs();

        if (std::find(outputs.begin(), outputs.end(), tensor) != outputs.end())
        {
            ret.first = true;
            ret.second = q;
            break;
        }
    }

    return ret;
}

bool Workflow::isCheckpoint(const Name& tensor) const
{
    return std::find(mCheckpoints.begin(), mCheckpoints.end(), tensor) != mCheckpoints.end();
}

bool Workflow::isPersistent(const Name& tensor) const
{
    WorkflowDB::TensorUsage usg = mWorkflowDB->findFirstTensor(tensor);
    return !usg.isOptimizeMem;
}

Workflow::Layers Workflow::findPathFromActivationTillCheckpoint(const Name& tensor, const std::unordered_set<Name>& alreadyFound) const
{
    std::pair<bool, size_t> found = findLayerByOutput(tensor);

    if (!found.first)
    {
        THROW_NONAME("Workflow", "tensor " + tensor + " not in outputs");
    }

    Layers tempRet;

    traverseGraphTillCheckpoint(mLayers[found.second].get(), tempRet, alreadyFound);

    std::reverse(tempRet.begin(), tempRet.end());

    Layers ret; // layers in forward order
    std::unordered_set<Name> visited;

    for (const auto& layer : tempRet)
    {
        if (visited.find(layer->getName()) == visited.end())
        {
            visited.insert(layer->getName());
            ret.push_back(layer);
        }
    }

    return ret;
}

void Workflow::traverseGraphTillCheckpoint(BasicLayer* layer, Layers& names, const std::unordered_set<Name>& alreadyFound) const
{
    names.push_back(layer);

    if (layer->getInputs().empty())
    {
        std::string outNames;
        for (const auto& name : layer->getOutputs())
        {
            outNames += name + " ";
        }
        THROW_NONAME("Workflow", "not enough checkpoints in layer " + layer->getName() + " outputs: " + outNames);
    }

    for (const auto& tName : layer->getInputs())
    {
        if (!isCheckpoint(tName) && !isPersistent(tName) && alreadyFound.find(tName) == alreadyFound.end())
        {
            std::pair<bool, size_t> found = findLayerByOutput(tName);

            if (!found.first)
            {
                THROW_NONAME("Workflow", "tensor " + tName + " not in outputs");
            }

            traverseGraphTillCheckpoint(mLayers[found.second].get(), names, alreadyFound);
        }
    }
}

void Workflow::removeDuplications(const Names& tNames, Appearance& appear) const
{
    for (const auto& tName : tNames)
    {
        for (auto& appearForward : appear)
        {
            auto lastForwardIt = std::find(appearForward.second.begin(), appearForward.second.end(), tName);
            if (lastForwardIt != appearForward.second.end())
            {
                appearForward.second.erase(lastForwardIt);
            }
        }
    }
}

void Workflow::removeDuplications(const Names& tNames, Appearance& appearA, Appearance& appearB) const
{
    removeDuplications(tNames, appearA);
    removeDuplications(tNames, appearB);
}

void Workflow::removeDuplications(const std::unordered_set<Name>& tNames, Appearance& appearA, Appearance& appearB) const
{
    for (const auto& tName : tNames)
    {
        for (auto& appearForward : appearA)
        {
            auto lastForwardIt = std::find(appearForward.second.begin(), appearForward.second.end(), tName);
            if (lastForwardIt != appearForward.second.end())
            {
                appearForward.second.erase(lastForwardIt);
            }
        }

        for (auto& appearForward : appearB)
        {
            auto lastForwardIt = std::find(appearForward.second.begin(), appearForward.second.end(), tName);
            if (lastForwardIt != appearForward.second.end())
            {
                appearForward.second.erase(lastForwardIt);
            }
        }
    }
}

bool Workflow::isUniqueNames(const Names& names) const
{
    std::unordered_set<Name> unique;
    for (const auto& tName : names)
    {
        auto it = unique.find(tName);
        if (it != unique.end())
        {
            return false;
        }
        else
        {
            unique.insert(tName);
        }
    }

    return true;
}

void Workflow::fillTrainPipeline(Pipeline& pipeline, BasicLayer* layer, const Appearance& first, const Appearance& last, Usage usage)
{
    if (usage != Usage::Forward && usage != Usage::Backward)
    {
        THROW_NONAME("Workflow", "incorrect usage parameter");
    }

    execTargetFillTrainPipeline(pipeline, layer, first, last, usage);
}

void Workflow::fillTrainPipelineCheckpointed(Pipeline& pipeline,
                                             BasicLayer* layer,
                                             const Appearance& first,
                                             const Appearance& last,
                                             std::shared_ptr<WorkflowPool<MemoryManager>>& pool,
                                             size_t recalcAmount,
                                             const Name& recalcTensor)
{
    const Name& lName = layer->getName();
    auto lItF = first.find(lName);
    auto lItL = last.find(lName);

    if (lItF != first.end())
    {
        for (const auto& tName : (*lItF).second)
        {
            if (mAllocationMode == AllocationMode::POOL)
            {
                if (recalcTensor == tName)
                {
                    Name tNamePool = tName + suffixRecalced;
                    pipeline.push_back(std::make_shared<Allocate<MemoryManager>>(mMemoryManager, tName, tNamePool, *this, pool));
                }
                else
                {
                    Name suffix = suffixForward + Conversions::toString(recalcAmount);
                    Name tNamePool = tName + suffix;
                    pipeline.push_back(std::make_shared<Allocate<MemoryManager>>(mMemoryManager, tName, tNamePool, *this, pool));
                }
            }
            else
            {
                pipeline.push_back(std::make_shared<Allocate<MemoryManager>>(mMemoryManager, tName, *this, pool));
            }

            auto tensorUsage = mWorkflowDB->getCell(mWorkflowDB->getLayersTable(), lName, tName);
            WorkflowDB::TensorUsage usg = mWorkflowDB->getUsage(tensorUsage[static_cast<size_t>(Usage::Forward)]);
            if (usg.isZero)
            {
                pipeline.push_back(std::make_shared<Zero<MemoryManager>>(mMemoryManager, tName));
            }
        }
    }

    pipeline.push_back(std::make_shared<Forward>(layer, NetworkMode::TrainCheckpointed, *this));

    if (lItL != last.end())
    {
        for (const auto& tName : (*lItL).second)
        {
            pipeline.push_back(std::make_shared<Deallocate<MemoryManager>>(mMemoryManager, tName));
        }
    }
}

void Workflow::fillTrainPipelineCompression(Pipeline& pipeline, BasicLayer* layer, const Appearance& appear, Usage usage)
{
    if (usage != Usage::Forward && usage != Usage::Backward)
    {
        THROW_NONAME("Workflow", "incorrect usage parameter");
    }

    const Name& lName = layer->getName();
    auto lItApp = appear.find(lName);

    if (lItApp != appear.end())
    {
        for (const auto& tName : (*lItApp).second)
        {
            auto tensorUsage = mWorkflowDB->getCell(mWorkflowDB->getLayersTable(), lName, tName);
            WorkflowDB::TensorUsage usg = mWorkflowDB->getUsage(tensorUsage[static_cast<size_t>(usage)]);
            if (usg.isCompress)
            {
                if (mExecutionTarget == ExecutionTarget::CPU)
                {
                    if (usage == Usage::Forward)
                    {
                        pipeline.push_back(std::make_shared<Compress<MemoryManager>>(mMemoryManager, tName, mNetworkParameters.mCompressionMode));
                    }

                    if (usage == Usage::Backward)
                    {
                        pipeline.push_back(std::make_shared<Decompress<MemoryManager>>(mMemoryManager, tName, mNetworkParameters.mCompressionMode));
                    }
                }
                else if (mExecutionTarget == ExecutionTarget::CPUFP16)
                {
                    if (usage == Usage::Forward)
                    {
                        pipeline.push_back(std::make_shared<Compress<MemoryManagerFP16>>(mMemoryManagerFP16, tName, mNetworkParameters.mCompressionMode));
                    }

                    if (usage == Usage::Backward)
                    {
                        pipeline.push_back(std::make_shared<Decompress<MemoryManagerFP16>>(mMemoryManagerFP16, tName, mNetworkParameters.mCompressionMode));
                    }
                }
            }
        }
    }
}

void Workflow::execTargetCreateAuxPipelines(const std::unordered_map<Name, size_t>& uniqueTensors)
{
    for (const auto& uniqueTensor : uniqueTensors)
    {
        WorkflowDB::TensorUsage usage = mWorkflowDB->getUsage(uniqueTensor.second);

        if (usage.shape.isBSDependent())
        {
            if (usage.isOptimizeMem)
            {
                mPipelineCreateBatched.push_back(newActionCreateShape(usage.tensorName, usage.layerExecutionTarget, usage.shape));
            }
            else
            {
                mPipelineCreateBatched.push_back(newActionCreateTensor(usage.tensorName, usage.layerExecutionTarget, usage.shape));
            }

            mPipelineDeleteBatched.push_back(newActionDeleteTensor(usage.tensorName, usage.layerExecutionTarget));
        }
        else
        {
            if (usage.isOptimizeMem)
            {
                mPipelineCreateNotBatched.push_back(newActionCreateShape(usage.tensorName, usage.layerExecutionTarget, usage.shape));
            }
            else
            {
                mPipelineCreateNotBatched.push_back(newActionCreateTensor(usage.tensorName, usage.layerExecutionTarget, usage.shape));
            }
        }

        if (usage.isZero)
        {
            if (!usage.isOptimizeMem)
            {
                mPipelineZeroTensors.push_back(newActionZero(usage.tensorName, usage.layerExecutionTarget));
            }
        }
    }
}

void Workflow::execTargetCreateForwardTestPipeline(const Appearance& timelineLayersFirst, const Appearance& timelineLayersLast)
{
    for (auto& layer : mLayers)
    {
        const Name& lName = layer->getName();
        auto lItF = timelineLayersFirst.find(lName);
        auto lItL = timelineLayersLast.find(lName);

        if (lItF != timelineLayersFirst.end())
        {
            for (const auto& tName : (*lItF).second)
            {
                std::vector<size_t> tensorUsage = mWorkflowDB->getCell(mWorkflowDB->getLayersTable(), lName, tName);
                WorkflowDB::TensorUsage usg = mWorkflowDB->getUsage(tensorUsage[static_cast<size_t>(Usage::Forward)]);

                mPipelineForwardTest.push_back(newActionAllocate(tName, usg.layerExecutionTarget, false));
            }
        }

        mPipelineForwardTest.push_back(std::make_shared<Forward>(layer.get(), NetworkMode::Test, *this));

        if (lItL != timelineLayersLast.end())
        {
            for (const auto& tName : (*lItL).second)
            {
                std::vector<size_t> tensorUsage = mWorkflowDB->getCell(mWorkflowDB->getLayersTable(), lName, tName);
                WorkflowDB::TensorUsage usg = mWorkflowDB->getUsage(tensorUsage[static_cast<size_t>(Usage::Forward)]);

                mPipelineForwardTest.push_back(newActionDeallocate(tName, usg.layerExecutionTarget));
            }
        }
    }
}

void Workflow::execTargetFillTrainPipeline(Pipeline& pipeline, BasicLayer* layer, const Appearance& first, const Appearance& last, Usage usage)
{
    const Name& lName = layer->getName();
    auto lItF = first.find(lName);
    auto lItL = last.find(lName);

    if (lItF != first.end())
    {
        for (const auto& tName : (*lItF).second)
        {
            auto tensorUsage = mWorkflowDB->getCell(mWorkflowDB->getLayersTable(), lName, tName);
            WorkflowDB::TensorUsage usg = mWorkflowDB->getUsage(tensorUsage[static_cast<size_t>(usage)]);

            pipeline.push_back(newActionAllocate(tName, usg.layerExecutionTarget, true));

            if (usg.isZero)
            {
                pipeline.push_back(newActionZero(tName, usg.layerExecutionTarget));
            }
        }
    }

    if (usage == Usage::Forward)
    {
        pipeline.push_back(std::make_shared<Forward>(layer, NetworkMode::Train, *this));
    }

    if (usage == Usage::Backward)
    {
        auto backward = std::make_shared<Backward>(layer, *this);
        auto it = mScalingStrategies.find(layer->getName());
        if (it != mScalingStrategies.end())
        {
            backward->setScaling(it->second);
        }
        pipeline.push_back(backward);
    }

    if (lItL != last.end())
    {
        for (const auto& tName : (*lItL).second)
        {
            std::vector<size_t> tensorUsage = mWorkflowDB->getCell(mWorkflowDB->getLayersTable(), lName, tName);
            WorkflowDB::TensorUsage usg = mWorkflowDB->getUsage(tensorUsage[static_cast<size_t>(usage)]);

            pipeline.push_back(newActionDeallocate(tName, usg.layerExecutionTarget));
        }
    }
}

Workflow::ActionMem Workflow::newActionCreateShape(const Name& name, LayerExecutionTarget layerExecutionTarget, const WShape& shape)
{
    ActionMem ret;

    ExecutionTarget target = mExecutionTarget;

    if (layerExecutionTarget != LayerExecutionTarget::Default)
    {
        target = static_cast<ExecutionTarget>(layerExecutionTarget);
    }

    if (target == ExecutionTarget::CPU)
    {
        ret = std::make_shared<CreateShape<MemoryManager>>(getMemoryManager<MemoryManager>(), name, shape, *this);
    }
    else if (target == ExecutionTarget::CPUFP16)
    {
        ret = std::make_shared<CreateShape<MemoryManagerFP16>>(getMemoryManager<MemoryManagerFP16>(), name, shape, *this);
    }
    else
    {
        THROW_NONAME("Workflow", "wrong execution target");
    }

    return ret;
}

Workflow::ActionMem Workflow::newActionCreateTensor(const Name& name, LayerExecutionTarget layerExecutionTarget, const WShape& shape)
{
    ActionMem ret;

    ExecutionTarget target = mExecutionTarget;

    if (layerExecutionTarget != LayerExecutionTarget::Default)
    {
        target = static_cast<ExecutionTarget>(layerExecutionTarget);
    }

    if (target == ExecutionTarget::CPU)
    {
        ret = std::make_shared<CreateTensor<MemoryManager>>(getMemoryManager<MemoryManager>(), name, shape, *this);
    }
    else if (target == ExecutionTarget::CPUFP16)
    {
        ret = std::make_shared<CreateTensor<MemoryManagerFP16>>(getMemoryManager<MemoryManagerFP16>(), name, shape, *this);
    }
    else
    {
        THROW_NONAME("Workflow", "wrong execution target");
    }

    return ret;
}

Workflow::ActionMem Workflow::newActionDeleteTensor(const Name& name, LayerExecutionTarget layerExecutionTarget)
{
    ActionMem ret;

    ExecutionTarget target = mExecutionTarget;

    if (layerExecutionTarget != LayerExecutionTarget::Default)
    {
        target = static_cast<ExecutionTarget>(layerExecutionTarget);
    }

    if (target == ExecutionTarget::CPU)
    {
        ret = std::make_shared<DeleteTensor<MemoryManager>>(getMemoryManager<MemoryManager>(), name);
    }
    else if (target == ExecutionTarget::CPUFP16)
    {
        ret = std::make_shared<DeleteTensor<MemoryManagerFP16>>(getMemoryManager<MemoryManagerFP16>(), name);
    }
    else
    {
        THROW_NONAME("Workflow", "wrong execution target");
    }

    return ret;
}

Workflow::ActionMem Workflow::newActionZero(const Name& name, LayerExecutionTarget layerExecutionTarget)
{
    ActionMem ret;

    ExecutionTarget target = mExecutionTarget;

    if (layerExecutionTarget != LayerExecutionTarget::Default)
    {
        target = static_cast<ExecutionTarget>(layerExecutionTarget);
    }

    if (target == ExecutionTarget::CPU)
    {
        ret = std::make_shared<Zero<MemoryManager>>(getMemoryManager<MemoryManager>(), name);
    }
    else if (target == ExecutionTarget::CPUFP16)
    {
        ret = std::make_shared<Zero<MemoryManagerFP16>>(getMemoryManager<MemoryManagerFP16>(), name);
    }
    else
    {
        THROW_NONAME("Workflow", "wrong execution target");
    }

    return ret;
}

Workflow::ActionMem Workflow::newActionAllocate(const Name& name, LayerExecutionTarget layerExecutionTarget, bool isTrain)
{
    ActionMem ret;

    ExecutionTarget target = mExecutionTarget;

    if (layerExecutionTarget != LayerExecutionTarget::Default)
    {
        target = static_cast<ExecutionTarget>(layerExecutionTarget);
    }

    if (target == ExecutionTarget::CPU)
    {
        ret = std::make_shared<Allocate<MemoryManager>>(getMemoryManager<MemoryManager>(), name, *this, isTrain ? mWorkflowPoolTrain : mWorkflowPoolTest);
    }
    else if (target == ExecutionTarget::CPUFP16)
    {
        ret = std::make_shared<Allocate<MemoryManagerFP16>>(getMemoryManager<MemoryManagerFP16>(), name, *this, isTrain ? mWorkflowPoolTrainFP16 : mWorkflowPoolTestFP16);
    }
    else
    {
        THROW_NONAME("Workflow", "wrong execution target");
    }

    return ret;
}

Workflow::ActionMem Workflow::newActionDeallocate(const Name& name, LayerExecutionTarget layerExecutionTarget)
{
    ActionMem ret;

    ExecutionTarget target = mExecutionTarget;

    if (layerExecutionTarget != LayerExecutionTarget::Default)
    {
        target = static_cast<ExecutionTarget>(layerExecutionTarget);
    }

    if (target == ExecutionTarget::CPU)
    {
        ret = std::make_shared<Deallocate<MemoryManager>>(getMemoryManager<MemoryManager>(), name);
    }
    else if (target == ExecutionTarget::CPUFP16)
    {
        ret = std::make_shared<Deallocate<MemoryManagerFP16>>(getMemoryManager<MemoryManagerFP16>(), name);
    }
    else
    {
        THROW_NONAME("Workflow", "wrong execution target");
    }

    return ret;
}

template std::vector<ParamAndGradImpl<TensorFP16>> Workflow::getTrainableParameters<MemoryManagerFP16>();
template std::vector<ParamAndGradImpl<Tensor>> Workflow::getTrainableParameters<MemoryManager>();
template std::vector<ParamAndGradImpl<TensorFP16>> Workflow::getTrainableParametersSafe<MemoryManagerFP16>();
template std::vector<ParamAndGradImpl<Tensor>> Workflow::getTrainableParametersSafe<MemoryManager>();
} // namespace raul