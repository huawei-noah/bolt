// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "Compiler.h"

#include <algorithm>

#include <training/base/layers/BasicLayer.h>
#include <training/base/layers/basic/ConvertPrecisionLayer.h>
#include <training/compiler/LayerBuilder.h>

namespace
{

const raul::Name suffixFP32 = "FP32";
const raul::Name suffixFP16 = "FP16";

/**
 * @brief Params for TrainableParamsConvertLayer
 *
 */
struct TrainableParamsConvertLayerParams : public raul::BasicParams
{
    TrainableParamsConvertLayerParams() = delete;

    TrainableParamsConvertLayerParams(const raul::Name& layerFromName, bool isBeforeLayer = true, raul::LayerExecutionTarget target = raul::LayerExecutionTarget::CPU)
        : raul::BasicParams({}, {})
        , mLayerFromName(layerFromName)
        , mIsBeforeLayer(isBeforeLayer)
        , mTarget(target)
    {
        if (layerFromName.empty())
        {
            THROW_NONAME("TrainableParamsConvertLayerParams", "provided layer name is empty");
        }
    }

    raul::Name mLayerFromName;
    bool mIsBeforeLayer;
    raul::LayerExecutionTarget mTarget;

    void print(std::ostream& stream) const override
    {
        raul::BasicParams::print(stream);
        stream << "Layer from: " << mLayerFromName << ", position: " << (mIsBeforeLayer ? "before" : "after");
    }
};

/**
 * @brief TrainableParamsConvertLayer
 * Do copy-convert operation with trainable params
 * Should be used only implicitly inside compiler
 *
 *
 */
class TrainableParamsConvertLayer : public raul::BasicLayer
{
  public:
    TrainableParamsConvertLayer(const raul::Name& name, const TrainableParamsConvertLayerParams& params, raul::NetworkParameters& networkParameters)
        : BasicLayer(name, "TrainableParamsConvert", params, networkParameters, { false, false })
        , mLayerFromName(params.mLayerFromName)
        , mIsBeforeLayer(params.mIsBeforeLayer)
        , mMasterWeightsInitiated(false)
    {
        if (!(params.mTarget == raul::LayerExecutionTarget::CPU && mNetworkParams.mWorkflow.getExecutionTarget() == raul::ExecutionTarget::CPUFP16))
        {
            THROW(mTypeName, mName, "unsupported combination of layer's and global execution targets provided");
        }

        if (!mNetworkParams.mWorkflow.isCompilerEnabled())
        {
            THROW(mTypeName, mName, "complier should be enabled to use this layer");
        }

        if (!mIsBeforeLayer)
        {
            const auto prevLayerTrainableParams = mNetworkParams.mWorkflow.getLayerTrainableParameterNames(mLayerFromName);

            if (prevLayerTrainableParams.empty())
            {
                THROW(mTypeName, mName, mLayerFromName + " layer should be trainable");
            }

            for (size_t i = 0; i < prevLayerTrainableParams.size(); ++i)
            {
                // Layer frozen if no grad declared
                auto isLayerFrozen = !mNetworkParams.mWorkflow.isTensorDeclared(prevLayerTrainableParams[i].grad());
                mNetworkParams.mWorkflow.copyDeclaration(mLayerFromName,
                                                         prevLayerTrainableParams[i],
                                                         prevLayerTrainableParams[i] + "_fp32",
                                                         raul::Workflow::Usage::ForwardAndBackward,
                                                         raul::Workflow::Mode::Read,
                                                         false,
                                                         false,
                                                         !isLayerFrozen,
                                                         false,
                                                         false,
                                                         params.mTarget);
                if (!isLayerFrozen)
                {
                    mNetworkParams.mWorkflow.copyDeclaration(
                        mLayerFromName, prevLayerTrainableParams[i] + "_fp32", raul::Name(prevLayerTrainableParams[i] + "_fp32").grad(), DEC_TRAINABLE_GRAD, params.mTarget);
                }
            }
        }
    }

    TrainableParamsConvertLayer(TrainableParamsConvertLayer&&) = default;
    TrainableParamsConvertLayer(const TrainableParamsConvertLayer&) = delete;
    TrainableParamsConvertLayer& operator=(const TrainableParamsConvertLayer&) = delete;

    void forwardComputeImpl(raul::NetworkMode) override
    {
        if (mIsBeforeLayer)
        {
            raul::Workflow& work = mNetworkParams.mWorkflow;

            const auto prevLayerTrainableParams = work.getLayerTrainableParameterNames(mLayerFromName);

            auto& memoryManagerFP16 = work.getMemoryManager<raul::MemoryManagerFP16>();
            auto& memoryManagerFP32 = work.getMemoryManager<raul::MemoryManager>();

            for (size_t i = 0; i < prevLayerTrainableParams.size(); ++i)
            {
                if (!memoryManagerFP16.tensorExists(prevLayerTrainableParams[i]))
                {
                    continue;
                }

                if (!memoryManagerFP32.tensorExists(prevLayerTrainableParams[i] + "_fp32"))
                {
                    THROW(mTypeName, mName, "FP32 copy of needed parameter does not exist");
                }

                auto& paramsFP16 = memoryManagerFP16[prevLayerTrainableParams[i]];
                auto& paramsFP32 = memoryManagerFP32[prevLayerTrainableParams[i] + "_fp32"];
                if (!mMasterWeightsInitiated)
                {
                    // Init master weights
                    std::transform(paramsFP16.begin(), paramsFP16.end(), paramsFP32.begin(), [](const raul::half& val) { return raul::toFloat32(val); });
                }
                else
                {
                    // Convert
                    std::transform(paramsFP32.begin(), paramsFP32.end(), paramsFP16.begin(), [](const raul::dtype& val) { return raul::toFloat16(val); });
                }
            }

            if (!mMasterWeightsInitiated)
            {
                mMasterWeightsInitiated = true;
            }
        }
    }

    void backwardComputeImpl() override
    {
        if (mIsBeforeLayer)
        {
            raul::Workflow& work = mNetworkParams.mWorkflow;

            const auto prevLayerTrainableParams = work.getLayerTrainableParameterNames(mLayerFromName);

            auto& memoryManagerFP16 = work.getMemoryManager<raul::MemoryManagerFP16>();
            auto& memoryManagerFP32 = work.getMemoryManager<raul::MemoryManager>();

            for (size_t i = 0; i < prevLayerTrainableParams.size(); ++i)
            {
                if (!memoryManagerFP16.tensorExists(prevLayerTrainableParams[i].grad()))
                {
                    continue;
                }

                if (!memoryManagerFP32.tensorExists(raul::Name(prevLayerTrainableParams[i] + "_fp32").grad()))
                {
                    THROW(mTypeName, mName, "FP32 copy of needed parameter's gradient does not exist");
                }

                auto& paramsGradFP16 = memoryManagerFP16[prevLayerTrainableParams[i].grad()];
                auto& paramsGradFP32 = memoryManagerFP32[raul::Name(prevLayerTrainableParams[i] + "_fp32").grad()];
                std::transform(paramsGradFP16.begin(), paramsGradFP16.end(), paramsGradFP32.begin(), [](const raul::half& val) { return raul::toFloat32(val); });
            }
        }
    }

  private:
    raul::Name mLayerFromName;
    bool mIsBeforeLayer;
    bool mMasterWeightsInitiated;
};

} // anonymous

namespace raul
{

Constraint::Constraint(const Name& layer, ConstraintImpl cImpl)
    : mLayerFrom(layer)
    , mOutputConversion(true)
    , mConstraintImpl(cImpl)
{
    if (layer.empty())
    {
        THROW_NONAME("Constraint", "Empty layer name");
    }
}

Constraint::Constraint(const Name& layerFrom, const Name& layerTo, ConstraintImpl cImpl)
    : mLayerFrom(layerFrom)
    , mLayerTo(layerTo)
    , mOutputConversion(true)
    , mConstraintImpl(cImpl)
{
    if (layerFrom.empty())
    {
        THROW_NONAME("Constraint", "Empty layerFrom name");
    }

    if (layerTo.empty())
    {
        THROW_NONAME("Constraint", "Empty layerTo name");
    }
}
Compiler::Compiler(ExecutionTarget executionTarget)
    : mImplementationResolved(false)
    , mGlobalExecutionTarget(executionTarget)
{
}

void Compiler::setImpl(const std::vector<LayerMem>& redefinedFrontLayers, size_t index, bool constrainActivated)
{
    std::unique_ptr<BasicImpl> foundImpl;

    const auto& front = redefinedFrontLayers.back();

    auto& frontref = *front;
    Name type = typeid(frontref).name();

    auto& map = getMapImpl(mPerLayerConstraintImpl[index]);

    const auto impl = map.find(type);
    if (impl != map.end())
    {
        foundImpl = impl->second->create(front.get());

        if (foundImpl == nullptr)
        {
            THROW_NONAME("Compiler", "Type \"" + type + "\" not instantiated");
        }
    }
    else
    {
        if (constrainActivated)
        {
            THROW_NONAME("Compiler", "Type \"" + type + "\" not resolved");
        }
    }

    if (foundImpl)
    {
        front->setImpl(std::move(foundImpl));
    }
}

std::vector<LayerMem> Compiler::resolveImplementation(Builders& builders, NetworkParameters& networkParams)
{
    // it is not possible to instantiate implementations several times
    // there might be duplication of tensor declarations
    if (mImplementationResolved)
    {
        THROW_NONAME("Compiler", "Already resolved");
    }

    std::vector<LayerMem> redefinedFrontLayers;

    for (size_t index = 0; index < builders.size(); ++index)
    {
        mMapLayerNameToBuilderIndex.insert({ builders[index].getName(), index });
    }

    if (!checkConstraints(builders))
    {
        THROW_NONAME("Compiler", "Constraints not correct");
    }

    std::unordered_map<size_t, std::pair<Names, Names>> inOutUnpair;
    std::set<Name> requireTrainableParamsFP32Copies;
    for (size_t index = 0; index < mConstraints.size(); ++index)
    {
        if (isConstraintApplicableForConversion(mConstraints[index]))
        {
            inOutUnpair.emplace(index, getUnpairNames(builders, mConstraints[index]));
        }
        else if (isConstraintRequireFP32CopyOfTrainableParams(mConstraints[index]))
        {
            const auto layerSeq = getConstraintSequenceLayerNames(builders, mConstraints[index]);
            requireTrainableParamsFP32Copies.insert(layerSeq.begin(), layerSeq.end());
        }
    }

    // layer name (from, to) to constraint index
    std::unordered_map<Name, size_t> mapLayerNameToConstraintFrom;
    std::unordered_map<Name, size_t> mapLayerNameToConstraintTo;

    for (size_t index = 0; index < mConstraints.size(); ++index)
    {
        if (isConstraintApplicableForConversion(mConstraints[index]))
        {
            auto& layerFrom = mConstraints[index].getLayerFrom();
            auto& layerTo = mConstraints[index].getLayerTo();

            mapLayerNameToConstraintFrom.insert({ layerFrom, index });

            // constraint with only one layer
            if (layerTo.empty())
            {
                mapLayerNameToConstraintTo.insert({ layerFrom, index });
            }
            else
            {
                mapLayerNameToConstraintTo.insert({ layerTo, index });
            }
        }
    }

    bool constrainActivated = false;
    size_t constrainIndex = 0;

    for (size_t index = 0; index < builders.size(); ++index)
    {
        auto& builder = builders[index];

        Name layerName = builder.getName();
        auto foundItFrom = mapLayerNameToConstraintFrom.find(layerName);
        if (foundItFrom != mapLayerNameToConstraintFrom.end())
        {
            constrainActivated = true;
            constrainIndex = (*foundItFrom).second;

            LayerExecutionTarget toTarget = constraintImplToLayerExecutionTarget(mConstraints[constrainIndex].getConstraintImpl());

            Names inUnpairTensors = inOutUnpair[constrainIndex].first;

            for (size_t q = 0; q < inUnpairTensors.size(); ++q)
            {
                Name convertorName = Name("TensorConvertor_conv") / Conversions::toString(index) / Conversions::toString(q);
                Name toName = inUnpairTensors[q];
                if (mConstraints[constrainIndex].getConstraintImpl() == ConstraintImpl::CPU)
                    toName += "_" + suffixFP32 + "_" + Conversions::toString(constrainIndex);
                else if (mConstraints[constrainIndex].getConstraintImpl() == ConstraintImpl::CPUFP16)
                    toName += "_" + suffixFP16 + "_" + Conversions::toString(constrainIndex);
                else
                    THROW_NONAME("Compiler", "Conversion not supported");

                bool optimizeMem = networkParams.mWorkflow.isTensorOptimizeMem(inUnpairTensors[q]);
                redefinedFrontLayers.emplace_back(std::make_unique<ConvertPrecisionLayer>(
                    convertorName, ConvertPrecisionParams{ { inUnpairTensors[q] }, { toName }, LayerExecutionTarget::Default, toTarget, optimizeMem }, networkParams.mWorkflow.getNetworkParameters()));
            }
        }

        // adjust names of inputs / outputs if conversion applied
        if (constrainActivated)
        {
            std::pair<Names, Names> inOut = inOutUnpair[constrainIndex];

            Names& layerIn = builder.getParams().inputs;
            Names& layerOut = builder.getParams().outputs;

            for (auto& inUnpair : inOut.first)
            {
                auto it = std::find(layerIn.begin(), layerIn.end(), inUnpair);
                if (it != layerIn.end())
                {
                    if (mConstraints[constrainIndex].getConstraintImpl() == ConstraintImpl::CPU)
                        (*it) += "_" + suffixFP32 + "_" + Conversions::toString(constrainIndex);
                    else if (mConstraints[constrainIndex].getConstraintImpl() == ConstraintImpl::CPUFP16)
                        (*it) += "_" + suffixFP16 + "_" + Conversions::toString(constrainIndex);
                    else
                        THROW_NONAME("Compiler", "Conversion not supported");
                }
            }

            if (mConstraints[constrainIndex].isOutputConversion())
            {
                for (auto& outUnpair : inOut.second)
                {
                    auto it = std::find(layerOut.begin(), layerOut.end(), outUnpair);
                    if (it != layerOut.end())
                    {
                        if (mConstraints[constrainIndex].getConstraintImpl() == ConstraintImpl::CPU)
                            (*it) += "_" + suffixFP32 + "_" + Conversions::toString(constrainIndex);
                        else if (mConstraints[constrainIndex].getConstraintImpl() == ConstraintImpl::CPUFP16)
                            (*it) += "_" + suffixFP16 + "_" + Conversions::toString(constrainIndex);
                        else
                            THROW_NONAME("Compiler", "Conversion not supported");
                    }
                }
            }
        }

        if (requireTrainableParamsFP32Copies.find(layerName) != requireTrainableParamsFP32Copies.end())
        {
            // networkParams.mWorkflow.overrideLayerExecutionTarget(constraintImplToLayerExecutionTarget(ConstraintImpl::CPUFP16FP32MasterWeights));
            Name convertorName = Name("TrainableParams_conv") / Conversions::toString(index) / "before";
            redefinedFrontLayers.emplace_back(std::make_unique<TrainableParamsConvertLayer>(
                convertorName, TrainableParamsConvertLayerParams{ layerName, true, LayerExecutionTarget::CPU }, networkParams.mWorkflow.getNetworkParameters()));
        }

        // needed to select proper memory manager in front layers tensors declaration
        if (constrainActivated)
        {
            networkParams.mWorkflow.overrideLayerExecutionTarget(constraintImplToLayerExecutionTarget(mConstraints[constrainIndex].getConstraintImpl()));
        }

        redefinedFrontLayers.emplace_back(builder.build(networkParams));

        if (constrainActivated)
        {
            networkParams.mWorkflow.resetLayerExecutionTargetOverride();
        }

        setImpl(redefinedFrontLayers, index, constrainActivated);

        auto foundItTo = mapLayerNameToConstraintTo.find(layerName);
        if (foundItTo != mapLayerNameToConstraintTo.end())
        {
            constrainActivated = false;

            LayerExecutionTarget fromTarget = constraintImplToLayerExecutionTarget(mConstraints[constrainIndex].getConstraintImpl());

            if (mConstraints[constrainIndex].isOutputConversion())
            {
                Names outUnpairTensors = inOutUnpair[constrainIndex].second;

                for (size_t q = 0; q < outUnpairTensors.size(); ++q)
                {
                    Name convertorName = Name("TensorConvertor_deconv") / Conversions::toString(index) / Conversions::toString(q);
                    Name fromName = outUnpairTensors[q];
                    if (mConstraints[constrainIndex].getConstraintImpl() == ConstraintImpl::CPU)
                        fromName += "_" + suffixFP32 + "_" + Conversions::toString(constrainIndex);
                    else if (mConstraints[constrainIndex].getConstraintImpl() == ConstraintImpl::CPUFP16)
                        fromName += "_" + suffixFP16 + "_" + Conversions::toString(constrainIndex);
                    else
                        THROW_NONAME("Compiler", "Conversion not supported");

                    bool optimizeMem = networkParams.mWorkflow.isTensorOptimizeMem(fromName);
                    redefinedFrontLayers.emplace_back(
                        std::make_unique<ConvertPrecisionLayer>(convertorName,
                                                                ConvertPrecisionParams{ { fromName }, { outUnpairTensors[q] }, fromTarget, LayerExecutionTarget::Default, optimizeMem },
                                                                networkParams.mWorkflow.getNetworkParameters()));
                }
            }
        }

        if (requireTrainableParamsFP32Copies.find(layerName) != requireTrainableParamsFP32Copies.end())
        {
            Name convertorName = Name("TrainableParams_conv") / Conversions::toString(index) / "after";
            redefinedFrontLayers.emplace_back(std::make_unique<TrainableParamsConvertLayer>(
                convertorName, TrainableParamsConvertLayerParams{ layerName, false, LayerExecutionTarget::CPU }, networkParams.mWorkflow.getNetworkParameters()));

            setImpl(redefinedFrontLayers, index);
        }
    }

    if (constrainActivated)
    {
        THROW_NONAME("Compiler", "Error - constraint not deactivated");
    }

    mImplementationResolved = true;

    return redefinedFrontLayers;
}

void Compiler::setConstraint(const Constraint& constraint)
{

    if (mImplementationResolved)
    {
        THROW_NONAME("Compiler", "Implementation already resolved");
    }

    ConstraintImpl globalImpl = executionTargetToConstraintImpl(mGlobalExecutionTarget);

    if (constraint.getConstraintImpl() == globalImpl)
    {
        THROW_NONAME("Compiler", "Redundant constraint");
    }

    if (globalImpl == ConstraintImpl::CPU && constraint.getConstraintImpl() == ConstraintImpl::CPUFP16FP32MasterWeights)
    {
        THROW_NONAME("Compiler", "Not possible combination");
    }

    if (globalImpl == ConstraintImpl::CPUFP16 && constraint.getConstraintImpl() == ConstraintImpl::CPUFP32FP16MixedLocal)
    {
        THROW_NONAME("Compiler", "Not possible combination");
    }

    mConstraints.push_back(constraint);
}

ImplFactory& Compiler::getImplFactory()
{
    return TheImplFactory::Instance();
}

bool Compiler::checkConstraints(const Builders& builders)
{
    bool constraintsCorrect = false;

    ConstraintImpl globalImpl = executionTargetToConstraintImpl(mGlobalExecutionTarget);

    mPerLayerConstraintImpl.clear();
    mPerLayerConstraintImpl.resize(builders.size(), globalImpl);

    std::vector<bool> isDefined(builders.size(), false);

    bool noOverlaps = true;

    for (auto& constraint : mConstraints)
    {
        auto& layerFrom = constraint.getLayerFrom();
        auto& layerTo = constraint.getLayerTo();

        auto foundItFrom = mMapLayerNameToBuilderIndex.find(layerFrom);

        if (foundItFrom == mMapLayerNameToBuilderIndex.end())
        {
            THROW_NONAME("Compiler", layerFrom + " incorrect layerFrom name");
        }

        size_t indexFrom = (*foundItFrom).second;

        // constraint with only one layer
        if (layerTo.empty())
        {
            if (!isDefined[indexFrom])
            {
                isDefined[indexFrom] = true;
                mPerLayerConstraintImpl[indexFrom] = constraint.getConstraintImpl();
            }
            else
            {
                noOverlaps = false;
                break;
            }
        }
        else
        {
            auto foundItTo = mMapLayerNameToBuilderIndex.find(layerTo);
            if (foundItTo == mMapLayerNameToBuilderIndex.end())
            {
                THROW_NONAME("Compiler", layerTo + " incorrect layerTo name");
            }

            size_t indexTo = (*foundItTo).second;

            if (indexTo < indexFrom)
            {
                THROW_NONAME("Compiler", "layers order in constraint not correct");
            }

            for (size_t q = indexFrom; q <= indexTo; ++q)
            {
                if (!isDefined[q])
                {
                    isDefined[q] = true;
                    mPerLayerConstraintImpl[q] = constraint.getConstraintImpl();
                }
                else
                {
                    noOverlaps = false;
                    break;
                }
            }

            if (!noOverlaps)
            {
                break;
            }
        }
    }

    if (noOverlaps)
    {
        constraintsCorrect = true;
    }

    return constraintsCorrect;
}

const ImplFactory::MapImpl& Compiler::getMapImpl(ConstraintImpl cImpl) const
{
    if (cImpl == ConstraintImpl::CPU)
    {
        return TheImplFactory::Instance().getCPUFP32Map();
    }
    else if (cImpl == ConstraintImpl::CPUFP16)
    {
        return TheImplFactory::Instance().getCPUFP16Map();
    }
    else if (cImpl == ConstraintImpl::CPUFP16FP32MasterWeights)
    {
        // Use of master weights do not change implementation
        // So FP16 impl is used when CPUFP16FP32MasterWeights enabled
        return TheImplFactory::Instance().getCPUFP16Map();
    }
    else if (cImpl == ConstraintImpl::CPUFP32FP16MixedLocal)
    {
        return TheImplFactory::Instance().getCPUFP32FP16MixedLocalMap();
    }

    THROW_NONAME("Compiler", "Unknown map type");
}

ConstraintImpl Compiler::executionTargetToConstraintImpl(ExecutionTarget target) const
{
    if (target == ExecutionTarget::CPU)
    {
        return ConstraintImpl::CPU;
    }
    else if (target == ExecutionTarget::CPUFP16)
    {
        return ConstraintImpl::CPUFP16;
    }

    THROW_NONAME("Compiler", "Unknown target type");
}

LayerExecutionTarget Compiler::constraintImplToLayerExecutionTarget(ConstraintImpl cImpl) const
{
    if (cImpl == ConstraintImpl::CPU)
    {
        return LayerExecutionTarget::CPU;
    }
    else if (cImpl == ConstraintImpl::CPUFP16)
    {
        return LayerExecutionTarget::CPUFP16;
    }

    THROW_NONAME("Compiler", "Unknown target type");
}

std::pair<Names, Names> Compiler::getUnpairNames(const Builders& builders, const Constraint& constraint) const
{
    std::pair<Names, Names> ret;

    auto& layerFrom = constraint.getLayerFrom();
    auto& layerTo = constraint.getLayerTo();

    auto foundItFrom = mMapLayerNameToBuilderIndex.find(layerFrom);

    if (foundItFrom == mMapLayerNameToBuilderIndex.end())
    {
        THROW_NONAME("Compiler", layerFrom + " incorrect layerFrom name");
    }

    size_t indexFrom = (*foundItFrom).second;

    // constraint with only one layer
    if (layerTo.empty())
    {
        ret.first = builders[indexFrom].getParams().inputs;
        ret.second = builders[indexFrom].getParams().outputs;
    }
    else
    {
        auto foundItTo = mMapLayerNameToBuilderIndex.find(layerTo);
        if (foundItTo == mMapLayerNameToBuilderIndex.end())
        {
            THROW_NONAME("Compiler", layerTo + " incorrect layerTo name");
        }

        size_t indexTo = (*foundItTo).second;

        std::set<Name> inputs;
        std::set<Name> outputs;

        for (size_t q = indexFrom; q <= indexTo; ++q)
        {
            auto& in = builders[q].getParams().inputs;
            std::copy(in.begin(), in.end(), std::inserter(inputs, inputs.end()));

            auto& out = builders[q].getParams().outputs;
            std::copy(out.begin(), out.end(), std::inserter(outputs, outputs.end()));
        }

        std::set_difference(inputs.begin(), inputs.end(), outputs.begin(), outputs.end(), std::back_inserter(ret.first));

        std::set_difference(outputs.begin(), outputs.end(), inputs.begin(), inputs.end(), std::back_inserter(ret.second));
    }

    return ret;
}

Names Compiler::getConstraintSequenceLayerNames(const Builders& builders, const Constraint& constraint) const
{
    Names layerSequence;

    auto& layerFrom = constraint.getLayerFrom();
    auto& layerTo = constraint.getLayerTo();

    auto foundItFrom = mMapLayerNameToBuilderIndex.find(layerFrom);

    if (foundItFrom == mMapLayerNameToBuilderIndex.end())
    {
        THROW_NONAME("Compiler", layerFrom + " incorrect layerFrom name");
    }

    size_t indexFrom = (*foundItFrom).second;

    // constraint with only one layer
    if (layerTo.empty())
    {
        layerSequence.push_back(layerFrom);
    }
    else
    {
        auto foundItTo = mMapLayerNameToBuilderIndex.find(layerTo);
        if (foundItTo == mMapLayerNameToBuilderIndex.end())
        {
            THROW_NONAME("Compiler", layerTo + " incorrect layerTo name");
        }

        size_t indexTo = (*foundItTo).second;

        for (size_t q = indexFrom; q <= indexTo; ++q)
        {
            layerSequence.push_back(builders[q].getName());
        }
    }

    return layerSequence;
}

bool Compiler::isConstraintApplicableForConversion(const Constraint& constraint) const
{
    // assume all other types no need for conversions
    return constraint.getConstraintImpl() == ConstraintImpl::CPU || constraint.getConstraintImpl() == ConstraintImpl::CPUFP16;
}

bool Compiler::isConstraintRequireFP32CopyOfTrainableParams(const Constraint& constraint) const
{
    // assume all other types no need for conversions
    return constraint.getConstraintImpl() == ConstraintImpl::CPUFP16FP32MasterWeights;
}
} // namespace raul
