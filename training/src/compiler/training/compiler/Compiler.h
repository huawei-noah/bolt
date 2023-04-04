// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef COMPILER_H
#define COMPILER_H

#include <memory>
#include <typeinfo>
#include <unordered_map>
#include <vector>

#include <training/system/Name.h>

#include <training/base/common/Common.h>
#include <training/base/impl/ImplFactory.h>

namespace raul
{

class BasicLayer;
class BasicImpl;
class BasicLayerBuilder;
struct NetworkParameters;

typedef std::unique_ptr<BasicLayer> LayerMem;

enum class ConstraintImpl
{
    CPU,
    CPUFP16,
    CPUFP16FP32MasterWeights, // input/output FP16, master weights FP32
    CPUFP32FP16MixedLocal     // input/output FP32, calculation FP16
};

class Constraint
{
  public:
    /**
     * @brief Define execution target for layer
     */
    Constraint(const Name& layer, ConstraintImpl cImpl);

    /**
     * @brief Define execution target for sequence of layers [layerFrom - layerTo]
     */
    Constraint(const Name& layerFrom, const Name& layerTo, ConstraintImpl cImpl);
    ~Constraint(){}
    [[nodiscard]] const Name& getLayerFrom() const { return mLayerFrom; }
    [[nodiscard]] const Name& getLayerTo() const { return mLayerTo; }

    [[nodiscard]] ConstraintImpl getConstraintImpl() const { return mConstraintImpl; }

    void disableOutputConversion() { mOutputConversion = false; }
    [[nodiscard]] bool isOutputConversion() const { return mOutputConversion; }

  private:
    Name mLayerFrom;
    Name mLayerTo; // inclusive

    bool mOutputConversion;

    ConstraintImpl mConstraintImpl;
};

class Compiler
{
  public:
    explicit Compiler(ExecutionTarget executionTarget);
             ~Compiler(){}
    typedef std::vector<BasicLayerBuilder> Builders;

    std::vector<std::unique_ptr<BasicLayer>> resolveImplementation(Builders& builders, NetworkParameters& networkParams);

    void setConstraint(const Constraint& constraint);

    typedef std::vector<Constraint> vConstraints;

    /**
     * @brief Must be used from user code if registration of other implementations needed
     */
    ImplFactory& getImplFactory();

    bool isResolved() const { return mImplementationResolved; }

  private:
    bool checkConstraints(const Builders& builders);

    const ImplFactory::MapImpl& getMapImpl(ConstraintImpl cImpl) const;

    ConstraintImpl executionTargetToConstraintImpl(ExecutionTarget target) const;
    LayerExecutionTarget constraintImplToLayerExecutionTarget(ConstraintImpl cImpl) const;

    /**
     * @ret Inputs / Outputs pair
     */
    std::pair<Names, Names> getUnpairNames(const Builders& builders, const Constraint& constraint) const;

    /**
     * @ret Sequence of layers with same constraint
     */
    Names getConstraintSequenceLayerNames(const Builders& builders, const Constraint& constraint) const;

    bool isConstraintApplicableForConversion(const Constraint& constraint) const;

    bool isConstraintRequireFP32CopyOfTrainableParams(const Constraint& constraint) const;

    bool mImplementationResolved;

    // if there will be need to change mGlobalExecutionTarget after Compiler constructed
    // make sure all added constraints are removed (otherwise there might be duplications with mGlobalExecutionTarget)
    ExecutionTarget mGlobalExecutionTarget;

    vConstraints mConstraints;

    std::vector<ConstraintImpl> mPerLayerConstraintImpl;

    // layer name to builder index
    std::unordered_map<Name, size_t> mMapLayerNameToBuilderIndex;

    // set implementation
    void setImpl(const std::vector<LayerMem>& redefinedFrontLayers, size_t index, bool constrainActivated = false);
};

} // raul namespace

#endif
