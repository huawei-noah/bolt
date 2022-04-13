// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <tests/tools/TestTools.h>

#include <training/base/layers/BasicImpl.h>
#include <training/base/layers/BasicLayer.h>
#include <training/base/layers/TrainableLayer.h>
#include <training/base/layers/basic/ConvertPrecisionLayer.h>
#include <training/compiler/Compiler.h>
#include <training/compiler/LayersResolver.h>

namespace
{

class TestLayer : public raul::BasicLayer
{
  public:
    TestLayer(const raul::Name& name, const raul::BasicParams& params, raul::NetworkParameters& networkParameters)
        : BasicLayer(name, "TestLayer", params, networkParameters, { false, false })
    {
        for (auto& input : params.getInputs())
        {
            mNetworkParams.mWorkflow.tensorNeeded(
                name, input, raul::WShape{ raul::BS(), 1u, 1u, 1u }, raul::Workflow::Usage::ForwardAndBackward, raul::Workflow::Mode::Read, true, true, false, false, false);
        }

        for (auto& output : params.getOutputs())
        {
            mNetworkParams.mWorkflow.tensorNeeded(name, output, raul::WShape{ raul::BS(), 1u, 1u, 1u }, raul::Workflow::Usage::Forward, raul::Workflow::Mode::Write, true, true, false, false, false);
        }
    }

    void forwardComputeImpl(raul::NetworkMode) override {}
    void backwardComputeImpl() override {}

    bool isImplResolved() const { return mImpl != nullptr; }
};

class TestImpl : public raul::BasicImpl
{
  public:
    TestImpl(TestLayer&) {}

    void forwardComputeImpl(raul::NetworkMode) override {}
    void backwardComputeImpl() override {}
};

class TestTrainableLayer : public raul::TrainableLayer
{
  public:
    TestTrainableLayer(const raul::Name& name, const raul::TrainableParams& params, raul::NetworkParameters& networkParameters)
        : TrainableLayer(name, "TestTrainableLayer", params, networkParameters, { false, false })
    {
        for (auto& input : params.getInputs())
        {
            mNetworkParams.mWorkflow.tensorNeeded(
                name, input, raul::WShape{ raul::BS(), 1u, 1u, 1u }, raul::Workflow::Usage::ForwardAndBackward, raul::Workflow::Mode::Read, true, true, false, false, false);
        }

        for (auto& output : params.getOutputs())
        {
            mNetworkParams.mWorkflow.tensorNeeded(name, output, raul::WShape{ raul::BS(), 1u, 1u, 1u }, raul::Workflow::Usage::Forward, raul::Workflow::Mode::Write, true, true, false, false, false);
        }

        // Declare trainable params
        mNetworkParams.mWorkflow.tensorNeeded(mName, mWeightsName, WShape{ 1u, 1u, 1u, 1u }, DEC_TRAINABLE);
        mNetworkParams.mWorkflow.tensorNeeded(mName, mBiasesName, WShape{ 1u, 1u, 1u, 1u }, DEC_TRAINABLE);

        if (!isFrozen())
        {
            mNetworkParams.mWorkflow.copyDeclaration(mName, mWeightsName, mWeightsName.grad(), DEC_TRAINABLE_GRAD);
            mNetworkParams.mWorkflow.copyDeclaration(mName, mBiasesName, mBiasesName.grad(), DEC_TRAINABLE_GRAD);
        }
    }

    void forwardComputeImpl(raul::NetworkMode) override {}
    void backwardComputeImpl() override {}

    bool isImplResolved() const { return mImpl != nullptr; }
};

class TestTrainableImpl : public raul::BasicImpl
{
  public:
    TestTrainableImpl(TestTrainableLayer&) {}

    void forwardComputeImpl(raul::NetworkMode) override {}
    void backwardComputeImpl() override {}
};

} // anonymous namespace

namespace UT
{

TEST(TestCompiler, RegistrationUnit)
{
    PROFILE_TEST

    auto& implFactory = Compiler(raul::ExecutionTarget::CPU).getImplFactory(); // factory from library

    size_t mapSizeCPUFP32 = implFactory.getCPUFP32Map().size();
    size_t mapSizeCPUFP16 = implFactory.getCPUFP16Map().size();
    size_t CPUFP32FP16MixedLocal = implFactory.getCPUFP32FP16MixedLocalMap().size();

    implFactory.regCPUFP32<TestLayer, TestImpl>();
    EXPECT_THROW((implFactory.regCPUFP32<TestLayer, TestImpl>()), raul::Exception); // double registration not possibe

    EXPECT_EQ(implFactory.getCPUFP32Map().size(), mapSizeCPUFP32 + 1u);
    EXPECT_EQ(implFactory.getCPUFP16Map().size(), mapSizeCPUFP16);
    EXPECT_EQ(implFactory.getCPUFP32FP16MixedLocalMap().size(), CPUFP32FP16MixedLocal);

    implFactory.clearRegistrationFromEveryMap(typeid(TestLayer).name());

    EXPECT_EQ(implFactory.getCPUFP32Map().size(), mapSizeCPUFP32);
    EXPECT_EQ(implFactory.getCPUFP16Map().size(), mapSizeCPUFP16);
    EXPECT_EQ(implFactory.getCPUFP32FP16MixedLocalMap().size(), CPUFP32FP16MixedLocal);
}

TEST(TestCompiler, ResolveSimpleUnit)
{
    PROFILE_TEST

    auto& implFactory = Compiler(raul::ExecutionTarget::CPU).getImplFactory(); // factory from library

    implFactory.regCPUFP32<TestLayer, TestImpl>();

    std::vector<BasicLayerBuilder> layers;

    raul::Workflow work;

    layers.emplace_back(raul::LayerBuilder<TestLayer, raul::BasicParams>("l1", raul::BasicParams{ {}, {} }));
    layers.emplace_back(raul::LayerBuilder<TestLayer, raul::BasicParams>("l2", raul::BasicParams{ {}, {} }));
    layers.emplace_back(raul::LayerBuilder<TestLayer, raul::BasicParams>("l3", raul::BasicParams{ {}, {} }));
    layers.emplace_back(raul::LayerBuilder<TestLayer, raul::BasicParams>("l4", raul::BasicParams{ {}, {} }));

    Compiler compiler(raul::ExecutionTarget::CPU);
    std::vector<LayerMem> fronts = compiler.resolveImplementation(layers, work.getNetworkParameters());
    EXPECT_THROW(compiler.resolveImplementation(layers, work.getNetworkParameters()), raul::Exception); // not possible to recompile

    EXPECT_EQ(fronts.size(), 4u);

    EXPECT_EQ(static_cast<TestLayer*>(fronts[0].get())->isImplResolved(), true);
    EXPECT_EQ(static_cast<TestLayer*>(fronts[1].get())->isImplResolved(), true);
    EXPECT_EQ(static_cast<TestLayer*>(fronts[2].get())->isImplResolved(), true);
    EXPECT_EQ(static_cast<TestLayer*>(fronts[3].get())->isImplResolved(), true);

    implFactory.clearRegistrationFromEveryMap(typeid(TestLayer).name());
}

TEST(TestCompiler, ResolveAdvancedUnit)
{
    PROFILE_TEST

    auto& implFactory = Compiler(raul::ExecutionTarget::CPU).getImplFactory(); // factory from library

    implFactory.regCPUFP32<TestLayer, TestImpl>();
    implFactory.regCPUFP16<TestLayer, TestImpl>();

    std::vector<BasicLayerBuilder> layers;

    layers.emplace_back(raul::LayerBuilder<TestLayer, raul::BasicParams>("l0", raul::BasicParams{ {}, { "l0" } }));
    layers.emplace_back(raul::LayerBuilder<TestLayer, raul::BasicParams>("l1", raul::BasicParams{ {}, { "l1" } }));
    layers.emplace_back(raul::LayerBuilder<TestLayer, raul::BasicParams>("l2", raul::BasicParams{ { "l1" }, { "l2" } }));
    layers.emplace_back(raul::LayerBuilder<TestLayer, raul::BasicParams>("l3", raul::BasicParams{ { "l0", "l2" }, { "l3", "l5" } }));
    layers.emplace_back(raul::LayerBuilder<TestLayer, raul::BasicParams>("l4", raul::BasicParams{ { "l3" }, { "l4" } }));

    {
        raul::Workflow work;
        Compiler compiler(raul::ExecutionTarget::CPU);
        compiler.setConstraint(raul::Constraint("l2", raul::ConstraintImpl::CPUFP16));
        compiler.setConstraint(raul::Constraint("l3", raul::ConstraintImpl::CPUFP32FP16MixedLocal)); // not used for conversions

        std::vector<BasicLayerBuilder> localLayers = layers;
        std::vector<LayerMem> fronts = compiler.resolveImplementation(localLayers, work.getNetworkParameters());

        EXPECT_EQ(fronts.size(), 7u);

        EXPECT_EQ(fronts[2]->getName(), "TensorConvertor_conv::2::0");
        EXPECT_EQ(fronts[2]->getInputs().size(), 1u);
        EXPECT_EQ(fronts[2]->getOutputs().size(), 1u);
        EXPECT_EQ(fronts[2]->getInputs()[0], "l1");
        EXPECT_EQ(fronts[2]->getOutputs()[0], "l1_FP16_0");

        EXPECT_EQ(fronts[3]->getInputs().size(), 1u);
        EXPECT_EQ(fronts[3]->getOutputs().size(), 1u);
        EXPECT_EQ(fronts[3]->getInputs()[0], "l1_FP16_0");
        EXPECT_EQ(fronts[3]->getOutputs()[0], "l2_FP16_0");

        EXPECT_EQ(fronts[4]->getName(), "TensorConvertor_deconv::2::0");
        EXPECT_EQ(fronts[4]->getInputs().size(), 1u);
        EXPECT_EQ(fronts[4]->getOutputs().size(), 1u);
        EXPECT_EQ(fronts[4]->getInputs()[0], "l2_FP16_0");
        EXPECT_EQ(fronts[4]->getOutputs()[0], "l2");
    }

    // disable output conversion
    {
        raul::Workflow work;
        Compiler compiler(raul::ExecutionTarget::CPU);
        auto constrain = raul::Constraint("l4", raul::ConstraintImpl::CPUFP16);
        constrain.disableOutputConversion();
        compiler.setConstraint(constrain);

        std::vector<BasicLayerBuilder> localLayers = layers;
        std::vector<LayerMem> fronts = compiler.resolveImplementation(localLayers, work.getNetworkParameters());

        EXPECT_EQ(fronts.size(), 6u);

        EXPECT_EQ(fronts[4]->getName(), "TensorConvertor_conv::4::0");
        EXPECT_EQ(fronts[4]->getInputs().size(), 1u);
        EXPECT_EQ(fronts[4]->getOutputs().size(), 1u);
        EXPECT_EQ(fronts[4]->getInputs()[0], "l3");
        EXPECT_EQ(fronts[4]->getOutputs()[0], "l3_FP16_0");

        EXPECT_EQ(fronts[5]->getInputs().size(), 1u);
        EXPECT_EQ(fronts[5]->getOutputs().size(), 1u);
        EXPECT_EQ(fronts[5]->getInputs()[0], "l3_FP16_0");
        EXPECT_EQ(fronts[5]->getOutputs()[0], "l4");
    }

    {
        raul::Workflow work;
        Compiler compiler(raul::ExecutionTarget::CPU);
        compiler.setConstraint(raul::Constraint("l2", "l3", raul::ConstraintImpl::CPUFP16));

        std::vector<BasicLayerBuilder> localLayers = layers;
        std::vector<LayerMem> fronts = compiler.resolveImplementation(localLayers, work.getNetworkParameters());

        EXPECT_EQ(fronts.size(), 9u);

        EXPECT_EQ(fronts[2]->getName(), "TensorConvertor_conv::2::0");
        EXPECT_EQ(fronts[2]->getInputs().size(), 1u);
        EXPECT_EQ(fronts[2]->getOutputs().size(), 1u);
        EXPECT_EQ(fronts[2]->getInputs()[0], "l0");
        EXPECT_EQ(fronts[2]->getOutputs()[0], "l0_FP16_0");

        EXPECT_EQ(fronts[3]->getName(), "TensorConvertor_conv::2::1");
        EXPECT_EQ(fronts[3]->getInputs().size(), 1u);
        EXPECT_EQ(fronts[3]->getOutputs().size(), 1u);
        EXPECT_EQ(fronts[3]->getInputs()[0], "l1");
        EXPECT_EQ(fronts[3]->getOutputs()[0], "l1_FP16_0");

        EXPECT_EQ(fronts[4]->getInputs().size(), 1u);
        EXPECT_EQ(fronts[4]->getOutputs().size(), 1u);
        EXPECT_EQ(fronts[4]->getInputs()[0], "l1_FP16_0");
        EXPECT_EQ(fronts[4]->getOutputs()[0], "l2");

        EXPECT_EQ(fronts[5]->getInputs().size(), 2u);
        EXPECT_EQ(fronts[5]->getOutputs().size(), 2u);
        EXPECT_EQ(fronts[5]->getInputs()[0], "l0_FP16_0");
        EXPECT_EQ(fronts[5]->getInputs()[1], "l2");
        EXPECT_EQ(fronts[5]->getOutputs()[0], "l3_FP16_0");
        EXPECT_EQ(fronts[5]->getOutputs()[1], "l5_FP16_0");

        EXPECT_EQ(fronts[6]->getName(), "TensorConvertor_deconv::3::0");
        EXPECT_EQ(fronts[6]->getInputs().size(), 1u);
        EXPECT_EQ(fronts[6]->getOutputs().size(), 1u);
        EXPECT_EQ(fronts[6]->getInputs()[0], "l3_FP16_0");
        EXPECT_EQ(fronts[6]->getOutputs()[0], "l3");

        EXPECT_EQ(fronts[7]->getName(), "TensorConvertor_deconv::3::1");
        EXPECT_EQ(fronts[7]->getInputs().size(), 1u);
        EXPECT_EQ(fronts[7]->getOutputs().size(), 1u);
        EXPECT_EQ(fronts[7]->getInputs()[0], "l5_FP16_0");
        EXPECT_EQ(fronts[7]->getOutputs()[0], "l5");
    }

    {
        raul::Workflow work;
        Compiler compiler(raul::ExecutionTarget::CPU);
        compiler.setConstraint(raul::Constraint("l0", "l1", raul::ConstraintImpl::CPUFP16));
        compiler.setConstraint(raul::Constraint("l3", "l4", raul::ConstraintImpl::CPUFP16));

        std::vector<BasicLayerBuilder> localLayers = layers;
        std::vector<LayerMem> fronts = compiler.resolveImplementation(localLayers, work.getNetworkParameters());

        EXPECT_EQ(fronts.size(), 11u);

        EXPECT_EQ(fronts[0]->getInputs().size(), 0u);
        EXPECT_EQ(fronts[0]->getOutputs().size(), 1u);
        EXPECT_EQ(fronts[0]->getOutputs()[0], "l0_FP16_0");

        EXPECT_EQ(fronts[1]->getInputs().size(), 0u);
        EXPECT_EQ(fronts[1]->getOutputs().size(), 1u);
        EXPECT_EQ(fronts[1]->getOutputs()[0], "l1_FP16_0");

        EXPECT_EQ(fronts[2]->getName(), "TensorConvertor_deconv::1::0");
        EXPECT_EQ(fronts[2]->getInputs().size(), 1u);
        EXPECT_EQ(fronts[2]->getOutputs().size(), 1u);
        EXPECT_EQ(fronts[2]->getInputs()[0], "l0_FP16_0");
        EXPECT_EQ(fronts[2]->getOutputs()[0], "l0");

        EXPECT_EQ(fronts[3]->getName(), "TensorConvertor_deconv::1::1");
        EXPECT_EQ(fronts[3]->getInputs().size(), 1u);
        EXPECT_EQ(fronts[3]->getOutputs().size(), 1u);
        EXPECT_EQ(fronts[3]->getInputs()[0], "l1_FP16_0");
        EXPECT_EQ(fronts[3]->getOutputs()[0], "l1");

        EXPECT_EQ(fronts[5]->getName(), "TensorConvertor_conv::3::0");
        EXPECT_EQ(fronts[5]->getInputs().size(), 1u);
        EXPECT_EQ(fronts[5]->getOutputs().size(), 1u);
        EXPECT_EQ(fronts[5]->getInputs()[0], "l0");
        EXPECT_EQ(fronts[5]->getOutputs()[0], "l0_FP16_1");

        EXPECT_EQ(fronts[6]->getName(), "TensorConvertor_conv::3::1");
        EXPECT_EQ(fronts[6]->getInputs().size(), 1u);
        EXPECT_EQ(fronts[6]->getOutputs().size(), 1u);
        EXPECT_EQ(fronts[6]->getInputs()[0], "l2");
        EXPECT_EQ(fronts[6]->getOutputs()[0], "l2_FP16_1");

        EXPECT_EQ(fronts[7]->getInputs().size(), 2u);
        EXPECT_EQ(fronts[7]->getOutputs().size(), 2u);
        EXPECT_EQ(fronts[7]->getInputs()[0], "l0_FP16_1");
        EXPECT_EQ(fronts[7]->getInputs()[1], "l2_FP16_1");
        EXPECT_EQ(fronts[7]->getOutputs()[0], "l3");
        EXPECT_EQ(fronts[7]->getOutputs()[1], "l5_FP16_1");

        EXPECT_EQ(fronts[8]->getInputs().size(), 1u);
        EXPECT_EQ(fronts[8]->getOutputs().size(), 1u);
        EXPECT_EQ(fronts[8]->getInputs()[0], "l3");
        EXPECT_EQ(fronts[8]->getOutputs()[0], "l4_FP16_1");

        EXPECT_EQ(fronts[9]->getName(), "TensorConvertor_deconv::4::0");
        EXPECT_EQ(fronts[9]->getInputs().size(), 1u);
        EXPECT_EQ(fronts[9]->getOutputs().size(), 1u);
        EXPECT_EQ(fronts[9]->getInputs()[0], "l4_FP16_1");
        EXPECT_EQ(fronts[9]->getOutputs()[0], "l4");

        EXPECT_EQ(fronts[10]->getName(), "TensorConvertor_deconv::4::1");
        EXPECT_EQ(fronts[10]->getInputs().size(), 1u);
        EXPECT_EQ(fronts[10]->getOutputs().size(), 1u);
        EXPECT_EQ(fronts[10]->getInputs()[0], "l5_FP16_1");
        EXPECT_EQ(fronts[10]->getOutputs()[0], "l5");
    }

    // usage of same input (l0) by two constraints
    {
        raul::Workflow work(CompressionMode::NONE, CalculationMode::DETERMINISTIC, AllocationMode::STANDARD, ExecutionTarget::CPU, true);

        work.add<TestLayer>("l0", raul::BasicParams{ {}, { "l0" } });
        work.add<TestLayer>("l1", raul::BasicParams{ {}, { "l1" } });
        work.add<TestLayer>("l2", raul::BasicParams{ { "l1" }, { "l2" } });
        work.add<TestLayer>("l3", raul::BasicParams{ { "l0", "l2" }, { "l3", "l5" } });
        work.add<TestLayer>("l4", raul::BasicParams{ { "l3" }, { "l4" } });
        work.add<TestLayer>("l5", raul::BasicParams{ { "l0" }, { "l6" } });

        auto& compiler = work.getCompiler();
        compiler.setConstraint(raul::Constraint("l3", raul::ConstraintImpl::CPUFP16));
        compiler.setConstraint(raul::Constraint("l5", raul::ConstraintImpl::CPUFP16));

        EXPECT_NO_THROW(work.preparePipelines());

        EXPECT_EQ(work["TensorConvertor_conv::3::0"]->getInputs().size(), 1u);
        EXPECT_EQ(work["TensorConvertor_conv::3::0"]->getOutputs().size(), 1u);
        EXPECT_EQ(work["TensorConvertor_conv::3::0"]->getInputs()[0], "l0");
        EXPECT_EQ(work["TensorConvertor_conv::3::0"]->getOutputs()[0], "l0_FP16_0");

        EXPECT_EQ(work["TensorConvertor_conv::3::1"]->getInputs().size(), 1u);
        EXPECT_EQ(work["TensorConvertor_conv::3::1"]->getOutputs().size(), 1u);
        EXPECT_EQ(work["TensorConvertor_conv::3::1"]->getInputs()[0], "l2");
        EXPECT_EQ(work["TensorConvertor_conv::3::1"]->getOutputs()[0], "l2_FP16_0");

        EXPECT_EQ(work["l3"]->getInputs().size(), 2u);
        EXPECT_EQ(work["l3"]->getOutputs().size(), 2u);
        EXPECT_EQ(work["l3"]->getInputs()[0], "l0_FP16_0");
        EXPECT_EQ(work["l3"]->getInputs()[1], "l2_FP16_0");
        EXPECT_EQ(work["l3"]->getOutputs()[0], "l3_FP16_0");
        EXPECT_EQ(work["l3"]->getOutputs()[1], "l5_FP16_0");

        EXPECT_EQ(work["TensorConvertor_deconv::3::0"]->getInputs().size(), 1u);
        EXPECT_EQ(work["TensorConvertor_deconv::3::0"]->getOutputs().size(), 1u);
        EXPECT_EQ(work["TensorConvertor_deconv::3::0"]->getInputs()[0], "l3_FP16_0");
        EXPECT_EQ(work["TensorConvertor_deconv::3::0"]->getOutputs()[0], "l3");

        EXPECT_EQ(work["TensorConvertor_deconv::3::1"]->getInputs().size(), 1u);
        EXPECT_EQ(work["TensorConvertor_deconv::3::1"]->getOutputs().size(), 1u);
        EXPECT_EQ(work["TensorConvertor_deconv::3::1"]->getInputs()[0], "l5_FP16_0");
        EXPECT_EQ(work["TensorConvertor_deconv::3::1"]->getOutputs()[0], "l5");

        EXPECT_EQ(work["TensorConvertor_conv::5::0"]->getInputs().size(), 1u);
        EXPECT_EQ(work["TensorConvertor_conv::5::0"]->getOutputs().size(), 1u);
        EXPECT_EQ(work["TensorConvertor_conv::5::0"]->getInputs()[0], "l0");
        EXPECT_EQ(work["TensorConvertor_conv::5::0"]->getOutputs()[0], "l0_FP16_1");

        EXPECT_EQ(work["l5"]->getInputs().size(), 1u);
        EXPECT_EQ(work["l5"]->getOutputs().size(), 1u);
        EXPECT_EQ(work["l5"]->getInputs()[0], "l0_FP16_1");
        EXPECT_EQ(work["l5"]->getOutputs()[0], "l6_FP16_1");

        EXPECT_EQ(work["TensorConvertor_deconv::5::0"]->getInputs().size(), 1u);
        EXPECT_EQ(work["TensorConvertor_deconv::5::0"]->getOutputs().size(), 1u);
        EXPECT_EQ(work["TensorConvertor_deconv::5::0"]->getInputs()[0], "l6_FP16_1");
        EXPECT_EQ(work["TensorConvertor_deconv::5::0"]->getOutputs()[0], "l6");
    }

    // direct usage of conversions
    {
        raul::Workflow work(CompressionMode::NONE, CalculationMode::DETERMINISTIC, AllocationMode::STANDARD, ExecutionTarget::CPU, true);

        EXPECT_THROW(work.add<raul::ConvertPrecisionLayer>("c1", raul::ConvertPrecisionParams{ { "l1" }, { "l2" }, LayerExecutionTarget::CPU, LayerExecutionTarget::CPUFP16, false }), raul::Exception);
    }

    implFactory.clearRegistrationFromEveryMap(typeid(TestLayer).name());
}

TEST(TestCompiler, ConstraintUnit)
{
    PROFILE_TEST

    auto& implFactory = Compiler(raul::ExecutionTarget::CPU).getImplFactory(); // factory from library

    implFactory.regCPUFP32<TestLayer, TestImpl>();
    implFactory.regCPUFP16<TestLayer, TestImpl>();

    std::vector<BasicLayerBuilder> layers;

    raul::Workflow work;

    layers.emplace_back(raul::LayerBuilder<TestLayer, raul::BasicParams>("l1", raul::BasicParams{ {}, {} }));
    layers.emplace_back(raul::LayerBuilder<TestLayer, raul::BasicParams>("l2", raul::BasicParams{ {}, {} }));
    layers.emplace_back(raul::LayerBuilder<TestLayer, raul::BasicParams>("l3", raul::BasicParams{ {}, {} }));
    layers.emplace_back(raul::LayerBuilder<TestLayer, raul::BasicParams>("l4", raul::BasicParams{ {}, {} }));

    {
        Compiler compiler(raul::ExecutionTarget::CPU);
        EXPECT_THROW(compiler.setConstraint(raul::Constraint("", raul::ConstraintImpl::CPUFP16)), raul::Exception);
        EXPECT_THROW(compiler.setConstraint(raul::Constraint("", "", raul::ConstraintImpl::CPUFP16)), raul::Exception);
    }

    {
        Compiler compiler(raul::ExecutionTarget::CPU);
        compiler.setConstraint(raul::Constraint("l2", raul::ConstraintImpl::CPUFP16));

        EXPECT_NO_THROW(compiler.resolveImplementation(layers, work.getNetworkParameters()));
    }

    {
        Compiler compiler(raul::ExecutionTarget::CPU);
        compiler.setConstraint(raul::Constraint("l2", "l4", raul::ConstraintImpl::CPUFP16));
        EXPECT_NO_THROW(compiler.resolveImplementation(layers, work.getNetworkParameters()));
    }

    {
        Compiler compiler(raul::ExecutionTarget::CPU);
        compiler.setConstraint(raul::Constraint("l1", "l5", raul::ConstraintImpl::CPUFP16)); // wrong to name
        EXPECT_THROW(compiler.resolveImplementation(layers, work.getNetworkParameters()), raul::Exception);
    }

    {
        Compiler compiler(raul::ExecutionTarget::CPU);
        compiler.setConstraint(raul::Constraint("lA", "l4", raul::ConstraintImpl::CPUFP16)); // wrong from name
        EXPECT_THROW(compiler.resolveImplementation(layers, work.getNetworkParameters()), raul::Exception);
    }

    {
        Compiler compiler(raul::ExecutionTarget::CPU);
        compiler.setConstraint(raul::Constraint("l2", "l4", raul::ConstraintImpl::CPUFP16));
        compiler.setConstraint(raul::Constraint("l3", raul::ConstraintImpl::CPUFP16)); // overlap
        EXPECT_THROW(compiler.resolveImplementation(layers, work.getNetworkParameters()), raul::Exception);
    }

    {
        Compiler compiler(raul::ExecutionTarget::CPU);
        compiler.setConstraint(raul::Constraint("l2", "l4", raul::ConstraintImpl::CPUFP16));
        compiler.setConstraint(raul::Constraint("l2", raul::ConstraintImpl::CPUFP16)); // overlap
        EXPECT_THROW(compiler.resolveImplementation(layers, work.getNetworkParameters()), raul::Exception);
    }

    {
        Compiler compiler(raul::ExecutionTarget::CPU);
        compiler.setConstraint(raul::Constraint("l2", "l4", raul::ConstraintImpl::CPUFP16));
        compiler.setConstraint(raul::Constraint("l4", raul::ConstraintImpl::CPUFP16)); // overlap
        EXPECT_THROW(compiler.resolveImplementation(layers, work.getNetworkParameters()), raul::Exception);
    }

    {
        Compiler compiler(raul::ExecutionTarget::CPU);
        compiler.setConstraint(raul::Constraint("l2", "l1", raul::ConstraintImpl::CPUFP16)); // wrong order
        compiler.setConstraint(raul::Constraint("l3", raul::ConstraintImpl::CPUFP16));
        EXPECT_THROW(compiler.resolveImplementation(layers, work.getNetworkParameters()), raul::Exception);
    }

    {
        Compiler compiler(raul::ExecutionTarget::CPU);
        compiler.setConstraint(raul::Constraint("l2", "l2", raul::ConstraintImpl::CPUFP16));
        compiler.setConstraint(raul::Constraint("l3", raul::ConstraintImpl::CPUFP16));
        EXPECT_NO_THROW(compiler.resolveImplementation(layers, work.getNetworkParameters()));
    }

    {
        Compiler compiler(raul::ExecutionTarget::CPU);
        compiler.setConstraint(raul::Constraint("l2", "l3", raul::ConstraintImpl::CPUFP16));
        compiler.setConstraint(raul::Constraint("l1", raul::ConstraintImpl::CPUFP16));
        compiler.setConstraint(raul::Constraint("l4", raul::ConstraintImpl::CPUFP16));
        EXPECT_NO_THROW(compiler.resolveImplementation(layers, work.getNetworkParameters()));
    }

    {
        Compiler compiler(raul::ExecutionTarget::CPU);
        compiler.setConstraint(raul::Constraint("l1", "l2", raul::ConstraintImpl::CPUFP16));
        compiler.setConstraint(raul::Constraint("l3", "l4", raul::ConstraintImpl::CPUFP16));
        EXPECT_NO_THROW(compiler.resolveImplementation(layers, work.getNetworkParameters()));
    }

    {
        Compiler compiler(raul::ExecutionTarget::CPU);
        compiler.setConstraint(raul::Constraint("l1", "l2", raul::ConstraintImpl::CPUFP16));
        compiler.setConstraint(raul::Constraint("l3", "l4", raul::ConstraintImpl::CPUFP16));
        compiler.setConstraint(raul::Constraint("l2", raul::ConstraintImpl::CPUFP16)); // overlap
        EXPECT_THROW(compiler.resolveImplementation(layers, work.getNetworkParameters()), raul::Exception);
    }

    {
        Compiler compiler(raul::ExecutionTarget::CPU);
        compiler.setConstraint(raul::Constraint("l1", "l2", raul::ConstraintImpl::CPUFP16));
        compiler.resolveImplementation(layers, work.getNetworkParameters());
        EXPECT_THROW(compiler.setConstraint(raul::Constraint("l3", "l4", raul::ConstraintImpl::CPUFP16)), raul::Exception); // already resolved
    }

    implFactory.clearRegistrationFromEveryMap(typeid(TestLayer).name());
}

TEST(TestCompiler, ConstraintCombinationUnit)
{
    PROFILE_TEST

    // redundant
    {
        Compiler compiler(raul::ExecutionTarget::CPU);
        EXPECT_THROW(compiler.setConstraint(raul::Constraint("l1", "l2", raul::ConstraintImpl::CPU)), raul::Exception);
    }

    // redundant2
    {
        Compiler compiler(raul::ExecutionTarget::CPUFP16);
        EXPECT_THROW(compiler.setConstraint(raul::Constraint("l1", "l2", raul::ConstraintImpl::CPUFP16)), raul::Exception);
    }

    {
        Compiler compiler(raul::ExecutionTarget::CPUFP16);
        EXPECT_THROW(compiler.setConstraint(raul::Constraint("l1", "l2", raul::ConstraintImpl::CPUFP32FP16MixedLocal)), raul::Exception);
    }

    {
        Compiler compiler(raul::ExecutionTarget::CPU);
        EXPECT_THROW(compiler.setConstraint(raul::Constraint("l1", "l2", raul::ConstraintImpl::CPUFP16FP32MasterWeights)), raul::Exception);
    }
}

TEST(TestCompiler, LinearUnit)
{
    using namespace frontend;
    auto topology = Graph{ { "l1", Linear{ 1 } }, { "l2", Linear{ 2 } }, { "l3", Linear{ 3 } }, { "l4", Linear{ 4 } } };

    std::vector<BasicLayerBuilder> layers;
    raul::Workflow work;

    auto resolver = LayersResolver(layers);
    topology.apply(resolver);
    resolver.resolveInputs();

    Compiler compiler(raul::ExecutionTarget::CPU);
    std::vector<LayerMem> fronts = compiler.resolveImplementation(layers, work.getNetworkParameters());
    EXPECT_EQ(fronts.size(), 5u);
}

TEST(TestCompiler, AutoMasterWeightsConstraintForTrainableLayerUnit)
{
    PROFILE_TEST

    auto& implFactory = Compiler(raul::ExecutionTarget::CPUFP16).getImplFactory(); // factory from library

    // Register one implementation for two situations
    implFactory.regCPUFP16<TestTrainableLayer, TestTrainableImpl>();
    implFactory.regCPUFP32<TestTrainableLayer, TestTrainableImpl>();

    std::vector<BasicLayerBuilder> layers;

    layers.emplace_back(raul::LayerBuilder<TestTrainableLayer, raul::TrainableParams>("l1", raul::TrainableParams{ raul::Names{}, {} }));
    layers.emplace_back(raul::LayerBuilder<TestTrainableLayer, raul::TrainableParams>("l2", raul::TrainableParams{ raul::Names{}, {} }));
    layers.emplace_back(raul::LayerBuilder<TestTrainableLayer, raul::TrainableParams>("l3", raul::TrainableParams{ raul::Names{}, {} }));
    layers.emplace_back(raul::LayerBuilder<TestTrainableLayer, raul::TrainableParams>("l4", raul::TrainableParams{ raul::Names{}, {} }));
    // Sharing
    layers.emplace_back(raul::LayerBuilder<TestTrainableLayer, raul::TrainableParams>("l5", raul::TrainableParams{ raul::Names{}, {}, "l1", false }));

    {
        raul::Workflow work(raul::CompressionMode::NONE, raul::CalculationMode::DETERMINISTIC, raul::AllocationMode::STANDARD, raul::ExecutionTarget::CPUFP16, true);
        Compiler compiler(raul::ExecutionTarget::CPUFP16);
        EXPECT_THROW(compiler.setConstraint(raul::Constraint("l1", raul::ConstraintImpl::CPUFP16)), raul::Exception);
        EXPECT_THROW(compiler.setConstraint(raul::Constraint("l1", "l4", raul::ConstraintImpl::CPUFP16)), raul::Exception);
    }

    {
        raul::Workflow work(raul::CompressionMode::NONE, raul::CalculationMode::DETERMINISTIC, raul::AllocationMode::STANDARD, raul::ExecutionTarget::CPUFP16, true);
        Compiler compiler(raul::ExecutionTarget::CPUFP16);
        EXPECT_THROW(compiler.setConstraint(raul::Constraint("l1", raul::ConstraintImpl::CPUFP32FP16MixedLocal)), raul::Exception);
        EXPECT_THROW(compiler.setConstraint(raul::Constraint("l1", "l4", raul::ConstraintImpl::CPUFP32FP16MixedLocal)), raul::Exception);
    }

    // No additional weights expected - 8 expected
    {
        raul::Workflow work(raul::CompressionMode::NONE, raul::CalculationMode::DETERMINISTIC, raul::AllocationMode::STANDARD, raul::ExecutionTarget::CPUFP16, true);
        Compiler compiler(raul::ExecutionTarget::CPUFP16);
        EXPECT_NO_THROW(compiler.setConstraint(raul::Constraint("l1", raul::ConstraintImpl::CPU)));
        EXPECT_NO_THROW(compiler.setConstraint(raul::Constraint("l3", "l4", raul::ConstraintImpl::CPU)));
        EXPECT_NO_THROW(compiler.resolveImplementation(layers, work.getNetworkParameters()));
        EXPECT_EQ(work.getLayerTrainableParameterNames("l1").size() + work.getLayerTrainableParameterNames("l2").size() + work.getLayerTrainableParameterNames("l3").size() +
                      work.getLayerTrainableParameterNames("l4").size(),
                  8);
    }

    // Additional weights expected - 4 * 2 initially + 3 * 2 copies
    {
        raul::Workflow work(raul::CompressionMode::NONE, raul::CalculationMode::DETERMINISTIC, raul::AllocationMode::STANDARD, raul::ExecutionTarget::CPUFP16, true);
        Compiler compiler(raul::ExecutionTarget::CPUFP16);
        EXPECT_NO_THROW(compiler.setConstraint(raul::Constraint("l1", raul::ConstraintImpl::CPUFP16FP32MasterWeights)));
        EXPECT_NO_THROW(compiler.setConstraint(raul::Constraint("l3", "l4", raul::ConstraintImpl::CPUFP16FP32MasterWeights)));
        EXPECT_NO_THROW(compiler.resolveImplementation(layers, work.getNetworkParameters()));

        // Calculate unique parameters
        std::unordered_set<raul::Name> uniqueTrainableParameterNames;
        for (size_t i = 0; i < layers.size(); ++i)
        {
            const auto trainableNames = work.getLayerTrainableParameterNames(layers[i].getName());
            uniqueTrainableParameterNames.insert(trainableNames.begin(), trainableNames.end());
        }
        EXPECT_EQ(uniqueTrainableParameterNames.size(), 14);
    }

    // No change from previous case due to sharing
    {
        raul::Workflow work(raul::CompressionMode::NONE, raul::CalculationMode::DETERMINISTIC, raul::AllocationMode::STANDARD, raul::ExecutionTarget::CPUFP16, true);
        Compiler compiler(raul::ExecutionTarget::CPUFP16);
        EXPECT_NO_THROW(compiler.setConstraint(raul::Constraint("l1", raul::ConstraintImpl::CPUFP16FP32MasterWeights)));
        EXPECT_NO_THROW(compiler.setConstraint(raul::Constraint("l3", "l4", raul::ConstraintImpl::CPUFP16FP32MasterWeights)));
        // Set constraint to shared layer - no additional copies required
        EXPECT_NO_THROW(compiler.setConstraint(raul::Constraint("l5", raul::ConstraintImpl::CPUFP16FP32MasterWeights)));
        EXPECT_NO_THROW(compiler.resolveImplementation(layers, work.getNetworkParameters()));

        // Calculate unique parameters
        std::unordered_set<raul::Name> uniqueTrainableParameterNames;
        for (size_t i = 0; i < layers.size(); ++i)
        {
            const auto trainableNames = work.getLayerTrainableParameterNames(layers[i].getName());
            uniqueTrainableParameterNames.insert(trainableNames.begin(), trainableNames.end());
        }
        EXPECT_EQ(uniqueTrainableParameterNames.size(), 14);
    }

    implFactory.clearRegistrationFromEveryMap(typeid(TestTrainableLayer).name());
}

} // UT namespace