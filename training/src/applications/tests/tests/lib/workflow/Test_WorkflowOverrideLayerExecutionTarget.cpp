// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <tests/tools/TestTools.h>

#include <chrono>

#include <training/common/Common.h>
#include <training/common/Conversions.h>
#include <training/common/MemoryManager.h>

#include <training/network/Workflow.h>

#include <training/network/Layers.h>

#include <training/optimizers/SGD.h>

#include <tests/tools/TestTools.h>

namespace {

class TestLayerOverride : public raul::BasicLayer
{
  public:
    TestLayerOverride(const raul::Name& name, const raul::BasicParams& params, std::function<void()> onForward, std::function<void()> onBackward, raul::NetworkParameters& networkParameters)
        : BasicLayer(name, "test", params, networkParameters, { false, false }),
        mOnForward(onForward),
        mOnBackward(onBackward)
    {

        if (mInputs.size() != 1)
        {
            THROW(mTypeName, mName, "wrong number of input names");
        }
        if (mOutputs.size() != 1)
        {
            THROW(mTypeName, mName, "wrong number of output names");
        }

        mNetworkParams.mWorkflow.copyDeclaration(mName, mInputs[0], Workflow::Usage::ForwardAndBackward, Workflow::Mode::Read);
        mNetworkParams.mWorkflow.copyDeclaration(mName, mInputs[0], mInputs[0].grad(), DEC_BACK_WRIT_ZERO);

        mNetworkParams.mWorkflow.tensorNeeded(mName, mOutputs[0], WShape{ BS(), 1u, 1u, 1u}, DEC_FORW_WRIT_COMP);
        mNetworkParams.mWorkflow.copyDeclaration(mName, mOutputs[0], mOutputs[0].grad(), DEC_BACK_READ);

        mLayerExecutionTarget = params.getLayerExecutionTarget();
    }

    void forwardComputeImpl(raul::NetworkMode) override
    {
        mOnForward();

        auto executionTarget = mNetworkParams.mWorkflow.getExecutionTarget();

        if (mLayerExecutionTarget != raul::LayerExecutionTarget::Default)
        {
            executionTarget = static_cast<raul::ExecutionTarget>(mLayerExecutionTarget);
        }

        if (executionTarget == raul::ExecutionTarget::CPU)
        {
            mNetworkParams.mWorkflow.getMemoryManager<MemoryManager>()[mOutputs[0]] = TORANGE(mNetworkParams.mWorkflow.getMemoryManager<MemoryManager>()[mInputs[0]]);
        }
        else if (executionTarget == raul::ExecutionTarget::CPUFP16)
        {
            mNetworkParams.mWorkflow.getMemoryManager<MemoryManagerFP16>()[mOutputs[0]] = TORANGE_FP16(mNetworkParams.mWorkflow.getMemoryManager<MemoryManagerFP16>()[mInputs[0]]);
        }
        else
        {
            THROW(mTypeName, mName, "Target not supported");
        }
    }

    void backwardComputeImpl() override
    {
        mOnBackward();

        auto executionTarget = mNetworkParams.mWorkflow.getExecutionTarget();

        if (mLayerExecutionTarget != raul::LayerExecutionTarget::Default)
        {
            executionTarget = static_cast<raul::ExecutionTarget>(mLayerExecutionTarget);
        }

        if (executionTarget == raul::ExecutionTarget::CPU)
        {
            mNetworkParams.mWorkflow.getMemoryManager<MemoryManager>()[mInputs[0].grad()] = TORANGE(mNetworkParams.mWorkflow.getMemoryManager<MemoryManager>()[mOutputs[0].grad()]);
        }
        else if (executionTarget == raul::ExecutionTarget::CPUFP16)
        {
            mNetworkParams.mWorkflow.getMemoryManager<MemoryManagerFP16>()[mInputs[0].grad()] = TORANGE_FP16(mNetworkParams.mWorkflow.getMemoryManager<MemoryManagerFP16>()[mOutputs[0].grad()]);
        }
        else
        {
            THROW(mTypeName, mName, "Target not supported");
        }
    }

  private:

      std::function<void()> mOnForward;
      std::function<void()> mOnBackward;

      raul::LayerExecutionTarget mLayerExecutionTarget;

};

} //anonymous

namespace UT
{
TEST(TestWorkflowOverrideLayerExecutionTarget, OverrideUnit)
{
    PROFILE_TEST

    {
        Workflow work(CompressionMode::NONE, CalculationMode::DETERMINISTIC, AllocationMode::STANDARD, ExecutionTarget::CPU);

        work.add<raul::DataLayer>("data", raul::DataParams{ { "data"}, 1u, 1u, 1u });
        work.add<TestLayerOverride>("o1", raul::BasicParams{ { "data"}, {"o1"} }, [](){}, [&]()
            {
                ASSERT_EQ(work.getMemoryManager<MemoryManager>()[raul::Name("o1").grad()][0], 33_dt);
            });

        EXPECT_THROW(work.overrideLayerExecutionTarget(raul::LayerExecutionTarget::Default), raul::Exception);

        work.overrideLayerExecutionTarget(raul::LayerExecutionTarget::CPUFP16);
        EXPECT_THROW(work.overrideLayerExecutionTarget(raul::LayerExecutionTarget::Default), raul::Exception);
        EXPECT_THROW(work.overrideLayerExecutionTarget(raul::LayerExecutionTarget::CPU), raul::Exception);
        ASSERT_TRUE(work.getOverrideLayerExecutionTarget() == raul::LayerExecutionTarget::CPUFP16);

        work.add<raul::ConvertPrecisionLayer>("converter", raul::ConvertPrecisionParams{ { "o1"}, { "o1_fp16" }, false });
        work.add<TestLayerOverride>("o2", raul::BasicParams{ { "o1_fp16"}, {"o2_fp16"} }, [](){}, [](){});
        work.add<raul::ConvertPrecisionLayer>("converter2", raul::ConvertPrecisionParams{ { "o2_fp16"}, { "o2"}, true });

        //reset before pipelines preparation
        EXPECT_THROW(work.preparePipelines(), raul::Exception);

        work.resetLayerExecutionTargetOverride();
        ASSERT_TRUE(work.getOverrideLayerExecutionTarget() == raul::LayerExecutionTarget::Default);

        work.add<TestLayerOverride>("o3", raul::BasicParams{ { "o2"}, {"o3"} }, [](){}, [&]()
            {
                work.getMemoryManager<MemoryManager>()[raul::Name("o3").grad()][0] = 33_dt;
                ASSERT_EQ(work.getMemoryManager<MemoryManager>()["o2"][0], 12_dt);
            });

        work.preparePipelines();
        work.setBatchSize(1u);
        work.prepareMemoryForTraining();

        work.getMemoryManager<MemoryManager>()["data"][0] = 12_dt;

        ASSERT_TRUE(work.getMemoryManager<MemoryManager>().tensorExists("data"));
        ASSERT_TRUE(work.getMemoryManager<MemoryManager>().tensorExists("o1"));
        ASSERT_TRUE(work.getMemoryManager<MemoryManagerFP16>().tensorExists("o1_fp16"));
        ASSERT_TRUE(work.getMemoryManager<MemoryManagerFP16>().tensorExists("o2_fp16"));
        ASSERT_TRUE(work.getMemoryManager<MemoryManager>().tensorExists("o2"));
        ASSERT_TRUE(work.getMemoryManager<MemoryManager>().tensorExists("o3"));

        work.forwardPassTraining();
        work.backwardPassTraining();
    }

    {
        Workflow work(CompressionMode::NONE, CalculationMode::DETERMINISTIC, AllocationMode::STANDARD, ExecutionTarget::CPUFP16);

        work.add<raul::DataLayer>("data", raul::DataParams{ { "data"}, 1u, 1u, 1u });
        work.add<TestLayerOverride>("o1", raul::BasicParams{ { "data"}, {"o1"} }, [](){}, [&]()
            {
                ASSERT_EQ(work.getMemoryManager<MemoryManagerFP16>()[raul::Name("o1").grad()][0], 33_hf);
            });

        work.overrideLayerExecutionTarget(raul::LayerExecutionTarget::CPU);
        ASSERT_TRUE(work.getOverrideLayerExecutionTarget() == raul::LayerExecutionTarget::CPU);

        work.add<raul::ConvertPrecisionLayer>("converter", raul::ConvertPrecisionParams{ { "o1"}, { "o1_fp32"}, false });
        work.add<TestLayerOverride>("o2", raul::BasicParams{ { "o1_fp32"}, {"o2_fp32"} }, [](){}, [](){});
        work.add<raul::ConvertPrecisionLayer>("converter2", raul::ConvertPrecisionParams{ { "o2_fp32"}, { "o2"}, true });

        work.resetLayerExecutionTargetOverride();
        ASSERT_TRUE(work.getOverrideLayerExecutionTarget() == raul::LayerExecutionTarget::Default);

        work.add<TestLayerOverride>("o3", raul::BasicParams{ { "o2"}, {"o3"} }, [](){}, [&]()
            {
                work.getMemoryManager<MemoryManagerFP16>()[raul::Name("o3").grad()][0] = 33_hf;
                ASSERT_EQ(work.getMemoryManager<MemoryManagerFP16>()["o2"][0], 12_hf);
            });

        work.preparePipelines();
        work.setBatchSize(1u);
        work.prepareMemoryForTraining();

        work.getMemoryManager<MemoryManagerFP16>()["data"][0] = 12_hf;

        ASSERT_TRUE(work.getMemoryManager<MemoryManagerFP16>().tensorExists("data"));
        ASSERT_TRUE(work.getMemoryManager<MemoryManagerFP16>().tensorExists("o1"));
        ASSERT_TRUE(work.getMemoryManager<MemoryManager>().tensorExists("o1_fp32"));
        ASSERT_TRUE(work.getMemoryManager<MemoryManager>().tensorExists("o2_fp32"));
        ASSERT_TRUE(work.getMemoryManager<MemoryManagerFP16>().tensorExists("o2"));
        ASSERT_TRUE(work.getMemoryManager<MemoryManagerFP16>().tensorExists("o3"));

        work.forwardPassTraining();
        work.backwardPassTraining();
    }
}

#if defined(ANDROID)
TEST(TestWorkflowOverrideLayerExecutionTarget, TrainingOverrideTarget)
{
    PROFILE_TEST

    const raul::dtype LEARNING_RATE = TODTYPE(0.1);
    const size_t BATCH_SIZE = 50;
    const raul::dtype EPSILON_ACCURACY = TODTYPE(2e-1);
    const raul::dtype EPSILON_LOSS = TODTYPE(1e-2);

    const size_t NUM_CLASSES = 10;

    const size_t MNIST_SIZE = 28;
    const size_t FC1_SIZE = 500;
    const size_t FC2_SIZE = 100;

    const raul::dtype acc1 = TODTYPE(3.24f);
    const raul::dtype acc2 = TODTYPE(91.51f);

    raul::MNIST mnist;
    ASSERT_EQ(mnist.loadingData(tools::getTestAssetsDir() / "MNIST"), true);

    raul::Workflow work(raul::CompressionMode::NONE, raul::CalculationMode::DETERMINISTIC, raul::AllocationMode::STANDARD, raul::ExecutionTarget::CPUFP16);
    work.add<raul::DataLayer>("data", raul::DataParams{ { "data", "labels" }, 1, MNIST_SIZE, MNIST_SIZE, NUM_CLASSES });
    work.add<raul::ReshapeLayer>("reshape", raul::ViewParams{ "data", "datar", 1, 1, -1 });
    work.add<raul::LinearLayer>("fc1", raul::LinearParams{ { "datar" }, { "fc1" }, FC1_SIZE });
    work.add<raul::TanhActivation>("tanh", raul::BasicParams{ { "fc1" }, { "tanh" } });
    work.add<raul::LinearLayer>("fc2", raul::LinearParams{ { "tanh" }, { "fc2" }, FC2_SIZE });
    work.overrideLayerExecutionTarget(raul::LayerExecutionTarget::CPU);
    work.add<raul::ConvertPrecisionLayer>("c1", raul::ConvertPrecisionParams{ { "fc2"}, { "fc2_fp32"}, false });
    work.add<raul::SigmoidActivation>("sigmoid", raul::BasicParams{ { "fc2_fp32" }, { "sigmoid_fp32" } });
    work.add<raul::ConvertPrecisionLayer>("c2", raul::ConvertPrecisionParams{ { "sigmoid_fp32"}, { "sigmoid"}, true });
    work.resetLayerExecutionTargetOverride();
    work.add<raul::LinearLayer>("fc3", raul::LinearParams{ { "sigmoid" }, { "fc3" }, NUM_CLASSES });
    work.add<raul::SoftMaxActivation>("softmax", raul::BasicParamsWithDim{ { "fc3" }, { "softmax" } });
    work.add<raul::CrossEntropyLoss>("loss", raul::LossParams{ { "softmax", "labels" }, { "loss" }, "batch_mean" });

    work.preparePipelines();
    work.setBatchSize(BATCH_SIZE);
    work.prepareMemoryForTraining();

    auto& memory_manager = work.getMemoryManager<raul::MemoryManagerFP16>();
    raul::DataLoader dataLoader;

    memory_manager["fc1::Weights"] = dataLoader.loadData(tools::getTestAssetsDir() / "test_fc_layer" / "mnist" / "0_fc1.weight.data", FC1_SIZE, MNIST_SIZE * MNIST_SIZE);
    memory_manager["fc1::Biases"] = dataLoader.loadData(tools::getTestAssetsDir() / "test_fc_layer" / "mnist" / "0_fc1.bias.data", 1, FC1_SIZE);
    memory_manager["fc2::Weights"] = dataLoader.loadData(tools::getTestAssetsDir() / "test_fc_layer" / "mnist" / "0_fc2.weight.data", FC2_SIZE, FC1_SIZE);
    memory_manager["fc2::Biases"] = dataLoader.loadData(tools::getTestAssetsDir() / "test_fc_layer" / "mnist" / "0_fc2.bias.data", 1, FC2_SIZE);
    memory_manager["fc3::Weights"] = dataLoader.loadData(tools::getTestAssetsDir() / "test_fc_layer" / "mnist" / "0_fc3.weight.data", NUM_CLASSES, FC2_SIZE);
    memory_manager["fc3::Biases"] = dataLoader.loadData(tools::getTestAssetsDir() / "test_fc_layer" / "mnist" / "0_fc3.bias.data", 1, NUM_CLASSES);

    raul::Common::transpose(memory_manager["fc1::Weights"], FC1_SIZE);
    raul::Common::transpose(memory_manager["fc2::Weights"], FC2_SIZE);
    raul::Common::transpose(memory_manager["fc3::Weights"], NUM_CLASSES);

    const size_t stepsAmountTrain = mnist.getTrainImageAmount() / BATCH_SIZE;
    raul::Tensor& idealLosses = dataLoader.createTensor(stepsAmountTrain / 100);
    raul::DataLoader::readArrayFromTextFile(tools::getTestAssetsDir() / "test_fc_layer" / "mnist" / "loss.data", idealLosses, 1, idealLosses.size());

    raul::dtype testAcc = mnist.testNetwork(work);
    CHECK_NEAR(testAcc, acc1, EPSILON_ACCURACY);
    printf("Test accuracy = %f\n", testAcc);
    auto sgd = std::make_shared<raul::optimizers::SGD>(LEARNING_RATE);
    for (size_t q = 0, idealLossIndex = 0; q < stepsAmountTrain; ++q)
    {
        raul::dtype testLoss = mnist.oneTrainIteration(work, sgd.get(), q);
        if (q % 100 == 0)
        {
            CHECK_NEAR(testLoss, idealLosses[idealLossIndex++], EPSILON_LOSS);
            printf("iteration = %d, loss = %f\n", static_cast<uint32_t>(q), testLoss);
        }
    }
    testAcc = mnist.testNetwork(work);
    CHECK_NEAR(testAcc, acc2, EPSILON_ACCURACY);

    printf("Test accuracy = %f\n", testAcc);
    printf("Time taken = %.3fs \n", mnist.getTrainingTime());
    printf("Memory taken = %.2fMB \n\n", static_cast<float>(work.getMemoryManager().getTotalMemory()) / (1024.0f * 1024.0f));
}
#endif
} // UT namespace