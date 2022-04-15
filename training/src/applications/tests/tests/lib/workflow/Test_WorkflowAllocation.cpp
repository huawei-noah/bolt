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
#include <training/common/MemoryManager.h>
#include <training/network/Workflow.h>
#include <training/optimizers/SGD.h>

#include <tests/tools/TestTools.h>

namespace UT
{
TEST(TestWorkflowAllocation, ConstructorUnit)
{
    PROFILE_TEST

    EXPECT_THROW(raul::Workflow(raul::CompressionMode::INT8, raul::CalculationMode::DETERMINISTIC, raul::AllocationMode::POOL), raul::Exception);
    EXPECT_THROW(raul::Workflow(raul::CompressionMode::FP16, raul::CalculationMode::DETERMINISTIC, raul::AllocationMode::POOL), raul::Exception);
}

TEST(TestWorkflowAllocation, ToyNetUnit)
{
    PROFILE_TEST

    constexpr auto lr = 0.05_dt;
    size_t batch_size = 50U;
    const size_t hidden_size = 500U;
    const size_t amount_of_classes = 10U;
    const size_t mnist_size = 28U;
    const size_t mnist_channels = 1U;
    const size_t print_freq = 100U;

    std::cout << "Loading MNIST..." << std::endl;
    raul::MNIST mnist;
    ASSERT_EQ(mnist.loadingData(tools::getTestAssetsDir() / "MNIST"), true);

    std::cout << "Building network..." << std::endl;
    const auto weights_path = tools::getTestAssetsDir() / "toy_network" / "784_500_10_seed_0";
    auto network = tools::toynet::buildUnique(mnist_size, mnist_size, mnist_channels, batch_size, amount_of_classes, hidden_size, tools::toynet::Nonlinearity::sigmoid, false, weights_path);
    auto networkPooled = tools::toynet::buildUnique(mnist_size, mnist_size, mnist_channels, batch_size, amount_of_classes, hidden_size, tools::toynet::Nonlinearity::sigmoid, true, weights_path);

    // Before train
    {
        std::cout << "Calculating acc...";
        raul::dtype testAcc = mnist.testNetwork(*network);
        raul::dtype testAccPooled = mnist.testNetwork(*networkPooled);
        std::cout << testAcc << std::endl;
        std::cout << testAccPooled << std::endl;
        EXPECT_FLOAT_EQ(testAcc, testAccPooled);
    }
    // Training
    size_t stepsAmountTrain = mnist.getTrainImageAmount() / batch_size;
    auto sgd = std::make_shared<raul::optimizers::SGD>(lr);
    for (size_t q = 0; q < stepsAmountTrain; ++q)
    {
        raul::dtype testLoss = mnist.oneTrainIteration(*network, sgd.get(), q);
        raul::dtype testLossPooled = mnist.oneTrainIteration(*networkPooled, sgd.get(), q);

        CHECK_NEAR(testLoss, testLossPooled, 1e-6);

        if (q % print_freq == 0)
        {
            printf("iteration = %d, loss = %f, lossPooled = %f\n", static_cast<uint32_t>(q), testLoss, testLossPooled);
        }
    }

    // After train
    {
        std::cout << "Calculating acc...";
        raul::dtype testAcc = mnist.testNetwork(*network);
        raul::dtype testAccPooled = mnist.testNetwork(*networkPooled);
        std::cout << testAcc << std::endl;
        std::cout << testAccPooled << std::endl;
        EXPECT_FLOAT_EQ(testAcc, testAccPooled);
    }

    batch_size *= 2u;
    network->setBatchSize(batch_size);
    networkPooled->setBatchSize(batch_size);

    // Training
    stepsAmountTrain = mnist.getTrainImageAmount() / batch_size;
    for (size_t q = 0; q < stepsAmountTrain; ++q)
    {
        raul::dtype testLoss = mnist.oneTrainIteration(*network, sgd.get(), q);
        raul::dtype testLossPooled = mnist.oneTrainIteration(*networkPooled, sgd.get(), q);
        if (q % print_freq == 0)
        {
            CHECK_NEAR(testLoss, testLossPooled, 1e-6);
            printf("iteration = %d, loss = %f, lossPooled = %f\n", static_cast<uint32_t>(q), testLoss, testLossPooled);
        }
    }

    // After train
    {
        std::cout << "Calculating acc...";
        raul::dtype testAcc = mnist.testNetwork(*network);
        raul::dtype testAccPooled = mnist.testNetwork(*networkPooled);
        std::cout << testAcc << std::endl;
        std::cout << testAccPooled << std::endl;
        EXPECT_FLOAT_EQ(testAcc, testAccPooled);
    }
}

TEST(TestWorkflowAllocation, ToyNetCheckpointedUnit)
{
    PROFILE_TEST

    constexpr auto lr = 0.05_dt;
    const size_t batch_size = 50U;
    const size_t hidden_size = 500U;
    const size_t amount_of_classes = 10U;
    const size_t mnist_size = 28U;
    const size_t mnist_channels = 1U;
    const size_t print_freq = 100U;

    std::cout << "Loading MNIST..." << std::endl;
    raul::MNIST mnist;
    ASSERT_EQ(mnist.loadingData(tools::getTestAssetsDir() / "MNIST"), true);

    std::cout << "Building network..." << std::endl;
    const auto weights_path = tools::getTestAssetsDir() / "toy_network" / "784_500_10_seed_0";
    auto network = tools::toynet::buildUnique(mnist_size, mnist_size, mnist_channels, batch_size, amount_of_classes, hidden_size, tools::toynet::Nonlinearity::sigmoid, false, weights_path);
    auto networkPool = tools::toynet::buildUnique(mnist_size, mnist_size, mnist_channels, batch_size, amount_of_classes, hidden_size, tools::toynet::Nonlinearity::sigmoid, true, weights_path);
    auto networkCPPool = tools::toynet::buildUnique(mnist_size, mnist_size, mnist_channels, batch_size, amount_of_classes, hidden_size, tools::toynet::Nonlinearity::sigmoid, true, weights_path);
    networkCPPool->preparePipelines(raul::Workflow::Execution::Checkpointed);

    const raul::Workflow::Pipeline& pipeBackwardTrain = network->getPipeline(raul::Workflow::Pipelines::BackwardTrain);
    const raul::Workflow::Pipeline& pipeBackwardTrainPool = networkPool->getPipeline(raul::Workflow::Pipelines::BackwardTrain);
    const raul::Workflow::Pipeline& pipeBackwardTrainCPPool = networkCPPool->getPipeline(raul::Workflow::Pipelines::BackwardTrain);

    EXPECT_EQ(pipeBackwardTrain.size(), pipeBackwardTrainPool.size());
    EXPECT_NE(pipeBackwardTrain.size(), pipeBackwardTrainCPPool.size());

    // Before train
    {
        std::cout << "Calculating acc...";
        raul::dtype testAcc = mnist.testNetwork(*network);
        raul::dtype testAccPool = mnist.testNetwork(*networkPool);
        raul::dtype testAccCPPool = mnist.testNetwork(*networkCPPool);
        std::cout << testAcc << std::endl;
        std::cout << testAccPool << std::endl;
        std::cout << testAccCPPool << std::endl;
        EXPECT_FLOAT_EQ(testAcc, testAccPool);
        EXPECT_FLOAT_EQ(testAcc, testAccCPPool);
    }
    // Training
    const size_t stepsAmountTrain = mnist.getTrainImageAmount() / batch_size;
    auto sgd = std::make_shared<raul::optimizers::SGD>(lr);
    for (size_t q = 0; q < stepsAmountTrain; ++q)
    {
        float timeBefore = mnist.getTrainingTime();
        raul::dtype testLoss = mnist.oneTrainIteration(*network, sgd.get(), q);
        float trainingTime = mnist.getTrainingTime() - timeBefore;

        timeBefore = mnist.getTrainingTime();
        raul::dtype testLossPool = mnist.oneTrainIteration(*networkPool, sgd.get(), q);
        float trainingTimePool = mnist.getTrainingTime() - timeBefore;

        timeBefore = mnist.getTrainingTime();
        raul::dtype testLossCPPool = mnist.oneTrainIteration(*networkCPPool, sgd.get(), q);
        float trainingTimeCPPool = mnist.getTrainingTime() - timeBefore;

        CHECK_NEAR(testLoss, testLossPool, 1e-6);
        CHECK_NEAR(testLoss, testLossCPPool, 1e-6);

        if (q % print_freq == 0)
        {
            printf("iteration = %d, loss = %f time = %f, lossPool = %f timePool = %f, lossCPPool = %f timeCPPool = %f\n",
                   static_cast<uint32_t>(q),
                   testLoss,
                   trainingTime,
                   testLossPool,
                   trainingTimePool,
                   testLossCPPool,
                   trainingTimeCPPool);
        }
    }

    // After train
    {
        std::cout << "Calculating acc...";
        raul::dtype testAcc = mnist.testNetwork(*network);
        raul::dtype testAccPool = mnist.testNetwork(*networkPool);
        raul::dtype testAccCPPool = mnist.testNetwork(*networkCPPool);
        std::cout << testAcc << std::endl;
        std::cout << testAccPool << std::endl;
        std::cout << testAccCPPool << std::endl;
        EXPECT_FLOAT_EQ(testAcc, testAccPool);
        EXPECT_FLOAT_EQ(testAcc, testAccCPPool);
    }
}
} // UT namespace