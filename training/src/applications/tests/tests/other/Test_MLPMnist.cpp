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
#include <cstdio>
#include <tests/tools/TestTools.h>

#include <training/api/API.h>
#include <training/common/Common.h>
#include <training/common/Conversions.h>
#include <training/common/DataLoader.h>
#include <training/common/MemoryManager.h>
#include <training/layers/BasicLayer.h>
#include <training/layers/parameters/LayerParameters.h>
#include <training/network/Layers.h>
#include <training/network/Workflow.h>
#include <training/optimizers/SGD.h>
#include <training/tools/Datasets.h>

namespace UT
{

TEST(TestMLPMnist, Training)
{
    PROFILE_TEST

    const raul::dtype LEARNING_RATE = TODTYPE(0.1);
    const size_t BATCH_SIZE = 50;
    const raul::dtype EPSILON_ACCURACY = TODTYPE(1e-2);
    const raul::dtype EPSILON_LOSS = TODTYPE(1e-5);

    const size_t NUM_CLASSES = 10;

    const size_t MNIST_SIZE = 28;
    const size_t FC1_SIZE = 500;
    const size_t FC2_SIZE = 100;

    const raul::dtype acc1 = TODTYPE(3.24f);
    const raul::dtype acc2 = TODTYPE(91.51f);

    raul::MNIST mnist;
    ASSERT_EQ(mnist.loadingData(tools::getTestAssetsDir() / "MNIST"), true);

    raul::Workflow work;
    work.add<raul::DataLayer>("data", raul::DataParams{ { "data", "labels" }, 1, MNIST_SIZE, MNIST_SIZE, NUM_CLASSES });
    work.add<raul::ReshapeLayer>("reshape", raul::ViewParams{ "data", "datar", 1, 1, -1 });
    work.add<raul::LinearLayer>("fc1", raul::LinearParams{ { "datar" }, { "fc1" }, FC1_SIZE });
    work.add<raul::TanhActivation>("tanh", raul::BasicParams{ { "fc1" }, { "tanh" } });
    work.add<raul::LinearLayer>("fc2", raul::LinearParams{ { "tanh" }, { "fc2" }, FC2_SIZE });
    work.add<raul::SigmoidActivation>("sigmoid", raul::BasicParams{ { "fc2" }, { "sigmoid" } });
    work.add<raul::LinearLayer>("fc3", raul::LinearParams{ { "sigmoid" }, { "fc3" }, NUM_CLASSES });
    work.add<raul::SoftMaxActivation>("softmax", raul::BasicParamsWithDim{ { "fc3" }, { "softmax" } });
    work.add<raul::CrossEntropyLoss>("loss", raul::LossParams{ { "softmax", "labels" }, { "loss" }, "batch_mean" });

    work.preparePipelines();
    work.setBatchSize(BATCH_SIZE);
    work.prepareMemoryForTraining();

    auto& memory_manager = work.getMemoryManager();
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

TEST(TestMLPMnist, GpuTraining)
{
    PROFILE_TEST

    GPU_ONLY_TEST

    const raul::dtype LEARNING_RATE = TODTYPE(0.1);
    const size_t BATCH_SIZE = 50;
    const raul::dtype EPSILON_ACCURACY = TODTYPE(2e-2);
    const raul::dtype EPSILON_LOSS = TODTYPE(1e-4);

    const size_t NUM_CLASSES = 10;

    const size_t MNIST_SIZE = 28;
    const size_t FC1_SIZE = 500;
    const size_t FC2_SIZE = 100;

    const raul::dtype acc1 = TODTYPE(3.24f);
    const raul::dtype acc2 = TODTYPE(91.51f);

    raul::MNIST mnist;
    ASSERT_EQ(mnist.loadingData(tools::getTestAssetsDir() / "MNIST"), true);

    raul::Workflow work(raul::CompressionMode::NONE, raul::CalculationMode::DETERMINISTIC, raul::AllocationMode::STANDARD, raul::ExecutionTarget::GPU);
    work.add<raul::DataLayer>("data", raul::DataParams{ { "data", "labels" }, 1, MNIST_SIZE, MNIST_SIZE, NUM_CLASSES });
    work.add<raul::ReshapeLayer>("reshape", raul::ViewParams{ "data", "datar", 1, 1, -1 });
    work.add<raul::LinearLayer>("fc1", raul::LinearParams{ { "datar" }, { "fc1" }, FC1_SIZE });
    work.add<raul::TanhActivation>("tanh", raul::BasicParams{ { "fc1" }, { "tanh" } });
    work.add<raul::LinearLayer>("fc2", raul::LinearParams{ { "tanh" }, { "fc2" }, FC2_SIZE });
    work.add<raul::SigmoidActivation>("sigmoid", raul::BasicParams{ { "fc2" }, { "sigmoid" } });
    work.add<raul::LinearLayer>("fc3", raul::LinearParams{ { "sigmoid" }, { "fc3" }, NUM_CLASSES });
    work.add<raul::SoftMaxActivation>("softmax", raul::BasicParamsWithDim{ { "fc3" }, { "softmax" } });
    work.add<raul::CrossEntropyLoss>("loss", raul::LossParams{ { "softmax", "labels" }, { "loss" }, "batch_mean" });

    work.preparePipelines();
    work.setBatchSize(BATCH_SIZE);
    work.prepareMemoryForTraining();

    raul::MemoryManagerGPU& memory_manager = work.getMemoryManager<raul::MemoryManagerGPU>();
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
}

TEST(TestMLPMnist, PoolGpuTraining)
{
    PROFILE_TEST

    GPU_ONLY_TEST

    const raul::dtype LEARNING_RATE = TODTYPE(0.1);
    const size_t BATCH_SIZE = 50;
    const raul::dtype EPSILON_ACCURACY = TODTYPE(2e-2);
    const raul::dtype EPSILON_LOSS = TODTYPE(1e-4);

    const size_t NUM_CLASSES = 10;

    const size_t MNIST_SIZE = 28;
    const size_t FC1_SIZE = 500;
    const size_t FC2_SIZE = 100;

    const raul::dtype acc1 = TODTYPE(3.24f);
    const raul::dtype acc2 = TODTYPE(91.51f);

    raul::MNIST mnist;
    ASSERT_EQ(mnist.loadingData(tools::getTestAssetsDir() / "MNIST"), true);

    raul::Workflow work(raul::CompressionMode::NONE, raul::CalculationMode::DETERMINISTIC, raul::AllocationMode::POOL, raul::ExecutionTarget::GPU);
    work.add<raul::DataLayer>("data", raul::DataParams{ { "data", "labels" }, 1, MNIST_SIZE, MNIST_SIZE, NUM_CLASSES });
    work.add<raul::ReshapeLayer>("reshape", raul::ViewParams{ "data", "datar", 1, 1, -1 });
    work.add<raul::LinearLayer>("fc1", raul::LinearParams{ { "datar" }, { "fc1" }, FC1_SIZE });
    work.add<raul::TanhActivation>("tanh", raul::BasicParams{ { "fc1" }, { "tanh" } });
    work.add<raul::LinearLayer>("fc2", raul::LinearParams{ { "tanh" }, { "fc2" }, FC2_SIZE });
    work.add<raul::SigmoidActivation>("sigmoid", raul::BasicParams{ { "fc2" }, { "sigmoid" } });
    work.add<raul::LinearLayer>("fc3", raul::LinearParams{ { "sigmoid" }, { "fc3" }, NUM_CLASSES });
    work.add<raul::SoftMaxActivation>("softmax", raul::BasicParamsWithDim{ { "fc3" }, { "softmax" } });
    work.add<raul::CrossEntropyLoss>("loss", raul::LossParams{ { "softmax", "labels" }, { "loss" }, "batch_mean" });

    work.preparePipelines();
    work.setBatchSize(BATCH_SIZE);
    work.prepareMemoryForTraining();

    raul::MemoryManagerGPU& memory_manager = work.getMemoryManager<raul::MemoryManagerGPU>();
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
}

#if defined(ANDROID)
TEST(TestMLPMnist, TrainingFP16)
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
    work.add<raul::SigmoidActivation>("sigmoid", raul::BasicParams{ { "fc2" }, { "sigmoid" } });
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

TEST(TestMLPMnist, TrainingMixedPrecision)
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

    raul::Workflow work(raul::CompressionMode::NONE, raul::CalculationMode::DETERMINISTIC, raul::AllocationMode::STANDARD, raul::ExecutionTarget::CPU);
    work.add<raul::DataLayer>("data", raul::DataParams{ { "data", "labels" }, 1, MNIST_SIZE, MNIST_SIZE, NUM_CLASSES });
    work.add<raul::ReshapeLayer>("reshape", raul::ViewParams{ "data", "datar", 1, 1, -1 });
    work.add<raul::LinearLayer>("fc1", raul::LinearParams{ { "datar" }, { "fc1" }, FC1_SIZE, raul::LayerExecutionTarget::CPUFP16 });
    work.add<raul::TanhActivation>("tanh", raul::BasicParams{ { "fc1" }, { "tanh" } });
    work.add<raul::LinearLayer>("fc2", raul::LinearParams{ { "tanh" }, { "fc2" }, FC2_SIZE, raul::LayerExecutionTarget::CPUFP16 });
    work.add<raul::SigmoidActivation>("sigmoid", raul::BasicParams{ { "fc2" }, { "sigmoid" } });
    work.add<raul::LinearLayer>("fc3", raul::LinearParams{ { "sigmoid" }, { "fc3" }, NUM_CLASSES, raul::LayerExecutionTarget::CPUFP16 });
    work.add<raul::SoftMaxActivation>("softmax", raul::BasicParamsWithDim{ { "fc3" }, { "softmax" } });
    work.add<raul::CrossEntropyLoss>("loss", raul::LossParams{ { "softmax", "labels" }, { "loss" }, "batch_mean" });

    work.preparePipelines();
    work.setBatchSize(BATCH_SIZE);
    work.prepareMemoryForTraining();

    auto& memory_manager = work.getMemoryManager();
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
        raul::dtype testLoss = mnist.oneTrainIterationMixedPrecision(work, sgd.get(), q);
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

TEST(TestMLPMnist, TrainingFP16PrecisionOverride)
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
    work.overrideLayerExecutionTarget(raul::LayerExecutionTarget::CPU);
    work.add<raul::ConvertPrecisionLayer>("c1", raul::ConvertPrecisionParams{ { "datar"}, { "datar_fp32"}, false });
    work.add<raul::LinearLayer>("fc1", raul::LinearParams{ { "datar_fp32" }, { "fc1_fp32" }, FC1_SIZE });
    work.add<raul::ConvertPrecisionLayer>("c2", raul::ConvertPrecisionParams{ { "fc1_fp32"}, { "fc1"}, true });
    work.resetLayerExecutionTargetOverride();
    work.add<raul::TanhActivation>("tanh", raul::BasicParams{ { "fc1" }, { "tanh" } });
    work.overrideLayerExecutionTarget(raul::LayerExecutionTarget::CPU);
    work.add<raul::ConvertPrecisionLayer>("c3", raul::ConvertPrecisionParams{ { "tanh"}, { "tanh_fp32"}, false });
    work.add<raul::LinearLayer>("fc2", raul::LinearParams{ { "tanh_fp32" }, { "fc2_fp32" }, FC2_SIZE });
    work.add<raul::ConvertPrecisionLayer>("c4", raul::ConvertPrecisionParams{ { "fc2_fp32"}, { "fc2"}, true });
    work.resetLayerExecutionTargetOverride();
    work.add<raul::SigmoidActivation>("sigmoid", raul::BasicParams{ { "fc2" }, { "sigmoid" } });
    work.overrideLayerExecutionTarget(raul::LayerExecutionTarget::CPU);
    work.add<raul::ConvertPrecisionLayer>("c5", raul::ConvertPrecisionParams{ { "sigmoid"}, { "sigmoid_fp32"}, false });
    work.add<raul::LinearLayer>("fc3", raul::LinearParams{ { "sigmoid_fp32" }, { "fc3_fp32" }, NUM_CLASSES });
    work.add<raul::ConvertPrecisionLayer>("c6", raul::ConvertPrecisionParams{ { "fc3_fp32"}, { "fc3"}, true });
    work.resetLayerExecutionTargetOverride();
    work.add<raul::SoftMaxActivation>("softmax", raul::BasicParamsWithDim{ { "fc3" }, { "softmax" } });
    work.add<raul::CrossEntropyLoss>("loss", raul::LossParams{ { "softmax", "labels" }, { "loss" }, "batch_mean" });

    work.preparePipelines();
    work.setBatchSize(BATCH_SIZE);
    work.prepareMemoryForTraining();

    auto& memory_manager = work.getMemoryManager<raul::MemoryManager>();
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
        raul::dtype testLoss = mnist.oneTrainIterationMixedPrecision(work, sgd.get(), q);
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

TEST(TestMLPMnist, TrainingMixedPrecisionOverride)
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

    raul::Workflow work(raul::CompressionMode::NONE, raul::CalculationMode::DETERMINISTIC, raul::AllocationMode::STANDARD, raul::ExecutionTarget::CPU);
    work.add<raul::DataLayer>("data", raul::DataParams{ { "data", "labels" }, 1, MNIST_SIZE, MNIST_SIZE, NUM_CLASSES });
    work.add<raul::ReshapeLayer>("reshape", raul::ViewParams{ "data", "datar", 1, 1, -1 });
    work.overrideLayerExecutionTarget(raul::LayerExecutionTarget::CPUFP16);
    work.add<raul::ConvertPrecisionLayer>("c1", raul::ConvertPrecisionParams{ { "datar"}, { "datar_fp16"}, false });
    work.add<raul::LinearLayer>("fc1", raul::LinearParams{ { "datar_fp16" }, { "fc1_fp16" }, FC1_SIZE });
    work.add<raul::ConvertPrecisionLayer>("c2", raul::ConvertPrecisionParams{ { "fc1_fp16"}, { "fc1"}, true });
    work.resetLayerExecutionTargetOverride();
    work.add<raul::TanhActivation>("tanh", raul::BasicParams{ { "fc1" }, { "tanh" } });
    work.overrideLayerExecutionTarget(raul::LayerExecutionTarget::CPUFP16);
    work.add<raul::ConvertPrecisionLayer>("c3", raul::ConvertPrecisionParams{ { "tanh"}, { "tanh_fp16"}, false });
    work.add<raul::LinearLayer>("fc2", raul::LinearParams{ { "tanh_fp16" }, { "fc2_fp16" }, FC2_SIZE });
    work.add<raul::ConvertPrecisionLayer>("c4", raul::ConvertPrecisionParams{ { "fc2_fp16"}, { "fc2"}, true });
    work.resetLayerExecutionTargetOverride();
    work.add<raul::SigmoidActivation>("sigmoid", raul::BasicParams{ { "fc2" }, { "sigmoid" } });
    work.overrideLayerExecutionTarget(raul::LayerExecutionTarget::CPUFP16);
    work.add<raul::ConvertPrecisionLayer>("c5", raul::ConvertPrecisionParams{ { "sigmoid"}, { "sigmoid_fp16"}, false });
    work.add<raul::LinearLayer>("fc3", raul::LinearParams{ { "sigmoid_fp16" }, { "fc3_fp16" }, NUM_CLASSES });
    work.add<raul::ConvertPrecisionLayer>("c6", raul::ConvertPrecisionParams{ { "fc3_fp16"}, { "fc3"}, true });
    work.resetLayerExecutionTargetOverride();
    work.add<raul::SoftMaxActivation>("softmax", raul::BasicParamsWithDim{ { "fc3" }, { "softmax" } });
    work.add<raul::CrossEntropyLoss>("loss", raul::LossParams{ { "softmax", "labels" }, { "loss" }, "batch_mean" });

    work.preparePipelines();
    work.setBatchSize(BATCH_SIZE);
    work.prepareMemoryForTraining();

    auto& memory_manager_fp16 = work.getMemoryManager<MemoryManagerFP16>();
    raul::DataLoader dataLoader;

    memory_manager_fp16["fc1::Weights"] = dataLoader.loadData(tools::getTestAssetsDir() / "test_fc_layer" / "mnist" / "0_fc1.weight.data", FC1_SIZE, MNIST_SIZE * MNIST_SIZE);
    memory_manager_fp16["fc1::Biases"] = dataLoader.loadData(tools::getTestAssetsDir() / "test_fc_layer" / "mnist" / "0_fc1.bias.data", 1, FC1_SIZE);
    memory_manager_fp16["fc2::Weights"] = dataLoader.loadData(tools::getTestAssetsDir() / "test_fc_layer" / "mnist" / "0_fc2.weight.data", FC2_SIZE, FC1_SIZE);
    memory_manager_fp16["fc2::Biases"] = dataLoader.loadData(tools::getTestAssetsDir() / "test_fc_layer" / "mnist" / "0_fc2.bias.data", 1, FC2_SIZE);
    memory_manager_fp16["fc3::Weights"] = dataLoader.loadData(tools::getTestAssetsDir() / "test_fc_layer" / "mnist" / "0_fc3.weight.data", NUM_CLASSES, FC2_SIZE);
    memory_manager_fp16["fc3::Biases"] = dataLoader.loadData(tools::getTestAssetsDir() / "test_fc_layer" / "mnist" / "0_fc3.bias.data", 1, NUM_CLASSES);

    raul::Common::transpose(memory_manager_fp16["fc1::Weights"], FC1_SIZE);
    raul::Common::transpose(memory_manager_fp16["fc2::Weights"], FC2_SIZE);
    raul::Common::transpose(memory_manager_fp16["fc3::Weights"], NUM_CLASSES);

    const size_t stepsAmountTrain = mnist.getTrainImageAmount() / BATCH_SIZE;
    raul::Tensor& idealLosses = dataLoader.createTensor(stepsAmountTrain / 100);
    raul::DataLoader::readArrayFromTextFile(tools::getTestAssetsDir() / "test_fc_layer" / "mnist" / "loss.data", idealLosses, 1, idealLosses.size());

    raul::dtype testAcc = mnist.testNetwork(work);
    CHECK_NEAR(testAcc, acc1, EPSILON_ACCURACY);
    printf("Test accuracy = %f\n", testAcc);
    auto sgd = std::make_shared<raul::optimizers::SGD>(LEARNING_RATE);
    for (size_t q = 0, idealLossIndex = 0; q < stepsAmountTrain; ++q)
    {
        raul::dtype testLoss = mnist.oneTrainIterationMixedPrecision(work, sgd.get(), q);
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
