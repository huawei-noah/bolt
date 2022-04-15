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

TEST(TestCNNMnist, Training)
{
    PROFILE_TEST

    const raul::dtype LEARNING_RATE = TODTYPE(0.1);
    const size_t BATCH_SIZE = 50;
    const raul::dtype EPSILON_ACCURACY = TODTYPE(1e-2);
    const raul::dtype EPSILON_LOSS = TODTYPE(1e-5);
    [[maybe_unused]] const raul::dtype EPSILON_WEIGHTS = TODTYPE(1e-2);
    const raul::dtype acc1 = TODTYPE(11.35f);
    const raul::dtype acc2 = TODTYPE(94.27f);
    const size_t NUM_CLASSES = 10;

    const size_t MNIST_SIZE = 28;
    const size_t MNIST_CHANNELS = 1;
    const size_t CONV1_FILTERS = 16;
    const size_t CONV1_KERNEL_SIZE = 5;
    const size_t CONV1_STRIDE = 2;
    const size_t CONV2_FILTERS = 32;
    const size_t CONV2_KERNEL_SIZE = 5;
    const size_t CONV2_STRIDE = 2;
    const size_t FC1_SIZE = 512;
    const size_t FC2_SIZE = 256;

    raul::MNIST mnist;
    ASSERT_EQ(mnist.loadingData(tools::getTestAssetsDir() / "MNIST"), true);

    raul::Workflow work;
    work.add<raul::DataLayer>("data", raul::DataParams{ { "data", "labels" }, MNIST_CHANNELS, MNIST_SIZE, MNIST_SIZE, NUM_CLASSES });
    work.add<raul::Convolution2DLayer>("conv1", raul::Convolution2DParams{ { "data" }, { "conv1" }, CONV1_KERNEL_SIZE, CONV1_FILTERS, CONV1_STRIDE });
    work.add<raul::ReLUActivation>("relu", raul::BasicParams{ { "conv1" }, { "relu" } });
    work.add<raul::Convolution2DLayer>("conv2", raul::Convolution2DParams{ { "relu" }, { "conv2" }, CONV2_KERNEL_SIZE, CONV2_FILTERS, CONV2_STRIDE });
    work.add<raul::ReshapeLayer>("reshape", raul::ViewParams{ "conv2", "conv2r", 1, 1, -1 });
    work.add<raul::LinearLayer>("fc1", raul::LinearParams{ { "conv2r" }, { "fc1" }, FC2_SIZE });
    work.add<raul::SigmoidActivation>("sigmoid", raul::BasicParams{ { "fc1" }, { "sigmoid" } });
    work.add<raul::LinearLayer>("fc2", raul::LinearParams{ { "sigmoid" }, { "fc2" }, NUM_CLASSES });
    work.add<raul::SoftMaxActivation>("softmax", raul::BasicParamsWithDim{ { "fc2" }, { "softmax" } });
    work.add<raul::CrossEntropyLoss>("loss", raul::LossParams{ { "softmax", "labels" }, { "loss" }, "batch_mean" });

    work.preparePipelines();
    work.setBatchSize(BATCH_SIZE);
    work.prepareMemoryForTraining();

    raul::MemoryManager& memory_manager = work.getMemoryManager();
    raul::DataLoader dataLoader;

    memory_manager["conv1::Weights"] =
        dataLoader.loadFilters((tools::getTestAssetsDir() / "test_cnn_layer" / "mnist" / "0_conv1.weight_").string(), 0, ".data", CONV1_KERNEL_SIZE, CONV1_KERNEL_SIZE, 1, CONV1_FILTERS);
    memory_manager["conv1::Biases"] = dataLoader.loadData(tools::getTestAssetsDir() / "test_cnn_layer" / "mnist" / "0_conv1.bias.data", 1, CONV1_FILTERS);

    memory_manager["conv2::Weights"] =
        dataLoader.loadFilters((tools::getTestAssetsDir() / "test_cnn_layer" / "mnist" / "0_conv2.weight_").string(), 0, ".data", CONV1_KERNEL_SIZE, CONV1_KERNEL_SIZE, CONV1_FILTERS, CONV2_FILTERS);
    memory_manager["conv2::Biases"] = dataLoader.loadData(tools::getTestAssetsDir() / "test_cnn_layer" / "mnist" / "0_conv2.bias.data", 1, CONV2_FILTERS);

    memory_manager["fc1::Weights"] = dataLoader.loadData(tools::getTestAssetsDir() / "test_cnn_layer" / "mnist" / "0_fc1.weight.data", FC2_SIZE, FC1_SIZE);
    memory_manager["fc1::Biases"] = dataLoader.loadData(tools::getTestAssetsDir() / "test_cnn_layer" / "mnist" / "0_fc1.bias.data", 1, FC2_SIZE);
    memory_manager["fc2::Weights"] = dataLoader.loadData(tools::getTestAssetsDir() / "test_cnn_layer" / "mnist" / "0_fc2.weight.data", NUM_CLASSES, FC2_SIZE);
    memory_manager["fc2::Biases"] = dataLoader.loadData(tools::getTestAssetsDir() / "test_cnn_layer" / "mnist" / "0_fc2.bias.data", 1, NUM_CLASSES);

    raul::Common::transpose(memory_manager["fc1::Weights"], FC2_SIZE);
    raul::Common::transpose(memory_manager["fc2::Weights"], NUM_CLASSES);

    const size_t stepsAmountTrain = mnist.getTrainImageAmount() / BATCH_SIZE;

    raul::Tensor& idealLosses = dataLoader.createTensor(stepsAmountTrain / 100);
    raul::DataLoader::readArrayFromTextFile(tools::getTestAssetsDir() / "test_cnn_layer" / "mnist" / "loss.data", idealLosses, 1, idealLosses.size());

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

TEST(TestCNNMnist, Training2)
{
    PROFILE_TEST

    const raul::dtype LEARNING_RATE = TODTYPE(0.1);
    const size_t BATCH_SIZE = 50;
    const raul::dtype EPSILON_ACCURACY = TODTYPE(1e-2);
    const raul::dtype EPSILON_LOSS = TODTYPE(1e-5);

    const size_t NUM_CLASSES = 10;

    const size_t MNIST_SIZE = 28;
    const size_t MNIST_CHANNELS = 1;

    const size_t CONV1_FILTERS = 1;
    const size_t CONV1_KERNEL_SIZE = 5;
    const size_t CONV1_STRIDE = 1;
    const size_t CONV1_PADDING = 2;

    const size_t CONV2_FILTERS = 16;
    const size_t CONV2_KERNEL_SIZE = 5;
    const size_t CONV2_STRIDE = 2;

    const size_t CONV3_FILTERS = 32;
    const size_t CONV3_KERNEL_SIZE = 5;
    const size_t CONV3_STRIDE = 2;

    const size_t FC1_SIZE = 512;
    const size_t FC2_SIZE = 256;

    const raul::dtype acc1 = TODTYPE(9.64f);
    const raul::dtype acc2 = TODTYPE(86.02f);

    raul::MNIST mnist;
    ASSERT_EQ(mnist.loadingData(tools::getTestAssetsDir() / "MNIST"), true);

    raul::Workflow work;
    work.add<raul::DataLayer>("data", raul::DataParams{ { "data", "labels" }, MNIST_CHANNELS, MNIST_SIZE, MNIST_SIZE, NUM_CLASSES });
    work.add<raul::Convolution2DLayer>("conv1", raul::Convolution2DParams{ { "data" }, { "conv1" }, CONV1_KERNEL_SIZE, CONV1_FILTERS, CONV1_STRIDE, CONV1_PADDING });
    work.add<raul::Convolution2DLayer>("conv2", raul::Convolution2DParams{ { "conv1" }, { "conv2" }, CONV2_KERNEL_SIZE, CONV2_FILTERS, CONV2_STRIDE });
    work.add<raul::Convolution2DLayer>("conv3", raul::Convolution2DParams{ { "conv2" }, { "conv3" }, CONV3_KERNEL_SIZE, CONV3_FILTERS, CONV3_STRIDE });
    work.add<raul::ReshapeLayer>("reshape", raul::ViewParams{ "conv3", "conv3r", 1, 1, -1 });
    work.add<raul::LinearLayer>("fc1", raul::LinearParams{ { "conv3r" }, { "fc1" }, FC2_SIZE });
    work.add<raul::LinearLayer>("fc2", raul::LinearParams{ { "fc1" }, { "fc2" }, NUM_CLASSES });
    work.add<raul::SoftMaxActivation>("softmax", raul::BasicParamsWithDim{ { "fc2" }, { "softmax" } });
    work.add<raul::CrossEntropyLoss>("loss", raul::LossParams{ { "softmax", "labels" }, { "loss" }, "batch_mean" });

    work.preparePipelines();
    work.setBatchSize(BATCH_SIZE);
    work.prepareMemoryForTraining();

    raul::MemoryManager& memory_manager = work.getMemoryManager();
    raul::DataLoader dataLoader;

    memory_manager["conv1::Weights"] =
        dataLoader.loadFilters((tools::getTestAssetsDir() / "test_cnn_layer" / "mnistPadding" / "0_conv1.weight_").string(), 0, ".data", CONV1_KERNEL_SIZE, CONV1_KERNEL_SIZE, 1, CONV1_FILTERS);
    memory_manager["conv1::Biases"] = dataLoader.loadData(tools::getTestAssetsDir() / "test_cnn_layer" / "mnistPadding" / "0_conv1.bias.data", 1, CONV1_FILTERS);

    memory_manager["conv2::Weights"] = dataLoader.loadFilters(
        (tools::getTestAssetsDir() / "test_cnn_layer" / "mnistPadding" / "0_conv2.weight_").string(), 0, ".data", CONV2_KERNEL_SIZE, CONV2_KERNEL_SIZE, CONV1_FILTERS, CONV2_FILTERS);
    memory_manager["conv2::Biases"] = dataLoader.loadData(tools::getTestAssetsDir() / "test_cnn_layer" / "mnistPadding" / "0_conv2.bias.data", 1, CONV2_FILTERS);

    memory_manager["conv3::Weights"] = dataLoader.loadFilters(
        (tools::getTestAssetsDir() / "test_cnn_layer" / "mnistPadding" / "0_conv3.weight_").string(), 0, ".data", CONV3_KERNEL_SIZE, CONV3_KERNEL_SIZE, CONV2_FILTERS, CONV3_FILTERS);
    memory_manager["conv3::Biases"] = dataLoader.loadData(tools::getTestAssetsDir() / "test_cnn_layer" / "mnistPadding" / "0_conv3.bias.data", 1, CONV3_FILTERS);

    memory_manager["fc1::Weights"] = dataLoader.loadData(tools::getTestAssetsDir() / "test_cnn_layer" / "mnistPadding" / "0_fc1.weight.data", FC2_SIZE, FC1_SIZE);
    memory_manager["fc1::Biases"] = dataLoader.loadData(tools::getTestAssetsDir() / "test_cnn_layer" / "mnistPadding" / "0_fc1.bias.data", 1, FC2_SIZE);
    memory_manager["fc2::Weights"] = dataLoader.loadData(tools::getTestAssetsDir() / "test_cnn_layer" / "mnistPadding" / "0_fc2.weight.data", NUM_CLASSES, FC2_SIZE);
    memory_manager["fc2::Biases"] = dataLoader.loadData(tools::getTestAssetsDir() / "test_cnn_layer" / "mnistPadding" / "0_fc2.bias.data", 1, NUM_CLASSES);

    raul::Common::transpose(memory_manager["fc1::Weights"], FC2_SIZE);
    raul::Common::transpose(memory_manager["fc2::Weights"], NUM_CLASSES);

    const size_t stepsAmountTrain = mnist.getTrainImageAmount() / BATCH_SIZE;
    raul::Tensor& idealLosses = dataLoader.createTensor(stepsAmountTrain / 100);
    raul::DataLoader::readArrayFromTextFile(tools::getTestAssetsDir() / "test_cnn_layer" / "mnistPadding" / "loss.data", idealLosses, 1, idealLosses.size());

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

} // UT namespace
