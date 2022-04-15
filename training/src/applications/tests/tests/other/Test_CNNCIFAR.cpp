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
#include <iterator>

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

TEST(TestCNNCIFAR, Training)
{
    PROFILE_TEST

    bool usePool = false;

    raul::CIFAR10 cifar10;
    try
    {
        ASSERT_EQ(cifar10.loadingData(tools::getTestAssetsDir() / "CIFAR"), true);
        const raul::dtype acc1 = TODTYPE(9.79f);
        const raul::dtype acc2 = TODTYPE(37.14f);
        const raul::dtype LEARNING_RATE = TODTYPE(0.01);
        const raul::dtype EPSILON_ACCURACY = TODTYPE(1e-2);
        const raul::dtype EPSILON_LOSS = TODTYPE(1e-6);
        [[maybe_unused]] const raul::dtype EPSILON_WEIGHTS = TODTYPE(1e-2);
        const size_t BATCH_SIZE = 50;
        const size_t NUM_CLASSES = 10;
        const size_t IMAGE_SIZE = 32;
        const size_t IMAGE_CHANNELS = 3;

        const size_t CONV1_FILTERS = 6;
        const size_t CONV1_KERNEL_SIZE = 5;

        const size_t FC1_SIZE = 6 * 28 * 28;
        const size_t FC2_SIZE = 84;

        raul::Workflow work(raul::CompressionMode::NONE, raul::CalculationMode::DETERMINISTIC, usePool ? raul::AllocationMode::POOL : raul::AllocationMode::STANDARD);

        work.add<raul::DataLayer>("data", raul::DataParams{ { "data", "labels" }, IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE, NUM_CLASSES });
        work.add<raul::Convolution2DLayer>("conv1", raul::Convolution2DParams{ { "data" }, { "conv1" }, CONV1_KERNEL_SIZE, CONV1_FILTERS });
        work.add<raul::ReshapeLayer>("reshape", raul::ViewParams{ "conv1", "conv1r", 1, 1, -1 });
        work.add<raul::LinearLayer>("fc1", raul::LinearParams{ { "conv1r" }, { "fc1" }, FC2_SIZE });
        work.add<raul::LinearLayer>("fc2", raul::LinearParams{ { "fc1" }, { "fc2" }, NUM_CLASSES });
        work.add<raul::SoftMaxActivation>("softmax", raul::BasicParamsWithDim{ { "fc2" }, { "softmax" } });
        work.add<raul::CrossEntropyLoss>("loss", raul::LossParams{ { "softmax", "labels" }, { "loss" }, "batch_mean" });

        work.preparePipelines();
        work.setBatchSize(BATCH_SIZE);
        work.prepareMemoryForTraining();

        raul::MemoryManager& memory_manager = work.getMemoryManager();
        raul::DataLoader dataLoader;

        memory_manager["conv1::Weights"] = dataLoader.loadFilters(
            (tools::getTestAssetsDir() / "test_cnn_layer" / "cifar" / "0_conv1.weight_").string(), 0, ".data", CONV1_KERNEL_SIZE, CONV1_KERNEL_SIZE, IMAGE_CHANNELS, CONV1_FILTERS);
        memory_manager["conv1::Biases"] = dataLoader.loadData(tools::getTestAssetsDir() / "test_cnn_layer" / "cifar" / "0_conv1.bias.data", 1, CONV1_FILTERS);

        memory_manager["fc1::Weights"] = dataLoader.loadData(tools::getTestAssetsDir() / "test_cnn_layer" / "cifar" / "0_fc1.weight.data", FC2_SIZE, FC1_SIZE);
        memory_manager["fc1::Biases"] = dataLoader.loadData(tools::getTestAssetsDir() / "test_cnn_layer" / "cifar" / "0_fc1.bias.data", 1, FC2_SIZE);
        memory_manager["fc2::Weights"] = dataLoader.loadData(tools::getTestAssetsDir() / "test_cnn_layer" / "cifar" / "0_fc2.weight.data", NUM_CLASSES, FC2_SIZE);
        memory_manager["fc2::Biases"] = dataLoader.loadData(tools::getTestAssetsDir() / "test_cnn_layer" / "cifar" / "0_fc2.bias.data", 1, NUM_CLASSES);

        raul::Common::transpose(memory_manager["fc1::Weights"], FC2_SIZE);
        raul::Common::transpose(memory_manager["fc2::Weights"], NUM_CLASSES);

        const size_t stepsAmountTrain = cifar10.getTrainImageAmount() / BATCH_SIZE;

        raul::Tensor& idealLosses = dataLoader.createTensor(stepsAmountTrain / 100);
        raul::DataLoader::readArrayFromTextFile(tools::getTestAssetsDir() / "test_cnn_layer" / "cifar" / "loss.data", idealLosses, 1, idealLosses.size());
        raul::dtype testAcc = cifar10.testNetwork(work);
        CHECK_NEAR(testAcc, acc1, EPSILON_ACCURACY);
        printf("Test accuracy = %f\n", testAcc);
        auto sgd = std::make_shared<raul::optimizers::SGD>(LEARNING_RATE);
        for (size_t q = 0, idealLossIndex = 0; q < stepsAmountTrain; ++q)
        {
            raul::dtype testLoss = cifar10.oneTrainIteration(work, sgd.get(), q);
            if (q % 100 == 0)
            {
                CHECK_NEAR(testLoss, idealLosses[idealLossIndex++], EPSILON_LOSS);
                printf("iteration = %d, loss = %f\n", static_cast<uint32_t>(q), testLoss);
            }
        }
        testAcc = cifar10.testNetwork(work);
        CHECK_NEAR(testAcc, acc2, EPSILON_ACCURACY);
        printf("Test accuracy = %f\n", testAcc);
    }
    catch (std::runtime_error& e)
    {

        std::cout << e.what() << std::endl;
        ASSERT_EQ(true, false);
    }
}
TEST(TestCNNCIFAR, Training2)
{
    PROFILE_TEST

    bool usePool = false;

    raul::CIFAR10 cifar10;
    ASSERT_EQ(cifar10.loadingData(tools::getTestAssetsDir() / "CIFAR"), true);
    const raul::dtype acc1 = TODTYPE(10.15f);
    const raul::dtype acc2 = TODTYPE(17.5f);
    const raul::dtype LEARNING_RATE = TODTYPE(0.01);
    const raul::dtype EPSILON_ACCURACY = TODTYPE(1e-2);
    const raul::dtype EPSILON_LOSS = TODTYPE(1e-6);
    [[maybe_unused]] const raul::dtype EPSILON_WEIGHTS = TODTYPE(1e-2);
    const size_t BATCH_SIZE = 50;
    const size_t NUM_CLASSES = 10;
    const size_t IMAGE_SIZE = 32;
    const size_t IMAGE_CHANNELS = 3;

    const size_t CONV1_FILTERS = 32;
    const size_t CONV1_KERNEL_SIZE = 3;

    const size_t CONV2_FILTERS = 64;

    const size_t CONV3_FILTERS = 128;

    const size_t AVGPOOL_KERNEL_SIZE = 2;
    const size_t AVGPOOL_STRIDE = 2;

    const size_t FC1_SIZE = 128;
    const size_t FC2_SIZE = 84;

    raul::Workflow work(raul::CompressionMode::NONE, raul::CalculationMode::DETERMINISTIC, usePool ? raul::AllocationMode::POOL : raul::AllocationMode::STANDARD);

    work.add<raul::DataLayer>("data", raul::DataParams{ { "data", "labels" }, IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE, NUM_CLASSES });
    work.add<raul::Convolution2DLayer>("conv1", raul::Convolution2DParams{ { "data" }, { "conv1" }, CONV1_KERNEL_SIZE, CONV1_FILTERS });
    work.add<raul::BatchNormLayer>("batch1", raul::BatchnormParams{ { "conv1" }, { "batch1" } });
    work.add<raul::Convolution2DLayer>("conv2", raul::Convolution2DParams{ { "batch1" }, { "conv2" }, CONV1_KERNEL_SIZE, CONV1_FILTERS });
    work.add<raul::BatchNormLayer>("batch2", raul::BatchnormParams{ { "conv2" }, { "batch2" } });
    work.add<raul::AveragePoolLayer>("avgpool1", raul::Pool2DParams{ { "batch2" }, { "avgpool1" }, AVGPOOL_KERNEL_SIZE, AVGPOOL_STRIDE });

    work.add<raul::Convolution2DLayer>("conv3", raul::Convolution2DParams{ { "avgpool1" }, { "conv3" }, CONV1_KERNEL_SIZE, CONV2_FILTERS });
    work.add<raul::BatchNormLayer>("batch3", raul::BatchnormParams{ { "conv3" }, { "batch3" } });
    work.add<raul::Convolution2DLayer>("conv4", raul::Convolution2DParams{ { "batch3" }, { "conv4" }, CONV1_KERNEL_SIZE, CONV2_FILTERS });
    work.add<raul::BatchNormLayer>("batch4", raul::BatchnormParams{ { "conv4" }, { "batch4" } });
    work.add<raul::AveragePoolLayer>("avgpool2", raul::Pool2DParams{ { "batch4" }, { "avgpool2" }, AVGPOOL_KERNEL_SIZE, AVGPOOL_STRIDE });

    work.add<raul::Convolution2DLayer>("conv5", raul::Convolution2DParams{ { "avgpool2" }, { "conv5" }, CONV1_KERNEL_SIZE, CONV3_FILTERS });
    work.add<raul::BatchNormLayer>("batch5", raul::BatchnormParams{ { "conv5" }, { "batch5" } });
    work.add<raul::Convolution2DLayer>("conv6", raul::Convolution2DParams{ { "batch5" }, { "conv6" }, CONV1_KERNEL_SIZE, CONV3_FILTERS });
    work.add<raul::BatchNormLayer>("batch6", raul::BatchnormParams{ { "conv6" }, { "batch6" } });

    work.add<raul::ReshapeLayer>("reshape", raul::ViewParams{ "batch6", "batch6r", 1, 1, -1 });
    work.add<raul::LinearLayer>("fc1", raul::LinearParams{ { "batch6r" }, { "fc1" }, FC2_SIZE });
    work.add<raul::LinearLayer>("fc2", raul::LinearParams{ { "fc1" }, { "fc2" }, NUM_CLASSES });
    work.add<raul::SoftMaxActivation>("softmax", raul::BasicParamsWithDim{ { "fc2" }, { "softmax" } });
    work.add<raul::CrossEntropyLoss>("loss", raul::LossParams{ { "softmax", "labels" }, { "loss" }, "batch_mean" });

    work.preparePipelines();
    work.setBatchSize(BATCH_SIZE);
    work.prepareMemoryForTraining();

    raul::MemoryManager& memory_manager = work.getMemoryManager();
    raul::DataLoader dataLoader;

    memory_manager["conv1::Weights"] =
        dataLoader.loadFilters((tools::getTestAssetsDir() / "test_cnn_layer" / "cifar2" / "0_conv1.weight_").string(), 0, ".data", CONV1_KERNEL_SIZE, CONV1_KERNEL_SIZE, IMAGE_CHANNELS, CONV1_FILTERS);
    memory_manager["conv1::Biases"] = dataLoader.loadData(tools::getTestAssetsDir() / "test_cnn_layer" / "cifar2" / "0_conv1.bias.data", 1, CONV1_FILTERS);

    memory_manager["conv2::Weights"] =
        dataLoader.loadFilters((tools::getTestAssetsDir() / "test_cnn_layer" / "cifar2" / "0_conv2.weight_").string(), 0, ".data", CONV1_KERNEL_SIZE, CONV1_KERNEL_SIZE, CONV1_FILTERS, CONV1_FILTERS);
    memory_manager["conv2::Biases"] = dataLoader.loadData(tools::getTestAssetsDir() / "test_cnn_layer" / "cifar2" / "0_conv2.bias.data", 1, CONV1_FILTERS);

    memory_manager["conv3::Weights"] =
        dataLoader.loadFilters((tools::getTestAssetsDir() / "test_cnn_layer" / "cifar2" / "0_conv3.weight_").string(), 0, ".data", CONV1_KERNEL_SIZE, CONV1_KERNEL_SIZE, CONV1_FILTERS, CONV2_FILTERS);
    memory_manager["conv3::Biases"] = dataLoader.loadData(tools::getTestAssetsDir() / "test_cnn_layer" / "cifar2" / "0_conv3.bias.data", 1, CONV2_FILTERS);

    memory_manager["conv4::Weights"] =
        dataLoader.loadFilters((tools::getTestAssetsDir() / "test_cnn_layer" / "cifar2" / "0_conv4.weight_").string(), 0, ".data", CONV1_KERNEL_SIZE, CONV1_KERNEL_SIZE, CONV2_FILTERS, CONV2_FILTERS);
    memory_manager["conv4::Biases"] = dataLoader.loadData(tools::getTestAssetsDir() / "test_cnn_layer" / "cifar2" / "0_conv4.bias.data", 1, CONV2_FILTERS);

    memory_manager["conv5::Weights"] =
        dataLoader.loadFilters((tools::getTestAssetsDir() / "test_cnn_layer" / "cifar2" / "0_conv5.weight_").string(), 0, ".data", CONV1_KERNEL_SIZE, CONV1_KERNEL_SIZE, CONV2_FILTERS, CONV3_FILTERS);
    memory_manager["conv5::Biases"] = dataLoader.loadData(tools::getTestAssetsDir() / "test_cnn_layer" / "cifar2" / "0_conv5.bias.data", 1, CONV3_FILTERS);

    memory_manager["conv6::Weights"] =
        dataLoader.loadFilters((tools::getTestAssetsDir() / "test_cnn_layer" / "cifar2" / "0_conv6.weight_").string(), 0, ".data", CONV1_KERNEL_SIZE, CONV1_KERNEL_SIZE, CONV3_FILTERS, CONV3_FILTERS);
    memory_manager["conv6::Biases"] = dataLoader.loadData(tools::getTestAssetsDir() / "test_cnn_layer" / "cifar2" / "0_conv6.bias.data", 1, CONV3_FILTERS);

    memory_manager["fc1::Weights"] = dataLoader.loadData(tools::getTestAssetsDir() / "test_cnn_layer" / "cifar2" / "0_fc1.weight.data", FC2_SIZE, FC1_SIZE);
    memory_manager["fc1::Biases"] = dataLoader.loadData(tools::getTestAssetsDir() / "test_cnn_layer" / "cifar2" / "0_fc1.bias.data", 1, FC2_SIZE);
    memory_manager["fc2::Weights"] = dataLoader.loadData(tools::getTestAssetsDir() / "test_cnn_layer" / "cifar2" / "0_fc2.weight.data", NUM_CLASSES, FC2_SIZE);
    memory_manager["fc2::Biases"] = dataLoader.loadData(tools::getTestAssetsDir() / "test_cnn_layer" / "cifar2" / "0_fc2.bias.data", 1, NUM_CLASSES);

    raul::Common::transpose(memory_manager["fc1::Weights"], FC2_SIZE);
    raul::Common::transpose(memory_manager["fc2::Weights"], NUM_CLASSES);

    const size_t stepsAmountTrain = cifar10.getTrainImageAmount() / BATCH_SIZE;

    raul::Tensor& idealLosses = dataLoader.createTensor(stepsAmountTrain / 100);
    raul::DataLoader::readArrayFromTextFile(tools::getTestAssetsDir() / "test_cnn_layer" / "cifar2" / "loss.data", idealLosses, 1, idealLosses.size());
    raul::dtype testAcc = cifar10.testNetwork(work);
    CHECK_NEAR(testAcc, acc1, EPSILON_ACCURACY);
    printf("Test accuracy = %f\n", testAcc);
    auto sgd = std::make_shared<raul::optimizers::SGD>(LEARNING_RATE);
    for (size_t q = 0, idealLossIndex = 0; q < stepsAmountTrain; ++q)
    {
        raul::dtype testLoss = cifar10.oneTrainIteration(work, sgd.get(), q);
        if (q % 100 == 0)
        {
            if (q == 0 || q == 100)
                CHECK_NEAR(testLoss, idealLosses[idealLossIndex++], EPSILON_LOSS * TODTYPE(10));
            else
                CHECK_NEAR(testLoss, idealLosses[idealLossIndex++], EPSILON_LOSS);
            printf("iteration = %d, loss = %f\n", static_cast<uint32_t>(q), testLoss);
        }
    }
    testAcc = cifar10.testNetwork(work);
    CHECK_NEAR(testAcc, acc2, EPSILON_ACCURACY);
    printf("Test accuracy = %f\n", testAcc);
}

TEST(TestCNNCIFAR, Loading)
{
    PROFILE_TEST
    raul::CIFAR10 cifar10;
    const auto path = tools::getTestAssetsDir() / "CIFAR";
    std::string files[] = { "data_batch_1.bin", "data_batch_2.bin", "data_batch_3.bin", "data_batch_4.bin", "data_batch_5.bin" };
    // auto n = std::size(files);
    size_t count = 0;
    for (auto file : files)
    {
        auto filePath = path / file;
        ASSERT_EQ(cifar10.load(filePath), true);
        ASSERT_EQ(cifar10.getTrainImageAmount() - count, 10000u);
        count = cifar10.getTrainImageAmount();
    }
}

} // UT namespace
