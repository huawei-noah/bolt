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
#include <training/layers/basic/AveragePoolLayer.h>
#include <training/layers/basic/DataLayer.h>
#include <training/layers/parameters/LayerParameters.h>
#include <training/network/Layers.h>
#include <training/network/Workflow.h>
#include <training/optimizers/SGD.h>
#include <training/tools/Datasets.h>

namespace UT
{
using namespace raul;
TEST(TestCNNAveragePool, Unit)
{
    PROFILE_TEST
    dtype eps = TODTYPE(1e-6);
    {
        MANAGERS_DEFINE
        NETWORK_PARAMS_DEFINE(networkParameters);

        Tensor raw = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
        size_t batch = 1;
        size_t stride_w = 1;
        size_t stride_h = 1;
        size_t in_w = 3;
        size_t in_h = 3;
        size_t padding_w = 0;
        size_t padding_h = 0;
        size_t depth = 1;
        size_t kernel_height = 3;
        size_t kernel_width = 3;

        work.add<DataLayer>("data", DataParams{ { "in" }, depth, in_h, in_w });
        auto params = Pool2DParams{ { "in" }, { "avg" }, kernel_width, kernel_height, stride_w, stride_h, padding_w, padding_h };
        AveragePoolLayer avgpool("avg1", params, networkParameters);
        TENSORS_CREATE(batch);
        memory_manager["in"] = TORANGE(raw);
        const Tensor& out = memory_manager["avg"];
        avgpool.forwardCompute(NetworkMode::Train);
        EXPECT_NEAR(TODTYPE(5.f), out[0], eps);
        printf(" - AveragePool with square kernel and stride = 1 is Ok.\n");

        memory_manager.clear();
    }
    {
        MANAGERS_DEFINE
        NETWORK_PARAMS_DEFINE(networkParameters);

        Tensor raw = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
        size_t batch = 1;
        size_t in_w = 4;
        size_t in_h = 3;
        size_t depth = 1;
        size_t kernel_height = 3;
        size_t kernel_width = 2;
        size_t stride_w = 1;
        size_t stride_h = 1;
        size_t padding_w = 0;
        size_t padding_h = 0;

        work.add<DataLayer>("data", DataParams{ { "in" }, depth, in_h, in_w });
        auto params = Pool2DParams{ { "in" }, { "avg" }, kernel_width, kernel_height, stride_w, stride_h, padding_w, padding_h };
        AveragePoolLayer avgpool("avg1", params, networkParameters);
        TENSORS_CREATE(batch);
        memory_manager["in"] = TORANGE(raw);
        const Tensor& out = memory_manager["avg"];
        avgpool.forwardCompute(NetworkMode::Train);

        EXPECT_NEAR(TODTYPE(5.5f), out[0], eps);
        EXPECT_NEAR(TODTYPE(6.5f), out[1], eps);
        EXPECT_NEAR(TODTYPE(7.5f), out[2], eps);
        printf(" - AveragePool with kernel = (2,3) and stride = 1 is Ok.\n");

        memory_manager.clear();
    }
    {
        MANAGERS_DEFINE
        NETWORK_PARAMS_DEFINE(networkParameters);

        Tensor raw = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
        size_t batch = 1;
        size_t in_w = 5;
        size_t in_h = 3;
        size_t depth = 1;
        size_t kernel_height = 2;
        size_t kernel_width = 3;
        size_t stride_w = 3;
        size_t stride_h = 1;
        size_t padding_w = 1;
        size_t padding_h = 0;

        work.add<DataLayer>("data", DataParams{ { "in" }, depth, in_h, in_w });
        auto params = Pool2DParams{ { "in" }, { "avg" }, kernel_width, kernel_height, stride_w, stride_h, padding_w, padding_h };
        AveragePoolLayer avgpool("avg1", params, networkParameters);
        TENSORS_CREATE(batch);
        memory_manager["in"] = TORANGE(raw);
        const Tensor& out = memory_manager["avg"];
        avgpool.forwardCompute(NetworkMode::Train);

        EXPECT_NEAR(TODTYPE(2.6666667f), out[0], eps);
        EXPECT_NEAR(TODTYPE(6.5f), out[1], eps);
        EXPECT_NEAR(TODTYPE(6.0f), out[2], eps);
        EXPECT_NEAR(TODTYPE(11.5f), out[3], eps);

        printf(" - AveragePool with kernel = (3,2) and stride = (3,1) and padding (1,0) is Ok.\n");
    }
    {
        MANAGERS_DEFINE
        NETWORK_PARAMS_DEFINE(networkParameters);

        size_t in_w = 5;
        size_t in_h = 3;
        size_t depth = 1;
        size_t kernel_height = 2;
        size_t kernel_width = 3;
        size_t stride_w = 3;
        size_t stride_h = 1;
        size_t padding_w = 2;
        size_t padding_h = 0;

        work.add<DataLayer>("data", DataParams{ { "in" }, depth, in_h, in_w });
        auto params = Pool2DParams{ { "in" }, { "out" }, kernel_width, kernel_height, stride_w, stride_h, padding_w, padding_h };
        EXPECT_THROW(AveragePoolLayer averagepool4("avg1", params, networkParameters), raul::Exception);
        printf(" - AveragePool with wrong Padding make throw - Ok.\n");
    }
    {
        MANAGERS_DEFINE
        NETWORK_PARAMS_DEFINE(networkParameters);
        size_t in_w = 0;
        size_t in_h = 0;
        size_t depth = 1;
        size_t kernel_height = 2;
        size_t kernel_width = 3;
        size_t stride_w = 1;
        size_t stride_h = 1;
        size_t padding_w = 2;
        size_t padding_h = 0;

        work.add<DataLayer>("data", DataParams{ { "in" }, depth, in_h, in_w });
        auto params = Pool2DParams{ { "in2" }, { "out2" }, kernel_width, kernel_height, stride_w, stride_h, padding_w, padding_h };
        EXPECT_THROW(AveragePoolLayer averagepool5("avg1", params, networkParameters), raul::Exception);
        printf(" - AveragePool with wrong Input size make throw - Ok.\n");
    }
}

TEST(TestCNNAveragePool, Training)
{
    PROFILE_TEST

    const dtype LEARNING_RATE = TODTYPE(0.1);
    const size_t BATCH_SIZE = 50;
    const dtype EPSILON_ACCURACY = TODTYPE(1e-2);
    const dtype EPSILON_LOSS = TODTYPE(1e-6);

    const size_t NUM_CLASSES = 10;
    const dtype acc1 = TODTYPE(10.32f);
    const dtype acc2 = TODTYPE(91.46f);
    const size_t MNIST_SIZE = 28;
    const size_t MNIST_CHANNELS = 1;
    const size_t CONV1_FILTERS = 16;
    const size_t CONV1_KERNEL_SIZE = 5;
    const size_t CONV1_STRIDE = 2;

    const size_t AVERAGEPOOL_KERNEL = 2;
    const size_t AVERAGEPOOL_STRIDE = 2;

    const size_t FC1_SIZE = 576;
    const size_t FC2_SIZE = 128;

    MNIST mnist;
    ASSERT_EQ(mnist.loadingData(tools::getTestAssetsDir() / "MNIST"), true);

    Workflow work;
    work.add<raul::DataLayer>("data", DataParams{ { "data", "labels" }, MNIST_CHANNELS, MNIST_SIZE, MNIST_SIZE, NUM_CLASSES });
    work.add<raul::Convolution2DLayer>("conv1", Convolution2DParams{ { "data" }, { "conv1" }, CONV1_KERNEL_SIZE, CONV1_FILTERS, CONV1_STRIDE });
    work.add<raul::AveragePoolLayer>("AVERAGEPOOL", Pool2DParams{ { "conv1" }, { "avg" }, AVERAGEPOOL_KERNEL, AVERAGEPOOL_STRIDE });
    work.add<raul::ReshapeLayer>("reshape", ViewParams{ "avg", "avgr", 1, 1, -1 });
    work.add<raul::LinearLayer>("fc1", LinearParams{ { "avgr" }, { "fc1" }, FC2_SIZE });
    work.add<raul::SigmoidActivation>("sigmoid", BasicParams{ { "fc1" }, { "sigmoid" } });
    work.add<raul::LinearLayer>("fc2", LinearParams{ { "sigmoid" }, { "fc2" }, NUM_CLASSES });
    work.add<raul::SoftMaxActivation>("softmax", BasicParamsWithDim{ { "fc2" }, { "softmax" } });
    work.add<raul::CrossEntropyLoss>("loss", LossParams{ { "softmax", "labels" }, { "loss" }, "batch_mean" });

    work.preparePipelines();
    work.setBatchSize(BATCH_SIZE);
    work.prepareMemoryForTraining();

    MemoryManager& memory_manager = work.getMemoryManager();
    DataLoader dataLoader;

    memory_manager["conv1::Weights"] =
        dataLoader.loadFilters((tools::getTestAssetsDir() / "test_cnn_layer" / "averagepool" / "0_conv1.weight_").string(), 0, ".data", CONV1_KERNEL_SIZE, CONV1_KERNEL_SIZE, 1, CONV1_FILTERS);
    memory_manager["conv1::Biases"] = dataLoader.loadData(tools::getTestAssetsDir() / "test_cnn_layer" / "maxpooltest" / "0_conv1.bias.data", 1, CONV1_FILTERS);

    memory_manager["fc1::Weights"] = dataLoader.loadData(tools::getTestAssetsDir() / "test_cnn_layer" / "averagepool" / "0_fc1.weight.data", FC2_SIZE, FC1_SIZE);
    memory_manager["fc1::Biases"] = dataLoader.loadData(tools::getTestAssetsDir() / "test_cnn_layer" / "averagepool" / "0_fc1.bias.data", 1, FC2_SIZE);
    memory_manager["fc2::Weights"] = dataLoader.loadData(tools::getTestAssetsDir() / "test_cnn_layer" / "averagepool" / "0_fc2.weight.data", NUM_CLASSES, FC2_SIZE);
    memory_manager["fc2::Biases"] = dataLoader.loadData(tools::getTestAssetsDir() / "test_cnn_layer" / "averagepool" / "0_fc2.bias.data", 1, NUM_CLASSES);

    Common::transpose(memory_manager["fc1::Weights"], FC2_SIZE);
    Common::transpose(memory_manager["fc2::Weights"], NUM_CLASSES);

    const size_t stepsAmountTrain = mnist.getTrainImageAmount() / BATCH_SIZE;
    Tensor& idealLosses = dataLoader.createTensor(stepsAmountTrain / 100);
    DataLoader::readArrayFromTextFile(tools::getTestAssetsDir() / "test_cnn_layer" / "averagepool" / "loss.data", idealLosses, 1, idealLosses.size());

    dtype testAcc = mnist.testNetwork(work);
    CHECK_NEAR(testAcc, acc1, EPSILON_ACCURACY);
    printf("Test accuracy = %f\n", testAcc);
    auto sgd = std::make_shared<optimizers::SGD>(LEARNING_RATE);
    for (size_t q = 0, idealLossIndex = 0; q < stepsAmountTrain; ++q)
    {
        dtype testLoss = mnist.oneTrainIteration(work, sgd.get(), q);
        if (q % 100 == 0)
        {
            CHECK_NEAR(testLoss, idealLosses[idealLossIndex++], EPSILON_LOSS);
            printf("iteration = %d, loss = %f\n", static_cast<uint32_t>(q), testLoss);
        }
    }
    testAcc = mnist.testNetwork(work);
    CHECK_NEAR(testAcc, acc2, EPSILON_ACCURACY);
    printf("Test accuracy = %f\n", testAcc);
}

} // UT namespace
