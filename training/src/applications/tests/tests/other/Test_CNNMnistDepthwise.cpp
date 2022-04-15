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

TEST(TestCNNDepthwiseMnist, Training)
{
    PROFILE_TEST

    const raul::dtype LEARNING_RATE = TODTYPE(0.1);
    const size_t BATCH_SIZE = 50;
    const raul::dtype EPSILON_ACCURACY = TODTYPE(1e-2);
    const raul::dtype EPSILON_LOSS = TODTYPE(1e-5);

    const size_t NUM_CLASSES = 10;

    const size_t MNIST_SIZE = 28;
    const size_t MNIST_CHANNELS = 1;

    const size_t CONV1_DEPTHWISE_FILTERS = 1;
    const size_t CONV1_DEPTHWISE_KERNEL_SIZE = 3;
    const size_t CONV1_DEPTHWISE_STRIDE = 2;
    const size_t CONV1_DEPTHWISE_PADDING = 1;

    const size_t CONV1_POINTWISE_FILTERS = 4;
    const size_t CONV1_POINTWISE_KERNEL_SIZE = 1;

    const size_t CONV2_DEPTHWISE_FILTERS = 4;
    const size_t CONV2_DEPTHWISE_KERNEL_SIZE = 3;
    const size_t CONV2_DEPTHWISE_STRIDE = 2;
    const size_t CONV2_DEPTHWISE_PADDING = 1;

    const size_t CONV2_POINTWISE_FILTERS = 8;
    const size_t CONV2_POINTWISE_KERNEL_SIZE = 1;

    const size_t CONV3_DEPTHWISE_FILTERS = 8;
    const size_t CONV3_DEPTHWISE_KERNEL_SIZE = 3;
    const size_t CONV3_DEPTHWISE_STRIDE = 2;
    const size_t CONV3_DEPTHWISE_PADDING = 1;

    const size_t CONV3_POINTWISE_FILTERS = 16;
    const size_t CONV3_POINTWISE_KERNEL_SIZE = 1;

    const size_t CONV4_DEPTHWISE_FILTERS = 16;
    const size_t CONV4_DEPTHWISE_KERNEL_SIZE = 3;
    const size_t CONV4_DEPTHWISE_STRIDE = 4;
    const size_t CONV4_DEPTHWISE_PADDING = 0;

    const size_t CONV4_POINTWISE_FILTERS = NUM_CLASSES;
    const size_t CONV4_POINTWISE_KERNEL_SIZE = 1;

    const raul::dtype acc1 = TODTYPE(8.23f);
    const raul::dtype acc2 = TODTYPE(85.57f);

    raul::MNIST mnist;
    ASSERT_EQ(mnist.loadingData(tools::getTestAssetsDir() / "MNIST"), true);

    raul::Workflow work;
    work.add<raul::DataLayer>("data", raul::DataParams{ { "data", "labels" }, MNIST_CHANNELS, MNIST_SIZE, MNIST_SIZE, NUM_CLASSES });
    work.add<raul::Convolution2DLayer>("conv1Depthwise",
                                       raul::Convolution2DParams{ { "data" }, { "conv1d" }, CONV1_DEPTHWISE_KERNEL_SIZE, CONV1_DEPTHWISE_FILTERS, CONV1_DEPTHWISE_STRIDE, CONV1_DEPTHWISE_PADDING });
    work.add<raul::Convolution2DLayer>("conv1Pointwise", raul::Convolution2DParams{ { "conv1d" }, { "conv1p" }, CONV1_POINTWISE_KERNEL_SIZE, CONV1_POINTWISE_FILTERS });
    work.add<raul::ConvolutionDepthwiseLayer>(
        "conv2Depthwise", raul::Convolution2DParams{ { "conv1p" }, { "conv2d" }, CONV2_DEPTHWISE_KERNEL_SIZE, CONV2_DEPTHWISE_FILTERS, CONV2_DEPTHWISE_STRIDE, CONV2_DEPTHWISE_PADDING });
    work.add<raul::Convolution2DLayer>("conv2Pointwise", raul::Convolution2DParams{ { "conv2d" }, { "conv2p" }, CONV2_POINTWISE_KERNEL_SIZE, CONV2_POINTWISE_FILTERS });
    work.add<raul::ConvolutionDepthwiseLayer>(
        "conv3Depthwise", raul::Convolution2DParams{ { "conv2p" }, { "conv3d" }, CONV3_DEPTHWISE_KERNEL_SIZE, CONV3_DEPTHWISE_FILTERS, CONV3_DEPTHWISE_STRIDE, CONV3_DEPTHWISE_PADDING });
    work.add<raul::Convolution2DLayer>("conv3Pointwise", raul::Convolution2DParams{ { "conv3d" }, { "conv3p" }, CONV3_POINTWISE_KERNEL_SIZE, CONV3_POINTWISE_FILTERS });
    work.add<raul::ConvolutionDepthwiseLayer>(
        "conv4Depthwise", raul::Convolution2DParams{ { "conv3p" }, { "conv4d" }, CONV4_DEPTHWISE_KERNEL_SIZE, CONV4_DEPTHWISE_FILTERS, CONV4_DEPTHWISE_STRIDE, CONV4_DEPTHWISE_PADDING });
    work.add<raul::Convolution2DLayer>("conv4Pointwise", raul::Convolution2DParams{ { "conv4d" }, { "conv4p" }, CONV4_POINTWISE_KERNEL_SIZE, CONV4_POINTWISE_FILTERS });
    work.add<raul::SoftMaxActivation>("softmax", raul::BasicParamsWithDim{ { "conv4p" }, { "softmax" } });
    work.add<raul::CrossEntropyLoss>("loss", raul::LossParams{ { "softmax", "labels" }, { "loss" }, "batch_mean" });

    work.preparePipelines();
    work.setBatchSize(BATCH_SIZE);
    work.prepareMemoryForTraining();

    raul::MemoryManager& memory_manager = work.getMemoryManager();
    raul::DataLoader dataLoader;

    memory_manager["conv1Depthwise::Weights"] = dataLoader.loadFilters((tools::getTestAssetsDir() / "test_cnn_layer" / "mnistDepthwise" / "0_conv1_depthwise.weight_").string(),
                                                                       0,
                                                                       ".data",
                                                                       CONV1_DEPTHWISE_KERNEL_SIZE,
                                                                       CONV1_DEPTHWISE_KERNEL_SIZE,
                                                                       MNIST_CHANNELS,
                                                                       CONV1_DEPTHWISE_FILTERS);
    memory_manager["conv1Depthwise::Biases"] = dataLoader.loadData(tools::getTestAssetsDir() / "test_cnn_layer" / "mnistDepthwise" / "0_conv1_depthwise.bias.data", 1, CONV1_DEPTHWISE_FILTERS);

    memory_manager["conv1Pointwise::Weights"] = dataLoader.loadFilters((tools::getTestAssetsDir() / "test_cnn_layer" / "mnistDepthwise" / "0_conv1_pointwise.weight_").string(),
                                                                       0,
                                                                       ".data",
                                                                       CONV1_POINTWISE_KERNEL_SIZE,
                                                                       CONV1_POINTWISE_KERNEL_SIZE,
                                                                       CONV1_DEPTHWISE_FILTERS,
                                                                       CONV1_POINTWISE_FILTERS);
    memory_manager["conv1Pointwise::Biases"] = dataLoader.loadData(tools::getTestAssetsDir() / "test_cnn_layer" / "mnistDepthwise" / "0_conv1_pointwise.bias.data", 1, CONV1_POINTWISE_FILTERS);

    memory_manager["conv2Depthwise::Weights"] = dataLoader.loadFilters((tools::getTestAssetsDir() / "test_cnn_layer" / "mnistDepthwise" / "0_conv2_depthwise.weight_").string(),
                                                                       0,
                                                                       ".data",
                                                                       CONV2_DEPTHWISE_KERNEL_SIZE,
                                                                       CONV2_DEPTHWISE_KERNEL_SIZE,
                                                                       1,
                                                                       CONV2_DEPTHWISE_FILTERS);
    memory_manager["conv2Depthwise::Biases"] = dataLoader.loadData(tools::getTestAssetsDir() / "test_cnn_layer" / "mnistDepthwise" / "0_conv2_depthwise.bias.data", 1, CONV2_DEPTHWISE_FILTERS);

    memory_manager["conv2Pointwise::Weights"] = dataLoader.loadFilters((tools::getTestAssetsDir() / "test_cnn_layer" / "mnistDepthwise" / "0_conv2_pointwise.weight_").string(),
                                                                       0,
                                                                       ".data",
                                                                       CONV2_POINTWISE_KERNEL_SIZE,
                                                                       CONV2_POINTWISE_KERNEL_SIZE,
                                                                       CONV2_DEPTHWISE_FILTERS,
                                                                       CONV2_POINTWISE_FILTERS);
    memory_manager["conv2Pointwise::Biases"] = dataLoader.loadData(tools::getTestAssetsDir() / "test_cnn_layer" / "mnistDepthwise" / "0_conv2_pointwise.bias.data", 1, CONV2_POINTWISE_FILTERS);

    memory_manager["conv3Depthwise::Weights"] = dataLoader.loadFilters((tools::getTestAssetsDir() / "test_cnn_layer" / "mnistDepthwise" / "0_conv3_depthwise.weight_").string(),
                                                                       0,
                                                                       ".data",
                                                                       CONV3_DEPTHWISE_KERNEL_SIZE,
                                                                       CONV3_DEPTHWISE_KERNEL_SIZE,
                                                                       1,
                                                                       CONV3_DEPTHWISE_FILTERS);
    memory_manager["conv3Depthwise::Biases"] = dataLoader.loadData(tools::getTestAssetsDir() / "test_cnn_layer" / "mnistDepthwise" / "0_conv3_depthwise.bias.data", 1, CONV3_DEPTHWISE_FILTERS);

    memory_manager["conv3Pointwise::Weights"] = dataLoader.loadFilters((tools::getTestAssetsDir() / "test_cnn_layer" / "mnistDepthwise" / "0_conv3_pointwise.weight_").string(),
                                                                       0,
                                                                       ".data",
                                                                       CONV3_POINTWISE_KERNEL_SIZE,
                                                                       CONV3_POINTWISE_KERNEL_SIZE,
                                                                       CONV3_DEPTHWISE_FILTERS,
                                                                       CONV3_POINTWISE_FILTERS);
    memory_manager["conv3Pointwise::Biases"] = dataLoader.loadData(tools::getTestAssetsDir() / "test_cnn_layer" / "mnistDepthwise" / "0_conv3_pointwise.bias.data", 1, CONV3_POINTWISE_FILTERS);

    memory_manager["conv4Depthwise::Weights"] = dataLoader.loadFilters((tools::getTestAssetsDir() / "test_cnn_layer" / "mnistDepthwise" / "0_conv4_depthwise.weight_").string(),
                                                                       0,
                                                                       ".data",
                                                                       CONV4_DEPTHWISE_KERNEL_SIZE,
                                                                       CONV4_DEPTHWISE_KERNEL_SIZE,
                                                                       1,
                                                                       CONV4_DEPTHWISE_FILTERS);
    memory_manager["conv4Depthwise::Biases"] = dataLoader.loadData(tools::getTestAssetsDir() / "test_cnn_layer" / "mnistDepthwise" / "0_conv4_depthwise.bias.data", 1, CONV4_DEPTHWISE_FILTERS);

    memory_manager["conv4Pointwise::Weights"] = dataLoader.loadFilters((tools::getTestAssetsDir() / "test_cnn_layer" / "mnistDepthwise" / "0_conv4_pointwise.weight_").string(),
                                                                       0,
                                                                       ".data",
                                                                       CONV4_POINTWISE_KERNEL_SIZE,
                                                                       CONV4_POINTWISE_KERNEL_SIZE,
                                                                       CONV4_DEPTHWISE_FILTERS,
                                                                       CONV4_POINTWISE_FILTERS);
    memory_manager["conv4Pointwise::Biases"] = dataLoader.loadData(tools::getTestAssetsDir() / "test_cnn_layer" / "mnistDepthwise" / "0_conv4_pointwise.bias.data", 1, CONV4_POINTWISE_FILTERS);

    const size_t stepsAmountTrain = mnist.getTrainImageAmount() / BATCH_SIZE;
    raul::Tensor& idealLosses = dataLoader.createTensor(stepsAmountTrain / 100);
    raul::DataLoader::readArrayFromTextFile(tools::getTestAssetsDir() / "test_cnn_layer" / "mnistDepthwise" / "loss.data", idealLosses, 1, idealLosses.size());

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
}

#if defined(ANDROID)
TEST(TestCNNDepthwiseMnist, TrainingFP16)
{
    PROFILE_TEST

    const raul::dtype LEARNING_RATE = TODTYPE(0.1);
    const size_t BATCH_SIZE = 50;
    const raul::dtype EPSILON_ACCURACY = TODTYPE(1e-2);

    const size_t NUM_CLASSES = 10;

    const size_t MNIST_SIZE = 28;
    const size_t MNIST_CHANNELS = 1;

    const size_t CONV1_DEPTHWISE_FILTERS = 1;
    const size_t CONV1_DEPTHWISE_KERNEL_SIZE = 3;
    const size_t CONV1_DEPTHWISE_STRIDE = 2;
    const size_t CONV1_DEPTHWISE_PADDING = 1;

    const size_t CONV1_POINTWISE_FILTERS = 4;
    const size_t CONV1_POINTWISE_KERNEL_SIZE = 1;

    const size_t CONV2_DEPTHWISE_FILTERS = 4;
    const size_t CONV2_DEPTHWISE_KERNEL_SIZE = 3;
    const size_t CONV2_DEPTHWISE_STRIDE = 2;
    const size_t CONV2_DEPTHWISE_PADDING = 1;

    const size_t CONV2_POINTWISE_FILTERS = 8;
    const size_t CONV2_POINTWISE_KERNEL_SIZE = 1;

    const size_t CONV3_DEPTHWISE_FILTERS = 8;
    const size_t CONV3_DEPTHWISE_KERNEL_SIZE = 3;
    const size_t CONV3_DEPTHWISE_STRIDE = 2;
    const size_t CONV3_DEPTHWISE_PADDING = 1;

    const size_t CONV3_POINTWISE_FILTERS = 16;
    const size_t CONV3_POINTWISE_KERNEL_SIZE = 1;

    const size_t CONV4_DEPTHWISE_FILTERS = 16;
    const size_t CONV4_DEPTHWISE_KERNEL_SIZE = 3;
    const size_t CONV4_DEPTHWISE_STRIDE = 4;
    const size_t CONV4_DEPTHWISE_PADDING = 0;

    const size_t CONV4_POINTWISE_FILTERS = NUM_CLASSES;
    const size_t CONV4_POINTWISE_KERNEL_SIZE = 1;

    const raul::dtype acc1 = TODTYPE(8.23f);
    const raul::dtype acc2 = TODTYPE(85.57f);

    raul::MNIST mnist;
    ASSERT_EQ(mnist.loadingData(tools::getTestAssetsDir() / "MNIST"), true);

    raul::Workflow work(raul::CompressionMode::NONE, raul::CalculationMode::DETERMINISTIC, raul::AllocationMode::STANDARD, raul::ExecutionTarget::CPUFP16);
    work.add<raul::DataLayer>("data", raul::DataParams{ { "data", "labels" }, MNIST_CHANNELS, MNIST_SIZE, MNIST_SIZE, NUM_CLASSES });
    work.add<raul::Convolution2DLayer>("conv1Depthwise",
                                       raul::Convolution2DParams{ { "data" }, { "conv1d" }, CONV1_DEPTHWISE_KERNEL_SIZE, CONV1_DEPTHWISE_FILTERS, CONV1_DEPTHWISE_STRIDE, CONV1_DEPTHWISE_PADDING });
    work.add<raul::Convolution2DLayer>("conv1Pointwise", raul::Convolution2DParams{ { "conv1d" }, { "conv1p" }, CONV1_POINTWISE_KERNEL_SIZE, CONV1_POINTWISE_FILTERS });
    work.add<raul::ConvolutionDepthwiseLayer>(
        "conv2Depthwise", raul::Convolution2DParams{ { "conv1p" }, { "conv2d" }, CONV2_DEPTHWISE_KERNEL_SIZE, CONV2_DEPTHWISE_FILTERS, CONV2_DEPTHWISE_STRIDE, CONV2_DEPTHWISE_PADDING });
    work.add<raul::Convolution2DLayer>("conv2Pointwise", raul::Convolution2DParams{ { "conv2d" }, { "conv2p" }, CONV2_POINTWISE_KERNEL_SIZE, CONV2_POINTWISE_FILTERS });
    work.add<raul::ConvolutionDepthwiseLayer>(
        "conv3Depthwise", raul::Convolution2DParams{ { "conv2p" }, { "conv3d" }, CONV3_DEPTHWISE_KERNEL_SIZE, CONV3_DEPTHWISE_FILTERS, CONV3_DEPTHWISE_STRIDE, CONV3_DEPTHWISE_PADDING });
    work.add<raul::Convolution2DLayer>("conv3Pointwise", raul::Convolution2DParams{ { "conv3d" }, { "conv3p" }, CONV3_POINTWISE_KERNEL_SIZE, CONV3_POINTWISE_FILTERS });
    work.add<raul::ConvolutionDepthwiseLayer>(
        "conv4Depthwise", raul::Convolution2DParams{ { "conv3p" }, { "conv4d" }, CONV4_DEPTHWISE_KERNEL_SIZE, CONV4_DEPTHWISE_FILTERS, CONV4_DEPTHWISE_STRIDE, CONV4_DEPTHWISE_PADDING });
    work.add<raul::Convolution2DLayer>("conv4Pointwise", raul::Convolution2DParams{ { "conv4d" }, { "conv4p" }, CONV4_POINTWISE_KERNEL_SIZE, CONV4_POINTWISE_FILTERS });
    work.add<raul::LogSoftMaxActivation>("softmax", raul::BasicParamsWithDim{ { "conv4p" }, { "softmax" } });
    work.add<raul::NLLLoss>("loss", raul::LossParams{ { "softmax", "labels" }, { "loss" }, "batch_mean" });

    work.preparePipelines();
    work.setBatchSize(BATCH_SIZE);
    work.prepareMemoryForTraining();

    auto& memory_manager = work.getMemoryManager<raul::MemoryManagerFP16>();
    raul::DataLoader dataLoader;

    memory_manager["conv1Depthwise::Weights"] = dataLoader.loadFilters((tools::getTestAssetsDir() / "test_cnn_layer" / "mnistDepthwise" / "0_conv1_depthwise.weight_").string(),
                                                                       0,
                                                                       ".txt",
                                                                       CONV1_DEPTHWISE_KERNEL_SIZE,
                                                                       CONV1_DEPTHWISE_KERNEL_SIZE,
                                                                       MNIST_CHANNELS,
                                                                       CONV1_DEPTHWISE_FILTERS);
    memory_manager["conv1Depthwise::Biases"] = dataLoader.loadData(tools::getTestAssetsDir() / "test_cnn_layer" / "mnistDepthwise" / "0_conv1_depthwise.bias.txt", 1, CONV1_DEPTHWISE_FILTERS);

    memory_manager["conv1Pointwise::Weights"] = dataLoader.loadFilters((tools::getTestAssetsDir() / "test_cnn_layer" / "mnistDepthwise" / "0_conv1_pointwise.weight_").string(),
                                                                       0,
                                                                       ".txt",
                                                                       CONV1_POINTWISE_KERNEL_SIZE,
                                                                       CONV1_POINTWISE_KERNEL_SIZE,
                                                                       CONV1_DEPTHWISE_FILTERS,
                                                                       CONV1_POINTWISE_FILTERS);
    memory_manager["conv1Pointwise::Biases"] = dataLoader.loadData(tools::getTestAssetsDir() / "test_cnn_layer" / "mnistDepthwise" / "0_conv1_pointwise.bias.txt", 1, CONV1_POINTWISE_FILTERS);

    memory_manager["conv2Depthwise::Weights"] = dataLoader.loadFilters((tools::getTestAssetsDir() / "test_cnn_layer" / "mnistDepthwise" / "0_conv2_depthwise.weight_").string(),
                                                                       0,
                                                                       ".txt",
                                                                       CONV2_DEPTHWISE_KERNEL_SIZE,
                                                                       CONV2_DEPTHWISE_KERNEL_SIZE,
                                                                       1,
                                                                       CONV2_DEPTHWISE_FILTERS);
    memory_manager["conv2Depthwise::Biases"] = dataLoader.loadData(tools::getTestAssetsDir() / "test_cnn_layer" / "mnistDepthwise" / "0_conv2_depthwise.bias.txt", 1, CONV2_DEPTHWISE_FILTERS);

    memory_manager["conv2Pointwise::Weights"] = dataLoader.loadFilters((tools::getTestAssetsDir() / "test_cnn_layer" / "mnistDepthwise" / "0_conv2_pointwise.weight_").string(),
                                                                       0,
                                                                       ".txt",
                                                                       CONV2_POINTWISE_KERNEL_SIZE,
                                                                       CONV2_POINTWISE_KERNEL_SIZE,
                                                                       CONV2_DEPTHWISE_FILTERS,
                                                                       CONV2_POINTWISE_FILTERS);
    memory_manager["conv2Pointwise::Biases"] = dataLoader.loadData(tools::getTestAssetsDir() / "test_cnn_layer" / "mnistDepthwise" / "0_conv2_pointwise.bias.txt", 1, CONV2_POINTWISE_FILTERS);

    memory_manager["conv3Depthwise::Weights"] = dataLoader.loadFilters((tools::getTestAssetsDir() / "test_cnn_layer" / "mnistDepthwise" / "0_conv3_depthwise.weight_").string(),
                                                                       0,
                                                                       ".txt",
                                                                       CONV3_DEPTHWISE_KERNEL_SIZE,
                                                                       CONV3_DEPTHWISE_KERNEL_SIZE,
                                                                       1,
                                                                       CONV3_DEPTHWISE_FILTERS);
    memory_manager["conv3Depthwise::Biases"] = dataLoader.loadData(tools::getTestAssetsDir() / "test_cnn_layer" / "mnistDepthwise" / "0_conv3_depthwise.bias.txt", 1, CONV3_DEPTHWISE_FILTERS);

    memory_manager["conv3Pointwise::Weights"] = dataLoader.loadFilters((tools::getTestAssetsDir() / "test_cnn_layer" / "mnistDepthwise" / "0_conv3_pointwise.weight_").string(),
                                                                       0,
                                                                       ".txt",
                                                                       CONV3_POINTWISE_KERNEL_SIZE,
                                                                       CONV3_POINTWISE_KERNEL_SIZE,
                                                                       CONV3_DEPTHWISE_FILTERS,
                                                                       CONV3_POINTWISE_FILTERS);
    memory_manager["conv3Pointwise::Biases"] = dataLoader.loadData(tools::getTestAssetsDir() / "test_cnn_layer" / "mnistDepthwise" / "0_conv3_pointwise.bias.txt", 1, CONV3_POINTWISE_FILTERS);

    memory_manager["conv4Depthwise::Weights"] = dataLoader.loadFilters((tools::getTestAssetsDir() / "test_cnn_layer" / "mnistDepthwise" / "0_conv4_depthwise.weight_").string(),
                                                                       0,
                                                                       ".txt",
                                                                       CONV4_DEPTHWISE_KERNEL_SIZE,
                                                                       CONV4_DEPTHWISE_KERNEL_SIZE,
                                                                       1,
                                                                       CONV4_DEPTHWISE_FILTERS);
    memory_manager["conv4Depthwise::Biases"] = dataLoader.loadData(tools::getTestAssetsDir() / "test_cnn_layer" / "mnistDepthwise" / "0_conv4_depthwise.bias.txt", 1, CONV4_DEPTHWISE_FILTERS);

    memory_manager["conv4Pointwise::Weights"] = dataLoader.loadFilters((tools::getTestAssetsDir() / "test_cnn_layer" / "mnistDepthwise" / "0_conv4_pointwise.weight_").string(),
                                                                       0,
                                                                       ".txt",
                                                                       CONV4_POINTWISE_KERNEL_SIZE,
                                                                       CONV4_POINTWISE_KERNEL_SIZE,
                                                                       CONV4_DEPTHWISE_FILTERS,
                                                                       CONV4_POINTWISE_FILTERS);
    memory_manager["conv4Pointwise::Biases"] = dataLoader.loadData(tools::getTestAssetsDir() / "test_cnn_layer" / "mnistDepthwise" / "0_conv4_pointwise.bias.txt", 1, CONV4_POINTWISE_FILTERS);

    const size_t stepsAmountTrain = mnist.getTrainImageAmount() / BATCH_SIZE;

    raul::dtype testAcc = mnist.testNetwork(work);
    CHECK_NEAR(testAcc, acc1, EPSILON_ACCURACY);
    printf("Test accuracy = %f\n", testAcc);
    auto sgd = std::make_shared<raul::optimizers::SGD>(LEARNING_RATE);
    for (size_t q = 0; q < stepsAmountTrain; ++q)
    {
        raul::dtype testLoss = mnist.oneTrainIteration(work, sgd.get(), q);
        if (q % 100 == 0)
        {
            printf("iteration = %d, loss = %f\n", static_cast<uint32_t>(q), testLoss);
        }
    }
    testAcc = mnist.testNetwork(work);
    CHECK_NEAR(testAcc, acc2, EPSILON_ACCURACY);
    printf("Test accuracy = %f\n", testAcc);
}
#endif

} // UT namespace
