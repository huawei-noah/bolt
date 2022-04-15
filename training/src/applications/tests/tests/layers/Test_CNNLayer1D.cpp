// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <tests/GTestExtensions.h>
#include <tests/tools/TestTools.h>

#include <training/api/API.h>
#include <training/layers/basic/DataLayer.h>
#include <training/layers/basic/trainable/Convolution1DLayer.h>
#include <training/network/Layers.h>
#include <training/network/Workflow.h>
#include <training/optimizers/SGD.h>

namespace UT
{

using dtype = raul::dtype;
using shape = raul::shape;
using dvec = std::vector<dtype>;

struct ConvParams
{
    size_t KERNEL_SIZE;
    size_t CHANNELS;
    size_t STRIDE;
    size_t PADDING;
    size_t DILATION;
    size_t GROUPS;
};

struct VariousConv1dLayerParameters : public testing::TestWithParam<std::tuple<ConvParams, shape, dvec, dvec, dvec, dvec>>
{
    static constexpr dtype EPSILON = 1e-6_dt;
    size_t kernelSize = std::get<0>(GetParam()).KERNEL_SIZE;
    size_t channels = std::get<0>(GetParam()).CHANNELS;
    size_t stride = std::get<0>(GetParam()).STRIDE;
    size_t padding = std::get<0>(GetParam()).PADDING;
    size_t dilation = std::get<0>(GetParam()).DILATION;
    size_t groups = std::get<0>(GetParam()).GROUPS;

    const dvec& forwardPassResult = std::get<2>(GetParam());
    const dvec& backwardPassResult = std::get<3>(GetParam());
    const dvec& weightsGradientResult = std::get<4>(GetParam());
    const dvec& biasGradientResult = std::get<5>(GetParam());

    dvec weightsGradientAfterTwoBackwardPasses;
    dvec biasGradientAfterTwoBackwardPasses;

    void SetUp() final
    {
        std::transform(weightsGradientResult.begin(), weightsGradientResult.end(), std::back_inserter(weightsGradientAfterTwoBackwardPasses), [](dtype v) { return v * 2; });
        std::transform(biasGradientResult.begin(), biasGradientResult.end(), std::back_inserter(biasGradientAfterTwoBackwardPasses), [](dtype v) { return v * 2; });
    }
};

// conv1d_test.py
TEST_P(VariousConv1dLayerParameters, Convolution1DUnit)
{
    PROFILE_TEST
    using namespace raul;

    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    work.add<DataLayer>("data", DataParams{ { "in" }, 2, 1, 3 });
    Convolution1DParams params{ "in", "cnn1", kernelSize, channels, stride, padding, dilation, groups, true };
    Convolution1DLayer cnnLayer("cnn1", params, networkParameters);
    params.print(std::cout);
    std::cout << std::endl;

    TENSORS_CREATE(2);

    Common::arange(memory_manager["in"].begin(), memory_manager["in"].end(), 1_dt);
    memory_manager["cnn1::Weights"] = 1_dt;
    memory_manager["cnn1::Biases"] = 1_dt;

    ASSERT_NO_THROW(cnnLayer.forwardCompute(raul::NetworkMode::Train));
    memory_manager[Name("cnn1").grad()] = 1_dt;
    ASSERT_EQ(memory_manager["cnn1"].getShape(), std::get<1>(GetParam()));
    ASSERT_INTERVALS_NEAR(memory_manager["cnn1"].begin(), memory_manager["cnn1"].end(), forwardPassResult.begin(), forwardPassResult.end(), EPSILON);

    ASSERT_NO_THROW(cnnLayer.backwardCompute());

    ASSERT_EQ(memory_manager["in"].getShape(), memory_manager[raul::Name("in").grad()].getShape());
    ASSERT_INTERVALS_NEAR(memory_manager[raul::Name("in").grad()].begin(), memory_manager[raul::Name("in").grad()].end(), backwardPassResult.begin(), backwardPassResult.end(), EPSILON);

    ASSERT_EQ(memory_manager["cnn1::WeightsGradient"].getShape(), memory_manager["cnn1::Weights"].getShape());
    ASSERT_INTERVALS_NEAR(memory_manager["cnn1::WeightsGradient"].begin(), memory_manager["cnn1::WeightsGradient"].end(), weightsGradientResult.begin(), weightsGradientResult.end(), EPSILON);

    ASSERT_EQ(memory_manager["cnn1::BiasesGradient"].getShape(), memory_manager["cnn1::Biases"].getShape());
    ASSERT_INTERVALS_NEAR(memory_manager["cnn1::BiasesGradient"].begin(), memory_manager["cnn1::BiasesGradient"].end(), biasGradientResult.begin(), biasGradientResult.end(), EPSILON);

    memory_manager.clear();
}

TEST_P(VariousConv1dLayerParameters, ShouldAccumulateGradientsDuringBackwardPassByDefaultUnit)
{
    PROFILE_TEST
    using namespace raul;

    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    work.add<DataLayer>("data", DataParams{ { "in" }, 2, 1, 3 });
    Convolution1DParams params{ "in", "cnn1", kernelSize, channels, stride, padding, dilation, groups, true };
    Convolution1DLayer cnnLayer("cnn1", params, networkParameters);
    params.print(std::cout);
    std::cout << std::endl;

    TENSORS_CREATE(2);

    Common::arange(memory_manager["in"].begin(), memory_manager["in"].end(), 1_dt);
    memory_manager["cnn1::Weights"] = 1_dt;
    memory_manager["cnn1::Biases"] = 1_dt;

    ASSERT_NO_THROW(cnnLayer.forwardCompute(raul::NetworkMode::Train));
    memory_manager[Name("cnn1").grad()] = 1_dt;
    ASSERT_NO_THROW(cnnLayer.backwardCompute());
    ASSERT_NO_THROW(cnnLayer.backwardCompute());

    ASSERT_INTERVALS_NEAR(memory_manager["cnn1::WeightsGradient"].begin(),
                          memory_manager["cnn1::WeightsGradient"].end(),
                          weightsGradientAfterTwoBackwardPasses.begin(),
                          weightsGradientAfterTwoBackwardPasses.end(),
                          EPSILON);

    ASSERT_INTERVALS_NEAR(
        memory_manager["cnn1::BiasesGradient"].begin(), memory_manager["cnn1::BiasesGradient"].end(), biasGradientAfterTwoBackwardPasses.begin(), biasGradientAfterTwoBackwardPasses.end(), EPSILON);

    memory_manager.clear();
}

INSTANTIATE_TEST_SUITE_P(TestCNN1DLayer,
                         VariousConv1dLayerParameters,
                         testing::Values(std::make_tuple(ConvParams{ 3u, 2u, 1u, 0u, 1u, 1u },
                                                         shape{ 2u, 2u, 1u, 1u },
                                                         dvec{ 22_dt, 22_dt, 58_dt, 58_dt },
                                                         dvec{ 2_dt, 2_dt, 2_dt, 2_dt, 2_dt, 2_dt, 2_dt, 2_dt, 2_dt, 2_dt, 2_dt, 2_dt },
                                                         dvec{ 8_dt, 10_dt, 12_dt, 14_dt, 16_dt, 18_dt, 8_dt, 10_dt, 12_dt, 14_dt, 16_dt, 18_dt },
                                                         dvec{ 2., 2. }),
                                         std::make_tuple(ConvParams{ 3u, 2u, 1u, 0, 1u, 2u },
                                                         shape{ 2u, 2u, 1u, 1u },
                                                         dvec{ 7_dt, 16_dt, 25_dt, 34_dt },
                                                         dvec{ 1_dt, 1_dt, 1_dt, 1_dt, 1_dt, 1_dt, 1_dt, 1_dt, 1_dt, 1_dt, 1_dt, 1_dt },
                                                         dvec{ 8_dt, 10_dt, 12_dt, 14_dt, 16_dt, 18_dt },
                                                         dvec{ 2., 2. }),
                                         std::make_tuple(ConvParams{ 3u, 2u, 1u, 2u, 1u, 1u },
                                                         shape{ 2u, 2u, 1u, 5u },
                                                         dvec{ 6., 13., 22., 17., 10., 6., 13., 22., 17., 10., 18., 37., 58., 41., 22., 18., 37., 58., 41., 22. },
                                                         dvec{ 6_dt, 6_dt, 6_dt, 6_dt, 6_dt, 6_dt, 6_dt, 6_dt, 6_dt, 6_dt, 6_dt, 6_dt },
                                                         dvec{ 30_dt, 30_dt, 30_dt, 48_dt, 48_dt, 48_dt, 30_dt, 30_dt, 30_dt, 48_dt, 48_dt, 48_dt },
                                                         dvec{ 10., 10. }),
                                         std::make_tuple(ConvParams{ 3u, 2u, 2u, 2u, 1u, 1u },
                                                         shape{ 2u, 2u, 1u, 3u },
                                                         dvec{ 6_dt, 22_dt, 10_dt, 6_dt, 22_dt, 10_dt, 18_dt, 58_dt, 22_dt, 18_dt, 58_dt, 22_dt },
                                                         dvec{ 4_dt, 2_dt, 4_dt, 4_dt, 2_dt, 4_dt, 4_dt, 2_dt, 4_dt, 4_dt, 2_dt, 4_dt },
                                                         dvec{ 20_dt, 10_dt, 20_dt, 32_dt, 16_dt, 32_dt, 20_dt, 10_dt, 20_dt, 32_dt, 16_dt, 32_dt },
                                                         dvec{ 6., 6. }),
                                         std::make_tuple(ConvParams{ 3u, 1u, 3u, 2u, 1u, 1u },
                                                         shape{ 2u, 1u, 1u, 2u },
                                                         dvec{ 6_dt, 17_dt, 18_dt, 41_dt },
                                                         dvec{ 1_dt, 1_dt, 1_dt, 1_dt, 1_dt, 1_dt, 1_dt, 1_dt, 1_dt, 1_dt, 1_dt, 1_dt },
                                                         dvec{ 10_dt, 12_dt, 8_dt, 16_dt, 18_dt, 14_dt },
                                                         dvec{ 4. }),
                                         std::make_tuple(ConvParams{ 3u, 1u, 5u, 2u, 1u, 1u },
                                                         shape{ 2u, 1u, 1u, 1u },
                                                         dvec{ 6_dt, 18_dt },
                                                         dvec{ 1_dt, 0_dt, 0_dt, 1_dt, 0_dt, 0_dt, 1_dt, 0_dt, 0_dt, 1_dt, 0_dt, 0_dt },
                                                         dvec{ 0_dt, 0_dt, 8_dt, 0_dt, 0_dt, 14_dt },
                                                         dvec{ 2. }),
                                         std::make_tuple(ConvParams{ 3u, 4u, 5u, 2u, 1u, 2u },
                                                         shape{ 2u, 4u, 1u, 1u },
                                                         dvec{ 2_dt, 2_dt, 5_dt, 5_dt, 8_dt, 8_dt, 11_dt, 11_dt },
                                                         dvec{ 2_dt, 0_dt, 0_dt, 2_dt, 0_dt, 0_dt, 2_dt, 0_dt, 0_dt, 2_dt, 0_dt, 0_dt },
                                                         dvec{ 0_dt, 0_dt, 8_dt, 0_dt, 0_dt, 8_dt, 0_dt, 0_dt, 14_dt, 0_dt, 0_dt, 14_dt },
                                                         dvec{ 2., 2., 2., 2. })));

// cnnBiasTrain1d.py
TEST(TestCNN1DLayer, BiasTrain)
{
    PROFILE_TEST
    using namespace raul;

    const auto LEARNING_RATE = 0.01_dt;
    const size_t BATCH_SIZE = 50;
    const auto EPSILON_ACCURACY = 1e-2_dt;
    const auto EPSILON_LOSS = 1e-5_dt;
    const auto acc1 = 12.36_dt;
    const auto acc2 = 89.6_dt;
    const size_t NUM_CLASSES = 10;

    const size_t MNIST_SIZE = 28;
    const size_t MNIST_CHANNELS = 1;
    const size_t CONV1_FILTERS = 16;
    const size_t CONV1_KERNEL_SIZE = 5;
    const size_t CONV1_STRIDE = 2;
    const size_t CONV1_PADDING = 2;
    const size_t CONV1_DILATION = 1;
    const size_t CONV1_GROUPS = 1;

    const size_t FC1_SIZE = CONV1_FILTERS * 392;

    MNIST mnist;
    ASSERT_EQ(mnist.loadingData(tools::getTestAssetsDir() / "MNIST"), true);

    Workflow work;
    work.add<DataLayer>("data", DataParams{ { "data", "labels" }, MNIST_CHANNELS, MNIST_SIZE, MNIST_SIZE, NUM_CLASSES });
    work.add<raul::ReshapeLayer>("reshape1", ViewParams{ "data", "datar", MNIST_CHANNELS, 1, -1 });
    work.add<raul::Convolution1DLayer>("conv1", Convolution1DParams{ { "datar" }, { "conv1" }, CONV1_KERNEL_SIZE, CONV1_FILTERS, CONV1_STRIDE, CONV1_PADDING, CONV1_DILATION, CONV1_GROUPS, true });
    work.add<raul::ReshapeLayer>("reshape2", ViewParams{ "conv1", "conv1r", 1, 1, -1 });
    work.add<raul::LinearLayer>("fc1", LinearParams{ { "conv1r" }, { "fc1" }, NUM_CLASSES, true });
    work.add<raul::SoftMaxActivation>("softmax", BasicParamsWithDim{ { "fc1" }, { "softmax" } });
    work.add<raul::CrossEntropyLoss>("loss", LossParams{ { "softmax", "labels" }, { "loss" }, "batch_mean" });

    work.preparePipelines();
    work.setBatchSize(BATCH_SIZE);
    work.prepareMemoryForTraining();

    MemoryManager& memory_manager = work.getMemoryManager();
    DataLoader dataLoader;

    memory_manager["conv1::Weights"] = dataLoader.loadData(tools::getTestAssetsDir() / "test_cnn_layer" / "conv1d" / "0_conv1.weight.data", CONV1_FILTERS, CONV1_KERNEL_SIZE);
    memory_manager["fc1::Weights"] = dataLoader.loadData(tools::getTestAssetsDir() / "test_cnn_layer" / "conv1d" / "0_fc1.weight.data", NUM_CLASSES, FC1_SIZE);
    memory_manager["conv1::Biases"] = dataLoader.loadData(tools::getTestAssetsDir() / "test_cnn_layer" / "conv1d" / "0_conv1.bias.data", CONV1_FILTERS, 1);
    memory_manager["fc1::Biases"] = dataLoader.loadData(tools::getTestAssetsDir() / "test_cnn_layer" / "conv1d" / "0_fc1.bias.data", NUM_CLASSES, 1);

    const size_t stepsAmountTrain = mnist.getTrainImageAmount() / BATCH_SIZE;

    Tensor idealLosses = { 2.577795_dt, 1.178468_dt, 0.626785_dt, 0.340559_dt, 0.544606_dt, 0.236532_dt, 0.439990_dt, 0.192519_dt, 0.243940_dt, 0.625617_dt, 0.338855_dt, 0.344555_dt };

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
    printf("Time taken = %.3fs \n", mnist.getTrainingTime());
    printf("Memory taken = %.2fMB \n\n", static_cast<float>(work.getMemoryManager().getTotalMemory()) / (1024.0f * 1024.0f));
}

#if defined(ANDROID)
TEST(TestCNN1DLayer, BiasTrainFP16)
{
    PROFILE_TEST

    using namespace raul;

    const auto LEARNING_RATE = 0.01_dt;
    const size_t BATCH_SIZE = 50;
    const auto EPSILON_ACCURACY = 1e-2_dt;
    const auto acc1 = 12.36_dt;
    const auto acc2 = 89.6_dt;
    const size_t NUM_CLASSES = 10;

    const size_t MNIST_SIZE = 28;
    const size_t MNIST_CHANNELS = 1;
    const size_t CONV1_FILTERS = 16;
    const size_t CONV1_KERNEL_SIZE = 5;
    const size_t CONV1_STRIDE = 2;
    const size_t CONV1_PADDING = 2;
    const size_t CONV1_DILATION = 1;
    const size_t CONV1_GROUPS = 1;

    const size_t FC1_SIZE = CONV1_FILTERS * 392;

    MNIST mnist;
    ASSERT_EQ(mnist.loadingData(tools::getTestAssetsDir() / "MNIST"), true);

    raul::Workflow work(raul::CompressionMode::NONE, raul::CalculationMode::DETERMINISTIC, raul::AllocationMode::STANDARD, raul::ExecutionTarget::CPUFP16);
    work.add<DataLayer>("data", DataParams{ { "data", "labels" }, MNIST_CHANNELS, MNIST_SIZE, MNIST_SIZE, NUM_CLASSES });
    work.add<raul::ReshapeLayer>("reshape1", ViewParams{ "data", "datar", MNIST_CHANNELS, 1, -1 });
    work.add<raul::Convolution1DLayer>("conv1", Convolution1DParams{ { "datar" }, { "conv1" }, CONV1_KERNEL_SIZE, CONV1_FILTERS, CONV1_STRIDE, CONV1_PADDING, CONV1_DILATION, CONV1_GROUPS, true });
    work.add<raul::ReshapeLayer>("reshape2", ViewParams{ "conv1", "conv1r", 1, 1, -1 });
    work.add<raul::LinearLayer>("fc1", LinearParams{ { "conv1r" }, { "fc1" }, NUM_CLASSES, true });
    work.add<raul::SoftMaxActivation>("softmax", BasicParamsWithDim{ { "fc1" }, { "softmax" } });
    work.add<raul::CrossEntropyLoss>("loss", LossParams{ { "softmax", "labels" }, { "loss" }, "batch_mean" });

    work.preparePipelines();
    work.setBatchSize(BATCH_SIZE);
    work.prepareMemoryForTraining();

    auto& memory_manager = work.getMemoryManager<raul::MemoryManagerFP16>();
    DataLoader dataLoader;

    memory_manager["conv1::Weights"] = dataLoader.loadData(tools::getTestAssetsDir() / "test_cnn_layer" / "conv1d" / "0_conv1.weight.txt", CONV1_FILTERS, CONV1_KERNEL_SIZE);
    memory_manager["fc1::Weights"] = dataLoader.loadData(tools::getTestAssetsDir() / "test_cnn_layer" / "conv1d" / "0_fc1.weight.txt", NUM_CLASSES, FC1_SIZE);
    memory_manager["conv1::Biases"] = dataLoader.loadData(tools::getTestAssetsDir() / "test_cnn_layer" / "conv1d" / "0_conv1.bias.txt", CONV1_FILTERS, 1);
    memory_manager["fc1::Biases"] = dataLoader.loadData(tools::getTestAssetsDir() / "test_cnn_layer" / "conv1d" / "0_fc1.bias.txt", NUM_CLASSES, 1);

    const size_t stepsAmountTrain = mnist.getTrainImageAmount() / BATCH_SIZE;

    dtype testAcc = mnist.testNetwork(work);
    CHECK_NEAR(testAcc, acc1, EPSILON_ACCURACY);
    printf("Test accuracy = %f\n", testAcc);
    auto sgd = std::make_shared<optimizers::SGD>(LEARNING_RATE);
    for (size_t q = 0; q < stepsAmountTrain; ++q)
    {
        dtype testLoss = mnist.oneTrainIteration(work, sgd.get(), q);
        if (q % 100 == 0)
        {
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

TEST(TestCNN1DLayer, SimpleDilationUnit)
{
    PROFILE_TEST

    using namespace raul;
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    const size_t filters = 1;
    const size_t kernelSize = 2;
    const size_t stride = 1;
    const size_t padding = 0;
    const size_t dilation = 2;
    const size_t groups = 1;
    const size_t BATCH_SIZE = 1;
    constexpr auto EPSILON = 1e-6_dt;

    const Tensor realOutput = { -0.47918504_dt, 0.13827267_dt, 0.21124730_dt };
    const Tensor realInputNabla = { -0.00374341_dt, -0.00374341_dt, 0.26447839_dt, 0.26822180_dt, 0.26822180_dt, -0.41152257_dt, -0.41152257_dt, -0.77949208_dt, -0.36796951_dt, -0.36796951_dt };
    const Tensor realWeightsGradient = { -2.69488049_dt, -2.07977104_dt, -0.28457478_dt, -0.81794238_dt };

    const Tensor x = { -2.17878938_dt, 0.56843126_dt, -1.08452237_dt, -1.39859545_dt, 0.40334684_dt, 0.83802634_dt, -0.71925759_dt, -0.40334353_dt, -0.59663534_dt, 0.18203649_dt };

    work.add<DataLayer>("data", DataParams{ { "in" }, 2, 1, 5 });
    Convolution1DLayer l("cnn1", { "in", "cnn1", kernelSize, filters, stride, padding, dilation, groups, false }, networkParameters);

    TENSORS_CREATE(BATCH_SIZE)

    memory_manager["in"] = TORANGE(x);
    memory_manager["cnn1::Weights"] = TORANGE((raul::Tensor{ -0.00374341_dt, 0.26822180_dt, -0.41152257_dt, -0.36796951_dt }));

    // Forward checks
    l.forwardCompute(raul::NetworkMode::Train);
    const auto& output = memory_manager["cnn1"];
    EXPECT_EQ(output.size(), realOutput.size());
    for (size_t i = 0; i < output.size(); ++i)
    {
        CHECK_NEAR(output[i], realOutput[i], EPSILON);
    }

    // Backward checks

    memory_manager[Name("cnn1").grad()] = 1_dt;
    ASSERT_NO_THROW(l.backwardCompute());
    const auto& inputNabla = memory_manager[Name("in").grad()];
    EXPECT_EQ(inputNabla.size(), realInputNabla.size());
    for (size_t i = 0; i < inputNabla.size(); ++i)
    {
        CHECK_NEAR(inputNabla[i], realInputNabla[i], EPSILON);
    }

    const auto& weightsGradient = memory_manager[Name("cnn1::Weights").grad()];
    EXPECT_EQ(weightsGradient.size(), realWeightsGradient.size());
    for (size_t i = 0; i < weightsGradient.size(); ++i)
    {
        CHECK_NEAR(weightsGradient[i], realWeightsGradient[i], EPSILON);
    }
}

TEST(TestCNN1DLayer, SimpleDilationGpuUnit)
{
    PROFILE_TEST

    GPU_ONLY_TEST

    using namespace raul;

    WorkflowEager work(CompressionMode::NONE, CalculationMode::DETERMINISTIC, AllocationMode::STANDARD, ExecutionTarget::GPU);
    auto& memory_manager = work.getMemoryManager<MemoryManagerGPU>();

    const size_t filters = 1;
    const size_t kernelSize = 2;
    const size_t stride = 1;
    const size_t padding = 0;
    const size_t dilation = 2;
    const size_t groups = 1;
    const size_t BATCH_SIZE = 1;
    constexpr auto EPSILON = 1e-6_dt;

    const Tensor realOutput = { -0.47918504_dt, 0.13827267_dt, 0.21124730_dt };
    const Tensor realInputNabla = { -0.00374341_dt, -0.00374341_dt, 0.26447839_dt, 0.26822180_dt, 0.26822180_dt, -0.41152257_dt, -0.41152257_dt, -0.77949208_dt, -0.36796951_dt, -0.36796951_dt };
    const Tensor realWeightsGradient = { -2.69488049_dt, -2.07977104_dt, -0.28457478_dt, -0.81794238_dt };

    const Tensor x = { -2.17878938_dt, 0.56843126_dt, -1.08452237_dt, -1.39859545_dt, 0.40334684_dt, 0.83802634_dt, -0.71925759_dt, -0.40334353_dt, -0.59663534_dt, 0.18203649_dt };

    work.add<DataLayer>("data", DataParams{ { "in" }, 2, 1, 5 });
    work.add<Convolution1DLayer>("cnn1", Convolution1DParams{ "in", "cnn1", kernelSize, filters, stride, padding, dilation, groups, false });

    TENSORS_CREATE(BATCH_SIZE)

    memory_manager["in"] = TORANGE(x);
    memory_manager["cnn1::Weights"] = TORANGE((raul::Tensor{ -0.00374341_dt, 0.26822180_dt, -0.41152257_dt, -0.36796951_dt }));

    // Forward checks
    ASSERT_NO_THROW(work.forwardPassTraining());
    const Tensor output = memory_manager["cnn1"];
    EXPECT_EQ(output.size(), realOutput.size());
    for (size_t i = 0; i < output.size(); ++i)
    {
        CHECK_NEAR(output[i], realOutput[i], EPSILON);
    }

    // Backward checks

    memory_manager[Name("cnn1").grad()] = 1_dt;
    ASSERT_NO_THROW(work.backwardPassTraining());
    const Tensor inputNabla = memory_manager[Name("in").grad()];
    EXPECT_EQ(inputNabla.size(), realInputNabla.size());
    for (size_t i = 0; i < inputNabla.size(); ++i)
    {
        CHECK_NEAR(inputNabla[i], realInputNabla[i], EPSILON);
    }

    const Tensor weightsGradient = memory_manager[Name("cnn1::Weights").grad()];
    EXPECT_EQ(weightsGradient.size(), realWeightsGradient.size());
    for (size_t i = 0; i < weightsGradient.size(); ++i)
    {
        CHECK_NEAR(weightsGradient[i], realWeightsGradient[i], EPSILON);
    }
}

TEST(TestCNN1DLayer, DilationUnit)
{
    PROFILE_TEST

    using namespace raul;
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    const size_t filters = 3;
    const size_t kernelSize = 3;
    const size_t stride = 2;
    const size_t padding = 1;
    const size_t dilation = 3;
    const size_t groups = 1;
    const size_t BATCH_SIZE = 2;
    const auto EPSILON = 1e-6_dt;

    const Tensor x = { -0.59103316_dt, -0.56924808_dt, 0.91997141_dt,  1.11081612_dt,  1.28987408_dt,  -1.47817397_dt, 2.56723285_dt,  -0.47311980_dt, 1.32527483_dt,  -1.62932599_dt,
                       -0.54974365_dt, -0.47983426_dt, -0.49968153_dt, -1.06698036_dt, 1.11493957_dt,  -0.14067143_dt, -0.08519430_dt, -0.09334823_dt, 0.68705022_dt,  -0.83831537_dt,
                       0.00089182_dt,  0.84189409_dt,  -0.40003416_dt, 1.03946197_dt,  0.27995500_dt,  0.07324605_dt,  1.11331844_dt,  0.28226724_dt,  0.43422565_dt,  -0.80249292_dt,
                       -1.29518616_dt, -0.75018150_dt, -0.92678940_dt, 0.20641631_dt,  -0.33344787_dt, -0.42883000_dt, 0.23291829_dt,  0.79688716_dt,  -0.18484163_dt, -0.37014726_dt };

    const Tensor realOutput = { 0.13889819_dt, 0.45487607_dt, -0.13117632_dt, -0.32477066_dt, 0.41605633_dt, -0.13640109_dt };
    const Tensor realInputNabla = { 0._dt, 0._dt, 0.26827440_dt,  0._dt, 0._dt, 0._dt, 0._dt, -0.05198006_dt, 0._dt, 0._dt, 0._dt, 0._dt, -0.19009148_dt, 0._dt, 0._dt,
                                    0._dt, 0._dt, -0.21962076_dt, 0._dt, 0._dt, 0._dt, 0._dt, 0.26827440_dt,  0._dt, 0._dt, 0._dt, 0._dt, -0.05198006_dt, 0._dt, 0._dt,
                                    0._dt, 0._dt, -0.19009148_dt, 0._dt, 0._dt, 0._dt, 0._dt, -0.21962076_dt, 0._dt, 0._dt };
    const Tensor realWeightsGradient = { 0._dt, 0.51993728_dt, 0._dt, 0._dt, -0.19085255_dt, 0._dt, 0._dt, -1.42647099_dt, 0._dt, 0._dt, 0.70353889_dt, 0._dt,
                                         0._dt, 0.51993728_dt, 0._dt, 0._dt, -0.19085255_dt, 0._dt, 0._dt, -1.42647099_dt, 0._dt, 0._dt, 0.70353889_dt, 0._dt,
                                         0._dt, 0.51993728_dt, 0._dt, 0._dt, -0.19085255_dt, 0._dt, 0._dt, -1.42647099_dt, 0._dt, 0._dt, 0.70353889_dt, 0._dt };
    const Tensor realBiasGradient = { 2.0_dt, 2.0_dt, 2.0_dt };

    work.add<DataLayer>("data", DataParams{ { "in" }, 1, 4, 5 });
    Convolution1DParams params{ "in", "cnn1", kernelSize, filters, stride, padding, dilation, groups, true };
    Convolution1DLayer cnnLayer("cnn1", params, networkParameters);

    TENSORS_CREATE(BATCH_SIZE);

    memory_manager["in"] = TORANGE(x);
    memory_manager["cnn1::Weights"] =
        TORANGE((raul::Tensor{ -0.04652963_dt, 0.03054590_dt,  0.26138845_dt,  -0.26779535_dt, -0.18173194_dt, -0.07308251_dt, -0.11252555_dt, 0.24941555_dt,  -0.18711334_dt,
                               -0.13288665_dt, -0.20168012_dt, -0.27036187_dt, -0.16851136_dt, 0.24814458_dt,  0.12881215_dt,  0.13991290_dt,  0.01518188_dt,  -0.14799897_dt,
                               0.04883941_dt,  -0.26953444_dt, -0.20858690_dt, -0.14882068_dt, 0.18213609_dt,  0.16925636_dt,  -0.12802599_dt, -0.01041609_dt, 0.18462527_dt,
                               0.28698149_dt,  0.11457001_dt,  0.03899795_dt,  0.19355273_dt,  -0.16997258_dt, 0.05379289_dt,  -0.22381142_dt, -0.20007673_dt, -0.14912483_dt }));
    memory_manager["cnn1::Biases"] = TORANGE((raul::Tensor{ 0.13061771_dt, 0.11609371_dt, -0.17099744_dt }));
    ASSERT_NO_THROW(cnnLayer.forwardCompute(raul::NetworkMode::Train));

    // Forward checks
    const auto& output = memory_manager["cnn1"];
    EXPECT_EQ(output.size(), realOutput.size());
    for (size_t i = 0; i < output.size(); ++i)
    {
        CHECK_NEAR(output[i], realOutput[i], EPSILON);
    }

    // Backward checks
    memory_manager[Name("cnn1").grad()] = 1.0_dt;
    ASSERT_NO_THROW(cnnLayer.backwardCompute());
    const auto& inputNabla = memory_manager[Name("in").grad()];
    EXPECT_EQ(inputNabla.size(), realInputNabla.size());
    for (size_t i = 0; i < inputNabla.size(); ++i)
    {
        CHECK_NEAR(inputNabla[i], realInputNabla[i], EPSILON);
    }

    const auto& weightsGradient = memory_manager[Name("cnn1::Weights").grad()];
    EXPECT_EQ(weightsGradient.size(), realWeightsGradient.size());
    for (size_t i = 0; i < weightsGradient.size(); ++i)
    {
        CHECK_NEAR(weightsGradient[i], realWeightsGradient[i], EPSILON);
    }

    const auto& biasGradient = memory_manager[raul::Name("cnn1::Biases").grad()];
    EXPECT_EQ(biasGradient.size(), realBiasGradient.size());
    for (size_t i = 0; i < biasGradient.size(); ++i)
    {
        CHECK_NEAR(biasGradient[i], realBiasGradient[i], EPSILON);
    }
}

TEST(TestCNN1DLayer, TFStyleSimpleUnit)
{
    PROFILE_TEST

    using namespace raul;
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    const size_t filters = 3;
    const size_t kernelSize = 2;
    const size_t stride = 1;
    const size_t padding = 0;
    const size_t dilation = 1;
    const size_t groups = 1;
    const size_t BATCH_SIZE = 2;
    const auto EPSILON = 1e-5_dt;

    const Tensor x = { 1.0_dt, 2.0_dt, 3.0_dt, 4.0_dt, 5.0_dt, 6.0_dt, 7.0_dt, 8.0_dt, 9.0_dt, 10.0_dt, 11.0_dt, 12.0_dt };
    const Tensor deltas = { 12.0_dt, 11.0_dt, 10.0_dt, 9.0_dt, 8.0_dt, 7.0_dt, 6.0_dt, 5.0_dt, 4.0_dt, 3.0_dt, 2.0_dt, 1.0_dt };

    const Tensor realOutput = { 10.738498_dt, 4.2573757_dt, 4.5316195_dt, 21.469763_dt, 8.778179_dt, 7.236649_dt, 42.932293_dt, 17.819784_dt, 12.646708_dt, 53.663555_dt, 22.34059_dt, 15.351738_dt };
    const Tensor realInputNabla = {
        35.35774_dt, 37.661076_dt, 34.63284_dt, 49.321957_dt, 6.6174064_dt, 15.02765_dt, 16.500051_dt, 18.874037_dt, 11.602078_dt, 18.48142_dt, 2.4443321_dt, 2.9741561_dt
    };
    const Tensor realWeightsGradient = { 108.0_dt, 88.0_dt, 68.0_dt, 138.0_dt, 114.0_dt, 90.0_dt, 168.0_dt, 140.0_dt, 112.0_dt, 198.0_dt, 166.0_dt, 134.0_dt };

    work.add<DataLayer>("data", DataParams{ { "in" }, 3, 1, 2 });
    Convolution1DParams params{ "in", "cnn1", kernelSize, filters, stride, padding, dilation, groups, false, false, true };
    Convolution1DLayer cnnLayer("cnn1", params, networkParameters);

    TENSORS_CREATE(BATCH_SIZE);
    memory_manager["in"] = TORANGE(x);
    memory_manager["cnn1::Weights"] = TORANGE((Tensor{ 1.764052345967664_dt,
                                                       0.4001572083672233_dt,
                                                       0.9787379841057392_dt,
                                                       2.240893199201458_dt,
                                                       1.8675579901499675_dt,
                                                       -0.977277879876411_dt,
                                                       0.9500884175255894_dt,
                                                       -0.1513572082976979_dt,
                                                       -0.10321885179355784_dt,
                                                       0.41059850193837233_dt,
                                                       0.144043571160878_dt,
                                                       1.454273506962975_dt }));
    ASSERT_NO_THROW(cnnLayer.forwardCompute(raul::NetworkMode::Train));

    // Forward checks
    const auto& output = memory_manager["cnn1"];
    EXPECT_EQ(output.size(), realOutput.size());
    for (size_t i = 0; i < output.size(); ++i)
    {
        CHECK_NEAR(output[i], realOutput[i], EPSILON);
    }

    memory_manager[Name("cnn1").grad()] = TORANGE(deltas);
    ASSERT_NO_THROW(cnnLayer.backwardCompute());
    const auto& inputNabla = memory_manager[raul::Name("in").grad()];
    EXPECT_EQ(inputNabla.size(), realInputNabla.size());
    for (size_t i = 0; i < inputNabla.size(); ++i)
    {
        CHECK_NEAR(inputNabla[i], realInputNabla[i], EPSILON);
    }

    const auto& weightsGradient = memory_manager[raul::Name("cnn1::Weights").grad()];
    EXPECT_EQ(weightsGradient.size(), realWeightsGradient.size());
    for (size_t i = 0; i < weightsGradient.size(); ++i)
    {
        CHECK_NEAR(weightsGradient[i], realWeightsGradient[i], EPSILON);
    }
}

TEST(TestCNN1DLayer, TFStyleSimpleGpuUnit)
{
    PROFILE_TEST

    GPU_ONLY_TEST

    using namespace raul;
    WorkflowEager work(CompressionMode::NONE, CalculationMode::DETERMINISTIC, AllocationMode::STANDARD, ExecutionTarget::GPU);
    auto& memory_manager = work.getMemoryManager<MemoryManagerGPU>();

    const size_t filters = 3;
    const size_t kernelSize = 2;
    const size_t stride = 1;
    const size_t padding = 0;
    const size_t dilation = 1;
    const size_t groups = 1;
    const size_t BATCH_SIZE = 2;
    const auto EPSILON = 1e-5_dt;

    const Tensor x = { 1.0_dt, 2.0_dt, 3.0_dt, 4.0_dt, 5.0_dt, 6.0_dt, 7.0_dt, 8.0_dt, 9.0_dt, 10.0_dt, 11.0_dt, 12.0_dt };
    const Tensor deltas = { 12.0_dt, 11.0_dt, 10.0_dt, 9.0_dt, 8.0_dt, 7.0_dt, 6.0_dt, 5.0_dt, 4.0_dt, 3.0_dt, 2.0_dt, 1.0_dt };

    const Tensor realOutput = { 10.738498_dt, 4.2573757_dt, 4.5316195_dt, 21.469763_dt, 8.778179_dt, 7.236649_dt, 42.932293_dt, 17.819784_dt, 12.646708_dt, 53.663555_dt, 22.34059_dt, 15.351738_dt };
    const Tensor realInputNabla = {
        35.35774_dt, 37.661076_dt, 34.63284_dt, 49.321957_dt, 6.6174064_dt, 15.02765_dt, 16.500051_dt, 18.874037_dt, 11.602078_dt, 18.48142_dt, 2.4443321_dt, 2.9741561_dt
    };
    const Tensor realWeightsGradient = { 108.0_dt, 88.0_dt, 68.0_dt, 138.0_dt, 114.0_dt, 90.0_dt, 168.0_dt, 140.0_dt, 112.0_dt, 198.0_dt, 166.0_dt, 134.0_dt };

    work.add<DataLayer>("data", DataParams{ { "in" }, 3, 1, 2 });
    Convolution1DParams params{ "in", "cnn1", kernelSize, filters, stride, padding, dilation, groups, false, false, true };
    work.add<Convolution1DLayer>("cnn1", params);

    TENSORS_CREATE(BATCH_SIZE);
    memory_manager["in"] = TORANGE(x);
    memory_manager["cnn1::Weights"] = TORANGE((Tensor{ 1.764052345967664_dt,
                                                       0.4001572083672233_dt,
                                                       0.9787379841057392_dt,
                                                       2.240893199201458_dt,
                                                       1.8675579901499675_dt,
                                                       -0.977277879876411_dt,
                                                       0.9500884175255894_dt,
                                                       -0.1513572082976979_dt,
                                                       -0.10321885179355784_dt,
                                                       0.41059850193837233_dt,
                                                       0.144043571160878_dt,
                                                       1.454273506962975_dt }));
    (work.forwardPassTraining());

    // Forward checks
    const Tensor output = memory_manager["cnn1"];
    EXPECT_EQ(output.size(), realOutput.size());
    for (size_t i = 0; i < output.size(); ++i)
    {
        CHECK_NEAR(output[i], realOutput[i], EPSILON);
    }

    memory_manager[Name("cnn1").grad()] = TORANGE(deltas);
    ASSERT_NO_THROW(work.backwardPassTraining());
    const Tensor inputNabla = memory_manager[raul::Name("in").grad()];
    EXPECT_EQ(inputNabla.size(), realInputNabla.size());
    for (size_t i = 0; i < inputNabla.size(); ++i)
    {
        CHECK_NEAR(inputNabla[i], realInputNabla[i], EPSILON);
    }

    const Tensor weightsGradient = memory_manager[raul::Name("cnn1::Weights").grad()];
    EXPECT_EQ(weightsGradient.size(), realWeightsGradient.size());
    for (size_t i = 0; i < weightsGradient.size(); ++i)
    {
        CHECK_NEAR(weightsGradient[i], realWeightsGradient[i], EPSILON);
    }
}

TEST(TestCNN1DLayer, TFStyleUnit)
{
    PROFILE_TEST

    using namespace raul;
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    const size_t filters = 2;
    const size_t kernelSize = 3;
    const size_t stride = 1;
    const size_t padding = 2;
    const size_t dilation = 2;
    const size_t groups = 1;
    const size_t BATCH_SIZE = 2;
    const auto EPSILON = 1e-4_dt;

    const Tensor x = { 1.0_dt,  2.0_dt,  3.0_dt,  4.0_dt,  5.0_dt,  6.0_dt,  7.0_dt,  8.0_dt,  9.0_dt,  10.0_dt, 11.0_dt, 12.0_dt, 13.0_dt, 14.0_dt, 15.0_dt,
                       16.0_dt, 17.0_dt, 18.0_dt, 19.0_dt, 20.0_dt, 21.0_dt, 22.0_dt, 23.0_dt, 24.0_dt, 25.0_dt, 26.0_dt, 27.0_dt, 28.0_dt, 29.0_dt, 30.0_dt };
    const Tensor deltas = { 20.0_dt, 19.0_dt, 18.0_dt, 17.0_dt, 16.0_dt, 15.0_dt, 14.0_dt, 13.0_dt, 12.0_dt, 11.0_dt, 10._dt, 9.0_dt, 8.0_dt, 7.0_dt, 6.0_dt, 5.0_dt, 4.0_dt, 3.0_dt, 2.0_dt, 1.0_dt };

    const Tensor realOutput = { 23.500664_dt, 6.707356_dt,  34.570343_dt, 12.598474_dt, 54.964226_dt, 20.439703_dt, 33.24925_dt,   27.39572_dt, 50.05303_dt,  37.52758_dt,
                                78.84906_dt,  36.162945_dt, 89.91874_dt,  42.05406_dt,  179.46786_dt, 74.851875_dt, 117.268166_dt, 78.05503_dt, 134.07195_dt, 88.18689_dt };
    const Tensor realInputNabla = { 50.353176_dt, 55.0102_dt,   45.733826_dt,  44.427296_dt, 47.956177_dt, 40.756634_dt, 56.033997_dt, 56.119232_dt, 61.763016_dt, 27.10075_dt,
                                    17.55472_dt,  44.3279_dt,   23.73786_dt,   15.384884_dt, 38.55342_dt,  20.72377_dt,  19.740091_dt, 20.847855_dt, 14.797889_dt, 12.68607_dt,
                                    15.870661_dt, 17.577457_dt, 13.0737505_dt, 23.987835_dt, 10.286309_dt, 6.7055464_dt, 15.455521_dt, 6.923421_dt,  4.535712_dt,  9.681044_dt };
    const Tensor realWeightsGradient = { 372.0_dt,  303.0_dt,  426.0_dt,  351.0_dt,  480.0_dt,  399.0_dt,  1100.0_dt, 955.0_dt,  1210.0_dt,
                                         1055.0_dt, 1320.0_dt, 1155.0_dt, 1116.0_dt, 1011.0_dt, 1194.0_dt, 1083.0_dt, 1272.0_dt, 1155.0_dt };

    work.add<DataLayer>("data", DataParams{ { "in" }, 1, 5, 3 });
    Convolution1DParams params{ "in", "cnn1", kernelSize, filters, stride, padding, dilation, groups, false, false, true };
    Convolution1DLayer cnnLayer("cnn1", params, networkParameters);

    TENSORS_CREATE(BATCH_SIZE);
    memory_manager["in"] = TORANGE(x);
    memory_manager["cnn1::Weights"] = TORANGE((raul::Tensor{ 1.7640524_dt,
                                                             0.4001572_dt,
                                                             0.978738_dt,
                                                             2.2408931_dt,
                                                             1.867558_dt,
                                                             -0.9772779_dt,
                                                             0.95008844_dt,
                                                             -0.1513572_dt,
                                                             -0.10321885_dt,
                                                             0.41059852_dt,
                                                             0.14404356_dt,
                                                             1.4542735_dt,
                                                             0.7610377_dt,
                                                             0.121675014_dt,
                                                             0.44386324_dt,
                                                             0.33367434_dt,
                                                             1.4940791_dt,
                                                             -0.20515826_dt }));
    ASSERT_NO_THROW(cnnLayer.forwardCompute(raul::NetworkMode::Train));

    // Forward checks
    const auto& output = memory_manager["cnn1"];
    EXPECT_EQ(output.size(), realOutput.size());
    for (size_t i = 0; i < output.size(); ++i)
    {
        CHECK_NEAR(output[i], realOutput[i], EPSILON);
    }

    memory_manager[Name("cnn1").grad()] = TORANGE(deltas);
    cnnLayer.backwardCompute();
    const auto& inputNabla = memory_manager[Name("in").grad()];
    EXPECT_EQ(inputNabla.size(), realInputNabla.size());
    for (size_t i = 0; i < inputNabla.size(); ++i)
    {
        CHECK_NEAR(inputNabla[i], realInputNabla[i], EPSILON);
    }

    const auto& weightsGradient = memory_manager[Name("cnn1::Weights").grad()];
    EXPECT_EQ(weightsGradient.size(), realWeightsGradient.size());
    for (size_t i = 0; i < weightsGradient.size(); ++i)
    {
        CHECK_NEAR(weightsGradient[i], realWeightsGradient[i], EPSILON);
    }
}

// conv1d_double_step.py
TEST(TestCNN1DLayer, DoubleStep1Unit)
{
    PROFILE_TEST

    using namespace raul;
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    const size_t filters = 3;
    const size_t kernelSize = 2;
    const size_t stride = 2;
    const size_t padding = 1;
    const size_t dilation = 2;
    const size_t groups = 1;
    const size_t BATCH_SIZE = 2;
    const auto EPSILON = 1e-6_dt;

    const Tensor x = { -1.55509508_dt, -0.34136039_dt, 1.85300612_dt, 0.46809641_dt,  -0.15771244_dt, 1.44366014_dt,  0.26604941_dt,  0.16645534_dt,  1.58633423_dt,  0.94629836_dt,
                       -0.84367675_dt, 0.93182659_dt,  1.25900924_dt, 2.00498056_dt,  0.17784372_dt,  -0.23033547_dt, -0.39175439_dt, 0.54329473_dt,  -0.39515755_dt, 0.20552567_dt,
                       -0.45032975_dt, -0.57307708_dt, 3.41050267_dt, -1.53118432_dt, -1.23413503_dt, 1.81972528_dt,  -0.55152869_dt, -1.32532597_dt, 0.18855357_dt,  -0.06907269_dt };

    const Tensor realOutput[] = {
        { -0.32944888_dt,
          -0.66623712_dt,
          -1.12647843_dt,
          -0.33430994_dt,
          -0.13762742_dt,
          -0.48640594_dt,
          0.47286037_dt,
          0.77560359_dt,
          0.02234657_dt,
          -0.25076321_dt,
          0.39784792_dt,
          0.20927989_dt,
          -0.32223558_dt,
          -0.39425132_dt,
          -0.14236981_dt,
          0.11043543_dt,
          0.43837607_dt,
          0.59051555_dt },
        { -0.29641706_dt, -0.35044304_dt, -0.47054291_dt, -0.26309419_dt, 0.51632619_dt, 0.55646789_dt, -0.02319643_dt, -0.21444283_dt, -0.12676388_dt, -0.22079742_dt, 0.14213601_dt, 0.13021433_dt }
    };
    const Tensor deltas = { 1.71789527_dt,  -0.85072410_dt, -0.84863919_dt, -0.18478005_dt, -1.19375718_dt, -0.22327407_dt,
                            -1.27057660_dt, 0.01933064_dt,  0.88683778_dt,  0.05517140_dt,  0.68803781_dt,  1.23262453_dt };
    const Tensor realWeightsGradient = { 0.65647715_dt,  -1.14853525_dt, 0.62395668_dt,  2.09043622_dt, 0.19213718_dt, 1.68474627_dt,  0.13944964_dt,  0.67229474_dt, -0.23088974_dt,
                                         -1.11849952_dt, -0.54473877_dt, -0.81111443_dt, 0.55264896_dt, 1.05027425_dt, -0.51726854_dt, -0.19892836_dt, 0.36443216_dt, -0.42750430_dt };
    const Tensor realBiasesGradient = { -0.52708888_dt, -0.11514002_dt, 0.73527777_dt };

    work.add<DataLayer>("data", DataParams{ { "in" }, 1, 3, 5 });
    Convolution1DParams params1{ "in", "cnn1", kernelSize, filters, stride, padding, dilation, groups, true, false };
    Convolution1DLayer cnnLayer1("cnn1", params1, networkParameters);
    Convolution1DParams params2{ { { "cnn1" }, { "out" }, { "cnn1::Weights", "cnn1::Biases" } }, kernelSize, filters, stride, padding, dilation, groups, true, false };
    Convolution1DLayer cnnLayer2("cnn2", params2, networkParameters);

    TENSORS_CREATE(BATCH_SIZE);
    memory_manager["in"] = TORANGE(x);
    memory_manager["cnn1::Weights"] = TORANGE((Tensor{ -0.00305648_dt,
                                                       0.21900219_dt,
                                                       -0.33600679_dt,
                                                       -0.30044585_dt,
                                                       -0.15723862_dt,
                                                       0.10947479_dt,
                                                       -0.00808870_dt,
                                                       0.32369578_dt,
                                                       -0.03622961_dt,
                                                       0.10802763_dt,
                                                       -0.12337797_dt,
                                                       -0.08024748_dt,
                                                       -0.39001942_dt,
                                                       -0.27037555_dt,
                                                       -0.16828938_dt,
                                                       0.01512298_dt,
                                                       0.16139492_dt,
                                                       0.24495828_dt }));
    memory_manager["cnn1::Biases"] = TORANGE((Tensor{ -0.27676830_dt, -0.17777696_dt, 0.14828277_dt }));

    // see conv1d_double_step.py
    // Forward
    // Step 1
    ASSERT_NO_THROW(cnnLayer1.forwardCompute(NetworkMode::Train));
    const auto interOutput = memory_manager["cnn1"];
    EXPECT_EQ(interOutput.size(), realOutput[0].size());
    for (size_t i = 0; i < interOutput.size(); ++i)
    {
        CHECK_NEAR(interOutput[i], realOutput[0][i], EPSILON);
    }
    // Step 2
    ASSERT_NO_THROW(cnnLayer2.forwardCompute(NetworkMode::Train));
    const auto output = memory_manager["out"];
    EXPECT_EQ(output.size(), realOutput[1].size());
    for (size_t i = 0; i < output.size(); ++i)
    {
        CHECK_NEAR(output[i], realOutput[1][i], EPSILON);
    }

    // Backward
    memory_manager[Name("out").grad()] = TORANGE(deltas);
    ASSERT_NO_THROW(cnnLayer2.backwardCompute());
    ASSERT_NO_THROW(cnnLayer1.backwardCompute());

    const auto weightsGradient = memory_manager[Name("cnn1::Weights").grad()];
    EXPECT_EQ(weightsGradient.size(), realWeightsGradient.size());
    for (size_t i = 0; i < weightsGradient.size(); ++i)
    {
        CHECK_NEAR(weightsGradient[i], realWeightsGradient[i], EPSILON);
    }
    const auto biasesGradient = memory_manager[Name("cnn1::Biases").grad()];
    EXPECT_EQ(biasesGradient.size(), realBiasesGradient.size());
    for (size_t i = 0; i < biasesGradient.size(); ++i)
    {
        CHECK_NEAR(biasesGradient[i], realBiasesGradient[i], EPSILON);
    }
}

TEST(TestCNN1DLayer, DoubleStep1GpuUnit)
{
    PROFILE_TEST

    GPU_ONLY_TEST

    using namespace raul;
    WorkflowEager work(CompressionMode::NONE, CalculationMode::DETERMINISTIC, AllocationMode::STANDARD, ExecutionTarget::GPU);
    auto& memory_manager = work.getMemoryManager<MemoryManagerGPU>();

    const size_t filters = 3;
    const size_t kernelSize = 2;
    const size_t stride = 2;
    const size_t padding = 1;
    const size_t dilation = 2;
    const size_t groups = 1;
    const size_t BATCH_SIZE = 2;
    const auto EPSILON = 1e-6_dt;

    const Tensor x = { -1.55509508_dt, -0.34136039_dt, 1.85300612_dt, 0.46809641_dt,  -0.15771244_dt, 1.44366014_dt,  0.26604941_dt,  0.16645534_dt,  1.58633423_dt,  0.94629836_dt,
                       -0.84367675_dt, 0.93182659_dt,  1.25900924_dt, 2.00498056_dt,  0.17784372_dt,  -0.23033547_dt, -0.39175439_dt, 0.54329473_dt,  -0.39515755_dt, 0.20552567_dt,
                       -0.45032975_dt, -0.57307708_dt, 3.41050267_dt, -1.53118432_dt, -1.23413503_dt, 1.81972528_dt,  -0.55152869_dt, -1.32532597_dt, 0.18855357_dt,  -0.06907269_dt };

    const Tensor realOutput[] = {
        { -0.32944888_dt,
          -0.66623712_dt,
          -1.12647843_dt,
          -0.33430994_dt,
          -0.13762742_dt,
          -0.48640594_dt,
          0.47286037_dt,
          0.77560359_dt,
          0.02234657_dt,
          -0.25076321_dt,
          0.39784792_dt,
          0.20927989_dt,
          -0.32223558_dt,
          -0.39425132_dt,
          -0.14236981_dt,
          0.11043543_dt,
          0.43837607_dt,
          0.59051555_dt },
        { -0.29641706_dt, -0.35044304_dt, -0.47054291_dt, -0.26309419_dt, 0.51632619_dt, 0.55646789_dt, -0.02319643_dt, -0.21444283_dt, -0.12676388_dt, -0.22079742_dt, 0.14213601_dt, 0.13021433_dt }
    };
    const Tensor deltas = { 1.71789527_dt,  -0.85072410_dt, -0.84863919_dt, -0.18478005_dt, -1.19375718_dt, -0.22327407_dt,
                            -1.27057660_dt, 0.01933064_dt,  0.88683778_dt,  0.05517140_dt,  0.68803781_dt,  1.23262453_dt };
    const Tensor realWeightsGradient = { 0.65647715_dt,  -1.14853525_dt, 0.62395668_dt,  2.09043622_dt, 0.19213718_dt, 1.68474627_dt,  0.13944964_dt,  0.67229474_dt, -0.23088974_dt,
                                         -1.11849952_dt, -0.54473877_dt, -0.81111443_dt, 0.55264896_dt, 1.05027425_dt, -0.51726854_dt, -0.19892836_dt, 0.36443216_dt, -0.42750430_dt };
    const Tensor realBiasesGradient = { -0.52708888_dt, -0.11514002_dt, 0.73527777_dt };

    work.add<DataLayer>("data", DataParams{ { "in" }, 1, 3, 5 });
    Convolution1DParams params1{ "in", "cnn1", kernelSize, filters, stride, padding, dilation, groups, true, false };
    work.add<Convolution1DLayer>("cnn1", params1);
    Convolution1DParams params2{ { { "cnn1" }, { "out" }, { "cnn1::Weights", "cnn1::Biases" } }, kernelSize, filters, stride, padding, dilation, groups, true, false };
    work.add<Convolution1DLayer>("cnn2", params2);

    TENSORS_CREATE(BATCH_SIZE);
    memory_manager["in"] = TORANGE(x);
    memory_manager["cnn1::Weights"] = TORANGE((Tensor{ -0.00305648_dt,
                                                       0.21900219_dt,
                                                       -0.33600679_dt,
                                                       -0.30044585_dt,
                                                       -0.15723862_dt,
                                                       0.10947479_dt,
                                                       -0.00808870_dt,
                                                       0.32369578_dt,
                                                       -0.03622961_dt,
                                                       0.10802763_dt,
                                                       -0.12337797_dt,
                                                       -0.08024748_dt,
                                                       -0.39001942_dt,
                                                       -0.27037555_dt,
                                                       -0.16828938_dt,
                                                       0.01512298_dt,
                                                       0.16139492_dt,
                                                       0.24495828_dt }));
    memory_manager["cnn1::Biases"] = TORANGE((Tensor{ -0.27676830_dt, -0.17777696_dt, 0.14828277_dt }));

    // see conv1d_double_step.py
    // Forward
    // Step 1
    ASSERT_NO_THROW(work.forwardPassTraining());
    const Tensor interOutput = memory_manager["cnn1"];
    EXPECT_EQ(interOutput.size(), realOutput[0].size());
    for (size_t i = 0; i < interOutput.size(); ++i)
    {
        CHECK_NEAR(interOutput[i], realOutput[0][i], EPSILON);
    }

    const Tensor output = memory_manager["out"];
    EXPECT_EQ(output.size(), realOutput[1].size());
    for (size_t i = 0; i < output.size(); ++i)
    {
        CHECK_NEAR(output[i], realOutput[1][i], EPSILON);
    }

    // Backward
    memory_manager[Name("out").grad()] = TORANGE(deltas);
    ASSERT_NO_THROW(work.backwardPassTraining());

    const Tensor weightsGradient = memory_manager[Name("cnn1::Weights").grad()];
    EXPECT_EQ(weightsGradient.size(), realWeightsGradient.size());
    for (size_t i = 0; i < weightsGradient.size(); ++i)
    {
        CHECK_NEAR(weightsGradient[i], realWeightsGradient[i], EPSILON);
    }
    const Tensor biasesGradient = memory_manager[Name("cnn1::Biases").grad()];
    EXPECT_EQ(biasesGradient.size(), realBiasesGradient.size());
    for (size_t i = 0; i < biasesGradient.size(); ++i)
    {
        CHECK_NEAR(biasesGradient[i], realBiasesGradient[i], EPSILON);
    }
}

// conv1d_double_step.py
TEST(TestCNN1DLayer, DoubleStep2Unit)
{
    PROFILE_TEST

    using namespace raul;
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    const size_t filters = 3;
    const size_t kernelSize = 2;
    const size_t stride = 2;
    const size_t padding = 1;
    const size_t dilation = 2;
    const size_t groups = 1;
    const size_t BATCH_SIZE = 2;
    const auto EPSILON = 1e-6_dt;

    const Tensor x = { -1.55509508_dt, -0.34136039_dt, 1.85300612_dt, 0.46809641_dt,  -0.15771244_dt, 1.44366014_dt,  0.26604941_dt,  0.16645534_dt,  1.58633423_dt,  0.94629836_dt,
                       -0.84367675_dt, 0.93182659_dt,  1.25900924_dt, 2.00498056_dt,  0.17784372_dt,  -0.23033547_dt, -0.39175439_dt, 0.54329473_dt,  -0.39515755_dt, 0.20552567_dt,
                       -0.45032975_dt, -0.57307708_dt, 3.41050267_dt, -1.53118432_dt, -1.23413503_dt, 1.81972528_dt,  -0.55152869_dt, -1.32532597_dt, 0.18855357_dt,  -0.06907269_dt };

    const Tensor realOutput[] = {
        { -0.32944888_dt,
          -0.66623712_dt,
          -1.12647843_dt,
          -0.33430994_dt,
          -0.13762742_dt,
          -0.48640594_dt,
          0.47286037_dt,
          0.77560359_dt,
          0.02234657_dt,
          -0.25076321_dt,
          0.39784792_dt,
          0.20927989_dt,
          -0.32223558_dt,
          -0.39425132_dt,
          -0.14236981_dt,
          0.11043543_dt,
          0.43837607_dt,
          0.59051555_dt },
        { -0.29641706_dt, -0.35044304_dt, -0.47054291_dt, -0.26309419_dt, 0.51632619_dt, 0.55646789_dt, -0.02319643_dt, -0.21444283_dt, -0.12676388_dt, -0.22079742_dt, 0.14213601_dt, 0.13021433_dt }
    };
    const Tensor deltas = { 1.71789527_dt,  -0.85072410_dt, -0.84863919_dt, -0.18478005_dt, -1.19375718_dt, -0.22327407_dt,
                            -1.27057660_dt, 0.01933064_dt,  0.88683778_dt,  0.05517140_dt,  0.68803781_dt,  1.23262453_dt };
    const Tensor realWeightsGradient = { 0.65647715_dt,  -1.14853525_dt, 0.62395668_dt,  2.09043622_dt, 0.19213718_dt, 1.68474627_dt,  0.13944964_dt,  0.67229474_dt, -0.23088974_dt,
                                         -1.11849952_dt, -0.54473877_dt, -0.81111443_dt, 0.55264896_dt, 1.05027425_dt, -0.51726854_dt, -0.19892836_dt, 0.36443216_dt, -0.42750430_dt };
    const Tensor realBiasesGradient = { -0.52708888_dt, -0.11514002_dt, 0.73527777_dt };

    work.add<DataLayer>("data", DataParams{ { "in" }, 1, 3, 5 });
    Convolution1DParams params1{ "in", "cnn1", kernelSize, filters, stride, padding, dilation, groups, true, false };
    Convolution1DLayer cnnLayer1("cnn1", params1, networkParameters);
    Convolution1DParams params2{ { { "cnn1" }, { "out" }, Name{ "cnn1" } }, kernelSize, filters, stride, padding, dilation, groups, true, false };
    Convolution1DLayer cnnLayer2("cnn2", params2, networkParameters);

    TENSORS_CREATE(BATCH_SIZE);
    memory_manager["in"] = TORANGE(x);
    memory_manager["cnn1::Weights"] = TORANGE((Tensor{ -0.00305648_dt,
                                                       0.21900219_dt,
                                                       -0.33600679_dt,
                                                       -0.30044585_dt,
                                                       -0.15723862_dt,
                                                       0.10947479_dt,
                                                       -0.00808870_dt,
                                                       0.32369578_dt,
                                                       -0.03622961_dt,
                                                       0.10802763_dt,
                                                       -0.12337797_dt,
                                                       -0.08024748_dt,
                                                       -0.39001942_dt,
                                                       -0.27037555_dt,
                                                       -0.16828938_dt,
                                                       0.01512298_dt,
                                                       0.16139492_dt,
                                                       0.24495828_dt }));
    memory_manager["cnn1::Biases"] = TORANGE((Tensor{ -0.27676830_dt, -0.17777696_dt, 0.14828277_dt }));

    // see conv1d_double_step.py
    // Forward
    // Step 1
    ASSERT_NO_THROW(cnnLayer1.forwardCompute(NetworkMode::Train));
    const auto interOutput = memory_manager["cnn1"];
    EXPECT_EQ(interOutput.size(), realOutput[0].size());
    for (size_t i = 0; i < interOutput.size(); ++i)
    {
        CHECK_NEAR(interOutput[i], realOutput[0][i], EPSILON);
    }
    // Step 2
    ASSERT_NO_THROW(cnnLayer2.forwardCompute(NetworkMode::Train));
    const auto output = memory_manager["out"];
    EXPECT_EQ(output.size(), realOutput[1].size());
    for (size_t i = 0; i < output.size(); ++i)
    {
        CHECK_NEAR(output[i], realOutput[1][i], EPSILON);
    }

    // Backward
    memory_manager[Name("out").grad()] = TORANGE(deltas);
    ASSERT_NO_THROW(cnnLayer2.backwardCompute());
    ASSERT_NO_THROW(cnnLayer1.backwardCompute());

    const auto weightsGradient = memory_manager[Name("cnn1::Weights").grad()];
    EXPECT_EQ(weightsGradient.size(), realWeightsGradient.size());
    for (size_t i = 0; i < weightsGradient.size(); ++i)
    {
        CHECK_NEAR(weightsGradient[i], realWeightsGradient[i], EPSILON);
    }
    const auto biasesGradient = memory_manager[Name("cnn1::Biases").grad()];
    EXPECT_EQ(biasesGradient.size(), realBiasesGradient.size());
    for (size_t i = 0; i < biasesGradient.size(); ++i)
    {
        CHECK_NEAR(biasesGradient[i], realBiasesGradient[i], EPSILON);
    }
}

TEST(TestCNN1DLayer, DepthwiseDilationUnit)
{
    PROFILE_TEST

    using namespace raul;
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    const size_t filters = 4;
    const size_t kernelSize = 3;
    const size_t stride = 2;
    const size_t padding = 1;
    const size_t dilation = 2;
    const size_t groups = 4;
    const size_t BATCH_SIZE = 2;
    const auto EPSILON = 1e-6_dt;

    const Tensor x = { -0.5911577344_dt, 0.2737623453_dt,  -0.9648520350_dt, -0.2358119786_dt, -0.6969724298_dt, -1.1607614756_dt, 0.6995424032_dt, 0.1990816295_dt,
                       0.1990565062_dt,  0.0457027778_dt,  0.1529569179_dt,  -0.4756788015_dt, -1.8821425438_dt, -0.7765450478_dt, 2.0242021084_dt, -0.0865411982_dt,
                       2.3571109772_dt,  -1.0373387337_dt, 1.5747981071_dt,  -0.6298472285_dt, 2.4069781303_dt,  0.2785662413_dt,  0.2467529178_dt, 1.1843266487_dt };

    const Tensor realOutput = { -0.0697628111_dt, 0.3788315654_dt, 0.0584749356_dt, 0.0244936608_dt, 0.1978868395_dt, -1.2811813354_dt, -0.1850008667_dt, 0.0395136252_dt };
    const Tensor realInputNabla = { 0.0000000000_dt,  -0.2548298240_dt, 0.0000000000_dt, 0.0000000000_dt, -0.5435388088_dt, 0.0000000000_dt,  0.0000000000_dt, 0.2937234044_dt,
                                    0.0000000000_dt,  0.0000000000_dt,  0.1601343751_dt, 0.0000000000_dt, 0.0000000000_dt,  -0.2548298240_dt, 0.0000000000_dt, 0.0000000000_dt,
                                    -0.5435388088_dt, 0.0000000000_dt,  0.0000000000_dt, 0.2937234044_dt, 0.0000000000_dt,  0.0000000000_dt,  0.1601343751_dt, 0.0000000000_dt };
    const Tensor realWeightsGradient = { 0.0000000000_dt, -0.5027827024_dt, 0.0000000000_dt, 0.0000000000_dt, 1.6601386070_dt, 0.0000000000_dt,
                                         0.0000000000_dt, -0.4307655990_dt, 0.0000000000_dt, 0.0000000000_dt, 0.3997098207_dt, 0.0000000000_dt };

    work.add<DataLayer>("data", DataParams{ { "in" }, 1, 4, 3 });
    Convolution1DParams params{ "in", "cnn1", kernelSize, filters, stride, padding, dilation, groups, false };
    Convolution1DLayer cnnLayer("cnn1", params, networkParameters);

    TENSORS_CREATE(BATCH_SIZE);
    memory_manager["in"] = TORANGE(x);
    memory_manager["cnn1::Weights"] = TORANGE((raul::Tensor{ 0.2974873185_dt,
                                                             -0.2548298240_dt,
                                                             -0.1119259894_dt,
                                                             0.2709902525_dt,
                                                             -0.5435388088_dt,
                                                             0.3462468982_dt,
                                                             -0.1187755764_dt,
                                                             0.2937234044_dt,
                                                             0.0802614689_dt,
                                                             -0.0706931353_dt,
                                                             0.1601343751_dt,
                                                             0.0284817219_dt }));
    ASSERT_NO_THROW(cnnLayer.forwardCompute(raul::NetworkMode::Train));

    // Forward checks
    const auto& output = memory_manager["cnn1"];
    EXPECT_EQ(output.size(), realOutput.size());
    for (size_t i = 0; i < output.size(); ++i)
    {
        CHECK_NEAR(output[i], realOutput[i], EPSILON);
    }

    // Backward checks
    memory_manager[Name("cnn1").grad()] = 1_dt;
    ASSERT_NO_THROW(cnnLayer.backwardCompute());
    const auto& inputNabla = memory_manager[Name("in").grad()];
    EXPECT_EQ(inputNabla.size(), realInputNabla.size());
    for (size_t i = 0; i < inputNabla.size(); ++i)
    {
        CHECK_NEAR(inputNabla[i], realInputNabla[i], EPSILON);
    }

    const auto& weightsGradient = memory_manager[Name("cnn1::Weights").grad()];
    EXPECT_EQ(weightsGradient.size(), realWeightsGradient.size());
    for (size_t i = 0; i < weightsGradient.size(); ++i)
    {
        CHECK_NEAR(weightsGradient[i], realWeightsGradient[i], EPSILON);
    }
}

TEST(TestCNN1DLayer, DepthwiseDilationGpuUnit)
{
    PROFILE_TEST

    GPU_ONLY_TEST

    using namespace raul;
    WorkflowEager work(CompressionMode::NONE, CalculationMode::DETERMINISTIC, AllocationMode::STANDARD, ExecutionTarget::GPU);
    auto& memory_manager = work.getMemoryManager<MemoryManagerGPU>();
    NETWORK_PARAMS_DEFINE(networkParameters);

    const size_t filters = 4;
    const size_t kernelSize = 3;
    const size_t stride = 2;
    const size_t padding = 1;
    const size_t dilation = 2;
    const size_t groups = 4;
    const size_t BATCH_SIZE = 2;
    const auto EPSILON = 1e-6_dt;

    const Tensor x = { -0.5911577344_dt, 0.2737623453_dt,  -0.9648520350_dt, -0.2358119786_dt, -0.6969724298_dt, -1.1607614756_dt, 0.6995424032_dt, 0.1990816295_dt,
                       0.1990565062_dt,  0.0457027778_dt,  0.1529569179_dt,  -0.4756788015_dt, -1.8821425438_dt, -0.7765450478_dt, 2.0242021084_dt, -0.0865411982_dt,
                       2.3571109772_dt,  -1.0373387337_dt, 1.5747981071_dt,  -0.6298472285_dt, 2.4069781303_dt,  0.2785662413_dt,  0.2467529178_dt, 1.1843266487_dt };

    const Tensor realOutput = { -0.0697628111_dt, 0.3788315654_dt, 0.0584749356_dt, 0.0244936608_dt, 0.1978868395_dt, -1.2811813354_dt, -0.1850008667_dt, 0.0395136252_dt };
    const Tensor realInputNabla = { 0.0000000000_dt,  -0.2548298240_dt, 0.0000000000_dt, 0.0000000000_dt, -0.5435388088_dt, 0.0000000000_dt,  0.0000000000_dt, 0.2937234044_dt,
                                    0.0000000000_dt,  0.0000000000_dt,  0.1601343751_dt, 0.0000000000_dt, 0.0000000000_dt,  -0.2548298240_dt, 0.0000000000_dt, 0.0000000000_dt,
                                    -0.5435388088_dt, 0.0000000000_dt,  0.0000000000_dt, 0.2937234044_dt, 0.0000000000_dt,  0.0000000000_dt,  0.1601343751_dt, 0.0000000000_dt };
    const Tensor realWeightsGradient = { 0.0000000000_dt, -0.5027827024_dt, 0.0000000000_dt, 0.0000000000_dt, 1.6601386070_dt, 0.0000000000_dt,
                                         0.0000000000_dt, -0.4307655990_dt, 0.0000000000_dt, 0.0000000000_dt, 0.3997098207_dt, 0.0000000000_dt };

    work.add<DataLayer>("data", DataParams{ { "in" }, 1, 4, 3 });
    Convolution1DParams params{ "in", "cnn1", kernelSize, filters, stride, padding, dilation, groups, false };
    work.add<Convolution1DLayer>("cnn1", params);

    TENSORS_CREATE(BATCH_SIZE);
    memory_manager["in"] = TORANGE(x);
    memory_manager["cnn1::Weights"] = TORANGE((raul::Tensor{ 0.2974873185_dt,
                                                             -0.2548298240_dt,
                                                             -0.1119259894_dt,
                                                             0.2709902525_dt,
                                                             -0.5435388088_dt,
                                                             0.3462468982_dt,
                                                             -0.1187755764_dt,
                                                             0.2937234044_dt,
                                                             0.0802614689_dt,
                                                             -0.0706931353_dt,
                                                             0.1601343751_dt,
                                                             0.0284817219_dt }));
    ASSERT_NO_THROW(work.forwardPassTraining());

    // Forward checks
    const Tensor output = memory_manager["cnn1"];
    EXPECT_EQ(output.size(), realOutput.size());
    for (size_t i = 0; i < output.size(); ++i)
    {
        CHECK_NEAR(output[i], realOutput[i], EPSILON);
    }

    // Backward checks
    memory_manager[Name("cnn1").grad()] = 1_dt;
    ASSERT_NO_THROW(work.backwardPassTraining());
    const Tensor inputNabla = memory_manager[Name("in").grad()];
    EXPECT_EQ(inputNabla.size(), realInputNabla.size());
    for (size_t i = 0; i < inputNabla.size(); ++i)
    {
        CHECK_NEAR(inputNabla[i], realInputNabla[i], EPSILON);
    }

    const Tensor weightsGradient = memory_manager[Name("cnn1::Weights").grad()];
    EXPECT_EQ(weightsGradient.size(), realWeightsGradient.size());
    for (size_t i = 0; i < weightsGradient.size(); ++i)
    {
        CHECK_NEAR(weightsGradient[i], realWeightsGradient[i], EPSILON);
    }
}

TEST(TestCNN1DLayer, DilationGpuUnit)
{
    PROFILE_TEST

    GPU_ONLY_TEST

    using namespace raul;

    const size_t filters = 3;
    const size_t kernelSize = 3;
    const size_t stride = 2;
    const size_t padding = 1;
    const size_t dilation = 3;
    const size_t groups = 1;
    const size_t BATCH_SIZE = 2;
    const auto EPSILON = 1e-6_dt;

    const Tensor x = { -0.59103316_dt, -0.56924808_dt, 0.91997141_dt,  1.11081612_dt,  1.28987408_dt,  -1.47817397_dt, 2.56723285_dt,  -0.47311980_dt, 1.32527483_dt,  -1.62932599_dt,
                       -0.54974365_dt, -0.47983426_dt, -0.49968153_dt, -1.06698036_dt, 1.11493957_dt,  -0.14067143_dt, -0.08519430_dt, -0.09334823_dt, 0.68705022_dt,  -0.83831537_dt,
                       0.00089182_dt,  0.84189409_dt,  -0.40003416_dt, 1.03946197_dt,  0.27995500_dt,  0.07324605_dt,  1.11331844_dt,  0.28226724_dt,  0.43422565_dt,  -0.80249292_dt,
                       -1.29518616_dt, -0.75018150_dt, -0.92678940_dt, 0.20641631_dt,  -0.33344787_dt, -0.42883000_dt, 0.23291829_dt,  0.79688716_dt,  -0.18484163_dt, -0.37014726_dt };

    const Tensor realOutput = { 0.13889819_dt, 0.45487607_dt, -0.13117632_dt, -0.32477066_dt, 0.41605633_dt, -0.13640109_dt };
    const Tensor realInputNabla = { 0._dt, 0._dt, 0.26827440_dt,  0._dt, 0._dt, 0._dt, 0._dt, -0.05198006_dt, 0._dt, 0._dt, 0._dt, 0._dt, -0.19009148_dt, 0._dt, 0._dt,
                                    0._dt, 0._dt, -0.21962076_dt, 0._dt, 0._dt, 0._dt, 0._dt, 0.26827440_dt,  0._dt, 0._dt, 0._dt, 0._dt, -0.05198006_dt, 0._dt, 0._dt,
                                    0._dt, 0._dt, -0.19009148_dt, 0._dt, 0._dt, 0._dt, 0._dt, -0.21962076_dt, 0._dt, 0._dt };
    const Tensor realWeightsGradient = { 0._dt, 0.51993728_dt, 0._dt, 0._dt, -0.19085255_dt, 0._dt, 0._dt, -1.42647099_dt, 0._dt, 0._dt, 0.70353889_dt, 0._dt,
                                         0._dt, 0.51993728_dt, 0._dt, 0._dt, -0.19085255_dt, 0._dt, 0._dt, -1.42647099_dt, 0._dt, 0._dt, 0.70353889_dt, 0._dt,
                                         0._dt, 0.51993728_dt, 0._dt, 0._dt, -0.19085255_dt, 0._dt, 0._dt, -1.42647099_dt, 0._dt, 0._dt, 0.70353889_dt, 0._dt };
    const Tensor realBiasGradient = { 2.0_dt, 2.0_dt, 2.0_dt };

    WorkflowEager work(CompressionMode::NONE, CalculationMode::DETERMINISTIC, AllocationMode::STANDARD, ExecutionTarget::GPU);

    work.add<DataLayer>("data", DataParams{ { "in" }, 1, 4, 5 });
    work.add<Convolution1DLayer>("cnn1", Convolution1DParams{ "in", "cnn1", kernelSize, filters, stride, padding, dilation, groups, true });

    TENSORS_CREATE(BATCH_SIZE)

    raul::MemoryManagerGPU& memory_manager = work.getMemoryManager<MemoryManagerGPU>();

    memory_manager["in"] = TORANGE(x);
    memory_manager["cnn1::Weights"] =
        TORANGE((Tensor{ -0.04652963_dt, 0.03054590_dt,  0.26138845_dt,  -0.26779535_dt, -0.18173194_dt, -0.07308251_dt, -0.11252555_dt, 0.24941555_dt,  -0.18711334_dt,
                         -0.13288665_dt, -0.20168012_dt, -0.27036187_dt, -0.16851136_dt, 0.24814458_dt,  0.12881215_dt,  0.13991290_dt,  0.01518188_dt,  -0.14799897_dt,
                         0.04883941_dt,  -0.26953444_dt, -0.20858690_dt, -0.14882068_dt, 0.18213609_dt,  0.16925636_dt,  -0.12802599_dt, -0.01041609_dt, 0.18462527_dt,
                         0.28698149_dt,  0.11457001_dt,  0.03899795_dt,  0.19355273_dt,  -0.16997258_dt, 0.05379289_dt,  -0.22381142_dt, -0.20007673_dt, -0.14912483_dt }));
    memory_manager["cnn1::Biases"] = TORANGE((Tensor{ 0.13061771_dt, 0.11609371_dt, -0.17099744_dt }));

    work.forwardPassTraining();

    // Forward checks
    const Tensor& output = memory_manager["cnn1"];
    EXPECT_EQ(output.size(), realOutput.size());
    for (size_t i = 0; i < output.size(); ++i)
    {
        CHECK_NEAR(output[i], realOutput[i], EPSILON);
    }

    // Backward checks
    memory_manager[Name("cnn1").grad()] = 1_dt;
    work.backwardPassTraining();
    const Tensor& inputNabla = memory_manager[Name("in").grad()];
    EXPECT_EQ(inputNabla.size(), realInputNabla.size());
    for (size_t i = 0; i < inputNabla.size(); ++i)
    {
        CHECK_NEAR(inputNabla[i], realInputNabla[i], EPSILON);
    }

    const Tensor& weightsGradient = memory_manager[Name("cnn1::Weights").grad()];
    EXPECT_EQ(weightsGradient.size(), realWeightsGradient.size());
    for (size_t i = 0; i < weightsGradient.size(); ++i)
    {
        CHECK_NEAR(weightsGradient[i], realWeightsGradient[i], EPSILON);
    }

    const Tensor& biasGradient = memory_manager[raul::Name("cnn1::Biases").grad()];
    EXPECT_EQ(biasGradient.size(), realBiasGradient.size());
    for (size_t i = 0; i < biasGradient.size(); ++i)
    {
        CHECK_NEAR(biasGradient[i], realBiasGradient[i], EPSILON);
    }
}

TEST(TestCNN1DLayer, TFStyleGpuUnit)
{
    PROFILE_TEST

    GPU_ONLY_TEST

    using namespace raul;

    const size_t filters = 2;
    const size_t kernelSize = 3;
    const size_t stride = 1;
    const size_t padding = 2;
    const size_t dilation = 2;
    const size_t groups = 1;
    const size_t BATCH_SIZE = 2;
    const auto EPSILON = 1e-4_dt;

    const Tensor x = { 1.0_dt,  2.0_dt,  3.0_dt,  4.0_dt,  5.0_dt,  6.0_dt,  7.0_dt,  8.0_dt,  9.0_dt,  10.0_dt, 11.0_dt, 12.0_dt, 13.0_dt, 14.0_dt, 15.0_dt,
                       16.0_dt, 17.0_dt, 18.0_dt, 19.0_dt, 20.0_dt, 21.0_dt, 22.0_dt, 23.0_dt, 24.0_dt, 25.0_dt, 26.0_dt, 27.0_dt, 28.0_dt, 29.0_dt, 30.0_dt };
    const Tensor deltas = { 20.0_dt, 19.0_dt, 18.0_dt, 17.0_dt, 16.0_dt, 15.0_dt, 14.0_dt, 13.0_dt, 12.0_dt, 11.0_dt, 10._dt, 9.0_dt, 8.0_dt, 7.0_dt, 6.0_dt, 5.0_dt, 4.0_dt, 3.0_dt, 2.0_dt, 1.0_dt };

    const Tensor realOutput = { 23.500664_dt, 6.707356_dt,  34.570343_dt, 12.598474_dt, 54.964226_dt, 20.439703_dt, 33.24925_dt,   27.39572_dt, 50.05303_dt,  37.52758_dt,
                                78.84906_dt,  36.162945_dt, 89.91874_dt,  42.05406_dt,  179.46786_dt, 74.851875_dt, 117.268166_dt, 78.05503_dt, 134.07195_dt, 88.18689_dt };
    const Tensor realInputNabla = { 50.353176_dt, 55.0102_dt,   45.733826_dt,  44.427296_dt, 47.956177_dt, 40.756634_dt, 56.033997_dt, 56.119232_dt, 61.763016_dt, 27.10075_dt,
                                    17.55472_dt,  44.3279_dt,   23.73786_dt,   15.384884_dt, 38.55342_dt,  20.72377_dt,  19.740091_dt, 20.847855_dt, 14.797889_dt, 12.68607_dt,
                                    15.870661_dt, 17.577457_dt, 13.0737505_dt, 23.987835_dt, 10.286309_dt, 6.7055464_dt, 15.455521_dt, 6.923421_dt,  4.535712_dt,  9.681044_dt };
    const Tensor realWeightsGradient = { 372.0_dt,  303.0_dt,  426.0_dt,  351.0_dt,  480.0_dt,  399.0_dt,  1100.0_dt, 955.0_dt,  1210.0_dt,
                                         1055.0_dt, 1320.0_dt, 1155.0_dt, 1116.0_dt, 1011.0_dt, 1194.0_dt, 1083.0_dt, 1272.0_dt, 1155.0_dt };

    WorkflowEager work(CompressionMode::NONE, CalculationMode::DETERMINISTIC, AllocationMode::STANDARD, ExecutionTarget::GPU);
    work.add<DataLayer>("data", DataParams{ { "in" }, 1, 5, 3 });
    work.add<Convolution1DLayer>("cnn1", Convolution1DParams{ "in", "cnn1", kernelSize, filters, stride, padding, dilation, groups, false, false, true });

    TENSORS_CREATE(BATCH_SIZE)

    raul::MemoryManagerGPU& memory_manager = work.getMemoryManager<MemoryManagerGPU>();
    memory_manager["in"] = TORANGE(x);
    memory_manager["cnn1::Weights"] = TORANGE((raul::Tensor{ 1.7640524_dt,
                                                             0.4001572_dt,
                                                             0.978738_dt,
                                                             2.2408931_dt,
                                                             1.867558_dt,
                                                             -0.9772779_dt,
                                                             0.95008844_dt,
                                                             -0.1513572_dt,
                                                             -0.10321885_dt,
                                                             0.41059852_dt,
                                                             0.14404356_dt,
                                                             1.4542735_dt,
                                                             0.7610377_dt,
                                                             0.121675014_dt,
                                                             0.44386324_dt,
                                                             0.33367434_dt,
                                                             1.4940791_dt,
                                                             -0.20515826_dt }));
    memory_manager[Name("cnn1").grad()] = TORANGE(deltas);

    work.forwardPassTraining();
    const Tensor& output = memory_manager["cnn1"];
    EXPECT_EQ(output.size(), realOutput.size());
    for (size_t i = 0; i < output.size(); ++i)
    {
        CHECK_NEAR(output[i], realOutput[i], EPSILON);
    }

    work.backwardPassTraining();
    const Tensor& inputNabla = memory_manager[Name("in").grad()];
    EXPECT_EQ(inputNabla.size(), realInputNabla.size());
    for (size_t i = 0; i < inputNabla.size(); ++i)
    {
        CHECK_NEAR(inputNabla[i], realInputNabla[i], EPSILON);
    }

    const Tensor& weightsGradient = memory_manager[Name("cnn1::Weights").grad()];
    EXPECT_EQ(weightsGradient.size(), realWeightsGradient.size());
    for (size_t i = 0; i < weightsGradient.size(); ++i)
    {
        CHECK_NEAR(weightsGradient[i], realWeightsGradient[i], EPSILON);
    }
}

TEST(TestCNN1DLayer, DoubleStep2GpuUnit)
{
    PROFILE_TEST

    GPU_ONLY_TEST

    using namespace raul;

    const size_t filters = 3;
    const size_t kernelSize = 2;
    const size_t stride = 2;
    const size_t padding = 1;
    const size_t dilation = 2;
    const size_t groups = 1;
    const size_t BATCH_SIZE = 2;
    const auto EPSILON = 1e-6_dt;

    const Tensor x = { -1.55509508_dt, -0.34136039_dt, 1.85300612_dt, 0.46809641_dt,  -0.15771244_dt, 1.44366014_dt,  0.26604941_dt,  0.16645534_dt,  1.58633423_dt,  0.94629836_dt,
                       -0.84367675_dt, 0.93182659_dt,  1.25900924_dt, 2.00498056_dt,  0.17784372_dt,  -0.23033547_dt, -0.39175439_dt, 0.54329473_dt,  -0.39515755_dt, 0.20552567_dt,
                       -0.45032975_dt, -0.57307708_dt, 3.41050267_dt, -1.53118432_dt, -1.23413503_dt, 1.81972528_dt,  -0.55152869_dt, -1.32532597_dt, 0.18855357_dt,  -0.06907269_dt };

    const Tensor realOutput[] = {
        { -0.32944888_dt,
          -0.66623712_dt,
          -1.12647843_dt,
          -0.33430994_dt,
          -0.13762742_dt,
          -0.48640594_dt,
          0.47286037_dt,
          0.77560359_dt,
          0.02234657_dt,
          -0.25076321_dt,
          0.39784792_dt,
          0.20927989_dt,
          -0.32223558_dt,
          -0.39425132_dt,
          -0.14236981_dt,
          0.11043543_dt,
          0.43837607_dt,
          0.59051555_dt },
        { -0.29641706_dt, -0.35044304_dt, -0.47054291_dt, -0.26309419_dt, 0.51632619_dt, 0.55646789_dt, -0.02319643_dt, -0.21444283_dt, -0.12676388_dt, -0.22079742_dt, 0.14213601_dt, 0.13021433_dt }
    };
    const Tensor deltas = { 1.71789527_dt,  -0.85072410_dt, -0.84863919_dt, -0.18478005_dt, -1.19375718_dt, -0.22327407_dt,
                            -1.27057660_dt, 0.01933064_dt,  0.88683778_dt,  0.05517140_dt,  0.68803781_dt,  1.23262453_dt };
    const Tensor realWeightsGradient = { 0.65647715_dt,  -1.14853525_dt, 0.62395668_dt,  2.09043622_dt, 0.19213718_dt, 1.68474627_dt,  0.13944964_dt,  0.67229474_dt, -0.23088974_dt,
                                         -1.11849952_dt, -0.54473877_dt, -0.81111443_dt, 0.55264896_dt, 1.05027425_dt, -0.51726854_dt, -0.19892836_dt, 0.36443216_dt, -0.42750430_dt };
    const Tensor realBiasesGradient = { -0.52708888_dt, -0.11514002_dt, 0.73527777_dt };

    WorkflowEager work(CompressionMode::NONE, CalculationMode::DETERMINISTIC, AllocationMode::STANDARD, ExecutionTarget::GPU);

    work.add<DataLayer>("data", DataParams{ { "in" }, 1, 3, 5 });
    work.add<Convolution1DLayer>("cnn1", Convolution1DParams{ "in", "cnn1", kernelSize, filters, stride, padding, dilation, groups, true, false });
    work.add<Convolution1DLayer>("cnn2", Convolution1DParams{ { { "cnn1" }, { "out" }, Name{ "cnn1" } }, kernelSize, filters, stride, padding, dilation, groups, true, false });

    TENSORS_CREATE(BATCH_SIZE)

    raul::MemoryManagerGPU& memory_manager = work.getMemoryManager<MemoryManagerGPU>();
    memory_manager["in"] = TORANGE(x);
    memory_manager["cnn1::Weights"] = TORANGE((Tensor{ -0.00305648_dt,
                                                       0.21900219_dt,
                                                       -0.33600679_dt,
                                                       -0.30044585_dt,
                                                       -0.15723862_dt,
                                                       0.10947479_dt,
                                                       -0.00808870_dt,
                                                       0.32369578_dt,
                                                       -0.03622961_dt,
                                                       0.10802763_dt,
                                                       -0.12337797_dt,
                                                       -0.08024748_dt,
                                                       -0.39001942_dt,
                                                       -0.27037555_dt,
                                                       -0.16828938_dt,
                                                       0.01512298_dt,
                                                       0.16139492_dt,
                                                       0.24495828_dt }));
    memory_manager["cnn1::Biases"] = TORANGE((Tensor{ -0.27676830_dt, -0.17777696_dt, 0.14828277_dt }));
    memory_manager[Name("out").grad()] = TORANGE(deltas);

    // see conv1d_double_step.py
    // Forward
    work.forwardPassTraining();
    // Step 1
    const Tensor& interOutput = memory_manager["cnn1"];
    EXPECT_EQ(interOutput.size(), realOutput[0].size());
    for (size_t i = 0; i < interOutput.size(); ++i)
    {
        CHECK_NEAR(interOutput[i], realOutput[0][i], EPSILON);
    }
    // Step 2
    const Tensor& output = memory_manager["out"];
    EXPECT_EQ(output.size(), realOutput[1].size());
    for (size_t i = 0; i < output.size(); ++i)
    {
        CHECK_NEAR(output[i], realOutput[1][i], EPSILON);
    }

    // Backward
    work.backwardPassTraining();

    const Tensor& weightsGradient = memory_manager[Name("cnn1::Weights").grad()];
    EXPECT_EQ(weightsGradient.size(), realWeightsGradient.size());
    for (size_t i = 0; i < weightsGradient.size(); ++i)
    {
        CHECK_NEAR(weightsGradient[i], realWeightsGradient[i], EPSILON);
    }
    const Tensor& biasesGradient = memory_manager[Name("cnn1::Biases").grad()];
    EXPECT_EQ(biasesGradient.size(), realBiasesGradient.size());
    for (size_t i = 0; i < biasesGradient.size(); ++i)
    {
        CHECK_NEAR(biasesGradient[i], realBiasesGradient[i], EPSILON);
    }
}

} // UT namespace