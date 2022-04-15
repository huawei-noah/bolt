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

#include <cstdio>
#include <tests/tools/TestTools.h>

#include <training/api/API.h>
#include <training/common/Test.h>
#include <training/common/Train.h>
#include <training/layers/BasicLayer.h>
#include <training/network/Layers.h>
#include <training/network/Workflow.h>
#include <training/optimizers/SGD.h>

#define CIFARTESTDIR "cifarNINTrained_BS16"
#define USE_DUMMY_DATASET 1

namespace UT
{

TEST(TestNINCifar, TopologyUnit)
{
    PROFILE_TEST
    const size_t golden_trainable_parameters = 966986U;

    const size_t BATCH_SIZE = 64;

    const size_t NUM_CLASSES = 10;
    const size_t IMAGE_SIZE = 32;
    const size_t IMAGE_CHANNELS = 3;

    const size_t CONV1_FILTERS = 192;
    const size_t CONV1_KERNEL_SIZE = 5;
    const size_t CONV1_STRIDE = 1;
    const size_t CONV1_PADDING = 2;

    const size_t CONV2_FILTERS = 160;
    const size_t CONV2_KERNEL_SIZE = 1;

    const size_t CONV3_FILTERS = 96;
    const size_t CONV3_KERNEL_SIZE = 1;

    const size_t MAXPOOL_KERNEL = 3;
    const size_t MAXPOOL_STRIDE = 2;
    const size_t MAXPOOL_PADDING = 1;

    const size_t CONV4_FILTERS = 192;
    const size_t CONV4_KERNEL_SIZE = 5;
    const size_t CONV4_STRIDE = 1;
    const size_t CONV4_PADDING = 2;

    const size_t CONV5_FILTERS = 192;
    const size_t CONV5_KERNEL_SIZE = 1;

    const size_t CONV6_FILTERS = 192;
    const size_t CONV6_KERNEL_SIZE = 1;

    const size_t AVGPOOL1_KERNEL = 3;
    const size_t AVGPOOL1_STRIDE = 2;
    const size_t AVGPOOL1_PADDING = 1;

    const size_t CONV7_FILTERS = 192;
    const size_t CONV7_KERNEL_SIZE = 3;
    const size_t CONV7_STRIDE = 1;
    const size_t CONV7_PADDING = 1;

    const size_t CONV8_FILTERS = 192;
    const size_t CONV8_KERNEL_SIZE = 1;

    const size_t CONV9_FILTERS = 10;
    const size_t CONV9_KERNEL_SIZE = 1;

    const size_t AVGPOOL2_KERNEL = 8;
    const size_t AVGPOOL2_STRIDE = 1;

    printf("begin creating graph\n");
    raul::Workflow work;
    work.add<raul::DataLayer>("data", raul::DataParams{ { "data", "labels" }, IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE, NUM_CLASSES });

    work.add<raul::Convolution2DLayer>("conv1", raul::Convolution2DParams{ { "data" }, { "conv1" }, CONV1_KERNEL_SIZE, CONV1_FILTERS, CONV1_STRIDE, CONV1_PADDING });
    work.add<raul::ReLUActivation>("relu1", raul::BasicParams{ { "conv1" }, { "relu1" } });
    work.add<raul::Convolution2DLayer>("conv2", raul::Convolution2DParams{ { "relu1" }, { "conv2" }, CONV2_KERNEL_SIZE, CONV2_FILTERS });
    work.add<raul::ReLUActivation>("relu2", raul::BasicParams{ { "conv2" }, { "relu2" } });
    work.add<raul::Convolution2DLayer>("conv3", raul::Convolution2DParams{ { "relu2" }, { "conv3" }, CONV3_KERNEL_SIZE, CONV3_FILTERS });
    work.add<raul::ReLUActivation>("relu3", raul::BasicParams{ { "conv3" }, { "relu3" } });
    work.add<raul::MaxPoolLayer2D>("mp", raul::Pool2DParams{ { "relu3" }, { "mp" }, MAXPOOL_KERNEL, MAXPOOL_STRIDE, MAXPOOL_PADDING });

    work.add<raul::Convolution2DLayer>("conv4", raul::Convolution2DParams{ { "mp" }, { "conv4" }, CONV4_KERNEL_SIZE, CONV4_FILTERS, CONV4_STRIDE, CONV4_PADDING });
    work.add<raul::ReLUActivation>("relu4", raul::BasicParams{ { "conv4" }, { "relu4" } });
    work.add<raul::Convolution2DLayer>("conv5", raul::Convolution2DParams{ { "relu4" }, { "conv5" }, CONV5_KERNEL_SIZE, CONV5_FILTERS });
    work.add<raul::ReLUActivation>("relu5", raul::BasicParams{ { "conv5" }, { "relu5" } });
    work.add<raul::Convolution2DLayer>("conv6", raul::Convolution2DParams{ { "relu5" }, { "conv6" }, CONV6_KERNEL_SIZE, CONV6_FILTERS });
    work.add<raul::ReLUActivation>("relu6", raul::BasicParams{ { "conv6" }, { "relu6" } });
    work.add<raul::AveragePoolLayer>("avg1", raul::Pool2DParams{ { "relu6" }, { "avg1" }, AVGPOOL1_KERNEL, AVGPOOL1_STRIDE, AVGPOOL1_PADDING });

    work.add<raul::Convolution2DLayer>("conv7", raul::Convolution2DParams{ { "avg1" }, { "conv7" }, CONV7_KERNEL_SIZE, CONV7_FILTERS, CONV7_STRIDE, CONV7_PADDING });
    work.add<raul::ReLUActivation>("relu7", raul::BasicParams{ { "conv7" }, { "relu7" } });
    work.add<raul::Convolution2DLayer>("conv8", raul::Convolution2DParams{ { "relu7" }, { "conv8" }, CONV8_KERNEL_SIZE, CONV8_FILTERS });
    work.add<raul::ReLUActivation>("relu8", raul::BasicParams{ { "conv8" }, { "relu8" } });
    work.add<raul::Convolution2DLayer>("conv9", raul::Convolution2DParams{ { "relu8" }, { "conv9" }, CONV9_KERNEL_SIZE, CONV9_FILTERS });
    work.add<raul::ReLUActivation>("relu9", raul::BasicParams{ { "conv9" }, { "relu9" } });
    work.add<raul::AveragePoolLayer>("avg2", raul::Pool2DParams{ { "relu9" }, { "avg2" }, AVGPOOL2_KERNEL, AVGPOOL2_STRIDE });

    work.add<raul::SoftMaxActivation>("softmax", raul::BasicParamsWithDim{ { "avg2" }, { "softmax" } });
    work.add<raul::CrossEntropyLoss>("loss", raul::LossParams{ { "softmax", "labels" }, { "loss" } });

    work.preparePipelines();
    work.setBatchSize(BATCH_SIZE);
    work.prepareMemoryForTraining();

    printf("end creating graph\n");

    // Checks
    EXPECT_EQ(tools::get_size_of_trainable_params(work), golden_trainable_parameters);
}

TEST(TestNINCifar, TrainingNINCifar)
{
    PROFILE_TEST
    const raul::dtype LEARNING_RATE = TODTYPE(0.01);
    const size_t BATCH_SIZE = 64;
    [[maybe_unused]] const raul::dtype EPSILON_ACCURACY = TODTYPE(1e-1);
    [[maybe_unused]] const raul::dtype EPSILON_LOSS = TODTYPE(1e-4);

    const size_t NUM_CLASSES = 10;
    [[maybe_unused]] const raul::dtype acc1 = TODTYPE(10.49f);
    [[maybe_unused]] const raul::dtype acc2 = TODTYPE(19.35f);
    const size_t IMAGE_SIZE = 32;
    const size_t IMAGE_CHANNELS = 3;

    const size_t CONV1_FILTERS = 192;
    const size_t CONV1_KERNEL_SIZE = 5;
    const size_t CONV1_STRIDE = 1;
    const size_t CONV1_PADDING = 2;

    const size_t CONV2_FILTERS = 160;
    const size_t CONV2_KERNEL_SIZE = 1;

    const size_t CONV3_FILTERS = 96;
    const size_t CONV3_KERNEL_SIZE = 1;

    const size_t MAXPOOL_KERNEL = 3;
    const size_t MAXPOOL_STRIDE = 2;
    const size_t MAXPOOL_PADDING = 1;

    const size_t CONV4_FILTERS = 192;
    const size_t CONV4_KERNEL_SIZE = 5;
    const size_t CONV4_STRIDE = 1;
    const size_t CONV4_PADDING = 2;

    const size_t CONV5_FILTERS = 192;
    const size_t CONV5_KERNEL_SIZE = 1;

    const size_t CONV6_FILTERS = 192;
    const size_t CONV6_KERNEL_SIZE = 1;

    const size_t AVGPOOL1_KERNEL = 3;
    const size_t AVGPOOL1_STRIDE = 2;
    const size_t AVGPOOL1_PADDING = 1;

    const size_t CONV7_FILTERS = 192;
    const size_t CONV7_KERNEL_SIZE = 3;
    const size_t CONV7_STRIDE = 1;
    const size_t CONV7_PADDING = 1;

    const size_t CONV8_FILTERS = 192;
    const size_t CONV8_KERNEL_SIZE = 1;

    const size_t CONV9_FILTERS = 10;
    const size_t CONV9_KERNEL_SIZE = 1;

    const size_t AVGPOOL2_KERNEL = 8;
    const size_t AVGPOOL2_STRIDE = 1;

    printf("begin creating graph\n");
    raul::Workflow work;
    work.add<raul::DataLayer>("data", raul::DataParams{ { "data", "labels" }, IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE, NUM_CLASSES });

    work.add<raul::Convolution2DLayer>("conv1", raul::Convolution2DParams{ { "data" }, { "conv1" }, CONV1_KERNEL_SIZE, CONV1_FILTERS, CONV1_STRIDE, CONV1_PADDING });
    work.add<raul::ReLUActivation>("relu1", raul::BasicParams{ { "conv1" }, { "relu1" } });
    work.add<raul::Convolution2DLayer>("conv2", raul::Convolution2DParams{ { "relu1" }, { "conv2" }, CONV2_KERNEL_SIZE, CONV2_FILTERS });
    work.add<raul::ReLUActivation>("relu2", raul::BasicParams{ { "conv2" }, { "relu2" } });
    work.add<raul::Convolution2DLayer>("conv3", raul::Convolution2DParams{ { "relu2" }, { "conv3" }, CONV3_KERNEL_SIZE, CONV3_FILTERS });
    work.add<raul::ReLUActivation>("relu3", raul::BasicParams{ { "conv3" }, { "relu3" } });
    work.add<raul::MaxPoolLayer2D>("mp", raul::Pool2DParams{ { "relu3" }, { "mp" }, MAXPOOL_KERNEL, MAXPOOL_STRIDE, MAXPOOL_PADDING });

    work.add<raul::Convolution2DLayer>("conv4", raul::Convolution2DParams{ { "mp" }, { "conv4" }, CONV4_KERNEL_SIZE, CONV4_FILTERS, CONV4_STRIDE, CONV4_PADDING });
    work.add<raul::ReLUActivation>("relu4", raul::BasicParams{ { "conv4" }, { "relu4" } });
    work.add<raul::Convolution2DLayer>("conv5", raul::Convolution2DParams{ { "relu4" }, { "conv5" }, CONV5_KERNEL_SIZE, CONV5_FILTERS });
    work.add<raul::ReLUActivation>("relu5", raul::BasicParams{ { "conv5" }, { "relu5" } });
    work.add<raul::Convolution2DLayer>("conv6", raul::Convolution2DParams{ { "relu5" }, { "conv6" }, CONV6_KERNEL_SIZE, CONV6_FILTERS });
    work.add<raul::ReLUActivation>("relu6", raul::BasicParams{ { "conv6" }, { "relu6" } });
    work.add<raul::AveragePoolLayer>("avg1", raul::Pool2DParams{ { "relu6" }, { "avg1" }, AVGPOOL1_KERNEL, AVGPOOL1_STRIDE, AVGPOOL1_PADDING });

    work.add<raul::Convolution2DLayer>("conv7", raul::Convolution2DParams{ { "avg1" }, { "conv7" }, CONV7_KERNEL_SIZE, CONV7_FILTERS, CONV7_STRIDE, CONV7_PADDING });
    work.add<raul::ReLUActivation>("relu7", raul::BasicParams{ { "conv7" }, { "relu7" } });
    work.add<raul::Convolution2DLayer>("conv8", raul::Convolution2DParams{ { "relu7" }, { "conv8" }, CONV8_KERNEL_SIZE, CONV8_FILTERS });
    work.add<raul::ReLUActivation>("relu8", raul::BasicParams{ { "conv8" }, { "relu8" } });
    work.add<raul::Convolution2DLayer>("conv9", raul::Convolution2DParams{ { "relu8" }, { "conv9" }, CONV9_KERNEL_SIZE, CONV9_FILTERS });
    work.add<raul::ReLUActivation>("relu9", raul::BasicParams{ { "conv9" }, { "relu9" } });
    work.add<raul::AveragePoolLayer>("avg2", raul::Pool2DParams{ { "relu9" }, { "avg2" }, AVGPOOL2_KERNEL, AVGPOOL2_STRIDE });

    work.add<raul::SoftMaxActivation>("softmax", raul::BasicParamsWithDim{ { "avg2" }, { "softmax" } });
    work.add<raul::CrossEntropyLoss>("loss", raul::LossParams{ { "softmax", "labels" }, { "loss" } });

    work.preparePipelines();
    work.setBatchSize(BATCH_SIZE);
    work.prepareMemoryForTraining();

    printf("end creating graph\n");
    printf("\n");
    work.printInfo(std::cout);
    printf("\n");
    raul::MemoryManager& memoryManager = work.getMemoryManager();
    raul::DataLoader dataLoader;

#if !USE_DUMMY_DATASET
    raul::CIFAR10 cifar;
    printf("begin loading dataset\n");
    ASSERT_EQ(cifar.loadingData(tools::getTestAssetsDir() / "CIFAR"), true);
    printf("end reading dataset\n");
#endif

    printf("begin reading weights\n");

    size_t filterSizes[] = { CONV1_FILTERS, CONV2_FILTERS, CONV3_FILTERS, CONV4_FILTERS, CONV5_FILTERS, CONV6_FILTERS, CONV7_FILTERS, CONV8_FILTERS, CONV9_FILTERS };
    size_t kernelSizes[] = {
        CONV1_KERNEL_SIZE, CONV2_KERNEL_SIZE, CONV3_KERNEL_SIZE, CONV4_KERNEL_SIZE, CONV5_KERNEL_SIZE, CONV6_KERNEL_SIZE, CONV7_KERNEL_SIZE, CONV8_KERNEL_SIZE, CONV9_KERNEL_SIZE
    };

    memoryManager["conv1::Weights"] =
        dataLoader.loadFilters((tools::getTestAssetsDir() / "test_cnn_layer" / CIFARTESTDIR / "0_conv1.weight_").string(), 0, ".data", kernelSizes[0], kernelSizes[0], IMAGE_CHANNELS, filterSizes[0]);
    memoryManager["conv1::Biases"] = dataLoader.loadData(tools::getTestAssetsDir() / "test_cnn_layer/" CIFARTESTDIR "/0_conv1.bias.data", 1, filterSizes[0]);

    for (size_t w = 0; w < 8; ++w)
    {
        memoryManager["conv" + Conversions::toString(w + 2) + "::Weights"] =
            dataLoader.loadFilters((tools::getTestAssetsDir() / "test_cnn_layer" / CIFARTESTDIR / ("0_conv" + Conversions::toString(w + 2) + ".weight_")).string(),
                                   0,
                                   ".data",
                                   kernelSizes[w + 1],
                                   kernelSizes[w + 1],
                                   filterSizes[w],
                                   filterSizes[w + 1]);
        memoryManager["conv" + Conversions::toString(w + 2) + "::Biases"] =
            dataLoader.loadData(tools::getTestAssetsDir() / ("test_cnn_layer/" CIFARTESTDIR "/0_conv" + Conversions::toString(w + 2) + ".bias.data"), 1, filterSizes[w + 1]);
    }

    printf("end reading weights\n");

#if !USE_DUMMY_DATASET
    const size_t stepsAmountTrain = cifar.getTrainImageAmount() / BATCH_SIZE;
    const size_t stepsAmountTest = cifar.getTestImageAmount() / BATCH_SIZE;
    raul::Tensor idealLosses(stepsAmountTrain / 100, 0);
    bool hasLoss = (0 == raul::DataLoader::readArrayFromTextFile(tools::getTestAssetsDir() / "test_cnn_layer/" CIFARTESTDIR "/loss.data", idealLosses, 1, idealLosses.size()));
    printf("Test on %zu images\n", cifar.getTestImageAmount());
    printf("begin inference\n");
    raul::dtype testAcc = cifar.testNetwork(work);
    printf("end inference\n");
    printf("Inference time = %.3fs \n", cifar.getTestingTime());
    CHECK_NEAR(testAcc, acc1, EPSILON_ACCURACY);
    printf("Test accuracy = %f\n", testAcc);
    printf("Train on %zu images\n", cifar.getTrainImageAmount());
    size_t idealLossIndex = 0;
    float lastTime = 0;
#else
    const size_t stepsAmountTrain = 50000 / BATCH_SIZE;
    [[maybe_unused]] const size_t stepsAmountTest = 10000 / BATCH_SIZE;
#endif
    printf("begin training\n");

    for (size_t q = 0; q < stepsAmountTrain; ++q)
    {
        auto sgd = std::make_shared<raul::optimizers::SGD>(LEARNING_RATE);
#if !USE_DUMMY_DATASET
        raul::dtype testLoss = cifar.oneTrainIteration(work, sgd.get(), q);
        if (q % 100 == 0)
        {
            if (hasLoss) CHECK_NEAR(testLoss, idealLosses[idealLossIndex++], EPSILON_LOSS);

            if (q != 0)
            {
                printf("iteration = %zu / %zu, loss = %f, time = %.3fs\n", q, stepsAmountTrain, testLoss, cifar.getTrainingTime() - lastTime);
                lastTime = cifar.getTrainingTime();
            }
            else
                printf("iteration = %zu / %zu, loss = %f\n", q, stepsAmountTrain, testLoss);
        }
#else
        raul::Tensor& inputs = work.getMemoryManager().getTensor("data");
        // raul::Tensor& softmax = work.getMemoryManager().getTensor("softmax");
        raul::Tensor& loss = work.getMemoryManager().getTensor("loss");
        raul::Tensor& labels = work.getMemoryManager().getTensor("labels");

        const size_t batchSize = work.getBatchSize();

        size_t mImageSize = 32;
        size_t mImageDepth = 3;
        size_t mNumClasses = 10;

        raul::Tensor trainImagesBatched("", batchSize, mImageDepth, mImageSize, mImageSize);
        raul::TensorU8 mTrainLabels(batchSize, 1);
        auto& mEncodedTrainLabels = dataLoader.buildOneHotVector(mTrainLabels, mNumClasses);

        inputs = TORANGE(trainImagesBatched);
        labels = TORANGE(mEncodedTrainLabels);

        work.forwardPassTraining();
        work.backwardPassTraining();

        auto paramsAndWeights = work.getTrainableParameters();
        for (auto s : paramsAndWeights)
            sgd->operator()(work.getMemoryManager(), s.Param, s.Gradient);

        raul::dtype totalLoss = loss[0] / static_cast<raul::dtype>(batchSize);

        if (q % 100 == 0) printf("iteration = %zu / %zu, loss = %f\n", q, stepsAmountTrain, totalLoss);
#endif
    }
    printf("end training\n");
#if !USE_DUMMY_DATASET
    printf("Training taken = %.3fs \n", cifar.getTrainingTime());
    printf("Test on %zu images\n", cifar.getTestImageAmount());
    printf("begin inference\n");
    testAcc = cifar.testNetwork(work);
    printf("end inference\n");
    printf("Inference time = %.3fs \n", cifar.getTestingTime());
    CHECK_NEAR(testAcc, acc2, EPSILON_ACCURACY);
    printf("Test accuracy = %f\n", testAcc);
#endif
    printf("Memory taken = %.2fMB \n\n", static_cast<float>(work.getMemoryManager().getTotalMemory()) / (1024.0f * 1024.0f));
}

TEST(TestNINCifar, TrainingNINCifarCanonical)
{
    PROFILE_TEST
    bool useMicroBatches = false;
    bool useCheckpointing = false;

    const auto LEARNING_RATE = 0.0001_dt;
    const size_t BATCH_SIZE = 32;
    const size_t MICROBATCH_SIZE = 16;
    const auto EPSILON_ACCURACY = 1e-1_dt;

    const size_t NUM_CLASSES = 10;
    const auto acc1 = 87.18_dt;
    const auto acc2 = 87.63_dt;
    const size_t IMAGE_SIZE = 32;
    const size_t IMAGE_CHANNELS = 3;

    const size_t CONV1_FILTERS = 192;
    const size_t CONV1_KERNEL_SIZE = 5;
    const size_t CONV1_STRIDE = 1;
    const size_t CONV1_PADDING = 2;

    const size_t CONV2_FILTERS = 160;
    const size_t CONV2_KERNEL_SIZE = 1;

    const size_t CONV3_FILTERS = 96;
    const size_t CONV3_KERNEL_SIZE = 1;

    const size_t MAXPOOL_KERNEL = 3;
    const size_t MAXPOOL_STRIDE = 2;
    const size_t MAXPOOL_PADDING = 1;

    const size_t CONV4_FILTERS = 192;
    const size_t CONV4_KERNEL_SIZE = 5;
    const size_t CONV4_STRIDE = 1;
    const size_t CONV4_PADDING = 2;

    const size_t CONV5_FILTERS = 192;
    const size_t CONV5_KERNEL_SIZE = 1;

    const size_t CONV6_FILTERS = 192;
    const size_t CONV6_KERNEL_SIZE = 1;

    const size_t AVGPOOL1_KERNEL = 3;
    const size_t AVGPOOL1_STRIDE = 2;
    const size_t AVGPOOL1_PADDING = 1;

    const size_t CONV7_FILTERS = 192;
    const size_t CONV7_KERNEL_SIZE = 3;
    const size_t CONV7_STRIDE = 1;
    const size_t CONV7_PADDING = 1;

    const size_t CONV8_FILTERS = 192;
    const size_t CONV8_KERNEL_SIZE = 1;

    const size_t CONV9_FILTERS = 10;
    const size_t CONV9_KERNEL_SIZE = 1;

    const size_t AVGPOOL2_KERNEL = 8;
    const size_t AVGPOOL2_STRIDE = 1;

    raul::Workflow work;
    work.add<raul::DataLayer>("data", raul::DataParams{ { "data", "labels" }, IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE, NUM_CLASSES });

    work.add<raul::Convolution2DLayer>("conv1", raul::Convolution2DParams{ { "data" }, { "conv1" }, CONV1_KERNEL_SIZE, CONV1_FILTERS, CONV1_STRIDE, CONV1_PADDING });
    work.add<raul::ReLUActivation>("relu1", raul::BasicParams{ { "conv1" }, { "relu1" } });
    work.add<raul::Convolution2DLayer>("conv2", raul::Convolution2DParams{ { "relu1" }, { "conv2" }, CONV2_KERNEL_SIZE, CONV2_FILTERS });
    work.add<raul::ReLUActivation>("relu2", raul::BasicParams{ { "conv2" }, { "relu2" } });
    work.add<raul::Convolution2DLayer>("conv3", raul::Convolution2DParams{ { "relu2" }, { "conv3" }, CONV3_KERNEL_SIZE, CONV3_FILTERS });
    work.add<raul::ReLUActivation>("relu3", raul::BasicParams{ { "conv3" }, { "relu3" } });
    work.add<raul::MaxPoolLayer2D>("mp", raul::Pool2DParams{ { "relu3" }, { "mp" }, MAXPOOL_KERNEL, MAXPOOL_STRIDE, MAXPOOL_PADDING });
    work.add<raul::DropoutLayer>("drop1", raul::DropoutParams{ { "mp" }, { "drop1" }, 0.5f });

    work.add<raul::Convolution2DLayer>("conv4", raul::Convolution2DParams{ { "drop1" }, { "conv4" }, CONV4_KERNEL_SIZE, CONV4_FILTERS, CONV4_STRIDE, CONV4_PADDING });
    work.add<raul::ReLUActivation>("relu4", raul::BasicParams{ { "conv4" }, { "relu4" } });
    work.add<raul::Convolution2DLayer>("conv5", raul::Convolution2DParams{ { "relu4" }, { "conv5" }, CONV5_KERNEL_SIZE, CONV5_FILTERS });
    work.add<raul::ReLUActivation>("relu5", raul::BasicParams{ { "conv5" }, { "relu5" } });
    work.add<raul::Convolution2DLayer>("conv6", raul::Convolution2DParams{ { "relu5" }, { "conv6" }, CONV6_KERNEL_SIZE, CONV6_FILTERS });
    work.add<raul::ReLUActivation>("relu6", raul::BasicParams{ { "conv6" }, { "relu6" } });
    work.add<raul::AveragePoolLayer>("avg1", raul::Pool2DParams{ { "relu6" }, { "avg1" }, AVGPOOL1_KERNEL, AVGPOOL1_STRIDE, AVGPOOL1_PADDING });
    work.add<raul::DropoutLayer>("drop2", raul::DropoutParams{ { "avg1" }, { "drop2" }, 0.5f });

    work.add<raul::Convolution2DLayer>("conv7", raul::Convolution2DParams{ { "drop2" }, { "conv7" }, CONV7_KERNEL_SIZE, CONV7_FILTERS, CONV7_STRIDE, CONV7_PADDING });
    work.add<raul::ReLUActivation>("relu7", raul::BasicParams{ { "conv7" }, { "relu7" } });
    work.add<raul::Convolution2DLayer>("conv8", raul::Convolution2DParams{ { "relu7" }, { "conv8" }, CONV8_KERNEL_SIZE, CONV8_FILTERS });
    work.add<raul::ReLUActivation>("relu8", raul::BasicParams{ { "conv8" }, { "relu8" } });
    work.add<raul::Convolution2DLayer>("conv9", raul::Convolution2DParams{ { "relu8" }, { "conv9" }, CONV9_KERNEL_SIZE, CONV9_FILTERS });
    work.add<raul::ReLUActivation>("relu9", raul::BasicParams{ { "conv9" }, { "relu9" } });
    work.add<raul::AveragePoolLayer>("avg2", raul::Pool2DParams{ { "relu9" }, { "avg2" }, AVGPOOL2_KERNEL, AVGPOOL2_STRIDE });

    work.add<raul::LogSoftMaxActivation>("softmax", raul::BasicParamsWithDim{ { "avg2" }, { "softmax" } });
    work.add<raul::NLLLoss>("loss", raul::LossParams{ { "softmax", "labels" }, { "loss" }, "custom_batch_mean" });

    if (useCheckpointing)
    {
        work.setCheckpoints({ "mp", "avg1", "avg2" });
        work.preparePipelines(raul::Workflow::Execution::Checkpointed);
    }
    else
    {
        work.preparePipelines();
    }
    work.setBatchSize(BATCH_SIZE);
    work.prepareMemoryForTraining();

    raul::MemoryManager& memoryManager = work.getMemoryManager();
    raul::DataLoader dataLoader;

    const std::string cifarDir = "cifarNINTrained_BS128_Canonical";

    size_t filterSizes[] = { CONV1_FILTERS, CONV2_FILTERS, CONV3_FILTERS, CONV4_FILTERS, CONV5_FILTERS, CONV6_FILTERS, CONV7_FILTERS, CONV8_FILTERS, CONV9_FILTERS };
    size_t kernelSizes[] = {
        CONV1_KERNEL_SIZE, CONV2_KERNEL_SIZE, CONV3_KERNEL_SIZE, CONV4_KERNEL_SIZE, CONV5_KERNEL_SIZE, CONV6_KERNEL_SIZE, CONV7_KERNEL_SIZE, CONV8_KERNEL_SIZE, CONV9_KERNEL_SIZE
    };
    size_t fileIndexes[] = { 2, 4, 8, 10, 12, 16, 18, 20 };

    memoryManager["conv1::Weights"] = dataLoader.loadFilters(
        (tools::getTestAssetsDir() / "test_cnn_layer" / cifarDir / "0_classifier.0.weight_").string(), 0, ".data", kernelSizes[0], kernelSizes[0], IMAGE_CHANNELS, filterSizes[0]);
    memoryManager["conv1::Biases"] = dataLoader.loadData(tools::getTestAssetsDir() / "test_cnn_layer" / cifarDir / "0_classifier.0.bias.data", 1, filterSizes[0]);

    for (size_t w = 0; w < 8; ++w)
    {
        memoryManager["conv" + Conversions::toString(w + 2) + "::Weights"] =
            dataLoader.loadFilters((tools::getTestAssetsDir() / "test_cnn_layer" / cifarDir / ("0_classifier." + Conversions::toString(fileIndexes[w]) + ".weight_")).string(),
                                   0,
                                   ".data",
                                   kernelSizes[w + 1],
                                   kernelSizes[w + 1],
                                   filterSizes[w],
                                   filterSizes[w + 1]);
        memoryManager["conv" + Conversions::toString(w + 2) + "::Biases"] =
            dataLoader.loadData(tools::getTestAssetsDir() / "test_cnn_layer" / cifarDir / ("0_classifier." + Conversions::toString(fileIndexes[w]) + ".bias.data"), 1, filterSizes[w + 1]);
    }

    auto trainData = raul::Dataset::CIFAR_Train(UT::tools::getTestAssetsDir() / "CIFAR");
    auto testData = raul::Dataset::CIFAR_Test(UT::tools::getTestAssetsDir() / "CIFAR");

    printf("Begin testing\n");
    raul::Test test(work, testData, { { { "images", "data" }, { "labels", "labels" } }, "softmax", "labels" });
    raul::dtype testAcc = test.run(useMicroBatches ? MICROBATCH_SIZE : BATCH_SIZE);
    CHECK_NEAR(testAcc, acc1, EPSILON_ACCURACY);
    printf("Test accuracy = %.2f\n", testAcc);

    raul::Train train(work, trainData, { { { "images", "data" }, { "labels", "labels" } }, "loss" });
    useMicroBatches ? train.useMicroBatches(BATCH_SIZE, MICROBATCH_SIZE) : train.useBatches(BATCH_SIZE);
    auto sgd = std::make_shared<raul::optimizers::SGD>(LEARNING_RATE);
    for (size_t epoch = 1; epoch <= 1; ++epoch)
    {
        printf("Epoch = %zu\n", epoch);

        raul::dtype averageLoss = 0.0_dt;

        for (size_t q = 0; q < train.numberOfIterations(); ++q)
        {
            raul::dtype testLoss = train.oneIteration(*sgd.get());
            averageLoss += testLoss;
            if (q % 100 == 0)
            {
                printf("iteration = %zu / %zu, loss = %f\n", q, static_cast<size_t>(train.numberOfIterations()), testLoss);
            }
        }

        printf("Average loss = %f\n", averageLoss / static_cast<float>(train.numberOfIterations()));
        testAcc = test.run(useMicroBatches ? MICROBATCH_SIZE : BATCH_SIZE);
        CHECK_NEAR(testAcc, acc2, EPSILON_ACCURACY);
        printf("Test accuracy = %.2f\n", testAcc);
    }

    printf("Memory taken = %.2fMB \n\n", static_cast<float>(work.getMemoryManager().getTotalMemory()) / (1024.0f * 1024.0f));
}

#if defined(ANDROID)
TEST(TestNINCifar, TrainingNINCifarCanonicalFP16)
{
    PROFILE_TEST
    bool useMicroBatches = false;
    bool useCheckpointing = false;

    const auto LEARNING_RATE = 0.0001_dt;
    const size_t BATCH_SIZE = 32;
    const size_t MICROBATCH_SIZE = 16;
    const auto EPSILON_ACCURACY = 1e-1_dt;

    const size_t NUM_CLASSES = 10;
    const auto acc1 = 87.18_dt;
    const auto acc2 = 87.63_dt;
    const size_t IMAGE_SIZE = 32;
    const size_t IMAGE_CHANNELS = 3;

    const size_t CONV1_FILTERS = 192;
    const size_t CONV1_KERNEL_SIZE = 5;
    const size_t CONV1_STRIDE = 1;
    const size_t CONV1_PADDING = 2;

    const size_t CONV2_FILTERS = 160;
    const size_t CONV2_KERNEL_SIZE = 1;

    const size_t CONV3_FILTERS = 96;
    const size_t CONV3_KERNEL_SIZE = 1;

    const size_t MAXPOOL_KERNEL = 3;
    const size_t MAXPOOL_STRIDE = 2;
    const size_t MAXPOOL_PADDING = 1;

    const size_t CONV4_FILTERS = 192;
    const size_t CONV4_KERNEL_SIZE = 5;
    const size_t CONV4_STRIDE = 1;
    const size_t CONV4_PADDING = 2;

    const size_t CONV5_FILTERS = 192;
    const size_t CONV5_KERNEL_SIZE = 1;

    const size_t CONV6_FILTERS = 192;
    const size_t CONV6_KERNEL_SIZE = 1;

    const size_t AVGPOOL1_KERNEL = 3;
    const size_t AVGPOOL1_STRIDE = 2;
    const size_t AVGPOOL1_PADDING = 1;

    const size_t CONV7_FILTERS = 192;
    const size_t CONV7_KERNEL_SIZE = 3;
    const size_t CONV7_STRIDE = 1;
    const size_t CONV7_PADDING = 1;

    const size_t CONV8_FILTERS = 192;
    const size_t CONV8_KERNEL_SIZE = 1;

    const size_t CONV9_FILTERS = 10;
    const size_t CONV9_KERNEL_SIZE = 1;

    const size_t AVGPOOL2_KERNEL = 8;
    const size_t AVGPOOL2_STRIDE = 1;

    raul::Workflow work(raul::CompressionMode::NONE, raul::CalculationMode::DETERMINISTIC, raul::AllocationMode::STANDARD, raul::ExecutionTarget::CPUFP16);
    work.add<raul::DataLayer>("data", raul::DataParams{ { "data", "labels" }, IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE, NUM_CLASSES });

    work.add<raul::Convolution2DLayer>("conv1", raul::Convolution2DParams{ { "data" }, { "conv1" }, CONV1_KERNEL_SIZE, CONV1_FILTERS, CONV1_STRIDE, CONV1_PADDING });
    work.add<raul::ReLUActivation>("relu1", raul::BasicParams{ { "conv1" }, { "relu1" } });
    work.add<raul::Convolution2DLayer>("conv2", raul::Convolution2DParams{ { "relu1" }, { "conv2" }, CONV2_KERNEL_SIZE, CONV2_FILTERS });
    work.add<raul::ReLUActivation>("relu2", raul::BasicParams{ { "conv2" }, { "relu2" } });
    work.add<raul::Convolution2DLayer>("conv3", raul::Convolution2DParams{ { "relu2" }, { "conv3" }, CONV3_KERNEL_SIZE, CONV3_FILTERS });
    work.add<raul::ReLUActivation>("relu3", raul::BasicParams{ { "conv3" }, { "relu3" } });
    work.add<raul::MaxPoolLayer2D>("mp", raul::Pool2DParams{ { "relu3" }, { "mp" }, MAXPOOL_KERNEL, MAXPOOL_STRIDE, MAXPOOL_PADDING });
    work.add<raul::DropoutLayer>("drop1", raul::DropoutParams{ { "mp" }, { "drop1" }, 0.5f });

    work.add<raul::Convolution2DLayer>("conv4", raul::Convolution2DParams{ { "drop1" }, { "conv4" }, CONV4_KERNEL_SIZE, CONV4_FILTERS, CONV4_STRIDE, CONV4_PADDING });
    work.add<raul::ReLUActivation>("relu4", raul::BasicParams{ { "conv4" }, { "relu4" } });
    work.add<raul::Convolution2DLayer>("conv5", raul::Convolution2DParams{ { "relu4" }, { "conv5" }, CONV5_KERNEL_SIZE, CONV5_FILTERS });
    work.add<raul::ReLUActivation>("relu5", raul::BasicParams{ { "conv5" }, { "relu5" } });
    work.add<raul::Convolution2DLayer>("conv6", raul::Convolution2DParams{ { "relu5" }, { "conv6" }, CONV6_KERNEL_SIZE, CONV6_FILTERS });
    work.add<raul::ReLUActivation>("relu6", raul::BasicParams{ { "conv6" }, { "relu6" } });
    work.add<raul::AveragePoolLayer>("avg1", raul::Pool2DParams{ { "relu6" }, { "avg1" }, AVGPOOL1_KERNEL, AVGPOOL1_STRIDE, AVGPOOL1_PADDING });
    work.add<raul::DropoutLayer>("drop2", raul::DropoutParams{ { "avg1" }, { "drop2" }, 0.5f });

    work.add<raul::Convolution2DLayer>("conv7", raul::Convolution2DParams{ { "drop2" }, { "conv7" }, CONV7_KERNEL_SIZE, CONV7_FILTERS, CONV7_STRIDE, CONV7_PADDING });
    work.add<raul::ReLUActivation>("relu7", raul::BasicParams{ { "conv7" }, { "relu7" } });
    work.add<raul::Convolution2DLayer>("conv8", raul::Convolution2DParams{ { "relu7" }, { "conv8" }, CONV8_KERNEL_SIZE, CONV8_FILTERS });
    work.add<raul::ReLUActivation>("relu8", raul::BasicParams{ { "conv8" }, { "relu8" } });
    work.add<raul::Convolution2DLayer>("conv9", raul::Convolution2DParams{ { "relu8" }, { "conv9" }, CONV9_KERNEL_SIZE, CONV9_FILTERS });
    work.add<raul::ReLUActivation>("relu9", raul::BasicParams{ { "conv9" }, { "relu9" } });
    work.add<raul::AveragePoolLayer>("avg2", raul::Pool2DParams{ { "relu9" }, { "avg2" }, AVGPOOL2_KERNEL, AVGPOOL2_STRIDE });

    work.add<raul::LogSoftMaxActivation>("softmax", raul::BasicParamsWithDim{ { "avg2" }, { "softmax" } });
    work.add<raul::NLLLoss>("loss", raul::LossParams{ { "softmax", "labels" }, { "loss" }, "custom_batch_mean" });

    if (useCheckpointing)
    {
        work.setCheckpoints({ "mp", "avg1", "avg2" });
        work.preparePipelines(raul::Workflow::Execution::Checkpointed);
    }
    else
    {
        work.preparePipelines();
    }
    work.setBatchSize(BATCH_SIZE);
    work.prepareMemoryForTraining();

    auto& memoryManager = work.getMemoryManager<raul::MemoryManagerFP16>();
    raul::DataLoader dataLoader;

    const std::string cifarDir = "cifarNINTrained_BS128_Canonical";

    size_t filterSizes[] = { CONV1_FILTERS, CONV2_FILTERS, CONV3_FILTERS, CONV4_FILTERS, CONV5_FILTERS, CONV6_FILTERS, CONV7_FILTERS, CONV8_FILTERS, CONV9_FILTERS };
    size_t kernelSizes[] = {
        CONV1_KERNEL_SIZE, CONV2_KERNEL_SIZE, CONV3_KERNEL_SIZE, CONV4_KERNEL_SIZE, CONV5_KERNEL_SIZE, CONV6_KERNEL_SIZE, CONV7_KERNEL_SIZE, CONV8_KERNEL_SIZE, CONV9_KERNEL_SIZE
    };
    size_t fileIndexes[] = { 2, 4, 8, 10, 12, 16, 18, 20 };

    memoryManager["conv1::Weights"] = dataLoader.loadFilters(
        (tools::getTestAssetsDir() / "test_cnn_layer" / cifarDir / "0_classifier.0.weight_").string(), 0, ".data", kernelSizes[0], kernelSizes[0], IMAGE_CHANNELS, filterSizes[0]);
    memoryManager["conv1::Biases"] = dataLoader.loadData(tools::getTestAssetsDir() / "test_cnn_layer" / cifarDir / "0_classifier.0.bias.data", 1, filterSizes[0]);

    for (size_t w = 0; w < 8; ++w)
    {
        memoryManager["conv" + Conversions::toString(w + 2) + "::Weights"] =
            dataLoader.loadFilters((tools::getTestAssetsDir() / "test_cnn_layer" / cifarDir / ("0_classifier." + Conversions::toString(fileIndexes[w]) + ".weight_")).string(),
                                   0,
                                   ".data",
                                   kernelSizes[w + 1],
                                   kernelSizes[w + 1],
                                   filterSizes[w],
                                   filterSizes[w + 1]);
        memoryManager["conv" + Conversions::toString(w + 2) + "::Biases"] =
            dataLoader.loadData(tools::getTestAssetsDir() / "test_cnn_layer" / cifarDir / ("0_classifier." + Conversions::toString(fileIndexes[w]) + ".bias.data"), 1, filterSizes[w + 1]);
    }

    auto trainData = raul::Dataset::CIFAR_Train(UT::tools::getTestAssetsDir() / "CIFAR");
    auto testData = raul::Dataset::CIFAR_Test(UT::tools::getTestAssetsDir() / "CIFAR");

    printf("Begin testing\n");
    raul::Test test(work, testData, { { { "images", "data" }, { "labels", "labels" } }, "softmax", "labels" });
    raul::dtype testAcc = test.run(useMicroBatches ? MICROBATCH_SIZE : BATCH_SIZE);
    CHECK_NEAR(testAcc, acc1, EPSILON_ACCURACY);
    printf("Test accuracy = %.2f\n", testAcc);

    raul::Train train(work, trainData, { { { "images", "data" }, { "labels", "labels" } }, "loss" });
    useMicroBatches ? train.useMicroBatches(BATCH_SIZE, MICROBATCH_SIZE) : train.useBatches(BATCH_SIZE);
    auto sgd = std::make_shared<raul::optimizers::SGD>(LEARNING_RATE);
    for (size_t epoch = 1; epoch <= 1; ++epoch)
    {
        printf("Epoch = %zu\n", epoch);

        raul::dtype averageLoss = 0.0_dt;

        for (size_t q = 0; q < train.numberOfIterations(); ++q)
        {
            raul::dtype testLoss = train.oneIteration(*sgd.get());
            averageLoss += testLoss;
            if (q % 100 == 0)
            {
                printf("iteration = %zu / %zu, loss = %f\n", q, static_cast<size_t>(train.numberOfIterations()), testLoss);
            }
        }

        printf("Average loss = %f\n", averageLoss / static_cast<float>(train.numberOfIterations()));
        testAcc = test.run(useMicroBatches ? MICROBATCH_SIZE : BATCH_SIZE);
        CHECK_NEAR(testAcc, acc2, EPSILON_ACCURACY);
        printf("Test accuracy = %.2f\n", testAcc);
    }

    printf("Memory taken = %.2fMB \n\n", static_cast<float>(work.getMemoryManager().getTotalMemory()) / (1024.0f * 1024.0f));
}
#endif

} // UT namespace
