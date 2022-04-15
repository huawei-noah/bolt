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

#include <algorithm>
#include <filesystem>
#include <tests/tools/TestTools.h>

#include "Bindings.h"
#include <training/network/Layers.h>

#define ASSERT_OK(x) ASSERT_EQ(x, STATUS_OK)
#define EXPECT_OK(x) EXPECT_EQ(x, STATUS_OK)
#define ASSERT_ERROR(x) ASSERT_NE(x, STATUS_OK)

namespace
{
template<typename T>
void transpose(T* matrix, size_t cols, size_t size)
{
    // https://stackoverflow.com/questions/9227747/in-place-transposition-of-a-matrix
    const size_t mn1 = size - 1;
    const size_t n = size / cols;
    std::vector<bool> visited(size);
    T* cycle = matrix;
    while (++cycle != (matrix + size))
    {
        if (visited[cycle - matrix]) continue;
        size_t a = cycle - matrix;
        do
        {
            a = a == mn1 ? mn1 : (n * a) % mn1;
            std::swap(*(matrix + a), *cycle);
            visited[a] = true;
        } while ((matrix + a) != cycle);
    }
}
} // anonymous namespace

namespace UT
{

TEST(TestBindings, MLPMnistTraining)
{
    PROFILE_TEST
    const raul::dtype LEARNING_RATE = 0.1f;
    const size_t BATCH_SIZE = 50;
    [[maybe_unused]] const raul::dtype EPSILON_ACCURACY = 1e-2f;
    const raul::dtype EPSILON_LOSS = 1e-5f;

    const size_t NUM_CLASSES = 10;

    const size_t MNIST_SIZE = 28;
    const size_t MNIST_DEPTH = 1;
    const size_t FC1_SIZE = 500;
    const size_t FC2_SIZE = 100;

    [[maybe_unused]] const raul::dtype acc1 = 3.24f;
    [[maybe_unused]] const raul::dtype acc2 = 91.51f;

    Graph_Description_t* desc = NULL;

    ASSERT_OK(create_graph_description_eager(&desc));

    {
        const char* tensors[] = { "data", "labels" };
        ASSERT_OK(add_data_layer_with_labels(desc, "data", tensors, 2, 1, MNIST_SIZE, MNIST_SIZE, NUM_CLASSES));
    }

    ASSERT_OK(add_reshape_layer(desc, "reshape", "data", "datar", 1, 1, -1));
    ASSERT_OK(add_linear_layer(desc, "fc1", "datar", "fc1", FC1_SIZE, true));
    ASSERT_OK(add_activation_layer(desc, "tanh", "fc1", "tanh", TANH_ACTIVATION));
    ASSERT_OK(add_linear_layer(desc, "fc2", "tanh", "fc2", FC2_SIZE, true));
    ASSERT_OK(add_activation_layer(desc, "sigmoid", "fc2", "sigmoid", SIGMOID_ACTIVATION));
    ASSERT_OK(add_linear_layer(desc, "fc3", "sigmoid", "fc3", NUM_CLASSES, true));
    ASSERT_OK(add_activation_layer(desc, "softmax", "fc3", "softmax", SOFTMAX_ACTIVATION));
    const char* loss_name = "loss";
    {
        const char* tensorsIn[] = { "softmax", "labels" };
        ASSERT_OK(add_loss_layer(desc, "loss", tensorsIn, loss_name, CROSS_ENTROPY_LOSS, 2, LOSS_REDUCTION_BATCH_MEAN));
    }

    Graph_t* graph = NULL;
    ASSERT_OK(create_graph(&desc, &graph, BATCH_SIZE));

    ASSERT_NE(graph, nullptr);

    raul::DataLoader dataLoader;

    auto weight = dataLoader.loadData(tools::getTestAssetsDir() / "test_fc_layer" / "mnist" / "0_fc1.weight.data", FC1_SIZE, MNIST_SIZE * MNIST_SIZE);
    auto bias = dataLoader.loadData(tools::getTestAssetsDir() / "test_fc_layer" / "mnist" / "0_fc1.bias.data", 1, FC1_SIZE);
    // transpose
    transpose(const_cast<raul::dtype*>(weight.first), FC1_SIZE, weight.second - weight.first);
    ASSERT_OK(set_tensor(graph, "fc1::Weights", weight.first, weight.second - weight.first));
    ASSERT_OK(set_tensor(graph, "fc1::Biases", bias.first, bias.second - bias.first));

    weight = dataLoader.loadData(tools::getTestAssetsDir() / "test_fc_layer" / "mnist" / "0_fc2.weight.data", FC2_SIZE, FC1_SIZE);
    bias = dataLoader.loadData(tools::getTestAssetsDir() / "test_fc_layer" / "mnist" / "0_fc2.bias.data", 1, FC2_SIZE);
    // transpose
    transpose(const_cast<raul::dtype*>(weight.first), FC2_SIZE, weight.second - weight.first);

    ASSERT_OK(set_tensor(graph, "fc2::Weights", weight.first, weight.second - weight.first));
    ASSERT_OK(set_tensor(graph, "fc2::Biases", bias.first, bias.second - bias.first));

    weight = dataLoader.loadData(tools::getTestAssetsDir() / "test_fc_layer" / "mnist" / "0_fc3.weight.data", NUM_CLASSES, FC2_SIZE);
    bias = dataLoader.loadData(tools::getTestAssetsDir() / "test_fc_layer" / "mnist" / "0_fc3.bias.data", 1, NUM_CLASSES);
    // transpose
    transpose(const_cast<raul::dtype*>(weight.first), NUM_CLASSES, weight.second - weight.first);
    ASSERT_OK(set_tensor(graph, "fc3::Weights", weight.first, weight.second - weight.first));
    ASSERT_OK(set_tensor(graph, "fc3::Biases", bias.first, bias.second - bias.first));

    Optimizer_t* sgd_optimizer = NULL;
    ASSERT_OK(create_sgd_optimizer(&sgd_optimizer, LEARNING_RATE));

    auto input_data = std::make_unique<raul::dtype[]>(MNIST_SIZE * MNIST_SIZE * MNIST_DEPTH * BATCH_SIZE);

    // read data
    raul::MNIST mnist;
    ASSERT_EQ(mnist.loadingData(tools::getTestAssetsDir() / "MNIST"), true);

    const size_t stepsAmountTrain = mnist.getTrainImageAmount() / BATCH_SIZE;
    raul::Tensor& idealLosses = dataLoader.createTensor(stepsAmountTrain / 100);
    raul::DataLoader::readArrayFromTextFile(tools::getTestAssetsDir() / "test_fc_layer" / "mnist" / "loss.data", idealLosses, 1, idealLosses.size());

    auto trainImages = mnist.getTrainImages();
    auto trainLabels = mnist.getTrainLabels();
    auto& encodedTrainLabels = dataLoader.buildOneHotVector(trainLabels, NUM_CLASSES);
    raul::Tensor& label_data = dataLoader.createTensor(NUM_CLASSES * BATCH_SIZE);

    size_t totalBatchSize = MNIST_SIZE * MNIST_SIZE * MNIST_DEPTH * BATCH_SIZE;
    raul::dtype totalTime = 0;

    for (size_t q = 0, idealLossIndex = 0; q < stepsAmountTrain; ++q)
    {
        std::transform(trainImages.begin() + q * totalBatchSize, trainImages.begin() + (q + 1) * totalBatchSize, input_data.get(), [](auto _n) { return _n / 255.f; });
        std::copy(encodedTrainLabels.begin() + q * BATCH_SIZE * NUM_CLASSES, encodedTrainLabels.begin() + (q + 1) * BATCH_SIZE * NUM_CLASSES, label_data.begin());

        ASSERT_OK(set_tensor(graph, "data", input_data.get(), totalBatchSize));
        ASSERT_OK(set_tensor(graph, "labels", &label_data[0], NUM_CLASSES * BATCH_SIZE));

        raul::dtype testLoss = 0.f;
        auto timeStart = std::chrono::steady_clock::now();
        ASSERT_OK(train_single_pass(graph, sgd_optimizer, &loss_name, 1u, &testLoss));
        totalTime += static_cast<raul::dtype>(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - timeStart).count());
        if (q % 100 == 0)
        {
            CHECK_NEAR(testLoss, idealLosses[idealLossIndex++], EPSILON_LOSS);
            printf("iteration = %d, loss = %f\n", static_cast<uint32_t>(q), testLoss);
        }
    }

    printf("Time taken = %.3fs \n", totalTime / 1000.f);
    ASSERT_OK(delete_graph(graph));
}

TEST(TestBindings, TrainingNINCifar)
{
    PROFILE_TEST
    const raul::dtype LEARNING_RATE = 0.0001f;
    const size_t BATCH_SIZE = 32;
    [[maybe_unused]] const raul::dtype EPSILON_ACCURACY = 1e-1f;

    const size_t NUM_CLASSES = 10;
    [[maybe_unused]] const raul::dtype acc1 = 80.20f;
    [[maybe_unused]] const raul::dtype acc2 = 85.36f;
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

    Graph_Description_t* desc = NULL;

    ASSERT_OK(create_graph_description_eager(&desc));

    {
        const char* tensors[] = { "data", "labels" };
        ASSERT_OK(add_data_layer_with_labels(desc, "data", tensors, 2, IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE, NUM_CLASSES));
    }

    ASSERT_OK(
        add_convolution_layer(desc, "conv1", "data", "conv1", CONVOLUTION_2D_LAYER, CONV1_KERNEL_SIZE, CONV1_KERNEL_SIZE, CONV1_FILTERS, CONV1_STRIDE, CONV1_STRIDE, CONV1_PADDING, CONV1_PADDING));
    ASSERT_OK(add_activation_layer(desc, "relu1", "conv1", "relu1", RELU_ACTIVATION));
    ASSERT_OK(add_convolution_layer(desc, "conv2", "relu1", "conv2", CONVOLUTION_2D_LAYER, CONV2_KERNEL_SIZE, CONV2_KERNEL_SIZE, CONV2_FILTERS, 1, 1, 0, 0));
    ASSERT_OK(add_activation_layer(desc, "relu2", "conv2", "relu2", RELU_ACTIVATION));
    ASSERT_OK(add_convolution_layer(desc, "conv3", "relu2", "conv3", CONVOLUTION_2D_LAYER, CONV3_KERNEL_SIZE, CONV3_KERNEL_SIZE, CONV3_FILTERS, 1, 1, 0, 0));
    ASSERT_OK(add_activation_layer(desc, "relu3", "conv3", "relu3", RELU_ACTIVATION));
    ASSERT_OK(add_pooling2d_layer(desc, "mp", "relu3", "mp", MAX_POOLING_2D_LAYER, MAXPOOL_KERNEL, MAXPOOL_KERNEL, MAXPOOL_STRIDE, MAXPOOL_STRIDE, MAXPOOL_PADDING, MAXPOOL_PADDING));
    ASSERT_OK(add_dropout_layer(desc, "drop1", "mp", "drop1", 0.5f));

    ASSERT_OK(
        add_convolution_layer(desc, "conv4", "drop1", "conv4", CONVOLUTION_2D_LAYER, CONV4_KERNEL_SIZE, CONV4_KERNEL_SIZE, CONV4_FILTERS, CONV4_STRIDE, CONV4_STRIDE, CONV4_PADDING, CONV4_PADDING));
    ASSERT_OK(add_activation_layer(desc, "relu4", "conv4", "relu4", RELU_ACTIVATION));
    ASSERT_OK(add_convolution_layer(desc, "conv5", "relu4", "conv5", CONVOLUTION_2D_LAYER, CONV5_KERNEL_SIZE, CONV5_KERNEL_SIZE, CONV5_FILTERS, 1, 1, 0, 0));
    ASSERT_OK(add_activation_layer(desc, "relu5", "conv5", "relu5", RELU_ACTIVATION));
    ASSERT_OK(add_convolution_layer(desc, "conv6", "relu5", "conv6", CONVOLUTION_2D_LAYER, CONV6_KERNEL_SIZE, CONV6_KERNEL_SIZE, CONV6_FILTERS, 1, 1, 0, 0));
    ASSERT_OK(add_activation_layer(desc, "relu6", "conv6", "relu6", RELU_ACTIVATION));
    ASSERT_OK(add_pooling2d_layer(desc, "avg1", "relu6", "avg1", AVERAGE_POOLING_2D_LAYER, AVGPOOL1_KERNEL, AVGPOOL1_KERNEL, AVGPOOL1_STRIDE, AVGPOOL1_STRIDE, AVGPOOL1_PADDING, AVGPOOL1_PADDING));
    ASSERT_OK(add_dropout_layer(desc, "drop2", "avg1", "drop2", 0.5f));

    ASSERT_OK(
        add_convolution_layer(desc, "conv7", "drop2", "conv7", CONVOLUTION_2D_LAYER, CONV7_KERNEL_SIZE, CONV7_KERNEL_SIZE, CONV7_FILTERS, CONV7_STRIDE, CONV7_STRIDE, CONV7_PADDING, CONV7_PADDING));
    ASSERT_OK(add_activation_layer(desc, "relu7", "conv7", "relu7", RELU_ACTIVATION));
    ASSERT_OK(add_convolution_layer(desc, "conv8", "relu7", "conv8", CONVOLUTION_2D_LAYER, CONV8_KERNEL_SIZE, CONV8_KERNEL_SIZE, CONV8_FILTERS, 1, 1, 0, 0));
    ASSERT_OK(add_activation_layer(desc, "relu8", "conv8", "relu8", RELU_ACTIVATION));
    ASSERT_OK(add_convolution_layer(desc, "conv9", "relu8", "conv9", CONVOLUTION_2D_LAYER, CONV9_KERNEL_SIZE, CONV9_KERNEL_SIZE, CONV9_FILTERS, 1, 1, 0, 0));
    ASSERT_OK(add_activation_layer(desc, "relu9", "conv9", "relu9", RELU_ACTIVATION));
    ASSERT_OK(add_pooling2d_layer(desc, "avg2", "relu9", "avg2", AVERAGE_POOLING_2D_LAYER, AVGPOOL2_KERNEL, AVGPOOL2_KERNEL, AVGPOOL2_STRIDE, AVGPOOL2_STRIDE, 0, 0));

    ASSERT_OK(add_activation_layer(desc, "softmax", "avg2", "softmax", LOG_SOFTMAX_ACTIVATION));
    const char* loss_name = "loss";
    {
        const char* tensorsIn[] = { "softmax", "labels" };
        ASSERT_OK(add_loss_layer(desc, "loss", tensorsIn, loss_name, NLL_LOSS, 2, LOSS_REDUCTION_SUM));
    }

    Graph_t* graph = NULL;
    ASSERT_OK(create_graph(&desc, &graph, BATCH_SIZE));

    ASSERT_NE(graph, nullptr);

    auto load_conv_params = [&graph](const char* name, int index, size_t count, int kernel_size, int depth) {
        std::string nameWeight = std::string(name) + "::Weights";
        std::string nameBias = std::string(name) + "::Biases";

        raul::DataLoader dataLoader;

        const std::string cifarDir = "cifarNINTrained_BS128_Canonical";
        auto weight = dataLoader.loadFilters(
            (tools::getTestAssetsDir() / "test_cnn_layer" / cifarDir / ("0_classifier." + Conversions::toString(index) + ".weight_")).string(), 0, ".data", kernel_size, kernel_size, depth, count);
        auto bias = dataLoader.loadData(tools::getTestAssetsDir() / "test_cnn_layer" / cifarDir / ("0_classifier." + Conversions::toString(index) + ".bias.data"), 1, count);

        ASSERT_OK(set_tensor(graph, nameWeight.c_str(), weight.first, weight.second - weight.first));
        ASSERT_OK(set_tensor(graph, nameBias.c_str(), bias.first, bias.second - bias.first));
    };

    load_conv_params("conv1", 0, CONV1_FILTERS, CONV1_KERNEL_SIZE, IMAGE_CHANNELS);
    load_conv_params("conv2", 2, CONV2_FILTERS, CONV2_KERNEL_SIZE, CONV1_FILTERS);
    load_conv_params("conv3", 4, CONV3_FILTERS, CONV3_KERNEL_SIZE, CONV2_FILTERS);
    load_conv_params("conv4", 8, CONV4_FILTERS, CONV4_KERNEL_SIZE, CONV3_FILTERS);
    load_conv_params("conv5", 10, CONV5_FILTERS, CONV5_KERNEL_SIZE, CONV4_FILTERS);
    load_conv_params("conv6", 12, CONV6_FILTERS, CONV6_KERNEL_SIZE, CONV5_FILTERS);
    load_conv_params("conv7", 16, CONV7_FILTERS, CONV7_KERNEL_SIZE, CONV6_FILTERS);
    load_conv_params("conv8", 18, CONV8_FILTERS, CONV8_KERNEL_SIZE, CONV7_FILTERS);
    load_conv_params("conv9", 20, CONV9_FILTERS, CONV9_KERNEL_SIZE, CONV8_FILTERS);

    Optimizer_t* sgd_optimizer = NULL;
    ASSERT_OK(create_sgd_optimizer(&sgd_optimizer, LEARNING_RATE));

    auto input_data = std::make_unique<raul::dtype[]>(IMAGE_SIZE * IMAGE_SIZE * IMAGE_CHANNELS * BATCH_SIZE);

    // read data
    raul::CIFAR10 cifar;
    ASSERT_EQ(cifar.loadingData(tools::getTestAssetsDir() / "CIFAR"), true);

    auto testImages = cifar.getTestImages();
    auto testLabels = cifar.getTestLabels();
    const size_t stepsAmountTest = cifar.getTestImageAmount() / BATCH_SIZE;
    auto totalBatchSize = IMAGE_SIZE * IMAGE_SIZE * IMAGE_CHANNELS * BATCH_SIZE;
    size_t labels[BATCH_SIZE];
    raul::dtype testAcc = 0;
    raul::dtype testTime = 0;
    printf("Begin testing\n");
    for (size_t q = 0; q < stepsAmountTest; ++q)
    {
        raul::dtype accuracy = 0;
        std::transform(testImages.begin() + q * totalBatchSize, testImages.begin() + (q + 1) * totalBatchSize, input_data.get(), [](auto _n) { return _n / 255.f; });
        std::copy_n(testLabels.begin() + q * BATCH_SIZE, BATCH_SIZE, labels);
        ASSERT_OK(set_tensor(graph, "data", input_data.get(), totalBatchSize));
        auto timeStart = std::chrono::steady_clock::now();
        ASSERT_OK(test_network(graph, "softmax", labels, &accuracy));
        testTime += static_cast<raul::dtype>(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - timeStart).count());
        testAcc += accuracy;
    }
    testAcc /= static_cast<raul::dtype>(stepsAmountTest);
    printf("Testing taken = %.3fs \n", testTime / 1000.f);
    printf("Test accuracy = %.2f\n", testAcc * 100);

    const size_t stepsAmountTrain = cifar.getTrainImageAmount() / BATCH_SIZE;

    raul::DataLoader dataLoader;

    auto trainImages = cifar.getTrainImages();
    auto trainLabels = cifar.getTrainLabels();
    auto& encodedTrainLabels = dataLoader.buildOneHotVector(trainLabels, NUM_CLASSES);

    raul::Tensor& label_data = dataLoader.createTensor(NUM_CLASSES * BATCH_SIZE);

    raul::dtype totalTime = 0;
    for (size_t epoch = 1; epoch <= 1; ++epoch)
    {
        printf("Epoch = %zu\n", epoch);

        raul::dtype averageLoss = 0;
        raul::dtype epochTime = 0;
        for (size_t q = 0; q < stepsAmountTrain; ++q)
        {
            std::transform(trainImages.begin() + q * totalBatchSize, trainImages.begin() + (q + 1) * totalBatchSize, input_data.get(), [](auto _n) { return _n / 255.f; });
            std::copy(encodedTrainLabels.begin() + q * BATCH_SIZE * NUM_CLASSES, encodedTrainLabels.begin() + (q + 1) * BATCH_SIZE * NUM_CLASSES, label_data.begin());

            ASSERT_OK(set_tensor(graph, "data", input_data.get(), totalBatchSize));
            ASSERT_OK(set_tensor(graph, "labels", &label_data[0], NUM_CLASSES * BATCH_SIZE));

            raul::dtype testLoss = 0.f;
            auto timeStart = std::chrono::steady_clock::now();
            ASSERT_OK(train_single_pass(graph, sgd_optimizer, &loss_name, 1u, &testLoss));
            epochTime += static_cast<raul::dtype>(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - timeStart).count());
            averageLoss += testLoss;
            if (q % (stepsAmountTrain / 3) == 0)
            {
                printf("iteration = %zu / %zu, loss = %f\n", q, stepsAmountTrain, testLoss);
            }
        }
        printf("Average loss = %f\n", averageLoss / static_cast<float>(stepsAmountTrain));
        printf("Epoch Training taken = %.3fs \n", epochTime / 1000.f);
        totalTime += epochTime;

        testAcc = 0;
        for (size_t q = 0; q < stepsAmountTest; ++q)
        {
            raul::dtype accuracy = 0;
            std::transform(testImages.begin() + q * totalBatchSize, testImages.begin() + (q + 1) * totalBatchSize, input_data.get(), [](auto _n) { return _n / 255.f; });
            std::copy_n(testLabels.begin() + q * BATCH_SIZE, BATCH_SIZE, labels);
            ASSERT_OK(set_tensor(graph, "data", input_data.get(), totalBatchSize));
            ASSERT_OK(test_network(graph, "softmax", labels, &accuracy));
            testAcc += accuracy;
        }
        testAcc /= static_cast<raul::dtype>(stepsAmountTest);
        printf("Test accuracy = %.2f\n", testAcc * 100);
    }

    printf("Time taken = %.3fs \n", totalTime / 1000.f);
    ASSERT_OK(delete_graph(graph));
}

TEST(TestBindings, SimpleTrainingUnit)
{
    PROFILE_TEST
    const raul::dtype LEARNING_RATE = 0.01f;
    const size_t BATCH_SIZE = 3;

    const size_t NUM_CLASSES = 2;
    const size_t IMAGE_WIDTH = 2;
    const size_t IMAGE_HEIGHT = 1;
    const size_t IMAGE_CHANNELS = 1;

    const size_t FC_SIZE = 2;

    Graph_Description_t* desc = NULL;

    ASSERT_OK(create_graph_description_eager(&desc));

    {
        const char* tensors[] = { "data", "labels" };
        ASSERT_OK(add_data_layer_with_labels(desc, "data", tensors, 2, IMAGE_CHANNELS, IMAGE_WIDTH, IMAGE_HEIGHT, NUM_CLASSES));
    }

    ASSERT_OK(add_reshape_layer(desc, "reshape", "data", "datar", 1, 1, -1));
    ASSERT_OK(add_linear_layer(desc, "fc", "datar", "fc", FC_SIZE, true));

    ASSERT_OK(add_activation_layer(desc, "softmax", "fc", "softmax", SOFTMAX_ACTIVATION));
    const char* loss_name = "loss";
    {
        const char* tensorsIn[] = { "softmax", "labels" };
        ASSERT_OK(add_loss_layer(desc, "loss", tensorsIn, loss_name, CROSS_ENTROPY_LOSS, 2, LOSS_REDUCTION_MEAN));
    }

    raul::dtype input_data[] = { 1, 0, 0, 1, 1, 1 };
    size_t labels[] = { 0, 1, 0 };
    raul::dtype label_data[] = { 1, 0, 0, 1, 1, 0 };

    Graph_t* graph = NULL;
    ASSERT_OK(create_graph(&desc, &graph, BATCH_SIZE));
    ASSERT_NE(graph, nullptr);

    Optimizer_t* sgd_optimizer = NULL;
    ASSERT_OK(create_sgd_optimizer(&sgd_optimizer, LEARNING_RATE));

    auto totalBatchSize = IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_CHANNELS * BATCH_SIZE;
    ASSERT_OK(set_tensor(graph, "data", input_data, totalBatchSize));
    ASSERT_OK(set_tensor(graph, "labels", label_data, NUM_CLASSES * BATCH_SIZE));

    raul::dtype accuracy = 0;
    ASSERT_OK(test_network(graph, "softmax", labels, &accuracy));
    printf("Accuracy = %.3f\n", accuracy * 100);

    size_t nepoch = 2000;
    size_t step = 100;
    for (size_t epoch = 1; epoch <= nepoch; ++epoch)
    {
        raul::dtype testLoss = 0.f;
        auto timeStart = std::chrono::steady_clock::now();
        ASSERT_OK(train_single_pass(graph, sgd_optimizer, &loss_name, 1u, &testLoss));
        auto epochTime = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - timeStart).count();

        if (epoch == 1 || epoch % step == 0)
        {
            printf("Epoch = %zu, Average loss = %f\n", epoch, testLoss);
            printf("Epoch Training taken = %.3fs \n", static_cast<float>(epochTime) / 1000.f);
        }
    }

    ASSERT_OK(test_network(graph, "softmax", labels, &accuracy));
    printf("Accuracy = %.3f\n", accuracy * 100);

    ASSERT_OK(delete_optimizer(sgd_optimizer));
    ASSERT_OK(delete_graph(graph));
}

TEST(TestBindings, ElementWiseMulForwardUnit)
{
    PROFILE_TEST
    const auto eps = 1e-6_dt;
    const auto tensor_size = 3U;

    Graph_Description_t* desc = NULL;

    ASSERT_OK(create_graph_description_eager(&desc));

    {
        const char* tensors[] = { "data", "labels" };
        const size_t tensor_amount = 2U;
        ASSERT_OK(add_data_layer_with_labels(desc, "data", tensors, tensor_amount, tensor_size, 1, 1, 1));
    }
    ASSERT_OK(add_activation_layer(desc, "relu", "data", "relu", RELU_ACTIVATION));
    {
        const char* tensors[] = { "data", "relu" };
        const size_t tensor_amount = 2U;
        ASSERT_OK(add_elementwise_layer(desc, "mul", tensors, "mul", ELEMENTWISE_MUL_LAYER, tensor_amount));
    }

    Graph_t* graph = NULL;
    ASSERT_OK(create_graph(&desc, &graph, 1U));
    ASSERT_NE(graph, nullptr);

    FLOAT_TYPE input_data[] = { 1.0, 2.0, 3.0 };

    ASSERT_OK(set_tensor(graph, "data", input_data, tensor_size));
    ASSERT_OK(network_forward(graph, true));

    FLOAT_TYPE output_data[tensor_size];
    size_t output_size = 0;
    ASSERT_OK(get_tensor(graph, "mul", nullptr, &output_size));
    ASSERT_OK(get_tensor(graph, "mul", output_data, &output_size));

    ASSERT_EQ(output_size, tensor_size);

    for (size_t i = 0; i < output_size; ++i)
    {
        EXPECT_NEAR(output_data[i], input_data[i] * input_data[i], eps);
    }
}

TEST(TestBindings, HSigmoidForwardUnit)
{
    PROFILE_TEST
    const auto eps = 1e-6_dt;
    const auto tensor_size = 7U;

    Graph_Description_t* desc = NULL;

    ASSERT_OK(create_graph_description_eager(&desc));

    {
        const char* tensors[] = { "data", "labels" };
        const size_t tensor_amount = 2U;
        ASSERT_OK(add_data_layer_with_labels(desc, "data", tensors, tensor_amount, tensor_size, 1, 1, 1));
    }
    ASSERT_OK(add_activation_layer(desc, "hsigm", "data", "hsigm", HSIGMOID_ACTIVATION));

    Graph_t* graph = NULL;
    ASSERT_OK(create_graph(&desc, &graph, 1U));
    ASSERT_NE(graph, nullptr);

    raul::dtype input_data[] = { -4.0, -3.0, -1.0, 0.0, 1.0, 3.0, 4.0 };

    ASSERT_OK(set_tensor(graph, "data", input_data, tensor_size));
    ASSERT_OK(network_forward(graph, true));

    raul::dtype output_data[tensor_size];
    size_t output_size = 0;
    ASSERT_OK(get_tensor(graph, "hsigm", nullptr, &output_size));
    ASSERT_OK(get_tensor(graph, "hsigm", output_data, &output_size));

    ASSERT_EQ(output_size, tensor_size);

    auto golden_hsigmoid = [](const raul::dtype x) { return std::min(std::max(x + 3.0_dt, .0_dt), 6.0_dt) / 6.0_dt; };

    for (size_t i = 0; i < output_size; ++i)
    {
        EXPECT_NEAR(output_data[i], golden_hsigmoid(input_data[i]), eps);
    }
}

TEST(TestBindings, LSTMCellForwardUnit)
{
    PROFILE_TEST
    // Test parameters
    const auto eps_rel = 1e-5_dt;
    const auto input_size = 4U;
    const auto hidden_size = 3U;
    const auto batch_size = 2U;
    const auto zonenout_prob = 0.0_dt;

    Graph_Description_t* desc = NULL;
    Graph_t* graph = NULL;

    const char* tensors_in[] = { "in" };
    const char* tensors_state[] = { "hidden", "cell" };
    ASSERT_OK(create_graph_description_eager(&desc));
    ASSERT_OK(add_data_layer(desc, "tensors_in", tensors_in, 1, 1, 1, input_size));
    ASSERT_OK(add_data_layer(desc, "tensors_state", tensors_state, 2, 1, 1, hidden_size));
    ASSERT_OK(add_lstm_cell_layer(desc, "lstm_cell", "in", "hidden", "cell", "new_hidden", "new_cell", true, zonenout_prob, false, 0.0_dt));
    //    std::cout << std::string(get_last_error()) << std::endl;
    ASSERT_OK(create_graph(&desc, &graph, batch_size));
    ASSERT_NE(graph, nullptr);

    // Initialization
    const FLOAT_TYPE input_init[] = { 1.752833e-01_dt, -9.315211e-01_dt, -1.505490e+00_dt, -6.609825e-01_dt, 1.323202e+00_dt, 3.711430e-02_dt, -2.849093e-01_dt, -1.334417e-01_dt };
    const FLOAT_TYPE hidden_init[] = { 1.892910e+00_dt, 3.111044e+00_dt, -4.583958e-01_dt, -3.359881e-01_dt, -1.569986e+00_dt, 1.231500e+00_dt };
    const FLOAT_TYPE cell_init[] = { 1.394632e+00_dt, 1.171102e+00_dt, 4.335119e-01_dt, -1.734250e+00_dt, -1.336049e+00_dt, 8.870960e-01_dt };

    ASSERT_OK(set_tensor(graph, "in", input_init, sizeof(input_init) / sizeof(FLOAT_TYPE)));
    ASSERT_OK(set_tensor(graph, "hidden", hidden_init, sizeof(hidden_init) / sizeof(FLOAT_TYPE)));
    ASSERT_OK(set_tensor(graph, "cell", cell_init, sizeof(cell_init) / sizeof(FLOAT_TYPE)));

    Initializer_t* initializer;
    create_constant_initializer(&initializer, 1.0);

    size_t param_count;
    size_t max_size;
    ASSERT_OK(get_model_parameters(graph, false, NULL, &param_count, &max_size));

    char** parameters = (char**)malloc(param_count * sizeof(char*));
    for (size_t i = 0; i < param_count; ++i)
    {
        parameters[i] = (char*)malloc((max_size + 1U) * sizeof(char));
    }

    ASSERT_OK(get_model_parameters(graph, false, parameters, &param_count, &max_size));

    for (size_t i = 0; i < param_count; ++i)
    {
        initialize_tensor(graph, initializer, parameters[i]);
    }

    delete_initializer(initializer);

    for (size_t i = 0; i < param_count; ++i)
    {
        free(parameters[i]);
    }
    free(parameters);

    ASSERT_OK(network_forward(graph, true));

    const FLOAT_TYPE hidden_golden[] = { 9.557552e-01_dt, 9.459500e-01_dt, 8.612298e-01_dt, -5.386918e-01_dt, -2.835236e-01_dt, 8.465633e-01_dt };
    const FLOAT_TYPE cell_golden[] = { 2.330955e+00_dt, 2.113240e+00_dt, 1.394835e+00_dt, -6.845742e-01_dt, -3.237444e-01_dt, 1.690755e+00_dt };

    // Checks
    raul::dtype hidden_new_tensor[hidden_size * batch_size];
    size_t hidden_output_size = 0;
    ASSERT_OK(get_tensor(graph, "new_hidden", nullptr, &hidden_output_size));
    ASSERT_OK(get_tensor(graph, "new_hidden", hidden_new_tensor, &hidden_output_size));

    ASSERT_EQ(hidden_output_size, hidden_size * batch_size);

    raul::dtype new_cell_tensor[hidden_size * batch_size];
    size_t cell_output_size = 0;
    ASSERT_OK(get_tensor(graph, "new_cell", nullptr, &cell_output_size));
    ASSERT_OK(get_tensor(graph, "new_cell", new_cell_tensor, &cell_output_size));

    ASSERT_EQ(cell_output_size, hidden_size * batch_size);

    for (size_t i = 0; i < hidden_output_size; ++i)
    {
        const auto hidden_val = hidden_new_tensor[i];
        const auto golden_hidden_val = hidden_golden[i];
        ASSERT_TRUE(tools::expect_near_relative(hidden_val, golden_hidden_val, eps_rel)) << "at " << i << ", expected: " << golden_hidden_val << ", got: " << hidden_val;
    }

    for (size_t i = 0; i < cell_output_size; ++i)
    {
        const auto cell_val = new_cell_tensor[i];
        const auto golden_cell_val = cell_golden[i];
        ASSERT_TRUE(tools::expect_near_relative(cell_val, golden_cell_val, eps_rel)) << "at " << i << ", expected: " << golden_cell_val << ", got: " << cell_val;
    }
}

namespace
{
bool EndsWith(const char* str, const char* suffix)
{
    if (!str || !suffix) return false;
    size_t lenstr = strlen(str);
    size_t lensuffix = strlen(suffix);
    if (lensuffix > lenstr)
    {
        return false;
    }
    return strncmp(str + lenstr - lensuffix, suffix, lensuffix) == 0;
}
}

TEST(TestBindings, BertForSequenceClassificationAPIUnit)
{
    PROFILE_TEST
    constexpr size_t BATCH_SIZE = 1;

    float ATTENTION_DROPOUT = 0.0f;
    const char* ACTIVATION = "gelu";
    float HIDDEN_DROPOUT = 0.0f;
    unsigned int HIDDEN_SIZE = 384;
    unsigned int INTERMEDIATE_SIZE = 1536;
    unsigned int MAX_POSITION_EMBEDDINGS = 512;
    unsigned int NUM_ATTENTION_HEADS = 12;
    unsigned int NUM_HIDDEN_LAYERS = 4;
    unsigned int TYPE_VOCAB_SIZE = 2;
    unsigned int VOCAB_SIZE = 30522;

    constexpr raul::dtype ERROR_EPS = 1e-5_dt;

    constexpr size_t LENGTH = 64;
    constexpr size_t NUM_LABELS = 2;

    Graph_Description_t* desc = NULL;

    ASSERT_OK(create_graph_description_eager(&desc));

    {
        const char* tensors[] = { "input_ids", "token_type_ids", "attention_mask", "labels" };
        ASSERT_OK(add_data_layer_with_labels(desc, "input", tensors, 4U, 1U, 1U, LENGTH, NUM_LABELS));
    }

    {
        const char* tensors_out[] = { "hidden_state", "pooled_output" };
        ASSERT_OK(add_bert_model(desc,
                                 "bert",
                                 "input_ids",
                                 "token_type_ids",
                                 "attention_mask",
                                 tensors_out,
                                 2U,
                                 VOCAB_SIZE,
                                 TYPE_VOCAB_SIZE,
                                 NUM_HIDDEN_LAYERS,
                                 HIDDEN_SIZE,
                                 INTERMEDIATE_SIZE,
                                 NUM_ATTENTION_HEADS,
                                 MAX_POSITION_EMBEDDINGS,
                                 ACTIVATION,
                                 HIDDEN_DROPOUT,
                                 ATTENTION_DROPOUT));
    }

    ASSERT_OK(add_dropout_layer(desc, "dropout", "pooled_output", "pooled_output_do", HIDDEN_DROPOUT));
    ASSERT_OK(add_linear_layer(desc, "classifier", "pooled_output_do", "logits", NUM_LABELS, true));
    ASSERT_OK(add_activation_layer(desc, "softmax", "logits", "softmax", LOG_SOFTMAX_ACTIVATION));
    const char* loss_name = "loss";
    {
        const char* tensors[] = { "softmax", "labels" };
        ASSERT_OK(add_loss_layer(desc, "loss", tensors, loss_name, NLL_LOSS, 2U, LOSS_REDUCTION_BATCH_MEAN));
    }

    Graph_t* graph = NULL;
    ASSERT_OK(create_graph(&desc, &graph, BATCH_SIZE));
    ASSERT_NE(graph, nullptr);

    // data for first batch of size 1
    FLOAT_TYPE input_ids[LENGTH * BATCH_SIZE] = { 1.0100e+02_dt, 2.0090e+03_dt, 1.0050e+03_dt, 1.0550e+03_dt, 1.0370e+03_dt, 1.1951e+04_dt, 1.9980e+03_dt, 2.4110e+03_dt, 1.2473e+04_dt, 4.9900e+03_dt,
                                                  1.0120e+03_dt, 1.0200e+02_dt, 0.0000e+00_dt, 0.0000e+00_dt, 0.0000e+00_dt, 0.0000e+00_dt, 0.0000e+00_dt, 0.0000e+00_dt, 0.0000e+00_dt, 0.0000e+00_dt,
                                                  0.0000e+00_dt, 0.0000e+00_dt, 0.0000e+00_dt, 0.0000e+00_dt, 0.0000e+00_dt, 0.0000e+00_dt, 0.0000e+00_dt, 0.0000e+00_dt, 0.0000e+00_dt, 0.0000e+00_dt,
                                                  0.0000e+00_dt, 0.0000e+00_dt, 0.0000e+00_dt, 0.0000e+00_dt, 0.0000e+00_dt, 0.0000e+00_dt, 0.0000e+00_dt, 0.0000e+00_dt, 0.0000e+00_dt, 0.0000e+00_dt,
                                                  0.0000e+00_dt, 0.0000e+00_dt, 0.0000e+00_dt, 0.0000e+00_dt, 0.0000e+00_dt, 0.0000e+00_dt, 0.0000e+00_dt, 0.0000e+00_dt, 0.0000e+00_dt, 0.0000e+00_dt,
                                                  0.0000e+00_dt, 0.0000e+00_dt, 0.0000e+00_dt, 0.0000e+00_dt, 0.0000e+00_dt, 0.0000e+00_dt, 0.0000e+00_dt, 0.0000e+00_dt, 0.0000e+00_dt, 0.0000e+00_dt,
                                                  0.0000e+00_dt, 0.0000e+00_dt, 0.0000e+00_dt, 0.0000e+00_dt };
    FLOAT_TYPE token_type_ids[LENGTH * BATCH_SIZE] = { 0_dt };
    FLOAT_TYPE attention_mask[LENGTH * BATCH_SIZE] = { 1_dt, 1_dt, 1_dt, 1_dt, 1_dt, 1_dt, 1_dt, 1_dt, 1_dt, 1_dt, 1_dt, 1_dt, 0_dt, 0_dt, 0_dt, 0_dt, 0_dt, 0_dt, 0_dt, 0_dt, 0_dt, 0_dt,
                                                       0_dt, 0_dt, 0_dt, 0_dt, 0_dt, 0_dt, 0_dt, 0_dt, 0_dt, 0_dt, 0_dt, 0_dt, 0_dt, 0_dt, 0_dt, 0_dt, 0_dt, 0_dt, 0_dt, 0_dt, 0_dt, 0_dt,
                                                       0_dt, 0_dt, 0_dt, 0_dt, 0_dt, 0_dt, 0_dt, 0_dt, 0_dt, 0_dt, 0_dt, 0_dt, 0_dt, 0_dt, 0_dt, 0_dt, 0_dt, 0_dt, 0_dt, 0_dt };
    FLOAT_TYPE labels[NUM_LABELS * BATCH_SIZE] = { 0_dt, 1_dt };

    EXPECT_OK(set_tensor(graph, "input_ids", input_ids, LENGTH * BATCH_SIZE));
    EXPECT_OK(set_tensor(graph, "token_type_ids", token_type_ids, LENGTH * BATCH_SIZE));
    EXPECT_OK(set_tensor(graph, "attention_mask", attention_mask, LENGTH * BATCH_SIZE));
    EXPECT_OK(set_tensor(graph, "labels", labels, NUM_LABELS * BATCH_SIZE));

    size_t params_count = 0;
    size_t max_param_length = 0;
    ASSERT_OK(get_model_parameters(graph, false, NULL, &params_count, &max_param_length));
    EXPECT_EQ(params_count, 73u);
    char** params = (char**)malloc(params_count * sizeof(char*));
    for (size_t i = 0; i < params_count; ++i)
    {
        params[i] = (char*)malloc(max_param_length * sizeof(char));
    }
    ASSERT_OK(get_model_parameters(graph, false, params, &params_count, &max_param_length));

    for (size_t i = 0; i < params_count; ++i)
    {
        if (EndsWith(params[i], "Biases"))
            EXPECT_OK(fill_tensor(graph, params[i], 0._dt));
        else
            EXPECT_OK(fill_tensor(graph, params[i], 1._dt));
    }

    Optimizer_t* sgd_optimizer = NULL;
    ASSERT_OK(create_sgd_optimizer(&sgd_optimizer, FLOAT_TYPE(0.01)));

    FLOAT_TYPE testLoss = FLOAT_TYPE(0.);

    ASSERT_OK(train_single_pass(graph, sgd_optimizer, &loss_name, 1u, &testLoss));
    EXPECT_NEAR(testLoss, 0.693147_dt, ERROR_EPS);

    for (size_t i = 0; i < params_count; ++i)
    {
        free(params[i]);
    }
    free(params);
    ASSERT_OK(delete_graph(graph));
}

struct PaddingLayerTests : public testing::TestWithParam<std::tuple<std::string, raul::PaddingLayerParams::FillingMode, bool, raul::dtype, raul::dtype>>
{
    std::filesystem::path datasetMnistPath = tools::getTestAssetsDir() / "MNIST";
    const size_t batchSize = 50;
    const size_t lossCheckStep = 100;
    raul::dtype learningRate = 0.01_dt;
    raul::dtype accuracyEpsilon = 1e-2_dt;
    raul::dtype lossEpsilon = 1e-6_dt;
    size_t mnistImgWidth = 28;
    size_t mnistImgHeight = 28;
    size_t mnistImgChannels = 1;
    size_t sizeOfOneDataBatch = mnistImgWidth * mnistImgHeight * mnistImgChannels * batchSize;
    size_t numberOfClasses = 10;
    size_t convKernelSize = 5;
    size_t convOutChannels = 1;
    size_t convStride = 1;
    size_t convPadding = 0;
    uint32_t commonPadding = 20;
    uint32_t tp = 20; // top padding
    uint32_t bp = 15; // bottom padding
    uint32_t lp = 10; // left padding
    uint32_t rp = 5;  // right padding
};

TEST_P(PaddingLayerTests, TrainingOfSimpleNetworkCreatedUsingC_API)
{
    PROFILE_TEST
    auto convWeightsPathPrefix = tools::getTestAssetsDir() / "test_cnn_layer" / std::get<0>(GetParam()) / "0_conv1.weight_";
    auto convBiases = tools::getTestAssetsDir() / "test_cnn_layer" / std::get<0>(GetParam()) / "0_conv1.bias.data";
    auto fcWeights = tools::getTestAssetsDir() / "test_cnn_layer" / std::get<0>(GetParam()) / "0_fc1.weight.data";
    auto fcBiases = tools::getTestAssetsDir() / "test_cnn_layer" / std::get<0>(GetParam()) / "0_fc1.bias.data";
    auto losses = tools::getTestAssetsDir() / "test_cnn_layer" / std::get<0>(GetParam()) / "loss.data";
    bool paddingShouldBeSymmetric = std::get<2>(GetParam());
    FILLING_MODE fm = CONSTANT;
    switch (std::get<1>(GetParam()))
    {
        case raul::PaddingLayerParams::USE_FILLING_VALUE:
        {
            fm = CONSTANT;
            break;
        }
        case raul::PaddingLayerParams::REFLECTION:
        {
            fm = REFLECTION;
            break;
        }
        case raul::PaddingLayerParams::REPLICATION:
        {
            fm = REPLICATION;
            break;
        }
        default:
            ASSERT_TRUE(false);
    }

    Graph_Description_t* desc = NULL;
    ASSERT_EQ(create_graph_description_eager(&desc), STATUS_OK);
    ASSERT_TRUE(desc);

    const char* data_layer_outputs[] = { "data", "labels" };
    ASSERT_EQ(add_data_layer_with_labels(desc, "data", data_layer_outputs, 2, mnistImgChannels, mnistImgHeight, mnistImgWidth, numberOfClasses), STATUS_OK);
    ASSERT_EQ(add_convolution_layer(desc, "conv1", "data", "conv1", CONVOLUTION_2D_LAYER, convKernelSize, convKernelSize, convOutChannels, convStride, convStride, convPadding, convPadding),
              STATUS_OK);
    if (paddingShouldBeSymmetric)
    {
        ASSERT_EQ(add_padding_layer(desc, "pad1", "conv1", "pad1", commonPadding, commonPadding, commonPadding, commonPadding, fm), STATUS_OK);
        ASSERT_EQ(add_padding_layer(desc, "pad2", "pad1", "pad2", commonPadding, commonPadding, commonPadding, commonPadding, fm), STATUS_OK);
        ASSERT_EQ(add_reshape_layer(desc, "reshape1", "pad2", "reshape1", 1, -1, 10816), STATUS_OK);
    }
    else
    {
        ASSERT_EQ(add_padding_layer(desc, "pad1", "conv1", "pad1", tp, bp, lp, rp, fm), STATUS_OK);
        ASSERT_EQ(add_padding_layer(desc, "pad2", "pad1", "pad2", tp, bp, lp, rp, fm), STATUS_OK);
        ASSERT_EQ(add_reshape_layer(desc, "reshape1", "pad2", "reshape1", 1, -1, 5076), STATUS_OK);
    }
    ASSERT_EQ(add_linear_layer(desc, "fc1", "reshape1", "fc1", numberOfClasses, true), STATUS_OK);
    ASSERT_EQ(add_activation_layer(desc, "softmax", "fc1", "softmax", SOFTMAX_ACTIVATION), STATUS_OK);
    const char* loss_layer_inputs[] = { "softmax", "labels" };
    const char* loss_name = "loss";
    ASSERT_EQ(add_loss_layer(desc, "loss", loss_layer_inputs, loss_name, CROSS_ENTROPY_LOSS, 2, LOSS_REDUCTION_BATCH_MEAN), STATUS_OK);

    Graph_t* graph = NULL;
    EXPECT_EQ(create_graph(&desc, &graph, batchSize), STATUS_OK);
    ASSERT_TRUE(graph);

    raul::DataLoader dataLoader;
    auto conv1_weights = dataLoader.loadFilters(convWeightsPathPrefix.string(), 0, ".data", convKernelSize, convKernelSize, 1, convOutChannels);
    set_tensor(graph, "conv1::Weights", conv1_weights.first, conv1_weights.second - conv1_weights.first);
    auto conv1_biases = dataLoader.loadData(convBiases, 1, convOutChannels);
    set_tensor(graph, "conv1::Biases", conv1_biases.first, conv1_biases.second - conv1_biases.first);
    auto fc1_weights = paddingShouldBeSymmetric ? dataLoader.loadData(fcWeights, numberOfClasses, 10816) : dataLoader.loadData(fcWeights, numberOfClasses, 5076);
    set_tensor(graph, "fc1::Weights", fc1_weights.first, fc1_weights.second - fc1_weights.first);
    auto fc1_biases = dataLoader.loadData(fcBiases, 1, numberOfClasses);
    set_tensor(graph, "fc1::Biases", fc1_biases.first, fc1_biases.second - fc1_biases.first);

    raul::MNIST mnist;
    ASSERT_EQ(mnist.loadingData(datasetMnistPath), true);

    const size_t stepsAmountTrain = mnist.getTrainImageAmount() / batchSize;
    raul::Tensor& idealLosses = dataLoader.createTensor(stepsAmountTrain / lossCheckStep);
    raul::DataLoader::readArrayFromTextFile(losses, idealLosses, 1, idealLosses.size());

    auto trainImages = mnist.getTrainImages();
    auto trainLabels = mnist.getTrainLabels();
    auto& encodedTrainLabels = dataLoader.buildOneHotVector(trainLabels, numberOfClasses);
    raul::Tensor& label_data = dataLoader.createTensor(numberOfClasses * batchSize);
    auto input_data = std::make_unique<raul::dtype[]>(sizeOfOneDataBatch);
    raul::dtype testLoss = 0._dt;
    Optimizer_t* sgd_optimizer = NULL;
    ASSERT_EQ(create_sgd_optimizer(&sgd_optimizer, learningRate), STATUS_OK);
    for (size_t q = 0; q <= lossCheckStep; ++q)
    {
        std::transform(trainImages.begin() + q * sizeOfOneDataBatch, trainImages.begin() + (q + 1) * sizeOfOneDataBatch, input_data.get(), [](auto _n) { return _n / 255.f; });
        std::copy(encodedTrainLabels.begin() + q * batchSize * numberOfClasses, encodedTrainLabels.begin() + (q + 1) * batchSize * numberOfClasses, label_data.begin());

        ASSERT_EQ(set_tensor(graph, "data", input_data.get(), sizeOfOneDataBatch), STATUS_OK);
        ASSERT_EQ(set_tensor(graph, "labels", &label_data[0], numberOfClasses * batchSize), STATUS_OK);
        ASSERT_EQ(train_single_pass(graph, sgd_optimizer, &loss_name, 1u, &testLoss), STATUS_OK);
    }
    EXPECT_NEAR(testLoss, idealLosses[1], lossEpsilon);

    ASSERT_EQ(delete_optimizer(sgd_optimizer), STATUS_OK);
    ASSERT_EQ(delete_graph(graph), STATUS_OK);
}

INSTANTIATE_TEST_SUITE_P(TestBindings,
                         PaddingLayerTests,
                         testing::Values(std::make_tuple("const_sym_pad", raul::PaddingLayerParams::USE_FILLING_VALUE, true, 5.01_dt, 89.90_dt),
                                         std::make_tuple("ref_sym_pad", raul::PaddingLayerParams::REFLECTION, true, 6.24_dt, 89.89_dt),
                                         std::make_tuple("rep_sym_pad", raul::PaddingLayerParams::REPLICATION, true, 9.20_dt, 90.13_dt),
                                         std::make_tuple("const_asym_pad", raul::PaddingLayerParams::USE_FILLING_VALUE, false, 7.28_dt, 89.88_dt),
                                         std::make_tuple("ref_asym_pad", raul::PaddingLayerParams::REFLECTION, false, 7.81_dt, 89.80_dt),
                                         std::make_tuple("rep_asym_pad", raul::PaddingLayerParams::REPLICATION, false, 7.17_dt, 90.57_dt)));

} // UT namespace
