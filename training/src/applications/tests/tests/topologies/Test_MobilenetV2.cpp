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
#include <training/api/API.h>
#include <training/network/Layers.h>

#include <training/layers/basic/trainable/Batchnorm.h>

#include <training/common/Test.h>
#include <training/common/Train.h>
#include <training/network/Workflow.h>
#include <training/optimizers/SGD.h>

#include <cstdio>

namespace
{
const size_t reproduceLayers = 16;

const size_t filterSizes[reproduceLayers][3] = { { 96, 96, 24 },   { 144, 144, 24 }, { 144, 144, 32 }, { 192, 192, 32 }, { 192, 192, 32 },  { 192, 192, 64 },  { 384, 384, 64 },  { 384, 384, 64 },
                                                 { 384, 384, 64 }, { 384, 384, 96 }, { 576, 576, 96 }, { 576, 576, 96 }, { 576, 576, 160 }, { 960, 960, 160 }, { 960, 960, 160 }, { 960, 960, 320 } };

const size_t lastLayerSize = 1280;

size_t createTopology(raul::Workflow& work, size_t IMAGE_SIZE, size_t IMAGE_CHANNELS, size_t NUM_CLASSES, float bnMomentum, bool bias)
{
    work.add<raul::DataLayer>("data", raul::DataParams{ { "data", "labels" }, IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE, NUM_CLASSES });

    // 0
    work.add<raul::Convolution2DLayer>("conv1", raul::Convolution2DParams{ { "data" }, { "conv1" }, 3, 32, 2, 1, bias });
    work.add<raul::BatchNormLayer>("bn1", raul::BatchnormParams{ { "conv1" }, { "bn1" }, bnMomentum, 1e-5f });
    // bnNames.push_back("bn1");
    work.add<raul::ReLU6Activation>("relu1", raul::BasicParams{ { "bn1" }, { "relu1" } });

    // 1
    work.add<raul::ConvolutionDepthwiseLayer>("conv2", raul::Convolution2DParams{ { "relu1" }, { "conv2" }, 3, 32, 1, 1, bias });
    work.add<raul::BatchNormLayer>("bn2", raul::BatchnormParams{ { "conv2" }, { "bn2" }, bnMomentum, 1e-5f });
    // bnNames.push_back("bn2");
    work.add<raul::ReLU6Activation>("relu2", raul::BasicParams{ { "bn2" }, { "relu2" } });

    work.add<raul::Convolution2DLayer>("conv3", raul::Convolution2DParams{ { "relu2" }, { "conv3" }, 1, 16, 1, 0, bias });
    work.add<raul::BatchNormLayer>("bn3", raul::BatchnormParams{ { "conv3" }, { "bn3" }, bnMomentum, 1e-5f });
    // bnNames.push_back("bn3");

    std::string inputName = "bn3";

    const size_t avgWidth = 7;

    const size_t strideSizes[reproduceLayers] = { 2, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1 };

    const bool residual[reproduceLayers] = { 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0 };

    size_t layerIndex = 4;

    for (size_t w = 0; w < reproduceLayers; ++w)
    {
        work.add<raul::Convolution2DLayer>("conv" + Conversions::toString(layerIndex),
                                           raul::Convolution2DParams{ { inputName }, { "conv" + Conversions::toString(layerIndex) }, 1, filterSizes[w][0], 1, 0, bias });
        work.add<raul::BatchNormLayer>("bn" + Conversions::toString(layerIndex),
                                       raul::BatchnormParams{ { "conv" + Conversions::toString(layerIndex) }, { "bn" + Conversions::toString(layerIndex) }, bnMomentum, 1e-5f });
        // bnNames.push_back("bn" + Conversions::toString(layerIndex));
        work.add<raul::ReLU6Activation>("relu" + Conversions::toString(layerIndex), raul::BasicParams{ { "bn" + Conversions::toString(layerIndex) }, { "relu" + Conversions::toString(layerIndex) } });

        ++layerIndex;

        work.add<raul::ConvolutionDepthwiseLayer>(
            "conv" + Conversions::toString(layerIndex),
            raul::Convolution2DParams{ { "relu" + Conversions::toString(layerIndex - 1) }, { "conv" + Conversions::toString(layerIndex) }, 3, filterSizes[w][1], strideSizes[w], 1, bias });
        work.add<raul::BatchNormLayer>("bn" + Conversions::toString(layerIndex),
                                       raul::BatchnormParams{ { "conv" + Conversions::toString(layerIndex) }, { "bn" + Conversions::toString(layerIndex) }, bnMomentum, 1e-5f });

        work.add<raul::ReLU6Activation>("relu" + Conversions::toString(layerIndex), raul::BasicParams{ { "bn" + Conversions::toString(layerIndex) }, { "relu" + Conversions::toString(layerIndex) } });

        ++layerIndex;

        work.add<raul::Convolution2DLayer>(
            "conv" + Conversions::toString(layerIndex),
            raul::Convolution2DParams{ { "relu" + Conversions::toString(layerIndex - 1) }, { "conv" + Conversions::toString(layerIndex) }, 1, filterSizes[w][2], 1, 0, bias });
        work.add<raul::BatchNormLayer>("bn" + Conversions::toString(layerIndex),
                                       raul::BatchnormParams{ { "conv" + Conversions::toString(layerIndex) }, { "bn" + Conversions::toString(layerIndex) }, bnMomentum, 1e-5f });
        // bnNames.push_back("bn" + Conversions::toString(layerIndex));

        if (residual[w])
        {
            work.add<raul::ElementWiseSumLayer>("sum" + Conversions::toString(layerIndex),
                                                raul::ElementWiseLayerParams{ { "bn" + Conversions::toString(layerIndex), inputName }, { "sum" + Conversions::toString(layerIndex) } });
            inputName = "sum" + Conversions::toString(layerIndex);
        }
        else
            inputName = "bn" + Conversions::toString(layerIndex);

        ++layerIndex;
    }

    // 18
    work.add<raul::Convolution2DLayer>("conv" + Conversions::toString(layerIndex),
                                       raul::Convolution2DParams{ { inputName }, { "conv" + Conversions::toString(layerIndex) }, 1, lastLayerSize, 1, 0, bias });
    work.add<raul::BatchNormLayer>("bn" + Conversions::toString(layerIndex),
                                   raul::BatchnormParams{ { "conv" + Conversions::toString(layerIndex) }, { "bn" + Conversions::toString(layerIndex) }, bnMomentum, 1e-5f });

    work.add<raul::ReLU6Activation>("relu" + Conversions::toString(layerIndex), raul::BasicParams{ { "bn" + Conversions::toString(layerIndex) }, { "relu" + Conversions::toString(layerIndex) } });

    work.add<raul::AveragePoolLayer>("avg", raul::Pool2DParams{ { "relu" + Conversions::toString(layerIndex) }, { "avg" }, avgWidth, 1 });
    work.add<raul::ReshapeLayer>("reshape", raul::ViewParams{ "avg", "avgr", 1, 1, -1 });
    work.add<raul::LinearLayer>("fc", raul::LinearParams{ { "avgr" }, { "fc" }, NUM_CLASSES, bias });

    work.add<raul::LogSoftMaxActivation>("softmax", raul::BasicParamsWithDim{ { "fc" }, { "softmax" } });
    work.add<raul::NLLLoss>("loss", raul::LossParams{ { "softmax", "labels" }, { "loss" }, "custom_batch_mean" });

    return layerIndex;
}

template<typename MM>
void loadModel(MM& memoryManager, raul::DataLoader& dataLoader, const std::string& cifarDir, size_t IMAGE_CHANNELS, size_t NUM_CLASSES, size_t layerIndex)
{
    memoryManager["conv1::Weights"] = dataLoader.loadFilters((UT::tools::getTestAssetsDir() / "test_cnn_layer" / cifarDir / "0_features.0.0.weight_").string(), 0, ".data", 3, 3, IMAGE_CHANNELS, 32);
    memoryManager["conv1::Biases"] = dataLoader.loadData(UT::tools::getTestAssetsDir() / "test_cnn_layer" / cifarDir / "0_features.0.0.bias.data", 1, 32);
    memoryManager["bn1::Weights"] = dataLoader.loadData(UT::tools::getTestAssetsDir() / "test_cnn_layer" / cifarDir / "0_features.0.1.weight.data", 1, 32);
    memoryManager["bn1::Biases"] = dataLoader.loadData(UT::tools::getTestAssetsDir() / "test_cnn_layer" / cifarDir / "0_features.0.1.bias.data", 1, 32);
    memoryManager["bn1::VarianceEval"] = dataLoader.loadData(UT::tools::getTestAssetsDir() / "test_cnn_layer" / cifarDir / "0_features.0.1.running_var.data", 1, 32);
    memoryManager["bn1::MeanEval"] = dataLoader.loadData(UT::tools::getTestAssetsDir() / "test_cnn_layer" / cifarDir / "0_features.0.1.running_mean.data", 1, 32);

    // dw
    memoryManager["conv2::Weights"] = dataLoader.loadFilters((UT::tools::getTestAssetsDir() / "test_cnn_layer" / cifarDir / "0_features.1.conv.0.weight_").string(), 0, ".data", 3, 3, 1, 32);
    memoryManager["conv2::Biases"] = dataLoader.loadData(UT::tools::getTestAssetsDir() / "test_cnn_layer" / cifarDir / "0_features.1.conv.0.bias.data", 1, 32);
    memoryManager["bn2::Weights"] = dataLoader.loadData(UT::tools::getTestAssetsDir() / "test_cnn_layer" / cifarDir / "0_features.1.conv.1.weight.data", 1, 32);
    memoryManager["bn2::Biases"] = dataLoader.loadData(UT::tools::getTestAssetsDir() / "test_cnn_layer" / cifarDir / "0_features.1.conv.1.bias.data", 1, 32);
    memoryManager["bn2::VarianceEval"] = dataLoader.loadData(UT::tools::getTestAssetsDir() / "test_cnn_layer" / cifarDir / "0_features.1.conv.1.running_var.data", 1, 32);
    memoryManager["bn2::MeanEval"] = dataLoader.loadData(UT::tools::getTestAssetsDir() / "test_cnn_layer" / cifarDir / "0_features.1.conv.1.running_mean.data", 1, 32);

    memoryManager["conv3::Weights"] = dataLoader.loadFilters((UT::tools::getTestAssetsDir() / "test_cnn_layer" / cifarDir / "0_features.1.conv.3.weight_").string(), 0, ".data", 1, 1, 32, 16);
    memoryManager["conv3::Biases"] = dataLoader.loadData(UT::tools::getTestAssetsDir() / "test_cnn_layer" / cifarDir / "0_features.1.conv.3.bias.data", 1, 16);
    memoryManager["bn3::Weights"] = dataLoader.loadData(UT::tools::getTestAssetsDir() / "test_cnn_layer" / cifarDir / "0_features.1.conv.4.weight.data", 1, 16);
    memoryManager["bn3::Biases"] = dataLoader.loadData(UT::tools::getTestAssetsDir() / "test_cnn_layer" / cifarDir / "0_features.1.conv.4.bias.data", 1, 16);
    memoryManager["bn3::VarianceEval"] = dataLoader.loadData(UT::tools::getTestAssetsDir() / "test_cnn_layer" / cifarDir / "0_features.1.conv.4.running_var.data", 1, 16);
    memoryManager["bn3::MeanEval"] = dataLoader.loadData(UT::tools::getTestAssetsDir() / "test_cnn_layer" / cifarDir / "0_features.1.conv.4.running_mean.data", 1, 16);

    const size_t firstFileIndex = 0;
    const size_t secondFileIndex = 3;
    const size_t thirdFileIndex = 6;

    size_t fileNames[reproduceLayers][3] = { { firstFileIndex, secondFileIndex, thirdFileIndex }, { firstFileIndex, secondFileIndex, thirdFileIndex },
                                             { firstFileIndex, secondFileIndex, thirdFileIndex }, { firstFileIndex, secondFileIndex, thirdFileIndex },
                                             { firstFileIndex, secondFileIndex, thirdFileIndex }, { firstFileIndex, secondFileIndex, thirdFileIndex },
                                             { firstFileIndex, secondFileIndex, thirdFileIndex }, { firstFileIndex, secondFileIndex, thirdFileIndex },
                                             { firstFileIndex, secondFileIndex, thirdFileIndex }, { firstFileIndex, secondFileIndex, thirdFileIndex },
                                             { firstFileIndex, secondFileIndex, thirdFileIndex }, { firstFileIndex, secondFileIndex, thirdFileIndex },
                                             { firstFileIndex, secondFileIndex, thirdFileIndex }, { firstFileIndex, secondFileIndex, thirdFileIndex },
                                             { firstFileIndex, secondFileIndex, thirdFileIndex }, { firstFileIndex, secondFileIndex, thirdFileIndex } };

    size_t layerIndexLoad = 4;

    for (size_t w = 0; w < reproduceLayers; ++w)
    {
        size_t prevFilterSizeFirst = 16;
        if (w != 0)
        {
            prevFilterSizeFirst = filterSizes[w - 1][2];
        }

        memoryManager["conv" + Conversions::toString(layerIndexLoad) + "::Weights"] = dataLoader.loadFilters(
            (UT::tools::getTestAssetsDir() / "test_cnn_layer" / cifarDir / ("0_features." + Conversions::toString(w + 2) + ".conv." + Conversions::toString(fileNames[w][0]) + ".weight_")).string(),
            0,
            ".data",
            1,
            1,
            prevFilterSizeFirst,
            filterSizes[w][0]);
        memoryManager["conv" + Conversions::toString(layerIndexLoad) + "::Biases"] = dataLoader.loadData(
            UT::tools::getTestAssetsDir() / "test_cnn_layer" / cifarDir / ("0_features." + Conversions::toString(w + 2) + ".conv." + Conversions::toString(fileNames[w][0]) + ".bias.data"),
            1,
            filterSizes[w][0]);

        memoryManager["bn" + Conversions::toString(layerIndexLoad) + "::Weights"] = dataLoader.loadData(
            UT::tools::getTestAssetsDir() / "test_cnn_layer" / cifarDir / ("0_features." + Conversions::toString(w + 2) + ".conv." + Conversions::toString(fileNames[w][0] + 1) + ".weight.data"),
            1,
            filterSizes[w][0]);
        memoryManager["bn" + Conversions::toString(layerIndexLoad) + "::Biases"] = dataLoader.loadData(
            UT::tools::getTestAssetsDir() / "test_cnn_layer" / cifarDir / ("0_features." + Conversions::toString(w + 2) + ".conv." + Conversions::toString(fileNames[w][0] + 1) + ".bias.data"),
            1,
            filterSizes[w][0]);

        memoryManager["bn" + Conversions::toString(layerIndexLoad) + "::VarianceEval"] = dataLoader.loadData(
            UT::tools::getTestAssetsDir() / "test_cnn_layer" / cifarDir / ("0_features." + Conversions::toString(w + 2) + ".conv." + Conversions::toString(fileNames[w][0] + 1) + ".running_var.data"),
            1,
            filterSizes[w][0]);
        memoryManager["bn" + Conversions::toString(layerIndexLoad) + "::MeanEval"] = dataLoader.loadData(
            UT::tools::getTestAssetsDir() / "test_cnn_layer" / cifarDir / ("0_features." + Conversions::toString(w + 2) + ".conv." + Conversions::toString(fileNames[w][0] + 1) + ".running_mean.data"),
            1,
            filterSizes[w][0]);

        ++layerIndexLoad;

        // dw
        memoryManager["conv" + Conversions::toString(layerIndexLoad) + "::Weights"] = dataLoader.loadFilters(
            (UT::tools::getTestAssetsDir() / "test_cnn_layer" / cifarDir / ("0_features." + Conversions::toString(w + 2) + ".conv." + Conversions::toString(fileNames[w][1]) + ".weight_")).string(),
            0,
            ".data",
            3,
            3,
            1,
            filterSizes[w][1]);
        memoryManager["conv" + Conversions::toString(layerIndexLoad) + "::Biases"] = dataLoader.loadData(
            UT::tools::getTestAssetsDir() / "test_cnn_layer" / cifarDir / ("0_features." + Conversions::toString(w + 2) + ".conv." + Conversions::toString(fileNames[w][1]) + ".bias.data"),
            1,
            filterSizes[w][1]);

        memoryManager["bn" + Conversions::toString(layerIndexLoad) + "::Weights"] = dataLoader.loadData(
            UT::tools::getTestAssetsDir() / "test_cnn_layer" / cifarDir / ("0_features." + Conversions::toString(w + 2) + ".conv." + Conversions::toString(fileNames[w][1] + 1) + ".weight.data"),
            1,
            filterSizes[w][1]);
        memoryManager["bn" + Conversions::toString(layerIndexLoad) + "::Biases"] = dataLoader.loadData(
            UT::tools::getTestAssetsDir() / "test_cnn_layer" / cifarDir / ("0_features." + Conversions::toString(w + 2) + ".conv." + Conversions::toString(fileNames[w][1] + 1) + ".bias.data"),
            1,
            filterSizes[w][1]);

        memoryManager["bn" + Conversions::toString(layerIndexLoad) + "::VarianceEval"] = dataLoader.loadData(
            UT::tools::getTestAssetsDir() / "test_cnn_layer" / cifarDir / ("0_features." + Conversions::toString(w + 2) + ".conv." + Conversions::toString(fileNames[w][1] + 1) + ".running_var.data"),
            1,
            filterSizes[w][1]);
        memoryManager["bn" + Conversions::toString(layerIndexLoad) + "::MeanEval"] = dataLoader.loadData(
            UT::tools::getTestAssetsDir() / "test_cnn_layer" / cifarDir / ("0_features." + Conversions::toString(w + 2) + ".conv." + Conversions::toString(fileNames[w][1] + 1) + ".running_mean.data"),
            1,
            filterSizes[w][1]);

        ++layerIndexLoad;

        memoryManager["conv" + Conversions::toString(layerIndexLoad) + "::Weights"] = dataLoader.loadFilters(
            (UT::tools::getTestAssetsDir() / "test_cnn_layer" / cifarDir / ("0_features." + Conversions::toString(w + 2) + ".conv." + Conversions::toString(fileNames[w][2]) + ".weight_")).string(),
            0,
            ".data",
            1,
            1,
            filterSizes[w][1],
            filterSizes[w][2]);
        memoryManager["conv" + Conversions::toString(layerIndexLoad) + "::Biases"] = dataLoader.loadData(
            UT::tools::getTestAssetsDir() / "test_cnn_layer" / cifarDir / ("0_features." + Conversions::toString(w + 2) + ".conv." + Conversions::toString(fileNames[w][2]) + ".bias.data"),
            1,
            filterSizes[w][2]);

        memoryManager["bn" + Conversions::toString(layerIndexLoad) + "::Weights"] = dataLoader.loadData(
            UT::tools::getTestAssetsDir() / "test_cnn_layer" / cifarDir / ("0_features." + Conversions::toString(w + 2) + ".conv." + Conversions::toString(fileNames[w][2] + 1) + ".weight.data"),
            1,
            filterSizes[w][2]);
        memoryManager["bn" + Conversions::toString(layerIndexLoad) + "::Biases"] = dataLoader.loadData(
            UT::tools::getTestAssetsDir() / "test_cnn_layer" / cifarDir / ("0_features." + Conversions::toString(w + 2) + ".conv." + Conversions::toString(fileNames[w][2] + 1) + ".bias.data"),
            1,
            filterSizes[w][2]);

        memoryManager["bn" + Conversions::toString(layerIndexLoad) + "::VarianceEval"] = dataLoader.loadData(
            UT::tools::getTestAssetsDir() / "test_cnn_layer" / cifarDir / ("0_features." + Conversions::toString(w + 2) + ".conv." + Conversions::toString(fileNames[w][2] + 1) + ".running_var.data"),
            1,
            filterSizes[w][2]);
        memoryManager["bn" + Conversions::toString(layerIndexLoad) + "::MeanEval"] = dataLoader.loadData(
            UT::tools::getTestAssetsDir() / "test_cnn_layer" / cifarDir / ("0_features." + Conversions::toString(w + 2) + ".conv." + Conversions::toString(fileNames[w][2] + 1) + ".running_mean.data"),
            1,
            filterSizes[w][2]);

        ++layerIndexLoad;
    }

    memoryManager["conv" + Conversions::toString(layerIndex) + "::Weights"] =
        dataLoader.loadFilters((UT::tools::getTestAssetsDir() / "test_cnn_layer" / cifarDir / ("0_features." + Conversions::toString(reproduceLayers + 2) + ".0.weight_")).string(),
                               0,
                               ".data",
                               1,
                               1,
                               filterSizes[reproduceLayers - 1][2],
                               lastLayerSize);
    memoryManager["conv" + Conversions::toString(layerIndex) + "::Biases"] =
        dataLoader.loadData(UT::tools::getTestAssetsDir() / "test_cnn_layer" / cifarDir / ("0_features." + Conversions::toString(reproduceLayers + 2) + ".0.bias.data"), 1, lastLayerSize);

    memoryManager["bn" + Conversions::toString(layerIndex) + "::Weights"] =
        dataLoader.loadData(UT::tools::getTestAssetsDir() / "test_cnn_layer" / cifarDir / ("0_features." + Conversions::toString(reproduceLayers + 2) + ".1.weight.data"), 1, lastLayerSize);
    memoryManager["bn" + Conversions::toString(layerIndex) + "::Biases"] =
        dataLoader.loadData(UT::tools::getTestAssetsDir() / "test_cnn_layer" / cifarDir / ("0_features." + Conversions::toString(reproduceLayers + 2) + ".1.bias.data"), 1, lastLayerSize);

    memoryManager["bn" + Conversions::toString(layerIndex) + "::VarianceEval"] =
        dataLoader.loadData(UT::tools::getTestAssetsDir() / "test_cnn_layer" / cifarDir / ("0_features." + Conversions::toString(reproduceLayers + 2) + ".1.running_var.data"), 1, lastLayerSize);
    memoryManager["bn" + Conversions::toString(layerIndex) + "::MeanEval"] =
        dataLoader.loadData(UT::tools::getTestAssetsDir() / "test_cnn_layer" / cifarDir / ("0_features." + Conversions::toString(reproduceLayers + 2) + ".1.running_mean.data"), 1, lastLayerSize);

    memoryManager["fc::Weights"] = dataLoader.loadData(UT::tools::getTestAssetsDir() / "test_cnn_layer" / cifarDir / "0_classifier.weight.data", NUM_CLASSES, lastLayerSize);
    raul::Common::transpose(memoryManager["fc::Weights"], NUM_CLASSES);

    memoryManager["fc::Biases"] = dataLoader.loadData(UT::tools::getTestAssetsDir() / "test_cnn_layer" / cifarDir / "0_classifier.bias.data", 1, NUM_CLASSES);
}

} // anonymous

namespace UT
{

TEST(TestMobileNetV2, TopologyUnit)
{
    PROFILE_TEST

    const size_t golden_trainable_parameters = 2253738U;

    bool useCheckpointing = false;
    bool usePool = false;

    const size_t BATCH_SIZE = 50;

    const size_t NUM_CLASSES = 10;
    const size_t IMAGE_SIZE = 224;
    const size_t IMAGE_CHANNELS = 3;

    const float bnMomentum = 0.1f;

    const bool bias = true;
    bool compressionConv = false;
    raul::CompressionMode compressionMode = compressionConv ? raul::CompressionMode::FP16 : raul::CompressionMode::NONE;
    // raul::CompressionMode compressionMode = raul::CompressionMode::INT8;

    raul::Workflow work(compressionMode, raul::CalculationMode::DETERMINISTIC, usePool ? raul::AllocationMode::POOL : raul::AllocationMode::STANDARD);

    createTopology(work, IMAGE_SIZE, IMAGE_CHANNELS, NUM_CLASSES, bnMomentum, bias);

    if (useCheckpointing)
    {
        // work.setCheckpoints({"bn3", "bn6", "bn9", "bn12"});
        raul::Names checkpointsAll = work.getPotentialCheckpoints();
        raul::Names checkpoints;
        for (raul::Name& checkP : checkpointsAll)
        {
            if (checkP.str().find("bn") != std::string::npos)
            {
                checkpoints.push_back(checkP);
            }
        }
        work.setCheckpoints(checkpoints);
        work.preparePipelines(raul::Workflow::Execution::Checkpointed);
    }
    else
    {
        work.preparePipelines();
    }
    work.setBatchSize(BATCH_SIZE);
    work.prepareMemoryForTraining();

    // Checks
    EXPECT_EQ(tools::get_size_of_trainable_params(work), golden_trainable_parameters);
}

TEST(TestMobileNetV2, Training)
{
    PROFILE_TEST
    bool useMicroBatches = false;
    bool useCheckpointing = false;
    bool usePool = false;

    const auto LEARNING_RATE = 0.05_dt;
    const size_t BATCH_SIZE = 50;
    const size_t MICRO_BATCH_SIZE = 10;
    auto EPSILON_ACCURACY = 0.5_dt;
    const auto EPSILON_LOSS = 0.1_dt;

    const size_t NUM_CLASSES = 10;
    const auto acc1 = 83.36_dt;
    const auto acc2 = 83.41_dt;
    const size_t IMAGE_SIZE = 224;
    const size_t IMAGE_CHANNELS = 3;

    const float bnMomentum = 0.1f;

    const bool bias = true;
    bool compressionConv = false;
    raul::CompressionMode compressionMode = compressionConv ? raul::CompressionMode::FP16 : raul::CompressionMode::NONE;
    // raul::CompressionMode compressionMode = raul::CompressionMode::INT8;

    if (compressionConv)
    {
        EPSILON_ACCURACY = TODTYPE(1e-0);
    }

    raul::Workflow work(compressionMode, raul::CalculationMode::DETERMINISTIC, usePool ? raul::AllocationMode::POOL : raul::AllocationMode::STANDARD);

    size_t layerIndex = createTopology(work, IMAGE_SIZE, IMAGE_CHANNELS, NUM_CLASSES, bnMomentum, bias);

    if (useCheckpointing)
    {
        // work.setCheckpoints({"bn3", "bn6", "bn9", "bn12"});
        raul::Names checkpointsAll = work.getPotentialCheckpoints();
        raul::Names checkpoints;
        for (raul::Name& checkP : checkpointsAll)
        {
            if (checkP.str().find("bn") != std::string::npos)
            {
                checkpoints.push_back(checkP);
            }
        }
        work.setCheckpoints(checkpoints);
        work.preparePipelines(raul::Workflow::Execution::Checkpointed);
    }
    else
    {
        work.preparePipelines();
    }
    work.setBatchSize(BATCH_SIZE);
    work.prepareMemoryForTraining();

    // work.printInfo(std::cout);

    printf("Topology created\n");

    raul::MemoryManager& memoryManager = work.getMemoryManager();
    raul::DataLoader dataLoader;

    const std::string cifarDir = "cifarMobilenetV2";

    loadModel(memoryManager, dataLoader, cifarDir, IMAGE_CHANNELS, NUM_CLASSES, layerIndex);

    printf("Model loaded\n");

    raul::Dataset trainData = raul::Dataset::CIFAR_Train(tools::getTestAssetsDir() / "CIFAR");
    trainData.applyTo("images", std::make_unique<raul::Resize>(224, 224));

    raul::Dataset testData = raul::Dataset::CIFAR_Test(tools::getTestAssetsDir() / "CIFAR");
    testData.applyTo("images", std::make_unique<raul::Resize>(224, 224));

    printf("Dataset loaded\n");

    printf("Memory taken = %.2fMB \n\n", static_cast<float>(work.getMemoryManager().getTotalMemory()) / (1024.0f * 1024.0f));

    raul::Tensor& idealLosses = dataLoader.createTensor(trainData.numberOfSamples() / BATCH_SIZE / 100);
    raul::DataLoader::readArrayFromTextFile(tools::getTestAssetsDir() / "test_cnn_layer" / "cifarMobilenetV2" / "loss.data", idealLosses, 1, idealLosses.size());

    raul::Test test(work, testData, { { { "images", "data" }, { "labels", "labels" } }, "softmax", "labels" });
    raul::dtype testAcc = test.run(useMicroBatches ? MICRO_BATCH_SIZE : BATCH_SIZE);
    CHECK_NEAR(testAcc, acc1, EPSILON_ACCURACY);
    printf("Test accuracy = %.2f\n", testAcc);

    raul::Train train(work, trainData, { { { "images", "data" }, { "labels", "labels" } }, "loss" });
    useMicroBatches ? train.useMicroBatches(BATCH_SIZE, MICRO_BATCH_SIZE) : train.useBatches(BATCH_SIZE);
    auto sgd = std::make_shared<raul::optimizers::SGD>(LEARNING_RATE);
    for (size_t epoch = 1; epoch <= 1; ++epoch)
    {
        printf("Epoch = %zu\n", epoch);

        for (size_t q = 0, idealLossIndex = 0; q < train.numberOfIterations(); ++q)
        {
            // std::chrono::steady_clock::time_point timeStart = std::chrono::steady_clock::now();

            raul::dtype testLoss = train.oneIteration(*sgd);
            if (q % 100 == 0)
            {
                if (epoch == 1)
                {
                    CHECK_NEAR(testLoss, idealLosses[idealLossIndex++], EPSILON_LOSS);
                }
                printf("iteration = %d/%d, loss = %f\n", static_cast<uint32_t>(q), static_cast<uint32_t>(train.numberOfIterations()), testLoss);
            }
            // float timeR = (std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - timeStart).count()) / 1000.0f;
            // printf("%f\n", timeR);
        }

        testAcc = test.run(useMicroBatches ? MICRO_BATCH_SIZE : BATCH_SIZE);
        printf("Test accuracy = %.2f\n", testAcc);
        CHECK_NEAR(testAcc, acc2, EPSILON_ACCURACY);
    }

    printf("Memory taken = %.2fMB \n\n", static_cast<float>(work.getMemoryManager().getTotalMemory()) / (1024.0f * 1024.0f));
}

#if defined(ANDROID)
TEST(TestMobileNetV2, TrainingFP16)
{
    PROFILE_TEST
    bool useMicroBatches = false;
    bool useCheckpointing = false;
    bool usePool = false;

    const auto LEARNING_RATE = 0.05_dt;
    const size_t BATCH_SIZE = 50;
    const size_t MICRO_BATCH_SIZE = 10;
    auto EPSILON_ACCURACY = 0.5_dt;

    const size_t NUM_CLASSES = 10;
    const auto acc1 = 83.36_dt;
    const auto acc2 = 83.41_dt;
    const size_t IMAGE_SIZE = 224;
    const size_t IMAGE_CHANNELS = 3;

    const float bnMomentum = 0.1f;

    const bool bias = true;
    bool compressionConv = false;
    raul::CompressionMode compressionMode = compressionConv ? raul::CompressionMode::FP16 : raul::CompressionMode::NONE;
    // raul::CompressionMode compressionMode = raul::CompressionMode::INT8;

    if (compressionConv)
    {
        EPSILON_ACCURACY = TODTYPE(1e-0);
    }

    raul::Workflow work(compressionMode, raul::CalculationMode::DETERMINISTIC, usePool ? raul::AllocationMode::POOL : raul::AllocationMode::STANDARD, raul::ExecutionTarget::CPUFP16);

    size_t layerIndex = createTopology(work, IMAGE_SIZE, IMAGE_CHANNELS, NUM_CLASSES, bnMomentum, bias);

    if (useCheckpointing)
    {
        // work.setCheckpoints({"bn3", "bn6", "bn9", "bn12"});
        raul::Names checkpointsAll = work.getPotentialCheckpoints();
        raul::Names checkpoints;
        for (raul::Name& checkP : checkpointsAll)
        {
            if (checkP.str().find("bn") != std::string::npos)
            {
                checkpoints.push_back(checkP);
            }
        }
        work.setCheckpoints(checkpoints);
        work.preparePipelines(raul::Workflow::Execution::Checkpointed);
    }
    else
    {
        work.preparePipelines();
    }
    work.setBatchSize(BATCH_SIZE);
    work.prepareMemoryForTraining();

    // work.printInfo(std::cout);

    printf("Topology created\n");

    auto& memoryManager = work.getMemoryManager<raul::MemoryManagerFP16>();
    raul::DataLoader dataLoader;

    const std::string cifarDir = "cifarMobilenetV2";

    loadModel(memoryManager, dataLoader, cifarDir, IMAGE_CHANNELS, NUM_CLASSES, layerIndex);

    printf("Model loaded\n");

    raul::Dataset trainData = raul::Dataset::CIFAR_Train(tools::getTestAssetsDir() / "CIFAR");
    trainData.applyTo("images", std::make_unique<raul::Resize>(224, 224));

    raul::Dataset testData = raul::Dataset::CIFAR_Test(tools::getTestAssetsDir() / "CIFAR");
    testData.applyTo("images", std::make_unique<raul::Resize>(224, 224));

    printf("Dataset loaded\n");

    printf("Memory taken = %.2fMB \n\n", static_cast<float>(work.getMemoryManager().getTotalMemory()) / (1024.0f * 1024.0f));

    raul::Test test(work, testData, { { { "images", "data" }, { "labels", "labels" } }, "softmax", "labels" });
    raul::dtype testAcc = test.run(useMicroBatches ? MICRO_BATCH_SIZE : BATCH_SIZE);
    CHECK_NEAR(testAcc, acc1, EPSILON_ACCURACY);
    printf("Test accuracy = %.2f\n", testAcc);

    raul::Train train(work, trainData, { { { "images", "data" }, { "labels", "labels" } }, "loss" });
    useMicroBatches ? train.useMicroBatches(BATCH_SIZE, MICRO_BATCH_SIZE) : train.useBatches(BATCH_SIZE);
    auto sgd = std::make_shared<raul::optimizers::SGD>(LEARNING_RATE);
    for (size_t epoch = 1; epoch <= 1; ++epoch)
    {
        printf("Epoch = %zu\n", epoch);

        for (size_t q = 0; q < train.numberOfIterations(); ++q)
        {
            // std::chrono::steady_clock::time_point timeStart = std::chrono::steady_clock::now();

            raul::dtype testLoss = train.oneIteration(*sgd);
            if (q % 100 == 0)
            {
                printf("iteration = %d/%d, loss = %f\n", static_cast<uint32_t>(q), static_cast<uint32_t>(train.numberOfIterations()), testLoss);
            }
            // float timeR = (std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - timeStart).count()) / 1000.0f;
            // printf("%f\n", timeR);
        }

        testAcc = test.run(useMicroBatches ? MICRO_BATCH_SIZE : BATCH_SIZE);
        printf("Test accuracy = %.2f\n", testAcc);
        CHECK_NEAR(testAcc, acc2, EPSILON_ACCURACY);
    }

    printf("Memory taken = %.2fMB \n\n", static_cast<float>(work.getMemoryManager().getTotalMemory()) / (1024.0f * 1024.0f));
}
#endif

TEST(TestMobileNetV2, CheckpointingTraining)
{
    PROFILE_TEST

    const auto LEARNING_RATE = 0.05_dt;
    const size_t BATCH_SIZE = 64;

    const size_t NUM_CLASSES = 10;
    const size_t IMAGE_SIZE = 224;
    const size_t IMAGE_CHANNELS = 3;

    const float bnMomentum = 0.1f;

    const bool bias = true;

    raul::Workflow work(raul::CompressionMode::NONE, raul::CalculationMode::DETERMINISTIC, raul::AllocationMode::STANDARD);

    size_t layerIndex = createTopology(work, IMAGE_SIZE, IMAGE_CHANNELS, NUM_CLASSES, bnMomentum, bias);

    raul::Names checkpointsAll = work.getPotentialCheckpoints();
    raul::Names checkpoints;
    for (raul::Name& checkP : checkpointsAll)
    {
        if (checkP.str().find("bn") != std::string::npos)
        {
            checkpoints.push_back(checkP);
        }
    }
    work.setCheckpoints(checkpoints);
    work.preparePipelines(raul::Workflow::Execution::Checkpointed);

    work.setBatchSize(BATCH_SIZE);
    work.prepareMemoryForTraining();

    printf("Topology created\n");

    raul::MemoryManager& memoryManager = work.getMemoryManager();
    raul::DataLoader dataLoader;

    const std::string cifarDir = "cifarMobilenetV2";

    loadModel(memoryManager, dataLoader, cifarDir, IMAGE_CHANNELS, NUM_CLASSES, layerIndex);

    printf("Model loaded\n");

    raul::CIFAR10 cifar;
    ASSERT_TRUE(cifar.loadingData(tools::getTestAssetsDir() / "CIFAR"));

    printf("Dataset loaded\n");

    auto sgd = std::make_shared<raul::optimizers::SGD>(LEARNING_RATE);

    #if defined(_BLAS)
        std::vector<raul::dtype> goldLoss = { 0.045231_dt, 0.005066_dt };
    #else
        std::vector<raul::dtype> goldLoss = { 0.045231_dt, 0.005067_dt };
    #endif

    for (size_t q = 0; q < 2; ++q)
    {
        std::chrono::steady_clock::time_point timeStart = std::chrono::steady_clock::now();
        raul::dtype testLoss = cifar.oneTrainIteration(work, sgd.get(), q, std::make_pair(224, 224));
        CHECK_NEAR(testLoss, goldLoss[q], 1e-6);
        float timeR = static_cast<float>(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - timeStart).count()) / 1000.0f;
        printf("%f\n", timeR);
    }
}

TEST(TestMobileNetV2, CheckpointingPoolTraining)
{
    PROFILE_TEST

    const auto LEARNING_RATE = 0.05_dt;
    const size_t BATCH_SIZE = 64;

    const size_t NUM_CLASSES = 10;
    const size_t IMAGE_SIZE = 224;
    const size_t IMAGE_CHANNELS = 3;

    const float bnMomentum = 0.1f;

    const bool bias = true;

    raul::Workflow work(raul::CompressionMode::NONE, raul::CalculationMode::DETERMINISTIC, raul::AllocationMode::POOL);

    size_t layerIndex = createTopology(work, IMAGE_SIZE, IMAGE_CHANNELS, NUM_CLASSES, bnMomentum, bias);

    raul::Names checkpointsAll = work.getPotentialCheckpoints();
    raul::Names checkpoints;
    for (raul::Name& checkP : checkpointsAll)
    {
        if (checkP.str().find("bn") != std::string::npos)
        {
            checkpoints.push_back(checkP);
        }
    }
    work.setCheckpoints(checkpoints);
    work.preparePipelines(raul::Workflow::Execution::Checkpointed);

    work.setBatchSize(BATCH_SIZE);
    work.prepareMemoryForTraining();

    printf("Topology created\n");

    raul::MemoryManager& memoryManager = work.getMemoryManager();
    raul::DataLoader dataLoader;

    const std::string cifarDir = "cifarMobilenetV2";

    loadModel(memoryManager, dataLoader, cifarDir, IMAGE_CHANNELS, NUM_CLASSES, layerIndex);

    printf("Model loaded\n");

    raul::CIFAR10 cifar;
    ASSERT_TRUE(cifar.loadingData(tools::getTestAssetsDir() / "CIFAR"));

    printf("Dataset loaded\n");

    auto sgd = std::make_shared<raul::optimizers::SGD>(LEARNING_RATE);

    std::vector<raul::dtype> goldLoss = { 0.045231_dt, 0.005066_dt };

    for (size_t q = 0; q < 2; ++q)
    {
        std::chrono::steady_clock::time_point timeStart = std::chrono::steady_clock::now();
        raul::dtype testLoss = cifar.oneTrainIteration(work, sgd.get(), q, std::make_pair(224, 224));
        CHECK_NEAR(testLoss, goldLoss[q], 1e-6);
        float timeR = static_cast<float>(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - timeStart).count()) / 1000.0f;
        printf("%f\n", timeR);
    }
}
} // UT namespace
