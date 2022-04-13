// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <tests/tools/TestTools.h>

#include <training/compiler/Layers.h>
#include <training/base/layers/basic/trainable/Batchnorm.h>
#include <training/compiler/Workflow.h>
#include <training/base/optimizers/SGD.h>

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

} // UT namespace