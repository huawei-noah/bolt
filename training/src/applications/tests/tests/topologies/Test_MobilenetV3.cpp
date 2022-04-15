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
#include <training/common/Common.h>
#include <training/common/DataLoader.h>
#include <training/network/Layers.h>
#include <training/tools/Datasets.h>

#include <training/layers/basic/GlobalAveragePoolLayer.h>
#include <training/layers/basic/trainable/Batchnorm.h>
#include <training/layers/parameters/LayerParameters.h>

#include <training/common/Conversions.h>
#include <training/common/MemoryManager.h>
#include <training/network/Workflow.h>
#include <training/optimizers/Adam.h>
#include <training/optimizers/SGD.h>

#include <chrono>
#include <cstdio>

namespace UT
{

enum class NonLinearType : char
{
    ReLU,
    HSwish
};

/*
 * Squeeze-and-Excite Block aka SE Block
 */
void add_se_block(size_t& block_cnt, raul::Workflow& work, raul::Name& input, const size_t in_size, const size_t reduction = 4, const float bnMomentum = 0.1f)
{
    const auto block_name = "se" + Conversions::toString(block_cnt);
    ++block_cnt;

    const size_t internal_size = in_size / reduction;

    work.add<raul::GlobAveragePoolLayer>(block_name + "::avg", raul::BasicParams{ { input }, { block_name + "::avg" } });
    work.add<raul::Convolution2DLayer>(block_name + "::conv1", raul::Convolution2DParams{ { block_name + "::avg" }, { block_name + "::conv1" }, 1, internal_size, 1, 0 });
    work.add<raul::BatchNormLayer>(block_name + "::bn1", raul::BatchnormParams{ { block_name + "::conv1" }, { block_name + "::bn1" }, bnMomentum });
    work.add<raul::ReLUActivation>(block_name + "::relu", raul::BasicParams{ { block_name + "::bn1" }, { block_name + "::relu" } });
    work.add<raul::Convolution2DLayer>(block_name + "::conv2", raul::Convolution2DParams{ { block_name + "::relu" }, { block_name + "::conv2" }, 1, in_size, 1, 0 });
    work.add<raul::BatchNormLayer>(block_name + "::bn2", raul::BatchnormParams{ { block_name + "::conv2" }, { block_name + "::bn2" }, bnMomentum });
    work.add<raul::HSigmoidActivation>(block_name + "::hsigmoid", raul::HSigmoidActivationParams{ { block_name + "::bn2" }, { block_name + "::hsigmoid" } });

    work.add<raul::ElementWiseMulLayer>(block_name + "::mul", raul::ElementWiseLayerParams{ { input, block_name + "::hsigmoid" }, { block_name + "::mul" }, true });
    input = block_name + "::mul";
}

/*
 * MobileNetV3 Block: Expand + depthwise + pointwise
 */
void add_mobilenetv3_block(size_t& block_cnt,
                           size_t& se_block_cnt,
                           raul::Workflow& work,
                           raul::Name& input,
                           const size_t kernel_size,
                           const size_t in_channels,
                           const size_t expand_channels,
                           const size_t out_channels,
                           const NonLinearType nonlinear,
                           const int semodule,
                           const size_t stride,
                           const float bnMomentum)
{
    const auto block_name = "block" + Conversions::toString(block_cnt);
    ++block_cnt;

    auto input_for_shortcut = input;

    // 0: 1x1 NL
    work.add<raul::Convolution2DLayer>(block_name + "::conv0", raul::Convolution2DParams{ { input }, { block_name + "::conv0" }, 1, expand_channels, 1, 0 });
    work.add<raul::BatchNormLayer>(block_name + "::bn0", raul::BatchnormParams{ { block_name + "::conv0" }, { block_name + "::bn0" }, bnMomentum });
    switch (nonlinear)
    {
        case NonLinearType::ReLU:
            work.add<raul::ReLUActivation>(block_name + "::relu0", raul::BasicParams{ { block_name + "::bn0" }, { block_name + "::relu0" } });
            input = block_name + "::relu0";
            break;
        case NonLinearType::HSwish:
            work.add<raul::HSwishActivation>(block_name + "::hswish0", raul::HSwishActivationParams{ { block_name + "::bn0" }, { block_name + "::hswish0" } });
            input = block_name + "::hswish0";
            break;
            // default: Do nothing
    }

    // 1: Dwise
    work.add<raul::ConvolutionDepthwiseLayer>(block_name + "::conv1", raul::Convolution2DParams{ { input }, { block_name + "::conv1" }, kernel_size, expand_channels, stride, kernel_size / 2 });
    work.add<raul::BatchNormLayer>(block_name + "::bn1", raul::BatchnormParams{ { block_name + "::conv1" }, { block_name + "::bn1" }, bnMomentum });
    switch (nonlinear)
    {
        case NonLinearType::ReLU:
            work.add<raul::ReLUActivation>(block_name + "::relu1", raul::BasicParams{ { block_name + "::bn1" }, { block_name + "::relu1" } });
            input = block_name + "::relu1";
            break;
        case NonLinearType::HSwish:
            work.add<raul::HSwishActivation>(block_name + "::hswish1", raul::HSwishActivationParams{ { block_name + "::bn1" }, { block_name + "::hswish1" } });
            input = block_name + "::hswish1";
            break;
            // default: Do nothing
    }

    // 2: Linear
    work.add<raul::Convolution2DLayer>(block_name + "::conv2", raul::Convolution2DParams{ { input }, { block_name + "::conv2" }, 1, out_channels, 1 });
    work.add<raul::BatchNormLayer>(block_name + "::bn2", raul::BatchnormParams{ { block_name + "::conv2" }, { block_name + "::bn2" }, bnMomentum });
    input = block_name + "::bn2";

    if (semodule > -1)
    {
        add_se_block(se_block_cnt, work, input, semodule);
    }

    if (stride == 1U)
    {
        if (in_channels != out_channels)
        {
            work.add<raul::Convolution2DLayer>(block_name + "::conv3", raul::Convolution2DParams{ { input_for_shortcut }, { block_name + "::conv3" }, 1, out_channels, 1 });
            work.add<raul::BatchNormLayer>(block_name + "::bn3", raul::BatchnormParams{ { block_name + "::conv3" }, { block_name + "::bn3" }, bnMomentum });
            input_for_shortcut = block_name + "::bn3";
        }

        work.add<raul::ElementWiseSumLayer>(block_name + "::sum", raul::ElementWiseLayerParams{ { input_for_shortcut, input }, { block_name + "::sum" } });
        input = block_name + "::sum";
    }
}

void add_input_block(raul::Workflow& work, raul::Name& input, const float bnMomentum)
{
    work.add<raul::Convolution2DLayer>("input::conv0", raul::Convolution2DParams{ { input }, { "input::conv0" }, 3, 16, 2, 1 });
    work.add<raul::BatchNormLayer>("input::bn0", raul::BatchnormParams{ { "input::conv0" }, { "input::bn0" }, bnMomentum });
    work.add<raul::HSwishActivation>("input::hswish0", raul::HSwishActivationParams({ { "input::bn0" }, { "input::hswish0" } }));
    input = "input::hswish0";
}

void add_output_block(raul::Workflow& work, raul::Name& input, const float bnMomentum)
{
    work.add<raul::Convolution2DLayer>("output::conv0", raul::Convolution2DParams{ { input }, { "output::conv0" }, 1, 576, 1, 0 });
    work.add<raul::BatchNormLayer>("output::bn0", raul::BatchnormParams{ { "output::conv0" }, { "output::bn0" }, bnMomentum });
    work.add<raul::HSwishActivation>("output::hswish0", raul::HSwishActivationParams({ { "output::bn0" }, { "output::hswish0" } }));

    work.add<raul::AveragePoolLayer>("output::avg", raul::Pool2DParams{ { "output::hswish0" }, { "output::avg" }, 7, 1, 0 });
    work.add<raul::ReshapeLayer>("reshape", raul::ViewParams{ "output::avg", "output::avgr", 1, 1, -1 });
    work.add<raul::LinearLayer>("output::fc0", raul::LinearParams{ { "output::avgr" }, { "output::fc0" }, 1024 });
    work.add<raul::HSwishActivation>("output::hswish1", raul::HSwishActivationParams({ { "output::fc0" }, { "output::hswish1" } }));

    work.add<raul::LinearLayer>("output::fc1", raul::LinearParams{ { "output::hswish1" }, { "output::fc1" }, 10 });
    input = "output::fc1";
}

raul::Name build_mobilenetv3_small(raul::Workflow& work, const size_t image_size = 224U, const size_t image_channels = 3U, const size_t labels_cnt = 10U, const float bnMomentum = 0.1f)
{
    size_t bneck_block_cnt = 0;
    size_t se_block_cnt = 0;

    raul::Name input = "data";
    work.add<raul::DataLayer>("data", raul::DataParams{ { input, "labels" }, image_channels, image_size, image_size, labels_cnt });
    add_input_block(work, input, bnMomentum);
    add_mobilenetv3_block(bneck_block_cnt, se_block_cnt, work, input, 3, 16, 16, 16, NonLinearType::ReLU, 16, 2, bnMomentum);
    add_mobilenetv3_block(bneck_block_cnt, se_block_cnt, work, input, 3, 16, 72, 24, NonLinearType::ReLU, -1, 2, bnMomentum);
    add_mobilenetv3_block(bneck_block_cnt, se_block_cnt, work, input, 3, 24, 88, 24, NonLinearType::ReLU, -1, 1, bnMomentum);
    add_mobilenetv3_block(bneck_block_cnt, se_block_cnt, work, input, 5, 24, 96, 40, NonLinearType::HSwish, 40, 2, bnMomentum);
    add_mobilenetv3_block(bneck_block_cnt, se_block_cnt, work, input, 5, 40, 240, 40, NonLinearType::HSwish, 40, 1, bnMomentum);
    add_mobilenetv3_block(bneck_block_cnt, se_block_cnt, work, input, 5, 40, 240, 40, NonLinearType::HSwish, 40, 1, bnMomentum);
    add_mobilenetv3_block(bneck_block_cnt, se_block_cnt, work, input, 5, 40, 120, 48, NonLinearType::HSwish, 48, 1, bnMomentum);
    add_mobilenetv3_block(bneck_block_cnt, se_block_cnt, work, input, 5, 48, 144, 48, NonLinearType::HSwish, 48, 1, bnMomentum);
    add_mobilenetv3_block(bneck_block_cnt, se_block_cnt, work, input, 5, 48, 288, 96, NonLinearType::HSwish, 96, 2, bnMomentum);
    add_mobilenetv3_block(bneck_block_cnt, se_block_cnt, work, input, 5, 96, 576, 96, NonLinearType::HSwish, 96, 1, bnMomentum);
    add_mobilenetv3_block(bneck_block_cnt, se_block_cnt, work, input, 5, 96, 576, 96, NonLinearType::HSwish, 96, 1, bnMomentum);
    add_output_block(work, input, bnMomentum);
    return input;
}

void load_mobilenetv3_small_weights(raul::MemoryManager& memory_manager, const std::filesystem::path& weights_path, const std::string file_prefix = "init_")
{
    std::map<raul::Name, std::pair<std::string, size_t>> pytorch_name_map{
        { "input::conv0::Weights", { "conv1.weight", 432 } },
        { "input::conv0::Biases", { "conv1.bias", 16 } },
        { "input::bn0::Weights", { "bn1.weight", 16 } },
        { "input::bn0::Biases", { "bn1.bias", 16 } },
        { "se0::conv1::Weights", { "bneck.0.se.se.1.weight", 64 } },
        { "se0::conv1::Biases", { "bneck.0.se.se.1.bias", 4 } },
        { "se0::bn1::Weights", { "bneck.0.se.se.2.weight", 4 } },
        { "se0::bn1::Biases", { "bneck.0.se.se.2.bias", 4 } },
        { "se0::conv2::Weights", { "bneck.0.se.se.4.weight", 64 } },
        { "se0::conv2::Biases", { "bneck.0.se.se.4.bias", 16 } },
        { "se0::bn2::Weights", { "bneck.0.se.se.5.weight", 16 } },
        { "se0::bn2::Biases", { "bneck.0.se.se.5.bias", 16 } },
        { "block0::conv0::Weights", { "bneck.0.conv1.weight", 256 } },
        { "block0::conv0::Biases", { "bneck.0.conv1.bias", 16 } },
        { "block0::bn0::Weights", { "bneck.0.bn1.weight", 16 } },
        { "block0::bn0::Biases", { "bneck.0.bn1.bias", 16 } },
        { "block0::conv1::Weights", { "bneck.0.conv2.weight", 144 } },
        { "block0::conv1::Biases", { "bneck.0.conv2.bias", 16 } },
        { "block0::bn1::Weights", { "bneck.0.bn2.weight", 16 } },
        { "block0::bn1::Biases", { "bneck.0.bn2.bias", 16 } },
        { "block0::conv2::Weights", { "bneck.0.conv3.weight", 256 } },
        { "block0::conv2::Biases", { "bneck.0.conv3.bias", 16 } },
        { "block0::bn2::Weights", { "bneck.0.bn3.weight", 16 } },
        { "block0::bn2::Biases", { "bneck.0.bn3.bias", 16 } },
        { "block1::conv0::Weights", { "bneck.1.conv1.weight", 1152 } },
        { "block1::conv0::Biases", { "bneck.1.conv1.bias", 72 } },
        { "block1::bn0::Weights", { "bneck.1.bn1.weight", 72 } },
        { "block1::bn0::Biases", { "bneck.1.bn1.bias", 72 } },
        { "block1::conv1::Weights", { "bneck.1.conv2.weight", 648 } },
        { "block1::conv1::Biases", { "bneck.1.conv2.bias", 72 } },
        { "block1::bn1::Weights", { "bneck.1.bn2.weight", 72 } },
        { "block1::bn1::Biases", { "bneck.1.bn2.bias", 72 } },
        { "block1::conv2::Weights", { "bneck.1.conv3.weight", 1728 } },
        { "block1::conv2::Biases", { "bneck.1.conv3.bias", 24 } },
        { "block1::bn2::Weights", { "bneck.1.bn3.weight", 24 } },
        { "block1::bn2::Biases", { "bneck.1.bn3.bias", 24 } },
        { "block2::conv0::Weights", { "bneck.2.conv1.weight", 2112 } },
        { "block2::conv0::Biases", { "bneck.2.conv1.bias", 88 } },
        { "block2::bn0::Weights", { "bneck.2.bn1.weight", 88 } },
        { "block2::bn0::Biases", { "bneck.2.bn1.bias", 88 } },
        { "block2::conv1::Weights", { "bneck.2.conv2.weight", 792 } },
        { "block2::conv1::Biases", { "bneck.2.conv2.bias", 88 } },
        { "block2::bn1::Weights", { "bneck.2.bn2.weight", 88 } },
        { "block2::bn1::Biases", { "bneck.2.bn2.bias", 88 } },
        { "block2::conv2::Weights", { "bneck.2.conv3.weight", 2112 } },
        { "block2::conv2::Biases", { "bneck.2.conv3.bias", 24 } },
        { "block2::bn2::Weights", { "bneck.2.bn3.weight", 24 } },
        { "block2::bn2::Biases", { "bneck.2.bn3.bias", 24 } },
        { "se1::conv1::Weights", { "bneck.3.se.se.1.weight", 400 } },
        { "se1::conv1::Biases", { "bneck.3.se.se.1.bias", 10 } },
        { "se1::bn1::Weights", { "bneck.3.se.se.2.weight", 10 } },
        { "se1::bn1::Biases", { "bneck.3.se.se.2.bias", 10 } },
        { "se1::conv2::Weights", { "bneck.3.se.se.4.weight", 400 } },
        { "se1::conv2::Biases", { "bneck.3.se.se.4.bias", 40 } },
        { "se1::bn2::Weights", { "bneck.3.se.se.5.weight", 40 } },
        { "se1::bn2::Biases", { "bneck.3.se.se.5.bias", 40 } },
        { "block3::conv0::Weights", { "bneck.3.conv1.weight", 2304 } },
        { "block3::conv0::Biases", { "bneck.3.conv1.bias", 96 } },
        { "block3::bn0::Weights", { "bneck.3.bn1.weight", 96 } },
        { "block3::bn0::Biases", { "bneck.3.bn1.bias", 96 } },
        { "block3::conv1::Weights", { "bneck.3.conv2.weight", 2400 } },
        { "block3::conv1::Biases", { "bneck.3.conv2.bias", 96 } },
        { "block3::bn1::Weights", { "bneck.3.bn2.weight", 96 } },
        { "block3::bn1::Biases", { "bneck.3.bn2.bias", 96 } },
        { "block3::conv2::Weights", { "bneck.3.conv3.weight", 3840 } },
        { "block3::conv2::Biases", { "bneck.3.conv3.bias", 40 } },
        { "block3::bn2::Weights", { "bneck.3.bn3.weight", 40 } },
        { "block3::bn2::Biases", { "bneck.3.bn3.bias", 40 } },
        { "se2::conv1::Weights", { "bneck.4.se.se.1.weight", 400 } },
        { "se2::conv1::Biases", { "bneck.4.se.se.1.bias", 10 } },
        { "se2::bn1::Weights", { "bneck.4.se.se.2.weight", 10 } },
        { "se2::bn1::Biases", { "bneck.4.se.se.2.bias", 10 } },
        { "se2::conv2::Weights", { "bneck.4.se.se.4.weight", 400 } },
        { "se2::conv2::Biases", { "bneck.4.se.se.4.bias", 40 } },
        { "se2::bn2::Weights", { "bneck.4.se.se.5.weight", 40 } },
        { "se2::bn2::Biases", { "bneck.4.se.se.5.bias", 40 } },
        { "block4::conv0::Weights", { "bneck.4.conv1.weight", 9600 } },
        { "block4::conv0::Biases", { "bneck.4.conv1.bias", 240 } },
        { "block4::bn0::Weights", { "bneck.4.bn1.weight", 240 } },
        { "block4::bn0::Biases", { "bneck.4.bn1.bias", 240 } },
        { "block4::conv1::Weights", { "bneck.4.conv2.weight", 6000 } },
        { "block4::conv1::Biases", { "bneck.4.conv2.bias", 240 } },
        { "block4::bn1::Weights", { "bneck.4.bn2.weight", 240 } },
        { "block4::bn1::Biases", { "bneck.4.bn2.bias", 240 } },
        { "block4::conv2::Weights", { "bneck.4.conv3.weight", 9600 } },
        { "block4::conv2::Biases", { "bneck.4.conv3.bias", 40 } },
        { "block4::bn2::Weights", { "bneck.4.bn3.weight", 40 } },
        { "block4::bn2::Biases", { "bneck.4.bn3.bias", 40 } },
        { "se3::conv1::Weights", { "bneck.5.se.se.1.weight", 400 } },
        { "se3::conv1::Biases", { "bneck.5.se.se.1.bias", 10 } },
        { "se3::bn1::Weights", { "bneck.5.se.se.2.weight", 10 } },
        { "se3::bn1::Biases", { "bneck.5.se.se.2.bias", 10 } },
        { "se3::conv2::Weights", { "bneck.5.se.se.4.weight", 400 } },
        { "se3::conv2::Biases", { "bneck.5.se.se.4.bias", 40 } },
        { "se3::bn2::Weights", { "bneck.5.se.se.5.weight", 40 } },
        { "se3::bn2::Biases", { "bneck.5.se.se.5.bias", 40 } },
        { "block5::conv0::Weights", { "bneck.5.conv1.weight", 9600 } },
        { "block5::conv0::Biases", { "bneck.5.conv1.bias", 240 } },
        { "block5::bn0::Weights", { "bneck.5.bn1.weight", 240 } },
        { "block5::bn0::Biases", { "bneck.5.bn1.bias", 240 } },
        { "block5::conv1::Weights", { "bneck.5.conv2.weight", 6000 } },
        { "block5::conv1::Biases", { "bneck.5.conv2.bias", 240 } },
        { "block5::bn1::Weights", { "bneck.5.bn2.weight", 240 } },
        { "block5::bn1::Biases", { "bneck.5.bn2.bias", 240 } },
        { "block5::conv2::Weights", { "bneck.5.conv3.weight", 9600 } },
        { "block5::conv2::Biases", { "bneck.5.conv3.bias", 40 } },
        { "block5::bn2::Weights", { "bneck.5.bn3.weight", 40 } },
        { "block5::bn2::Biases", { "bneck.5.bn3.bias", 40 } },
        { "se4::conv1::Weights", { "bneck.6.se.se.1.weight", 576 } },
        { "se4::conv1::Biases", { "bneck.6.se.se.1.bias", 12 } },
        { "se4::bn1::Weights", { "bneck.6.se.se.2.weight", 12 } },
        { "se4::bn1::Biases", { "bneck.6.se.se.2.bias", 12 } },
        { "se4::conv2::Weights", { "bneck.6.se.se.4.weight", 576 } },
        { "se4::conv2::Biases", { "bneck.6.se.se.4.bias", 48 } },
        { "se4::bn2::Weights", { "bneck.6.se.se.5.weight", 48 } },
        { "se4::bn2::Biases", { "bneck.6.se.se.5.bias", 48 } },
        { "block6::conv0::Weights", { "bneck.6.conv1.weight", 4800 } },
        { "block6::conv0::Biases", { "bneck.6.conv1.bias", 120 } },
        { "block6::bn0::Weights", { "bneck.6.bn1.weight", 120 } },
        { "block6::bn0::Biases", { "bneck.6.bn1.bias", 120 } },
        { "block6::conv1::Weights", { "bneck.6.conv2.weight", 3000 } },
        { "block6::conv1::Biases", { "bneck.6.conv2.bias", 120 } },
        { "block6::bn1::Weights", { "bneck.6.bn2.weight", 120 } },
        { "block6::bn1::Biases", { "bneck.6.bn2.bias", 120 } },
        { "block6::conv2::Weights", { "bneck.6.conv3.weight", 5760 } },
        { "block6::conv2::Biases", { "bneck.6.conv3.bias", 48 } },
        { "block6::bn2::Weights", { "bneck.6.bn3.weight", 48 } },
        { "block6::bn2::Biases", { "bneck.6.bn3.bias", 48 } },
        { "block6::conv3::Weights", { "bneck.6.shortcut.0.weight", 1920 } },
        { "block6::conv3::Biases", { "bneck.6.shortcut.0.bias", 48 } },
        { "block6::bn3::Weights", { "bneck.6.shortcut.1.weight", 48 } },
        { "block6::bn3::Biases", { "bneck.6.shortcut.1.bias", 48 } },
        { "se5::conv1::Weights", { "bneck.7.se.se.1.weight", 576 } },
        { "se5::conv1::Biases", { "bneck.7.se.se.1.bias", 12 } },
        { "se5::bn1::Weights", { "bneck.7.se.se.2.weight", 12 } },
        { "se5::bn1::Biases", { "bneck.7.se.se.2.bias", 12 } },
        { "se5::conv2::Weights", { "bneck.7.se.se.4.weight", 576 } },
        { "se5::conv2::Biases", { "bneck.7.se.se.4.bias", 48 } },
        { "se5::bn2::Weights", { "bneck.7.se.se.5.weight", 48 } },
        { "se5::bn2::Biases", { "bneck.7.se.se.5.bias", 48 } },
        { "block7::conv0::Weights", { "bneck.7.conv1.weight", 6912 } },
        { "block7::conv0::Biases", { "bneck.7.conv1.bias", 144 } },
        { "block7::bn0::Weights", { "bneck.7.bn1.weight", 144 } },
        { "block7::bn0::Biases", { "bneck.7.bn1.bias", 144 } },
        { "block7::conv1::Weights", { "bneck.7.conv2.weight", 3600 } },
        { "block7::conv1::Biases", { "bneck.7.conv2.bias", 144 } },
        { "block7::bn1::Weights", { "bneck.7.bn2.weight", 144 } },
        { "block7::bn1::Biases", { "bneck.7.bn2.bias", 144 } },
        { "block7::conv2::Weights", { "bneck.7.conv3.weight", 6912 } },
        { "block7::conv2::Biases", { "bneck.7.conv3.bias", 48 } },
        { "block7::bn2::Weights", { "bneck.7.bn3.weight", 48 } },
        { "block7::bn2::Biases", { "bneck.7.bn3.bias", 48 } },
        { "se6::conv1::Weights", { "bneck.8.se.se.1.weight", 2304 } },
        { "se6::conv1::Biases", { "bneck.8.se.se.1.bias", 24 } },
        { "se6::bn1::Weights", { "bneck.8.se.se.2.weight", 24 } },
        { "se6::bn1::Biases", { "bneck.8.se.se.2.bias", 24 } },
        { "se6::conv2::Weights", { "bneck.8.se.se.4.weight", 2304 } },
        { "se6::conv2::Biases", { "bneck.8.se.se.4.bias", 96 } },
        { "se6::bn2::Weights", { "bneck.8.se.se.5.weight", 96 } },
        { "se6::bn2::Biases", { "bneck.8.se.se.5.bias", 96 } },
        { "block8::conv0::Weights", { "bneck.8.conv1.weight", 13824 } },
        { "block8::conv0::Biases", { "bneck.8.conv1.bias", 288 } },
        { "block8::bn0::Weights", { "bneck.8.bn1.weight", 288 } },
        { "block8::bn0::Biases", { "bneck.8.bn1.bias", 288 } },
        { "block8::conv1::Weights", { "bneck.8.conv2.weight", 7200 } },
        { "block8::conv1::Biases", { "bneck.8.conv2.bias", 288 } },
        { "block8::bn1::Weights", { "bneck.8.bn2.weight", 288 } },
        { "block8::bn1::Biases", { "bneck.8.bn2.bias", 288 } },
        { "block8::conv2::Weights", { "bneck.8.conv3.weight", 27648 } },
        { "block8::conv2::Biases", { "bneck.8.conv3.bias", 96 } },
        { "block8::bn2::Weights", { "bneck.8.bn3.weight", 96 } },
        { "block8::bn2::Biases", { "bneck.8.bn3.bias", 96 } },
        { "se7::conv1::Weights", { "bneck.9.se.se.1.weight", 2304 } },
        { "se7::conv1::Biases", { "bneck.9.se.se.1.bias", 24 } },
        { "se7::bn1::Weights", { "bneck.9.se.se.2.weight", 24 } },
        { "se7::bn1::Biases", { "bneck.9.se.se.2.bias", 24 } },
        { "se7::conv2::Weights", { "bneck.9.se.se.4.weight", 2304 } },
        { "se7::conv2::Biases", { "bneck.9.se.se.4.bias", 96 } },
        { "se7::bn2::Weights", { "bneck.9.se.se.5.weight", 96 } },
        { "se7::bn2::Biases", { "bneck.9.se.se.5.bias", 96 } },
        { "block9::conv0::Weights", { "bneck.9.conv1.weight", 55296 } },
        { "block9::conv0::Biases", { "bneck.9.conv1.bias", 576 } },
        { "block9::bn0::Weights", { "bneck.9.bn1.weight", 576 } },
        { "block9::bn0::Biases", { "bneck.9.bn1.bias", 576 } },
        { "block9::conv1::Weights", { "bneck.9.conv2.weight", 14400 } },
        { "block9::conv1::Biases", { "bneck.9.conv2.bias", 576 } },
        { "block9::bn1::Weights", { "bneck.9.bn2.weight", 576 } },
        { "block9::bn1::Biases", { "bneck.9.bn2.bias", 576 } },
        { "block9::conv2::Weights", { "bneck.9.conv3.weight", 55296 } },
        { "block9::conv2::Biases", { "bneck.9.conv3.bias", 96 } },
        { "block9::bn2::Weights", { "bneck.9.bn3.weight", 96 } },
        { "block9::bn2::Biases", { "bneck.9.bn3.bias", 96 } },
        { "se8::conv1::Weights", { "bneck.10.se.se.1.weight", 2304 } },
        { "se8::conv1::Biases", { "bneck.10.se.se.1.bias", 24 } },
        { "se8::bn1::Weights", { "bneck.10.se.se.2.weight", 24 } },
        { "se8::bn1::Biases", { "bneck.10.se.se.2.bias", 24 } },
        { "se8::conv2::Weights", { "bneck.10.se.se.4.weight", 2304 } },
        { "se8::conv2::Biases", { "bneck.10.se.se.4.bias", 96 } },
        { "se8::bn2::Weights", { "bneck.10.se.se.5.weight", 96 } },
        { "se8::bn2::Biases", { "bneck.10.se.se.5.bias", 96 } },
        { "block10::conv0::Weights", { "bneck.10.conv1.weight", 55296 } },
        { "block10::conv0::Biases", { "bneck.10.conv1.bias", 576 } },
        { "block10::bn0::Weights", { "bneck.10.bn1.weight", 576 } },
        { "block10::bn0::Biases", { "bneck.10.bn1.bias", 576 } },
        { "block10::conv1::Weights", { "bneck.10.conv2.weight", 14400 } },
        { "block10::conv1::Biases", { "bneck.10.conv2.bias", 576 } },
        { "block10::bn1::Weights", { "bneck.10.bn2.weight", 576 } },
        { "block10::bn1::Biases", { "bneck.10.bn2.bias", 576 } },
        { "block10::conv2::Weights", { "bneck.10.conv3.weight", 55296 } },
        { "block10::conv2::Biases", { "bneck.10.conv3.bias", 96 } },
        { "block10::bn2::Weights", { "bneck.10.bn3.weight", 96 } },
        { "block10::bn2::Biases", { "bneck.10.bn3.bias", 96 } },
        { "output::conv0::Weights", { "conv2.weight", 55296 } },
        { "output::conv0::Biases", { "conv2.bias", 576 } },
        { "output::bn0::Weights", { "bn2.weight", 576 } },
        { "output::bn0::Biases", { "bn2.bias", 576 } },
        { "output::fc0::Weights", { "linear3.weight", 589824 } },
        { "output::fc0::Biases", { "linear3.bias", 1024 } },
        { "output::fc1::Weights", { "linear4.weight", 10240 } },
        { "output::fc1::Biases", { "linear4.bias", 10 } },
        { "input::bn0::MeanEval", { "bn1.running_mean", 16 } },
        { "input::bn0::VarianceEval", { "bn1.running_var", 16 } },
        { "se0::bn1::MeanEval", { "bneck.0.se.se.2.running_mean", 4 } },
        { "se0::bn1::VarianceEval", { "bneck.0.se.se.2.running_var", 4 } },
        { "se0::bn2::MeanEval", { "bneck.0.se.se.5.running_mean", 16 } },
        { "se0::bn2::VarianceEval", { "bneck.0.se.se.5.running_var", 16 } },
        { "block0::bn0::MeanEval", { "bneck.0.bn1.running_mean", 16 } },
        { "block0::bn0::VarianceEval", { "bneck.0.bn1.running_var", 16 } },
        { "block0::bn1::MeanEval", { "bneck.0.bn2.running_mean", 16 } },
        { "block0::bn1::VarianceEval", { "bneck.0.bn2.running_var", 16 } },
        { "block0::bn2::MeanEval", { "bneck.0.bn3.running_mean", 16 } },
        { "block0::bn2::VarianceEval", { "bneck.0.bn3.running_var", 16 } },
        { "block1::bn0::MeanEval", { "bneck.1.bn1.running_mean", 72 } },
        { "block1::bn0::VarianceEval", { "bneck.1.bn1.running_var", 72 } },
        { "block1::bn1::MeanEval", { "bneck.1.bn2.running_mean", 72 } },
        { "block1::bn1::VarianceEval", { "bneck.1.bn2.running_var", 72 } },
        { "block1::bn2::MeanEval", { "bneck.1.bn3.running_mean", 24 } },
        { "block1::bn2::VarianceEval", { "bneck.1.bn3.running_var", 24 } },
        { "block2::bn0::MeanEval", { "bneck.2.bn1.running_mean", 88 } },
        { "block2::bn0::VarianceEval", { "bneck.2.bn1.running_var", 88 } },
        { "block2::bn1::MeanEval", { "bneck.2.bn2.running_mean", 88 } },
        { "block2::bn1::VarianceEval", { "bneck.2.bn2.running_var", 88 } },
        { "block2::bn2::MeanEval", { "bneck.2.bn3.running_mean", 24 } },
        { "block2::bn2::VarianceEval", { "bneck.2.bn3.running_var", 24 } },
        { "se1::bn1::MeanEval", { "bneck.3.se.se.2.running_mean", 10 } },
        { "se1::bn1::VarianceEval", { "bneck.3.se.se.2.running_var", 10 } },
        { "se1::bn2::MeanEval", { "bneck.3.se.se.5.running_mean", 40 } },
        { "se1::bn2::VarianceEval", { "bneck.3.se.se.5.running_var", 40 } },
        { "block3::bn0::MeanEval", { "bneck.3.bn1.running_mean", 96 } },
        { "block3::bn0::VarianceEval", { "bneck.3.bn1.running_var", 96 } },
        { "block3::bn1::MeanEval", { "bneck.3.bn2.running_mean", 96 } },
        { "block3::bn1::VarianceEval", { "bneck.3.bn2.running_var", 96 } },
        { "block3::bn2::MeanEval", { "bneck.3.bn3.running_mean", 40 } },
        { "block3::bn2::VarianceEval", { "bneck.3.bn3.running_var", 40 } },
        { "se2::bn1::MeanEval", { "bneck.4.se.se.2.running_mean", 10 } },
        { "se2::bn1::VarianceEval", { "bneck.4.se.se.2.running_var", 10 } },
        { "se2::bn2::MeanEval", { "bneck.4.se.se.5.running_mean", 40 } },
        { "se2::bn2::VarianceEval", { "bneck.4.se.se.5.running_var", 40 } },
        { "block4::bn0::MeanEval", { "bneck.4.bn1.running_mean", 240 } },
        { "block4::bn0::VarianceEval", { "bneck.4.bn1.running_var", 240 } },
        { "block4::bn1::MeanEval", { "bneck.4.bn2.running_mean", 240 } },
        { "block4::bn1::VarianceEval", { "bneck.4.bn2.running_var", 240 } },
        { "block4::bn2::MeanEval", { "bneck.4.bn3.running_mean", 40 } },
        { "block4::bn2::VarianceEval", { "bneck.4.bn3.running_var", 40 } },
        { "se3::bn1::MeanEval", { "bneck.5.se.se.2.running_mean", 10 } },
        { "se3::bn1::VarianceEval", { "bneck.5.se.se.2.running_var", 10 } },
        { "se3::bn2::MeanEval", { "bneck.5.se.se.5.running_mean", 40 } },
        { "se3::bn2::VarianceEval", { "bneck.5.se.se.5.running_var", 40 } },
        { "block5::bn0::MeanEval", { "bneck.5.bn1.running_mean", 240 } },
        { "block5::bn0::VarianceEval", { "bneck.5.bn1.running_var", 240 } },
        { "block5::bn1::MeanEval", { "bneck.5.bn2.running_mean", 240 } },
        { "block5::bn1::VarianceEval", { "bneck.5.bn2.running_var", 240 } },
        { "block5::bn2::MeanEval", { "bneck.5.bn3.running_mean", 40 } },
        { "block5::bn2::VarianceEval", { "bneck.5.bn3.running_var", 40 } },
        { "se4::bn1::MeanEval", { "bneck.6.se.se.2.running_mean", 12 } },
        { "se4::bn1::VarianceEval", { "bneck.6.se.se.2.running_var", 12 } },
        { "se4::bn2::MeanEval", { "bneck.6.se.se.5.running_mean", 48 } },
        { "se4::bn2::VarianceEval", { "bneck.6.se.se.5.running_var", 48 } },
        { "block6::bn0::MeanEval", { "bneck.6.bn1.running_mean", 120 } },
        { "block6::bn0::VarianceEval", { "bneck.6.bn1.running_var", 120 } },
        { "block6::bn1::MeanEval", { "bneck.6.bn2.running_mean", 120 } },
        { "block6::bn1::VarianceEval", { "bneck.6.bn2.running_var", 120 } },
        { "block6::bn2::MeanEval", { "bneck.6.bn3.running_mean", 48 } },
        { "block6::bn2::VarianceEval", { "bneck.6.bn3.running_var", 48 } },
        { "block6::bn3::MeanEval", { "bneck.6.shortcut.1.running_mean", 48 } },
        { "block6::bn3::VarianceEval", { "bneck.6.shortcut.1.running_var", 48 } },
        { "se5::bn1::MeanEval", { "bneck.7.se.se.2.running_mean", 12 } },
        { "se5::bn1::VarianceEval", { "bneck.7.se.se.2.running_var", 12 } },
        { "se5::bn2::MeanEval", { "bneck.7.se.se.5.running_mean", 48 } },
        { "se5::bn2::VarianceEval", { "bneck.7.se.se.5.running_var", 48 } },
        { "block7::bn0::MeanEval", { "bneck.7.bn1.running_mean", 144 } },
        { "block7::bn0::VarianceEval", { "bneck.7.bn1.running_var", 144 } },
        { "block7::bn1::MeanEval", { "bneck.7.bn2.running_mean", 144 } },
        { "block7::bn1::VarianceEval", { "bneck.7.bn2.running_var", 144 } },
        { "block7::bn2::MeanEval", { "bneck.7.bn3.running_mean", 48 } },
        { "block7::bn2::VarianceEval", { "bneck.7.bn3.running_var", 48 } },
        { "se6::bn1::MeanEval", { "bneck.8.se.se.2.running_mean", 24 } },
        { "se6::bn1::VarianceEval", { "bneck.8.se.se.2.running_var", 24 } },
        { "se6::bn2::MeanEval", { "bneck.8.se.se.5.running_mean", 96 } },
        { "se6::bn2::VarianceEval", { "bneck.8.se.se.5.running_var", 96 } },
        { "block8::bn0::MeanEval", { "bneck.8.bn1.running_mean", 288 } },
        { "block8::bn0::VarianceEval", { "bneck.8.bn1.running_var", 288 } },
        { "block8::bn1::MeanEval", { "bneck.8.bn2.running_mean", 288 } },
        { "block8::bn1::VarianceEval", { "bneck.8.bn2.running_var", 288 } },
        { "block8::bn2::MeanEval", { "bneck.8.bn3.running_mean", 96 } },
        { "block8::bn2::VarianceEval", { "bneck.8.bn3.running_var", 96 } },
        { "se7::bn1::MeanEval", { "bneck.9.se.se.2.running_mean", 24 } },
        { "se7::bn1::VarianceEval", { "bneck.9.se.se.2.running_var", 24 } },
        { "se7::bn2::MeanEval", { "bneck.9.se.se.5.running_mean", 96 } },
        { "se7::bn2::VarianceEval", { "bneck.9.se.se.5.running_var", 96 } },
        { "block9::bn0::MeanEval", { "bneck.9.bn1.running_mean", 576 } },
        { "block9::bn0::VarianceEval", { "bneck.9.bn1.running_var", 576 } },
        { "block9::bn1::MeanEval", { "bneck.9.bn2.running_mean", 576 } },
        { "block9::bn1::VarianceEval", { "bneck.9.bn2.running_var", 576 } },
        { "block9::bn2::MeanEval", { "bneck.9.bn3.running_mean", 96 } },
        { "block9::bn2::VarianceEval", { "bneck.9.bn3.running_var", 96 } },
        { "se8::bn1::MeanEval", { "bneck.10.se.se.2.running_mean", 24 } },
        { "se8::bn1::VarianceEval", { "bneck.10.se.se.2.running_var", 24 } },
        { "se8::bn2::MeanEval", { "bneck.10.se.se.5.running_mean", 96 } },
        { "se8::bn2::VarianceEval", { "bneck.10.se.se.5.running_var", 96 } },
        { "block10::bn0::MeanEval", { "bneck.10.bn1.running_mean", 576 } },
        { "block10::bn0::VarianceEval", { "bneck.10.bn1.running_var", 576 } },
        { "block10::bn1::MeanEval", { "bneck.10.bn2.running_mean", 576 } },
        { "block10::bn1::VarianceEval", { "bneck.10.bn2.running_var", 576 } },
        { "block10::bn2::MeanEval", { "bneck.10.bn3.running_mean", 96 } },
        { "block10::bn2::VarianceEval", { "bneck.10.bn3.running_var", 96 } },
        { "output::bn0::MeanEval", { "bn2.running_mean", 576 } },
        { "output::bn0::VarianceEval", { "bn2.running_var", 576 } },
    };

    raul::DataLoader dataLoader;
    std::cout << "Loading weights...";
    for (auto [tensor, value] : pytorch_name_map)
    {
        const auto [filename, size] = value;
        memory_manager[tensor] = dataLoader.loadData(weights_path / (file_prefix + filename + ".data"), size, 1, 1);
    }

    if (memory_manager.tensorExists("output::fc0::Weights")) raul::Common::transpose(memory_manager["output::fc0::Weights"], 1024);
    if (memory_manager.tensorExists("output::fc1::Weights")) raul::Common::transpose(memory_manager["output::fc1::Weights"], 10);

    std::cout << "done" << std::endl;
}

TEST(TestMobileNetV3, SeBlockBuildingUnit)
{
    PROFILE_TEST
    const size_t block_in_size = 8U;
    // This value is got from pytorch model (mobilenetv3_experiments.ipynb)
    const size_t golden_trainable_parameters = 62U;

    size_t se_block_cnt = 0;

    raul::Workflow work;
    raul::Name input = "data";

    // Build block
    work.add<raul::DataLayer>("data", raul::DataParams{ { input, "labels" }, block_in_size, 1, 1, 0 });
    add_se_block(se_block_cnt, work, input, block_in_size);

    work.preparePipelines();
    work.setBatchSize(1u);
    work.prepareMemoryForTraining();

    work.printInfo(std::cout);

    // Checks
    EXPECT_EQ(tools::get_size_of_trainable_params(work), golden_trainable_parameters);
}

TEST(TestMobileNetV3, BottleneckBlockBuildingNoSEUnit)
{
    PROFILE_TEST
    const size_t image_size = 56U;
    const size_t image_channels = 16U;
    const size_t kernel_size = 3U;
    const size_t expand_channels = 72U;
    const size_t out_channels = 24U;
    const NonLinearType nonlinear = NonLinearType::ReLU;
    const int semodule = -1;
    const size_t stride = 2U;
    const float bnMomentum = 0.1f;

    // This value is got from pytorch model (mobilenetv3_experiments.ipynb)
    const size_t golden_trainable_parameters = 4'032U;

    size_t se_block_cnt = 0;
    size_t bneck_block_cnt = 0;

    raul::Workflow work;
    raul::Name input = "data";

    // Build block
    work.add<raul::DataLayer>("data", raul::DataParams{ { input, "labels" }, image_channels, image_size, image_size, 0 });
    add_mobilenetv3_block(bneck_block_cnt, se_block_cnt, work, input, kernel_size, image_channels, expand_channels, out_channels, nonlinear, semodule, stride, bnMomentum);

    work.preparePipelines();
    work.setBatchSize(1u);
    work.prepareMemoryForTraining();

    work.printInfo(std::cout);

    // Checks
    EXPECT_EQ(tools::get_size_of_trainable_params(work), golden_trainable_parameters);
}

TEST(TestMobileNetV3, BottleneckBlockBuildingSEUnit)
{
    PROFILE_TEST
    const size_t image_size = 14U;
    const size_t image_channels = 40U;
    const size_t kernel_size = 5U;
    const size_t expand_channels = 120U;
    const size_t out_channels = 48U;
    const NonLinearType nonlinear = NonLinearType::HSwish;
    const int semodule = 48;
    const size_t stride = 1U;
    const float bnMomentum = 0.1f;

    // This value is got from pytorch model (mobilenetv3_experiments.ipynb)
    const size_t golden_trainable_parameters = 17'820U;

    size_t se_block_cnt = 0;
    size_t bneck_block_cnt = 0;

    raul::Workflow work;
    raul::Name input = "data";

    // Build block
    work.add<raul::DataLayer>("data", raul::DataParams{ { input, "labels" }, image_channels, image_size, image_size, 0 });
    add_mobilenetv3_block(bneck_block_cnt, se_block_cnt, work, input, kernel_size, image_channels, expand_channels, out_channels, nonlinear, semodule, stride, bnMomentum);

    work.preparePipelines();
    work.setBatchSize(1u);
    work.prepareMemoryForTraining();

    work.printInfo(std::cout);

    // Checks
    EXPECT_EQ(tools::get_size_of_trainable_params(work), golden_trainable_parameters);
}

TEST(TestMobileNetV3, OutputBlockBuildingUnit)
{
    PROFILE_TEST
    const size_t image_size = 7U;
    const size_t image_channels = 96U;
    const float bnMomentum = 0.1f;

    // This value is got from pytorch model (mobilenetv3_experiments.ipynb)
    const size_t golden_trainable_parameters = 658'122U;

    raul::Workflow work;
    raul::Name input = "data";

    // Build block
    work.add<raul::DataLayer>("data", raul::DataParams{ { input, "labels" }, image_channels, image_size, image_size, 0 });
    add_output_block(work, input, bnMomentum);

    work.preparePipelines();
    work.setBatchSize(1u);
    work.prepareMemoryForTraining();

    work.printInfo(std::cout);

    // Checks
    EXPECT_EQ(tools::get_size_of_trainable_params(work), golden_trainable_parameters);
}

TEST(TestMobileNetV3, InputBlockBuildingUnit)
{
    PROFILE_TEST
    const size_t image_size = 7U;
    const size_t image_channels = 3U;
    const float bnMomentum = 0.1f;

    // This value is got from pytorch model (mobilenetv3_experiments.ipynb)
    const size_t golden_trainable_parameters = 480U;

    raul::Workflow work;
    raul::Name input = "data";

    // Build block
    work.add<raul::DataLayer>("data", raul::DataParams{ { input, "labels" }, image_channels, image_size, image_size, 0 });
    add_input_block(work, input, bnMomentum);

    work.preparePipelines();
    work.setBatchSize(1u);
    work.prepareMemoryForTraining();

    work.printInfo(std::cout);

    // Checks
    EXPECT_EQ(tools::get_size_of_trainable_params(work), golden_trainable_parameters);
}

TEST(TestMobileNetV3, SmallNetBuildingUnit)
{
    PROFILE_TEST
    // This value is got from pytorch model (mobilenetv3_experiments.ipynb)
    const size_t golden_trainable_parameters = 1'095'496U;

    raul::Workflow work;
    build_mobilenetv3_small(work, 224U, 3U, 10U);

    work.preparePipelines();
    work.setBatchSize(1u);
    work.prepareMemoryForTraining();

    work.printInfo(std::cout);

    EXPECT_EQ(tools::get_size_of_trainable_params(work), golden_trainable_parameters);
}

TEST(TestMobileNetV3, SeBlockInferenceUnit)
{
    PROFILE_TEST
    constexpr auto eps = 1e-6_dt;
    const size_t block_in_size = 4U;
    const size_t reduction = 4U;
    const raul::dtype in_val = 5.0_dt;
    const size_t batch_size = 2U;

    const raul::Tensor golden_tensor_our{ 2.7205095291137695_dt, 2.248157024383545_dt, 2.3361964225769043_dt, 1.7038803100585938_dt,
                                          2.7205095291137695_dt, 2.248157024383545_dt, 2.3361964225769043_dt, 1.7038803100585938_dt };

    size_t se_block_cnt = 0;

    raul::Workflow work;
    raul::Name input = "data";

    // Build block
    work.add<raul::DataLayer>("data", raul::DataParams{ { input, "labels" }, block_in_size, 1, 1, 0 });
    add_se_block(se_block_cnt, work, input, block_in_size);

    work.preparePipelines();
    work.setBatchSize(batch_size);
    work.prepareMemoryForTraining();

    // Initialization of SE block
    auto& memory_manager = work.getMemoryManager();
    raul::DataLoader dataLoader;

    const auto weights_path = tools::getTestAssetsDir() / "mobilenet_v3" / "se" / "4_seed_0";
    const raul::Name se_prefix = "se0::";
    memory_manager[se_prefix + "conv1::Weights"] = dataLoader.loadData(weights_path / "init_se.1.weight.data", block_in_size, 1, 1);
    memory_manager[se_prefix + "conv1::Biases"] = dataLoader.loadData(weights_path / "init_se.1.bias.data", block_in_size / reduction, 1, 1);
    memory_manager[se_prefix + "bn1::Weights"] = dataLoader.loadData(weights_path / "init_se.2.weight.data", block_in_size / reduction, 1, 1);
    memory_manager[se_prefix + "bn1::Biases"] = dataLoader.loadData(weights_path / "init_se.2.bias.data", block_in_size / reduction, 1, 1);
    memory_manager[se_prefix + "conv2::Weights"] = dataLoader.loadData(weights_path / "init_se.4.weight.data", block_in_size, 1, 1);
    memory_manager[se_prefix + "conv2::Biases"] = dataLoader.loadData(weights_path / "init_se.4.bias.data", block_in_size, 1, 1);
    memory_manager[se_prefix + "bn2::Weights"] = dataLoader.loadData(weights_path / "init_se.5.weight.data", block_in_size, 1, 1);
    memory_manager[se_prefix + "bn2::Biases"] = dataLoader.loadData(weights_path / "init_se.5.bias.data", block_in_size, 1, 1);

    // Initialization in_tensor and out_tensor
    raul::Tensor& in_tensor = memory_manager.getTensor("data");
    const raul::Tensor& out_tensor = memory_manager.getTensor(input);

    std::fill(in_tensor.begin(), in_tensor.end(), in_val);

    // Apply SE block
    work.forwardPassTesting();

    for (size_t q = 0; q < out_tensor.size(); ++q)
    {
        CHECK_NEAR(out_tensor[q], golden_tensor_our[q], eps);
    }
}

TEST(TestMobileNetV3, BottleneckBlockInferenceUnit)
{
    PROFILE_TEST
    constexpr auto eps = 1e-6_dt;
    const raul::dtype in_val = 5.0_dt;
    const size_t image_size = 2U;
    const size_t in_channels = 10U;
    const size_t kernel_size = 5U;
    const size_t expand_channels = 15U;
    const size_t out_channels = 4U;
    const NonLinearType nonlinear = NonLinearType::HSwish;
    const int semodule = out_channels;
    const size_t stride = 1U;
    const float bnMomentum = 0.1f;
    const size_t reduction = 4U;
    const size_t batch_size = 2U;

    size_t se_block_cnt = 0;
    size_t bneck_block_cnt = 0;

    raul::Workflow work;
    raul::Name input = "data";

    // Build block
    work.add<raul::DataLayer>("data", raul::DataParams{ { input, "labels" }, in_channels, image_size, image_size, 0 });
    add_mobilenetv3_block(bneck_block_cnt, se_block_cnt, work, input, kernel_size, in_channels, expand_channels, out_channels, nonlinear, semodule, stride, bnMomentum);

    work.preparePipelines();
    work.setBatchSize(batch_size);
    work.prepareMemoryForTraining();

    const raul::Tensor golden_tensor_out{ -2.6895761489868164_dt, -2.73490047454834_dt,   -2.733654260635376_dt,  -2.7092490196228027_dt, 0.17908255755901337_dt, 0.2319057583808899_dt,
                                          0.23007869720458984_dt, 0.1677435338497162_dt,  -0.8000085353851318_dt, -0.7773728966712952_dt, -0.8467975854873657_dt, -0.801381528377533_dt,
                                          1.9343833923339844_dt,  1.930648922920227_dt,   1.80897855758667_dt,    1.9036997556686401_dt,  -2.6895761489868164_dt, -2.73490047454834_dt,
                                          -2.733654260635376_dt,  -2.7092490196228027_dt, 0.17908255755901337_dt, 0.2319057583808899_dt,  0.23007869720458984_dt, 0.1677435338497162_dt,
                                          -0.8000085353851318_dt, -0.7773728966712952_dt, -0.8467975854873657_dt, -0.801381528377533_dt,  1.9343833923339844_dt,  1.930648922920227_dt,
                                          1.80897855758667_dt,    1.9036997556686401_dt };

    // Initialization of Bottleneck block
    auto& memory_manager = work.getMemoryManager();
    const auto weights_path = tools::getTestAssetsDir() / "mobilenet_v3" / "bneck" / "im.2_in.10_ker.5_exp.15_out.4_s.1_seed_0";
    const raul::Name bneck_prefix = "block0::";

    raul::DataLoader dataLoader;

    // Convolution2D [block0::conv0]
    memory_manager[bneck_prefix + "conv0::Weights"] = dataLoader.loadData(weights_path / "init_conv1.weight.data", in_channels * expand_channels, 1, 1);
    memory_manager[bneck_prefix + "conv0::Biases"] = dataLoader.loadData(weights_path / "init_conv1.bias.data", expand_channels, 1, 1);

    // BatchNorm [block0::bn0]
    memory_manager[bneck_prefix + "bn0::Weights"] = dataLoader.loadData(weights_path / "init_bn1.weight.data", expand_channels, 1, 1);
    memory_manager[bneck_prefix + "bn0::Biases"] = dataLoader.loadData(weights_path / "init_bn1.bias.data", expand_channels, 1, 1);

    // ConvolutionDepthwise2D [block0::conv1]
    memory_manager[bneck_prefix + "conv1::Weights"] = dataLoader.loadData(weights_path / "init_conv2.weight.data", expand_channels * kernel_size * kernel_size, 1, 1);
    memory_manager[bneck_prefix + "conv1::Biases"] = dataLoader.loadData(weights_path / "init_conv2.bias.data", expand_channels, 1, 1);

    // BatchNorm [block0::bn1]
    memory_manager[bneck_prefix + "bn1::Weights"] = dataLoader.loadData(weights_path / "init_bn2.weight.data", expand_channels, 1, 1);
    memory_manager[bneck_prefix + "bn1::Biases"] = dataLoader.loadData(weights_path / "init_bn2.bias.data", expand_channels, 1, 1);

    // Convolution2D [block0::conv2]
    memory_manager[bneck_prefix + "conv2::Weights"] = dataLoader.loadData(weights_path / "init_conv3.weight.data", expand_channels * out_channels, 1, 1);
    memory_manager[bneck_prefix + "conv2::Biases"] = dataLoader.loadData(weights_path / "init_conv3.bias.data", out_channels, 1, 1);

    // BatchNorm [block0::bn2]
    memory_manager[bneck_prefix + "bn2::Weights"] = dataLoader.loadData(weights_path / "init_bn3.weight.data", out_channels, 1, 1);
    memory_manager[bneck_prefix + "bn2::Biases"] = dataLoader.loadData(weights_path / "init_bn3.bias.data", out_channels, 1, 1);

    // SE block
    if (se_block_cnt)
    {
        const raul::Name se_prefix = "se0::";
        // Convolution2D [se0::conv1]
        memory_manager[se_prefix + "conv1::Weights"] = dataLoader.loadData(weights_path / "init_se.se.1.weight.data", out_channels, 1, 1);
        memory_manager[se_prefix + "conv1::Biases"] = dataLoader.loadData(weights_path / "init_se.se.1.bias.data", out_channels / reduction, 1, 1);

        // BatchNorm [se0::bn1]
        memory_manager[se_prefix + "bn1::Weights"] = dataLoader.loadData(weights_path / "init_se.se.2.weight.data", out_channels / reduction, 1, 1);
        memory_manager[se_prefix + "bn1::Biases"] = dataLoader.loadData(weights_path / "init_se.se.2.bias.data", out_channels / reduction, 1, 1);

        // Convolution2D [se0::conv2]
        memory_manager[se_prefix + "conv2::Weights"] = dataLoader.loadData(weights_path / "init_se.se.4.weight.data", out_channels, 1, 1);
        memory_manager[se_prefix + "conv2::Biases"] = dataLoader.loadData(weights_path / "init_se.se.4.bias.data", out_channels, 1, 1);

        // BatchNorm [se0::bn2]
        memory_manager[se_prefix + "bn2::Weights"] = dataLoader.loadData(weights_path / "init_se.se.5.weight.data", out_channels, 1, 1);
        memory_manager[se_prefix + "bn2::Biases"] = dataLoader.loadData(weights_path / "init_se.se.5.bias.data", out_channels, 1, 1);
    }
    // Shortcut
    // Convolution2D [block0::conv3]
    memory_manager[bneck_prefix + "conv3::Weights"] = dataLoader.loadData(weights_path / "init_shortcut.0.weight.data", in_channels * out_channels, 1, 1);
    memory_manager[bneck_prefix + "conv3::Biases"] = dataLoader.loadData(weights_path / "init_shortcut.0.bias.data", out_channels, 1, 1);

    // BatchNorm [block0::bn3]
    memory_manager[bneck_prefix + "bn3::Weights"] = dataLoader.loadData(weights_path / "init_shortcut.1.weight.data", out_channels, 1, 1);
    memory_manager[bneck_prefix + "bn3::Biases"] = dataLoader.loadData(weights_path / "init_shortcut.1.bias.data", out_channels, 1, 1);

    // Initialization in_tensor and out_tensor
    raul::Tensor& in_tensor = memory_manager.getTensor("data");
    const raul::Tensor& out_tensor = memory_manager.getTensor(input);

    std::fill(in_tensor.begin(), in_tensor.end(), in_val);

    // Apply Bottleneck block
    work.forwardPassTesting();

    for (size_t q = 0; q < out_tensor.size(); ++q)
    {
        CHECK_NEAR(out_tensor[q], golden_tensor_out[q], eps);
    }
}

TEST(TestMobileNetV3, Inference)
{
    PROFILE_TEST
    constexpr auto relative_eps = 1e-3_dt;
    const size_t image_size = 224U;
    const size_t image_channels = 3U;
    const size_t image_classes = 10U;
    const size_t batch_size = 1U;
    const raul::dtype in_val = 5.0_dt;

    const raul::Tensor golden_tensor_out{ 5.78190828548486e-09_dt,   3.2652163195479034e-09_dt,  -6.43653352838669e-10_dt,  2.140289723229216e-09_dt,  -2.9097784182141595e-09_dt,
                                          -7.537933055523638e-10_dt, -3.2835432151046007e-09_dt, 3.1988978133057344e-09_dt, -1.241775682281343e-09_dt, -2.4757718097845327e-10_dt };

    // Build work
    raul::Workflow work;
    const auto output_name = build_mobilenetv3_small(work, image_size, image_channels, image_classes);

    work.preparePipelines();
    work.setBatchSize(batch_size);
    work.prepareMemoryForTraining();

    // Initialization of the work
    auto& memory_manager = work.getMemoryManager();
    const auto weights_path = tools::getTestAssetsDir() / "mobilenet_v3" / "small" / "seed_0";
    load_mobilenetv3_small_weights(memory_manager, weights_path);

    // Initialization in_tensor and out_tensor
    raul::Tensor& in_tensor = memory_manager.getTensor("data");
    const raul::Tensor& out_tensor = memory_manager.getTensor(output_name);

    std::fill(in_tensor.begin(), in_tensor.end(), in_val);

    // Apply MobilenetV3
    std::cout << "Forward...";
    work.forwardPassTesting();
    std::cout << "done" << std::endl;

    for (size_t q = 0; q < out_tensor.size(); ++q)
    {
        std::cout << out_tensor[q] << " == " << golden_tensor_out[q] << std::endl;
        ASSERT_TRUE(tools::expect_near_relative(out_tensor[q], golden_tensor_out[q], relative_eps));
    }
}

TEST(TestMobileNetV3, CifarPretrainedModelOneEpochTrain)
{
    PROFILE_TEST
    constexpr auto acc_eps = 1.0_dt;
    constexpr auto lr = 0.05_dt;
    const size_t image_size = 224U;
    const size_t image_channels = 3U;
    const size_t image_classes = 10U;
    const size_t batch_size = 50U;
    const size_t print_freq = 100U;

    bool useCheckpointing = false;
    bool usePool = false;

    const auto golden_acc_before = 80.43_dt;
    const auto golden_acc_after = 82.68_dt;

    // Build work
    raul::Workflow work(raul::CompressionMode::NONE, raul::CalculationMode::DETERMINISTIC, usePool ? raul::AllocationMode::POOL : raul::AllocationMode::STANDARD);
    const auto output_name = build_mobilenetv3_small(work, image_size, image_channels, image_classes);
    work.add<raul::LogSoftMaxActivation>("softmax", raul::BasicParamsWithDim{ { output_name }, { "softmax" } });
    work.add<raul::NLLLoss>("loss", raul::LossParams{ { "softmax", "labels" }, { "loss" }, "batch_mean" });

    if (useCheckpointing)
    {
        raul::Names checkpointsAll = work.getPotentialCheckpoints();
        raul::Names checkpoints;

        raul::Names blocks = { "block3", "block6", "block9" };
        std::vector<bool> isBlock(blocks.size(), false);

        for (raul::Name& checkP : checkpointsAll)
        {
            for (size_t q = 0; q < blocks.size(); ++q)
            {
                if (!isBlock[q] && checkP.str().find(blocks[q]) != std::string::npos)
                {
                    isBlock[q] = true;
                    checkpoints.push_back(checkP);
                }
            }
        }

        work.setCheckpoints(checkpoints);

        // work.setCheckpoints({"block3::conv0", "block6::conv0", "block9::conv0"});

        work.preparePipelines(raul::Workflow::Execution::Checkpointed);
    }
    else
    {
        work.preparePipelines();
    }
    work.setBatchSize(batch_size);
    work.prepareMemoryForTraining();

    // Initialization of the work
    auto& memory_manager = work.getMemoryManager();
    const auto weights_path = tools::getTestAssetsDir() / "mobilenet_v3" / "small" / "seed_0_epoch20_acc_80.43";
    load_mobilenetv3_small_weights(memory_manager, weights_path, "80.43_mobilenetv3.");

    // Dataset initialization
    raul::CIFAR10 cifar;
    std::cout << "Loading CIFAR...";
    ASSERT_TRUE(cifar.loadingData(tools::getTestAssetsDir() / "CIFAR"));
    std::cout << "done" << std::endl;

    printf("Memory taken = %.2fMB \n\n", static_cast<float>(work.getMemoryManager().getTotalMemory()) / (1024.0f * 1024.0f));

    // Before train
    {
        std::cout << "Calculating acc..." << std::endl;
        raul::dtype testAcc = cifar.testNetwork(work, std::make_pair(224, 224));
        printf("Testing taken = %.3fs \n", cifar.getTestingTime());
        printf("Test accuracy = %.2f\n", testAcc);
        EXPECT_NEAR(testAcc, golden_acc_before, acc_eps);
    }
    // Trainings
    const size_t stepsAmountTrain = cifar.getTrainImageAmount() / batch_size;
    auto sgd = std::make_shared<raul::optimizers::SGD>(lr);
    std::cout << "Training..." << std::endl;

    for (size_t epoch = 1; epoch <= 1; ++epoch)
    {
        printf("Epoch = %zu\n", epoch);

        for (size_t q = 0; q < stepsAmountTrain; ++q)
        {
            raul::dtype testLoss = cifar.oneTrainIteration(work, sgd.get(), q, std::make_pair(224, 224));
            if (q % print_freq == 0)
            {
                // TODO(ck): review loss difference
                printf("iteration = %zu/%zu, loss = %f\n", q, stepsAmountTrain, testLoss);
            }
        }

        printf("Epoch Training taken = %.3fs \n", cifar.getTrainingTime());

        // After train
        {
            std::cout << "Calculating acc..." << std::endl;
            raul::dtype testAcc = cifar.testNetwork(work, std::make_pair(224, 224));
            printf("Test accuracy = %.2f\n", testAcc);
            EXPECT_NEAR(testAcc, golden_acc_after, acc_eps);
        }
    }

    printf("Memory taken = %.2fMB \n\n", static_cast<float>(work.getMemoryManager().getTotalMemory()) / (1024.0f * 1024.0f));
}

TEST(TestMobileNetV3, CifarPretrainedModelCheckpointsTrain)
{
    PROFILE_TEST
    constexpr auto lr = 0.05_dt;
    const size_t image_size = 224U;
    const size_t image_channels = 3U;
    const size_t image_classes = 10U;
    const size_t batch_size = 50U;

    // Build work
    raul::Workflow work(raul::CompressionMode::NONE, raul::CalculationMode::DETERMINISTIC, raul::AllocationMode::STANDARD);
    const auto output_name = build_mobilenetv3_small(work, image_size, image_channels, image_classes);
    work.add<raul::LogSoftMaxActivation>("softmax", raul::BasicParamsWithDim{ { output_name }, { "softmax" } });
    work.add<raul::NLLLoss>("loss", raul::LossParams{ { "softmax", "labels" }, { "loss" }, "batch_mean" });

    raul::Names checkpointsAll = work.getPotentialCheckpoints();
    raul::Names checkpoints;

    raul::Names blocks = { "block3", "block6", "block9" };
    std::vector<bool> isBlock(blocks.size(), false);

    for (raul::Name& checkP : checkpointsAll)
    {
        for (size_t q = 0; q < blocks.size(); ++q)
        {
            if (!isBlock[q] && checkP.str().find(blocks[q]) != std::string::npos)
            {
                isBlock[q] = true;
                checkpoints.push_back(checkP);
            }
        }
    }

    work.setCheckpoints(checkpoints);

    work.preparePipelines(raul::Workflow::Execution::Checkpointed);

    work.setBatchSize(batch_size);
    work.prepareMemoryForTraining();

    // Initialization of the work
    auto& memory_manager = work.getMemoryManager();
    const auto weights_path = tools::getTestAssetsDir() / "mobilenet_v3" / "small" / "seed_0_epoch20_acc_80.43";
    load_mobilenetv3_small_weights(memory_manager, weights_path, "80.43_mobilenetv3.");

    // Dataset initialization
    raul::CIFAR10 cifar;
    std::cout << "Loading CIFAR...";
    ASSERT_TRUE(cifar.loadingData(tools::getTestAssetsDir() / "CIFAR"));
    std::cout << "done" << std::endl;

    printf("Memory taken = %.2fMB \n\n", static_cast<float>(work.getMemoryManager().getTotalMemory()) / (1024.0f * 1024.0f));

    auto sgd = std::make_shared<raul::optimizers::SGD>(lr);

    std::vector<raul::dtype> goldLoss = { 0.0141768642_dt, 0.15402_dt };

    for (size_t q = 0; q < 2; ++q)
    {
        raul::dtype testLoss = cifar.oneTrainIteration(work, sgd.get(), q, std::make_pair(224, 224));
        EXPECT_NEAR(testLoss, goldLoss[q], 1e-5);
    }
}

TEST(TestMobileNetV3, CifarPretrainedModelCheckpointsSpecialCaseTrain)
{
    PROFILE_TEST
    constexpr auto lr = 0.05_dt;
    const size_t image_size = 224U;
    const size_t image_channels = 3U;
    const size_t image_classes = 10U;
    const size_t batch_size = 50U;

    // Build work
    raul::Workflow work(raul::CompressionMode::NONE, raul::CalculationMode::DETERMINISTIC, raul::AllocationMode::STANDARD);
    const auto output_name = build_mobilenetv3_small(work, image_size, image_channels, image_classes);
    work.add<raul::LogSoftMaxActivation>("softmax", raul::BasicParamsWithDim{ { output_name }, { "softmax" } });
    work.add<raul::NLLLoss>("loss", raul::LossParams{ { "softmax", "labels" }, { "loss" }, "batch_mean" });

    work.setCheckpoints({ "block3::bn2", "block6::bn2", "block9::bn2" });

    work.preparePipelines(raul::Workflow::Execution::Checkpointed);

    work.setBatchSize(batch_size);
    work.prepareMemoryForTraining();

    // Initialization of the work
    auto& memory_manager = work.getMemoryManager();
    const auto weights_path = tools::getTestAssetsDir() / "mobilenet_v3" / "small" / "seed_0_epoch20_acc_80.43";
    load_mobilenetv3_small_weights(memory_manager, weights_path, "80.43_mobilenetv3.");

    // Dataset initialization
    raul::CIFAR10 cifar;
    std::cout << "Loading CIFAR...";
    ASSERT_TRUE(cifar.loadingData(tools::getTestAssetsDir() / "CIFAR"));
    std::cout << "done" << std::endl;

    printf("Memory taken = %.2fMB \n\n", static_cast<float>(work.getMemoryManager().getTotalMemory()) / (1024.0f * 1024.0f));

    auto sgd = std::make_shared<raul::optimizers::SGD>(lr);

    std::vector<raul::dtype> goldLoss = { 0.0141768642_dt, 0.15402_dt };

    for (size_t q = 0; q < 2; ++q)
    {
        raul::dtype testLoss = cifar.oneTrainIteration(work, sgd.get(), q, std::make_pair(224, 224));
        EXPECT_NEAR(testLoss, goldLoss[q], 1e-5);
    }
}

TEST(TestMobileNetV3, CifarPretrainedModelCheckpointsPoolTrain)
{
    PROFILE_TEST
    constexpr auto lr = 0.05_dt;
    const size_t image_size = 224U;
    const size_t image_channels = 3U;
    const size_t image_classes = 10U;
    const size_t batch_size = 50U;

    // Build work
    raul::Workflow work(raul::CompressionMode::NONE, raul::CalculationMode::DETERMINISTIC, raul::AllocationMode::POOL);
    const auto output_name = build_mobilenetv3_small(work, image_size, image_channels, image_classes);
    work.add<raul::LogSoftMaxActivation>("softmax", raul::BasicParamsWithDim{ { output_name }, { "softmax" } });
    work.add<raul::NLLLoss>("loss", raul::LossParams{ { "softmax", "labels" }, { "loss" }, "batch_mean" });

    raul::Names checkpointsAll = work.getPotentialCheckpoints();
    raul::Names checkpoints;

    raul::Names blocks = { "block3", "block6", "block9" };
    std::vector<bool> isBlock(blocks.size(), false);

    for (raul::Name& checkP : checkpointsAll)
    {
        for (size_t q = 0; q < blocks.size(); ++q)
        {
            if (!isBlock[q] && checkP.str().find(blocks[q]) != std::string::npos)
            {
                isBlock[q] = true;
                checkpoints.push_back(checkP);
            }
        }
    }

    work.setCheckpoints(checkpoints);

    work.preparePipelines(raul::Workflow::Execution::Checkpointed);

    work.setBatchSize(batch_size);
    work.prepareMemoryForTraining();

    // Initialization of the work
    auto& memory_manager = work.getMemoryManager();
    const auto weights_path = tools::getTestAssetsDir() / "mobilenet_v3" / "small" / "seed_0_epoch20_acc_80.43";
    load_mobilenetv3_small_weights(memory_manager, weights_path, "80.43_mobilenetv3.");

    // Dataset initialization
    raul::CIFAR10 cifar;
    std::cout << "Loading CIFAR...";
    ASSERT_TRUE(cifar.loadingData(tools::getTestAssetsDir() / "CIFAR"));
    std::cout << "done" << std::endl;

    printf("Memory taken = %.2fMB \n\n", static_cast<float>(work.getMemoryManager().getTotalMemory()) / (1024.0f * 1024.0f));

    auto sgd = std::make_shared<raul::optimizers::SGD>(lr);

    std::vector<raul::dtype> goldLoss = { 0.0141768642_dt, 0.15402_dt };

    for (size_t q = 0; q < 2; ++q)
    {
        raul::dtype testLoss = cifar.oneTrainIteration(work, sgd.get(), q, std::make_pair(224, 224));
        EXPECT_NEAR(testLoss, goldLoss[q], 1e-5);
    }
}

/*
TEST(TestMobileNetV3, Experiments)
{
    PROFILE_TEST
    const char* env_raul_mn3_batch = std::getenv("RAUL_MN3_BATCH");

    constexpr auto acc_eps = 1e-1_dt;
    constexpr auto lr = 0.05_dt;
    const size_t image_size = 224U;
    const size_t image_channels = 3U;
    const size_t image_classes = 10U;
    const size_t batch_size = env_raul_mn3_batch == nullptr ? 50U : std::stoi(env_raul_mn3_batch);
    const size_t print_freq = 100U;
    const size_t epochs = 10U;

    const auto golden_acc_before = 80.43_dt;
    const auto golden_acc_after = 82.68_dt;

    // Config
    std::cout << "Batch size = " << batch_size << std::endl;

    {
        std::cout << "App loaded" << std::endl;
        std::time_t result = std::time(nullptr);
        std::cout << std::asctime(std::localtime(&result)) << result << " seconds since the Epoch\n";
    }

    // Build work
    raul::Workflow work;
    const auto output_name = build_mobilenetv3_small(work, image_size, image_channels, image_classes, batch_size);
    work.add<raul::LogSoftMaxActivation>(  "softmax", raul::BasicParamsWithDim{ { output_name }, { "softmax" } });
    work.add<raul::NLLLoss>(  "loss", raul::LossParams{ { "softmax", "labels" }, { "loss" }, "mean" });
    raul::Graph work(std::move(work), BATCH_SIZE);

    {
        std::cout << "Network built" << std::endl;
        std::time_t result = std::time(nullptr);
        std::cout << std::asctime(std::localtime(&result)) << result << " seconds since the Epoch\n";
    }

    // Initialization of the work
    auto& memory_manager = work.getMemoryManager();
    const auto weights_path = tools::getTestAssetsDir() / "mobilenet_v3" / "small" / "seed_0_epoch20_acc_80.43";
    load_mobilenetv3_small_weights(memory_manager, weights_path, "80.43_mobilenetv3.");

    {
        std::cout << "Weights loaded" << std::endl;
        std::time_t result = std::time(nullptr);
        std::cout << std::asctime(std::localtime(&result)) << result << " seconds since the Epoch\n";
    }

    // Dataset initialization
    raul::CIFAR10 cifar;
    std::cout << "Loading CIFAR...";
    ASSERT_TRUE(cifar.loadingData(tools::getTestAssetsDir() / "CIFAR"));
    std::cout << "done" << std::endl;

    {
        std::cout << "CIFAR loaded" << std::endl;
        std::time_t result = std::time(nullptr);
        std::cout << std::asctime(std::localtime(&result)) << result << " seconds since the Epoch\n";
    }

    printf("Memory taken = %.2fMB \n\n", work.getMemoryManager().getTotalMemory() / (1024.0f * 1024.0f));

    // Before train
     {
         std::cout << "Calculating acc..." << std::endl;
         raul::dtype testAcc = cifar.testNetwork(work, std::make_pair(224, 224));
         printf("Testing taken = %.3fs \n", cifar.getTestingTime());
         printf("Test accuracy = %.2f\n", testAcc);
         fflush(stdout);
         EXPECT_NEAR(testAcc, golden_acc_before, acc_eps);
     }
    // Trainings
    const size_t stepsAmountTrain = cifar.getTrainImageAmount() / batch_size;
    auto sgd = std::make_shared<raul::optimizers::SGD>(lr);

    std::cout << "Training with " << *sgd << std::endl;

    for (size_t epoch = 1; epoch <= 1; ++epoch)
    {
        printf("Epoch = %zu\n", epoch);

        for (size_t q = 0; q < 5; ++q)
        {
            raul::dtype testLoss = cifar.oneTrainIteration(work, sgd.get(), q, std::make_pair(224, 224));
            if (q % print_freq == 0)
            {
                printf("iteration = %zu/%zu, loss = %f\n", q, stepsAmountTrain, testLoss);
                fflush(stdout);
            }

            {
                std::cout << "Step finished" << std::endl;
                std::time_t result = std::time(nullptr);
                std::cout << std::asctime(std::localtime(&result)) << result << " seconds since the Epoch\n";
            }
        }

        printf("Epoch Training taken = %.3fs \n", cifar.getTrainingTime());

        // After train
         {
             std::cout << "Calculating acc..." << std::endl;
             raul::dtype testAcc = cifar.testNetwork(work, std::make_pair(224, 224));
             printf("Test accuracy = %.2f\n", testAcc);
             fflush(stdout);
             // EXPECT_NEAR(testAcc, golden_acc_after, acc_eps);
         }
    }

    {
        std::cout << "Train finished" << std::endl;
        std::time_t result = std::time(nullptr);
        std::cout << std::asctime(std::localtime(&result)) << result << " seconds since the Epoch\n";
    }

    printf("Memory taken = %.2fMB \n\n", work.getMemoryManager().getTotalMemory() / (1024.0f * 1024.0f));
}
*/

} // namespace UT
