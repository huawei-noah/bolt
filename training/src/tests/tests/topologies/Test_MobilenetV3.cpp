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

#include <training/base/common/Common.h>
#include <training/compiler/Layers.h> 
#include <training/base/layers/basic/GlobalAveragePoolLayer.h>
#include <training/base/layers/basic/trainable/Batchnorm.h>
#include <training/base/layers/parameters/LayerParameters.h>
#include <training/base/common/Conversions.h>
#include <training/base/common/MemoryManager.h>
#include <training/compiler/Workflow.h>
#include <training/base/optimizers/Adam.h>
#include <training/base/optimizers/SGD.h>

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

} // namespace UT