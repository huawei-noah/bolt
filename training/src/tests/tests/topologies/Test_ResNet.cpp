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
#include <training/base/optimizers/SGD.h>
#include <training/base/common/quantization/SymmetricQuantizer.h>

namespace UT
{

using ResNetT = std::array<size_t, 4U>;
using downsampleT = std::function<void(raul::Workflow&, raul::Name&, const raul::Name&)>;

void add_resnet_basic_block(size_t& block_cnt,
                            raul::Workflow& work,
                            raul::Name& input,
                            const raul::Name& name_prefix,
                            const size_t planes,
                            const size_t stride = 1U,
                            const std::optional<downsampleT> downsample = std::nullopt,
                            const size_t padding = 1U, // dilation in the reference
                            const float bnMomentum = 0.1f,
                            const bool bias = false,
                            const bool quantize = false)
{
    if(padding != 1U)
    {
        THROW_NONAME("add_resnet_basic_block with padding != 1", "not implemented");
    }

    const auto kernel_size = 3U;
    const auto default_stride = 1U;
    const auto block_name = name_prefix / "block" + Conversions::toString(block_cnt);
    auto input_for_shortcut = input;

    ++block_cnt;

    if (quantize)
    {
        work.add<raul::FakeQuantLayer>(block_name + "::fq1", raul::FakeQuantParams{ { input }, { block_name + "::fq1" } });
        input = block_name + "::fq1";
    }

    work.add<raul::Convolution2DLayer>(block_name + "::conv1", raul::Convolution2DParams{ { input }, { block_name + "::conv1" }, kernel_size, planes, stride, padding, bias, quantize });
    work.add<raul::BatchNormLayer>(block_name + "::bn1", raul::BatchnormParams{ { block_name + "::conv1" }, { block_name + "::bn1" }, bnMomentum });
    work.add<raul::ReLUActivation>(block_name + "::relu1", raul::BasicParams{ { block_name + "::bn1" }, { block_name + "::relu1" } });
    input = block_name + "::relu1";
    if (quantize)
    {
        work.add<raul::FakeQuantLayer>(block_name + "::fq2", raul::FakeQuantParams{ { input }, { block_name + "::fq2" } });
        input = block_name + "::fq2";
    }

    work.add<raul::Convolution2DLayer>(block_name + "::conv2", raul::Convolution2DParams{ { input }, { block_name + "::conv2" }, kernel_size, planes, default_stride, padding, bias, quantize });
    work.add<raul::BatchNormLayer>(block_name + "::bn2", raul::BatchnormParams{ { block_name + "::conv2" }, { block_name + "::bn2" }, bnMomentum });
    input = block_name + "::bn2";

    if (downsample)
    {
        (*downsample)(work, input_for_shortcut, block_name);
    }

    work.add<raul::ElementWiseSumLayer>(block_name + "::sum", raul::ElementWiseLayerParams{ { input_for_shortcut, input }, { block_name + "::sum" } });
    work.add<raul::ReLUActivation>(block_name + "::relu2", raul::BasicParams{ { block_name + "::sum" }, { block_name + "::relu2" } });
    input = block_name + "::relu2";
}

void add_input_block(raul::Workflow& work, raul::Name& input, const float bnMomentum = 0.1f, const bool bias = false)
{
    const auto conv_in_planes = 64U;
    const auto conv_kernel_size = 7U;
    const auto conv_stride = 2U;
    const auto conv_padding = 3U;

    const auto max_pool_kernel_size = 3;
    const auto max_pool_stride = 2;
    const auto max_pool_padding = 1U;

    work.add<raul::Convolution2DLayer>("input::conv1", raul::Convolution2DParams{ { input }, { "input::conv1" }, conv_kernel_size, conv_in_planes, conv_stride, conv_padding, bias });
    work.add<raul::BatchNormLayer>("input::bn1", raul::BatchnormParams{ { "input::conv1" }, { "input::bn1" }, bnMomentum });
    work.add<raul::ReLUActivation>("input::relu", raul::BasicParams{ { "input::bn1" }, { "input::relu" } });
    work.add<raul::MaxPoolLayer2D>("input::maxpool", raul::Pool2DParams{ { "input::relu" }, { "input::maxpool" }, max_pool_kernel_size, max_pool_stride, max_pool_padding });
    input = "input::maxpool";
}

void add_output_block(raul::Workflow& work, raul::Name& input, const size_t num_classes = 10U)
{
    work.add<raul::GlobAveragePoolLayer>("output::avg", raul::BasicParams{ { input }, { "output::avg" } });
    work.add<raul::ReshapeLayer>("output::reshape", raul::ViewParams{ "output::avg", "output::avgr", 1, 1, -1 });
    work.add<raul::LinearLayer>("output::fc0", raul::LinearParams{ { "output::avgr" }, { "output::fc0" }, num_classes });
    input = "output::fc0";
}

void add_resnet_layer(size_t& layer_cnt,
                      raul::Workflow& work,
                      raul::Name& input,
                      size_t& inplanes,
                      size_t expansion,
                      const size_t planes,
                      const size_t blocks,
                      const size_t stride = 1U,
                      const size_t padding = 1U,
                      const float bnMomentum = 0.1f,
                      const bool bias = false,
                      const bool quantize = false)
{
    const auto layer_name = "layer" + Conversions::toString(layer_cnt);
    size_t block_cnt = 0U;
    std::optional<downsampleT> downsample = std::nullopt;

    ++layer_cnt;

    if (stride != 1U || inplanes != planes * expansion)
    {
        downsample = [layer_name, bias, quantize, planes, expansion, stride, bnMomentum](raul::Workflow& work, raul::Name& input, const raul::Name& name_prefix) {
            const auto conv_kernel_size = 1U;
            const auto downsample_name = name_prefix + "::downsample";
            if (quantize)
            {
                work.add<raul::FakeQuantLayer>(downsample_name + "::fq1", raul::FakeQuantParams{ { input }, { downsample_name + "::fq1" } });
                input = downsample_name + "::fq1";
            }
            work.add<raul::Convolution2DLayer>(downsample_name + "::conv1",
                                               raul::Convolution2DParams{ { input }, { downsample_name + "::conv1" }, conv_kernel_size, planes * expansion, stride, 0U, bias, quantize });
            work.add<raul::BatchNormLayer>(downsample_name + "::bn1", raul::BatchnormParams{ { downsample_name + "::conv1" }, { downsample_name + "::bn1" }, bnMomentum });
            input = downsample_name + "::bn1";
        };
    }
    add_resnet_basic_block(block_cnt, work, input, layer_name, planes, stride, downsample, padding, bnMomentum, bias, quantize);
    inplanes = planes * expansion;
    downsample = std::nullopt;
    for (size_t i = 1U; i < blocks; ++i)
    {
        const auto default_stride = 1U;
        add_resnet_basic_block(block_cnt, work, input, layer_name, planes, default_stride, downsample, padding, bnMomentum, bias, quantize);
    }
}

raul::Name build_resnet(raul::Workflow& work,
                        const ResNetT layers,
                        const size_t image_size = 224U,
                        const size_t image_channels = 3U,
                        const size_t labels_cnt = 10U,
                        const float bnMomentum = 0.1f,
                        const bool bias = false,
                        const bool quantize = false)
{
    // This parameters are actual only for BasicBlock (see notebook for details)
    size_t inplanes = 64U;
    const size_t expansion = 1U;

    size_t layer_cnt = 1U;
    raul::Name input = "data";
    work.add<raul::DataLayer>("data", raul::DataParams{ { input, "labels" }, image_channels, image_size, image_size, labels_cnt });
    add_input_block(work, input);
    add_resnet_layer(layer_cnt, work, input, inplanes, expansion, 64U, layers[0], 1U, 1U, bnMomentum, bias, quantize);
    add_resnet_layer(layer_cnt, work, input, inplanes, expansion, 128U, layers[1], 2U, 1U, bnMomentum, bias, quantize);
    add_resnet_layer(layer_cnt, work, input, inplanes, expansion, 256U, layers[2], 2U, 1U, bnMomentum, bias, quantize);
    add_resnet_layer(layer_cnt, work, input, inplanes, expansion, 512U, layers[3], 2U, 1U, bnMomentum, bias, quantize);
    add_output_block(work, input);
    return input;
}

raul::Name build_resnet18(raul::Workflow& work,
                          const size_t image_size = 224U,
                          const size_t image_channels = 3U,
                          const size_t labels_cnt = 10U,
                          const float bnMomentum = 0.1f,
                          const bool bias = false,
                          const bool quantize = false)
{
    const ResNetT layers{ 2U, 2U, 2U, 2U };
    return build_resnet(work, layers, image_size, image_channels, labels_cnt, bnMomentum, bias, quantize);
}

TEST(TestResNet, BasicBlockBuildingUnit)
{
    PROFILE_TEST
    const size_t block_in_planes = 10U;
    const size_t block_planes = 20U;
    const size_t golden_trainable_parameters = 5480U;

    const auto name_prefix = "";
    size_t block_cnt = 0U;
    raul::Name input{ "data" };
    raul::Workflow work;
    work.add<raul::DataLayer>("data", raul::DataParams{ { input, "labels" }, block_in_planes, 1U, 1U, 0U });
    add_resnet_basic_block(block_cnt, work, input, name_prefix, block_planes);

    work.preparePipelines();
    work.setBatchSize(1u);
    work.prepareMemoryForTraining();

    work.printInfo(std::cout);

    // Checks
    EXPECT_EQ(tools::get_size_of_trainable_params(work), golden_trainable_parameters);
}

TEST(TestResNet, InputBlockBuildingUnit)
{
    PROFILE_TEST
    const size_t block_in_planes = 3U;
    const size_t golden_trainable_parameters = 9536U;

    raul::Name input{ "data" };
    raul::Workflow work;
    work.add<raul::DataLayer>("data", raul::DataParams{ { input, "labels" }, block_in_planes, 1U, 1U, 0U });
    add_input_block(work, input);

    work.preparePipelines();
    work.setBatchSize(1u);
    work.prepareMemoryForTraining();

    work.printInfo(std::cout);

    // Checks
    EXPECT_EQ(tools::get_size_of_trainable_params(work), golden_trainable_parameters);
}

TEST(TestResNet, ResNet18BuildingUnit)
{
    PROFILE_TEST
    const size_t image_size = 224U;
    const size_t image_channels = 3U;
    const size_t image_classes = 10U;
    const size_t batch_size = 50U;

    const size_t golden_trainable_parameters = 11'181'642U;

    raul::Workflow work;
    const auto output_name = build_resnet18(work, image_size, image_channels, image_classes);

    work.preparePipelines();
    work.setBatchSize(batch_size);
    work.prepareMemoryForTraining();

    work.printInfo(std::cout);

    // Checks
    EXPECT_EQ(tools::get_size_of_trainable_params(work), golden_trainable_parameters);
}

TEST(TestResNet, ResNet18QuantizedBuildingUnit)
{
    PROFILE_TEST
    const size_t image_size = 224U;
    const size_t image_channels = 3U;
    const size_t image_classes = 10U;
    const size_t batch_size = 50U;
    const float batchtornm_momentum = 0.1f;
    const bool bias = false;
    const bool quantized = true;

    const size_t golden_trainable_parameters = 11'181'642U;

    auto quantizer = raul::quantization::SymmetricQuantizer(static_cast<raul::dtype (*)(raul::dtype)>(std::trunc));
    raul::Workflow work(raul::CompressionMode::FP16, raul::CalculationMode::DETERMINISTIC, raul::AllocationMode::STANDARD, raul::ExecutionTarget::CPU, false, &quantizer);
    const auto output_name = build_resnet18(work, image_size, image_channels, image_classes, batchtornm_momentum, bias, quantized);

    work.preparePipelines();
    work.setBatchSize(batch_size);
    work.prepareMemoryForTraining();

    work.printInfo(std::cout);

    // Checks
    EXPECT_EQ(tools::get_size_of_trainable_params(work), golden_trainable_parameters);
}

} // namespace UT