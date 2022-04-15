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
#include <training/network/Layers.h>
#include <training/tools/Datasets.h>

#include <training/layers/basic/GlobalAveragePoolLayer.h>
#include <training/layers/basic/trainable/Batchnorm.h>
#include <training/layers/parameters/LayerParameters.h>

#include <training/common/Conversions.h>
#include <training/common/MemoryManager.h>
#include <training/network/Workflow.h>
#include <training/optimizers/SGD.h>

#include <training/common/quantization/SymmetricQuantizer.h>

#include <chrono>
#include <cstdio>

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
                            const size_t dilation = 1U,
                            const float bnMomentum = 0.1f,
                            const bool bias = false,
                            const bool quantize = false)
{
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

    work.add<raul::Convolution2DLayer>(block_name + "::conv1", raul::Convolution2DParams{ { input }, { block_name + "::conv1" }, kernel_size, planes, stride, dilation, bias, quantize });
    work.add<raul::BatchNormLayer>(block_name + "::bn1", raul::BatchnormParams{ { block_name + "::conv1" }, { block_name + "::bn1" }, bnMomentum });
    work.add<raul::ReLUActivation>(block_name + "::relu1", raul::BasicParams{ { block_name + "::bn1" }, { block_name + "::relu1" } });
    input = block_name + "::relu1";
    if (quantize)
    {
        work.add<raul::FakeQuantLayer>(block_name + "::fq2", raul::FakeQuantParams{ { input }, { block_name + "::fq2" } });
        input = block_name + "::fq2";
    }

    work.add<raul::Convolution2DLayer>(block_name + "::conv2", raul::Convolution2DParams{ { input }, { block_name + "::conv2" }, kernel_size, planes, default_stride, dilation, bias, quantize });
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
                      const size_t dilation = 1U,
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
    add_resnet_basic_block(block_cnt, work, input, layer_name, planes, stride, downsample, dilation, bnMomentum, bias, quantize);
    inplanes = planes * expansion;
    downsample = std::nullopt;
    for (size_t i = 1U; i < blocks; ++i)
    {
        const auto default_stride = 1U;
        add_resnet_basic_block(block_cnt, work, input, layer_name, planes, default_stride, downsample, dilation, bnMomentum, bias, quantize);
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

/// @todo(ck): need to write an automatic weight loading mechanism
void load_resnet_weights(raul::MemoryManager& memory_manager,
                         const std::filesystem::path& weights_path,
                         const std::string file_prefix = "init_",
                         const bool extended = false,
                         const bool shifted = false)
{
    using namespace std;
    std::map<raul::Name, std::pair<std::string, size_t>> pytorch_name_map{ { "input::conv1::Weights", { "conv1.weight", 9408 } },
                                                                           { "layer1::block0::conv1::Weights", { "layer1.0.conv1.weight", 36864 } },
                                                                           { "layer1::block0::conv2::Weights", { "layer1.0.conv2.weight", 36864 } },
                                                                           { "layer1::block1::conv1::Weights", { "layer1.1.conv1.weight", 36864 } },
                                                                           { "layer1::block1::conv2::Weights", { "layer1.1.conv2.weight", 36864 } },
                                                                           { "layer2::block0::conv1::Weights", { "layer2.0.conv1.weight", 73728 } },
                                                                           { "layer2::block0::conv2::Weights", { "layer2.0.conv2.weight", 147456 } },
                                                                           { "layer2::block0::downsample::conv1::Weights", { "layer2.0.downsample."s + (shifted ? "1" : "0") + ".weight", 8192 } },
                                                                           { "layer2::block1::conv1::Weights", { "layer2.1.conv1.weight", 147456 } },
                                                                           { "layer2::block1::conv2::Weights", { "layer2.1.conv2.weight", 147456 } },
                                                                           { "layer3::block0::conv1::Weights", { "layer3.0.conv1.weight", 294912 } },
                                                                           { "layer3::block0::conv2::Weights", { "layer3.0.conv2.weight", 589824 } },
                                                                           { "layer3::block0::downsample::conv1::Weights", { "layer3.0.downsample."s + (shifted ? "1" : "0") + ".weight", 32768 } },
                                                                           { "layer3::block1::conv1::Weights", { "layer3.1.conv1.weight", 589824 } },
                                                                           { "layer3::block1::conv2::Weights", { "layer3.1.conv2.weight", 589824 } },
                                                                           { "layer4::block0::conv1::Weights", { "layer4.0.conv1.weight", 1179648 } },
                                                                           { "layer4::block0::conv2::Weights", { "layer4.0.conv2.weight", 2359296 } },
                                                                           { "layer4::block0::downsample::conv1::Weights", { "layer4.0.downsample."s + (shifted ? "1" : "0") + ".weight", 131072 } },
                                                                           { "layer4::block1::conv1::Weights", { "layer4.1.conv1.weight", 2359296 } },
                                                                           { "layer4::block1::conv2::Weights", { "layer4.1.conv2.weight", 2359296 } },
                                                                           { "output::fc0::Biases", { "fc.bias", 10 } },
                                                                           { "output::fc0::Weights", { "fc.weight", 5120 } } };

    std::map<raul::Name, std::pair<std::string, size_t>> pytorch_name_map_extended{
        { "input::bn1::Biases", { "bn1.bias", 64 } },
        { "input::bn1::MeanEval", { "bn1.running_mean", 64 } },
        { "input::bn1::VarianceEval", { "bn1.running_var", 64 } },
        { "input::bn1::Weights", { "bn1.weight", 64 } },
        { "layer1::block0::bn1::Biases", { "layer1.0.bn1.bias", 64 } },
        { "layer1::block0::bn1::MeanEval", { "layer1.0.bn1.running_mean", 64 } },
        { "layer1::block0::bn1::VarianceEval", { "layer1.0.bn1.running_var", 64 } },
        { "layer1::block0::bn1::Weights", { "layer1.0.bn1.weight", 64 } },
        { "layer1::block0::bn2::Biases", { "layer1.0.bn2.bias", 64 } },
        { "layer1::block0::bn2::MeanEval", { "layer1.0.bn2.running_mean", 64 } },
        { "layer1::block0::bn2::VarianceEval", { "layer1.0.bn2.running_var", 64 } },
        { "layer1::block0::bn2::Weights", { "layer1.0.bn2.weight", 64 } },
        { "layer1::block1::bn1::Biases", { "layer1.1.bn1.bias", 64 } },
        { "layer1::block1::bn1::MeanEval", { "layer1.1.bn1.running_mean", 64 } },
        { "layer1::block1::bn1::VarianceEval", { "layer1.1.bn1.running_var", 64 } },
        { "layer1::block1::bn1::Weights", { "layer1.1.bn1.weight", 64 } },
        { "layer1::block1::bn2::Biases", { "layer1.1.bn2.bias", 64 } },
        { "layer1::block1::bn2::MeanEval", { "layer1.1.bn2.running_mean", 64 } },
        { "layer1::block1::bn2::VarianceEval", { "layer1.1.bn2.running_var", 64 } },
        { "layer1::block1::bn2::Weights", { "layer1.1.bn2.weight", 64 } },
        { "layer2::block0::bn1::Biases", { "layer2.0.bn1.bias", 128 } },
        { "layer2::block0::bn1::MeanEval", { "layer2.0.bn1.running_mean", 128 } },
        { "layer2::block0::bn1::VarianceEval", { "layer2.0.bn1.running_var", 128 } },
        { "layer2::block0::bn1::Weights", { "layer2.0.bn1.weight", 128 } },
        { "layer2::block0::bn2::Biases", { "layer2.0.bn2.bias", 128 } },
        { "layer2::block0::bn2::MeanEval", { "layer2.0.bn2.running_mean", 128 } },
        { "layer2::block0::bn2::VarianceEval", { "layer2.0.bn2.running_var", 128 } },
        { "layer2::block0::bn2::Weights", { "layer2.0.bn2.weight", 128 } },
        { "layer2::block0::downsample::bn1::Biases", { "layer2.0.downsample."s + (shifted ? "2" : "1") + ".bias", 128 } },
        { "layer2::block0::downsample::bn1::MeanEval", { "layer2.0.downsample."s + (shifted ? "2" : "1") + ".running_mean", 128 } },
        { "layer2::block0::downsample::bn1::VarianceEval", { "layer2.0.downsample."s + (shifted ? "2" : "1") + ".running_var", 128 } },
        { "layer2::block0::downsample::bn1::Weights", { "layer2.0.downsample."s + (shifted ? "2" : "1") + ".weight", 128 } },
        { "layer2::block1::bn1::Biases", { "layer2.1.bn1.bias", 128 } },
        { "layer2::block1::bn1::MeanEval", { "layer2.1.bn1.running_mean", 128 } },
        { "layer2::block1::bn1::VarianceEval", { "layer2.1.bn1.running_var", 128 } },
        { "layer2::block1::bn1::Weights", { "layer2.1.bn1.weight", 128 } },
        { "layer2::block1::bn2::Biases", { "layer2.1.bn2.bias", 128 } },
        { "layer2::block1::bn2::MeanEval", { "layer2.1.bn2.running_mean", 128 } },
        { "layer2::block1::bn2::VarianceEval", { "layer2.1.bn2.running_var", 128 } },
        { "layer2::block1::bn2::Weights", { "layer2.1.bn2.weight", 128 } },
        { "layer3::block0::bn1::Biases", { "layer3.0.bn1.bias", 256 } },
        { "layer3::block0::bn1::MeanEval", { "layer3.0.bn1.running_mean", 256 } },
        { "layer3::block0::bn1::VarianceEval", { "layer3.0.bn1.running_var", 256 } },
        { "layer3::block0::bn1::Weights", { "layer3.0.bn1.weight", 256 } },
        { "layer3::block0::bn2::Biases", { "layer3.0.bn2.bias", 256 } },
        { "layer3::block0::bn2::MeanEval", { "layer3.0.bn2.running_mean", 256 } },
        { "layer3::block0::bn2::VarianceEval", { "layer3.0.bn2.running_var", 256 } },
        { "layer3::block0::bn2::Weights", { "layer3.0.bn2.weight", 256 } },
        { "layer3::block0::downsample::bn1::Biases", { "layer3.0.downsample."s + (shifted ? "2" : "1") + ".bias", 256 } },
        { "layer3::block0::downsample::bn1::MeanEval", { "layer3.0.downsample."s + (shifted ? "2" : "1") + ".running_mean", 256 } },
        { "layer3::block0::downsample::bn1::VarianceEval", { "layer3.0.downsample."s + (shifted ? "2" : "1") + ".running_var", 256 } },
        { "layer3::block0::downsample::bn1::Weights", { "layer3.0.downsample."s + (shifted ? "2" : "1") + ".weight", 256 } },
        { "layer3::block1::bn1::Biases", { "layer3.1.bn1.bias", 256 } },
        { "layer3::block1::bn1::MeanEval", { "layer3.1.bn1.running_mean", 256 } },
        { "layer3::block1::bn1::VarianceEval", { "layer3.1.bn1.running_var", 256 } },
        { "layer3::block1::bn1::Weights", { "layer3.1.bn1.weight", 256 } },
        { "layer3::block1::bn2::Biases", { "layer3.1.bn2.bias", 256 } },
        { "layer3::block1::bn2::MeanEval", { "layer3.1.bn2.running_mean", 256 } },
        { "layer3::block1::bn2::VarianceEval", { "layer3.1.bn2.running_var", 256 } },
        { "layer3::block1::bn2::Weights", { "layer3.1.bn2.weight", 256 } },
        { "layer4::block0::bn1::Biases", { "layer4.0.bn1.bias", 512 } },
        { "layer4::block0::bn1::MeanEval", { "layer4.0.bn1.running_mean", 512 } },
        { "layer4::block0::bn1::VarianceEval", { "layer4.0.bn1.running_var", 512 } },
        { "layer4::block0::bn1::Weights", { "layer4.0.bn1.weight", 512 } },
        { "layer4::block0::bn2::Biases", { "layer4.0.bn2.bias", 512 } },
        { "layer4::block0::bn2::MeanEval", { "layer4.0.bn2.running_mean", 512 } },
        { "layer4::block0::bn2::VarianceEval", { "layer4.0.bn2.running_var", 512 } },
        { "layer4::block0::bn2::Weights", { "layer4.0.bn2.weight", 512 } },
        { "layer4::block0::downsample::bn1::Biases", { "layer4.0.downsample."s + (shifted ? "2" : "1") + ".bias", 512 } },
        { "layer4::block0::downsample::bn1::MeanEval", { "layer4.0.downsample."s + (shifted ? "2" : "1") + ".running_mean", 512 } },
        { "layer4::block0::downsample::bn1::VarianceEval", { "layer4.0.downsample."s + (shifted ? "2" : "1") + ".running_var", 512 } },
        { "layer4::block0::downsample::bn1::Weights", { "layer4.0.downsample."s + (shifted ? "2" : "1") + ".weight", 512 } },
        { "layer4::block1::bn1::Biases", { "layer4.1.bn1.bias", 512 } },
        { "layer4::block1::bn1::MeanEval", { "layer4.1.bn1.running_mean", 512 } },
        { "layer4::block1::bn1::VarianceEval", { "layer4.1.bn1.running_var", 512 } },
        { "layer4::block1::bn1::Weights", { "layer4.1.bn1.weight", 512 } },
        { "layer4::block1::bn2::Biases", { "layer4.1.bn2.bias", 512 } },
        { "layer4::block1::bn2::MeanEval", { "layer4.1.bn2.running_mean", 512 } },
        { "layer4::block1::bn2::VarianceEval", { "layer4.1.bn2.running_var", 512 } },
        { "layer4::block1::bn2::Weights", { "layer4.1.bn2.weight", 512 } }
    };

    raul::DataLoader dataLoader;
    std::cout << "Loading weights...";
    for (auto [tensor, value] : pytorch_name_map)
    {
        const auto [filename, size] = value;
        memory_manager[tensor] = dataLoader.loadData(weights_path / (file_prefix + filename + ".data"), size, 1, 1);
    }

    if (extended)
    {
        for (auto [tensor, value] : pytorch_name_map_extended)
        {
            const auto [filename, size] = value;
            memory_manager[tensor] = dataLoader.loadData(weights_path / (file_prefix + filename + ".data"), size, 1, 1);
        }
    }

    if (memory_manager.tensorExists("output::fc0::Weights")) raul::Common::transpose(memory_manager["output::fc0::Weights"], 10U);

    std::cout << "done" << std::endl;
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

TEST(TestResNet, ResNet18InferenceUnit)
{
    PROFILE_TEST
    constexpr auto relative_eps = 1e-3_dt;
    const size_t image_size = 224U;
    const size_t image_channels = 3U;
    const size_t image_classes = 10U;
    const size_t batch_size = 1U;
    const raul::dtype in_val = 5.0_dt;

    const raul::Tensor golden_tensor_out{ 1.403602e+00_dt,  9.141974e-03_dt,  -2.788210e+00_dt, 2.950551e+00_dt, -1.179150e+00_dt,
                                          -2.007151e+00_dt, -3.162097e-01_dt, -8.549929e-01_dt, 1.268970e+00_dt, 3.833183e-01_dt };

    // Build work
    raul::Workflow work;
    const auto output_name = build_resnet18(work, image_size, image_channels, image_classes);

    work.preparePipelines();
    work.setBatchSize(batch_size);
    work.prepareMemoryForTraining();

    // Initialization of the work
    auto& memory_manager = work.getMemoryManager();
    const auto weights_path = tools::getTestAssetsDir() / "resnet" / "18" / "seed_0";
    load_resnet_weights(memory_manager, weights_path);

    // Initialization in_tensor and out_tensor
    raul::Tensor& in_tensor = memory_manager.getTensor("data");
    const raul::Tensor& out_tensor = memory_manager.getTensor(output_name);

    std::fill(in_tensor.begin(), in_tensor.end(), in_val);

    // Apply ResNet 18
    std::cout << "Forward...";
    work.forwardPassTesting();
    std::cout << "done" << std::endl;

    for (size_t q = 0; q < out_tensor.size(); ++q)
    {
        std::cout << out_tensor[q] << " == " << golden_tensor_out[q] << std::endl;
        ASSERT_TRUE(tools::expect_near_relative(out_tensor[q], golden_tensor_out[q], relative_eps));
    }
}

TEST(TestResNet, ResNet18CifarPretrainedModelOneEpochTrain)
{
    PROFILE_TEST
    constexpr auto acc_eps = 1e-1_dt;
    constexpr auto loss_eps_rel = 1e-1_dt;
    constexpr auto lr = 0.05_dt;
    const size_t image_size = 224U;
    const size_t image_channels = 3U;
    const size_t image_classes = 10U;
    const size_t batch_size = 50U;
    const size_t print_freq = 100U;

    bool useCheckpointing = false;
    bool usePool = false;

    const auto golden_acc_before = 83.24_dt;
    const auto golden_acc_after = 85.24_dt;

    const raul::Tensor idealLosses{ 3.608587e-02_dt, 1.817846e-01_dt, 1.034031e-01_dt, 4.845553e-03_dt, 3.461881e-02_dt,
                                    1.446302e-02_dt, 2.096494e-02_dt, 8.074160e-03_dt, 2.517764e-02_dt, 1.755940e-02_dt };

    // Build work
    raul::Workflow work(raul::CompressionMode::NONE, raul::CalculationMode::DETERMINISTIC, usePool ? raul::AllocationMode::POOL : raul::AllocationMode::STANDARD);
    const auto output_name = build_resnet18(work, image_size, image_channels, image_classes);
    work.add<raul::LogSoftMaxActivation>("softmax", raul::BasicParamsWithDim{ { output_name }, { "softmax" } });
    work.add<raul::NLLLoss>("loss", raul::LossParams{ { "softmax", "labels" }, { "loss" }, "batch_mean" });

    if (useCheckpointing)
    {
        raul::Names checkpointsAll = work.getPotentialCheckpoints();
        raul::Names checkpoints;

        raul::Names blocks = { "layer1", "layer2", "layer3" };
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
    }
    else
    {
        work.preparePipelines();
    }
    work.setBatchSize(batch_size);
    work.prepareMemoryForTraining();

    // Initialization of the work
    auto& memory_manager = work.getMemoryManager();
    const auto weights_path = tools::getTestAssetsDir() / "resnet" / "18" / "seed_0_epoch10_acc_83.24";
    load_resnet_weights(memory_manager, weights_path, "83.24_resnet.", true);

    // Dataset initialization
    raul::CIFAR10 cifar;
    std::cout << "Loading CIFAR...";
    ASSERT_TRUE(cifar.loadingData(tools::getTestAssetsDir() / "CIFAR"));
    std::cout << "done" << std::endl;

    printf("Memory taken = %.2fMB \n\n", static_cast<float>(work.getMemoryManager().getTotalMemory()) / (1024.0f * 1024.0f));
    fflush(stdout);

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
    std::cout << "Training..." << std::endl;

    for (size_t epoch = 1; epoch <= 1; ++epoch)
    {
        printf("Epoch = %zu\n", epoch);

        for (size_t q = 0; q < stepsAmountTrain; ++q)
        {
            raul::dtype testLoss = cifar.oneTrainIteration(work, sgd.get(), q, std::make_pair(224, 224));
            if (q % print_freq == 0)
            {
                EXPECT_TRUE(tools::expect_near_relative(testLoss, idealLosses[q / print_freq], loss_eps_rel)) << "expected: " << idealLosses[q / print_freq] << ", got: " << testLoss;
                printf("iteration = %zu/%zu, loss = %f\n", q, stepsAmountTrain, testLoss);
                fflush(stdout);
            }
        }

        printf("Epoch Training taken = %.3fs \n", cifar.getTrainingTime());

        // After train
        {
            std::cout << "Calculating acc..." << std::endl;
            raul::dtype testAcc = cifar.testNetwork(work, std::make_pair(224, 224));
            printf("Test accuracy = %.2f\n", testAcc);
            fflush(stdout);
            EXPECT_NEAR(testAcc, golden_acc_after, acc_eps);
        }
    }

    printf("Memory taken = %.2fMB \n\n", static_cast<float>(work.getMemoryManager().getTotalMemory()) / (1024.0f * 1024.0f));
}

TEST(TestResNet, ResNet18CifarPretrainedModelCheckpointedTrain)
{
    PROFILE_TEST
    constexpr auto lr = 0.05_dt;
    const size_t image_size = 224U;
    const size_t image_channels = 3U;
    const size_t image_classes = 10U;
    const size_t batch_size = 50U;

    // Build work
    raul::Workflow work(raul::CompressionMode::NONE, raul::CalculationMode::DETERMINISTIC, raul::AllocationMode::STANDARD);
    const auto output_name = build_resnet18(work, image_size, image_channels, image_classes);
    work.add<raul::LogSoftMaxActivation>("softmax", raul::BasicParamsWithDim{ { output_name }, { "softmax" } });
    work.add<raul::NLLLoss>("loss", raul::LossParams{ { "softmax", "labels" }, { "loss" }, "batch_mean" });

    raul::Names checkpointsAll = work.getPotentialCheckpoints();
    raul::Names checkpoints;

    raul::Names blocks = { "layer1", "layer2", "layer3" };
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
    const auto weights_path = tools::getTestAssetsDir() / "resnet" / "18" / "seed_0_epoch10_acc_83.24";
    load_resnet_weights(memory_manager, weights_path, "83.24_resnet.", true);

    // Dataset initialization
    raul::CIFAR10 cifar;
    std::cout << "Loading CIFAR...";
    ASSERT_TRUE(cifar.loadingData(tools::getTestAssetsDir() / "CIFAR"));
    std::cout << "done" << std::endl;

    printf("Memory taken = %.2fMB \n\n", static_cast<float>(work.getMemoryManager().getTotalMemory()) / (1024.0f * 1024.0f));
    fflush(stdout);

    auto sgd = std::make_shared<raul::optimizers::SGD>(lr);

    std::vector<raul::dtype> goldLoss = { 0.0360929221_dt, 0.0294999164_dt };

    for (size_t q = 0; q < 2; ++q)
    {
        raul::dtype testLoss = cifar.oneTrainIteration(work, sgd.get(), q, std::make_pair(224, 224));
        EXPECT_NEAR(testLoss, goldLoss[q], 1e-6);
    }
}

TEST(TestResNet, ResNet18CifarPretrainedModelCheckpointedPoolTrain)
{
    PROFILE_TEST
    constexpr auto lr = 0.05_dt;
    const size_t image_size = 224U;
    const size_t image_channels = 3U;
    const size_t image_classes = 10U;
    const size_t batch_size = 50U;

    // Build work
    raul::Workflow work(raul::CompressionMode::NONE, raul::CalculationMode::DETERMINISTIC, raul::AllocationMode::POOL);
    const auto output_name = build_resnet18(work, image_size, image_channels, image_classes);
    work.add<raul::LogSoftMaxActivation>("softmax", raul::BasicParamsWithDim{ { output_name }, { "softmax" } });
    work.add<raul::NLLLoss>("loss", raul::LossParams{ { "softmax", "labels" }, { "loss" }, "batch_mean" });

    raul::Names checkpointsAll = work.getPotentialCheckpoints();
    raul::Names checkpoints;

    raul::Names blocks = { "layer1", "layer2", "layer3" };
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
    const auto weights_path = tools::getTestAssetsDir() / "resnet" / "18" / "seed_0_epoch10_acc_83.24";
    load_resnet_weights(memory_manager, weights_path, "83.24_resnet.", true);

    // Dataset initialization
    raul::CIFAR10 cifar;
    std::cout << "Loading CIFAR...";
    ASSERT_TRUE(cifar.loadingData(tools::getTestAssetsDir() / "CIFAR"));
    std::cout << "done" << std::endl;

    printf("Memory taken = %.2fMB \n\n", static_cast<float>(work.getMemoryManager().getTotalMemory()) / (1024.0f * 1024.0f));
    fflush(stdout);

    auto sgd = std::make_shared<raul::optimizers::SGD>(lr);

    std::vector<raul::dtype> goldLoss = { 0.0360929221_dt, 0.0294999164_dt };

    for (size_t q = 0; q < 2; ++q)
    {
        raul::dtype testLoss = cifar.oneTrainIteration(work, sgd.get(), q, std::make_pair(224, 224));
        EXPECT_NEAR(testLoss, goldLoss[q], 1e-6);
    }
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
    raul::Workflow work(raul::CompressionMode::FP16, raul::CalculationMode::DETERMINISTIC, raul::AllocationMode::STANDARD, raul::ExecutionTarget::CPU, &quantizer);
    const auto output_name = build_resnet18(work, image_size, image_channels, image_classes, batchtornm_momentum, bias, quantized);

    work.preparePipelines();
    work.setBatchSize(batch_size);
    work.prepareMemoryForTraining();

    work.printInfo(std::cout);

    // Checks
    EXPECT_EQ(tools::get_size_of_trainable_params(work), golden_trainable_parameters);
}

// TEST(TestResNet, Experiments)
//{
//    const char* env_raul_resnet_batch = std::getenv("RAUL_RESNET_BATCH");
//    const char* env_raul_resnet_quant = std::getenv("RAUL_RESNET_QUANT");
//
//    constexpr auto acc_eps = 1e-1_dt;
//    constexpr auto lr = 0.05_dt;
//    const size_t image_size = 224U;
//    const size_t image_channels = 3U;
//    const size_t image_classes = 10U;
//    const size_t batch_size = env_raul_resnet_batch == nullptr ? 50U : std::stoi(env_raul_resnet_batch);
//    const bool quantized = env_raul_resnet_quant == nullptr ? false : std::stoi(env_raul_resnet_quant);
//    const float batchtornm_momentum = 0.1f;
//    const bool bias = false;
//    const size_t print_freq = 100U;
//    const size_t epochs = 10U;
//
//    const auto golden_acc_before = 83.24_dt;
//
//    // Config
//    std::cout << "Batch size = " << batch_size << std::endl;
//    std::cout << "Quantized = " << quantized << std::endl;
//
//    {
//        std::cout << "App loaded" << std::endl;
//        std::time_t result = std::time(nullptr);
//        std::cout << std::asctime(std::localtime(&result)) << result << " seconds since the Epoch\n";
//    }
//
//    // Build work
//    raul::Workflow work;
//    const auto output_name = build_resnet18(work, image_size, image_channels, image_classes, batchtornm_momentum, bias, quantized);
//    work.add<raul::LogSoftMaxActivation>(  "softmax", raul::BasicParamsWithDim{ { output_name }, { "softmax" } });
//    work.add<raul::NLLLoss>(  "loss", raul::LossParams{ { "softmax", "labels" }, { "loss" }, "mean" });
//    auto quantizer = raul::quantization::SymmetricQuantizer(static_cast<raul::dtype (*)(raul::dtype)>(std::trunc));
//    raul::Graph work(std::move(work), batch_size, {}, raul::CompressionMode::FP16, raul::CalculationMode::DETERMINISTIC, &quantizer);
//
//    // Initialization of the work
//    auto& memory_manager = work.getMemoryManager();
//    const auto weights_path = tools::getTestAssetsDir() / "resnet" / "18" / "seed_0_epoch20_acc_85.94";
//    load_resnet_weights(memory_manager, weights_path, "85.94_resnet.", true, true);
//
//    // Dataset initialization
//    raul::CIFAR10 cifar;
//    std::cout << "Loading CIFAR...";
//    ASSERT_TRUE(cifar.loadingData(tools::getTestAssetsDir() / "CIFAR"));
//    std::cout << "done" << std::endl;
//
//    printf("Memory taken = %.2fMB \n\n", work.getMemoryManager().getTotalMemory() / (1024.0f * 1024.0f));
//    fflush(stdout);
//
//    // Before train
//    {
//        std::cout << "Calculating acc..." << std::endl;
//        raul::dtype testAcc = cifar.testNetwork(work, std::make_pair(224, 224));
//        printf("Testing taken = %.3fs \n", cifar.getTestingTime());
//        printf("Test accuracy = %.2f\n", testAcc);
//        fflush(stdout);
//        EXPECT_NEAR(testAcc, golden_acc_before, acc_eps);
//    }
//    // Trainings
//    const size_t stepsAmountTrain = cifar.getTrainImageAmount() / batch_size;
//    auto sgd = std::make_shared<raul::optimizers::SGD>(lr);
//    std::cout << "Training..." << std::endl;
//
//    for (size_t epoch = 1; epoch <= epochs; ++epoch)
//    {
//        printf("Epoch = %zu\n", epoch);
//
//        for (size_t q = 0; q < stepsAmountTrain; ++q)
//        {
//            raul::dtype testLoss = cifar.oneTrainIteration(work, sgd.get(), q, std::make_pair(224, 224));
//            if (q % print_freq == 0)
//            {
//                printf("iteration = %zu/%zu, loss = %f\n", q, stepsAmountTrain, testLoss);
//                fflush(stdout);
//            }
//        }
//
//        printf("Epoch Training taken = %.3fs \n", cifar.getTrainingTime());
//
//        // After train
//        {
//            std::cout << "Calculating acc..." << std::endl;
//            raul::dtype testAcc = cifar.testNetwork(work, std::make_pair(224, 224));
//            printf("Test accuracy = %.2f\n", testAcc);
//            fflush(stdout);
//        }
//    }
//
//    printf("Memory taken = %.2fMB \n\n", work.getMemoryManager().getTotalMemory() / (1024.0f * 1024.0f));
//}

} // namespace UT
