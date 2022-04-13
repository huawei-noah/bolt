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

#include <training/base/common/quantization/SymmetricQuantizer.h>
#include <training/base/layers/basic/trainable/Convolution2DLayer.h>
#include <training/compiler/Layers.h>
#include <training/compiler/Workflow.h>
#include <training/base/optimizers/SGD.h>

#include <chrono>

namespace UT
{

using namespace raul;

TEST(TestCNN2DLayer, BiasesUnit)
{
    PROFILE_TEST
    const size_t KERNEL_SIZE = 3;
    const size_t FILTERS = 2;
    const size_t STRIDE = 1;
    const size_t PADDING = 0;

    const dtype EPSILON = TODTYPE(1e-6);

    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    Tensor input = { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f };

    work.add<raul::DataLayer>("data", DataParams{ { "in" }, 1, 3, 3 });
    Convolution2DLayer cnnLayer("cnn1", Convolution2DParams{ { "in" }, { "cnn1" }, KERNEL_SIZE, FILTERS, STRIDE, PADDING, true }, networkParameters);
    TENSORS_CREATE(1);

    memory_manager["in"] = TORANGE(input);
    memory_manager["cnn1::Weights"] = 1_dt;
    memory_manager["cnn1::Biases"] = TORANGE((Tensor{ 2.0f, 3.0f }));
    ASSERT_NO_THROW(memory_manager["cnn1::Biases"]);
    ASSERT_NO_THROW(memory_manager["cnn1::BiasesGradient"]);

    ASSERT_NO_THROW(cnnLayer.forwardCompute(NetworkMode::Train));

    EXPECT_EQ(memory_manager["cnn1"].size(), static_cast<size_t>(FILTERS));
    CHECK_NEAR(memory_manager["cnn1"][0], 11.0f, EPSILON);
    CHECK_NEAR(memory_manager["cnn1"][1], 12.0f, EPSILON);

    memory_manager[Name("cnn1").grad()] = 1_dt;

    ASSERT_NO_THROW(cnnLayer.backwardCompute());
}

TEST(TestCNN2DLayer, Biases2Unit)
{
    PROFILE_TEST

    const size_t KERNEL_SIZE = 3;
    const size_t FILTERS = 2;
    const size_t STRIDE = 1;
    const size_t PADDING = 0;

    const dtype EPSILON = TODTYPE(1e-6);

    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    work.add<raul::DataLayer>("data", DataParams{ { "in" }, 1, 3, 3 });
    Convolution2DLayer cnnLayer("cnn1", Convolution2DParams{ { "in" }, { "cnn1" }, KERNEL_SIZE, FILTERS, STRIDE, PADDING, false }, networkParameters);
    TENSORS_CREATE(1);

    memory_manager["in"] = 1_dt;
    memory_manager["cnn1::Weights"] = 1_dt;
    ASSERT_THROW(memory_manager["cnn1::Biases"], raul::Exception);
    ASSERT_THROW(memory_manager["cnn1::BiasesGradient"], raul::Exception);

    ASSERT_NO_THROW(cnnLayer.forwardCompute(NetworkMode::Train));

    EXPECT_EQ(memory_manager["cnn1"].size(), static_cast<size_t>(FILTERS));
    CHECK_NEAR(memory_manager["cnn1"][0], 9.0f, EPSILON);
    CHECK_NEAR(memory_manager["cnn1"][1], 9.0f, EPSILON);

    memory_manager[Name("cnn1").grad()] = 1_dt;

    ASSERT_NO_THROW(cnnLayer.backwardCompute());
}

TEST(TestCNN2DLayer, QuantizeWeightsUnit)
{
    PROFILE_TEST

    const size_t KERNEL_SIZE = 3;
    const size_t FILTERS = 1;
    const size_t STRIDE = 1;
    const size_t PADDING = 0;

    const dtype EPSILON = 1e-6_dt;

    auto quantizer = quantization::SymmetricQuantizer(static_cast<dtype (*)(dtype)>(std::trunc));

    WorkflowEager work(CompressionMode::NONE, CalculationMode::DETERMINISTIC, raul::AllocationMode::STANDARD, raul::ExecutionTarget::CPU, false, &quantizer);
    MemoryManager& memory_manager = work.getMemoryManager();
    NETWORK_PARAMS_DEFINE(net_params)

    work.add<raul::DataLayer>("data", DataParams{ { "in" }, 1, 3, 3 });
    Convolution2DLayer cnnLayer("cnn", Convolution2DParams{ { "in" }, { "cnn" }, KERNEL_SIZE, FILTERS, STRIDE, PADDING, false, true }, net_params);
    TENSORS_CREATE(1)
    memory_manager["in"] = 1.0_dt;
    memory_manager["cnn::Weights"] = 1.0_dt;

    ASSERT_NO_THROW(memory_manager["cnn::Weights_backup"]);

    ASSERT_NO_THROW(cnnLayer.forwardCompute(NetworkMode::Train));

    EXPECT_EQ(memory_manager["cnn"].size(), FILTERS);
    CHECK_NEAR(memory_manager["cnn"][0], 9.0_dt, EPSILON);

    for (const auto& val : memory_manager["cnn::Weights_backup"])
    {
        CHECK_NEAR(val, 1.0_dt, EPSILON);
    }

    memory_manager[Name("cnn").grad()] = 1.0_dt;

    ASSERT_NO_THROW(cnnLayer.backwardCompute());
}

TEST(TestCNN2DLayer, TimeMeasurementUnit)
{
    PROFILE_TEST
    const size_t BATCH_SIZE = 200;
    const size_t KERNEL_SIZE = 3;
    const size_t FILTERS = 16;
    const size_t STRIDE = 1;
    const size_t PADDING = 0;
    const size_t INITIAL_SIZE = 28;
    const size_t INITIAL_DEPTH = 3;

    const size_t FORWARD_STEPS = 100;
    std::vector<double> time(FORWARD_STEPS, 0.);

    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);
    work.add<raul::DataLayer>("data", DataParams{ { "in" }, INITIAL_DEPTH, INITIAL_SIZE, INITIAL_SIZE });
    Convolution2DLayer cnnLayer("cnn", Convolution2DParams{ { "in" }, { "cnn" }, KERNEL_SIZE, FILTERS, STRIDE, PADDING }, networkParameters);

    TENSORS_CREATE(BATCH_SIZE)
    memory_manager["in"] = 1_dt;
    memory_manager["cnn::Weights"] = 1_dt;

    for (size_t step = 0; step < FORWARD_STEPS; ++step)
    {
        auto start = std::chrono::high_resolution_clock::now();
        cnnLayer.forwardCompute(NetworkMode::Train);
        time[step] = static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count());
    }
    auto mean = std::accumulate(time.begin(), time.end(), 0.0) / FORWARD_STEPS;
    auto dispersion = 0.0;
    std::for_each(time.begin(), time.end(), [&](const auto d) { dispersion += (d - mean) * (d - mean); });
    auto stddev = std::sqrt(dispersion / static_cast<double>(time.size() - 1));
    std::cout << "ConvolutionLayer forward time: mean = " << mean << ", standard deviation = " << stddev << std::endl;
}

TEST(TestCNN2DLayer, SimpleDilationUnit)
{
    PROFILE_TEST
    const size_t KERNEL_SIZE = 2;
    const size_t FILTERS = 1;
    const size_t STRIDE = 1;
    const size_t PADDING = 0;
    const size_t DILATION = 2;

    const dtype EPSILON = TODTYPE(1e-6);

    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    const Tensor input = { 1.92691529_dt,  1.48728406_dt,  0.90071720_dt,  -2.10552096_dt, 0.67841846_dt,  -1.23454487_dt, -0.04306748_dt, -1.60466695_dt, -0.75213528_dt,
                           -0.68662298_dt, -0.49335635_dt, 0.24148779_dt,  -1.11090386_dt, 0.09154566_dt,  -2.31692266_dt, -0.21680473_dt, -1.38467371_dt, -0.39571050_dt,
                           0.80340934_dt,  -0.62159538_dt, -0.59200054_dt, -0.06307438_dt, -0.82855427_dt, 0.33089843_dt,  -1.55757248_dt };

    const Tensor realOutput = { 0.84618479_dt, -0.70337778_dt, 1.48831379_dt, -0.19095580_dt, 0.07261647_dt, 0.21341163_dt, 0.25238249_dt, -0.07215345_dt, 0.29681736_dt };

    const Tensor realInputNabla = { -0.00374341_dt, -0.00374341_dt, 0.26447839_dt,  0.26822180_dt,  0.26822180_dt,  -0.00374341_dt, -0.00374341_dt, 0.26447839_dt,  0.26822180_dt,
                                    0.26822180_dt,  -0.41526598_dt, -0.41526598_dt, -0.51501369_dt, -0.09974772_dt, -0.09974772_dt, -0.41152257_dt, -0.41152257_dt, -0.77949208_dt,
                                    -0.36796951_dt, -0.36796951_dt, -0.41152257_dt, -0.41152257_dt, -0.77949208_dt, -0.36796951_dt, -0.36796951_dt };

    const Tensor realWeightsGrad = { 0.06986499_dt, -6.90609121_dt, -4.84359074_dt, -5.60540581_dt };

    work.add<raul::DataLayer>("data", DataParams{ { "in" }, 1, 5, 5 });
    Convolution2DLayer cnnLayer("cnn1", Convolution2DParams{ { "in" }, { "cnn1" }, KERNEL_SIZE, FILTERS, STRIDE, PADDING, false, false, DILATION }, networkParameters);

    TENSORS_CREATE(1);

    memory_manager["in"] = TORANGE(input);
    memory_manager["cnn1::Weights"] = TORANGE((Tensor{ -0.00374341_dt, 0.26822180_dt, -0.41152257_dt, -0.36796951_dt }));
    ASSERT_THROW(memory_manager["cnn1::Biases"], raul::Exception);
    ASSERT_THROW(memory_manager["cnn1::BiasesGradient"], raul::Exception);

    ASSERT_NO_THROW(cnnLayer.forwardCompute(NetworkMode::Train));

    const auto output = memory_manager["cnn1"];
    EXPECT_EQ(output.size(), realOutput.size());
    for (size_t i = 0; i < output.size(); ++i)
    {
        CHECK_NEAR(output[i], realOutput[i], EPSILON);
    }

    memory_manager[Name("cnn1").grad()] = 1_dt;

    ASSERT_NO_THROW(cnnLayer.backwardCompute());
    const auto inputNabla = memory_manager[Name("in").grad()];
    EXPECT_EQ(inputNabla.size(), realInputNabla.size());
    for (size_t i = 0; i < inputNabla.size(); ++i)
    {
        CHECK_NEAR(inputNabla[i], realInputNabla[i], EPSILON);
    }
    const auto weightsGrad = memory_manager[Name("cnn1::Weights").grad()];
    EXPECT_EQ(weightsGrad.size(), realWeightsGrad.size());
    for (size_t i = 0; i < weightsGrad.size(); ++i)
    {
        CHECK_NEAR(weightsGrad[i], realWeightsGrad[i], EPSILON);
    }
}

TEST(TestCNN2DLayer, DilationUnit)
{
    PROFILE_TEST
    const size_t KERNEL_SIZE = 3;
    const size_t FILTERS = 3;
    const size_t STRIDE = 2;
    const size_t PADDING = 2;
    const size_t DILATION = 3;
    const size_t BATCH = 2;

    const dtype EPSILON = TODTYPE(1e-6);

    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    const Tensor input = { 1.92691529_dt,  1.48728406_dt,  0.90071720_dt,  -2.10552096_dt, 0.67841846_dt,  -1.23454487_dt, -0.04306748_dt, -1.60466695_dt, -0.75213528_dt, 1.64872301_dt,
                           -0.39247864_dt, -1.40360713_dt, -0.72788131_dt, -0.55943018_dt, -0.76883888_dt, 0.76244539_dt,  1.64231694_dt,  -0.15959747_dt, -0.49739754_dt, 0.43958926_dt,
                           -0.75813115_dt, 1.07831764_dt,  0.80080056_dt,  1.68062055_dt,  1.27912438_dt,  1.29642284_dt,  0.61046648_dt,  1.33473778_dt,  -0.23162432_dt, 0.04175949_dt,
                           -0.25157529_dt, 0.85985851_dt,  -1.38467371_dt, -0.87123615_dt, -0.22336592_dt, 1.71736145_dt,  0.31888032_dt,  -0.42451897_dt, 0.30572093_dt,  -0.77459252_dt,
                           -1.55757248_dt, 0.99563611_dt,  -0.87978584_dt, -0.60114205_dt, -1.27415121_dt, 2.12278509_dt,  -1.23465312_dt, -0.48791388_dt, -0.91382301_dt, -0.65813726_dt,
                           0.07802387_dt,  0.52580875_dt,  -0.48799172_dt, 1.19136906_dt,  -0.81400764_dt, -0.73599279_dt, -1.40324783_dt, 0.03600367_dt,  -0.06347727_dt, 0.67561489_dt,
                           -0.09780689_dt, 1.84459400_dt,  -1.18453741_dt, 1.38354933_dt,  1.44513381_dt,  0.85641253_dt,  2.21807575_dt,  0.52316552_dt,  0.34664667_dt,  -0.19733144_dt,
                           -1.05458891_dt, 1.27799559_dt,  -0.17219013_dt, 0.52378845_dt,  0.05662182_dt,  0.42629614_dt,  0.57500505_dt,  -0.64172411_dt, -2.20639849_dt, -0.75080305_dt,
                           0.01086814_dt,  -0.33874235_dt, -1.34067953_dt, -0.58537054_dt, 0.64075530_dt,  0.58324742_dt,  1.06692672_dt,  -0.45015338_dt, -0.67875296_dt, 0.57431608_dt,
                           0.18774910_dt,  -0.35762301_dt, 0.26490951_dt,  1.27316833_dt,  -0.00131086_dt, -0.30360377_dt, -0.98643863_dt, 0.12329912_dt,  0.34986776_dt,  0.61728072_dt };

    const Tensor realOutput = { -0.03776294_dt, 0.72608572_dt,  -0.14190413_dt, 0.18520108_dt,  1.15060508_dt,  -0.36858386_dt, 0.20764011_dt, -0.14685461_dt,
                                -0.83236200_dt, -0.10481618_dt, 0.23594269_dt,  -0.63483143_dt, 0.39375004_dt,  0.35512355_dt,  0.06034210_dt, -0.08678990_dt,
                                0.09048741_dt,  -0.23272288_dt, -0.37920359_dt, 0.49226546_dt,  -0.36308318_dt, -0.27676097_dt, 0.05685749_dt, -0.07759568_dt };

    const Tensor realInputNabla = { -0.29914585_dt, 0.22641060_dt,  0._dt, 0.22641060_dt,  -0.00320783_dt, 0.13650024_dt,  -0.12689337_dt, 0._dt, -0.12689337_dt, 0.11874340_dt,
                                    0._dt,          0._dt,          0._dt, 0._dt,          0._dt,          0.13650024_dt,  -0.12689337_dt, 0._dt, -0.12689337_dt, 0.11874340_dt,
                                    -0.00278408_dt, -0.00824748_dt, 0._dt, -0.00824748_dt, 0.02219512_dt,  -0.27779582_dt, -0.07090232_dt, 0._dt, -0.07090232_dt, 0.03219472_dt,
                                    -0.42158729_dt, 0.03904085_dt,  0._dt, 0.03904085_dt,  -0.09919342_dt, 0._dt,          0._dt,          0._dt, 0._dt,          0._dt,
                                    -0.42158729_dt, 0.03904085_dt,  0._dt, 0.03904085_dt,  -0.09919342_dt, 0.13454917_dt,  0.02205629_dt,  0._dt, 0.02205629_dt,  -0.04748112_dt,
                                    -0.29914585_dt, 0.22641060_dt,  0._dt, 0.22641060_dt,  -0.00320783_dt, 0.13650024_dt,  -0.12689337_dt, 0._dt, -0.12689337_dt, 0.11874340_dt,
                                    0._dt,          0._dt,          0._dt, 0._dt,          0._dt,          0.13650024_dt,  -0.12689337_dt, 0._dt, -0.12689337_dt, 0.11874340_dt,
                                    -0.00278408_dt, -0.00824748_dt, 0._dt, -0.00824748_dt, 0.02219512_dt,  -0.27779582_dt, -0.07090232_dt, 0._dt, -0.07090232_dt, 0.03219472_dt,
                                    -0.42158729_dt, 0.03904085_dt,  0._dt, 0.03904085_dt,  -0.09919342_dt, 0._dt,          0._dt,          0._dt, 0._dt,          0._dt,
                                    -0.42158729_dt, 0.03904085_dt,  0._dt, 0.03904085_dt,  -0.09919342_dt, 0.13454917_dt,  0.02205629_dt,  0._dt, 0.02205629_dt,  -0.04748112_dt };

    const Tensor realWeightsGrad = { 2.00493908_dt, 1.09894097_dt,  -0.13558918_dt, -0.35167974_dt, 1.44771397_dt, 2.56659555_dt,  -1.81272006_dt, 4.56072235_dt,  1.33574617_dt,
                                     1.72271895_dt, -1.25255132_dt, -0.70904356_dt, -1.61053061_dt, 0.37454879_dt, -0.85807270_dt, 1.81918132_dt,  -2.78504705_dt, -0.04085654_dt,
                                     2.00493908_dt, 1.09894097_dt,  -0.13558918_dt, -0.35167974_dt, 1.44771397_dt, 2.56659555_dt,  -1.81272006_dt, 4.56072235_dt,  1.33574617_dt,
                                     1.72271895_dt, -1.25255132_dt, -0.70904356_dt, -1.61053061_dt, 0.37454879_dt, -0.85807270_dt, 1.81918132_dt,  -2.78504705_dt, -0.04085654_dt,
                                     2.00493908_dt, 1.09894097_dt,  -0.13558918_dt, -0.35167974_dt, 1.44771397_dt, 2.56659555_dt,  -1.81272006_dt, 4.56072235_dt,  1.33574617_dt,
                                     1.72271895_dt, -1.25255132_dt, -0.70904356_dt, -1.61053061_dt, 0.37454879_dt, -0.85807270_dt, 1.81918132_dt,  -2.78504705_dt, -0.04085654_dt };

    work.add<raul::DataLayer>("data", DataParams{ { "in" }, 2, 5, 5 });
    Convolution2DLayer cnnLayer("cnn1", Convolution2DParams{ { "in" }, { "cnn1" }, KERNEL_SIZE, FILTERS, STRIDE, PADDING, false, false, DILATION }, networkParameters);
    TENSORS_CREATE(BATCH);

    memory_manager["in"] = TORANGE(input);
    memory_manager["cnn1::Weights"] =
        TORANGE((Tensor{ -0.00176466_dt, 0.12644097_dt,  -0.19399360_dt, -0.17346250_dt, -0.09078176_dt, 0.06320529_dt,  -0.00467001_dt, 0.18688585_dt,  -0.02091717_dt, 0.06236978_dt, -0.07123230_dt,
                         -0.04633091_dt, -0.22517779_dt, -0.15610139_dt, -0.09716192_dt, 0.00873125_dt,  0.09318140_dt,  0.14142673_dt,  -0.15979224_dt, -0.10263957_dt, 0.08561110_dt, 0.19572432_dt,
                         -0.04850757_dt, 0.17637877_dt,  -0.03799128_dt, 0.02494062_dt,  0.21342279_dt,  -0.21865401_dt, -0.14838351_dt, -0.05967162_dt, -0.09187673_dt, 0.20364694_dt, -0.15277740_dt,
                         -0.10850150_dt, -0.16467114_dt, -0.22074954_dt, -0.13758895_dt, 0.20260920_dt,  0.10517468_dt,  0.11423842_dt,  0.01239595_dt,  -0.12084066_dt, 0.03987721_dt, -0.22007395_dt,
                         -0.17031050_dt, -0.12151159_dt, 0.14871350_dt,  0.13819724_dt,  -0.10453279_dt, -0.00850470_dt, 0.15074590_dt,  0.23431942_dt,  0.09354603_dt,  0.03184169_dt }));
    ASSERT_THROW(memory_manager["cnn1::Biases"], raul::Exception);
    ASSERT_THROW(memory_manager["cnn1::BiasesGradient"], raul::Exception);

    ASSERT_NO_THROW(cnnLayer.forwardCompute(NetworkMode::Train));

    const auto output = memory_manager["cnn1"];
    EXPECT_EQ(output.size(), realOutput.size());
    for (size_t i = 0; i < output.size(); ++i)
    {
        CHECK_NEAR(output[i], realOutput[i], EPSILON);
    }

    memory_manager[Name("cnn1").grad()] = 1_dt;

    ASSERT_NO_THROW(cnnLayer.backwardCompute());
    const auto inputNabla = memory_manager[Name("in").grad()];
    EXPECT_EQ(inputNabla.size(), realInputNabla.size());
    for (size_t i = 0; i < inputNabla.size(); ++i)
    {
        CHECK_NEAR(inputNabla[i], realInputNabla[i], EPSILON);
    }
    const auto weightsGrad = memory_manager[Name("cnn1::Weights").grad()];
    EXPECT_EQ(weightsGrad.size(), realWeightsGrad.size());
    for (size_t i = 0; i < weightsGrad.size(); ++i)
    {
        CHECK_NEAR(weightsGrad[i], realWeightsGrad[i], EPSILON);
    }
}

TEST(TestCNN2DLayer, Dilation2Unit)
{
    PROFILE_TEST
    const size_t KERNEL_SIZE_H = 1;
    const size_t KERNEL_SIZE_W = 3;
    const size_t FILTERS = 2;
    const size_t STRIDE_H = 1;
    const size_t STRIDE_W = 3;
    const size_t PADDING_H = 3;
    const size_t PADDING_W = 1;
    const size_t DILATION_H = 2;
    const size_t DILATION_W = 3;
    const size_t BATCH = 2;

    const dtype EPSILON = TODTYPE(1e-6);

    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    const Tensor input = { 1.92691529_dt,  1.48728406_dt,  0.90071720_dt,  -2.10552096_dt, 0.67841846_dt,  -1.23454487_dt, -0.04306748_dt, -1.60466695_dt, -0.75213528_dt, 1.64872301_dt,
                           -0.39247864_dt, -1.40360713_dt, -0.72788131_dt, -0.55943018_dt, -0.76883888_dt, 0.76244539_dt,  1.64231694_dt,  -0.15959747_dt, -0.49739754_dt, 0.43958926_dt,
                           -0.75813115_dt, 1.07831764_dt,  0.80080056_dt,  1.68062055_dt,  1.27912438_dt,  1.29642284_dt,  0.61046648_dt,  1.33473778_dt,  -0.23162432_dt, 0.04175949_dt,
                           -0.25157529_dt, 0.85985851_dt,  -1.38467371_dt, -0.87123615_dt, -0.22336592_dt, 1.71736145_dt,  0.31888032_dt,  -0.42451897_dt, 0.30572093_dt,  -0.77459252_dt,
                           -1.55757248_dt, 0.99563611_dt,  -0.87978584_dt, -0.60114205_dt, -1.27415121_dt, 2.12278509_dt,  -1.23465312_dt, -0.48791388_dt, -0.91382301_dt, -0.65813726_dt,
                           0.07802387_dt,  0.52580875_dt,  -0.48799172_dt, 1.19136906_dt,  -0.81400764_dt, -0.73599279_dt, -1.40324783_dt, 0.03600367_dt,  -0.06347727_dt, 0.67561489_dt,
                           -0.09780689_dt, 1.84459400_dt,  -1.18453741_dt, 1.38354933_dt,  1.44513381_dt,  0.85641253_dt,  2.21807575_dt,  0.52316552_dt,  0.34664667_dt,  -0.19733144_dt,
                           -1.05458891_dt, 1.27799559_dt,  -0.17219013_dt, 0.52378845_dt,  0.05662182_dt,  0.42629614_dt,  0.57500505_dt,  -0.64172411_dt, -2.20639849_dt, -0.75080305_dt,
                           0.01086814_dt,  -0.33874235_dt, -1.34067953_dt, -0.58537054_dt, 0.53618813_dt,  0.52462262_dt,  1.14120162_dt,  0.05164360_dt,  0.74395198_dt,  -0.48158440_dt,
                           -1.04946613_dt, 0.60389882_dt,  -1.72229505_dt, -0.82776886_dt, 1.33470297_dt,  0.48353928_dt,  -2.50954437_dt, 0.48800105_dt,  0.78458685_dt,  0.02864719_dt,
                           0.64075530_dt,  0.58324742_dt,  1.06692672_dt,  -0.45015338_dt, -0.18526748_dt, 0.75275886_dt,  0.40475780_dt,  0.17846599_dt,  0.26490951_dt,  1.27316833_dt,
                           -0.00131086_dt, -0.30360377_dt, -1.45702910_dt, -0.10233524_dt, -0.59915304_dt, 0.47705641_dt,  0.72617722_dt,  0.09115186_dt,  -0.38906521_dt, 0.52791649_dt,
                           -0.01268548_dt, 0.24083632_dt,  0.13253537_dt,  0.76424062_dt,  1.09500968_dt,  0.33989096_dt,  0.71996748_dt,  0.41140762_dt,  1.93116057_dt,  1.01186383_dt,
                           -1.43640649_dt, -1.12985981_dt, -0.13603453_dt, 1.63540959_dt,  -0.73280275_dt, 0.10429783_dt,  1.04140103_dt,  -0.39973062_dt, -2.29333448_dt, 0.49756259_dt,
                           -0.42572311_dt, -1.33714700_dt, -1.19545376_dt, 0.81233692_dt,  -0.30627838_dt, -0.33015838_dt, -0.98080349_dt, 0.19473360_dt,  -1.65352094_dt, 0.68141943_dt };

    const Tensor realOutput = { -0.22598037_dt, -0.22598037_dt, -0.22598037_dt, -0.36525360_dt, -0.32563144_dt, -0.61470342_dt, -0.00329676_dt, -0.06565411_dt, -0.22598037_dt,
                                -0.22598037_dt, -0.22598037_dt, -0.14515428_dt, -0.14515428_dt, -0.14515428_dt, -0.59485489_dt, 0.32692224_dt,  -0.13420853_dt, 0.13408725_dt,
                                -0.14080381_dt, -0.14515428_dt, -0.14515428_dt, -0.14515428_dt, -0.22598037_dt, -0.22598037_dt, -0.22598037_dt, -0.36897355_dt, -0.52457917_dt,
                                -0.13533276_dt, -0.86160851_dt, -0.10426680_dt, -0.22598037_dt, -0.22598037_dt, -0.22598037_dt, -0.14515428_dt, -0.14515428_dt, -0.14515428_dt,
                                -0.26182932_dt, -0.06742202_dt, 0.11862217_dt,  -0.14931199_dt, -0.19791129_dt, -0.14515428_dt, -0.14515428_dt, -0.14515428_dt };

    const Tensor realInputNabla = { 0._dt, 0._dt, 0.07807684_dt,  0._dt, 0._dt, 0._dt, 0._dt, 0.07807684_dt,  0._dt, 0._dt, 0._dt, 0._dt, 0.07807684_dt,  0._dt, 0._dt,
                                    0._dt, 0._dt, 0.07807684_dt,  0._dt, 0._dt, 0._dt, 0._dt, 0.07807684_dt,  0._dt, 0._dt, 0._dt, 0._dt, -0.34914550_dt, 0._dt, 0._dt,
                                    0._dt, 0._dt, -0.34914550_dt, 0._dt, 0._dt, 0._dt, 0._dt, -0.34914550_dt, 0._dt, 0._dt, 0._dt, 0._dt, -0.34914550_dt, 0._dt, 0._dt,
                                    0._dt, 0._dt, -0.34914550_dt, 0._dt, 0._dt, 0._dt, 0._dt, 0.39607489_dt,  0._dt, 0._dt, 0._dt, 0._dt, 0.39607489_dt,  0._dt, 0._dt,
                                    0._dt, 0._dt, 0.39607489_dt,  0._dt, 0._dt, 0._dt, 0._dt, 0.39607489_dt,  0._dt, 0._dt, 0._dt, 0._dt, 0.39607489_dt,  0._dt, 0._dt,
                                    0._dt, 0._dt, 0.07807684_dt,  0._dt, 0._dt, 0._dt, 0._dt, 0.07807684_dt,  0._dt, 0._dt, 0._dt, 0._dt, 0.07807684_dt,  0._dt, 0._dt,
                                    0._dt, 0._dt, 0.07807684_dt,  0._dt, 0._dt, 0._dt, 0._dt, 0.07807684_dt,  0._dt, 0._dt, 0._dt, 0._dt, -0.34914550_dt, 0._dt, 0._dt,
                                    0._dt, 0._dt, -0.34914550_dt, 0._dt, 0._dt, 0._dt, 0._dt, -0.34914550_dt, 0._dt, 0._dt, 0._dt, 0._dt, -0.34914550_dt, 0._dt, 0._dt,
                                    0._dt, 0._dt, -0.34914550_dt, 0._dt, 0._dt, 0._dt, 0._dt, 0.39607489_dt,  0._dt, 0._dt, 0._dt, 0._dt, 0.39607489_dt,  0._dt, 0._dt,
                                    0._dt, 0._dt, 0.39607489_dt,  0._dt, 0._dt, 0._dt, 0._dt, 0.39607489_dt,  0._dt, 0._dt, 0._dt, 0._dt, 0.39607489_dt,  0._dt, 0._dt };
    const Tensor realWeightsGrad = { 0._dt, -3.95568180_dt, 0._dt, 0._dt, -1.83010375_dt, 0._dt, 0._dt, -2.41062760_dt, 0._dt,
                                     0._dt, -3.95568180_dt, 0._dt, 0._dt, -1.83010375_dt, 0._dt, 0._dt, -2.41062760_dt, 0._dt };
    const Tensor realBiasGrad = { 22.0_dt, 22.0_dt };

    work.add<raul::DataLayer>("data", DataParams{ { "in" }, 3, 5, 5 });
    Convolution2DLayer cnnLayer(
        "cnn1", Convolution2DParams{ { "in" }, { "cnn1" }, KERNEL_SIZE_W, KERNEL_SIZE_H, FILTERS, STRIDE_W, STRIDE_H, PADDING_W, PADDING_H, true, false, DILATION_W, DILATION_H }, networkParameters);
    TENSORS_CREATE(BATCH);

    memory_manager["in"] = TORANGE(input);
    memory_manager["cnn1::Weights"] = TORANGE((Tensor{
        -0.00249561_dt,
        0.17881453_dt,
        -0.27434838_dt,
        -0.24531302_dt,
        -0.12838480_dt,
        0.08938579_dt,
        -0.00660439_dt,
        0.26429650_dt,
        -0.02958135_dt,
        0.08820419_dt,
        -0.10073769_dt,
        -0.06552180_dt,
        -0.31844950_dt,
        -0.22076070_dt,
        -0.13740771_dt,
        0.01234786_dt,
        0.13177840_dt,
        0.20000760_dt,
    }));
    memory_manager["cnn1::Biases"] = TORANGE((Tensor{ -0.22598037_dt, -0.14515428_dt }));

    ASSERT_NO_THROW(cnnLayer.forwardCompute(NetworkMode::Train));

    const auto output = memory_manager["cnn1"];
    EXPECT_EQ(output.size(), realOutput.size());
    for (size_t i = 0; i < output.size(); ++i)
    {
        CHECK_NEAR(output[i], realOutput[i], EPSILON);
    }

    memory_manager[Name("cnn1").grad()] = 1_dt;

    ASSERT_NO_THROW(cnnLayer.backwardCompute());
    const auto inputNabla = memory_manager[Name("in").grad()];
    EXPECT_EQ(inputNabla.size(), realInputNabla.size());
    for (size_t i = 0; i < inputNabla.size(); ++i)
    {
        CHECK_NEAR(inputNabla[i], realInputNabla[i], EPSILON);
    }
    const auto weightsGrad = memory_manager[Name("cnn1::Weights").grad()];
    EXPECT_EQ(weightsGrad.size(), realWeightsGrad.size());
    for (size_t i = 0; i < weightsGrad.size(); ++i)
    {
        CHECK_NEAR(weightsGrad[i], realWeightsGrad[i], EPSILON);
    }
    const auto biasGrad = memory_manager[Name("cnn1::Biases").grad()];
    EXPECT_EQ(biasGrad.size(), realBiasGrad.size());
    for (size_t i = 0; i < biasGrad.size(); ++i)
    {
        CHECK_NEAR(biasGrad[i], realBiasGrad[i], EPSILON);
    }
}

TEST(TestCNN2DLayer, IncorrectGroupsUnit)
{
    PROFILE_TEST
    const size_t KERNEL_SIZE_H = 1;
    const size_t KERNEL_SIZE_W = 3;
    const size_t FILTERS = 2;
    const size_t STRIDE_H = 1;
    const size_t STRIDE_W = 3;
    const size_t PADDING_H = 3;
    const size_t PADDING_W = 1;
    const size_t DILATION_H = 2;
    const size_t DILATION_W = 3;
    const size_t GROUPS = 0;

    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    ASSERT_THROW(Convolution2DLayer cnnLayer(
                     "cnn1",
                     Convolution2DParams{ { "in" }, { "cnn1" }, KERNEL_SIZE_W, KERNEL_SIZE_H, FILTERS, STRIDE_W, STRIDE_H, PADDING_W, PADDING_H, true, false, DILATION_W, DILATION_H, GROUPS },
                     networkParameters),
                 raul::Exception);
}

TEST(TestCNN2DLayer, SimpleDepthwiseUnit)
{
    PROFILE_TEST
    const size_t KERNEL_SIZE = 3;
    const size_t FILTERS = 2;
    const size_t STRIDE = 1;
    const size_t PADDING = 0;
    const size_t DILATION = 1;
    const size_t BATCH = 2;
    const size_t GROUPS = 2;

    const dtype EPSILON = TODTYPE(1e-6);

    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    const Tensor input = { 0.6995424032_dt,  0.1990816295_dt,  0.8656923771_dt,  0.2444039285_dt,  -0.6629113555_dt, 0.8073082566_dt,  0.4391415715_dt,  1.1712007523_dt,  -2.2455577850_dt,
                           -1.4464579821_dt, 0.0611552820_dt,  -0.6177445054_dt, -0.7980698347_dt, -0.1316232085_dt, -0.7984398007_dt, 0.3357305229_dt,  0.1577706039_dt,  -0.7734549046_dt,
                           0.1990565062_dt,  0.0457027778_dt,  1.1651384830_dt,  2.0153918266_dt,  0.2151824534_dt,  -0.5241936445_dt, -1.8033639193_dt, -1.3083208799_dt, 0.4532545805_dt,
                           1.1421611309_dt,  -3.3311536312_dt, -0.7478722334_dt, 1.1173496246_dt,  0.2981353104_dt,  0.1098855436_dt,  -0.6463385224_dt, 0.4285422862_dt,  1.4760777950_dt };

    const Tensor realOutput = { 0.5070145130_dt, -0.0075466228_dt, 0.0179799497_dt, -0.2341260314_dt };

    const Tensor realInputNabla = { 0.1717543900_dt,  -0.1471260786_dt, -0.0646204948_dt, 0.1564563215_dt, -0.3138123155_dt, 0.1999057829_dt,  -0.0685751140_dt, 0.1695813239_dt, 0.0463390052_dt,
                                    -0.0408147275_dt, 0.0924536288_dt,  0.0164439380_dt,  0.1217427254_dt, -0.1299003363_dt, -0.0243029296_dt, -0.0300091207_dt, 0.0483146608_dt, -0.0013315976_dt,
                                    0.1717543900_dt,  -0.1471260786_dt, -0.0646204948_dt, 0.1564563215_dt, -0.3138123155_dt, 0.1999057829_dt,  -0.0685751140_dt, 0.1695813239_dt, 0.0463390052_dt,
                                    -0.0408147275_dt, 0.0924536288_dt,  0.0164439380_dt,  0.1217427254_dt, -0.1299003363_dt, -0.0243029296_dt, -0.0300091207_dt, 0.0483146608_dt, -0.0013315976_dt };
    const Tensor realWeightsGrad = { 0.8985989094_dt,  0.2447844148_dt,  2.0308308601_dt,  2.2597956657_dt, -0.4477289021_dt, 0.2831146121_dt,  -1.3642222881_dt, -0.1371201277_dt, -1.7923032045_dt,
                                     -0.3042968512_dt, -3.2699983120_dt, -1.3656167984_dt, 0.3192797899_dt, 0.1665121019_dt,  -0.6885542870_dt, -0.3106079996_dt, 0.5863128901_dt,  0.7026228905_dt };

    work.add<raul::DataLayer>("data", DataParams{ { "in" }, 2, 3, 3 });
    Convolution2DLayer cnnLayer("cnn1", Convolution2DParams{ { "in" }, { "cnn1" }, KERNEL_SIZE, FILTERS, STRIDE, PADDING, false, false, DILATION, GROUPS }, networkParameters);
    TENSORS_CREATE(BATCH);

    memory_manager["in"] = TORANGE(input);
    memory_manager["cnn1::Weights"] = TORANGE((Tensor{ 0.1717543900_dt,
                                                       -0.1471260786_dt,
                                                       -0.0646204948_dt,
                                                       0.1564563215_dt,
                                                       -0.3138123155_dt,
                                                       0.1999057829_dt,
                                                       -0.0685751140_dt,
                                                       0.1695813239_dt,
                                                       0.0463390052_dt,
                                                       -0.0408147275_dt,
                                                       0.0924536288_dt,
                                                       0.0164439380_dt,
                                                       0.1217427254_dt,
                                                       -0.1299003363_dt,
                                                       -0.0243029296_dt,
                                                       -0.0300091207_dt,
                                                       0.0483146608_dt,
                                                       -0.0013315976_dt }));

    ASSERT_NO_THROW(cnnLayer.forwardCompute(NetworkMode::Train));

    const auto output = memory_manager["cnn1"];
    EXPECT_EQ(output.size(), realOutput.size());
    for (size_t i = 0; i < output.size(); ++i)
    {
        CHECK_NEAR(output[i], realOutput[i], EPSILON);
    }

    memory_manager[Name("cnn1").grad()] = 1_dt;

    ASSERT_NO_THROW(cnnLayer.backwardCompute());
    const auto inputNabla = memory_manager[Name("in").grad()];
    EXPECT_EQ(inputNabla.size(), realInputNabla.size());
    for (size_t i = 0; i < inputNabla.size(); ++i)
    {
        CHECK_NEAR(inputNabla[i], realInputNabla[i], EPSILON);
    }

    const auto weightsGrad = memory_manager[Name("cnn1::Weights").grad()];
    EXPECT_EQ(weightsGrad.size(), realWeightsGrad.size());
    for (size_t i = 0; i < weightsGrad.size(); ++i)
    {
        CHECK_NEAR(weightsGrad[i], realWeightsGrad[i], EPSILON);
    }
}

TEST(TestCNN2DLayer, DepthwiseUnit)
{
    PROFILE_TEST
    const size_t KERNEL_SIZE_H = 1;
    const size_t KERNEL_SIZE_W = 3;
    const size_t FILTERS = 4;
    const size_t STRIDE_H = 1;
    const size_t STRIDE_W = 3;
    const size_t PADDING_H = 3;
    const size_t PADDING_W = 1;
    const size_t DILATION_H = 2;
    const size_t DILATION_W = 3;
    const size_t BATCH = 2;
    const size_t GROUPS = 4;

    const dtype EPSILON = TODTYPE(1e-6);

    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    const Tensor input = {
        1.9269152880_dt,  1.4872840643_dt,  0.9007171988_dt,  -2.1055209637_dt, 0.6784183979_dt,  -1.2345448732_dt, -0.0430674776_dt, -1.6046669483_dt, -0.7521352768_dt, 1.6487230062_dt,
        -0.3924786448_dt, -1.4036071301_dt, -0.7278813124_dt, -0.5594301820_dt, -0.7688388824_dt, 0.7624453902_dt,  1.6423169374_dt,  -0.1595974714_dt, -0.4973975420_dt, 0.4395892620_dt,
        -0.7581311464_dt, 1.0783176422_dt,  0.8008005619_dt,  1.6806205511_dt,  1.2791243792_dt,  1.2964228392_dt,  0.6104664803_dt,  1.3347377777_dt,  -0.2316243201_dt, 0.0417594910_dt,
        -0.2515752912_dt, 0.8598585129_dt,  -1.3846737146_dt, -0.8712361455_dt, -0.2233659178_dt, 1.7173614502_dt,  0.3188803196_dt,  -0.4245189726_dt, 0.3057209253_dt,  -0.7745925188_dt,
        -1.5575724840_dt, 0.9956361055_dt,  -0.8797858357_dt, -0.6011420488_dt, -1.2741512060_dt, 2.1227850914_dt,  -1.2346531153_dt, -0.4879138768_dt, -0.9138230085_dt, -0.6581372619_dt,
        0.0780238733_dt,  0.5258087516_dt,  -0.4879917204_dt, 1.1913690567_dt,  -0.8140076399_dt, -0.7359927893_dt, -1.4032478333_dt, 0.0360036679_dt,  -0.0634772703_dt, 0.6756148934_dt,
        -0.0978068933_dt, 1.8445940018_dt,  -1.1845374107_dt, 1.3835493326_dt,  1.4451338053_dt,  0.8564125299_dt,  2.2180757523_dt,  0.5231655240_dt,  0.3466466665_dt,  -0.1973314434_dt,
        -1.0545889139_dt, 1.2779955864_dt,  -0.1721901298_dt, 0.5237884521_dt,  0.0566218197_dt,  0.4262961447_dt,  0.5750050545_dt,  -0.6417241096_dt, -2.2063984871_dt, -0.7508030534_dt,
        0.0108681442_dt,  -0.3387423456_dt, -1.3406795263_dt, -0.5853705406_dt, 0.5361881256_dt,  0.5246226192_dt,  1.1412016153_dt,  0.0516435951_dt,  0.7439519763_dt,  -0.4815843999_dt,
        -1.0494660139_dt, 0.6038988233_dt,  -1.7222950459_dt, -0.8277688622_dt, 1.3347029686_dt,  0.4835392833_dt,  -2.5095443726_dt, 0.4880010486_dt,  0.7845868468_dt,  0.0286471862_dt,
        0.6407552958_dt,  0.5832474232_dt,  1.0669267178_dt,  -0.4501533806_dt, -0.1852674782_dt, 0.7527588606_dt,  0.4047577977_dt,  0.1784659922_dt,  0.2649095058_dt,  1.2731683254_dt,
        -0.0013108636_dt, -0.3036037683_dt, -1.4570291042_dt, -0.1023352370_dt, -0.5991530418_dt, 0.4770564139_dt,  0.7261772156_dt,  0.0911518633_dt,  -0.3890652061_dt, 0.5279164910_dt,
        -0.0126854787_dt, 0.2408363223_dt,  0.1325353682_dt,  0.7642406225_dt,  1.0950096846_dt,  0.3398909569_dt,  0.7199674845_dt,  0.4114076197_dt,  1.9311605692_dt,  1.0118638277_dt,
        -1.4364064932_dt, -1.1298598051_dt, -0.1360345334_dt, 1.6354095936_dt,  0.6547407508_dt,  0.5760045648_dt,  1.1415079832_dt,  0.0185645763_dt,  -1.8058050871_dt, 0.9254348874_dt,
        -0.3753443658_dt, 1.0330873728_dt,  -0.6866509318_dt, 0.6368136406_dt,  -0.9726738930_dt, 0.9584577680_dt,  1.6192004681_dt,  1.4506098032_dt,  0.2694815397_dt,  -0.2103759795_dt,
        -0.7328027487_dt, 0.1042978317_dt,  0.3487516940_dt,  0.9675941467_dt,  -0.4656884372_dt, 1.6047972441_dt,  -2.4801201820_dt, -0.4175437391_dt, -1.1954537630_dt, 0.8123369217_dt,
        -1.9005532265_dt, 0.2285765260_dt,  0.0248594042_dt,  -0.3459502459_dt, 0.2868328094_dt,  -0.7308424115_dt, 0.1748202592_dt,  -1.0939292908_dt, -1.6021603346_dt, 1.3528969288_dt,
        1.2888276577_dt,  0.0522954725_dt,  -1.5468504429_dt, 0.7567060590_dt,  0.7755194902_dt,  2.0265355110_dt,  0.0358176120_dt,  0.1205887273_dt,  -0.8056637049_dt, -0.2075768262_dt,
        -0.9319477677_dt, -1.5909662247_dt, -1.1359757185_dt, -0.5225976110_dt, -0.1593310684_dt, -0.4249436855_dt, 0.9442309737_dt,  -0.1849345267_dt, 1.0607991219_dt,  0.2083034515_dt,
        -0.5778480768_dt, 0.3254614472_dt,  0.2617779672_dt,  -0.7599348426_dt, -2.0461385250_dt, -1.5294533968_dt, 0.4048692286_dt,  0.6318764687_dt,  0.3125324845_dt,  -0.0335015208_dt
    };

    const Tensor realOutput = { 0.0000000000_dt,  0.0000000000_dt,  0.0000000000_dt,  0.2789663970_dt,  -0.4969908297_dt, -0.2254363894_dt, -0.0494298674_dt, 0.2480206341_dt,  0.0000000000_dt,
                                0.0000000000_dt,  0.0000000000_dt,  0.0000000000_dt,  0.0000000000_dt,  0.0000000000_dt,  -0.2968042791_dt, 0.3079084754_dt,  0.0943998545_dt,  0.1956370771_dt,
                                0.1084969118_dt,  0.0000000000_dt,  0.0000000000_dt,  0.0000000000_dt,  0.0000000000_dt,  0.0000000000_dt,  0.0000000000_dt,  -0.2233904153_dt, 0.0164815784_dt,
                                -0.5422515869_dt, 0.2394921035_dt,  -0.0788243338_dt, 0.0000000000_dt,  0.0000000000_dt,  0.0000000000_dt,  0.0000000000_dt,  0.0000000000_dt,  0.0000000000_dt,
                                0.1119698137_dt,  0.2339255065_dt,  -0.0090109184_dt, 0.3005108535_dt,  -0.0851477832_dt, 0.0000000000_dt,  0.0000000000_dt,  0.0000000000_dt,  0.0000000000_dt,
                                0.0000000000_dt,  0.0000000000_dt,  0.3304441273_dt,  0.0552737489_dt,  -0.4512650371_dt, 0.0282311775_dt,  0.0410483070_dt,  0.0000000000_dt,  0.0000000000_dt,
                                0.0000000000_dt,  0.0000000000_dt,  0.0000000000_dt,  0.0000000000_dt,  -0.0914842933_dt, 0.0302498620_dt,  -0.0041281860_dt, 0.1526898742_dt,  -0.3225706220_dt,
                                0.0000000000_dt,  0.0000000000_dt,  0.0000000000_dt,  0.0000000000_dt,  0.0000000000_dt,  0.0000000000_dt,  0.1596498042_dt,  -0.1911410838_dt, 0.0113800140_dt,
                                -0.5007734895_dt, -0.7081094384_dt, 0.0000000000_dt,  0.0000000000_dt,  0.0000000000_dt,  0.0000000000_dt,  0.0000000000_dt,  0.0000000000_dt,  -0.0210406575_dt,
                                0.1982082129_dt,  0.0322678909_dt,  -0.0456757508_dt, -0.1102515683_dt, 0.0000000000_dt,  0.0000000000_dt,  0.0000000000_dt };

    const Tensor realInputNabla = {
        0.0000000000_dt, 0.0000000000_dt, 0.3097158670_dt,  0.0000000000_dt, 0.0000000000_dt, 0.0000000000_dt, 0.0000000000_dt, 0.3097158670_dt,  0.0000000000_dt, 0.0000000000_dt,
        0.0000000000_dt, 0.0000000000_dt, 0.3097158670_dt,  0.0000000000_dt, 0.0000000000_dt, 0.0000000000_dt, 0.0000000000_dt, 0.3097158670_dt,  0.0000000000_dt, 0.0000000000_dt,
        0.0000000000_dt, 0.0000000000_dt, 0.3097158670_dt,  0.0000000000_dt, 0.0000000000_dt, 0.0000000000_dt, 0.0000000000_dt, -0.2223689854_dt, 0.0000000000_dt, 0.0000000000_dt,
        0.0000000000_dt, 0.0000000000_dt, -0.2223689854_dt, 0.0000000000_dt, 0.0000000000_dt, 0.0000000000_dt, 0.0000000000_dt, -0.2223689854_dt, 0.0000000000_dt, 0.0000000000_dt,
        0.0000000000_dt, 0.0000000000_dt, -0.2223689854_dt, 0.0000000000_dt, 0.0000000000_dt, 0.0000000000_dt, 0.0000000000_dt, -0.2223689854_dt, 0.0000000000_dt, 0.0000000000_dt,
        0.0000000000_dt, 0.0000000000_dt, 0.4577749968_dt,  0.0000000000_dt, 0.0000000000_dt, 0.0000000000_dt, 0.0000000000_dt, 0.4577749968_dt,  0.0000000000_dt, 0.0000000000_dt,
        0.0000000000_dt, 0.0000000000_dt, 0.4577749968_dt,  0.0000000000_dt, 0.0000000000_dt, 0.0000000000_dt, 0.0000000000_dt, 0.4577749968_dt,  0.0000000000_dt, 0.0000000000_dt,
        0.0000000000_dt, 0.0000000000_dt, 0.4577749968_dt,  0.0000000000_dt, 0.0000000000_dt, 0.0000000000_dt, 0.0000000000_dt, -0.1744827926_dt, 0.0000000000_dt, 0.0000000000_dt,
        0.0000000000_dt, 0.0000000000_dt, -0.1744827926_dt, 0.0000000000_dt, 0.0000000000_dt, 0.0000000000_dt, 0.0000000000_dt, -0.1744827926_dt, 0.0000000000_dt, 0.0000000000_dt,
        0.0000000000_dt, 0.0000000000_dt, -0.1744827926_dt, 0.0000000000_dt, 0.0000000000_dt, 0.0000000000_dt, 0.0000000000_dt, -0.1744827926_dt, 0.0000000000_dt, 0.0000000000_dt,
        0.0000000000_dt, 0.0000000000_dt, 0.3097158670_dt,  0.0000000000_dt, 0.0000000000_dt, 0.0000000000_dt, 0.0000000000_dt, 0.3097158670_dt,  0.0000000000_dt, 0.0000000000_dt,
        0.0000000000_dt, 0.0000000000_dt, 0.3097158670_dt,  0.0000000000_dt, 0.0000000000_dt, 0.0000000000_dt, 0.0000000000_dt, 0.3097158670_dt,  0.0000000000_dt, 0.0000000000_dt,
        0.0000000000_dt, 0.0000000000_dt, 0.3097158670_dt,  0.0000000000_dt, 0.0000000000_dt, 0.0000000000_dt, 0.0000000000_dt, -0.2223689854_dt, 0.0000000000_dt, 0.0000000000_dt,
        0.0000000000_dt, 0.0000000000_dt, -0.2223689854_dt, 0.0000000000_dt, 0.0000000000_dt, 0.0000000000_dt, 0.0000000000_dt, -0.2223689854_dt, 0.0000000000_dt, 0.0000000000_dt,
        0.0000000000_dt, 0.0000000000_dt, -0.2223689854_dt, 0.0000000000_dt, 0.0000000000_dt, 0.0000000000_dt, 0.0000000000_dt, -0.2223689854_dt, 0.0000000000_dt, 0.0000000000_dt,
        0.0000000000_dt, 0.0000000000_dt, 0.4577749968_dt,  0.0000000000_dt, 0.0000000000_dt, 0.0000000000_dt, 0.0000000000_dt, 0.4577749968_dt,  0.0000000000_dt, 0.0000000000_dt,
        0.0000000000_dt, 0.0000000000_dt, 0.4577749968_dt,  0.0000000000_dt, 0.0000000000_dt, 0.0000000000_dt, 0.0000000000_dt, 0.4577749968_dt,  0.0000000000_dt, 0.0000000000_dt,
        0.0000000000_dt, 0.0000000000_dt, 0.4577749968_dt,  0.0000000000_dt, 0.0000000000_dt, 0.0000000000_dt, 0.0000000000_dt, -0.1744827926_dt, 0.0000000000_dt, 0.0000000000_dt,
        0.0000000000_dt, 0.0000000000_dt, -0.1744827926_dt, 0.0000000000_dt, 0.0000000000_dt, 0.0000000000_dt, 0.0000000000_dt, -0.1744827926_dt, 0.0000000000_dt, 0.0000000000_dt,
        0.0000000000_dt, 0.0000000000_dt, -0.1744827926_dt, 0.0000000000_dt, 0.0000000000_dt, 0.0000000000_dt, 0.0000000000_dt, -0.1744827926_dt, 0.0000000000_dt, 0.0000000000_dt
    };
    const Tensor realWeightsGrad = { 0.0000000000_dt, -0.7785770893_dt, 0.0000000000_dt, 0.0000000000_dt, -0.7842580080_dt, 0.0000000000_dt,
                                     0.0000000000_dt, -3.9702625275_dt, 0.0000000000_dt, 0.0000000000_dt, -3.4717206955_dt, 0.0000000000_dt };

    work.add<raul::DataLayer>("data", DataParams{ { "in" }, 4, 5, 5 });
    Convolution2DLayer cnnLayer(
        "cnn1",
        Convolution2DParams{ { "in" }, { "cnn1" }, KERNEL_SIZE_W, KERNEL_SIZE_H, FILTERS, STRIDE_W, STRIDE_H, PADDING_W, PADDING_H, false, false, DILATION_W, DILATION_H, GROUPS },
        networkParameters);
    TENSORS_CREATE(BATCH);

    memory_manager["in"] = TORANGE(input);
    memory_manager["cnn1::Weights"] = TORANGE((Tensor{ -0.0043225288_dt,
                                                       0.3097158670_dt,
                                                       -0.4751853347_dt,
                                                       -0.4248945713_dt,
                                                       -0.2223689854_dt,
                                                       0.1548207402_dt,
                                                       -0.0114391446_dt,
                                                       0.4577749968_dt,
                                                       -0.0512363911_dt,
                                                       0.1527741551_dt,
                                                       -0.1744827926_dt,
                                                       -0.1134870648_dt }));

    ASSERT_NO_THROW(cnnLayer.forwardCompute(NetworkMode::Train));

    const auto output = memory_manager["cnn1"];
    EXPECT_EQ(output.size(), realOutput.size());
    for (size_t i = 0; i < output.size(); ++i)
    {
        CHECK_NEAR(output[i], realOutput[i], EPSILON);
    }

    memory_manager[Name("cnn1").grad()] = 1_dt;

    ASSERT_NO_THROW(cnnLayer.backwardCompute());
    const auto inputNabla = memory_manager[Name("in").grad()];
    EXPECT_EQ(inputNabla.size(), realInputNabla.size());
    for (size_t i = 0; i < inputNabla.size(); ++i)
    {
        CHECK_NEAR(inputNabla[i], realInputNabla[i], EPSILON);
    }

    const auto weightsGrad = memory_manager[Name("cnn1::Weights").grad()];
    EXPECT_EQ(weightsGrad.size(), realWeightsGrad.size());
    for (size_t i = 0; i < weightsGrad.size(); ++i)
    {
        CHECK_NEAR(weightsGrad[i], realWeightsGrad[i], EPSILON);
    }
}

} // UT namespace