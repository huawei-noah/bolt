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

#include <training/base/layers/basic/trainable/TransposedConvolution2DLayer.h>
#include <training/compiler/Workflow.h>

namespace UT
{

TEST(TestTransposedCNN2DLayer, BiasesUnit)
{
    const size_t KERNEL_SIZE = 3;
    const size_t FILTERS = 2;

    const size_t realOutputSize = 50;

    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    const raul::Tensor input = { 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt };

    work.tensorNeeded("in", "in", raul::WShape{ raul::BS(), 1u, 3u, 3u }, DEC_FORW_READ_NOMEMOPT);
    raul::TransposedConvolution2DLayer cnnLayer("cnn1", raul::TransposedConvolution2DParams{ { "in" }, { "cnn1" }, KERNEL_SIZE, FILTERS }, networkParameters);
    TENSORS_CREATE(1);
    memory_manager["in"] = TORANGE(input);
    memory_manager["cnn1::Weights"] = TORANGE((raul::Tensor{
        1.0_dt,
        1.0_dt,
        1.0_dt,
        1.0_dt,
        1.0_dt,
        1.0_dt,
        1.0_dt,
        1.0_dt,
        1.0_dt,

        1.0_dt,
        1.0_dt,
        1.0_dt,
        1.0_dt,
        1.0_dt,
        1.0_dt,
        1.0_dt,
        1.0_dt,
        1.0_dt,
    }));
    memory_manager["cnn1::Biases"] = TORANGE((raul::Tensor{ 2.0f, 3.0f }));
    ASSERT_NO_THROW(memory_manager["cnn1::Biases"]);
    ASSERT_NO_THROW(memory_manager["cnn1::BiasesGradient"]);

    ASSERT_NO_THROW(cnnLayer.forwardCompute(raul::NetworkMode::Train));

    EXPECT_EQ(memory_manager["cnn1"].getDepth(), static_cast<size_t>(FILTERS));
    EXPECT_EQ(memory_manager["cnn1"].size(), realOutputSize);
    EXPECT_EQ(memory_manager["cnn1"][0], 3.0_dt);
    EXPECT_EQ(memory_manager["cnn1"][1], 4.0_dt);

    ASSERT_NO_THROW(cnnLayer.backwardCompute());
}

TEST(TestTransposedCNN2DLayer, SimpleUnit)
{
    const size_t KERNEL_SIZE = 2;
    const size_t FILTERS = 1;
    const size_t STRIDE = 1;
    const size_t PADDING = 0;
    const size_t OUTPUT_PADDING = 0;
    const size_t DILATION = 1;

    const raul::dtype EPSILON = TODTYPE(1e-6);

    const raul::Tensor input = { -2.17878938_dt, 0.56843126_dt, -1.08452237_dt, -1.39859545_dt };

    const raul::Tensor realOutput = { 0.00815610_dt, -0.58652669_dt, 0.15246566_dt, 0.90068078_dt, 0.28214878_dt, -0.58429915_dt, 0.44630542_dt, 0.97462475_dt, 0.51464051_dt };

    const raul::Tensor realInputNabla = { -0.51501369_dt, -0.51501369_dt, -0.51501369_dt, -0.51501369_dt };

    const raul::Tensor realWeightsGrad = { -4.09347582_dt, -4.09347582_dt, -4.09347582_dt, -4.09347582_dt };

    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    work.tensorNeeded("in", "in", raul::WShape{ raul::BS(), 1u, 2u, 2u }, DEC_FORW_READ_NOMEMOPT);

    raul::TransposedConvolution2DLayer cnnLayer(
        "cnn1", raul::TransposedConvolution2DParams{ { "in" }, { "cnn1" }, KERNEL_SIZE, FILTERS, STRIDE, PADDING, OUTPUT_PADDING, false, DILATION }, networkParameters);
    TENSORS_CREATE(1);
    memory_manager["in"] = TORANGE(input);

    memory_manager["cnn1::Weights"] = TORANGE((raul::Tensor{ -0.00374341_dt, 0.26822180_dt, -0.41152257_dt, -0.36796951_dt }));
    ASSERT_THROW(memory_manager["cnn1::Biases"], raul::Exception);
    ASSERT_THROW(memory_manager["cnn1::BiasesGradient"], raul::Exception);

    // Forward
    ASSERT_NO_THROW(cnnLayer.forwardCompute(raul::NetworkMode::Train));

    const auto& output = memory_manager["cnn1"];
    EXPECT_EQ(output.size(), realOutput.size());
    for (size_t i = 0; i < output.size(); ++i)
    {
        CHECK_NEAR(output[i], realOutput[i], EPSILON);
    }

    // Backward
    memory_manager[raul::Name("cnn1").grad()] = TORANGE(*memory_manager.createTensor("gradient", output.getShape(), 1.0_dt));

    ASSERT_NO_THROW(cnnLayer.backwardCompute());
    const auto inputNabla = memory_manager[raul::Name("in").grad()];
    EXPECT_EQ(inputNabla.size(), realInputNabla.size());
    for (size_t i = 0; i < inputNabla.size(); ++i)
    {
        CHECK_NEAR(inputNabla[i], realInputNabla[i], EPSILON);
    }
    const auto weightsGrad = memory_manager[raul::Name("cnn1::Weights").grad()];
    EXPECT_EQ(weightsGrad.size(), realWeightsGrad.size());
    for (size_t i = 0; i < weightsGrad.size(); ++i)
    {
        CHECK_NEAR(weightsGrad[i], realWeightsGrad[i], EPSILON);
    }
}

TEST(TestTransposedCNN2DLayer, NonTrivialUnit)
{
    const size_t KERNEL_SIZE_W = 1;
    const size_t KERNEL_SIZE_H = 3;
    const size_t FILTERS = 2;
    const size_t STRIDE_W = 3;
    const size_t STRIDE_H = 2;
    const size_t PADDING_W = 1;
    const size_t PADDING_H = 2;
    const size_t DILATION_W = 3;
    const size_t DILATION_H = 2;

    const raul::dtype EPSILON = TODTYPE(1e-5);

    const raul::Tensor input = { -1.12583983_dt, -1.15236020_dt, -0.25057858_dt, -0.43387881_dt, 0.84871036_dt,  0.69200915_dt,  -0.31601277_dt, -2.11521935_dt, 0.32227492_dt,
                                 -1.26333475_dt, 0.34998319_dt,  0.30813393_dt,  0.11984151_dt,  1.23765790_dt,  1.11677718_dt,  -0.24727815_dt, -1.35265374_dt, -1.69593120_dt,
                                 0.56665063_dt,  0.79350835_dt,  0.59883946_dt,  -1.55509508_dt, -0.34136039_dt, 1.85300612_dt,  0.75018948_dt,  -0.58549756_dt, -0.17339675_dt,
                                 0.18347794_dt,  1.38936615_dt,  1.58633423_dt,  0.94629836_dt,  -0.84367675_dt, -0.61358309_dt, 0.03159274_dt,  -0.49267697_dt, 0.24841475_dt,
                                 0.43969584_dt,  0.11241119_dt,  0.54329473_dt,  -0.39515755_dt, 0.20552567_dt,  -0.45032975_dt, -0.57307708_dt, -0.55535841_dt, 0.59432304_dt,
                                 1.54194260_dt,  1.81972528_dt,  -0.55152869_dt, -1.32532597_dt, 0.18855357_dt,  -0.06907269_dt, -0.49492535_dt, -1.49591494_dt, -0.19383712_dt };

    const raul::Tensor deltas = { -0.47311980_dt, 0.33555076_dt,  1.50912189_dt,  2.08195543_dt,  1.70671165_dt,  2.38036752_dt,  -1.12560165_dt, -0.31699809_dt, -0.14067143_dt, 0.80575359_dt,
                                  0.32761431_dt,  -0.76070720_dt, -1.59908199_dt, 0.01848667_dt,  -0.75042683_dt, 0.18540798_dt,  1.03946197_dt,  0.35815310_dt,  -0.00330387_dt, -0.53444070_dt,
                                  1.16868782_dt,  0.39450276_dt,  1.94146204_dt,  0.79149806_dt,  0.03353186_dt,  0.71008658_dt,  -1.53528714_dt, -0.41267914_dt, 0.96630329_dt,  1.62478316_dt,
                                  -0.36561880_dt, -1.30244040_dt, -0.22824179_dt, 0.27995500_dt,  0.07324605_dt,  1.11331844_dt,  0.28226724_dt,  0.43422565_dt,  -0.80249292_dt, -1.29518616_dt,
                                  0.78130794_dt,  -0.92678940_dt, 0.20641631_dt,  -0.33344787_dt, -0.42883000_dt, 0.23291829_dt,  0.79688716_dt,  -0.18484163_dt, -0.92146200_dt, -0.05619479_dt,
                                  -0.70152360_dt, 1.03668678_dt,  -0.60367012_dt, -1.27876520_dt, 0.09295023_dt,  -0.66609973_dt, 0.92337352_dt,  1.38729525_dt,  1.37503791_dt,  0.65963107_dt,
                                  0.47655711_dt,  -1.01630747_dt, 0.18036698_dt,  0.10833187_dt,  1.95065713_dt,  -1.06309855_dt, 1.14035070_dt,  -0.08988207_dt, 0.72979623_dt,  -1.84531903_dt,
                                  -0.02501994_dt, 1.36938095_dt,  -0.31263882_dt, 0.24578547_dt,  0.37718192_dt,  1.10123479_dt,  -1.14277780_dt, 0.03758540_dt,  2.69627643_dt,  1.23576367_dt,
                                  -0.20106392_dt, -0.11792699_dt, -0.82936686_dt, -1.40725660_dt, 1.91589761_dt,  0.69019532_dt,  -2.32170153_dt, -1.19641018_dt, 0.19702816_dt,  -1.17733157_dt,
                                  -0.06614405_dt, -0.35835508_dt, -1.39517200_dt, 0.47511867_dt,  -0.81372583_dt, 0.92423636_dt,  -0.24734129_dt, -1.41538787_dt, -1.07865798_dt, -0.72090983_dt };

    const raul::Tensor realOutput = { -0.27676830_dt, -0.27676830_dt, -0.50986284_dt, -0.27676830_dt, -0.27676830_dt, -0.27676830_dt, -0.27676830_dt, -0.27676830_dt, -0.27676830_dt, -0.27676830_dt,
                                      -0.27676830_dt, -0.27676830_dt, 0.88876522_dt,  -0.27676830_dt, -0.27676830_dt, -0.27676830_dt, -0.27676830_dt, -0.27676830_dt, -0.27676830_dt, -0.27676830_dt,
                                      -0.27676830_dt, -0.27676830_dt, -1.29211509_dt, -0.27676830_dt, -0.27676830_dt, -0.17777696_dt, -0.17777696_dt, -0.03814605_dt, -0.17777696_dt, -0.17777696_dt,
                                      -0.17777696_dt, -0.17777696_dt, -0.17777696_dt, -0.17777696_dt, -0.17777696_dt, -0.17777696_dt, -0.17777696_dt, 0.00164664_dt,  -0.17777696_dt, -0.17777696_dt,
                                      -0.17777696_dt, -0.17777696_dt, -0.17777696_dt, -0.17777696_dt, -0.17777696_dt, -0.17777696_dt, -0.17777696_dt, 0.13718286_dt,  -0.17777696_dt, -0.17777696_dt,
                                      -0.27676830_dt, -0.27676830_dt, -0.50073957_dt, -0.27676830_dt, -0.27676830_dt, -0.27676830_dt, -0.27676830_dt, -0.27676830_dt, -0.27676830_dt, -0.27676830_dt,
                                      -0.27676830_dt, -0.27676830_dt, -0.63370347_dt, -0.27676830_dt, -0.27676830_dt, -0.27676830_dt, -0.27676830_dt, -0.27676830_dt, -0.27676830_dt, -0.27676830_dt,
                                      -0.27676830_dt, -0.27676830_dt, 0.08432946_dt,  -0.27676830_dt, -0.27676830_dt, -0.17777696_dt, -0.17777696_dt, 0.16211960_dt,  -0.17777696_dt, -0.17777696_dt,
                                      -0.17777696_dt, -0.17777696_dt, -0.17777696_dt, -0.17777696_dt, -0.17777696_dt, -0.17777696_dt, -0.17777696_dt, 0.61419845_dt,  -0.17777696_dt, -0.17777696_dt,
                                      -0.17777696_dt, -0.17777696_dt, -0.17777696_dt, -0.17777696_dt, -0.17777696_dt, -0.17777696_dt, -0.17777696_dt, -0.33588973_dt, -0.17777696_dt, -0.17777696_dt };

    const raul::Tensor realInputNabla = { 0._dt, 0.98022926_dt,  0._dt, 0._dt, -0.97168428_dt, 0._dt, 0._dt, 0.32867494_dt,  0._dt, 0._dt, 0.56250048_dt, 0._dt, 0._dt, -0.68348289_dt, 0._dt,
                                          0._dt, 0.71109134_dt,  0._dt, 0._dt, -0.09915833_dt, 0._dt, 0._dt, -0.46439925_dt, 0._dt, 0._dt, 0.07548344_dt, 0._dt, 0._dt, -0.32969624_dt, 0._dt,
                                          0._dt, 0.16827486_dt,  0._dt, 0._dt, 0.51299018_dt,  0._dt, 0._dt, -0.11056838_dt, 0._dt, 0._dt, 0.33984596_dt, 0._dt, 0._dt, -0.05727647_dt, 0._dt,
                                          0._dt, -0.15414071_dt, 0._dt, 0._dt, -0.29994714_dt, 0._dt, 0._dt, -0.23234649_dt, 0._dt };

    const raul::Tensor realWeightsGrad = { 5.08365631_dt,  -8.03968811_dt, 4.00481939_dt,  -0.71099371_dt, 2.99399972_dt, -1.12538338_dt, 3.80654287_dt, -3.93423772_dt, 1.79923403_dt,
                                           -0.42595136_dt, 1.18739843_dt,  -0.50218743_dt, 0.03746638_dt,  0.00981903_dt, -1.66235399_dt, 1.68344891_dt, 1.59264016_dt,  -2.03635502_dt };

    const raul::Tensor realBiasGrad = { 13.82497597_dt, -6.50798988_dt };

    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    work.tensorNeeded("in", "in", raul::WShape{ raul::BS(), 3u, 3u, 3u }, DEC_FORW_READ_NOMEMOPT);

    raul::TransposedConvolution2DLayer cnnLayer(
        "cnn1",
        raul::TransposedConvolution2DParams{ { "in" }, { "cnn1" }, KERNEL_SIZE_W, KERNEL_SIZE_H, FILTERS, STRIDE_W, STRIDE_H, PADDING_W, PADDING_H, 0, 0, true, DILATION_W, DILATION_H },
        networkParameters);
    TENSORS_CREATE(2);
    memory_manager["in"] = TORANGE(input);

    memory_manager["cnn1::Weights"] = TORANGE((raul::Tensor{ -0.00305650_dt,
                                                             0.21900219_dt,
                                                             -0.33600679_dt,
                                                             -0.30044585_dt,
                                                             -0.15723860_dt,
                                                             0.10947478_dt,
                                                             -0.00808871_dt,
                                                             0.32369578_dt,
                                                             -0.03622961_dt,
                                                             0.10802764_dt,
                                                             -0.12337798_dt,
                                                             -0.08024749_dt,
                                                             -0.39001942_dt,
                                                             -0.27037555_dt,
                                                             -0.16828938_dt,
                                                             0.01512298_dt,
                                                             0.16139495_dt,
                                                             0.24495828_dt }));
    memory_manager["cnn1::Biases"] = TORANGE((raul::Tensor{ -0.27676830_dt, -0.17777696_dt }));

    ASSERT_NO_THROW(cnnLayer.forwardCompute(raul::NetworkMode::Train));

    const auto output = memory_manager["cnn1"];
    EXPECT_EQ(output.size(), realOutput.size());
    for (size_t i = 0; i < output.size(); ++i)
    {
        CHECK_NEAR(output[i], realOutput[i], EPSILON);
    }

    memory_manager[raul::Name("cnn1").grad()] = TORANGE(deltas);

    ASSERT_NO_THROW(cnnLayer.backwardCompute());
    const auto inputNabla = memory_manager[raul::Name("in").grad()];
    EXPECT_EQ(inputNabla.size(), realInputNabla.size());
    for (size_t i = 0; i < inputNabla.size(); ++i)
    {
        CHECK_NEAR(inputNabla[i], realInputNabla[i], EPSILON);
    }

    const auto weightsGrad = memory_manager[raul::Name("cnn1::Weights").grad()];
    EXPECT_EQ(weightsGrad.size(), realWeightsGrad.size());
    for (size_t i = 0; i < weightsGrad.size(); ++i)
    {
        CHECK_NEAR(weightsGrad[i], realWeightsGrad[i], EPSILON);
    }

    const auto biasGrad = memory_manager[raul::Name("cnn1::Biases").grad()];
    EXPECT_EQ(biasGrad.size(), realBiasGrad.size());
    for (size_t i = 0; i < biasGrad.size(); ++i)
    {
        CHECK_NEAR(biasGrad[i], realBiasGrad[i], EPSILON);
    }
}

}