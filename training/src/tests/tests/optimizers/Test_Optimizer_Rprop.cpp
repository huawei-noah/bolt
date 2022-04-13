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
#include <iostream>

#include <training/base/optimizers/Rprop.h>

namespace UT
{

TEST(TestOptimizerRprop, StreamUnit)
{
    PROFILE_TEST
    std::ostringstream stream;
    {
        raul::optimizers::Rprop optimizer{ 0.01_dt };
        stream << optimizer;
        ASSERT_STREQ(stream.str().c_str(), "Rprop(lr=1.000000e-02, alpha=5.000000e-01, beta=1.200000e+00, min step=1.000000e-06, max step=5.000000e+01)");
    }
}

// see Rprop.py
TEST(TestOptimizerRprop, StepUnit)
{
    PROFILE_TEST
    raul::MemoryManager memory_manager;
    constexpr auto lr = 0.1_dt;

    const raul::Tensor param{ 0.4963_dt, 0.7682_dt, 0.0885_dt, 0.1320_dt, 0.3074_dt, 0.6341_dt, 0.4901_dt, 0.8964_dt, 0.4556_dt, 0.6323_dt, 0.3489_dt, 0.4017_dt,
                              0.0223_dt, 0.1689_dt, 0.2939_dt, 0.5185_dt, 0.6977_dt, 0.8000_dt, 0.1610_dt, 0.2823_dt, 0.6816_dt, 0.9152_dt, 0.3971_dt, 0.8742_dt };
    const raul::Tensor grad{ 0.4681_dt,  -0.1577_dt, 1.4437_dt, 0.2660_dt,  0.1665_dt,  0.8744_dt, -0.1435_dt, -0.1116_dt, -0.6731_dt, 0.8728_dt,  1.0554_dt, 0.1778_dt,
                             -0.2303_dt, -0.3918_dt, 0.5433_dt, -0.3952_dt, -0.4462_dt, 0.7440_dt, 1.5210_dt,  3.4105_dt,  -1.5312_dt, -1.2341_dt, 1.8197_dt, -0.5515_dt };
    const raul::Tensor updatedParam{ 0.3963_dt, 0.8682_dt, -0.0115_dt, 0.0320_dt, 0.2074_dt, 0.5341_dt, 0.5901_dt, 0.9964_dt, 0.5556_dt, 0.5323_dt, 0.2489_dt, 0.3017_dt,
                                     0.1223_dt, 0.2689_dt, 0.1939_dt,  0.6185_dt, 0.7977_dt, 0.7000_dt, 0.0610_dt, 0.1823_dt, 0.7816_dt, 1.0152_dt, 0.2971_dt, 0.9742_dt };
    {
        raul::optimizers::Rprop optimizer{ lr };
        auto& params = *memory_manager.createTensor("params", 1, 2, 3, 4, param);
        auto& gradients = *memory_manager.createTensor("gradients", 1, 2, 3, 4, grad);

        optimizer(memory_manager, params, gradients);

        for (size_t i = 0; i < updatedParam.size(); ++i)
        {
            EXPECT_FLOAT_EQ(updatedParam[i], params[i]);
        }
    }
}

TEST(TestOptimizerRprop, DoubleStepUnit)
{
    PROFILE_TEST
    raul::MemoryManager memory_manager;
    constexpr auto lr = 0.1_dt;
    constexpr auto alpha = 0.5_dt;
    constexpr auto beta = 1.5_dt;

    const raul::Tensor param{ 0.4963_dt, 0.7682_dt, 0.0885_dt, 0.1320_dt, 0.3074_dt, 0.6341_dt, 0.4901_dt, 0.8964_dt, 0.4556_dt, 0.6323_dt, 0.3489_dt, 0.4017_dt,
                              0.0223_dt, 0.1689_dt, 0.2939_dt, 0.5185_dt, 0.6977_dt, 0.8000_dt, 0.1610_dt, 0.2823_dt, 0.6816_dt, 0.9152_dt, 0.3971_dt, 0.8742_dt };
    const raul::Tensor grad{ 0.4681_dt,  -0.1577_dt, 1.4437_dt, 0.2660_dt,  0.1665_dt,  0.8744_dt, -0.1435_dt, -0.1116_dt, -0.6731_dt, 0.8728_dt,  1.0554_dt, 0.1778_dt,
                             -0.2303_dt, -0.3918_dt, 0.5433_dt, -0.3952_dt, -0.4462_dt, 0.7440_dt, 1.5210_dt,  3.4105_dt,  -1.5312_dt, -1.2341_dt, 1.8197_dt, -0.5515_dt };
    const raul::Tensor updatedParam{ 0.2463_dt, 1.0182_dt, -0.1615_dt, -0.1180_dt, 0.0574_dt, 0.3841_dt, 0.7401_dt,  1.1464_dt, 0.7056_dt, 0.3823_dt, 0.0989_dt, 0.1517_dt,
                                     0.2723_dt, 0.4189_dt, 0.0439_dt,  0.7685_dt,  0.9477_dt, 0.5500_dt, -0.0890_dt, 0.0323_dt, 0.9316_dt, 1.1652_dt, 0.1471_dt, 1.1242_dt };
    {
        raul::optimizers::Rprop optimizer{ lr, alpha, beta };
        auto& params = *memory_manager.createTensor("params", 1, 2, 3, 4, param);
        auto& gradients = *memory_manager.createTensor("gradients", 1, 2, 3, 4, grad);

        optimizer(memory_manager, params, gradients);
        optimizer(memory_manager, params, gradients);

        for (size_t i = 0; i < updatedParam.size(); ++i)
        {
            EXPECT_FLOAT_EQ(updatedParam[i], params[i]);
        }
    }
}

TEST(TestOptimizerRprop, NStepsWithDifferentSignsGradientsUnit)
{
    PROFILE_TEST
    raul::MemoryManager memory_manager;
    constexpr auto lr = 0.1_dt;
    constexpr auto alpha = 0.1_dt;
    constexpr auto beta = 20.0_dt;
    constexpr auto minStep = 0.001_dt;
    constexpr auto maxStep = 1.2_dt;

    const raul::Tensor param{ 0.4963_dt, 0.7682_dt, 0.0885_dt, 0.1320_dt, 0.3074_dt, 0.6341_dt };
    const raul::Tensor grads[]{ { 0.4901_dt, 0.8964_dt, 0.4556_dt, 0.6323_dt, 0.3489_dt, 0.4017_dt },
                                { 0.0223_dt, 0.1689_dt, 0.2939_dt, 0.5185_dt, 0.6977_dt, 0.8000_dt },
                                { 0.1610_dt, 0.2823_dt, 0.6816_dt, 0.9152_dt, 0.3971_dt, 0.8742_dt },
                                { -0.1612_dt, 0.1058_dt, 0.9055_dt, -0.9277_dt, -0.6295_dt, -0.2532_dt },
                                { -0.3898_dt, 0.8640_dt, -0.6482_dt, -0.4603_dt, -0.6986_dt, -0.9366_dt },
                                { 0.2081_dt, 0.9298_dt, 0.7231_dt, 0.7423_dt, 0.5263_dt, 0.2437_dt },
                                { 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt },
                                { 0.1692_dt, -0.9337_dt, -0.7226_dt, -0.5155_dt, 0.6309_dt, 0.5863_dt } };
    const raul::Tensor updatedParam{ -1.8957_dt, -4.1318_dt, -3.6115_dt, -2.2360_dt, -2.0846_dt, -1.7579_dt };

    raul::optimizers::Rprop optimizer{ lr, alpha, beta, minStep, maxStep };
    auto& params = *memory_manager.createTensor("params", 1, 2, 3, 1, param);
    auto& gradients = *memory_manager.createTensor("gradients", 1, 2, 3, 1);
    for (size_t i = 0; i < std::size(grads); ++i)
    {
        gradients = TORANGE(grads[i]);
        optimizer(memory_manager, params, gradients);
    }
    for (size_t q = 0; q < params.size(); ++q)
    {
        EXPECT_FLOAT_EQ(updatedParam[q], params[q]);
    }
}

#ifdef ANDROID
TEST(TestOptimizerRprop, NStepsWithDifferentSignsGradientsFP16Unit)
{
    PROFILE_TEST
    raul::MemoryManagerFP16 memory_manager;
    constexpr auto lr = 0.1_dt;
    constexpr auto alpha = 0.1_dt;
    constexpr auto beta = 20.0_dt;
    constexpr auto minStep = 0.001_dt;
    constexpr auto maxStep = 1.2_dt;
    const auto eps = 1e-3_hf;

    const raul::TensorFP16 param{ 0.4963_hf, 0.7682_hf, 0.0885_hf, 0.1320_hf, 0.3074_hf, 0.6341_hf };
    const raul::TensorFP16 grads[]{ { 0.4901_hf, 0.8964_hf, 0.4556_hf, 0.6323_hf, 0.3489_hf, 0.4017_hf },
                                { 0.0223_hf, 0.1689_hf, 0.2939_hf, 0.5185_hf, 0.6977_hf, 0.8000_hf },
                                { 0.1610_hf, 0.2823_hf, 0.6816_hf, 0.9152_hf, 0.3971_hf, 0.8742_hf },
                                { -0.1612_hf, 0.1058_hf, 0.9055_hf, -0.9277_hf, -0.6295_hf, -0.2532_hf },
                                { -0.3898_hf, 0.8640_hf, -0.6482_hf, -0.4603_hf, -0.6986_hf, -0.9366_hf },
                                { 0.2081_hf, 0.9298_hf, 0.7231_hf, 0.7423_hf, 0.5263_hf, 0.2437_hf },
                                { 0.0_hf, 0.0_hf, 0.0_hf, 0.0_hf, 0.0_hf, 0.0_hf },
                                { 0.1692_hf, -0.9337_hf, -0.7226_hf, -0.5155_hf, 0.6309_hf, 0.5863_hf } };
    const raul::TensorFP16 updatedParam{ -1.8957_hf, -4.1318_hf, -3.6115_hf, -2.2360_hf, -2.0846_hf, -1.7579_hf };

    raul::optimizers::Rprop optimizer{ lr, alpha, beta, minStep, maxStep };
    auto& params = *memory_manager.createTensor("params", 1, 2, 3, 1, param);
    auto& gradients = *memory_manager.createTensor("gradients", 1, 2, 3, 1);
    for (size_t i = 0; i < std::size(grads); ++i)
    {
        gradients = TORANGE_FP16(grads[i]);
        optimizer(memory_manager, params, gradients);
    }
    for (size_t q = 0; q < params.size(); ++q)
    {
        ASSERT_TRUE(tools::expect_near_relative(updatedParam[q], params[q], eps));
    }
}
#endif // ANDROID

}