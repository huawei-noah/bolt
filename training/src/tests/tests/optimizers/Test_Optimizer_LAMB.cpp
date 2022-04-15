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

#include <training/base/optimizers/LAMB.h>

namespace UT
{

TEST(TestOptimizerLAMB, StreamUnit)
{
    PROFILE_TEST
    std::ostringstream stream;
    {
        raul::optimizers::LAMB optimizer{ 0.01_dt };
        stream << optimizer;
        ASSERT_STREQ(stream.str().c_str(), "LAMB(lr=1.000000e-02, beta1=9.000000e-01, beta2=9.990000e-01, epsilon=1.000000e-06, weight decay=0.000000e+00, adam: false)");
    }
}

// see LAMB.py
TEST(TestOptimizerLAMB, NoWeightDecayTwoStepsUnit)
{
    PROFILE_TEST
    raul::MemoryManager memory_manager;
    constexpr auto lr = 0.1_dt;
    constexpr auto beta1 = 0.5_dt;
    constexpr auto beta2 = 0.8_dt;
    constexpr auto eps = 1.0_dt;
    constexpr auto EPS = 1e-6_dt;

    const raul::Tensor param{ 0.49625659_dt, 0.76822180_dt, 0.08847743_dt, 0.13203049_dt, 0.30742282_dt, 0.63407868_dt, 0.49009341_dt, 0.89644474_dt,
                              0.45562798_dt, 0.63230628_dt, 0.34889346_dt, 0.40171731_dt, 0.02232575_dt, 0.16885895_dt, 0.29388845_dt, 0.51852179_dt,
                              0.69766760_dt, 0.80001140_dt, 0.16102946_dt, 0.28226858_dt, 0.68160856_dt, 0.91519397_dt, 0.39709991_dt, 0.87415588_dt };
    const raul::Tensor grad{ 0.41940832_dt, 0.55290705_dt, 0.95273811_dt, 0.03616482_dt, 0.18523103_dt, 0.37341738_dt, 0.30510002_dt, 0.93200040_dt,
                             0.17591017_dt, 0.26983356_dt, 0.15067977_dt, 0.03171951_dt, 0.20812976_dt, 0.92979902_dt, 0.72310919_dt, 0.74233627_dt,
                             0.52629578_dt, 0.24365824_dt, 0.58459234_dt, 0.03315264_dt, 0.13871688_dt, 0.24223500_dt, 0.81546897_dt, 0.79316062_dt };
    const raul::Tensor updatedParamFirstStep{ 0.44785520_dt, 0.70746839_dt, -0.00308317_dt, 0.12715299_dt, 0.28397900_dt,  0.59022534_dt, 0.45329982_dt, 0.80629081_dt,
                                              0.43327782_dt, 0.59930772_dt, 0.32954654_dt,  0.39743096_dt, -0.00376947_dt, 0.07885540_dt, 0.21900323_dt, 0.44214168_dt,
                                              0.63928115_dt, 0.76989931_dt, 0.09751603_dt,  0.27779141_dt, 0.66370791_dt,  0.88524061_dt, 0.31520593_dt, 0.79391563_dt };
    const raul::Tensor updatedParamSecondStep{ 0.40191659_dt, 0.65054995_dt, -0.08619092_dt, 0.12230027_dt, 0.26112473_dt,  0.54840213_dt,  0.41794428_dt, 0.72434324_dt,
                                               0.41146380_dt, 0.56746948_dt, 0.31060183_dt,  0.39316359_dt, -0.02913539_dt, -0.00296792_dt, 0.14986515_dt, 0.37173176_dt,
                                               0.58444470_dt, 0.74075562_dt, 0.03818277_dt,  0.27333498_dt, 0.64615172_dt,  0.85624558_dt,  0.24013832_dt, 0.72023946_dt };
    {
        raul::optimizers::LAMB optimizer{ lr, beta1, beta2, eps };
        auto& params = *memory_manager.createTensor("params", 1, 2, 3, 4, param);
        auto& gradients = *memory_manager.createTensor("gradients", 1, 2, 3, 4, grad);

        // First step
        optimizer(memory_manager, params, gradients);

        for (size_t i = 0; i < params.size(); ++i)
        {
            EXPECT_NEAR(updatedParamFirstStep[i], params[i], EPS);
        }

        // Second step
        optimizer(memory_manager, params, gradients);
        for (size_t i = 0; i < params.size(); ++i)
        {
            EXPECT_NEAR(updatedParamSecondStep[i], params[i], EPS);
        }
    }
}

TEST(TestOptimizerLAMB, AdamStyleTwoStepsUnit)
{
    PROFILE_TEST
    raul::MemoryManager memory_manager;
    constexpr auto lr = 0.1_dt;
    constexpr auto beta1 = 0.5_dt;
    constexpr auto beta2 = 0.8_dt;
    constexpr auto eps = 1.0_dt;
    constexpr auto weightDecay = 0.1_dt;
    constexpr auto adam = true;
    constexpr auto EPS = 1e-6_dt;

    const raul::Tensor param{ 0.27825248_dt, 0.48195881_dt, 0.81978035_dt, 0.99706656_dt, 0.69844109_dt, 0.56754643_dt, 0.83524317_dt, 0.20559883_dt,
                              0.59317201_dt, 0.11234725_dt, 0.15345693_dt, 0.24170822_dt, 0.72623652_dt, 0.70108020_dt, 0.20382375_dt, 0.65105355_dt,
                              0.77448601_dt, 0.43689132_dt, 0.51909077_dt, 0.61585236_dt, 0.81018829_dt, 0.98009706_dt, 0.11468822_dt, 0.31676513_dt };
    const raul::Tensor grad{ 0.69650495_dt, 0.91427469_dt, 0.93510365_dt, 0.94117838_dt, 0.59950727_dt, 0.06520867_dt, 0.54599625_dt, 0.18719733_dt,
                             0.03402293_dt, 0.94424623_dt, 0.88017988_dt, 0.00123602_dt, 0.59358603_dt, 0.41576999_dt, 0.41771942_dt, 0.27112156_dt,
                             0.69227809_dt, 0.20384824_dt, 0.68329567_dt, 0.75285405_dt, 0.85793579_dt, 0.68695557_dt, 0.00513238_dt, 0.17565155_dt };
    const raul::Tensor updatedParamFirstStep{ 0.24891593_dt, 0.44469225_dt, 0.77861434_dt, 0.95397699_dt, 0.66781878_dt, 0.55870295_dt, 0.80494869_dt, 0.19490603_dt,
                                              0.58556461_dt, 0.07802896_dt, 0.12034364_dt, 0.23922937_dt, 0.69552076_dt, 0.67654026_dt, 0.18418710_dt, 0.63245285_dt,
                                              0.74031019_dt, 0.42318153_dt, 0.48773158_dt, 0.58153260_dt, 0.77108449_dt, 0.94402057_dt, 0.11328530_dt, 0.30545455_dt };
    const raul::Tensor updatedParamSecondStep{ 0.20958513_dt, 0.39596522_dt, 0.72590190_dt, 0.89932436_dt, 0.62807232_dt, 0.54840940_dt, 0.76605421_dt, 0.18033487_dt,
                                               0.57720828_dt, 0.03204196_dt, 0.07594071_dt, 0.23674443_dt, 0.65573812_dt, 0.64481789_dt, 0.15729472_dt, 0.60863918_dt,
                                               0.69622344_dt, 0.40532726_dt, 0.44650817_dt, 0.53682250_dt, 0.72089487_dt, 0.89809638_dt, 0.11176870_dt, 0.29048216_dt };
    {
        raul::optimizers::LAMB optimizer{ lr, beta1, beta2, eps, weightDecay, adam };
        auto& params = *memory_manager.createTensor("params", 1, 2, 3, 4, param);
        auto& gradients = *memory_manager.createTensor("gradients", 1, 2, 3, 4, grad);

        // First step
        optimizer(memory_manager, params, gradients);

        for (size_t i = 0; i < params.size(); ++i)
        {
            EXPECT_NEAR(updatedParamFirstStep[i], params[i], EPS);
        }

        // Second step
        optimizer(memory_manager, params, gradients);
        for (size_t i = 0; i < params.size(); ++i)
        {
            EXPECT_NEAR(updatedParamSecondStep[i], params[i], EPS);
        }
    }
}

TEST(TestOptimizerLAMB, TwoStepsUnit)
{
    PROFILE_TEST
    raul::MemoryManager memory_manager;
    constexpr auto lr = 0.1_dt;
    constexpr auto beta1 = 0.5_dt;
    constexpr auto beta2 = 0.8_dt;
    constexpr auto eps = 1.0_dt;
    constexpr auto weightDecay = 0.1_dt;
    constexpr auto adam = false;
    constexpr auto EPS = 1e-6_dt;

    const raul::Tensor param{ 0.74965751_dt, 0.60465068_dt, 0.10995799_dt, 0.21209025_dt, 0.97037464_dt, 0.83690894_dt, 0.28198743_dt, 0.37415761_dt,
                              0.02370095_dt, 0.49101293_dt, 0.12347054_dt, 0.11432165_dt, 0.47245020_dt, 0.57507253_dt, 0.29523486_dt, 0.79668880_dt,
                              0.19573045_dt, 0.95368505_dt, 0.84264994_dt, 0.07835853_dt, 0.37555784_dt, 0.52256131_dt, 0.57295054_dt, 0.61858714_dt };
    const raul::Tensor grad{ 0.69621414_dt, 0.52995008_dt, 0.25603563_dt, 0.73659450_dt, 0.02037555_dt, 0.20364666_dt, 0.37483507_dt, 0.25644332_dt,
                             0.32508332_dt, 0.09018916_dt, 0.39364243_dt, 0.60687822_dt, 0.17426711_dt, 0.47434032_dt, 0.85792542_dt, 0.44859987_dt,
                             0.51389611_dt, 0.45686555_dt, 0.60119069_dt, 0.81791973_dt, 0.97362310_dt, 0.81752795_dt, 0.97470677_dt, 0.46383917_dt };
    const raul::Tensor updatedParamFirstStep{ 0.67769986_dt,  0.54659086_dt, 0.08335369_dt, 0.14904755_dt, 0.94772899_dt, 0.79949188_dt, 0.24209835_dt, 0.34193403_dt,
                                              -0.00679680_dt, 0.47147155_dt, 0.08548463_dt, 0.06145668_dt, 0.44537714_dt, 0.52155775_dt, 0.22346349_dt, 0.74035889_dt,
                                              0.14742965_dt,  0.89343238_dt, 0.77476233_dt, 0.01340881_dt, 0.29593199_dt, 0.44824433_dt, 0.48909670_dt, 0.56491083_dt };
    const raul::Tensor updatedParamSecondStep{ 0.61265975_dt,  0.49345976_dt, 0.05728398_dt, 0.08968264_dt,  0.93134212_dt, 0.76726788_dt, 0.20425665_dt, 0.31197339_dt,
                                               -0.03712438_dt, 0.45486891_dt, 0.04858941_dt, 0.01077493_dt,  0.42108607_dt, 0.47247416_dt, 0.15677491_dt, 0.68977797_dt,
                                               0.10129340_dt,  0.83999664_dt, 0.71378452_dt, -0.04816509_dt, 0.22276920_dt, 0.38020468_dt, 0.41300461_dt, 0.51589596_dt };
    {
        raul::optimizers::LAMB optimizer{ lr, beta1, beta2, eps, weightDecay, adam };
        auto& params = *memory_manager.createTensor("params", 1, 2, 3, 4, param);
        auto& gradients = *memory_manager.createTensor("gradients", 1, 2, 3, 4, grad);

        // First step
        optimizer(memory_manager, params, gradients);

        for (size_t i = 0; i < params.size(); ++i)
        {
            EXPECT_NEAR(updatedParamFirstStep[i], params[i], EPS);
        }

        // Second step
        optimizer(memory_manager, params, gradients);
        for (size_t i = 0; i < params.size(); ++i)
        {
            EXPECT_NEAR(updatedParamSecondStep[i], params[i], EPS);
        }
    }
}

#ifdef ANDROID
TEST(TestOptimizerLAMB, TwoStepsFP16Unit)
{
    PROFILE_TEST
    raul::MemoryManagerFP16 memory_manager;
    constexpr auto lr = 0.1_dt;
    constexpr auto beta1 = 0.5_dt;
    constexpr auto beta2 = 0.8_dt;
    constexpr auto eps = 1.0_dt;
    constexpr auto weightDecay = 0.1_dt;
    constexpr auto adam = false;
    const auto EPS = 1e-2_hf;

    const raul::TensorFP16 param{ 0.74965751_hf, 0.60465068_hf, 0.10995799_hf, 0.21209025_hf, 0.97037464_hf, 0.83690894_hf, 0.28198743_hf, 0.37415761_hf,
                              0.02370095_hf, 0.49101293_hf, 0.12347054_hf, 0.11432165_hf, 0.47245020_hf, 0.57507253_hf, 0.29523486_hf, 0.79668880_hf,
                              0.19573045_hf, 0.95368505_hf, 0.84264994_hf, 0.07835853_hf, 0.37555784_hf, 0.52256131_hf, 0.57295054_hf, 0.61858714_hf };
    const raul::TensorFP16 grad{ 0.69621414_hf, 0.52995008_hf, 0.25603563_hf, 0.73659450_hf, 0.02037555_hf, 0.20364666_hf, 0.37483507_hf, 0.25644332_hf,
                             0.32508332_hf, 0.09018916_hf, 0.39364243_hf, 0.60687822_hf, 0.17426711_hf, 0.47434032_hf, 0.85792542_hf, 0.44859987_hf,
                             0.51389611_hf, 0.45686555_hf, 0.60119069_hf, 0.81791973_hf, 0.97362310_hf, 0.81752795_hf, 0.97470677_hf, 0.46383917_hf };
    const raul::TensorFP16 updatedParamFirstStep{ 0.67769986_hf,  0.54659086_hf, 0.08335369_hf, 0.14904755_hf, 0.94772899_hf, 0.79949188_hf, 0.24209835_hf, 0.34193403_hf,
                                              -0.00679680_hf, 0.47147155_hf, 0.08548463_hf, 0.06145668_hf, 0.44537714_hf, 0.52155775_hf, 0.22346349_hf, 0.74035889_hf,
                                              0.14742965_hf,  0.89343238_hf, 0.77476233_hf, 0.01340881_hf, 0.29593199_hf, 0.44824433_hf, 0.48909670_hf, 0.56491083_hf };
    const raul::TensorFP16 updatedParamSecondStep{ 0.61265975_hf,  0.49345976_hf, 0.05728398_hf, 0.08968264_hf,  0.93134212_hf, 0.76726788_hf, 0.20425665_hf, 0.31197339_hf,
                                               -0.03712438_hf, 0.45486891_hf, 0.04858941_hf, 0.01077493_hf,  0.42108607_hf, 0.47247416_hf, 0.15677491_hf, 0.68977797_hf,
                                               0.10129340_hf,  0.83999664_hf, 0.71378452_hf, -0.04816509_hf, 0.22276920_hf, 0.38020468_hf, 0.41300461_hf, 0.51589596_hf };
    {
        raul::optimizers::LAMB optimizer{ lr, beta1, beta2, eps, weightDecay, adam };
        auto& params = *memory_manager.createTensor("params", 1, 2, 3, 4, param);
        auto& gradients = *memory_manager.createTensor("gradients", 1, 2, 3, 4, grad);

        // First step
        optimizer(memory_manager, params, gradients);

        for (size_t i = 0; i < params.size(); ++i)
        {
            ASSERT_TRUE(tools::expect_near_relative(updatedParamFirstStep[i], params[i], EPS));
        }

        // Second step
        optimizer(memory_manager, params, gradients);
        for (size_t i = 0; i < params.size(); ++i)
        {
            ASSERT_TRUE(tools::expect_near_relative(updatedParamSecondStep[i], params[i], EPS));
        }
    }
}
#endif // ANDROID

}