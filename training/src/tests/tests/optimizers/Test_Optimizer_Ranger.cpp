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

#include <training/base/optimizers/Ranger.h>

namespace UT
{

TEST(TestOptimizerRanger, StreamUnit)
{
    PROFILE_TEST
    std::ostringstream stream;
    {
        raul::optimizers::Ranger optimizer{ 0.01_dt };
        stream << optimizer;
        ASSERT_STREQ(
            stream.str().c_str(),
            "Ranger(lr=1.000000e-02, alpha=5.000000e-01, k=6, nSMaThreshold=5.000000e+00, beta1=9.500000e-01, beta2=9.990000e-01, epsilon=1.000000e-05, weightDecay=0.000000e+00, useGc=true)");
    }
}

TEST(TestOptimizerRanger, DoubleStepsUnit)
{
    PROFILE_TEST
    raul::MemoryManager memory_manager;
    constexpr auto lr = 0.1_dt;
    constexpr auto alpha = 0.5_dt;
    constexpr auto k = 1U;
    constexpr auto nSmaThreshold = 0.0_dt;
    constexpr auto beta1 = 0.5_dt;
    constexpr auto beta2 = 0.8_dt;
    constexpr auto eps = 1.0_dt;
    constexpr auto weightDecay = 0.3_dt;
    constexpr auto EPS = 1e-6_dt;

    const raul::Tensor param{ 0.49625659_dt, 0.76822180_dt, 0.08847743_dt, 0.13203049_dt, 0.30742282_dt, 0.63407868_dt, 0.49009341_dt, 0.89644474_dt,
                              0.45562798_dt, 0.63230628_dt, 0.34889346_dt, 0.40171731_dt, 0.02232575_dt, 0.16885895_dt, 0.29388845_dt, 0.51852179_dt,
                              0.69766760_dt, 0.80001140_dt, 0.16102946_dt, 0.28226858_dt, 0.68160856_dt, 0.91519397_dt, 0.39709991_dt, 0.87415588_dt };
    const raul::Tensor grad{ 0.41940832_dt, 0.55290705_dt, 0.95273811_dt, 0.03616482_dt, 0.18523103_dt, 0.37341738_dt, 0.30510002_dt, 0.93200040_dt,
                             0.17591017_dt, 0.26983356_dt, 0.15067977_dt, 0.03171951_dt, 0.20812976_dt, 0.92979902_dt, 0.72310919_dt, 0.74233627_dt,
                             0.52629578_dt, 0.24365824_dt, 0.58459234_dt, 0.03315264_dt, 0.13871688_dt, 0.24223500_dt, 0.81546897_dt, 0.79316062_dt };
    const raul::Tensor updatedParamFirstStep{ 0.48905683_dt, 0.75444406_dt, 0.07885379_dt, 0.13665354_dt, 0.30717474_dt, 0.62568694_dt, 0.48509878_dt, 0.87497157_dt,
                                              0.45330477_dt, 0.62578964_dt, 0.34856620_dt, 0.40235800_dt, 0.02598595_dt, 0.15832844_dt, 0.28442001_dt, 0.50539047_dt,
                                              0.68542385_dt, 0.79142129_dt, 0.15580700_dt, 0.28468072_dt, 0.67647505_dt, 0.90489990_dt, 0.38471335_dt, 0.85493547_dt };
    const raul::Tensor updatedParamSecondStep{ 0.48178747_dt, 0.74252260_dt, 0.07554363_dt, 0.13631819_dt, 0.30371904_dt, 0.61660457_dt, 0.47845405_dt, 0.85978478_dt,
                                               0.44769484_dt, 0.61719465_dt, 0.34462768_dt, 0.39805260_dt, 0.02665381_dt, 0.15389833_dt, 0.27882481_dt, 0.49640673_dt,
                                               0.67466360_dt, 0.78045678_dt, 0.15272006_dt, 0.28213549_dt, 0.66766453_dt, 0.89223933_dt, 0.37727112_dt, 0.84051979_dt };
    {
        raul::optimizers::Ranger optimizer{ lr, alpha, k, nSmaThreshold, beta1, beta2, eps, weightDecay };
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

TEST(TestOptimizerRanger, TenStepsUnit)
{
    PROFILE_TEST
    raul::MemoryManager memory_manager;
    constexpr auto lr = 0.1_dt;
    constexpr auto alpha = 0.5_dt;
    constexpr auto k = 6U;
    constexpr auto nSmaThreshold = 5.0_dt;
    constexpr auto beta1 = 0.5_dt;
    constexpr auto beta2 = 0.8_dt;
    constexpr auto eps = 1.0_dt;
    constexpr auto weightDecay = 0.3_dt;
    constexpr auto EPS = 1e-6_dt;
    constexpr auto steps = 10U;

    const raul::Tensor param{ 0.27825248_dt, 0.48195881_dt, 0.81978035_dt, 0.99706656_dt, 0.69844109_dt, 0.56754643_dt, 0.83524317_dt, 0.20559883_dt,
                              0.59317201_dt, 0.11234725_dt, 0.15345693_dt, 0.24170822_dt, 0.72623652_dt, 0.70108020_dt, 0.20382375_dt, 0.65105355_dt,
                              0.77448601_dt, 0.43689132_dt, 0.51909077_dt, 0.61585236_dt, 0.81018829_dt, 0.98009706_dt, 0.11468822_dt, 0.31676513_dt };
    const raul::Tensor grad{ 0.69650495_dt, 0.91427469_dt, 0.93510365_dt, 0.94117838_dt, 0.59950727_dt, 0.06520867_dt, 0.54599625_dt, 0.18719733_dt,
                             0.03402293_dt, 0.94424623_dt, 0.88017988_dt, 0.00123602_dt, 0.59358603_dt, 0.41576999_dt, 0.41771942_dt, 0.27112156_dt,
                             0.69227809_dt, 0.20384824_dt, 0.68329567_dt, 0.75285405_dt, 0.85793579_dt, 0.68695557_dt, 0.00513238_dt, 0.17565155_dt };
    const raul::Tensor updatedParamTenthStep{ 0.15163589_dt, 0.23519443_dt,  0.50189996_dt,  0.64359951_dt, 0.53234357_dt, 0.63823181_dt, 0.66647679_dt, 0.30110210_dt,
                                              0.66979969_dt, -0.07530542_dt, -0.01910310_dt, 0.39583641_dt, 0.55741066_dt, 0.61427820_dt, 0.21001345_dt, 0.63120264_dt,
                                              0.55594164_dt, 0.48264995_dt,  0.35231444_dt,  0.40351400_dt, 0.52182317_dt, 0.72489077_dt, 0.29145336_dt, 0.39550915_dt };
    {
        raul::optimizers::Ranger optimizer{ lr, alpha, k, nSmaThreshold, beta1, beta2, eps, weightDecay };
        auto& params = *memory_manager.createTensor("params", 1, 2, 3, 4, param);
        auto& gradients = *memory_manager.createTensor("gradients", 1, 2, 3, 4, grad);

        // Ten steps
        for (size_t i = 0; i < steps; ++i)
        {
            optimizer(memory_manager, params, gradients);
        }

        // Check
        for (size_t i = 0; i < params.size(); ++i)
        {
            EXPECT_NEAR(updatedParamTenthStep[i], params[i], EPS);
        }
    }
}

#ifdef ANDROID
TEST(TestOptimizerRanger, DoubleStepsFP16Unit)
{
    PROFILE_TEST
    raul::MemoryManagerFP16 memory_manager;
    constexpr auto lr = 0.1_dt;
    constexpr auto alpha = 0.5_dt;
    constexpr auto k = 1U;
    constexpr auto nSmaThreshold = 0.0_dt;
    constexpr auto beta1 = 0.5_dt;
    constexpr auto beta2 = 0.8_dt;
    constexpr auto eps = 1.0_dt;
    constexpr auto weightDecay = 0.3_dt;
    const auto EPS = 1e-2_hf;

    const raul::TensorFP16 param{ 0.49625659_hf, 0.76822180_hf, 0.08847743_hf, 0.13203049_hf, 0.30742282_hf, 0.63407868_hf, 0.49009341_hf, 0.89644474_hf,
                              0.45562798_hf, 0.63230628_hf, 0.34889346_hf, 0.40171731_hf, 0.02232575_hf, 0.16885895_hf, 0.29388845_hf, 0.51852179_hf,
                              0.69766760_hf, 0.80001140_hf, 0.16102946_hf, 0.28226858_hf, 0.68160856_hf, 0.91519397_hf, 0.39709991_hf, 0.87415588_hf };
    const raul::TensorFP16 grad{ 0.41940832_hf, 0.55290705_hf, 0.95273811_hf, 0.03616482_hf, 0.18523103_hf, 0.37341738_hf, 0.30510002_hf, 0.93200040_hf,
                             0.17591017_hf, 0.26983356_hf, 0.15067977_hf, 0.03171951_hf, 0.20812976_hf, 0.92979902_hf, 0.72310919_hf, 0.74233627_hf,
                             0.52629578_hf, 0.24365824_hf, 0.58459234_hf, 0.03315264_hf, 0.13871688_hf, 0.24223500_hf, 0.81546897_hf, 0.79316062_hf };
    const raul::TensorFP16 updatedParamFirstStep{ 0.48905683_hf, 0.75444406_hf, 0.07885379_hf, 0.13665354_hf, 0.30717474_hf, 0.62568694_hf, 0.48509878_hf, 0.87497157_hf,
                                              0.45330477_hf, 0.62578964_hf, 0.34856620_hf, 0.40235800_hf, 0.02598595_hf, 0.15832844_hf, 0.28442001_hf, 0.50539047_hf,
                                              0.68542385_hf, 0.79142129_hf, 0.15580700_hf, 0.28468072_hf, 0.67647505_hf, 0.90489990_hf, 0.38471335_hf, 0.85493547_hf };
    const raul::TensorFP16 updatedParamSecondStep{ 0.48178747_hf, 0.74252260_hf, 0.07554363_hf, 0.13631819_hf, 0.30371904_hf, 0.61660457_hf, 0.47845405_hf, 0.85978478_hf,
                                               0.44769484_hf, 0.61719465_hf, 0.34462768_hf, 0.39805260_hf, 0.02665381_hf, 0.15389833_hf, 0.27882481_hf, 0.49640673_hf,
                                               0.67466360_hf, 0.78045678_hf, 0.15272006_hf, 0.28213549_hf, 0.66766453_hf, 0.89223933_hf, 0.37727112_hf, 0.84051979_hf };
    {
        raul::optimizers::Ranger optimizer{ lr, alpha, k, nSmaThreshold, beta1, beta2, eps, weightDecay };
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