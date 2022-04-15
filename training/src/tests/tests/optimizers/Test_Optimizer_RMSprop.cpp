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

#include <training/base/optimizers/RMSprop.h>

namespace UT
{

TEST(TestOptimizerRMSprop, StreamUnit)
{
    PROFILE_TEST
    std::ostringstream stream;
    {
        raul::optimizers::RMSprop optimizer{ 0.01_dt };
        stream << optimizer;
        ASSERT_STREQ(stream.str().c_str(), "RMSprop(lr=1.000000e-02, alpha=9.900000e-01, eps=1.000000e-08, weight decay=0.000000e+00, momentum: 0.000000e+00, centered: false, style: pytorch)");
    }
}

// see RMSprop.py
TEST(TestOptimizerRMSprop, TwoStepsUnit)
{
    PROFILE_TEST
    raul::MemoryManager memory_manager;
    constexpr auto lr = 0.1_dt;
    constexpr auto alpha = 0.9_dt;
    constexpr auto eps = 0.1_dt;
    constexpr auto decay = 0.1_dt;
    constexpr auto momentum = 0.1_dt;
    constexpr auto centered = true;
    constexpr auto EPS = 1.0e-5_dt;

    const raul::Tensor param{ 0.496257_dt, 0.768222_dt, 0.088477_dt, 0.132030_dt, 0.307423_dt, 0.634079_dt, 0.490093_dt, 0.896445_dt, 0.455628_dt, 0.632306_dt, 0.348893_dt, 0.401717_dt,
                              0.022326_dt, 0.168859_dt, 0.293888_dt, 0.518522_dt, 0.697668_dt, 0.800011_dt, 0.161029_dt, 0.282269_dt, 0.681609_dt, 0.915194_dt, 0.397100_dt, 0.874156_dt };
    const raul::Tensor grad{ 0.419408_dt, 0.552907_dt, 0.952738_dt, 0.036165_dt, 0.185231_dt, 0.373417_dt, 0.305100_dt, 0.932000_dt, 0.175910_dt, 0.269834_dt, 0.150680_dt, 0.031720_dt,
                             0.208130_dt, 0.929799_dt, 0.723109_dt, 0.742336_dt, 0.526296_dt, 0.243658_dt, 0.584592_dt, 0.033153_dt, 0.138717_dt, 0.242235_dt, 0.815469_dt, 0.793161_dt };
    const raul::Tensor paramAfterFirstStep{ 0.301402_dt, 0.550261_dt, -0.159050_dt, 0.089031_dt, 0.176365_dt,  0.445016_dt,  0.318390_dt, 0.645114_dt,
                                            0.322565_dt, 0.465707_dt, 0.229687_dt,  0.342580_dt, -0.106645_dt, -0.077670_dt, 0.062883_dt, 0.283733_dt,
                                            0.483886_dt, 0.635799_dt, -0.053345_dt, 0.230434_dt, 0.553956_dt,  0.748422_dt,  0.157254_dt, 0.632354_dt };
    const raul::Tensor paramAfterSecondStep{ 0.121381_dt, 0.351013_dt, -0.382218_dt, 0.046700_dt, 0.051836_dt,  0.269886_dt,  0.158099_dt,  0.418925_dt,
                                             0.196239_dt, 0.309829_dt, 0.115852_dt,  0.284721_dt, -0.229299_dt, -0.300042_dt, -0.147013_dt, 0.070777_dt,
                                             0.288081_dt, 0.481993_dt, -0.249640_dt, 0.179576_dt, 0.432488_dt,  0.592395_dt,  -0.059773_dt, 0.413759_dt };
    {
        raul::optimizers::RMSprop optimizer{ lr, alpha, eps, decay, momentum, centered };
        auto& params = *memory_manager.createTensor("params", 1, 2, 3, 4, param);
        auto& gradients = *memory_manager.createTensor("gradients", 1, 2, 3, 4, grad);

        // First step
        optimizer(memory_manager, params, gradients);

        for (size_t i = 0; i < params.size(); ++i)
        {
            EXPECT_NEAR(paramAfterFirstStep[i], params[i], EPS);
        }

        // Second step
        optimizer(memory_manager, params, gradients);

        for (size_t i = 0; i < params.size(); ++i)
        {
            EXPECT_NEAR(paramAfterSecondStep[i], params[i], EPS);
        }
    }
}

TEST(TestOptimizerRMSprop, TwoStepsNotCenteredUnit)
{
    PROFILE_TEST
    raul::MemoryManager memory_manager;
    constexpr auto lr = 0.1_dt;
    constexpr auto alpha = 0.9_dt;
    constexpr auto eps = 0.1_dt;
    constexpr auto decay = 0.1_dt;
    constexpr auto momentum = 0.1_dt;
    constexpr auto centered = false;
    constexpr auto EPS = 1.0e-5_dt;

    const raul::Tensor param{ 0.278252_dt, 0.481959_dt, 0.819780_dt, 0.997067_dt, 0.698441_dt, 0.567546_dt, 0.835243_dt, 0.205599_dt, 0.593172_dt, 0.112347_dt, 0.153457_dt, 0.241708_dt,
                              0.726237_dt, 0.701080_dt, 0.203824_dt, 0.651054_dt, 0.774486_dt, 0.436891_dt, 0.519091_dt, 0.615852_dt, 0.810188_dt, 0.980097_dt, 0.114688_dt, 0.316765_dt };
    const raul::Tensor grad{ 0.696505_dt, 0.914275_dt, 0.935104_dt, 0.941178_dt, 0.599507_dt, 0.065209_dt, 0.545996_dt, 0.187197_dt, 0.034023_dt, 0.944246_dt, 0.880180_dt, 0.001236_dt,
                             0.593586_dt, 0.415770_dt, 0.417719_dt, 0.271122_dt, 0.692278_dt, 0.203848_dt, 0.683296_dt, 0.752854_dt, 0.857936_dt, 0.686956_dt, 0.005132_dt, 0.175652_dt };
    const raul::Tensor paramAfterFirstStep{ 0.058127_dt, 0.243936_dt,  0.578554_dt,  0.754525_dt, 0.483677_dt, 0.479530_dt, 0.624752_dt, 0.080216_dt,
                                            0.521104_dt, -0.125246_dt, -0.080246_dt, 0.218191_dt, 0.511796_dt, 0.509524_dt, 0.020164_dt, 0.488093_dt,
                                            0.550343_dt, 0.298042_dt,  0.297971_dt,  0.388068_dt, 0.573630_dt, 0.754680_dt, 0.098915_dt, 0.191539_dt };
    const raul::Tensor paramAfterSecondStep{ -0.134884_dt, 0.037550_dt,  0.369809_dt,  0.544814_dt, 0.294736_dt, 0.395875_dt, 0.439078_dt,  -0.036314_dt,
                                             0.451959_dt,  -0.331315_dt, -0.283434_dt, 0.194984_dt, 0.323103_dt, 0.338558_dt, -0.144557_dt, 0.340064_dt,
                                             0.354301_dt,  0.170046_dt,  0.104209_dt,  0.189293_dt, 0.368327_dt, 0.557680_dt, 0.083281_dt,  0.075143_dt };
    {
        raul::optimizers::RMSprop optimizer{ lr, alpha, eps, decay, momentum, centered };
        auto& params = *memory_manager.createTensor("params", 1, 2, 3, 4, param);
        auto& gradients = *memory_manager.createTensor("gradients", 1, 2, 3, 4, grad);

        // First step
        optimizer(memory_manager, params, gradients);

        for (size_t i = 0; i < params.size(); ++i)
        {
            EXPECT_NEAR(paramAfterFirstStep[i], params[i], EPS);
        }

        // Second step
        optimizer(memory_manager, params, gradients);

        for (size_t i = 0; i < params.size(); ++i)
        {
            EXPECT_NEAR(paramAfterSecondStep[i], params[i], EPS);
        }
    }
}

TEST(TestOptimizerRMSprop, TwoStepsNotCenteredNoMomentumUnit)
{
    PROFILE_TEST
    raul::MemoryManager memory_manager;
    constexpr auto lr = 0.1_dt;
    constexpr auto alpha = 0.9_dt;
    constexpr auto eps = 0.1_dt;
    constexpr auto decay = 0.1_dt;
    constexpr auto momentum = 0.0_dt;
    constexpr auto centered = false;
    constexpr auto EPS = 1.0e-5_dt;

    const raul::Tensor param{ 0.749658_dt, 0.604651_dt, 0.109958_dt, 0.212090_dt, 0.970375_dt, 0.836909_dt, 0.281987_dt, 0.374158_dt, 0.023701_dt, 0.491013_dt, 0.123471_dt, 0.114322_dt,
                              0.472450_dt, 0.575073_dt, 0.295235_dt, 0.796689_dt, 0.195730_dt, 0.953685_dt, 0.842650_dt, 0.078359_dt, 0.375558_dt, 0.522561_dt, 0.572951_dt, 0.618587_dt };
    const raul::Tensor grad{ 0.696214_dt, 0.529950_dt, 0.256036_dt, 0.736594_dt, 0.020376_dt, 0.203647_dt, 0.374835_dt, 0.256443_dt, 0.325083_dt, 0.090189_dt, 0.393642_dt, 0.606878_dt,
                             0.174267_dt, 0.474340_dt, 0.857925_dt, 0.448600_dt, 0.513896_dt, 0.456866_dt, 0.601191_dt, 0.817920_dt, 0.973623_dt, 0.817528_dt, 0.974707_dt, 0.463839_dt };
    const raul::Tensor paramAfterFirstStep{ 0.525392_dt,  0.398720_dt, -0.034819_dt, -0.011030_dt, 0.884753_dt, 0.686363_dt, 0.104791_dt, 0.221841_dt,
                                            -0.137170_dt, 0.394315_dt, -0.054295_dt, -0.094901_dt, 0.342186_dt, 0.376759_dt, 0.062086_dt, 0.598875_dt,
                                            -0.002808_dt, 0.752603_dt, 0.626254_dt,  -0.150302_dt, 0.134665_dt, 0.290650_dt, 0.330894_dt, 0.421135_dt };
    const raul::Tensor paramAfterSecondStep{ 0.351683_dt,  0.237147_dt, -0.153323_dt, -0.183991_dt, 0.811821_dt,  0.563624_dt, -0.037048_dt, 0.097808_dt,
                                             -0.267402_dt, 0.312554_dt, -0.196533_dt, -0.258680_dt, 0.234502_dt,  0.220331_dt, -0.117376_dt, 0.442787_dt,
                                             -0.159389_dt, 0.594299_dt, 0.457711_dt,  -0.326867_dt, -0.049745_dt, 0.111985_dt, 0.145745_dt,  0.265293_dt };
    {
        raul::optimizers::RMSprop optimizer{ lr, alpha, eps, decay, momentum, centered };
        auto& params = *memory_manager.createTensor("params", 1, 2, 3, 4, param);
        auto& gradients = *memory_manager.createTensor("gradients", 1, 2, 3, 4, grad);

        // First step
        optimizer(memory_manager, params, gradients);

        for (size_t i = 0; i < params.size(); ++i)
        {
            EXPECT_NEAR(paramAfterFirstStep[i], params[i], EPS);
        }

        // Second step
        optimizer(memory_manager, params, gradients);

        for (size_t i = 0; i < params.size(); ++i)
        {
            EXPECT_NEAR(paramAfterSecondStep[i], params[i], EPS);
        }
    }
}

TEST(TestOptimizerRMSprop, TwoStepsNotCenteredTFStyleUnit)
{
    PROFILE_TEST
    raul::MemoryManager memory_manager;
    constexpr auto lr = 0.1_dt;
    constexpr auto alpha = 0.9_dt;
    constexpr auto eps = 0.1_dt;
    constexpr auto decay = 0.0_dt;
    constexpr auto momentum = 0.1_dt;
    constexpr auto centered = false;
    constexpr auto tfStyle = true;
    constexpr auto EPS = 1.0e-5_dt;

    const raul::Tensor param{ 1.5110626_dt,   0.42292204_dt,  -0.41969493_dt, -1.0360372_dt,  -1.2368279_dt, 0.47027302_dt, -0.01397489_dt, 1.1888583_dt,
                              0.60253334_dt,  0.5997111_dt,   -0.7057119_dt,  -0.43297544_dt, 0.7936245_dt,  -0.6974926_dt, -0.9598332_dt,  -0.9006969_dt,
                              -0.36081055_dt, -0.22377317_dt, 0.30383846_dt,  0.52152544_dt,  0.1554326_dt,  1.5885501_dt,  -0.7958055_dt,  0.07794423_dt };
    const raul::Tensor paramAfterFirstStep{ 1.2473527_dt,  0.29974532_dt,  -0.29731688_dt, -0.8085083_dt, -0.9909208_dt, 0.335698_dt,    -0.00955607_dt, 0.94685745_dt,
                                            0.43933123_dt, 0.4370709_dt,   -0.5233781_dt,  -0.3073284_dt, 0.59704304_dt, -0.51658463_dt, -0.7408552_dt,  -0.68906116_dt,
                                            -0.2534846_dt, -0.15471771_dt, 0.21190614_dt,  0.37529635_dt, 0.10686368_dt, 1.3209327_dt,   -0.5988931_dt,  0.05337063_dt };
    const raul::Tensor paramAfterSecondStep{ 1.0372865_dt,   0.2026748_dt,   -0.2008816_dt,  -0.6277918_dt, -0.79534674_dt, 0.229552_dt,    -0.0060927_dt,  0.75444734_dt,
                                             0.31031582_dt,  0.3085053_dt,   -0.37902588_dt, -0.2082925_dt, 0.4412466_dt,   -0.37337673_dt, -0.56703365_dt, -0.52115375_dt,
                                             -0.16900417_dt, -0.10049157_dt, 0.13961849_dt,  0.25985277_dt, 0.06876031_dt,  1.1076612_dt,   -0.44283062_dt, 0.03410574_dt };
    {
        raul::optimizers::RMSprop optimizer{ lr, alpha, eps, decay, momentum, centered, tfStyle };
        auto& params = *memory_manager.createTensor("params", 1, 2, 3, 4, param);

        // First step
        optimizer(memory_manager, params, params);

        for (size_t i = 0; i < params.size(); ++i)
        {
            EXPECT_NEAR(paramAfterFirstStep[i], params[i], EPS);
        }

        // Second step
        optimizer(memory_manager, params, params);

        for (size_t i = 0; i < params.size(); ++i)
        {
            EXPECT_NEAR(paramAfterSecondStep[i], params[i], EPS);
        }
    }
}

TEST(TestOptimizerRMSprop, TwoStepsTFStyleUnit)
{
    PROFILE_TEST
    raul::MemoryManager memory_manager;
    constexpr auto lr = 0.1_dt;
    constexpr auto alpha = 0.9_dt;
    constexpr auto eps = 0.1_dt;
    constexpr auto decay = 0.0_dt;
    constexpr auto momentum = 0.1_dt;
    constexpr auto centered = true;
    constexpr auto tfStyle = true;
    constexpr auto EPS = 1.0e-5_dt;

    const raul::Tensor param{ 1.0372865_dt,   0.2026748_dt,   -0.2008816_dt,  -0.6277918_dt, -0.79534674_dt, 0.229552_dt,    -0.0060927_dt,  0.75444734_dt,
                              0.31031582_dt,  0.3085053_dt,   -0.37902588_dt, -0.2082925_dt, 0.4412466_dt,   -0.37337673_dt, -0.56703365_dt, -0.52115375_dt,
                              -0.16900417_dt, -0.10049157_dt, 0.13961849_dt,  0.25985277_dt, 0.06876031_dt,  1.1076612_dt,   -0.44283062_dt, 0.03410574_dt };
    const raul::Tensor paramAfterFirstStep{ 0.8034859_dt,   0.13973624_dt,  -0.1384803_dt,  -0.45722586_dt, -0.59457576_dt, 0.15862368_dt,  -0.00416605_dt, 0.56044185_dt,
                                            0.2161798_dt,   0.21487507_dt,  -0.26623726_dt, -0.14367414_dt, 0.31253427_dt,  -0.26208052_dt, -0.40912014_dt, -0.37342036_dt,
                                            -0.11623431_dt, -0.06885678_dt, 0.09584951_dt,  0.18006864_dt,  0.04706251_dt,  0.8661924_dt,   -0.31372544_dt, 0.0233262_dt };
    const raul::Tensor paramAfterSecondStep{ 0.6130109_dt,   0.0902389_dt,   -0.08940828_dt, -0.3204778_dt,  -0.43241364_dt, 0.10279267_dt,  -0.00265599_dt, 0.40401128_dt,
                                             0.14184621_dt,  0.14094658_dt,  -0.17689902_dt, -0.09284654_dt, 0.21027792_dt,  -0.17394763_dt, -0.28288537_dt, -0.2555893_dt,
                                             -0.07477544_dt, -0.04404078_dt, 0.06148778_dt,  0.1171969_dt,   0.03004941_dt,  0.6690629_dt,   -0.21114905_dt, 0.01487664_dt };
    {
        raul::optimizers::RMSprop optimizer{ lr, alpha, eps, decay, momentum, centered, tfStyle };
        auto& params = *memory_manager.createTensor("params", 1, 2, 3, 4, param);

        // First step
        optimizer(memory_manager, params, params);

        for (size_t i = 0; i < params.size(); ++i)
        {
            EXPECT_NEAR(paramAfterFirstStep[i], params[i], EPS);
        }

        // Second step
        optimizer(memory_manager, params, params);

        for (size_t i = 0; i < params.size(); ++i)
        {
            EXPECT_NEAR(paramAfterSecondStep[i], params[i], EPS);
        }
    }
}

#ifdef ANDROID
TEST(TestOptimizerRMSprop, TwoStepsFP16Unit)
{
    PROFILE_TEST
    raul::MemoryManagerFP16 memory_manager;
    constexpr auto lr = 0.1_dt;
    constexpr auto alpha = 0.9_dt;
    constexpr auto eps = 0.1_dt;
    constexpr auto decay = 0.1_dt;
    constexpr auto momentum = 0.1_dt;
    constexpr auto centered = true;
    constexpr auto EPS = 1.0e-2_dt;

    const raul::TensorFP16 param{ 0.496257_hf, 0.768222_hf, 0.088477_hf, 0.132030_hf, 0.307423_hf, 0.634079_hf, 0.490093_hf, 0.896445_hf, 0.455628_hf, 0.632306_hf, 0.348893_hf, 0.401717_hf,
        0.022326_hf, 0.168859_hf, 0.293888_hf, 0.518522_hf, 0.697668_hf, 0.800011_hf, 0.161029_hf, 0.282269_hf, 0.681609_hf, 0.915194_hf, 0.397100_hf, 0.874156_hf };
    const raul::TensorFP16 grad{ 0.419408_hf, 0.552907_hf, 0.952738_hf, 0.036165_hf, 0.185231_hf, 0.373417_hf, 0.305100_hf, 0.932000_hf, 0.175910_hf, 0.269834_hf, 0.150680_hf, 0.031720_hf,
        0.208130_hf, 0.929799_hf, 0.723109_hf, 0.742336_hf, 0.526296_hf, 0.243658_hf, 0.584592_hf, 0.033153_hf, 0.138717_hf, 0.242235_hf, 0.815469_hf, 0.793161_hf };
    const raul::TensorFP16 paramAfterFirstStep{ 0.301402_hf, 0.550261_hf, -0.159050_hf, 0.089031_hf, 0.176365_hf,  0.445016_hf,  0.318390_hf, 0.645114_hf,
        0.322565_hf, 0.465707_hf, 0.229687_hf,  0.342580_hf, -0.106645_hf, -0.077670_hf, 0.062883_hf, 0.283733_hf,
        0.483886_hf, 0.635799_hf, -0.053345_hf, 0.230434_hf, 0.553956_hf,  0.748422_hf,  0.157254_hf, 0.632354_hf };
    const raul::TensorFP16 paramAfterSecondStep{ 0.121381_hf, 0.351013_hf, -0.382218_hf, 0.046700_hf, 0.051836_hf,  0.269886_hf,  0.158099_hf,  0.418925_hf,
        0.196239_hf, 0.309829_hf, 0.115852_hf,  0.284721_hf, -0.229299_hf, -0.300042_hf, -0.147013_hf, 0.070777_hf,
        0.288081_hf, 0.481993_hf, -0.249640_hf, 0.179576_hf, 0.432488_hf,  0.592395_hf,  -0.059773_hf, 0.413759_hf };
    {
        raul::optimizers::RMSprop optimizer{ lr, alpha, eps, decay, momentum, centered };
        auto& params = *memory_manager.createTensor("params", 1, 2, 3, 4, param);
        auto& gradients = *memory_manager.createTensor("gradients", 1, 2, 3, 4, grad);

        // First step
        optimizer(memory_manager, params, gradients);

        for (size_t i = 0; i < params.size(); ++i)
        {
            EXPECT_NEAR(TODTYPE(paramAfterFirstStep[i]), TODTYPE(params[i]), EPS);
        }

        // Second step
        optimizer(memory_manager, params, gradients);

        for (size_t i = 0; i < params.size(); ++i)
        {
            EXPECT_NEAR(TODTYPE(paramAfterSecondStep[i]), TODTYPE(params[i]), EPS);
        }
    }
}

TEST(TestOptimizerRMSprop, TwoStepsNotCenteredNoMomentumFP16Unit)
{
    PROFILE_TEST
    raul::MemoryManagerFP16 memory_manager;
    constexpr auto lr = 0.1_dt;
    constexpr auto alpha = 0.9_dt;
    constexpr auto eps = 0.1_dt;
    constexpr auto decay = 0.1_dt;
    constexpr auto momentum = 0.0_dt;
    constexpr auto centered = false;
    constexpr auto EPS = 1.0e-3_dt;

    const raul::TensorFP16 param{ 0.749658_hf, 0.604651_hf, 0.109958_hf, 0.212090_hf, 0.970375_hf, 0.836909_hf, 0.281987_hf, 0.374158_hf, 0.023701_hf, 0.491013_hf, 0.123471_hf, 0.114322_hf,
                              0.472450_hf, 0.575073_hf, 0.295235_hf, 0.796689_hf, 0.195730_hf, 0.953685_hf, 0.842650_hf, 0.078359_hf, 0.375558_hf, 0.522561_hf, 0.572951_hf, 0.618587_hf };
    const raul::TensorFP16 grad{ 0.696214_hf, 0.529950_hf, 0.256036_hf, 0.736594_hf, 0.020376_hf, 0.203647_hf, 0.374835_hf, 0.256443_hf, 0.325083_hf, 0.090189_hf, 0.393642_hf, 0.606878_hf,
                             0.174267_hf, 0.474340_hf, 0.857925_hf, 0.448600_hf, 0.513896_hf, 0.456866_hf, 0.601191_hf, 0.817920_hf, 0.973623_hf, 0.817528_hf, 0.974707_hf, 0.463839_hf };
    const raul::TensorFP16 paramAfterFirstStep{ 0.525392_hf,  0.398720_hf, -0.034819_hf, -0.011030_hf, 0.884753_hf, 0.686363_hf, 0.104791_hf, 0.221841_hf,
                                            -0.137170_hf, 0.394315_hf, -0.054295_hf, -0.094901_hf, 0.342186_hf, 0.376759_hf, 0.062086_hf, 0.598875_hf,
                                            -0.002808_hf, 0.752603_hf, 0.626254_hf,  -0.150302_hf, 0.134665_hf, 0.290650_hf, 0.330894_hf, 0.421135_hf };
    const raul::TensorFP16 paramAfterSecondStep{ 0.351683_hf,  0.237147_hf, -0.153323_hf, -0.183991_hf, 0.811821_hf,  0.563624_hf, -0.037048_hf, 0.097808_hf,
                                             -0.267402_hf, 0.312554_hf, -0.196533_hf, -0.258680_hf, 0.234502_hf,  0.220331_hf, -0.117376_hf, 0.442787_hf,
                                             -0.159389_hf, 0.594299_hf, 0.457711_hf,  -0.326867_hf, -0.049745_hf, 0.111985_hf, 0.145745_hf,  0.265293_hf };
    {
        raul::optimizers::RMSprop optimizer{ lr, alpha, eps, decay, momentum, centered };
        auto& params = *memory_manager.createTensor("params", 1, 2, 3, 4, param);
        auto& gradients = *memory_manager.createTensor("gradients", 1, 2, 3, 4, grad);

        // First step
        optimizer(memory_manager, params, gradients);

        for (size_t i = 0; i < params.size(); ++i)
        {
            EXPECT_NEAR(TODTYPE(paramAfterFirstStep[i]), TODTYPE(params[i]), EPS);
        }

        // Second step
        optimizer(memory_manager, params, gradients);

        for (size_t i = 0; i < params.size(); ++i)
        {
            EXPECT_NEAR(TODTYPE(paramAfterSecondStep[i]), TODTYPE(params[i]), EPS);
        }
    }
}
#endif // ANDROID

}