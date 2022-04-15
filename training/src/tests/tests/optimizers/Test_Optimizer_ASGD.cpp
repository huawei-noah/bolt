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

#include <training/base/common/Common.h>
#include <training/base/common/MemoryManager.h>
#include <training/base/optimizers/ASGD.h>

namespace UT
{

TEST(TestOptimizerASGD, StreamUnit)
{
    PROFILE_TEST
    std::ostringstream stream;
    const auto learning_rate = TODTYPE(0.01);
    {
        raul::optimizers::ASGD optimizer{ learning_rate };
        stream << optimizer;
        ASSERT_STREQ(stream.str().c_str(), "ASGD(lr=1.000000e-02, lambda=1.000000e-04, start point=1.000000e+06, weight decay=0.000000e+00)");
    }
}

TEST(TestOptimizerASGD, AssertLRNegativeUnit)
{
    PROFILE_TEST
    std::ostringstream stream;
    {
        const auto learning_rate = TODTYPE(-0.5);
        ASSERT_THROW(raul::optimizers::ASGD{ learning_rate }, raul::Exception);
    }
}

TEST(TestOptimizerASGD, OptimizerTwoStepsUnit)
{
    PROFILE_TEST
    raul::MemoryManager memory_manager;
    const auto learning_rate = TODTYPE(0.1);
    const auto lambda = TODTYPE(0.01);
    const auto alpha = TODTYPE(0.75);
    const auto start = TODTYPE(1.0e6);
    const auto wDecay = TODTYPE(0.1);
    const auto eps = TODTYPE(1.0e-6);
    const raul::Tensor param{ 0.496257_dt, 0.768222_dt, 0.088477_dt, 0.132030_dt, 0.307423_dt, 0.634079_dt, 0.490093_dt, 0.896445_dt, 0.455628_dt, 0.632306_dt, 0.348893_dt, 0.401717_dt,
                              0.022326_dt, 0.168859_dt, 0.293888_dt, 0.518522_dt, 0.697668_dt, 0.800011_dt, 0.161029_dt, 0.282269_dt, 0.681609_dt, 0.915194_dt, 0.397100_dt, 0.874156_dt };
    const raul::Tensor grad{ 0.419408_dt, 0.552907_dt, 0.952738_dt, 0.036165_dt, 0.185231_dt, 0.373417_dt, 0.305100_dt, 0.932000_dt, 0.175910_dt, 0.269834_dt, 0.150680_dt, 0.031720_dt,
                             0.208130_dt, 0.929799_dt, 0.723109_dt, 0.742336_dt, 0.526296_dt, 0.243658_dt, 0.584592_dt, 0.033153_dt, 0.138717_dt, 0.242235_dt, 0.815469_dt, 0.793161_dt };
    const raul::Tensor paramAfterFirstStep{
        0.448857_dt, 0.704481_dt, -0.007770_dt, 0.126962_dt, 0.285518_dt, 0.589762_dt, 0.454192_dt, 0.793384_dt, 0.433025_dt, 0.598368_dt, 0.329988_dt, 0.394126_dt,
        0.001267_dt, 0.074022_dt, 0.218345_dt,  0.438584_dt, 0.637364_dt, 0.766845_dt, 0.100799_dt, 0.275848_dt, 0.660239_dt, 0.880903_dt, 0.311185_dt, 0.785224_dt
    };
    const raul::Tensor paramAfterSecondStep{ 0.402014_dt, 0.641488_dt, -0.102887_dt, 0.121952_dt, 0.263871_dt,  0.545966_dt,  0.418713_dt, 0.691533_dt,
                                             0.410688_dt, 0.564827_dt, 0.311304_dt,  0.386625_dt, -0.019544_dt, -0.019702_dt, 0.143688_dt, 0.359586_dt,
                                             0.577768_dt, 0.734069_dt, 0.041276_dt,  0.269504_dt, 0.639121_dt,  0.847015_dt,  0.226279_dt, 0.697336_dt };
    raul::optimizers::ASGD optimizer{ learning_rate, lambda, alpha, start, wDecay };
    auto& params = *memory_manager.createTensor("params", 1, 2, 3, 4, param);
    auto& gradients = *memory_manager.createTensor("gradients", 1, 2, 3, 4, grad);

    // First step
    optimizer(memory_manager, params, gradients);
    EXPECT_EQ(params.size(), gradients.size());
    for (size_t i = 0; i < params.size(); ++i)
    {
        EXPECT_NEAR(params[i], paramAfterFirstStep[i], eps);
    }

    // Second step
    optimizer(memory_manager, params, gradients);
    EXPECT_EQ(params.size(), gradients.size());
    for (size_t i = 0; i < params.size(); ++i)
    {
        EXPECT_NEAR(params[i], paramAfterSecondStep[i], eps);
    }
}

#ifdef ANDROID
TEST(TestOptimizerASGD, OptimizerTwoStepsFP16Unit)
{
    PROFILE_TEST
    raul::MemoryManagerFP16 memory_manager;
    const auto learning_rate = TODTYPE(0.1);
    const auto lambda = TODTYPE(0.01);
    const auto alpha = TODTYPE(0.75);
    const auto start = TODTYPE(1.0e6);
    const auto wDecay = TODTYPE(0.1);
    const auto eps = TODTYPE(1.0e-3);
    const raul::TensorFP16 param{ 0.496257_hf, 0.768222_hf, 0.088477_hf, 0.132030_hf, 0.307423_hf, 0.634079_hf, 0.490093_hf, 0.896445_hf, 0.455628_hf, 0.632306_hf, 0.348893_hf, 0.401717_hf,
                              0.022326_hf, 0.168859_hf, 0.293888_hf, 0.518522_hf, 0.697668_hf, 0.800011_hf, 0.161029_hf, 0.282269_hf, 0.681609_hf, 0.915194_hf, 0.397100_hf, 0.874156_hf };
    const raul::TensorFP16 grad{ 0.419408_hf, 0.552907_hf, 0.952738_hf, 0.036165_hf, 0.185231_hf, 0.373417_hf, 0.305100_hf, 0.932000_hf, 0.175910_hf, 0.269834_hf, 0.150680_hf, 0.031720_hf,
                             0.208130_hf, 0.929799_hf, 0.723109_hf, 0.742336_hf, 0.526296_hf, 0.243658_hf, 0.584592_hf, 0.033153_hf, 0.138717_hf, 0.242235_hf, 0.815469_hf, 0.793161_hf };
    const raul::TensorFP16 paramAfterFirstStep{
        0.448857_hf, 0.704481_hf, -0.007770_hf, 0.126962_hf, 0.285518_hf, 0.589762_hf, 0.454192_hf, 0.793384_hf, 0.433025_hf, 0.598368_hf, 0.329988_hf, 0.394126_hf,
        0.001267_hf, 0.074022_hf, 0.218345_hf,  0.438584_hf, 0.637364_hf, 0.766845_hf, 0.100799_hf, 0.275848_hf, 0.660239_hf, 0.880903_hf, 0.311185_hf, 0.785224_hf
    };
    const raul::Tensor paramAfterSecondStep{ 0.402014_hf, 0.641488_hf, -0.102887_hf, 0.121952_hf, 0.263871_hf,  0.545966_hf,  0.418713_hf, 0.691533_hf,
                                             0.410688_hf, 0.564827_hf, 0.311304_hf,  0.386625_hf, -0.019544_hf, -0.019702_hf, 0.143688_hf, 0.359586_hf,
                                             0.577768_hf, 0.734069_hf, 0.041276_hf,  0.269504_hf, 0.639121_hf,  0.847015_hf,  0.226279_hf, 0.697336_hf };
    raul::optimizers::ASGD optimizer{ learning_rate, lambda, alpha, start, wDecay };
    auto& params = *memory_manager.createTensor("params", 1, 2, 3, 4, param);
    auto& gradients = *memory_manager.createTensor("gradients", 1, 2, 3, 4, grad);

    // First step
    optimizer(memory_manager, params, gradients);
    EXPECT_EQ(params.size(), gradients.size());
    for (size_t i = 0; i < params.size(); ++i)
    {
        EXPECT_NEAR(TODTYPE(params[i]), TODTYPE(paramAfterFirstStep[i]), eps);
    }

    // Second step
    optimizer(memory_manager, params, gradients);
    EXPECT_EQ(params.size(), gradients.size());
    for (size_t i = 0; i < params.size(); ++i)
    {
        EXPECT_NEAR(TODTYPE(params[i]), TODTYPE(paramAfterSecondStep[i]), eps);
    }
}
#endif // ANDROID

} // UT namespace