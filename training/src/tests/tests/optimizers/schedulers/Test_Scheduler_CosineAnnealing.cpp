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
#include <training/base/optimizers/SGD.h>
#include <training/base/optimizers/schedulers/strategies/CosineAnnealing.h>

#include <tests/tools/TestTools.h>

namespace UT
{

using namespace std;

TEST(TestSchedulerCosineAnnealing, MultiStepUnit)
{
    using namespace raul::optimizers::Scheduler;
    PROFILE_TEST
    constexpr auto acc_eps = 1e-6_dt;
    const auto baseLR = 2.0_dt;
    const size_t numLoops = 1000;
    const float warmUpPow = 1.f;
    const float annealingPow = 2.f;
    const float warmUpPercentage = 0.01f;
    LrScheduler scheduler{ std::make_unique<Strategies::CosineAnnealing>(numLoops, 1.f, 0.f, warmUpPercentage, warmUpPow, annealingPow), std::make_unique<raul::optimizers::SGD>(baseLR) };

    vector<float> goldenLR = {4.89434837e-02f, 1.98394199e+00f, 1.91980636e+00f, 1.81056471e+00f,
                              1.66289814e+00f, 1.48562952e+00f, 1.28899123e+00f, 1.08379106e+00f,
                              8.80554690e-01f, 6.88725022e-01f, 5.15991828e-01f, 3.67811918e-01f,
                              2.47160479e-01f, 1.54530966e-01f, 8.81760996e-02f, 4.45586213e-02f,
                              1.89598363e-02f, 6.17877874e-03f, 1.24644664e-03f, 7.88907588e-05f,
                              0.00000000e+00f};

    for (size_t i = 0; i <= numLoops; ++i)
    {
        scheduler.step();
        if (i % 50 == 0)
        {
            EXPECT_NEAR(scheduler.getLearningRate(), goldenLR[i / 50], acc_eps);
        }
    }
}

TEST(TestSchedulerCosineAnnealing, MultiStep2Unit)
{
    using namespace raul::optimizers::Scheduler;
    PROFILE_TEST
    constexpr auto acc_eps = 1e-6_dt;
    const auto baseLR = 2.0_dt;
    const size_t numLoops = 1000;
    const float warmUpPow = 2.f;
    const float annealingPow = 2.f;
    const float warmUpPercentage = 0.1f;

    LrScheduler scheduler{ std::make_unique<Strategies::CosineAnnealing>(numLoops, 1.f, 0.f, warmUpPercentage, warmUpPow, annealingPow), std::make_unique<raul::optimizers::SGD>(baseLR) };

    vector<float> goldenLR = {1.21741336e-07f, 5.31904077e-01f, 2.00000000e+00f, 1.96973091e+00f,
                              1.88120373e+00f, 1.74102540e+00f, 1.55945649e+00f, 1.34937557e+00f,
                              1.12500000e+00f, 9.00509033e-01f, 6.88725022e-01f, 5.00000000e-01f,
                              3.41428667e-01f, 2.16468746e-01f, 1.25000000e-01f, 6.38003459e-02f,
                              2.73676013e-02f, 8.97459622e-03f, 1.81848999e-03f, 1.15402184e-04f,
                              0.00000000e+00f};

    for (size_t i = 0; i <= numLoops; ++i)
    {
        scheduler.step();
        if (i % 50 == 0)
        {
            EXPECT_NEAR(scheduler.getLearningRate(), goldenLR[i / 50], acc_eps);
        }
    }
}

} // UT namespace
