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
#include <training/base/optimizers/schedulers/strategies/Lambda.h>

#include <tests/tools/TestTools.h>

namespace UT
{

TEST(TestSchedulerLambda, StreamUnit)
{
    PROFILE_TEST
    std::ostringstream testStream;
    std::ostringstream goldenStream;
    const auto lambda = [](raul::dtype baseLR, long long step, raul::dtype& currLR) { currLR = baseLR + 0.1_dt * static_cast<raul::dtype>(step); };
    raul::optimizers::Scheduler::Strategies::Lambda strategy{ lambda };
    testStream << strategy;
    goldenStream << "Strategies::Lambda()";
    ASSERT_STREQ(testStream.str().c_str(), goldenStream.str().c_str());
}

TEST(TestSchedulerLambda, OneStepUnit)
{
    using namespace raul::optimizers::Scheduler;
    PROFILE_TEST
    constexpr auto acc_eps = 1e-6_dt;
    const auto baseLR = 1.0_dt;
    const auto lambda = [](raul::dtype baseLR, long long step, raul::dtype& currLR) { currLR = baseLR + 0.1_dt * static_cast<raul::dtype>(step); };
    LrScheduler scheduler{ std::make_unique<Strategies::Lambda>(lambda), std::make_unique<raul::optimizers::SGD>(baseLR) };
    scheduler.step();
    const auto newLR = scheduler.getLearningRate();
    EXPECT_NEAR(newLR, baseLR + 0.1_dt, acc_eps);
}

TEST(TestSchedulerLambda, MultiStepUnit)
{
    using namespace raul::optimizers::Scheduler;
    PROFILE_TEST
    constexpr auto acc_eps = 1e-6_dt;
    const auto baseLR = 1.0_dt;
    const auto decay = -0.1_dt;
    const size_t steps = 15;
    const auto lambda = [&](raul::dtype baseLR, long long step, raul::dtype& currLR) { currLR = baseLR + decay * static_cast<raul::dtype>(step); };
    LrScheduler scheduler{ std::make_unique<Strategies::Lambda>(lambda), std::make_unique<raul::optimizers::SGD>(baseLR) };
    for (size_t i = 0; i < steps; ++i)
    {
        scheduler.step();
    }
    EXPECT_NEAR(scheduler.getLearningRate(), baseLR + steps * decay, acc_eps);
    scheduler.reset();
    EXPECT_NEAR(scheduler.getLearningRate(), baseLR, acc_eps);
}

TEST(TestSchedulerLambda, MultiResetUnit)
{
    using namespace raul::optimizers::Scheduler;
    PROFILE_TEST
    constexpr auto acc_eps = 1e-6_dt;
    const auto baseLR = 1.0_dt;
    const auto decay = -0.1_dt;
    const size_t steps = 15;
    const auto lambda = [&](raul::dtype baseLR, long long step, raul::dtype& currLR) { currLR = baseLR + decay * static_cast<raul::dtype>(step); };
    Strategies::Lambda strategy{ lambda };
    LrScheduler scheduler{ std::make_unique<Strategies::Lambda>(lambda), std::make_unique<raul::optimizers::SGD>(baseLR) };
    scheduler.reset(steps);
    EXPECT_NEAR(scheduler.getLearningRate(), baseLR + steps * decay, acc_eps);
    scheduler.reset();
    EXPECT_NEAR(scheduler.getLearningRate(), baseLR, acc_eps);
}

TEST(TestSchedulerLambda, CompositionUnit)
{
    using namespace raul::optimizers::Scheduler;
    PROFILE_TEST
    constexpr auto acc_eps = 1e-6_dt;
    const auto baseLR = 1.0_dt;
    const auto decay1 = -0.1_dt;
    const auto decay2 = -0.3_dt;
    const auto lambda1 = [&]([[maybe_unused]] raul::dtype baseLR, [[maybe_unused]] size_t step, raul::dtype& currLR) { currLR = currLR * decay1; };
    const auto lambda2 = [&]([[maybe_unused]] raul::dtype baseLR, [[maybe_unused]] size_t step, raul::dtype& currLR) { currLR = currLR * decay2; };
    LrScheduler scheduler{ std::make_unique<Strategies::Lambda>(lambda1, std::make_unique<Strategies::Lambda>(lambda2)), std::make_unique<raul::optimizers::SGD>(baseLR) };
    scheduler.step();
    EXPECT_NEAR(scheduler.getLearningRate(), baseLR * decay1 * decay2, acc_eps);
}

} // UT namespace
