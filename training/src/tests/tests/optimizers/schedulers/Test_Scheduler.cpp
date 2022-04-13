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
#include <training/base/optimizers/Adam.h>
#include <training/base/optimizers/SGD.h>
#include <training/base/optimizers/schedulers/strategies/ClipLower.h>
#include <training/base/optimizers/schedulers/strategies/ClipUpper.h>
#include <training/base/optimizers/schedulers/strategies/Exponential.h>
#include <training/base/optimizers/schedulers/strategies/Lambda.h>
#include <training/base/optimizers/schedulers/strategies/StepOffset.h>
#include <training/base/optimizers/schedulers/strategies/WarmUp.h>

#include <tests/tools/TestTools.h>

namespace UT
{

TEST(TestScheduler, CustomFunctionalUnit)
{
    using namespace raul::optimizers::Scheduler;
    PROFILE_TEST
    const auto acc_eps = 1e-6_dt;
    const size_t tacotron_start_decay = 20'000;
    const size_t tacotron_decay_steps = 40'000;
    const raul::dtype tacotron_decay_rate = 0.5_dt;
    const auto initial_learning_rate = 1e-3_dt;
    const raul::dtype tacotron_final_learning_rate = 1e-4_dt;
    const bool warmup_enable = true;
    const size_t warmup_num_steps = 1000;

    const std::vector<std::pair<size_t, raul::dtype>> phase1Data{ { 50, 5.000e-05_dt },  { 100, 1.000e-04_dt }, { 150, 1.500e-04_dt }, { 200, 2.000e-04_dt }, { 250, 2.500e-04_dt },
                                                                  { 300, 3.000e-04_dt }, { 350, 3.500e-04_dt }, { 400, 4.000e-04_dt }, { 450, 4.500e-04_dt }, { 500, 5.000e-04_dt },
                                                                  { 550, 5.500e-04_dt }, { 600, 6.000e-04_dt }, { 650, 6.500e-04_dt }, { 700, 7.000e-04_dt }, { 750, 7.500e-04_dt },
                                                                  { 800, 8.000e-04_dt }, { 850, 8.500e-04_dt }, { 900, 9.000e-04_dt }, { 950, 9.500e-04_dt }, { 1000, 1.000e-03_dt },
                                                                  { 1050, 1.000e-03_dt } };
    const std::vector<std::pair<size_t, raul::dtype>> phase2Data{ { 10000, 1.000e-03_dt },  { 20000, 1.000e-03_dt },  { 30000, 8.409e-04_dt },  { 40000, 7.071e-04_dt },  { 50000, 5.946e-04_dt },
                                                                  { 60000, 5.000e-04_dt },  { 70000, 4.204e-04_dt },  { 80000, 3.536e-04_dt },  { 90000, 2.973e-04_dt },  { 100000, 2.500e-04_dt },
                                                                  { 110000, 2.102e-04_dt }, { 120000, 1.768e-04_dt }, { 130000, 1.487e-04_dt }, { 140000, 1.250e-04_dt }, { 150000, 1.051e-04_dt },
                                                                  { 160000, 1.000e-04_dt }, { 170000, 1.000e-04_dt }, { 180000, 1.000e-04_dt }, { 190000, 1.000e-04_dt } };

    const auto lambda = [&](raul::dtype baseLR, size_t step, raul::dtype& currLR) {
        // 1. Natural exponential decay
        const auto step_a = static_cast<raul::dtype>(step) - static_cast<raul::dtype>(tacotron_start_decay);
        const auto step_b = static_cast<raul::dtype>(tacotron_decay_steps);

        auto lr = baseLR * std::pow(tacotron_decay_rate, step_a / step_b);

        // 2. Clip learning rate by max and min values
        lr = std::min(std::max(lr, tacotron_final_learning_rate), baseLR);

        // 3. Warmup
        if (warmup_enable)
        {
            const auto warmup_percent_done = static_cast<raul::dtype>(step) / static_cast<raul::dtype>(warmup_num_steps);
            const auto warmup_learning_rate = baseLR * warmup_percent_done;
            lr = std::min(lr, warmup_learning_rate);
        }

        currLR = lr;
    };

    LrScheduler scheduler{ std::make_unique<Strategies::Lambda>(lambda), std::make_unique<raul::optimizers::SGD>(initial_learning_rate) };

    for (const auto& phaseData : { phase1Data, phase2Data })
    {
        for (auto [step, lr] : phaseData)
        {
            scheduler.reset(step);
            ASSERT_NEAR(scheduler.getLearningRate(), lr, acc_eps);
        }
    }
}

TEST(TestScheduler, CustomFunctionalCompositUnit)
{
    using namespace raul::optimizers::Scheduler;
    PROFILE_TEST
    const auto acc_eps = 1e-6_dt;
    const int tacotron_start_decay = 20'000;
    const size_t tacotron_decay_steps = 40'000;
    const raul::dtype tacotron_decay_rate = 0.5_dt;
    const auto initial_learning_rate = 1e-3_dt;
    const raul::dtype tacotron_final_learning_rate = 1e-4_dt;
    const size_t warmup_num_steps = 1000;

    const std::vector<std::pair<size_t, raul::dtype>> phase1Data{ { 50, 5.000e-05_dt },  { 100, 1.000e-04_dt }, { 150, 1.500e-04_dt }, { 200, 2.000e-04_dt }, { 250, 2.500e-04_dt },
                                                                  { 300, 3.000e-04_dt }, { 350, 3.500e-04_dt }, { 400, 4.000e-04_dt }, { 450, 4.500e-04_dt }, { 500, 5.000e-04_dt },
                                                                  { 550, 5.500e-04_dt }, { 600, 6.000e-04_dt }, { 650, 6.500e-04_dt }, { 700, 7.000e-04_dt }, { 750, 7.500e-04_dt },
                                                                  { 800, 8.000e-04_dt }, { 850, 8.500e-04_dt }, { 900, 9.000e-04_dt }, { 950, 9.500e-04_dt }, { 1000, 1.000e-03_dt },
                                                                  { 1050, 1.000e-03_dt } };
    const std::vector<std::pair<size_t, raul::dtype>> phase2Data{ { 10000, 1.000e-03_dt },  { 20000, 1.000e-03_dt },  { 30000, 8.409e-04_dt },  { 40000, 7.071e-04_dt },  { 50000, 5.946e-04_dt },
                                                                  { 60000, 5.000e-04_dt },  { 70000, 4.204e-04_dt },  { 80000, 3.536e-04_dt },  { 90000, 2.973e-04_dt },  { 100000, 2.500e-04_dt },
                                                                  { 110000, 2.102e-04_dt }, { 120000, 1.768e-04_dt }, { 130000, 1.487e-04_dt }, { 140000, 1.250e-04_dt }, { 150000, 1.051e-04_dt },
                                                                  { 160000, 1.000e-04_dt }, { 170000, 1.000e-04_dt }, { 180000, 1.000e-04_dt }, { 190000, 1.000e-04_dt } };

    auto strategyExpOffset = std::make_unique<Strategies::StepOffset>(-tacotron_start_decay, std::make_unique<Strategies::Exponential>(tacotron_decay_rate, tacotron_decay_steps));
    auto strategyClipped = std::make_unique<Strategies::ClipLower>(tacotron_final_learning_rate, std::make_unique<Strategies::ClipUpper>(std::move(strategyExpOffset)));
    auto strategy = std::make_unique<Strategies::WarmUp>(warmup_num_steps, true, std::move(strategyClipped));

    LrScheduler scheduler{ std::move(strategy), std::make_unique<raul::optimizers::SGD>(initial_learning_rate) };

    for (const auto& phaseData : { phase1Data, phase2Data })
    {
        for (auto [step, lr] : phaseData)
        {
            scheduler.reset(step);
            ASSERT_NEAR(scheduler.getLearningRate(), lr, acc_eps);
        }
    }
}

} // UT namespace
