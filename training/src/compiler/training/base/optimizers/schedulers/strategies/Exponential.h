// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef EXPONENTIAL_SCHEDULER_H
#define EXPONENTIAL_SCHEDULER_H

#include <training/base/optimizers/schedulers/LrScheduler.h>
#include <iostream>
#include <utility>

namespace raul::optimizers::Scheduler::Strategies
{
/**
 * @brief Exponential scheduler strategy
 *
 */
struct Exponential : public Base
{

    explicit Exponential(dtype decayRate)
        : decayRate(decayRate)
        , decaySteps(1U)
        , iterativeMode(false)
    {
    }

    Exponential(dtype decayRate, std::unique_ptr<Base> child)
        : Base(std::move(child))
        , decayRate(decayRate)
        , decaySteps(1U)
        , iterativeMode(false)
    {
    }

    Exponential(dtype decayRate, size_t decaySteps)
        : decayRate(decayRate)
        , decaySteps(decaySteps)
        , iterativeMode(false)
    {
    }

    Exponential(dtype decayRate, size_t decaySteps, std::unique_ptr<Base> child)
        : Base(std::move(child))
        , decayRate(decayRate)
        , decaySteps(decaySteps)
        , iterativeMode(false)
    {
    }

    void calculateLR(dtype baseLR, long long step, dtype& currentLR) override;

  private:
    std::ostream& as_ostream(std::ostream& out) const final;

    dtype decayRate;
    size_t decaySteps;
    bool iterativeMode;
};
} // raul::optimizers::Scheduler::Strategies

#endif // EXPONENTIAL_SCHEDULER_H