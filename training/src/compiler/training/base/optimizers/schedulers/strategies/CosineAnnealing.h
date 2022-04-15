// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef COSINE_ANNEALING_SCHEDULER_H
#define COSINE_ANNEALING_SCHEDULER_H

#include "training/base/optimizers/schedulers/LrScheduler.h"
#include <iostream>
#include <utility>

namespace raul::optimizers::Scheduler::Strategies
{
/**
 * @brief Cosine Annealing with Warm-up
 *
 *  Conforms to original paper with default arguments
 *  Modifications:
 *    1. Adds warm-up stage
 *    2. Allows to specify separate powers for warmup and annealing stages.
 *
 *  @see
 *  - Ilya Loshchilov, Frank Hutter, �SGDR: Stochastic Gradient Descent with Warm Restarts� arXiv:1608.03983v5 [cs.LG], May 2017.
 */
struct CosineAnnealing : public Base
{

    CosineAnnealing(size_t num_loops, float max_a = 1.f, float min_a = 0.f, float warmup_percentage = 0.00f, float warmup_pow = 1.f, float annealing_pow = 1.f)
        : mNumLoops(num_loops)
        , mMaxA(max_a)
        , mMinA(min_a)
        , mWarmupPercentage(warmup_percentage)
        , mWarmupPow(warmup_pow)
        , mAnnealingPow(annealing_pow)
    {
        mWarmupLoops = static_cast<size_t>(mNumLoops * mWarmupPercentage);
        mAnnealingLoops = mNumLoops - mWarmupLoops;
    }

    CosineAnnealing(size_t num_loops, float max_a, float min_a, float warmup_percentage, float warmup_pow, float annealing_pow, std::unique_ptr<Base> child)
        : Base(std::move(child))
        , mNumLoops(num_loops)
        , mMaxA(max_a)
        , mMinA(min_a)
        , mWarmupPercentage(warmup_percentage)
        , mWarmupPow(warmup_pow)
        , mAnnealingPow(annealing_pow)
    {
        mWarmupLoops = static_cast<size_t>(mNumLoops * mWarmupPercentage);
        mAnnealingLoops = mNumLoops - mWarmupLoops;
    }

    void calculateLR(dtype baseLR, long long step, dtype& currentLR) override;

  private:
    std::ostream& as_ostream(std::ostream& out) const final;

    size_t mNumLoops;
    float mMaxA;
    float mMinA;
    float mWarmupPercentage;
    float mWarmupPow;
    float mAnnealingPow;

    size_t mWarmupLoops;
    size_t mAnnealingLoops;
};
} // raul::optimizers::Scheduler::Strategies

#endif // COSINE_ANNEALING_SCHEDULER_H