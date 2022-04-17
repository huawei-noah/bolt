// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef SCHEDULER_H
#define SCHEDULER_H

#include <iostream>
#include <utility>

#include <training/base/common/Common.h>
#include <training/base/common/MemoryManager.h>
#include <training/base/optimizers/IOptimizer.h>

#include "training/base/optimizers/schedulers/strategies/Base.h"

namespace raul::optimizers::Scheduler
{

struct LrScheduler : IOptimizer
{
    LrScheduler(std::unique_ptr<Strategies::Base> strategy, std::unique_ptr<IOptimizer> optimizer)
        : mStrategy(std::move(strategy))
        , mOptimizer(std::move(optimizer))
        , currentStep(0)
        , baseLR(mOptimizer->getLearningRate())
    {
    }

    LrScheduler(const LrScheduler& other) = delete;
    LrScheduler(const LrScheduler&& other) = delete;
    LrScheduler& operator=(const LrScheduler& other) = delete;
    LrScheduler& operator=(const LrScheduler&& other) = delete;

    void step();
    void reset();
    void reset(size_t steps);

    void operator()(MemoryManager& memory_manager, Tensor& params, Tensor& grads) override;
    void operator()(MemoryManagerFP16& memory_manager, TensorFP16& params, TensorFP16& grads) override;
    void setLearningRate(dtype lr) override;
    [[nodiscard]] dtype getLearningRate() override;

    friend std::ostream& operator<<(std::ostream& out, const LrScheduler& instance)
    {
        out << "Scheduler: " << *instance.mStrategy;
        return out;
    }

  private:
    void calculateLR();

    void forward(size_t numberOfSteps);

    std::unique_ptr<Strategies::Base> mStrategy;
    std::unique_ptr<IOptimizer> mOptimizer;
    size_t currentStep;
    const dtype baseLR;
};

} // raul::optimizers::Scheduler

#endif // SCHEDULER_H