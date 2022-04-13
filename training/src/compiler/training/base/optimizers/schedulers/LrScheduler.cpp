// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "LrScheduler.h"
#include "training/base/optimizers/schedulers/strategies/Base.h"

namespace raul::optimizers::Scheduler
{

void LrScheduler::step()
{
    ++currentStep;
    calculateLR();
}

void LrScheduler::reset()
{
    currentStep = 0;
    mOptimizer->setLearningRate(baseLR);
}

void LrScheduler::reset(size_t steps)
{
    reset();
    forward(steps);
}

void LrScheduler::operator()(MemoryManager& memory_manager, Tensor& params, Tensor& grads)
{
    mOptimizer->operator()(memory_manager, params, grads);
}

void LrScheduler::operator()(MemoryManagerFP16& memory_manager, TensorFP16& params, TensorFP16& grads)
{
    mOptimizer->operator()(memory_manager, params, grads);
}

void LrScheduler::setLearningRate(dtype lr)
{
    mOptimizer->setLearningRate(lr);
}
[[nodiscard]] dtype LrScheduler::getLearningRate()
{
    return mOptimizer->getLearningRate();
}

void LrScheduler::calculateLR()
{
    auto lr = mOptimizer->getLearningRate();
    mStrategy->calculateLR(baseLR, currentStep, lr);
    mOptimizer->setLearningRate(lr);
}

void LrScheduler::forward(size_t numberOfSteps)
{
    for (size_t i = 0; i < numberOfSteps; ++i)
    {
        step();
    }
}

} // raul::optimizers::Scheduler