// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef STEP_OFFSET_SCHEDULER_H
#define STEP_OFFSET_SCHEDULER_H

#include "training/base/optimizers/schedulers/LrScheduler.h"
#include <iostream>
#include <utility>

namespace raul::optimizers::Scheduler::Strategies
{
/**
 * @brief Step offset scheduler strategy
 *
 */
struct StepOffset : public Base
{

    StepOffset(long long offset, std::unique_ptr<Base> child)
        : Base(std::move(child))
        , offset(offset)
    {
    }

    void calculateLR(dtype baseLR, long long step, dtype& currentLR) override;

  private:
    std::ostream& as_ostream(std::ostream& out) const final;

    const long long offset;
};
} // raul::optimizers::Scheduler::Strategies

#endif // STEP_OFFSET_SCHEDULER_H