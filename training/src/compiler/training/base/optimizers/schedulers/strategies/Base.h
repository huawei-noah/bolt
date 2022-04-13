// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef BASE_STRATEGY_H
#define BASE_STRATEGY_H

#include <iostream>
#include <utility>

#include <training/base/common/Common.h>
#include <training/base/common/MemoryManager.h>
#include <training/base/optimizers/IOptimizer.h>

namespace raul::optimizers::Scheduler::Strategies
{

/**
 * @brief Scheduler Base class
 */
struct Base
{
    Base() = default;

    explicit Base(std::unique_ptr<Base> child)
        : child(std::move(child))
    {
    }

    virtual ~Base() = default;
    virtual void calculateLR(dtype baseLR, long long step, dtype& currentLR);
    friend std::ostream& operator<<(std::ostream& out, const Base& instance) { return instance.as_ostream(out); }

  protected:
    std::unique_ptr<Base> child;

  private:
    virtual std::ostream& as_ostream(std::ostream& out) const { return out; }
};

} // raul::optimizers::Scheduler::Strategies

#endif // BASE_STRATEGY_H