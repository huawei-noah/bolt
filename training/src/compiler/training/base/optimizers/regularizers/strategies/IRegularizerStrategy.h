// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef I_REGULARIZER_STRATEGY_H
#define I_REGULARIZER_STRATEGY_H

#include <iostream>

#include <training/base/common/Common.h>
#include <training/compiler/Workflow.h>

namespace raul::optimizers::Regularizers::Strategies
{
/**
 * @brief Regularizer strategy interface
 */
struct IRegularizerStrategy
{
    IRegularizerStrategy() = default;
    virtual ~IRegularizerStrategy() = default;

    IRegularizerStrategy(const IRegularizerStrategy& other) = delete;
    IRegularizerStrategy(const IRegularizerStrategy&& other) = delete;
    IRegularizerStrategy& operator=(const IRegularizerStrategy& other) = delete;
    IRegularizerStrategy& operator=(const IRegularizerStrategy&& other) = delete;

    virtual dtype operator()(dtype weight) const = 0;
    virtual dtype getPenalty(Workflow& net) const = 0;
    friend std::ostream& operator<<(std::ostream& out, const IRegularizerStrategy& instance) { return instance.as_ostream(out); }

  private:
    virtual std::ostream& as_ostream(std::ostream& out) const = 0;
};
} // namespace raul::optimizers::Regularizers::Strategies

#endif // I_REGULARIZER_STRATEGY_H