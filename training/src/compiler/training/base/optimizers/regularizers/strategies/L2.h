// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef L2_H
#define L2_H

#include "IRegularizerStrategy.h"
#include <training/compiler/Workflow.h>

namespace raul::optimizers::Regularizers::Strategies
{

struct L2 final : public IRegularizerStrategy
{
    explicit L2(const dtype lambda)
        : mLambda(lambda)
    {
    }

    /**
     * @brief Return additional correction to the weight
     *
     *  The regularization term in loss function gives the following correction
     *  \f[
     *  \lambda \frac{\partial}{\partial \omega_k}\sum_{i} \omega_i^2 = 2 \lambda \omega_k
     *  \f]
     *
     * @param weight (omega_k in formula above)
     * @return
     */
    dtype operator()(dtype weight) const override;

    /**
     * @brief Returns penalty to the loss function
     *
     * \f[
     * L = L_{0} + \lambda \sum_{i} \omega_i^2
     * \f]
     *
     * @param net
     * @return penalty
     */
    dtype getPenalty(Workflow& net) const override;

  private:
    std::ostream& as_ostream(std::ostream& out) const override;
    const dtype mLambda;
};

} // namespace raul::optimizers::Regularizers::Strategies

#endif // L2_H