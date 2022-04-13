// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef RPROP_H
#define RPROP_H

#include "Optimizer.h"
#include <iostream>

namespace raul::optimizers
{

/**
 * @brief Resilient backpropagation algorithm (Rprop)
 *
 *  RProp is a popular gradient descent algorithm that only
 *  uses the signs of gradients to compute updates. Parameters:
 *  1. learning rate (lr).
 *  2. alpha and beta.
 *  3. minimum and maximum step sizes (minStep, maxStep)
 *  An optimization algorithm works according to the following formula:
 *
 *  \f[
 *      \theta_{t} =  \theta_{t-1} - \eta_{t-1} sign (\nabla_{\theta} E(\theta_{t-1})),
 *  \f]
 *  where
 *  - \f$\theta\f$ is a tuned parameter at specific step of the algorithm,
 *  - \f$\eta\f$ depends on sign of previous derivitive. Possibilities:
 *  1.$\eta_{t} =  min(\eta_{t-1} * \alpha, \eta_{max}), if \nabla_{\theta} E(\theta_{t-1}) * \nabla_{\theta} E(\theta_{t}) > 0$
 *  2.$\eta_{t} =  max(\eta_{t-1} * \beta, \eta_{min}), if \nabla_{\theta} E(\theta_{t-1}) * \nabla_{\theta} E(\theta_{t}) < 0$
 *  3.$\eta_{t} =  \eta_{t-1}, otherwise$
 *  - \f$E(\theta)\f$ is an objective function (error function in our case).
 *
 *  @see
 *  - M. Riedmiller, H. Braun, “A direct adaptive method for faster backpropagation learning: The RPROP algorithm” IEEE International Conference on (pp. 586-591), 1993
 */
struct Rprop : public Optimizer
{
    explicit Rprop(const dtype lr, const dtype alpha = 0.5_dt, const dtype beta = 1.2_dt, const dtype minStep = 1.0e-6_dt, const dtype maxStep = 50.0_dt);
    void setLearningRate(dtype lr) final { mLearningRate = lr; }
    [[nodiscard]] dtype getLearningRate() final { return mLearningRate; }

  private:
    void optimize(MemoryManager& memory_manager, Tensor& param, const Tensor& grad) final;
    void optimize(MemoryManagerFP16& memory_manager, TensorFP16& param, const TensorFP16& grad) final;
    std::ostream& as_ostream(std::ostream& out) const final;

  private:
    dtype mLearningRate;
    dtype mAlpha;
    dtype mBeta;
    dtype mMinStep;
    dtype mMaxStep;
};

} // raul::optimizers

#endif