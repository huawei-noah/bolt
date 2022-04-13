// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef ADAMAX_H
#define ADAMAX_H

#include "Optimizer.h"
#include <iostream>

namespace raul::optimizers
{

/**
 * @brief AdaMax (Adaptive moment estimation Maximum)
 *
 * The AdaMax method is an extension of Adam optimization method based on the infinity norm.
 *
 *  \f[
 *      m_t =  \beta_1 m_{t-1} - (1-\beta_1) \nabla_{\theta} E(\theta_{t-1}),\\
 *      \u_t =  \max{\beta_2 \u_{t-1}, |\nabla_{\theta} E(\theta_{t-1})|),\\
 *      \theta_{t} =  \theta_{t-1} - \alpha \frac{m_t}{u_t(1-\beta_1^t)},
 *  \f]
 *  where
 *  - \f$m\f$ is the 1st moment vector (the mean of gradient),
 *  - \f$u\f$ is the Lp norm of 2st moment vector (the uncentered variance of gradient),
 *  - \f$\beta_1\f$ is the exponential decay rate for 1st moment,
 *  - \f$\beta_2\f$ is the exponential decay rate for 2st moment,
 *  - \f$\theta\f$ is a tuned parameter at specific step of the algorithm,
 *  - \f$\alpha\f$ is a learning rate,
 *  - \f$E(\theta)\f$ is an objective function (error function in our case).
 *
 *  Good default settings from the original article:
 *  - \f$\alpha = 0.002\f$
 *  - \f$\beta_1 = 0.9\f$
 *  - \f$\beta_2 = 0.999\f$
 *
 *  @see
 *  - D. P. Kingma and J. Ba, “Adam: A Method for Stochastic Optimization” arXiv:1412.6980 [cs], Jan. 2017.
 */
struct Adamax : public Optimizer
{
    explicit Adamax(const dtype alpha, const dtype beta_1 = 0.9_dt, const dtype beta_2 = 0.999_dt, const dtype epsilon = 1e-8_dt);
    void setLearningRate(dtype lr) final { m_alpha = lr; }
    [[nodiscard]] dtype getLearningRate() final { return m_alpha; }

  private:
    void optimize(MemoryManager& memory_manager, Tensor& param, const Tensor& grad) final;
    void optimize(MemoryManagerFP16& memory_manager, TensorFP16& param, const TensorFP16& grad) final;
    std::ostream& as_ostream(std::ostream& out) const final;

  private:
    dtype m_alpha;
    dtype m_beta_1;
    dtype m_beta_2;
    dtype m_epsilon;
};

} // raul::optimizers

#endif