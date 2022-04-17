// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef ADAGRAD_H
#define ADAGRAD_H

#include "Optimizer.h"
#include <iostream>

namespace raul::optimizers
{
/**
 * @brief AdaGrad (Adaptive gradient)
 *
 *  AdaGrad is the simplest adaptive gradient descent method.
 *  A per-component learning rate reduces according to the history of its changes.
 *
 *  Technically, the sums of the squares of the gradient components are calculating and then
 *  its square roots are using as normalizing factors for the learning rate.
 *
 *  \f[
 *      g = g + \nabla^2_{\theta} E(\theta_{t-1}),\\
 *      \theta_{t} =  \theta_{t-1} - \alpha \frac{\nabla_{\theta} E(\theta_{t-1})}{\sqrt{g + \epsilon}},
 *  \f]
 *  where
 *  - \f$g\f$ is accumulated squares of the gradient components,
 *  - \f$\epsilon\f$ is a small value to avoid division by zero,
 *  - \f$\theta\f$ is a tuned parameter at specific step of the algorithm,
 *  - \f$\alpha\f$ is a learning rate,
 *  - \f$E(\theta)\f$ is an objective function (error function in our case).
 *
 *  Default parameters:
 *  - \f$\epsilon = 10^{-10}\f$
 *
 *  @see
 *  - J. Duchi, E. Hazan, and Y. Singer, “Adaptive Subgradient Methods for Online Learning and Stochastic Optimization” Journal of Machine Learning Research, vol. 12, no. Jul, pp. 2121–2159, 2011.
 *  - D. P. Kingma and J. Ba, “Adam: A Method for Stochastic Optimization” arXiv:1412.6980 [cs], Jan. 2017.
 */
struct Adagrad : public Optimizer
{
    explicit Adagrad(const dtype alpha, const dtype epsilon = 1e-10_dt);
    void setLearningRate(dtype lr) final { m_alpha = lr; }
    [[nodiscard]] dtype getLearningRate() final { return m_alpha; }

  private:
    void optimize(MemoryManager& memory_manager, Tensor& param, const Tensor& grad) final;
    void optimize(MemoryManagerFP16& memory_manager, TensorFP16& param, const TensorFP16& grad) final;
    std::ostream& as_ostream(std::ostream& out) const override;

  private:
    dtype m_alpha;
    dtype m_epsilon;
};
} // raul::optimizers

#endif