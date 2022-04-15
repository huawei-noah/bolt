// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef NESTEROV_H
#define NESTEROV_H

#include "Optimizer.h"
#include <iostream>

namespace raul::optimizers
{
/**
 * @brief Nesterov accelerated gradient (NAG) method
 *
 *  \f[
 *      \nu_{t} = \mu nu_{t-1} - \alpha \nabla_{\theta} E(\theta_{t-1}),\\
 *      \theta_{t} = \theta_{t-1} - \mu \nu_{t-1} + (1 + \mu) \nu_{t},
 *  \f]
 *  where
 *  - \f$\nu\f$ is a velocity,
 *  - \f$\mu\f$ is a momentum parameter,
 *  - \f$\theta\f$ is a tuned parameter at specific step of the algorithm,
 *  - \f$\alpha\f$ is a learning rate,
 *  - \f$E(\theta)\f$ is an objective function (error function in our case).
 *
 *  @see
 *  - Y. Bengio, N. Boulanger-Lewandowski, and R. Pascanu, “Advances in Optimizing Recurrent Networks” arXiv:1212.0901 [cs], Dec. 2012.
 */
struct Nesterov : public Optimizer
{
    Nesterov(const dtype lr, const dtype momentum);

  private:
    void optimize(MemoryManager& memory_manager, Tensor& param, const Tensor& grad) final;
    void optimize(MemoryManagerFP16& memory_manager, TensorFP16& param, const TensorFP16& grad) final;
    void setLearningRate(dtype lr) final { m_learning_rate = lr; }
    [[nodiscard]] dtype getLearningRate() final { return m_learning_rate; }

    std::ostream& as_ostream(std::ostream& out) const final;

  private:
    dtype m_learning_rate;
    dtype m_momentum;
};
} // raul::optimizers

#endif