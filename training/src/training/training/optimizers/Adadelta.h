// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef ADADELTA_H
#define ADADELTA_H

#include "Optimizer.h"
#include <iostream>

namespace raul::optimizers
{
/**
 * @brief Adadelta (Adaptive gradient)
 *
 * Adadelta is a per-dimension learning rate method for gradient descent
 * which is improved version of Adagrad.
 *
 * Adadelta improves two main drawbacks of the Adagrad:
 * 1. the continual decay of learning rates throughout training,
 * 2. the need for a manually selected global learning rate.
 *
 *  \f[
 *      g_t = \nabla_{\theta} E(\theta_{t}),\\
 *      \mathrm{M}[\Delta g^2]_{t} = \rho \mathrm{M}[\Delta g^2]_{t-1} - (1-\rho) g^2_{t},\\
 *      \Delta \theta_{t} = - \frac{\sqrt{\mathrm{M}[\Delta \theta^2]_{t-1} + \epsilon}}{\sqrt{\mathrm{M}[\Delta g^2]_{t} + \epsilon}} g_t,\\
 *      \mathrm{M}[\Delta \theta^2]_{t} = \rho \mathrm{M}[\Delta \theta^2]_{t-1} - (1-\rho) \Delta \theta^2_{t},\\
 *      \theta_{t} =  \theta_{t-1} + \Delta \theta_{t},
 *  \f]
 *  where
 *  - \f$\epsilon\f$ is a small value to avoid division by zero,
 *  - \f$\theta\f$ is a tuned parameter at specific step of the algorithm,
 *  - \f$\rho\f$ is a momentum,
 *  - \f$E(\theta)\f$ is an objective function (error function in our case).
 *
 *  @see
 *  - M. D. Zeiler, “ADADELTA: An Adaptive Learning Rate Method” arXiv:1212.5701 [cs], Dec. 2012.
 */
struct Adadelta : public Optimizer
{
    explicit Adadelta(const dtype rho, const dtype epsilon = 1e-8_dt);

  private:
    void optimize(MemoryManager& memory_manager, Tensor& param, const Tensor& grad) final;
    void optimize(MemoryManagerFP16& memory_manager, TensorFP16& param, const TensorFP16& grad) final;
    std::ostream& as_ostream(std::ostream& out) const final;

  private:
    dtype m_rho;
    dtype m_epsilon;
};
} // raul::optimizers

#endif