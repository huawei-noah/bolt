// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef ADAM_H
#define ADAM_H

#include "Optimizer.h"
#include <iostream>

namespace raul::optimizers
{
/**
 * @brief Adam (Adaptive moment estimation)
 *
 *  The Adam method computes individual adaptive learning rates for
 *  different parameters from estimates of first
 *  and second moments of the gradients. This method is combination
 *  of AdaGrad and RMSProp. AdaGrad works well with sparse gradients.
 *  RMSProp works well in on-line and non-stationary settings.
 *
 *  \f[
 *      m_t =  \beta_1 m_{t-1} - (1-\beta_1) \nabla_{\theta} E(\theta_{t-1}),\\
 *      \nu_t =  \beta_2 \nu_{t-1} - (1-\beta_2) \nabla^2_{\theta} E(\theta_{t-1}),\\
 *      \hat m_t = \frac{m}{1-\beta_1^t}, \\
 *      \hat \nu_t = \frac{\nu}{1-\beta_2^t}, \\
 *      \theta_{t} =  \theta_{t-1} - \alpha \frac{m_{t}}{\sqrt{\hat \nu_t} + \epsilon},
 *  \f]
 *  where
 *  - \f$m\f$ is the 1st moment vector (the mean of gradient),
 *  - \f$\nu\f$ is the 2st moment vector (the uncentered variance of gradient),
 *  - \f$\beta_1\f$ is the exponential decay rate for 1st moment,
 *  - \f$\beta_2\f$ is the exponential decay rate for 2st moment,
 *  - \f$\hat m\f$ is the bias-corrected 1st moment vector,
 *  - \f$\hat \nu\f$ is the bias-corrected 2st moment vector,
 *  - \f$\theta\f$ is a tuned parameter at specific step of the algorithm,
 *  - \f$\alpha\f$ is a learning rate,
 *  - \f$E(\theta)\f$ is an objective function (error function in our case).
 *
 *  Good default settings from the original article:
 *  - \f$\alpha = 0.0001\f$
 *  - \f$\beta_1 = 0.9\f$
 *  - \f$\beta_2 = 0.999\f$
 *  - \f$\epsilon = 10^{-8}\f$
 *
 *  @see
 *  - D. P. Kingma and J. Ba, “Adam: A Method for Stochastic Optimization” arXiv:1412.6980 [cs], Jan. 2017.
 */
struct Adam : public Optimizer
{
    explicit Adam(const dtype alpha = 0.0001_dt, const dtype beta_1 = 0.9_dt, const dtype beta_2 = 0.999_dt, const dtype epsilon = 1e-8_dt, bool use_simple_epsilon = false);

    void setLearningRate(dtype lr) final { m_alpha = lr; }
    [[nodiscard]] dtype getLearningRate() final { return m_alpha; }

  private:
    void optimize(MemoryManager& memory_manager, Tensor& param, const Tensor& grad) final;
    void optimize(MemoryManagerFP16& memory_manager, TensorFP16& param, const TensorFP16& grad) final;
    void optimize(OpenCLKernelManager& kernel_manager, MemoryManagerGPU& memory_manager, TensorGPU& param, const TensorGPU& grad) final;
    std::ostream& as_ostream(std::ostream& out) const final;

  private:
    dtype m_alpha;
    dtype m_beta_1;
    dtype m_beta_2;
    dtype m_epsilon;
    bool m_use_simple_epsilon;
};
} // raul::optimizers

#endif