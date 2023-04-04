// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef LAMB_H
#define LAMB_H

#include "Optimizer.h"
#include <iostream>

namespace raul::optimizers
{

/**
 * @brief LAMB (Layer-wise Adaptive Moments optimizer for Batch training)
 *
 * This method can be summarized as LARS (Layerwise Adaptive Rate Scaling)
 * applied to Adam, since it’s just multiplying the old update step
 * by the trust ratio. Update rule:
 *
 *  \f[
 *      \m_t =  \beta_1 m_{t-1} + (1-\beta_1) \nabla_{\theta} E(\theta_{t-1}),\\
 *      \nu_t =  \beta_2 \nu_{t-1} + (1-\beta_2) \nabla^2_{\theta} E(\theta_{t-1}),\\
 *      \theta_{t} =  \theta_{t-1} - \frac{\r_1}{\r_2} \eta (\frac{\m_t}{\sqrt{\nu_t} + \epsilon} + \lambda * \theta_{t-1})
 *  \f]
 * Where:
 * r1 - L2-norm of weights.
 * r2 - L2-norm of the Adam update rule with weight decay.
 *
 *  @see
 *  - Yang You, Jing Li, Sashank Reddi, Jonathan Hseu, Sanjiv Kumar, Srinadh Bhojanapalli, Xiaodan Song, James Demmel, Kurt Keutzer, Cho-Jui Hsieh,
 *    “Large Batch Optimization for Deep Learning: Training BERT in 76 minutes” arXiv:1904.00962 [cs], Apr. 2019.
 *
 */
struct LAMB : public Optimizer
{
    explicit LAMB(const dtype lr, const dtype beta1 = 0.9_dt, const dtype beta2 = 0.999_dt, const dtype epsilon = 1e-6_dt, const dtype weightDecay = 0.0_dt, const bool adam = false);

    void setLearningRate(dtype lr) final { mLearningRate = lr; }
    [[nodiscard]] dtype getLearningRate() final { return mLearningRate; }

  private:
    void optimize(MemoryManager& memory_manager, Tensor& param, const Tensor& grad) final;
    void optimize(MemoryManagerFP16& memory_manager, TensorFP16& param, const TensorFP16& grad) final;
    std::ostream& as_ostream(std::ostream& out) const final;

  private:
    dtype mLearningRate;
    dtype mBeta1;
    dtype mBeta2;
    dtype mEpsilon;
    dtype mWeightDecay;
    bool mAdam;
};

} // raul::optimizers

#endif