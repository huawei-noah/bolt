// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef RMSPROP_H
#define RMSPROP_H

#include "Optimizer.h"
#include <iostream>

namespace raul::optimizers
{

/**
 * @brief RMSprop algorithm
 *
 * By default, the implementation here takes the square root of the gradient average before
 * adding epsilon (PyTorch behaviour). The effective learning rate is thus :math:`\alpha/(\sqrt{v} + \epsilon)`
 * where :math:`\alpha` is the scheduled learning rate and :math:`v` is the weighted moving average
 * of the squared gradient.
 * Behaviour can be changed to TF by setting tfStyle = true.
 *
 */
struct RMSprop : public Optimizer
{
    explicit RMSprop(const dtype lr,
                     const dtype alpha = 0.99_dt,
                     const dtype eps = 1.0e-8_dt,
                     const dtype weightDecay = 0.0_dt,
                     const dtype momentum = 0.0_dt,
                     bool centered = false,
                     bool tfStyle = false);
    void setLearningRate(dtype lr) final { mLearningRate = lr; }
    [[nodiscard]] dtype getLearningRate() final { return mLearningRate; }

  private:
    void optimize(MemoryManager& memory_manager, Tensor& param, const Tensor& grad) final;
    void optimize(MemoryManagerFP16& memory_manager, TensorFP16& param, const TensorFP16& grad) final;
    std::ostream& as_ostream(std::ostream& out) const final;

  private:
    dtype mLearningRate;
    dtype mAlpha;
    dtype mEps;
    dtype mWeightDecay;
    dtype mMomentum;
    bool mCentered;
    bool mTFStyle;
};

} // raul::optimizers

#endif