// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef RANGER_H
#define RANGER_H

#include "Optimizer.h"
#include <array>
#include <iostream>
#include <tuple>

namespace raul::optimizers
{

/**
 * @brief Ranger algorithm
 *
 * The Ranger optimizer combines two very new developments - RAdam + Lookahead + Gradient Centralization - into
 * a single optimizer for deep learning.
 *
 *  @see
 *  - Gradient Centralization: H. Yong, J. Huang, X. Hua, L. Zhang, "Gradient Centralization: A New Optimization Technique for Deep Neural Networks"  arXiv:2004.01461v2 [cs.CV],
 *  - LookAhead: M. Zhang, J. Lucas, G. Hinton, J. Ba, "Lookahead Optimizer: k steps forward, 1 step back" arXiv:1907.08610 [cs], Dec 2019
 *  - RAdam: L. Liu, H. Jiang, P. He, W. Chen, X. Liu, J. Gao, J. Han, "On the variance of the adaptive learning rate and beyond" arXiv:1908.03265 [cs], Apr 2020
 */
struct Ranger : public Optimizer
{
    explicit Ranger(const dtype lr,
                    const dtype alpha = 0.5_dt,
                    const size_t k = 6,
                    const dtype nSmaThreshold = 5.0_dt,
                    const dtype beta1 = 0.95_dt,
                    const dtype beta2 = 0.999_dt,
                    const dtype eps = 1.0e-5_dt,
                    const dtype weightDecay = 0.0_dt,
                    bool useGc = true);
    void setLearningRate(dtype lr) final { mLearningRate = lr; }
    [[nodiscard]] dtype getLearningRate() final { return mLearningRate; }

  private:
    void optimize(MemoryManager& memory_manager, Tensor& param, const Tensor& grad) final;
    void optimize(MemoryManagerFP16& memory_manager, TensorFP16& param, const TensorFP16& grad) final;
    std::ostream& as_ostream(std::ostream& out) const final;

  private:
    dtype mLearningRate;
    dtype mAlpha;
    size_t mK;
    dtype mNSmaThreshold;
    dtype mBeta1;
    dtype mBeta2;
    dtype mEpsilon;
    dtype mWeightDecay;
    bool mUseGc;
    int mStep;

    std::array<std::tuple<int, dtype, dtype>, 10> mBuffered;
};

} // raul::optimizers

#endif