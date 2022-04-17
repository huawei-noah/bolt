// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "ASGD.h"
#include <iostream>
#include <stdexcept>

namespace raul::optimizers
{

ASGD::ASGD(const dtype lr, const dtype lambda, const dtype alpha, const dtype startPoint, const dtype weightDecay)
    : mLearningRate(lr)
    , mLambda(lambda)
    , mAlpha(alpha)
    , mStartPoint(startPoint)
    , mWeightDecay(weightDecay)
    , mStep(0.0_dt)
    , mEta(lr)
    , mMu(1.0_dt)
{
    if (lr < 0.0_dt)
    {
        THROW_NONAME("ASGD", "reset lr>=0 (current lr=" + Conversions::toString(lr) + ")");
    }
    if (weightDecay < 0.0_dt)
    {
        THROW_NONAME("ASGD", "reset weight decay>=0 (current weight decay=" + Conversions::toString(weightDecay) + ")");
    }
}

void ASGD::optimize(MemoryManager&, Tensor& param, const Tensor& grad)
{
    if (param.size() != grad.size())
    {
        THROW_NONAME("ASGD", "parameters and gradients must have the same size (" + Conversions::toString(param.size()) + " != " + Conversions::toString(grad.size()) + ")");
    }

    mStep += 1.0_dt;

    size_t n = param.size();
    size_t incx = 1U;
    dtype* sy = &(param[0]);
    const dtype* sx = &(grad[0]);

    size_t incy = 1U;
    size_t xOffset = 0U;
    size_t yOffset = 0U;

    // Decay term
    OPENBLAS_CONST dtype sa2 = TODTYPE(-1.0) * mEta * (mLambda + mWeightDecay);
    Common::axpy(n, sa2, sy, incx, sy, incy, xOffset, yOffset);

    // Update param
    OPENBLAS_CONST dtype sa3 = TODTYPE(-1.0) * mEta;
    Common::axpy(n, sa3, sx, incx, sy, incy, xOffset, yOffset);

    // Update eta and mu
    mEta = mLearningRate / std::pow(1.0_dt + mLambda * mLearningRate * mStep, mAlpha);
    mMu = 1.0_dt / std::max(1.0_dt, mStep - mStartPoint);
}

void ASGD::optimize(MemoryManagerFP16&, TensorFP16& param, const TensorFP16& grad)
{
    if (param.size() != grad.size())
    {
        THROW_NONAME("ASGD", "parameters and gradients must have the same size (" + Conversions::toString(param.size()) + " != " + Conversions::toString(grad.size()) + ")");
    }

    mStep += 1.0_dt;

    size_t n = param.size();
    size_t incx = 1U;
    half* sy = &(param[0]);
    const half* sx = &(grad[0]);

    size_t incy = 1U;
    size_t xOffset = 0U;
    size_t yOffset = 0U;

    // Decay term
    OPENBLAS_CONST dtype sa2 = TODTYPE(-1.0) * mEta * (mLambda + mWeightDecay);
    Common::axpy(n, sa2, sy, incx, sy, incy, xOffset, yOffset);

    // Update param
    OPENBLAS_CONST dtype sa3 = TODTYPE(-1.0) * mEta;
    Common::axpy(n, sa3, sx, incx, sy, incy, xOffset, yOffset);

    // Update eta and mu
    mEta = mLearningRate / std::pow(1.0_dt + mLambda * mLearningRate * mStep, mAlpha);
    mMu = 1.0_dt / std::max(1.0_dt, mStep - mStartPoint);
}

std::ostream& ASGD::as_ostream(std::ostream& out) const
{
    out << "ASGD(lr=" << std::scientific << mLearningRate << ", lambda=" << mLambda << ", start point=" << mStartPoint << ", weight decay=" << mWeightDecay << ")";
    return out;
}

} // raul::optimizers