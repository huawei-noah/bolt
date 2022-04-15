// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "LAMB.h"
#include <algorithm>
#include <iostream>
#include <stdexcept>

namespace
{
constexpr raul::dtype lowerBoundary = 0.0_dt;
constexpr raul::dtype upperBoundary = 1.0_dt;

constexpr raul::dtype weightNormLowerBoundary = 0.0_dt;
constexpr raul::dtype weightNormUpperBoundary = 10.0_dt;
}

namespace raul::optimizers
{

LAMB::LAMB(const dtype lr, const dtype beta1, const dtype beta2, const dtype epsilon, const dtype weightDecay, const bool adam)
    : mLearningRate(lr)
    , mBeta1(beta1)
    , mBeta2(beta2)
    , mEpsilon(epsilon)
    , mWeightDecay(weightDecay)
    , mAdam(adam)
{
    if (lr < lowerBoundary)
    {
        THROW_NONAME("LAMB", "reset lr>" + Conversions::toString(lowerBoundary) + " (current lr=" + Conversions::toString(lr) + ")");
    }
    if (epsilon < lowerBoundary)
    {
        THROW_NONAME("LAMB", "reset epsilon>" + Conversions::toString(lowerBoundary) + " (current epsilon=" + Conversions::toString(epsilon) + ")");
    }

    if (beta1 < lowerBoundary || beta1 >= upperBoundary)
    {
        THROW_NONAME("LAMB", "reset beta1 from [" + Conversions::toString(lowerBoundary) + ", " + Conversions::toString(upperBoundary) + ") (current beta1=" + Conversions::toString(beta1) + ")");
    }
    if (beta2 < lowerBoundary || beta2 >= upperBoundary)
    {
        THROW_NONAME("LAMB", "reset beta2 from [" + Conversions::toString(lowerBoundary) + ", " + Conversions::toString(upperBoundary) + ") (current beta2=" + Conversions::toString(beta1) + ")");
    }
}

void LAMB::optimize(MemoryManager& memory_manager, Tensor& param, const Tensor& grad)
{
    if (param.size() != grad.size())
    {
        THROW_NONAME("LAMB", "parameters and gradients must have the same size (" + Conversions::toString(param.size()) + " != " + Conversions::toString(grad.size()) + ")");
    }

    if (!memory_manager.tensorExists(Name("LAMB") / param.getName() / "m"))
    {
        // Exponential moving average of gradient values
        memory_manager.createTensor(Name("LAMB") / param.getName() / "m", 1, param.size(), 1, 1, 0.0_dt);
        // Exponential moving average of squared gradient values
        memory_manager.createTensor(Name("LAMB") / param.getName() / "v", 1, param.size(), 1, 1, 0.0_dt);
        // Temporal tensor to keep actual parameter updates
        memory_manager.createTensor(Name("LAMB") / param.getName() / "adam_steps", 1, param.size(), 1, 1, 0.0_dt);
    }

    Tensor& m = memory_manager[Name("LAMB") / param.getName() / "m"];
    Tensor& v = memory_manager[Name("LAMB") / param.getName() / "v"];
    Tensor& adamSteps = memory_manager[Name("LAMB") / param.getName() / "adam_steps"];

    // Calculate weight norm and adam norm
    dtype weightNorm = 0.0_dt;
    dtype adamNorm = 0.0_dt;
#if defined(_OPENMP)
#pragma omp parallel for reduction(+ : weightNorm, adamNorm)
#endif
    for (size_t i = 0; i < grad.size(); ++i)
    {
        // Decay the first and second moment running average coefficient
        m[i] = m[i] * mBeta1 + (1.0_dt - mBeta1) * grad[i];
        v[i] = v[i] * mBeta2 + (1.0_dt - mBeta2) * grad[i] * grad[i];
        // Adam steps
        adamSteps[i] = m[i] / (std::sqrt(v[i]) + mEpsilon) + mWeightDecay * param[i];
        // Weight and adam norms
        weightNorm += param[i] * param[i];
        adamNorm += adamSteps[i] * adamSteps[i];
    }
    weightNorm = std::clamp(std::sqrt(weightNorm), weightNormLowerBoundary, weightNormUpperBoundary);
    adamNorm = std::sqrt(adamNorm);

    // Update parameter
    OPENBLAS_CONST dtype sa = -mLearningRate * ((mAdam || weightNorm == 0.0_dt || adamNorm == 0.0_dt) ? 1.0_dt : weightNorm / adamNorm);
    OPENBLAS_CONST dtype* sx = &(adamSteps[0]);
    size_t incx = 1U;
    dtype* sy = &(param[0]);
    size_t incy = 1U;
    size_t xOffset = 0U;
    size_t yOffset = 0U;
    Common::axpy(param.size(), sa, sx, incx, sy, incy, xOffset, yOffset);
}

void LAMB::optimize(MemoryManagerFP16& memory_manager, TensorFP16& param, const TensorFP16& grad)
{
    if (param.size() != grad.size())
    {
        THROW_NONAME("LAMB", "parameters and gradients must have the same size (" + Conversions::toString(param.size()) + " != " + Conversions::toString(grad.size()) + ")");
    }

    if (!memory_manager.tensorExists(Name("LAMB") / param.getName() / "m"))
    {
        // Exponential moving average of gradient values
        memory_manager.createTensor(Name("LAMB") / param.getName() / "m", 1, param.size(), 1, 1, 0.0_hf);
        // Exponential moving average of squared gradient values
        memory_manager.createTensor(Name("LAMB") / param.getName() / "v", 1, param.size(), 1, 1, 0.0_hf);
        // Temporal tensor to keep actual parameter updates
        memory_manager.createTensor(Name("LAMB") / param.getName() / "adam_steps", 1, param.size(), 1, 1, 0.0_hf);
    }

    TensorFP16& m = memory_manager[Name("LAMB") / param.getName() / "m"];
    TensorFP16& v = memory_manager[Name("LAMB") / param.getName() / "v"];
    TensorFP16& adamSteps = memory_manager[Name("LAMB") / param.getName() / "adam_steps"];

    // Calculate weight norm and adam norm
    dtype weightNorm = 0.0_dt;
    dtype adamNorm = 0.0_dt;
#if defined(_OPENMP)
#pragma omp parallel for reduction(+ : weightNorm, adamNorm)
#endif
    for (size_t i = 0; i < grad.size(); ++i)
    {
        // Decay the first and second moment running average coefficient
        const auto mTmp = TODTYPE(m[i]) * mBeta1 + (1.0_dt - mBeta1) * TODTYPE(grad[i]);
        m[i] = TOHTYPE(mTmp);
        const auto vTmp = TODTYPE(v[i]) * mBeta2 + (1.0_dt - mBeta2) * TODTYPE(grad[i]) * TODTYPE(grad[i]);
        v[i] = TOHTYPE(vTmp);
        // Adam steps
        const auto adamStepTmp = mTmp / (std::sqrt(vTmp) + mEpsilon) + mWeightDecay * TODTYPE(param[i]);
        adamSteps[i] = TOHTYPE(adamStepTmp);
        // Weight and adam norms
        weightNorm += TODTYPE(param[i]) * TODTYPE(param[i]);
        adamNorm += adamStepTmp * adamStepTmp;
    }
    weightNorm = std::clamp(std::sqrt(weightNorm), weightNormLowerBoundary, weightNormUpperBoundary);
    adamNorm = std::sqrt(adamNorm);

    // Update parameter
    OPENBLAS_CONST dtype sa = -mLearningRate * ((mAdam || weightNorm == 0.0_dt || adamNorm == 0.0_dt) ? 1.0_dt : weightNorm / adamNorm);
    OPENBLAS_CONST half* sx = &(adamSteps[0]);
    size_t incx = 1U;
    half* sy = &(param[0]);
    size_t incy = 1U;
    size_t xOffset = 0U;
    size_t yOffset = 0U;
    Common::axpy(param.size(), sa, sx, incx, sy, incy, xOffset, yOffset);
}

std::ostream& LAMB::as_ostream(std::ostream& out) const
{
    out << "LAMB(lr=" << std::scientific << mLearningRate << ", beta1=" << mBeta1 << ", beta2=" << mBeta2;
    out << ", epsilon=" << mEpsilon << ", weight decay=" << mWeightDecay << ", adam: " << (mAdam ? "true" : "false") << ")";
    return out;
}

} // raul::optimizers