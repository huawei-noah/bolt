// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "Rprop.h"
#include <iostream>
#include <stdexcept>

using namespace raul;

namespace
{

template<typename T>
T sign(const T val)
{
    return static_cast<T>((TODTYPE(val) > 0.0_dt) - (TODTYPE(val) < 0.0_dt));
}
#if !defined(ANDROID)
template<>
half sign(const half val)
{
    return half_float::half_cast<half>((val > 0.0_hf) - (val < 0.0_hf));
}
#endif

constexpr raul::dtype lr_lower_boundary = 0.0_dt;
constexpr raul::dtype alpha_lower_boundary = 0.0_dt;
constexpr raul::dtype beta_lower_boundary = 1.0_dt;

}

namespace raul::optimizers
{

Rprop::Rprop(const dtype lr, const dtype alpha, const dtype beta, const dtype minStep, const dtype maxStep)
    : mLearningRate(lr)
    , mAlpha(alpha)
    , mBeta(beta)
    , mMinStep(minStep)
    , mMaxStep(maxStep)
{
    if (lr <= lr_lower_boundary)
    {
        THROW_NONAME("Rprop", "reset lr>" + Conversions::toString(lr_lower_boundary) + " (current lr=" + Conversions::toString(lr) + ")");
    }
    if (alpha < alpha_lower_boundary || alpha > beta_lower_boundary)
    {
        THROW_NONAME("Rprop",
                     "reset alpha from [" + Conversions::toString(alpha_lower_boundary) + "," + Conversions::toString(beta_lower_boundary) + "] (current alpha=" + Conversions::toString(alpha) + ")");
    }
    if (beta < beta_lower_boundary)
    {
        THROW_NONAME("Rprop", "reset beta from [" + Conversions::toString(beta_lower_boundary) + ",) (current beta=" + Conversions::toString(beta) + ")");
    }
}

void Rprop::optimize(MemoryManager& memory_manager, Tensor& param, const Tensor& grad)
{
    if (param.size() != grad.size())
    {
        THROW_NONAME("Rprop", "parameters and gradients must have the same size (" + Conversions::toString(param.size()) + " != " + Conversions::toString(grad.size()) + ")");
    }

    if (!memory_manager.tensorExists(Name("Rprop") / param.getName() / "prevGradSigns"))
    {
        Tensor* prevGradSigns = memory_manager.createTensor(Name("Rprop") / param.getName() / "prevGradSigns", 1, param.size(), 1, 1, 1.0_dt);
        memory_manager.createTensor(Name("Rprop") / param.getName() / "prevLRs", 1, param.size(), 1, 1, mLearningRate);
#if defined(_OPENMP)
#pragma omp parallel for
#endif
        for (size_t i = 0; i < param.size(); i++)
        {
            (*prevGradSigns)[i] = sign(grad[i]);
            param[i] = param[i] - mLearningRate * (*prevGradSigns)[i];
        }
        return;
    }

    Tensor& signs = memory_manager[Name("Rprop") / param.getName() / "prevGradSigns"];
    Tensor& LRs = memory_manager[Name("Rprop") / param.getName() / "prevLRs"];

#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (size_t i = 0; i < param.size(); i++)
    {
        if (signs[i] * sign(grad[i]) > 0.0_dt)
        {
            LRs[i] = std::min(LRs[i] * mBeta, mMaxStep);
        }
        else
        {
            if (signs[i] * sign(grad[i]) < 0.0_dt)
            {
                LRs[i] = std::max(LRs[i] * mAlpha, mMinStep);
                signs[i] = 0.0_dt;
                continue;
            }
        }
        signs[i] = sign(grad[i]);
        param[i] = param[i] - LRs[i] * signs[i];
    }
}

void Rprop::optimize(MemoryManagerFP16& memory_manager, TensorFP16& param, const TensorFP16& grad)
{
    if (param.size() != grad.size())
    {
        THROW_NONAME("Rprop", "parameters and gradients must have the same size (" + Conversions::toString(param.size()) + " != " + Conversions::toString(grad.size()) + ")");
    }

    if (!memory_manager.tensorExists(Name("Rprop") / param.getName() / "prevGradSigns"))
    {
        TensorFP16* prevGradSigns = memory_manager.createTensor(Name("Rprop") / param.getName() / "prevGradSigns", 1, param.size(), 1, 1, 1.0_hf);
        memory_manager.createTensor(Name("Rprop") / param.getName() / "prevLRs", 1, param.size(), 1, 1, TOHTYPE(mLearningRate));
#if defined(_OPENMP)
#pragma omp parallel for
#endif
        for (size_t i = 0; i < param.size(); i++)
        {
            (*prevGradSigns)[i] = sign(grad[i]);
            param[i] = TOHTYPE(TODTYPE(param[i]) - mLearningRate * TODTYPE((*prevGradSigns)[i]));
        }
        return;
    }

    TensorFP16& signs = memory_manager[Name("Rprop") / param.getName() / "prevGradSigns"];
    TensorFP16& LRs = memory_manager[Name("Rprop") / param.getName() / "prevLRs"];

#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (size_t i = 0; i < param.size(); i++)
    {
        if (signs[i] * sign(grad[i]) > 0.0_hf)
        {
            LRs[i] = TOHTYPE(std::min(TODTYPE(LRs[i]) * mBeta, mMaxStep));
        }
        else
        {
            if (signs[i] * sign(grad[i]) < 0.0_hf)
            {
                LRs[i] = TOHTYPE(std::max(TODTYPE(LRs[i]) * mAlpha, mMinStep));
                signs[i] = 0.0_hf;
                continue;
            }
        }
        signs[i] = sign(grad[i]);
        param[i] = param[i] - LRs[i] * signs[i];
    }
}

std::ostream& Rprop::as_ostream(std::ostream& out) const
{
    out << "Rprop(lr=" << std::scientific << mLearningRate << ", alpha=" << mAlpha << ", beta=" << mBeta << ", min step=" << mMinStep << ", max step=" << mMaxStep << ")";
    return out;
}

} // raul::optimizers