// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "AdamW.h"
#include <iostream>
#include <stdexcept>

namespace
{

constexpr raul::dtype alpha_lower_boundary = 0.0_dt;
constexpr raul::dtype beta_lower_boundary = 0.0_dt;
constexpr raul::dtype beta_upper_boundary = 1.0_dt;

}

namespace raul::optimizers
{

AdamW::AdamW(const dtype alpha, const dtype beta_1, const dtype beta_2, const dtype epsilon, const dtype lambda)
    : mAlpha(alpha)
    , mBeta_1(beta_1)
    , mBeta_2(beta_2)
    , mEpsilon(epsilon)
    , mLambda(lambda)
{
    if (alpha <= alpha_lower_boundary)
    {
        THROW_NONAME("AdamW", "reset alpha >" + Conversions::toString(alpha_lower_boundary) + " (current alpha=" + Conversions::toString(alpha) + ")");
    }

    if (beta_1 < beta_lower_boundary || beta_1 >= beta_upper_boundary)
    {
        THROW_NONAME("AdamW",
                     "reset beta_1 from (" + Conversions::toString(beta_lower_boundary) + ", " + Conversions::toString(beta_upper_boundary) + ") (current beta_1=" + Conversions::toString(beta_1) +
                         ")");
    }

    if (beta_2 < beta_lower_boundary || beta_2 >= beta_upper_boundary)
    {
        THROW_NONAME("AdamW",
                     "reset beta_2 from (" + Conversions::toString(beta_lower_boundary) + ", " + Conversions::toString(beta_upper_boundary) + ") (current beta_2=" + Conversions::toString(beta_2) +
                         ")");
    }
}

void AdamW::optimize(MemoryManager& memory_manager, Tensor& param, const Tensor& grad)
{
    if (param.size() != grad.size())
    {
        THROW_NONAME("AdamW", "parameters and gradients must have the same size (" + Conversions::toString(param.size()) + " != " + Conversions::toString(grad.size()) + ")");
    }

    const auto n = param.size();
    Tensor *b1tp, *b2tp, *mp, *vp;
    if (!memory_manager.tensorExists(Name("AdamW") / param.getName() / "beta_1_t"))
    {
        b1tp = memory_manager.createTensor(Name("AdamW") / param.getName() / "beta_1_t", 1, 1, 1, 1, mBeta_1);
    }
    b1tp = &memory_manager.getTensor(Name("AdamW") / param.getName() / "beta_1_t");

    if (!memory_manager.tensorExists(Name("AdamW") / param.getName() / "beta_2_t"))
    {
        b2tp = memory_manager.createTensor(Name("AdamW") / param.getName() / "beta_2_t", 1, 1, 1, 1, mBeta_2);
    }
    b2tp = &memory_manager.getTensor(Name("AdamW") / param.getName() / "beta_2_t");

    if (!memory_manager.tensorExists(Name("AdamW") / param.getName() / "m"))
    {
        mp = memory_manager.createTensor(Name("AdamW") / param.getName() / "m", 1, n, 1, 1);
    }
    mp = &memory_manager.getTensor(Name("AdamW") / param.getName() / "m");

    if (!memory_manager.tensorExists(Name("AdamW") / param.getName() / "v"))
    {
        vp = memory_manager.createTensor(Name("AdamW") / param.getName() / "v", 1, n, 1, 1);
    }
    vp = &memory_manager.getTensor(Name("AdamW") / param.getName() / "v");

    Tensor& beta_1_t = *b1tp;
    Tensor& beta_2_t = *b2tp;
    Tensor& m = *mp;
    Tensor& v = *vp;

    const auto alpha_new = mAlpha * std::sqrt(1.0_dt - beta_2_t[0]) / (1.0_dt - beta_1_t[0]);
    const auto epsilon_new = mEpsilon * std::sqrt(1.0_dt - beta_2_t[0]);
    const auto lambdaW = (1.0_dt - mAlpha * mLambda);

#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (size_t i = 0; i < n; i++)
    {
        // m_new = beta_1*m + (1-beta_1)*grad
        m[i] = mBeta_1 * m[i] + (1.0_dt - mBeta_1) * grad[i];
        // v_new = beta_2*v + (1-beta_2)*grad*grad
        v[i] = mBeta_2 * v[i] + (1.0_dt - mBeta_2) * grad[i] * grad[i];
        // param_new = param - alpha_new*m_new/(sqrt(v_new) + epsilon_new) + lambda * param
        param[i] = lambdaW * param[i] - alpha_new * m[i] / (std::sqrt(v[i]) + epsilon_new);
    }

    beta_1_t[0] *= mBeta_1;
    beta_2_t[0] *= mBeta_2;
}

void AdamW::optimize(MemoryManagerFP16& memory_manager, TensorFP16& param, const TensorFP16& grad)
{
    if (param.size() != grad.size())
    {
        THROW_NONAME("AdamW", "parameters and gradients must have the same size (" + Conversions::toString(param.size()) + " != " + Conversions::toString(grad.size()) + ")");
    }

    const auto n = param.size();
    TensorFP16 *b1tp, *b2tp, *mp, *vp;
    if (!memory_manager.tensorExists(Name("AdamW") / param.getName() / "beta_1_t"))
    {
        b1tp = memory_manager.createTensor(Name("AdamW") / param.getName() / "beta_1_t", 1, 1, 1, 1, TOHTYPE(mBeta_1));
    }
    b1tp = &memory_manager.getTensor(Name("AdamW") / param.getName() / "beta_1_t");

    if (!memory_manager.tensorExists(Name("AdamW") / param.getName() / "beta_2_t"))
    {
        b2tp = memory_manager.createTensor(Name("AdamW") / param.getName() / "beta_2_t", 1, 1, 1, 1, TOHTYPE(mBeta_2));
    }
    b2tp = &memory_manager.getTensor(Name("AdamW") / param.getName() / "beta_2_t");

    if (!memory_manager.tensorExists(Name("AdamW") / param.getName() / "m"))
    {
        mp = memory_manager.createTensor(Name("AdamW") / param.getName() / "m", 1, n, 1, 1);
    }
    mp = &memory_manager.getTensor(Name("AdamW") / param.getName() / "m");

    if (!memory_manager.tensorExists(Name("AdamW") / param.getName() / "v"))
    {
        vp = memory_manager.createTensor(Name("AdamW") / param.getName() / "v", 1, n, 1, 1);
    }
    vp = &memory_manager.getTensor(Name("AdamW") / param.getName() / "v");

    TensorFP16& beta_1_t = *b1tp;
    TensorFP16& beta_2_t = *b2tp;
    TensorFP16& m = *mp;
    TensorFP16& v = *vp;

    const auto alpha_new = mAlpha * std::sqrt(1.0_dt - TODTYPE(beta_2_t[0])) / (1.0_dt - TODTYPE(beta_1_t[0]));
    const auto epsilon_new = mEpsilon * std::sqrt(1.0_dt - TODTYPE(beta_2_t[0]));
    const auto lambdaW = (1.0_dt - mAlpha * mLambda);

#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (size_t i = 0; i < n; i++)
    {
        // m_new = beta_1*m + (1-beta_1)*grad
        const auto mTmp = mBeta_1 * TODTYPE(m[i]) + (1.0_dt - mBeta_1) * TODTYPE(grad[i]);
        m[i] = TOHTYPE(mTmp);
        // v_new = beta_2*v + (1-beta_2)*grad*grad
        const auto vTmp = mBeta_2 * TODTYPE(v[i]) + (1.0_dt - mBeta_2) * TODTYPE(grad[i]) * TODTYPE(grad[i]);
        v[i] = TOHTYPE(vTmp);
        // param_new = param - alpha_new*m_new/(sqrt(v_new) + epsilon_new) + lambda * param
        param[i] = TOHTYPE(lambdaW * TODTYPE(param[i]) - alpha_new * mTmp / (std::sqrt(vTmp) + epsilon_new));
    }

    beta_1_t[0] = TOHTYPE(mBeta_1 * TODTYPE(beta_1_t[0]));
    beta_2_t[0] = TOHTYPE(mBeta_2 * TODTYPE(beta_2_t[0]));
}

std::ostream& AdamW::as_ostream(std::ostream& out) const
{
    out << "AdamW(alpha=" << std::scientific << this->mAlpha << ", beta_1=" << mBeta_1 << ", beta_2=" << mBeta_2;
    out << ", epsilon=" << this->mEpsilon << ", lambda=" << this->mLambda << ")";
    return out;
}

} // raul::optimizers