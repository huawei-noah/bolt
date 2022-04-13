// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "Adamax.h"
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

Adamax::Adamax(const dtype alpha, const dtype beta_1, const dtype beta_2, const dtype epsilon)
    : m_alpha(alpha)
    , m_beta_1(beta_1)
    , m_beta_2(beta_2)
    , m_epsilon(epsilon)
{
    if (alpha <= alpha_lower_boundary)
    {
        THROW_NONAME("Adamax", "reset alpha>" + Conversions::toString(alpha_lower_boundary) + " (current alpha=" + Conversions::toString(alpha) + ")");
    }

    if (beta_1 < beta_lower_boundary || beta_1 >= beta_upper_boundary)
    {
        THROW_NONAME("Adamax",
                     "reset beta_1 from [" + Conversions::toString(beta_lower_boundary) + ", " + Conversions::toString(beta_upper_boundary) + ") (current beta_1=" + Conversions::toString(beta_1) +
                         ")");
    }

    if (beta_2 < beta_lower_boundary || beta_2 >= beta_upper_boundary)
    {
        THROW_NONAME("Adamax",
                     "reset beta_2 from [" + Conversions::toString(beta_lower_boundary) + ", " + Conversions::toString(beta_upper_boundary) + ") (current beta_2=" + Conversions::toString(beta_2) +
                         ")");
    }
}

void Adamax::optimize(MemoryManager& memory_manager, Tensor& param, const Tensor& grad)
{
    if (param.size() != grad.size())
    {
        THROW_NONAME("Adamax", "parameters and gradients must have the same size (" + Conversions::toString(param.size()) + " != " + Conversions::toString(grad.size()) + ")");
    }

    Tensor *b1tp, *mp, *up;
    if (!memory_manager.tensorExists(Name("Adamax") / param.getName() / "beta_1_t"))
    {
        b1tp = memory_manager.createTensor(Name("Adamax") / param.getName() / "beta_1_t", 1, 1, 1, 1, this->m_beta_1);
    }
    b1tp = &memory_manager.getTensor(Name("Adamax") / param.getName() / "beta_1_t");

    if (!memory_manager.tensorExists(Name("Adamax") / param.getName() / "m"))
    {
        mp = memory_manager.createTensor(Name("Adamax") / param.getName() / "m", 1, param.size(), 1, 1);
    }
    mp = &memory_manager.getTensor(Name("Adamax") / param.getName() / "m");

    if (!memory_manager.tensorExists(Name("Adamax") / param.getName() / "u"))
    {
        up = memory_manager.createTensor(Name("Adamax") / param.getName() / "u", 1, param.size(), 1, 1);
    }
    up = &memory_manager.getTensor(Name("Adamax") / param.getName() / "u");

    Tensor& beta_1_t = *b1tp;
    Tensor& m = *mp;
    Tensor& u = *up;

    const auto n = param.size();

#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (size_t i = 0; i < n; i++)
    {
        // m_new = beta_1*m + (1-beta_1)*grad
        m[i] = this->m_beta_1 * m[i] + (1.0_dt - this->m_beta_1) * grad[i];
        // u_new = max[u*beta_2, abs(grad)]
        u[i] = std::max(this->m_beta_2 * u[i], std::abs(grad[i]));
        // param_new = param - alpha m_new/(u_new (1-\beta_1^t)),
        param[i] = param[i] - this->m_alpha * m[i] / (u[i] * (1 - beta_1_t[0]) + this->m_epsilon);
    }

    beta_1_t[0] *= this->m_beta_1;
}

void Adamax::optimize(MemoryManagerFP16& memory_manager, TensorFP16& param, const TensorFP16& grad)
{
    if (param.size() != grad.size())
    {
        THROW_NONAME("Adamax", "parameters and gradients must have the same size (" + Conversions::toString(param.size()) + " != " + Conversions::toString(grad.size()) + ")");
    }

    TensorFP16 *b1tp, *mp, *up;
    if (!memory_manager.tensorExists(Name("Adamax") / param.getName() / "beta_1_t"))
    {
        b1tp = memory_manager.createTensor(Name("Adamax") / param.getName() / "beta_1_t", 1, 1, 1, 1, TOHTYPE(this->m_beta_1));
    }
    b1tp = &memory_manager.getTensor(Name("Adamax") / param.getName() / "beta_1_t");

    if (!memory_manager.tensorExists(Name("Adamax") / param.getName() / "m"))
    {
        mp = memory_manager.createTensor(Name("Adamax") / param.getName() / "m", 1, param.size(), 1, 1);
    }
    mp = &memory_manager.getTensor(Name("Adamax") / param.getName() / "m");

    if (!memory_manager.tensorExists(Name("Adamax") / param.getName() / "u"))
    {
        up = memory_manager.createTensor(Name("Adamax") / param.getName() / "u", 1, param.size(), 1, 1);
    }
    up = &memory_manager.getTensor(Name("Adamax") / param.getName() / "u");

    TensorFP16& beta_1_t = *b1tp;
    TensorFP16& m = *mp;
    TensorFP16& u = *up;

    const auto n = param.size();

#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (size_t i = 0; i < n; i++)
    {
        // m_new = beta_1*m + (1-beta_1)*grad
        const auto mTmp = this->m_beta_1 * TODTYPE(m[i]) + (1.0_dt - this->m_beta_1) * TODTYPE(grad[i]);
        m[i] = TOHTYPE(mTmp);
        // u_new = max[u*beta_2, abs(grad)]
        const auto uTmp = std::max(this->m_beta_2 * TODTYPE(u[i]), std::abs(TODTYPE(grad[i])));
        u[i] = TOHTYPE(uTmp);
        // param_new = param - alpha m_new/(u_new (1-\beta_1^t)),
        param[i] = TOHTYPE(TODTYPE(param[i]) - this->m_alpha * mTmp / (uTmp * (1.0_dt - TODTYPE(beta_1_t[0])) + this->m_epsilon));
    }

    beta_1_t[0] = TOHTYPE(this->m_beta_1 * TODTYPE(beta_1_t[0]));
}

std::ostream& Adamax::as_ostream(std::ostream& out) const
{
    out << "Adamax(alpha=" << std::scientific << this->m_alpha << ", beta_1=" << this->m_beta_1 << ", beta_2=" << this->m_beta_2 << ")";
    return out;
}

} // raul::optimizers