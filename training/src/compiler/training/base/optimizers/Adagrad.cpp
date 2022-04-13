// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "Adagrad.h"
#include <iostream>
#include <stdexcept>

namespace
{
constexpr raul::dtype alpha_lower_boundary = 0.0_dt;
constexpr raul::dtype epsilon_lower_boundary = 0.0_dt;
constexpr raul::dtype epsilon_upper_boundary = 0.1_dt;
}

namespace raul::optimizers
{

Adagrad::Adagrad(const dtype alpha, const dtype epsilon)
    : m_alpha(alpha)
    , m_epsilon(epsilon)
{
    if (alpha <= alpha_lower_boundary)
    {
        THROW_NONAME("Adagrad", "reset alpha>" + Conversions::toString(alpha_lower_boundary) + " (current alpha=" + Conversions::toString(alpha) + ")");
    }

    if (epsilon < epsilon_lower_boundary || epsilon >= epsilon_upper_boundary)
    {
        THROW_NONAME("Adagrad",
                     "reset epsilon from [" + Conversions::toString(epsilon_lower_boundary) + ", " + Conversions::toString(epsilon_upper_boundary) +
                         ") (current epsilon=" + Conversions::toString(epsilon) + ")");
    }
}

void Adagrad::optimize(MemoryManager& memory_manager, Tensor& param, const Tensor& grad)
{
    if (param.size() != grad.size())
    {
        THROW_NONAME("Adagrad", "parameters and gradients must have the same size (" + Conversions::toString(param.size()) + " != " + Conversions::toString(grad.size()) + ")");
    }

    Tensor* gp;
    if (!memory_manager.tensorExists(Name("Adagrad") / param.getName() / "g"))
    {
        gp = memory_manager.createTensor(Name("Adagrad") / param.getName() / "g", 1, param.size(), 1, 1);
    }
    gp = &memory_manager.getTensor(Name("Adagrad") / param.getName() / "g");

    Tensor& g = *gp;

    const auto n = param.size();

#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (size_t i = 0; i < n; i++)
    {
        g[i] += grad[i] * grad[i];
        // param_new = param - alpha*grad/(sqrt[g + epsilon])
        param[i] = param[i] - this->m_alpha * grad[i] / (std::sqrt(g[i]) + this->m_epsilon);
    }
}

void Adagrad::optimize(MemoryManagerFP16& memory_manager, TensorFP16& param, const TensorFP16& grad)
{
    if (param.size() != grad.size())
    {
        THROW_NONAME("Adagrad", "parameters and gradients must have the same size (" + Conversions::toString(param.size()) + " != " + Conversions::toString(grad.size()) + ")");
    }

    TensorFP16* gp;
    if (!memory_manager.tensorExists(Name("Adagrad") / param.getName() / "g"))
    {
        gp = memory_manager.createTensor(Name("Adagrad") / param.getName() / "g", 1, param.size(), 1, 1);
    }
    gp = &memory_manager.getTensor(Name("Adagrad") / param.getName() / "g");

    TensorFP16& g = *gp;

    const auto n = param.size();

#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (size_t i = 0; i < n; i++)
    {
        g[i] += grad[i] * grad[i];
        // param_new = param - alpha*grad/(sqrt[g + epsilon])
        param[i] = param[i] - TOHTYPE(this->m_alpha * TODTYPE(grad[i]) / (std::sqrt(TODTYPE(g[i])) + this->m_epsilon));
    }
}

std::ostream& Adagrad::as_ostream(std::ostream& out) const
{
    out << "Adagrad(alpha=" << std::scientific << this->m_alpha << ", epsilon=" << this->m_epsilon << ")";
    return out;
}

} // raul::optimizers