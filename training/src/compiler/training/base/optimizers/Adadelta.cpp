// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "Adadelta.h"
#include <iostream>
#include <stdexcept>

namespace
{
constexpr raul::dtype rho_lower_boundary = 0.0_dt;
constexpr raul::dtype rho_upper_boundary = 1.0_dt;
constexpr raul::dtype epsilon_lower_boundary = 0.0_dt;
constexpr raul::dtype epsilon_upper_boundary = 0.1_dt;
}

namespace raul::optimizers
{

Adadelta::Adadelta(const dtype rho, const dtype epsilon)
    : m_rho(rho)
    , m_epsilon(epsilon)
{
    if (rho < rho_lower_boundary || rho >= rho_upper_boundary)
    {
        THROW_NONAME("Adadelta",
                     "reset rho from [" + Conversions::toString(rho_lower_boundary) + ", " + Conversions::toString(rho_upper_boundary) + ") (current rho=" + Conversions::toString(epsilon) + ")");
    }

    if (epsilon < epsilon_lower_boundary || epsilon >= epsilon_upper_boundary)
    {
        THROW_NONAME("Adadelta",
                     "reset epsilon from [" + Conversions::toString(epsilon_lower_boundary) + ", " + Conversions::toString(epsilon_upper_boundary) +
                         ") (current epsilon=" + Conversions::toString(epsilon) + ")");
    }
}

void Adadelta::optimize(MemoryManager& memory_manager, Tensor& param, const Tensor& grad)
{
    if (param.size() != grad.size())
    {
        THROW_NONAME("Adadelta", "parameters and gradients must have the same size (" + Conversions::toString(param.size()) + " != " + Conversions::toString(grad.size()) + ")");
    }

    Tensor *gp, *gu;
    if (!memory_manager.tensorExists(Name("Adadelta") / param.getName() / "g"))
    {
        gp = memory_manager.createTensor(Name("Adadelta") / param.getName() / "g", 1, param.size(), 1, 1);
    }
    gp = &memory_manager.getTensor(Name("Adadelta") / param.getName() / "g");

    if (!memory_manager.tensorExists(Name("Adadelta") / param.getName() / "u"))
    {
        gu = memory_manager.createTensor(Name("Adadelta") / param.getName() / "u", 1, param.size(), 1, 1);
    }
    gu = &memory_manager.getTensor(Name("Adadelta") / param.getName() / "u");

    Tensor& g = *gp;
    Tensor& u = *gu;

    const auto n = param.size();

#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (size_t i = 0; i < n; i++)
    {
        // Accumulate gradients
        g[i] = this->m_rho * g[i] + (1.0_dt - this->m_rho) * grad[i] * grad[i];
        // Compute update
        const auto delta = -std::sqrt(u[i] + this->m_epsilon) / std::sqrt(g[i] + this->m_epsilon) * grad[i];
        // Accumulate updates
        u[i] = this->m_rho * u[i] + (1.0_dt - this->m_rho) * delta * delta;
        // Apply updates
        param[i] = param[i] + delta;
    }
}

void Adadelta::optimize(MemoryManagerFP16& memory_manager, TensorFP16& param, const TensorFP16& grad)
{
    if (param.size() != grad.size())
    {
        THROW_NONAME("Adadelta", "parameters and gradients must have the same size (" + Conversions::toString(param.size()) + " != " + Conversions::toString(grad.size()) + ")");
    }

    TensorFP16 *gp, *gu;
    if (!memory_manager.tensorExists(Name("Adadelta") / param.getName() / "g"))
    {
        gp = memory_manager.createTensor(Name("Adadelta") / param.getName() / "g", 1, param.size(), 1, 1);
    }
    gp = &memory_manager.getTensor(Name("Adadelta") / param.getName() / "g");

    if (!memory_manager.tensorExists(Name("Adadelta") / param.getName() / "u"))
    {
        gu = memory_manager.createTensor(Name("Adadelta") / param.getName() / "u", 1, param.size(), 1, 1);
    }
    gu = &memory_manager.getTensor(Name("Adadelta") / param.getName() / "u");

    TensorFP16& g = *gp;
    TensorFP16& u = *gu;

    const auto n = param.size();

#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (size_t i = 0; i < n; i++)
    {
        // Accumulate gradients
        const auto gTmp = this->m_rho * TODTYPE(g[i]) + (1.0_dt - this->m_rho) * TODTYPE(grad[i]) * TODTYPE(grad[i]);
        g[i] = TOHTYPE(gTmp);
        // Compute update
        const auto delta = -std::sqrt(TODTYPE(u[i]) + this->m_epsilon) / std::sqrt(gTmp + this->m_epsilon) * TODTYPE(grad[i]);
        // Accumulate updates
        u[i] = TOHTYPE(this->m_rho * TODTYPE(u[i]) + (1.0_dt - this->m_rho) * delta * delta);
        // Apply updates
        param[i] = TOHTYPE(TODTYPE(param[i]) + delta);
    }
}

std::ostream& Adadelta::as_ostream(std::ostream& out) const
{
    out << "Adadelta(rho=" << std::scientific << this->m_rho << ", epsilon=" << this->m_epsilon << ")";
    return out;
}

} // raul::optimizers