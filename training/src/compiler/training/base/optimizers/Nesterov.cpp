// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "Nesterov.h"
#include <iostream>
#include <stdexcept>

namespace
{
constexpr raul::dtype lr_lower_boundary = 0.0_dt;
constexpr raul::dtype momentum_lower_boundary = 0.0_dt;
constexpr raul::dtype momentum_upper_boundary = 1.0_dt;
}

namespace raul::optimizers
{

Nesterov::Nesterov(const dtype lr, const dtype momentum)
    : m_learning_rate(lr)
    , m_momentum(momentum)
{
    if (lr <= lr_lower_boundary)
    {
        THROW_NONAME("Nesterov", "reset lr>" + Conversions::toString(lr_lower_boundary) + " (current lr=" + Conversions::toString(lr) + ")");
    }
    if (momentum < momentum_lower_boundary || momentum > momentum_upper_boundary)
    {
        THROW_NONAME("Nesterov",
                     "reset momentum from [" + Conversions::toString(momentum_lower_boundary) + "," + Conversions::toString(momentum_upper_boundary) +
                         "] (current momentum=" + Conversions::toString(momentum) + ")");
    }
}

void Nesterov::optimize(MemoryManager& memory_manager, Tensor& param, const Tensor& grad)
{
    if (param.size() != grad.size())
    {
        THROW_NONAME("Nesterov", "parameters and gradients must have the same size (" + Conversions::toString(param.size()) + " != " + Conversions::toString(grad.size()) + ")");
    }

    Tensor* velP;
    if (!memory_manager.tensorExists(Name("Nesterov") / param.getName() / "v"))
    {
        velP = memory_manager.createTensor(Name("Nesterov") / param.getName() / "v", 1, param.size(), 1, 1, 0.0_dt);
    }
    velP = &memory_manager.getTensor(Name("Nesterov") / param.getName() / "v");

    Tensor& velocity_vector = *velP;

    const auto n = param.size();

#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (size_t i = 0; i < n; i++)
    {
        const auto prev_v = velocity_vector[i];
        // v_new = momentum*v - lr*grad
        velocity_vector[i] = this->m_momentum * velocity_vector[i] - this->m_learning_rate * grad[i];
        // param_new = param - momentum*v + (1 + momentum)*v_new
        param[i] = param[i] - this->m_momentum * prev_v + (1.0_dt + this->m_momentum) * velocity_vector[i];
    }
}

void Nesterov::optimize(MemoryManagerFP16& memory_manager, TensorFP16& param, const TensorFP16& grad)
{
    if (param.size() != grad.size())
    {
        THROW_NONAME("Nesterov", "parameters and gradients must have the same size (" + Conversions::toString(param.size()) + " != " + Conversions::toString(grad.size()) + ")");
    }

    TensorFP16* velP;
    if (!memory_manager.tensorExists(Name("Nesterov") / param.getName() / "v"))
    {
        velP = memory_manager.createTensor(Name("Nesterov") / param.getName() / "v", 1, param.size(), 1, 1, 0.0_hf);
    }
    velP = &memory_manager.getTensor(Name("Nesterov") / param.getName() / "v");

    TensorFP16& velocity_vector = *velP;

    const auto n = param.size();

#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (size_t i = 0; i < n; i++)
    {
        const auto prev_v = velocity_vector[i];
        // v_new = momentum*v - lr*grad
        velocity_vector[i] = TOHTYPE(this->m_momentum) * velocity_vector[i] - TOHTYPE(this->m_learning_rate) * grad[i];
        // param_new = param - momentum*v + (1 + momentum)*v_new
        param[i] = param[i] - TOHTYPE(this->m_momentum) * prev_v + TOHTYPE(1.0_dt + this->m_momentum) * velocity_vector[i];
    }
}

std::ostream& Nesterov::as_ostream(std::ostream& out) const
{
    out << "Nesterov(lr=" << std::scientific << this->m_learning_rate << ", momentum=" << this->m_momentum << ")";
    return out;
}

} // raul::optimizers