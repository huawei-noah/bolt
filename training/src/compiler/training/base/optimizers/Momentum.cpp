// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "Momentum.h"
#include "training/base/common/Conversions.h"
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

Momentum::Momentum(const dtype lr, const dtype momentum)
    : m_learning_rate(lr)
    , m_momentum(momentum)
{

    if (lr <= lr_lower_boundary)
    {
        THROW_NONAME("Momentum", "reset lr>" + Conversions::toString(lr_lower_boundary) + " (current lr=" + Conversions::toString(lr) + ")");
    }
    if (momentum < momentum_lower_boundary || momentum > momentum_upper_boundary)
    {
        THROW_NONAME("Momentum",
                     "reset momentum from [" + Conversions::toString(momentum_lower_boundary) + "," + Conversions::toString(momentum_upper_boundary) +
                         "] (current momentum=" + Conversions::toString(momentum) + ")");
    }
}

void Momentum::optimize(MemoryManager& memory_manager, Tensor& param, const Tensor& grad)
{
    if (param.size() != grad.size())
    {
        THROW_NONAME("Momentum", "parameters and gradients must have the same size (" + Conversions::toString(param.size()) + " != " + Conversions::toString(grad.size()) + ")");
    }

    Tensor* velP;
    if (!memory_manager.tensorExists(Name("Momentum") / param.getName() / "v"))
    {
        velP = memory_manager.createTensor(Name("Momentum") / param.getName() / "v", 1, param.size(), 1, 1, 0.0_dt);
    }
    velP = &memory_manager.getTensor(Name("Momentum") / param.getName() / "v");

    Tensor& velocity_vector = *velP;

    // velocity_new = momentum*velocity + lr*grad
    {
        OPENBLAS_CONST size_t n = velocity_vector.size();
        OPENBLAS_CONST dtype alpha = this->m_learning_rate;
        OPENBLAS_CONST dtype* x = &(grad[0]);
        OPENBLAS_CONST size_t incx = 1U;
        OPENBLAS_CONST dtype beta = this->m_momentum;
        dtype* y = &(velocity_vector[0]);
        OPENBLAS_CONST size_t incy = 1U;
        size_t xOffset = 0U;
        size_t yOffset = 0U;
        Common::axpby(n, alpha, x, incx, beta, y, incy, xOffset, yOffset);
    }
    // param_new = param - velocity_new = param - momentum*velocity - lr*grad
    {
        size_t n = velocity_vector.size();
        OPENBLAS_CONST dtype sa = -1.0_dt;
        OPENBLAS_CONST dtype* sx = &(velocity_vector[0]);
        size_t incx = 1U;
        dtype* sy = &(param[0]);
        size_t incy = 1U;
        size_t xOffset = 0U;
        size_t yOffset = 0U;
        Common::axpy(n, sa, sx, incx, sy, incy, xOffset, yOffset);
    }
}

void Momentum::optimize(MemoryManagerFP16& memory_manager, TensorFP16& param, const TensorFP16& grad)
{
    if (param.size() != grad.size())
    {
        THROW_NONAME("Momentum", "parameters and gradients must have the same size (" + Conversions::toString(param.size()) + " != " + Conversions::toString(grad.size()) + ")");
    }

    TensorFP16* velP;
    if (!memory_manager.tensorExists(Name("Momentum") / param.getName() / "v"))
    {
        velP = memory_manager.createTensor(Name("Momentum") / param.getName() / "v", 1, param.size(), 1, 1, 0.0_hf);
    }
    velP = &memory_manager.getTensor(Name("Momentum") / param.getName() / "v");

    TensorFP16& velocity_vector = *velP;

    // velocity_new = momentum*velocity + lr*grad
    {
        OPENBLAS_CONST size_t n = velocity_vector.size();
        OPENBLAS_CONST dtype alpha = this->m_learning_rate;
        OPENBLAS_CONST half* x = &(grad[0]);
        OPENBLAS_CONST size_t incx = 1U;
        OPENBLAS_CONST dtype beta = this->m_momentum;
        half* y = &(velocity_vector[0]);
        OPENBLAS_CONST size_t incy = 1U;
        size_t xOffset = 0U;
        size_t yOffset = 0U;
        Common::axpby(n, alpha, x, incx, beta, y, incy, xOffset, yOffset);
    }
    // param_new = param - velocity_new = param - momentum*velocity - lr*grad
    {
        size_t n = velocity_vector.size();
        OPENBLAS_CONST dtype sa = -1.0_dt;
        OPENBLAS_CONST half* sx = &(velocity_vector[0]);
        size_t incx = 1U;
        half* sy = &(param[0]);
        size_t incy = 1U;
        size_t xOffset = 0U;
        size_t yOffset = 0U;
        Common::axpy(n, sa, sx, incx, sy, incy, xOffset, yOffset);
    }
}

std::ostream& Momentum::as_ostream(std::ostream& out) const
{
    out << "Momentum(lr=" << std::scientific << this->m_learning_rate << ", momentum=" << this->m_momentum << ")";
    return out;
}

} // raul::optimizers