// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "SGD.h"
#include <iostream>
#include <stdexcept>

namespace raul::optimizers
{

SGD::SGD(const dtype lr)
    : m_learning_rate(lr)
{
    if (lr <= .0)
    {
        THROW_NONAME("SGD", "reset lr>0 (current lr=" + Conversions::toString(lr) + ")");
    }
}

void SGD::optimize(MemoryManager&, Tensor& param, const Tensor& grad)
{
    if (param.size() != grad.size())
    {
        THROW_NONAME("SGD", "parameters and gradients must have the same size (" + Conversions::toString(param.size()) + " != " + Conversions::toString(grad.size()) + ")");
    }
    size_t n = param.size();
    OPENBLAS_CONST dtype sa = TODTYPE(-1.0) * this->m_learning_rate;
    OPENBLAS_CONST dtype* sx = &(grad[0]);
    size_t incx = 1U;
    dtype* sy = &(param[0]);
    size_t incy = 1U;
    size_t xOffset = 0U;
    size_t yOffset = 0U;
    Common::axpy(nullptr, "", n, sa, sx, incx, sy, incy, xOffset, yOffset);
}

void SGD::optimize(MemoryManagerGPU& manager, TensorGPU& param, const TensorGPU& grad)
{
    if (param.size() != grad.size())
    {
        THROW_NONAME("SGD", "parameters and gradients must have the same size (" + Conversions::toString(param.size()) + " != " + Conversions::toString(grad.size()) + ")");
    }

    // d.polubotko: implement using GPU

    size_t n = param.getShape().total_size(); // due to alignment don`t use size()
    OPENBLAS_CONST dtype sa = TODTYPE(-1.0) * this->m_learning_rate;

    size_t incx = 1U;

    auto paramGPU = manager[param.getName()];
    Tensor paramCPU(paramGPU);

    const Tensor gradCPU(manager[grad.getName()]);

    OPENBLAS_CONST dtype* sx = &(gradCPU[0]);

    dtype* sy = &(paramCPU[0]);
    size_t incy = 1U;
    size_t xOffset = 0U;
    size_t yOffset = 0U;
    Common::axpy(nullptr, "", n, sa, sx, incx, sy, incy, xOffset, yOffset);

    paramGPU = paramCPU;
}

void SGD::optimize(MemoryManagerFP16&, TensorFP16& param, const TensorFP16& grad)
{
    if (param.size() != grad.size())
    {
        THROW_NONAME("SGD", "parameters and gradients must have the same size (" + Conversions::toString(param.size()) + " != " + Conversions::toString(grad.size()) + ")");
    }
    size_t n = param.size();
    OPENBLAS_CONST dtype sa = TODTYPE(-1.0) * this->m_learning_rate;
    OPENBLAS_CONST half* sx = &(grad[0]);
    size_t incx = 1U;
    half* sy = &(param[0]);
    size_t incy = 1U;
    size_t xOffset = 0U;
    size_t yOffset = 0U;
    Common::axpy(nullptr, "", n, sa, sx, incx, sy, incy, xOffset, yOffset);
}

void SGD::optimize(OpenCLKernelManager& kernel_manager, MemoryManagerGPU&, TensorGPU& param, const TensorGPU& grad)
{
    if (param.getShape().total_size() != grad.getShape().total_size())
    {
        THROW_NONAME("Adam",
                     "parameters and gradients must have the same size (" + Conversions::toString(param.getShape().total_size()) + " != " + Conversions::toString(grad.getShape().total_size()) + ")");
    }

    OPENBLAS_CONST dtype sa = TODTYPE(-1.0) * this->m_learning_rate;
    size_t incx = 1U;
    size_t incy = 1U;
    size_t xOffset = 0U;
    size_t yOffset = 0U;
    Common::axpy(&kernel_manager, "SGD[::optimize]", param.getShape().total_size(), sa, grad.getBuffer(), incx, param.getBuffer(), incy, xOffset, yOffset);
}

std::ostream& SGD::as_ostream(std::ostream& out) const
{
    out << "SGD(lr=" << std::scientific << this->m_learning_rate << ")";
    return out;
}

} // raul::optimizers