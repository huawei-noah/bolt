// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "Adam.h"
#include <iostream>
#include <stdexcept>

#include <training/opencl/GPUCommon.h>

using namespace raul;

namespace
{
constexpr raul::dtype alpha_lower_boundary = 0.0_dt;
constexpr raul::dtype beta_lower_boundary = 0.0_dt;
constexpr raul::dtype beta_upper_boundary = 1.0_dt;
}

namespace raul::optimizers
{
Adam::Adam(const dtype alpha, const dtype beta_1, const dtype beta_2, const dtype epsilon, bool use_simple_epsilon)
    : m_alpha(alpha)
    , m_beta_1(beta_1)
    , m_beta_2(beta_2)
    , m_epsilon(epsilon)
    , m_use_simple_epsilon(use_simple_epsilon)
{
    if (alpha <= alpha_lower_boundary)
    {
        THROW_NONAME("Adam", "reset alpha>" + Conversions::toString(alpha_lower_boundary) + " (current alpha=" + Conversions::toString(alpha) + ")");
    }

    if (beta_1 < beta_lower_boundary || beta_1 >= beta_upper_boundary)
    {
        THROW_NONAME("Adam",
                     "reset beta_1 from [" + Conversions::toString(beta_lower_boundary) + ", " + Conversions::toString(beta_upper_boundary) + ") (current beta_1=" + Conversions::toString(beta_1) +
                         ")");
    }

    if (beta_2 < beta_lower_boundary || beta_2 >= beta_upper_boundary)
    {
        THROW_NONAME("Adam",
                     "reset beta_2 from [" + Conversions::toString(beta_lower_boundary) + ", " + Conversions::toString(beta_upper_boundary) + ") (current beta_2=" + Conversions::toString(beta_2) +
                         ")");
    }
}

void Adam::optimize(MemoryManager& memory_manager, Tensor& param, const Tensor& grad)
{
    if (param.size() != grad.size())
    {
        THROW_NONAME("Adam", "parameters and gradients must have the same size (" + Conversions::toString(param.size()) + " != " + Conversions::toString(grad.size()) + ")");
    }

    Tensor *b1tp, *b2tp, *mp, *vp;
    if (!memory_manager.tensorExists(Name("Adam") / param.getName() / "beta_1_t"))
    {
        b1tp = memory_manager.createTensor(Name("Adam") / param.getName() / "beta_1_t", 1, 1, 1, 1, this->m_beta_1);
    }
    b1tp = &memory_manager.getTensor(Name("Adam") / param.getName() / "beta_1_t");

    if (!memory_manager.tensorExists(Name("Adam") / param.getName() / "beta_2_t"))
    {
        b2tp = memory_manager.createTensor(Name("Adam") / param.getName() / "beta_2_t", 1, 1, 1, 1, this->m_beta_2);
    }
    b2tp = &memory_manager.getTensor(Name("Adam") / param.getName() / "beta_2_t");

    if (!memory_manager.tensorExists(Name("Adam") / param.getName() / "m"))
    {
        mp = memory_manager.createTensor(Name("Adam") / param.getName() / "m", param.getShape());
    }
    mp = &memory_manager.getTensor(Name("Adam") / param.getName() / "m");

    if (!memory_manager.tensorExists(Name("Adam") / param.getName() / "v"))
    {
        vp = memory_manager.createTensor(Name("Adam") / param.getName() / "v", param.getShape());
    }
    vp = &memory_manager.getTensor(Name("Adam") / param.getName() / "v");

    Tensor& beta_1_t = *b1tp;
    Tensor& beta_2_t = *b2tp;
    Tensor& m = *mp;
    Tensor& v = *vp;

    const auto sqrt_beta_2_t_0 = std::sqrt(1.0_dt - beta_2_t[0]);
    const auto alpha_new = this->m_alpha * sqrt_beta_2_t_0 / (1.0_dt - beta_1_t[0]);
    const auto epsilon_new = m_use_simple_epsilon ? this->m_epsilon : this->m_epsilon * sqrt_beta_2_t_0;
    const auto n = param.size();

#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (size_t i = 0; i < n; i++)
    {
        // m_new = beta_1*m + (1-beta_1)*grad
        m[i] = this->m_beta_1 * m[i] + (1.0_dt - this->m_beta_1) * grad[i];
        // v_new = beta_2*v + (1-beta_2)*grad*grad
        v[i] = this->m_beta_2 * v[i] + (1.0_dt - this->m_beta_2) * grad[i] * grad[i];
        // param_new = param - alpha_new*m_new/(sqrt(v_new) + epsilon_new)
        param[i] = param[i] - alpha_new * m[i] / (std::sqrt(v[i]) + epsilon_new);
    }

    beta_1_t[0] *= this->m_beta_1;
    beta_2_t[0] *= this->m_beta_2;
}

void Adam::optimize(MemoryManagerFP16& memory_manager, TensorFP16& param, const TensorFP16& grad)
{
    if (param.size() != grad.size())
    {
        THROW_NONAME("Adam", "parameters and gradients must have the same size (" + Conversions::toString(param.size()) + " != " + Conversions::toString(grad.size()) + ")");
    }

    TensorFP16 *b1tp, *b2tp, *mp, *vp;
    if (!memory_manager.tensorExists(Name("Adam") / param.getName() / "beta_1_t"))
    {
        b1tp = memory_manager.createTensor(Name("Adam") / param.getName() / "beta_1_t", 1, 1, 1, 1, TOHTYPE(this->m_beta_1));
    }
    b1tp = &memory_manager.getTensor(Name("Adam") / param.getName() / "beta_1_t");

    if (!memory_manager.tensorExists(Name("Adam") / param.getName() / "beta_2_t"))
    {
        b2tp = memory_manager.createTensor(Name("Adam") / param.getName() / "beta_2_t", 1, 1, 1, 1, TOHTYPE(this->m_beta_2));
    }
    b2tp = &memory_manager.getTensor(Name("Adam") / param.getName() / "beta_2_t");

    if (!memory_manager.tensorExists(Name("Adam") / param.getName() / "m"))
    {
        mp = memory_manager.createTensor(Name("Adam") / param.getName() / "m", param.getShape());
    }
    mp = &memory_manager.getTensor(Name("Adam") / param.getName() / "m");

    if (!memory_manager.tensorExists(Name("Adam") / param.getName() / "v"))
    {
        vp = memory_manager.createTensor(Name("Adam") / param.getName() / "v", param.getShape());
    }
    vp = &memory_manager.getTensor(Name("Adam") / param.getName() / "v");

    TensorFP16& beta_1_t = *b1tp;
    TensorFP16& beta_2_t = *b2tp;
    TensorFP16& m = *mp;
    TensorFP16& v = *vp;

    const auto sqrt_beta_2_t_0 = std::sqrt(1.0_dt - beta_2_t[0]);
    const auto alpha_new = this->m_alpha * sqrt_beta_2_t_0 / (1.0_dt - beta_1_t[0]);
    const auto epsilon_new = m_use_simple_epsilon ? this->m_epsilon : this->m_epsilon * sqrt_beta_2_t_0;
    const auto n = param.size();

#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (size_t i = 0; i < n; i++)
    {
        // m_new = beta_1*m + (1-beta_1)*grad
        const auto mTmp = this->m_beta_1 * TODTYPE(m[i]) + (1.0_dt - this->m_beta_1) * TODTYPE(grad[i]);
        m[i] = TOHTYPE(mTmp);
        // v_new = beta_2*v + (1-beta_2)*grad*grad
        const auto vTmp = this->m_beta_2 * TODTYPE(v[i]) + (1.0_dt - this->m_beta_2) * TODTYPE(grad[i]) * TODTYPE(grad[i]);
        v[i] = TOHTYPE(vTmp);
        // param_new = param - alpha_new*m_new/(sqrt(v_new) + epsilon_new)
        param[i] = TOHTYPE(TODTYPE(param[i]) - alpha_new * mTmp / (std::sqrt(vTmp) + epsilon_new));
    }

    beta_1_t[0] = TOHTYPE(TODTYPE(beta_1_t[0]) * this->m_beta_1);
    beta_2_t[0] = TOHTYPE(TODTYPE(beta_2_t[0]) * this->m_beta_2);
}

void Adam::optimize(OpenCLKernelManager& kernel_manager, MemoryManagerGPU& memory_manager, TensorGPU& param, const TensorGPU& grad)
{
    if (param.getShape().total_size() != grad.getShape().total_size())
    {
        THROW_NONAME("Adam",
                     "parameters and gradients must have the same size (" + Conversions::toString(param.getShape().total_size()) + " != " + Conversions::toString(grad.getShape().total_size()) + ")");
    }

    TensorGPU *b1tp, *b2tp, *mp, *vp;
    if (!memory_manager.tensorExists(Name("Adam") / param.getName() / "beta_1_t"))
    {
        b1tp = memory_manager.createTensor(Name("Adam") / param.getName() / "beta_1_t", 1, 1, 1, 1, this->m_beta_1);
    }
    b1tp = &memory_manager.getTensor(Name("Adam") / param.getName() / "beta_1_t");

    if (!memory_manager.tensorExists(Name("Adam") / param.getName() / "beta_2_t"))
    {
        b2tp = memory_manager.createTensor(Name("Adam") / param.getName() / "beta_2_t", 1, 1, 1, 1, this->m_beta_2);
    }
    b2tp = &memory_manager.getTensor(Name("Adam") / param.getName() / "beta_2_t");

    if (!memory_manager.tensorExists(Name("Adam") / param.getName() / "m"))
    {
        mp = memory_manager.createTensor(Name("Adam") / param.getName() / "m", param.getShape(), 0.0_dt);
    }
    mp = &memory_manager.getTensor(Name("Adam") / param.getName() / "m");

    if (!memory_manager.tensorExists(Name("Adam") / param.getName() / "v"))
    {
        vp = memory_manager.createTensor(Name("Adam") / param.getName() / "v", param.getShape(), 0.0_dt);
    }
    vp = &memory_manager.getTensor(Name("Adam") / param.getName() / "v");

    TensorGPU& beta_1_t = *b1tp;
    TensorGPU& beta_2_t = *b2tp;
    TensorGPU& m = *mp;
    TensorGPU& v = *vp;

    gpu::adam(kernel_manager,
              "",
              param.getBatchSize() * param.getDepth() * param.getHeight() * param.getWidth(),
              m_alpha,
              m_beta_1,
              m_beta_2,
              m_epsilon,
              static_cast<size_t>(m_use_simple_epsilon),
              grad.getBuffer(),
              beta_1_t.getBuffer(),
              beta_2_t.getBuffer(),
              m.getBuffer(),
              v.getBuffer(),
              param.getBuffer());

    // Updata betas
    OPENBLAS_CONST dtype sa1 = this->m_beta_1;
    OPENBLAS_CONST dtype sa2 = this->m_beta_2;
    Common::axpby(&kernel_manager, "", 1u, sa1, beta_1_t.getBuffer(), 1u, 0.0_dt, beta_1_t.getBuffer(), 1u, 0, 0);
    Common::axpby(&kernel_manager, "", 1u, sa2, beta_2_t.getBuffer(), 1u, 0.0_dt, beta_2_t.getBuffer(), 1u, 0, 0);
}

std::ostream& Adam::as_ostream(std::ostream& out) const
{
    out << "Adam(alpha=" << std::scientific << this->m_alpha << ", beta_1=" << this->m_beta_1 << ", beta_2=" << this->m_beta_2 << ", epsilon=" << this->m_epsilon << ")";
    return out;
}

} // raul::optimizers