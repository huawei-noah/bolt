// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <tests/tools/TestTools.h>
#include <iostream>

#include <training/base/layers/activations/SigmoidActivation.h>
#include <training/base/optimizers/Adadelta.h>

namespace UT
{

TEST(TestOptimizerAdadelta, StreamUnit)
{
    PROFILE_TEST
    raul::MemoryManager memory_manager;
    std::ostringstream stream;
    const auto rho = 1e-2_dt;
    const auto epsilon = 1e-6_dt;
    {
        raul::optimizers::Adadelta optimizer{ rho, epsilon };
        stream << optimizer;
        ASSERT_STREQ(stream.str().c_str(), "Adadelta(rho=1.000000e-02, epsilon=1.000000e-06)");
    }
}

TEST(TestOptimizerAdadelta, StepUnit)
{
    PROFILE_TEST
    raul::MemoryManager memory_manager;
    const auto rho = 0.5_dt;
    const auto grad_val = 1.0_dt;
    const auto epsilon = 1e-10_dt;
    auto amount_of_element = 10U;
    {
        raul::optimizers::Adadelta optimizer{ rho, epsilon };
        auto& params = *memory_manager.createTensor("params", 1, amount_of_element, 1, 1, 1.0_dt);
        auto& gradients = *memory_manager.createTensor("gradients", 1, amount_of_element, 1, 1, grad_val);
        optimizer(memory_manager, params, gradients);

        EXPECT_EQ(params.size(), amount_of_element);
        EXPECT_EQ(gradients.size(), amount_of_element);

        const auto g_el_golden = 1.0_dt - rho;
        for (raul::dtype g_el : memory_manager["Adadelta::params::g"])
            EXPECT_FLOAT_EQ(g_el, g_el_golden);

        const auto delta = -std::sqrt(epsilon) / std::sqrt(g_el_golden + epsilon);
        const auto u_el_golden = (1.0_dt - rho) * delta * delta;
        for (raul::dtype u_el : memory_manager["Adadelta::params::u"])
            EXPECT_FLOAT_EQ(u_el, u_el_golden);

        const auto res = 1.0_dt + delta;

        for (raul::dtype param_tensor_el : params)
            EXPECT_FLOAT_EQ(param_tensor_el, res);
    }
}

TEST(TestOptimizerAdadelta, DoubleStepUnit)
{
    PROFILE_TEST
    raul::MemoryManager memory_manager;
    const auto rho = 0.5_dt;
    const auto grad_val = 1.0_dt;
    const auto epsilon = 1e-10_dt;
    auto amount_of_element = 10U;
    {
        raul::optimizers::Adadelta optimizer{ rho, epsilon };
        auto& params = *memory_manager.createTensor("params", 1, amount_of_element, 1, 1, 1.0_dt);
        auto& gradients = *memory_manager.createTensor("gradients", 1, amount_of_element, 1, 1, grad_val);
        optimizer(memory_manager, params, gradients);
        std::fill(gradients.begin(), gradients.end(), grad_val);
        optimizer(memory_manager, params, gradients);

        EXPECT_EQ(params.size(), amount_of_element);
        EXPECT_EQ(gradients.size(), amount_of_element);

        const auto g_el_golden = 1.0_dt - rho * rho;
        for (raul::dtype g_el : memory_manager["Adadelta::params::g"])
            EXPECT_FLOAT_EQ(g_el, g_el_golden);

        const auto delta_prev = -std::sqrt(epsilon) / std::sqrt(1.0_dt - rho + epsilon);
        const auto u_el_golden_prev = (1.0_dt - rho) * delta_prev * delta_prev;

        const auto delta = -std::sqrt(u_el_golden_prev + epsilon) / std::sqrt(g_el_golden + epsilon);
        const auto u_el_golden = rho * u_el_golden_prev + (1.0_dt - rho) * delta * delta;
        for (raul::dtype u_el : memory_manager["Adadelta::params::u"])
            EXPECT_FLOAT_EQ(u_el, u_el_golden);

        const auto res = 1.0_dt + delta_prev + delta;

        for (raul::dtype param_tensor_el : params)
            EXPECT_FLOAT_EQ(param_tensor_el, res);
    }
}

#ifdef ANDROID
TEST(TestOptimizerAdadelta, DoubleStepFP16Unit)
{
    PROFILE_TEST
    raul::MemoryManagerFP16 memory_manager;
    const auto rho = 0.5_dt;
    const auto epsilon = 1e-10_dt;
    auto amount_of_element = 10U;
    const auto eps = 1.0e-4_dt;
    {
        raul::optimizers::Adadelta optimizer{ rho, epsilon };
        auto& params = *memory_manager.createTensor("params", 1, amount_of_element, 1, 1, 1.0_hf);
        auto& gradients = *memory_manager.createTensor("gradients", 1, amount_of_element, 1, 1, 1.0_hf);
        optimizer(memory_manager, params, gradients);
        std::fill(gradients.begin(), gradients.end(), 1.0_hf);
        optimizer(memory_manager, params, gradients);

        EXPECT_EQ(params.size(), amount_of_element);
        EXPECT_EQ(gradients.size(), amount_of_element);

        const auto g_el_golden = 1.0_dt - rho * rho;
        for (raul::half g_el : memory_manager["Adadelta::params::g"])
        {
            EXPECT_FLOAT_EQ(TODTYPE(g_el), g_el_golden);
        }

        const auto delta_prev = -std::sqrt(epsilon) / std::sqrt(1.0_dt - rho + epsilon);
        const auto u_el_golden_prev = (1.0_dt - rho) * delta_prev * delta_prev;

        const auto delta = -std::sqrt(u_el_golden_prev + epsilon) / std::sqrt(g_el_golden + epsilon);
        const auto u_el_golden = rho * u_el_golden_prev + (1.0_dt - rho) * delta * delta;
        for (raul::half u_el : memory_manager["Adadelta::params::u"])
        {
            EXPECT_NEAR(TODTYPE(u_el), u_el_golden, eps);
        }

        const auto res = 1.0_dt + delta_prev + delta;

        for (raul::half param_tensor_el : params)
        {
            EXPECT_NEAR(TODTYPE(param_tensor_el), res, eps);
        }
    }
}
#endif // ANDROID

} // UT namespace