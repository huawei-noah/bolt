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
#include <training/base/optimizers/Adagrad.h>

namespace UT
{

TEST(TestOptimizerAdagrad, StreamUnit)
{
    PROFILE_TEST
    raul::MemoryManager memory_manager;
    std::ostringstream stream;
    const auto alpha = 1e-2_dt;
    const auto epsilon = 1e-6_dt;
    {
        raul::optimizers::Adagrad optimizer{ alpha, epsilon };
        stream << optimizer;
        ASSERT_STREQ(stream.str().c_str(), "Adagrad(alpha=1.000000e-02, epsilon=1.000000e-06)");
    }
}

TEST(TestOptimizerAdagrad, StepUnit)
{
    PROFILE_TEST
    raul::MemoryManager memory_manager;
    const auto alpha = 0.8_dt;
    const auto grad_val = 2.0_dt;
    auto amount_of_element = 10U;
    {
        raul::optimizers::Adagrad optimizer{ alpha };
        auto& params = *memory_manager.createTensor("params", 1, amount_of_element, 1, 1, 1.0_dt);
        auto& gradients = *memory_manager.createTensor("gradients", 1, amount_of_element, 1, 1, grad_val);
        optimizer(memory_manager, params, gradients);

        EXPECT_EQ(params.size(), amount_of_element);
        EXPECT_EQ(gradients.size(), amount_of_element);

        EXPECT_FLOAT_EQ(memory_manager["Adagrad::params::g"][0], grad_val * grad_val);

        const auto res = 1.0_dt - alpha;

        for (raul::dtype param_tensor_el : params)
            EXPECT_FLOAT_EQ(param_tensor_el, res);
    }
}

TEST(TestOptimizerAdagrad, DoubleStepUnit)
{
    PROFILE_TEST
    raul::MemoryManager memory_manager;
    auto alpha = 0.8_dt;
    const auto grad_val = 2.0_dt;
    auto amount_of_element = 10U;
    {
        raul::optimizers::Adagrad optimizer{ alpha };
        auto& params = *memory_manager.createTensor("params", 1, amount_of_element, 1, 1, 1.0_dt);
        auto& gradients = *memory_manager.createTensor("gradients", 1, amount_of_element, 1, 1, grad_val);
        optimizer(memory_manager, params, gradients);
        std::fill(gradients.begin(), gradients.end(), grad_val);
        optimizer(memory_manager, params, gradients);

        EXPECT_EQ(params.size(), amount_of_element);
        EXPECT_EQ(gradients.size(), amount_of_element);
        EXPECT_FLOAT_EQ(memory_manager["Adagrad::params::g"][0], 2.0_dt * grad_val * grad_val);

        const auto res = 1.0_dt - alpha * (1.0_dt + 1.0_dt / std::sqrt(2.0_dt));

        for (raul::dtype param_tensor_el : params)
            EXPECT_FLOAT_EQ(param_tensor_el, res);
    }
}

#ifdef ANDROID
TEST(TestOptimizerAdagrad, DoubleStepFP16Unit)
{
    PROFILE_TEST
    raul::MemoryManagerFP16 memory_manager;
    auto alpha = 0.8_dt;
    const auto grad_val = 2.0_dt;
    const auto eps = 1.0e-4_dt;
    auto amount_of_element = 10U;
    {
        raul::optimizers::Adagrad optimizer{ alpha };
        auto& params = *memory_manager.createTensor("params", 1, amount_of_element, 1, 1, 1.0_dt);
        auto& gradients = *memory_manager.createTensor("gradients", 1, amount_of_element, 1, 1, grad_val);
        optimizer(memory_manager, params, gradients);
        std::fill(gradients.begin(), gradients.end(), grad_val);
        optimizer(memory_manager, params, gradients);

        EXPECT_EQ(params.size(), amount_of_element);
        EXPECT_EQ(gradients.size(), amount_of_element);
        EXPECT_FLOAT_EQ(TODTYPE(memory_manager["Adagrad::params::g"][0]), 2.0_dt * grad_val * grad_val);

        const auto res = 1.0_dt - alpha * (1.0_dt + 1.0_dt / std::sqrt(2.0_dt));

        for (raul::half param_tensor_el : params)
        {
            EXPECT_NEAR(TODTYPE(param_tensor_el), res, eps);
        }
    }
}
#endif // ANDROID

} // UT namespace