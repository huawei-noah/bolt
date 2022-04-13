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
#include <training/base/optimizers/Adamax.h>

namespace UT
{

TEST(TestOptimizerAdamax, StreamUnit)
{
    PROFILE_TEST
    raul::MemoryManager memory_manager;
    std::ostringstream stream;
    auto alpha = 0.002_dt;
    {
        raul::optimizers::Adamax optimizer{ alpha };
        stream << optimizer;
        ASSERT_STREQ(stream.str().c_str(), "Adamax(alpha=2.000000e-03, beta_1=9.000000e-01, beta_2=9.990000e-01)");
    }
}

TEST(TestOptimizerAdamax, StepUnit)
{
    PROFILE_TEST
    raul::MemoryManager memory_manager;
    auto alpha = 0.8_dt;
    auto beta_1 = 0.5_dt;
    auto beta_2 = 0.75_dt;
    auto amount_of_element = 10U;
    {
        raul::optimizers::Adamax optimizer{ alpha, beta_1, beta_2 };
        auto& params = *memory_manager.createTensor("params", 1, amount_of_element, 1, 1, 1.0_dt);
        auto& gradients = *memory_manager.createTensor("gradients", 1, amount_of_element, 1, 1, 1.0_dt);
        optimizer(memory_manager, params, gradients);

        EXPECT_EQ(params.size(), amount_of_element);
        EXPECT_EQ(gradients.size(), amount_of_element);

        EXPECT_FLOAT_EQ(memory_manager["Adamax::params::beta_1_t"][0], beta_1 * beta_1);

        const auto m_t = (1.0_dt - beta_1);
        const auto u_t = 1.0_dt;

        for (raul::dtype m_tensor_t_el : memory_manager["Adamax::params::m"])
            EXPECT_FLOAT_EQ(m_tensor_t_el, m_t);

        for (raul::dtype v_tensor_t_el : memory_manager["Adamax::params::u"])
            EXPECT_FLOAT_EQ(v_tensor_t_el, u_t);

        const auto res = 1.0_dt - alpha;

        for (raul::dtype param_tensor_el : params)
            EXPECT_FLOAT_EQ(param_tensor_el, res);
    }
}

TEST(TestOptimizerAdamax, DoubleStepUnit)
{
    PROFILE_TEST
    raul::MemoryManager memory_manager;
    auto alpha = 0.8_dt;
    auto beta_1 = 0.5_dt;
    auto beta_2 = 0.75_dt;
    auto amount_of_element = 10U;
    {
        raul::optimizers::Adamax optimizer{ alpha, beta_1, beta_2 };
        auto& params = *memory_manager.createTensor("params", 1, amount_of_element, 1, 1, 1.0_dt);
        auto& gradients = *memory_manager.createTensor("gradients", 1, amount_of_element, 1, 1, 1.0_dt);
        optimizer(memory_manager, params, gradients);
        std::fill(gradients.begin(), gradients.end(), 1_dt);
        optimizer(memory_manager, params, gradients);

        EXPECT_EQ(params.size(), amount_of_element);
        EXPECT_EQ(gradients.size(), amount_of_element);

        EXPECT_FLOAT_EQ(memory_manager["Adamax::params::beta_1_t"][0], beta_1 * beta_1 * beta_1);

        // beta*old + (1-beta)*1 = beta*(1-beta)*1 + (1-beta)*1 = (1+beta)*(1-beta) = (1-beta^2)
        const auto m_t = (1.0_dt - beta_1 * beta_1);
        // Because beta_2 < 1 we always choose |grad| here
        const auto u_t = 1.0_dt;

        for (raul::dtype m_tensor_t_el : memory_manager["Adamax::params::m"])
            EXPECT_FLOAT_EQ(m_tensor_t_el, m_t);

        for (raul::dtype v_tensor_t_el : memory_manager["Adamax::params::u"])
            EXPECT_FLOAT_EQ(v_tensor_t_el, u_t);

        const auto res = 1.0_dt - 2 * alpha / (1.0_dt);

        for (raul::dtype param_tensor_el : params)
            EXPECT_FLOAT_EQ(param_tensor_el, res);
    }
}

#ifdef ANDROID
TEST(TestOptimizerAdamax, DoubleStepFP16Unit)
{
    PROFILE_TEST
    raul::MemoryManagerFP16 memory_manager;
    auto alpha = 0.8_dt;
    auto beta_1 = 0.5_dt;
    auto beta_2 = 0.75_dt;
    auto amount_of_element = 10U;
    const auto eps = 1.0e-3_dt;
    {
        raul::optimizers::Adamax optimizer{ alpha, beta_1, beta_2 };
        auto& params = *memory_manager.createTensor("params", 1, amount_of_element, 1, 1, 1.0_hf);
        auto& gradients = *memory_manager.createTensor("gradients", 1, amount_of_element, 1, 1, 1.0_hf);
        optimizer(memory_manager, params, gradients);
        std::fill(gradients.begin(), gradients.end(), 1.0_hf);
        optimizer(memory_manager, params, gradients);

        EXPECT_EQ(params.size(), amount_of_element);
        EXPECT_EQ(gradients.size(), amount_of_element);

        EXPECT_FLOAT_EQ(TODTYPE(memory_manager["Adamax::params::beta_1_t"][0]), beta_1 * beta_1 * beta_1);

        // beta*old + (1-beta)*1 = beta*(1-beta)*1 + (1-beta)*1 = (1+beta)*(1-beta) = (1-beta^2)
        const auto m_t = (1.0_dt - beta_1 * beta_1);
        // Because beta_2 < 1 we always choose |grad| here
        const auto u_t = 1.0_dt;

        for (raul::half m_tensor_t_el : memory_manager["Adamax::params::m"])
        {
            EXPECT_FLOAT_EQ(TODTYPE(m_tensor_t_el), m_t);
        }

        for (raul::half v_tensor_t_el : memory_manager["Adamax::params::u"])
        {
            EXPECT_FLOAT_EQ(TODTYPE(v_tensor_t_el), u_t);
        }

        const auto res = 1.0_dt - 2 * alpha / (1.0_dt);

        for (raul::half param_tensor_el : params)
        {
            EXPECT_NEAR(TODTYPE(param_tensor_el), res, eps);
        }
    }
}
#endif // ANDROID

} // UT namespace