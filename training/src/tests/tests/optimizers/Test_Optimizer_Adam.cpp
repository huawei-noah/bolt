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
#include <training/base/optimizers/Adam.h>

namespace UT
{

TEST(TestOptimizerAdam, StreamUnit)
{
    PROFILE_TEST
    raul::MemoryManager memory_manager;
    std::ostringstream stream;
    auto alpha = 0.01_dt;
    {
        raul::optimizers::Adam optimizer{ alpha };
        stream << optimizer;
        ASSERT_STREQ(stream.str().c_str(), "Adam(alpha=1.000000e-02, beta_1=9.000000e-01, beta_2=9.990000e-01, epsilon=1.000000e-08)");
    }
}

TEST(TestOptimizerAdam, StepUnit)
{
    PROFILE_TEST
    raul::MemoryManager memory_manager;
    auto alpha = 0.8_dt;
    auto beta_1 = 0.5_dt;
    auto beta_2 = 0.75_dt;
    auto epsilon = 0.6_dt;
    auto amount_of_element = 10U;
    {
        raul::optimizers::Adam optimizer{ alpha, beta_1, beta_2, epsilon };
        auto& params = *memory_manager.createTensor("params", 1, amount_of_element, 1, 1, 1.0_dt);
        auto& gradients = *memory_manager.createTensor("gradients", 1, amount_of_element, 1, 1, 1.0_dt);
        optimizer(memory_manager, params, gradients);

        EXPECT_EQ(params.size(), amount_of_element);
        EXPECT_EQ(gradients.size(), amount_of_element);

        EXPECT_FLOAT_EQ(memory_manager["Adam::params::beta_1_t"][0], beta_1 * beta_1);
        EXPECT_FLOAT_EQ(memory_manager["Adam::params::beta_2_t"][0], beta_2 * beta_2);

        const auto m_t = (1.0_dt - beta_1);
        const auto v_t = (1.0_dt - beta_2);

        for (raul::dtype m_tensor_t_el : memory_manager["Adam::params::m"])
            EXPECT_FLOAT_EQ(m_tensor_t_el, m_t);

        for (raul::dtype v_tensor_t_el : memory_manager["Adam::params::v"])
            EXPECT_FLOAT_EQ(v_tensor_t_el, v_t);

        const auto res = 1.0_dt - alpha / (1.0_dt + epsilon);

        for (raul::dtype param_tensor_el : params)
            EXPECT_FLOAT_EQ(param_tensor_el, res);
    }
}

TEST(TestOptimizerAdam, DoubleStepUnit)
{
    PROFILE_TEST
    raul::MemoryManager memory_manager;
    auto alpha = 0.8_dt;
    auto beta_1 = 0.5_dt;
    auto beta_2 = 0.75_dt;
    auto epsilon = 0.6_dt;
    auto amount_of_element = 10U;
    {
        raul::optimizers::Adam optimizer{ alpha, beta_1, beta_2, epsilon };
        auto& params = *memory_manager.createTensor("params", 1, amount_of_element, 1, 1, 1.0_dt);
        auto& gradients = *memory_manager.createTensor("gradients", 1, amount_of_element, 1, 1, 1.0_dt);
        optimizer(memory_manager, params, gradients);
        std::fill(gradients.begin(), gradients.end(), 1_dt);
        optimizer(memory_manager, params, gradients);

        EXPECT_EQ(params.size(), amount_of_element);
        EXPECT_EQ(gradients.size(), amount_of_element);

        EXPECT_FLOAT_EQ(memory_manager["Adam::params::beta_1_t"][0], beta_1 * beta_1 * beta_1);
        EXPECT_FLOAT_EQ(memory_manager["Adam::params::beta_2_t"][0], beta_2 * beta_2 * beta_2);

        // beta*old + (1-beta)*1 = beta*(1-beta)*1 + (1-beta)*1 = (1+beta)*(1-beta) = (1-beta^2)
        const auto m_t = (1.0_dt - beta_1 * beta_1);
        const auto v_t = (1.0_dt - beta_2 * beta_2);

        for (raul::dtype m_tensor_t_el : memory_manager.getTensor("Adam::params::m"))
            EXPECT_FLOAT_EQ(m_tensor_t_el, m_t);

        for (raul::dtype v_tensor_t_el : memory_manager.getTensor("Adam::params::v"))
            EXPECT_FLOAT_EQ(v_tensor_t_el, v_t);

        const auto res = 1.0_dt - 2 * alpha / (1.0_dt + epsilon);

        for (raul::dtype param_tensor_el : params)
            EXPECT_FLOAT_EQ(param_tensor_el, res);
    }
}

TEST(TestOptimizerAdamQuantized, MappersUnit)
{
    PROFILE_TEST

    constexpr dtype EPSILON = 0.01_dt;

    {
        auto lin = raul::optimizers::AdamQuantized::linspace(0.0f, 1.0f, 0);
        EXPECT_EQ(lin.size(), 0u);
    }

    {
        auto lin = raul::optimizers::AdamQuantized::linspace(-1.0f, 1.0f, 1);
        EXPECT_EQ(lin.size(), 1u);

        EXPECT_NEAR(lin[0], -1, EPSILON);
    }

    {
        auto lin = raul::optimizers::AdamQuantized::linspace(-1.0f, 1.0f, 2);
        EXPECT_EQ(lin.size(), 2u);

        EXPECT_NEAR(lin[0], -1, EPSILON);
        EXPECT_NEAR(lin[1], 1, EPSILON);
    }

    {
        auto lin = raul::optimizers::AdamQuantized::linspace(0.0f, 1.0f, 11);
        EXPECT_EQ(lin.size(), 11u);

        dtype elem = 0.0_dt;
        for (auto val : lin)
        {
            EXPECT_NEAR(val, elem, EPSILON);
            elem += 0.1_dt;
        }
    }

    {
        auto lin = raul::optimizers::AdamQuantized::linspace(-1.0f, 1.0f, 21);
        EXPECT_EQ(lin.size(), 21u);

        dtype elem = -1.0_dt;
        for (auto val : lin)
        {
            EXPECT_NEAR(val, elem, EPSILON);
            elem += 0.1_dt;
        }
    }

    {
        auto map = raul::optimizers::AdamQuantized::createNormalQuantileMap(true);
        EXPECT_EQ(map.size(), 256u);
        EXPECT_NEAR(map[0], -1.0_dt, EPSILON);
        EXPECT_NEAR(map[127], 0.0_dt, EPSILON);
        EXPECT_NEAR(map[255], 1.0_dt, EPSILON);
    }

    {
        auto map = raul::optimizers::AdamQuantized::createNormalQuantileMap(false);
        EXPECT_EQ(map.size(), 256u);
        EXPECT_NEAR(map[0], 0.0_dt, EPSILON);
        EXPECT_NEAR(map[255], 1.0_dt, EPSILON);
    }

    {
        auto map = raul::optimizers::AdamQuantized::createDynamicMap(true);
        EXPECT_EQ(map.size(), 256u);
        EXPECT_NEAR(map[0], -0.993_dt, EPSILON);
        EXPECT_NEAR(map[127], 0.0_dt, EPSILON);
        EXPECT_NEAR(map[255], 1.0_dt, EPSILON);
    }

    {
        auto map = raul::optimizers::AdamQuantized::createDynamicMap(false);
        EXPECT_EQ(map.size(), 256u);
        EXPECT_NEAR(map[0], 0.0_dt, EPSILON);
        EXPECT_NEAR(map[255], 1.0_dt, EPSILON);
    }
}

} // UT namespace
