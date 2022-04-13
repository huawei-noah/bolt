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
#include <training/base/optimizers/Nesterov.h>

namespace UT
{

TEST(TestOptimizerNesterov, StreamUnit)
{
    PROFILE_TEST
    std::ostringstream stream;
    constexpr auto learning_rate = 0.01_dt;
    constexpr auto momentum = 1.0_dt;
    {
        raul::optimizers::Nesterov optimizer{ learning_rate, momentum };
        stream << optimizer;
        ASSERT_STREQ(stream.str().c_str(), "Nesterov(lr=1.000000e-02, momentum=1.000000e+00)");
    }
}

TEST(TestOptimizerNesterov, StepUnit)
{
    PROFILE_TEST
    raul::MemoryManager memory_manager;
    const auto learning_rate = 0.1_dt;
    const auto momentum = 1.0_dt;
    const auto amount_of_element = 10U;
    {
        raul::optimizers::Nesterov optimizer{ learning_rate, momentum };
        auto& params = *memory_manager.createTensor("params", 1, amount_of_element, 1, 1, 1.0_dt);
        auto& gradients = *memory_manager.createTensor("gradients", 1, amount_of_element, 1, 1, 1.0_dt);
        optimizer(memory_manager, params, gradients);

        EXPECT_EQ(params.size(), amount_of_element);
        EXPECT_EQ(gradients.size(), amount_of_element);

        for (raul::dtype val_el : memory_manager["Nesterov::params::v"])
            EXPECT_FLOAT_EQ(val_el, -learning_rate);

        for (raul::dtype d : params)
            EXPECT_EQ(d, 1.0_dt - 2.0_dt * learning_rate);
    }
}

TEST(TestOptimizerNesterov, DoubleStepUnit)
{
    PROFILE_TEST
    raul::MemoryManager memory_manager;
    const auto learning_rate = 0.1_dt;
    const auto momentum = 1.0_dt;
    const auto amount_of_element = 10U;
    {
        raul::optimizers::Nesterov optimizer{ learning_rate, momentum };
        auto& params = *memory_manager.createTensor("params", 1, amount_of_element, 1, 1, 1.0_dt);
        auto& gradients = *memory_manager.createTensor("gradients", 1, amount_of_element, 1, 1, 1.0_dt);
        optimizer(memory_manager, params, gradients);
        std::fill(gradients.begin(), gradients.end(), 1_dt);
        optimizer(memory_manager, params, gradients);

        EXPECT_EQ(params.size(), amount_of_element);
        EXPECT_EQ(gradients.size(), amount_of_element);

        const auto v_el_golden = -(1.0_dt + momentum) * learning_rate;
        for (raul::dtype val_el : memory_manager["Nesterov::params::v"])
            EXPECT_FLOAT_EQ(val_el, v_el_golden);

        for (raul::dtype d : params)
            EXPECT_EQ(d, 1.0_dt - 5.0_dt * learning_rate);
    }
}

#ifdef ANDROID
TEST(TestOptimizerNesterov, DoubleStepFP16Unit)
{
    PROFILE_TEST
    raul::MemoryManagerFP16 memory_manager;
    const auto learning_rate = 0.1_dt;
    const auto momentum = 1.0_dt;
    const auto amount_of_element = 10U;
    {
        raul::optimizers::Nesterov optimizer{ learning_rate, momentum };
        auto& params = *memory_manager.createTensor("params", 1, amount_of_element, 1, 1, 1.0_hf);
        auto& gradients = *memory_manager.createTensor("gradients", 1, amount_of_element, 1, 1, 1.0_hf);
        optimizer(memory_manager, params, gradients);
        std::fill(gradients.begin(), gradients.end(), 1_hf);
        optimizer(memory_manager, params, gradients);

        EXPECT_EQ(params.size(), amount_of_element);
        EXPECT_EQ(gradients.size(), amount_of_element);

        const auto v_el_golden = -(1.0_dt + momentum) * TOHTYPE(learning_rate);
        for (raul::half val_el : memory_manager["Nesterov::params::v"])
        {
            EXPECT_FLOAT_EQ(TODTYPE(val_el), v_el_golden);
        }

        for (raul::half d : params)
        {
            EXPECT_EQ(TODTYPE(d), 1.0_dt - 5.0_dt * learning_rate);
        }
    }
}
#endif // ANDROID

} // UT namespace