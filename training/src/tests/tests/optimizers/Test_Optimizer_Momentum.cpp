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

#include "training/base/optimizers/Momentum.h"
#include <training/base/layers/activations/SigmoidActivation.h>

namespace UT
{

TEST(TestOptimizerMomentum, StreamUnit)
{
    PROFILE_TEST
    raul::MemoryManager memory_manager;
    std::ostringstream stream;
    const auto learning_rate = 0.01_dt;
    const auto momentum = 1.0_dt;
    {
        raul::optimizers::Momentum optimizer{ learning_rate, momentum };
        stream << optimizer;
        ASSERT_STREQ(stream.str().c_str(), "Momentum(lr=1.000000e-02, momentum=1.000000e+00)");
    }
}

TEST(TestOptimizerMomentum, StepUnit)
{
    PROFILE_TEST
    raul::MemoryManager memory_manager;
    const auto learning_rate = 0.1_dt;
    const auto momentum = 1.0_dt;
    const auto amount_of_element = 10U;
    {
        raul::optimizers::Momentum optimizer{ learning_rate, momentum };
        auto& params = *memory_manager.createTensor("params", 1, amount_of_element, 1, 1, 1.0_dt);
        auto& gradients = *memory_manager.createTensor("gradients", 1, amount_of_element, 1, 1, 1.0_dt);
        optimizer(memory_manager, params, gradients);

        EXPECT_EQ(params.size(), amount_of_element);
        EXPECT_EQ(gradients.size(), amount_of_element);
        for (raul::dtype d : params)
            EXPECT_EQ(d, 1.0_dt - learning_rate);
    }
}

TEST(TestOptimizerMomentum, DoubleStepUnit)
{
    PROFILE_TEST
    // We expect here after 2 steps of the optimizer
    // 1st: velocity_new = 0*momentum+lr*1 = lr
    //      param_new = param - velocity_new = 1-lr
    // 2st: velocity_new = lr*momentum+lr*1 = 2*lr
    //      param_new = param - velocity_new = (1-lr) - 2*lr = 1-3*lr
    raul::MemoryManager memory_manager;
    const auto learning_rate = 0.1_dt;
    const auto momentum = 1.0_dt;
    const auto amount_of_element = 10U;
    {
        raul::optimizers::Momentum optimizer{ learning_rate, momentum };
        auto& params = *memory_manager.createTensor("params", 1, amount_of_element, 1, 1, 1.0_dt);
        auto& gradients = *memory_manager.createTensor("gradients", 1, amount_of_element, 1, 1, 1.0_dt);
        optimizer(memory_manager, params, gradients);
        std::fill(gradients.begin(), gradients.end(), 1_dt);
        optimizer(memory_manager, params, gradients);

        EXPECT_EQ(params.size(), amount_of_element);
        EXPECT_EQ(gradients.size(), amount_of_element);
        for (raul::dtype d : params)
            EXPECT_EQ(d, 1.0_dt - 3.0_dt * learning_rate);
    }
}

TEST(TestOptimizerMomentum, SmallMomentumDoubleStepRandUnit)
{
    PROFILE_TEST
    // We expect here after 2 steps of the optimizer
    // 1st: velocity_new = 0*momentum+lr*1 = lr
    //      param_new = param - velocity_new = 1-lr
    // 2st: velocity_new = lr*momentum+lr*1 = (momentum+1)*lr
    //      param_new = param - velocity_new = (1-lr)-(momentum+1)*lr = 1-(2+momentum)*lr
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(.0_dt, std::nextafter(1.0, std::numeric_limits<raul::dtype>::max())); // [0,1]

    raul::MemoryManager memory_manager;
    const auto learning_rate = 0.1_dt;
    const auto momentum = static_cast<raul::dtype>(dis(gen));
    const auto amount_of_element = 10U;
    {
        raul::optimizers::Momentum optimizer{ learning_rate, momentum };
        auto& params = *memory_manager.createTensor("params", 1, amount_of_element, 1, 1, 1.0_dt);
        auto& gradients1 = *memory_manager.createTensor("gradients1", 1, amount_of_element, 1, 1, 1.0_dt);
        auto& gradients2 = *memory_manager.createTensor("gradients2", 1, amount_of_element, 1, 1, 1.0_dt);

        optimizer(memory_manager, params, gradients1);
        optimizer(memory_manager, params, gradients2);

        EXPECT_EQ(params.size(), amount_of_element);
        EXPECT_EQ(gradients1.size(), amount_of_element);
        EXPECT_EQ(gradients2.size(), amount_of_element);
        for (raul::dtype d : params)

            EXPECT_FLOAT_EQ(d, 1.0_dt - (2.0_dt + momentum) * learning_rate);
    }
}

#ifdef ANDROID
TEST(TestOptimizerMomentum, DoubleStepFP16Unit)
{
    PROFILE_TEST
    raul::MemoryManagerFP16 memory_manager;
    const auto learning_rate = 0.1_dt;
    const auto momentum = 1.0_dt;
    const auto amount_of_element = 10U;
    const auto eps = 1e-3_dt;
    {
        raul::optimizers::Momentum optimizer{ learning_rate, momentum };
        auto& params = *memory_manager.createTensor("params", 1, amount_of_element, 1, 1, 1.0_hf);
        auto& gradients = *memory_manager.createTensor("gradients", 1, amount_of_element, 1, 1, 1.0_hf);
        optimizer(memory_manager, params, gradients);
        std::fill(gradients.begin(), gradients.end(), 1.0_hf);
        optimizer(memory_manager, params, gradients);

        EXPECT_EQ(params.size(), amount_of_element);
        EXPECT_EQ(gradients.size(), amount_of_element);
        for (raul::half d : params)
        {
            EXPECT_NEAR(TODTYPE(d), 1.0_dt - 3.0_dt * learning_rate, eps);
        }
    }
}
#endif // ANDROID

} // UT namespace