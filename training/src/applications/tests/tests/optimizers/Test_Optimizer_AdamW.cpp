// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

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

#include <training/optimizers/AdamW.h>

namespace UT
{

TEST(TestOptimizerAdamW, StreamUnit)
{
    PROFILE_TEST
    std::ostringstream stream;
    {
        raul::optimizers::AdamW optimizer{ 0.01_dt };
        stream << optimizer;
        ASSERT_STREQ(stream.str().c_str(), "AdamW(alpha=1.000000e-02, beta_1=9.000000e-01, beta_2=9.990000e-01, epsilon=1.000000e-08, lambda=1.000000e-02)");
    }
}

// see adamw.py
TEST(TestOptimizerAdamW, StepUnit)
{
    PROFILE_TEST
    raul::MemoryManager memory_manager;
    constexpr auto alpha = 0.8_dt;
    constexpr auto beta_1 = 0.5_dt;
    constexpr auto beta_2 = 0.75_dt;
    constexpr auto epsilon = 0.6_dt;
    constexpr auto lambda = 0.2_dt;
    constexpr auto amount_of_element = 10U;
    constexpr auto res = 0.34_dt;
    {
        raul::optimizers::AdamW optimizer{ alpha, beta_1, beta_2, epsilon, lambda };
        auto& params = *memory_manager.createTensor("params", 1, amount_of_element, 1, 1, 1.0_dt);
        auto& gradients = *memory_manager.createTensor("gradients", 1, amount_of_element, 1, 1, 1.0_dt);
        optimizer(memory_manager, params, gradients);

        EXPECT_EQ(params.size(), amount_of_element);
        EXPECT_EQ(gradients.size(), amount_of_element);

        EXPECT_FLOAT_EQ(memory_manager["AdamW::params::beta_1_t"][0], beta_1 * beta_1);
        EXPECT_FLOAT_EQ(memory_manager["AdamW::params::beta_2_t"][0], beta_2 * beta_2);

        const auto m_t = (1.0_dt - beta_1);
        const auto v_t = (1.0_dt - beta_2);

        for (raul::dtype m_tensor_t_el : memory_manager["AdamW::params::m"])
        {
            EXPECT_FLOAT_EQ(m_tensor_t_el, m_t);
        }

        for (raul::dtype v_tensor_t_el : memory_manager["AdamW::params::v"])
        {
            EXPECT_FLOAT_EQ(v_tensor_t_el, v_t);
        }

        for (raul::dtype param_tensor_el : params)
        {
            EXPECT_FLOAT_EQ(param_tensor_el, res);
        }
    }
}

TEST(TestOptimizerAdamW, DoubleStepUnit)
{
    PROFILE_TEST
    raul::MemoryManager memory_manager;
    constexpr auto alpha = 0.8_dt;
    constexpr auto beta_1 = 0.5_dt;
    constexpr auto beta_2 = 0.75_dt;
    constexpr auto epsilon = 0.6_dt;
    constexpr auto lambda = 0.2_dt;
    constexpr auto amount_of_element = 10U;
    constexpr auto res = -0.2144_dt;
    {
        raul::optimizers::AdamW optimizer{ alpha, beta_1, beta_2, epsilon, lambda };
        auto& params = *memory_manager.createTensor("params", 1, amount_of_element, 1, 1, 1.0_dt);
        auto& gradients = *memory_manager.createTensor("gradients", 1, amount_of_element, 1, 1, 1.0_dt);
        optimizer(memory_manager, params, gradients);
        std::fill(gradients.begin(), gradients.end(), 1_dt);
        optimizer(memory_manager, params, gradients);

        EXPECT_EQ(params.size(), amount_of_element);
        EXPECT_EQ(gradients.size(), amount_of_element);

        EXPECT_FLOAT_EQ(memory_manager["AdamW::params::beta_1_t"][0], beta_1 * beta_1 * beta_1);
        EXPECT_FLOAT_EQ(memory_manager["AdamW::params::beta_2_t"][0], beta_2 * beta_2 * beta_2);

        const auto m_t = (1.0_dt - beta_1 * beta_1);
        const auto v_t = (1.0_dt - beta_2 * beta_2);

        for (raul::dtype m_tensor_t_el : memory_manager.getTensor("AdamW::params::m"))
        {
            EXPECT_FLOAT_EQ(m_tensor_t_el, m_t);
        }

        for (raul::dtype v_tensor_t_el : memory_manager.getTensor("AdamW::params::v"))
        {
            EXPECT_FLOAT_EQ(v_tensor_t_el, v_t);
        }

        for (raul::dtype param_tensor_el : params)
        {
            EXPECT_FLOAT_EQ(param_tensor_el, res);
        }
    }
}

TEST(TestOptimizerAdamW, ToyNetTraining)
{
    PROFILE_TEST
    constexpr auto acc_eps = 1e-2_dt;
    constexpr auto loss_eps_rel = 1e-5_dt;
    constexpr auto lr = 1e-3_dt;
    const size_t batch_size = 50U;
    const size_t hidden_size = 500U;
    const size_t amount_of_classes = 10U;
    const size_t mnist_size = 28U;
    const size_t mnist_channels = 1U;
    const size_t print_freq = 100U;

    const auto golden_acc_before = 10.28_dt;
    const auto golden_acc_after = 93.05_dt;

    const raul::Tensor idealLosses{ 2.349840_dt, 0.605465_dt, 0.503376_dt, 0.248671_dt, 0.505824_dt, 0.160106_dt, 0.354624_dt, 0.101470_dt, 0.229938_dt, 0.447792_dt, 0.211203_dt, 0.222475_dt };

    std::cout << "Loading MNIST..." << std::endl;
    raul::MNIST mnist;
    ASSERT_EQ(mnist.loadingData(tools::getTestAssetsDir() / "MNIST"), true);

    std::cout << "Building network..." << std::endl;
    const auto weights_path = tools::getTestAssetsDir() / "toy_network" / "784_500_10_seed_0";
    auto network = tools::toynet::buildUnique(mnist_size, mnist_size, mnist_channels, batch_size, amount_of_classes, hidden_size, tools::toynet::Nonlinearity::sigmoid, false, weights_path);

    // Before train
    {
        std::cout << "Calculating acc...";
        raul::dtype testAcc = mnist.testNetwork(*network);
        std::cout << testAcc << std::endl;
        EXPECT_NEAR(testAcc, golden_acc_before, acc_eps);
    }
    // Training
    const size_t stepsAmountTrain = mnist.getTrainImageAmount() / batch_size;
    auto adam = std::make_shared<raul::optimizers::AdamW>(lr);
    for (size_t q = 0; q < stepsAmountTrain; ++q)
    {
        raul::dtype testLoss = mnist.oneTrainIteration(*network, adam.get(), q);
        if (q % print_freq == 0)
        {
            EXPECT_TRUE(tools::expect_near_relative(testLoss, idealLosses[q / print_freq], loss_eps_rel)) << "expected: " << idealLosses[q / print_freq] << ", got: " << testLoss;
            printf("iteration = %d, loss = %f\n", static_cast<uint32_t>(q), testLoss);
            fflush(stdout);
        }
    }

    // After train
    {
        std::cout << "Calculating acc...";
        raul::dtype testAcc = mnist.testNetwork(*network);
        std::cout << testAcc << std::endl;
        EXPECT_NEAR(testAcc, golden_acc_after, acc_eps);
    }
}

#ifdef ANDROID
TEST(TestOptimizerAdamW, DoubleStepFP16Unit)
{
    PROFILE_TEST
    raul::MemoryManagerFP16 memory_manager;
    constexpr auto alpha = 0.8_dt;
    constexpr auto beta_1 = 0.5_dt;
    constexpr auto beta_2 = 0.75_dt;
    constexpr auto epsilon = 0.6_dt;
    constexpr auto lambda = 0.2_dt;
    constexpr auto amount_of_element = 10U;
    constexpr auto res = -0.2144_dt;
    constexpr auto eps = 1.0e-3_dt;
    {
        raul::optimizers::AdamW optimizer{ alpha, beta_1, beta_2, epsilon, lambda };
        auto& params = *memory_manager.createTensor("params", 1, amount_of_element, 1, 1, 1.0_hf);
        auto& gradients = *memory_manager.createTensor("gradients", 1, amount_of_element, 1, 1, 1.0_hf);
        optimizer(memory_manager, params, gradients);
        std::fill(gradients.begin(), gradients.end(), 1.0_hf);
        optimizer(memory_manager, params, gradients);

        EXPECT_EQ(params.size(), amount_of_element);
        EXPECT_EQ(gradients.size(), amount_of_element);

        EXPECT_FLOAT_EQ(TODTYPE(memory_manager["AdamW::params::beta_1_t"][0]), beta_1 * beta_1 * beta_1);
        EXPECT_FLOAT_EQ(TODTYPE(memory_manager["AdamW::params::beta_2_t"][0]), beta_2 * beta_2 * beta_2);

        const auto m_t = (1.0_dt - beta_1 * beta_1);
        const auto v_t = (1.0_dt - beta_2 * beta_2);

        for (raul::half m_tensor_t_el : memory_manager.getTensor("AdamW::params::m"))
        {
            EXPECT_FLOAT_EQ(TODTYPE(m_tensor_t_el), m_t);
        }

        for (raul::half v_tensor_t_el : memory_manager.getTensor("AdamW::params::v"))
        {
            EXPECT_FLOAT_EQ(TODTYPE(v_tensor_t_el), v_t);
        }

        for (raul::half param_tensor_el : params)
        {
            EXPECT_NEAR(TODTYPE(param_tensor_el), res, eps);
        }
    }
}
#endif // ANDROID

}