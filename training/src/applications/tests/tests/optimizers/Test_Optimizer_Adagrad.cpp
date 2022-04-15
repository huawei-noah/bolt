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

#include <training/api/API.h>
#include <training/layers/activations/SigmoidActivation.h>
#include <training/optimizers/Adagrad.h>

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

TEST(TestOptimizerAdagrad, ToyNetTraining)
{
    PROFILE_TEST
    constexpr auto acc_eps = 1e-2_dt;
    constexpr auto loss_eps_rel = 1e-6_dt;
    constexpr auto lr = 1e-2_dt;
    constexpr auto epsilon = 1e-10_dt;
    const size_t batch_size = 50U;
    const size_t hidden_size = 500U;
    const size_t amount_of_classes = 10U;
    const size_t mnist_size = 28U;
    const size_t mnist_channels = 1U;
    const size_t print_freq = 100U;

    const auto golden_acc_before = 10.28_dt;
    const auto golden_acc_after = 92.18_dt;

    const raul::Tensor idealLosses{ 2.349840e+00_dt, 5.051338e-01_dt, 4.570299e-01_dt, 2.119440e-01_dt, 4.868208e-01_dt, 1.713404e-01_dt,
                                    3.266897e-01_dt, 1.343931e-01_dt, 1.952383e-01_dt, 4.680505e-01_dt, 2.441438e-01_dt, 2.500688e-01_dt };

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
    auto adagrad = std::make_shared<raul::optimizers::Adagrad>(lr, epsilon);
    for (size_t q = 0; q < stepsAmountTrain; ++q)
    {
        raul::dtype testLoss = mnist.oneTrainIteration(*network, adagrad.get(), q);
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
