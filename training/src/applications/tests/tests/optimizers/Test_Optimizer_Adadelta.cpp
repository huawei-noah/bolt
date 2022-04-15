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
#include <training/optimizers/Adadelta.h>

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

TEST(TestOptimizerAdadelta, ToyNetTraining)
{
    PROFILE_TEST
    constexpr auto acc_eps = 1e-2_dt;
    constexpr auto loss_eps_rel = 1e-6_dt;
    //    constexpr auto lr = 1e-2_dt;
    constexpr auto rho = 0.9_dt;
    constexpr auto epsilon = 1e-6_dt;
    const size_t batch_size = 50U;
    const size_t hidden_size = 500U;
    const size_t amount_of_classes = 10U;
    const size_t mnist_size = 28U;
    const size_t mnist_channels = 1U;
    const size_t print_freq = 100U;

    const auto golden_acc_before = 10.28_dt;
    const auto golden_acc_after = 91.87_dt;

    const raul::Tensor idealLosses{ 2.349840e+00_dt, 6.942629e-01_dt, 5.802511e-01_dt, 1.705905e-01_dt, 5.064788e-01_dt, 1.809439e-01_dt,
                                    3.887126e-01_dt, 1.262087e-01_dt, 1.543944e-01_dt, 5.208972e-01_dt, 2.165053e-01_dt, 2.640768e-01_dt };

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
    auto adadelta = std::make_shared<raul::optimizers::Adadelta>(rho, epsilon);
    for (size_t q = 0; q < stepsAmountTrain; ++q)
    {
        raul::dtype testLoss = mnist.oneTrainIteration(*network, adadelta.get(), q);
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
