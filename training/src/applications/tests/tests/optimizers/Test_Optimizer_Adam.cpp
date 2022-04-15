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
#include <training/optimizers/Adam.h>

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

TEST(TestOptimizerAdam, ToyNetTraining)
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
    const auto golden_acc_after = 93.08_dt;

    const raul::Tensor idealLosses{ 2.349840e+00_dt, 6.051041e-01_dt, 5.031426e-01_dt, 2.480776e-01_dt, 5.054188e-01_dt, 1.595692e-01_dt,
                                    3.540441e-01_dt, 1.008065e-01_dt, 2.288596e-01_dt, 4.473625e-01_dt, 2.100571e-01_dt, 2.217926e-01_dt };

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
    auto adam = std::make_shared<raul::optimizers::Adam>(lr);
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

TEST(TestOptimizerAdam, StepGpuUnit)
{
    PROFILE_TEST

    GPU_ONLY_TEST

    raul::WorkflowEager work{ raul::CompressionMode::NONE, raul::CalculationMode::DETERMINISTIC, raul::AllocationMode::STANDARD, raul::ExecutionTarget::GPU };
    auto& memory_manager = work.getMemoryManager<raul::MemoryManagerGPU>();

    auto alpha = 0.8_dt;
    auto beta_1 = 0.5_dt;
    auto beta_2 = 0.75_dt;
    auto epsilon = 0.6_dt;
    auto amount_of_element = 10U;
    {
        raul::optimizers::Adam optimizer{ alpha, beta_1, beta_2, epsilon };
        auto& params = *memory_manager.createTensor("params", 1, amount_of_element, 1, 1, 1.0_dt);
        auto& gradients = *memory_manager.createTensor("gradients", 1, amount_of_element, 1, 1, 1.0_dt);
        optimizer(work.getKernelManager(), memory_manager, params, gradients);

        EXPECT_EQ(params.size(), amount_of_element);
        EXPECT_EQ(gradients.size(), amount_of_element);

        const raul::Tensor& beta_1_t = memory_manager["Adam::params::beta_1_t"];
        const raul::Tensor& beta_2_t = memory_manager["Adam::params::beta_2_t"];
        EXPECT_FLOAT_EQ(beta_1_t[0], beta_1 * beta_1);
        EXPECT_FLOAT_EQ(beta_2_t[0], beta_2 * beta_2);

        const auto m_t = (1.0_dt - beta_1);
        const auto v_t = (1.0_dt - beta_2);

        const raul::Tensor& m = memory_manager["Adam::params::m"];
        for (raul::dtype m_tensor_t_el : m)
        {
            EXPECT_FLOAT_EQ(m_tensor_t_el, m_t);
        }

        const raul::Tensor& v = memory_manager["Adam::params::v"];
        for (raul::dtype v_tensor_t_el : v)
        {
            EXPECT_FLOAT_EQ(v_tensor_t_el, v_t);
        }

        const auto res = 1.0_dt - alpha / (1.0_dt + epsilon);
        const raul::Tensor& p = memory_manager["params"];
        for (raul::dtype param_tensor_el : p)
        {
            EXPECT_FLOAT_EQ(param_tensor_el, res);
        }
    }
}

TEST(TestOptimizerAdam, DoubleStepGpuUnit)
{
    PROFILE_TEST

    GPU_ONLY_TEST

    raul::WorkflowEager work{ raul::CompressionMode::NONE, raul::CalculationMode::DETERMINISTIC, raul::AllocationMode::STANDARD, raul::ExecutionTarget::GPU };
    auto& memory_manager = work.getMemoryManager<raul::MemoryManagerGPU>();

    auto alpha = 0.8_dt;
    auto beta_1 = 0.5_dt;
    auto beta_2 = 0.75_dt;
    auto epsilon = 0.6_dt;
    auto amount_of_element = 10U;
    auto eps = 1.0e-6_dt;
    {
        raul::optimizers::Adam optimizer{ alpha, beta_1, beta_2, epsilon };
        auto& params = *memory_manager.createTensor("params", 1, amount_of_element, 1, 1, 1.0_dt);
        auto& gradients = *memory_manager.createTensor("gradients", 1, amount_of_element, 1, 1, 1.0_dt);

        optimizer(work.getKernelManager(), memory_manager, params, gradients);
        optimizer(work.getKernelManager(), memory_manager, params, gradients);

        EXPECT_EQ(params.size(), amount_of_element);
        EXPECT_EQ(gradients.size(), amount_of_element);

        const raul::Tensor& beta_1_t = memory_manager["Adam::params::beta_1_t"];
        const raul::Tensor& beta_2_t = memory_manager["Adam::params::beta_2_t"];
        EXPECT_FLOAT_EQ(beta_1_t[0], beta_1 * beta_1 * beta_1);
        EXPECT_FLOAT_EQ(beta_2_t[0], beta_2 * beta_2 * beta_2);

        const auto m_t = (1.0_dt - beta_1 * beta_1);
        const auto v_t = (1.0_dt - beta_2 * beta_2);

        const raul::Tensor& m = memory_manager["Adam::params::m"];
        for (raul::dtype m_tensor_t_el : m)
        {
            EXPECT_FLOAT_EQ(m_tensor_t_el, m_t);
        }

        const raul::Tensor& v = memory_manager["Adam::params::v"];
        for (raul::dtype v_tensor_t_el : v)
        {
            EXPECT_FLOAT_EQ(v_tensor_t_el, v_t);
        }

        const auto res = 1.0_dt - 2 * alpha / (1.0_dt + epsilon);
        const raul::Tensor& p = memory_manager["params"];
        for (raul::dtype param_tensor_el : p)
        {
            EXPECT_NEAR(param_tensor_el, res, eps);
        }
    }
}

} // UT namespace
