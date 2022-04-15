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

#include "training/optimizers/Momentum.h"
#include <training/api/API.h>
#include <training/layers/activations/SigmoidActivation.h>

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

TEST(TestOptimizerMomentum, ToyNetTraining)
{
    PROFILE_TEST
    constexpr auto acc_eps = 1e-2_dt;
    constexpr auto loss_eps_rel = 1e-6_dt;
    constexpr auto lr = 1e-2_dt;
    constexpr auto momentum = 0.9_dt;
    const size_t batch_size = 50U;
    const size_t hidden_size = 500U;
    const size_t amount_of_classes = 10U;
    const size_t mnist_size = 28U;
    const size_t mnist_channels = 1U;
    const size_t print_freq = 100U;

    const auto golden_acc_before = 10.28_dt;
    const auto golden_acc_after = 88.27_dt;

    const raul::Tensor idealLosses{ 2.349840e+00_dt, 1.810462e+00_dt, 1.108999e+00_dt, 6.941411e-01_dt, 8.150463e-01_dt, 4.028075e-01_dt,
                                    5.807914e-01_dt, 2.615516e-01_dt, 3.698479e-01_dt, 7.210506e-01_dt, 4.273896e-01_dt, 4.469023e-01_dt };

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
    auto momentum_optimizer = std::make_shared<raul::optimizers::Momentum>(lr, momentum);
    for (size_t q = 0; q < stepsAmountTrain; ++q)
    {
        raul::dtype testLoss = mnist.oneTrainIteration(*network, momentum_optimizer.get(), q);
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
