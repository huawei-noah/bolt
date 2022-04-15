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
#include <training/common/Common.h>
#include <training/common/Conversions.h>
#include <training/common/DataLoader.h>
#include <training/common/MemoryManager.h>
#include <training/layers/BasicLayer.h>
#include <training/layers/activations/SigmoidActivation.h>
#include <training/layers/parameters/LayerParameters.h>
#include <training/network/Layers.h>
#include <training/network/Workflow.h>
#include <training/optimizers/SGD.h>
#include <training/optimizers/regularizers/Regularizer.h>
#include <training/optimizers/regularizers/strategies/L1.h>
#include <training/optimizers/regularizers/strategies/L2.h>
#include <training/tools/Datasets.h>

namespace UT
{

TEST(TestRegularizersL1, ToyNetTraining)
{

    PROFILE_TEST

    constexpr auto acc_eps = 1e-2_dt;
    [[maybe_unused]] constexpr auto loss_eps_rel = 1e-6_dt;
    constexpr auto lr = 0.05_dt;
    constexpr auto lambda = 0.0005_dt;
    const size_t batch_size = 50U;
    const size_t hidden_size = 500U;
    const size_t amount_of_classes = 10U;
    const size_t mnist_size = 28U;
    const size_t mnist_channels = 1U;
    const size_t print_freq = 100U;

    const auto golden_acc_before = 10.28_dt;
    const auto golden_acc_after = 82.21_dt;

    std::cout << "Loading MNIST..." << std::endl;
    raul::MNIST mnist;
    ASSERT_EQ(mnist.loadingData(tools::getTestAssetsDir() / "MNIST"), true);

    std::cout << "Building network..." << std::endl;
    const auto weights_path = tools::getTestAssetsDir() / "toy_network" / "784_500_10_seed_0";
    auto network = tools::toynet::buildUnique(mnist_size, mnist_size, mnist_channels, batch_size, amount_of_classes, hidden_size, tools::toynet::Nonlinearity::sigmoid, false, weights_path);

    // Before train
    {
        std::cout << "Calculating acc..." << std::endl;
        raul::dtype testAcc = mnist.testNetwork(*network);
        std::cout << testAcc << std::endl;
        EXPECT_NEAR(testAcc, golden_acc_before, acc_eps);
    }
    // Training
    const size_t stepsAmountTrain = mnist.getTrainImageAmount() / batch_size;

    auto reg_sgd = std::make_shared<raul::optimizers::Regularizers::Regularizer>(std::make_unique<raul::optimizers::Regularizers::Strategies::L1>(lambda), std::make_unique<raul::optimizers::SGD>(lr));

    for (size_t q = 0; q < stepsAmountTrain; ++q)
    {
        raul::dtype testLoss = mnist.oneTrainIteration(*network, reg_sgd.get(), q);
        if (q % print_freq == 0)
        {
            const auto penalty = reg_sgd->getPenalty(*network);
            printf("iteration = %d, loss = %f, penalty = %f, regularized loss = %f\n", static_cast<uint32_t>(q), testLoss, penalty, testLoss + penalty);
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

} // UT namespace
