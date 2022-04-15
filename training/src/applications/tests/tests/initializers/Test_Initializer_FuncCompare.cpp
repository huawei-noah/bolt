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
#include <training/common/MemoryManager.h>
#include <training/initializers/ConstantInitializer.h>
#include <training/initializers/IInitializer.h>
#include <training/initializers/RandomNormInitializer.h>
#include <training/initializers/RandomUniformInitializer.h>
#include <training/initializers/XavierInitializer.h>
#include <training/layers/activations/SigmoidActivation.h>
#include <training/optimizers/SGD.h>

namespace UT
{

TEST(TestInitializersFuncCompare, ToyNetTraining)
{
    PROFILE_TEST
    const raul::dtype LEARNING_RATE = TODTYPE(0.1);
    // Network parameters
    const size_t batch_size = 50U;
    const size_t hidden_size = 500U;
    const size_t amount_of_classes = 10U;
    const size_t mnist_size = 28U;
    const size_t mnist_channels = 1U;
    const size_t print_freq = 100U;
    // Initializers parameters
    const size_t xavier_norm_seed = 0;
    const size_t xavier_uniform_seed = 1;
    const size_t random_norm_seed = 2;
    const size_t random_uniform_seed = 3;
    const raul::dtype minval = TODTYPE(-1.0);
    const raul::dtype maxval = TODTYPE(1.0);
    const raul::dtype mean = TODTYPE(0.0);
    const raul::dtype stddev = TODTYPE(1.0);
    // Initializers
    auto const_zero_initializer = std::make_unique<raul::initializers::ConstantInitializer>(0.0_dt);
    auto const_one_initializer = std::make_unique<raul::initializers::ConstantInitializer>(1.0_dt);
    auto xn_initializer = std::make_unique<raul::initializers::XavierNormInitializer>(xavier_norm_seed);
    auto xu_initializer = std::make_unique<raul::initializers::XavierUniformInitializer>(xavier_uniform_seed);
    auto rn_initializer = std::make_unique<raul::initializers::RandomNormInitializer>(random_norm_seed, mean, stddev);
    auto ru_initializer = std::make_unique<raul::initializers::RandomUniformInitializer>(random_uniform_seed, minval, maxval);
    std::array<std::unique_ptr<raul::initializers::IInitializer>, 6> initializers{ std::make_unique<raul::initializers::ConstantInitializer>(0.0_dt),
                                                                                   std::make_unique<raul::initializers::ConstantInitializer>(1.0_dt),
                                                                                   std::make_unique<raul::initializers::RandomNormInitializer>(random_norm_seed, mean, stddev),
                                                                                   std::make_unique<raul::initializers::RandomUniformInitializer>(random_uniform_seed, minval, maxval),
                                                                                   std::make_unique<raul::initializers::XavierNormInitializer>(xavier_norm_seed),
                                                                                   std::make_unique<raul::initializers::XavierUniformInitializer>(xavier_uniform_seed) };

    std::cout << "Loading MNIST..." << std::endl;
    raul::MNIST mnist;
    ASSERT_EQ(mnist.loadingData(tools::getTestAssetsDir() / "MNIST"), true);
    const size_t stepsAmountTrain = mnist.getTrainImageAmount() / batch_size;
    auto sgd = std::make_shared<raul::optimizers::SGD>(LEARNING_RATE);

    {
        for (auto& initializer : initializers)
        {
            std::cout << "Building network..." << std::endl;
            auto network = tools::toynet::buildUnique(mnist_size, mnist_size, mnist_channels, batch_size, amount_of_classes, hidden_size, tools::toynet::Nonlinearity::sigmoid);
            std::cout << "Initialiazing network with " << *initializer << std::endl;
            raul::MemoryManager& memory_manager = network->getMemoryManager();
            (*initializer)(memory_manager["fc_in::Weights"]);
            (*initializer)(memory_manager["fc_in::Biases"]);
            (*initializer)(memory_manager["fc_out::Weights"]);
            (*initializer)(memory_manager["fc_out::Biases"]);

            {
                std::cout << "Before training accuracy = ";
                raul::dtype acc = mnist.testNetwork(*network);
                std::cout << acc << std::endl;
            }
            // Training
            for (size_t q = 0; q < stepsAmountTrain; ++q)
            {
                mnist.oneTrainIteration(*network, sgd.get(), q);
                if (q % print_freq == 0)
                {
                    std::cout << *initializer << ": iteration = " << static_cast<uint32_t>(q) << " accuracy = " << mnist.testNetwork(*network) << std::endl;
                    fflush(stdout);
                }
            }
            // Final result
            std::cout << *initializer << ": final accuracy = " << mnist.testNetwork(*network) << std::endl;
        }
    }
}

} // UT namespace