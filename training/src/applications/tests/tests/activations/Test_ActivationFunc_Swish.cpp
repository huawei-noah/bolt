// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <chrono>
#include <cstdio>
#include <tests/tools/TestTools.h>

#include <training/layers/activations/SwishActivation.h>
#include <training/network/Layers.h>
#include <training/network/Workflow.h>
#include <training/optimizers/SGD.h>

namespace UT
{

namespace
{

raul::dtype golden_sigmoid(const raul::dtype x)
{
    return 1.0_dt / (1.0_dt + std::exp(-x));
}

raul::dtype golden_swish(const raul::dtype x)
{
    return x * golden_sigmoid(x);
}

// See https://www.wolframalpha.com/input/?i=derivative+x*+sigmoid%28x%29
raul::dtype golden_swish_grad(const raul::dtype x, const raul::dtype grad)
{
    return grad * (golden_sigmoid(x) + x * golden_sigmoid(x) * (1.0_dt - golden_sigmoid(x)));
}

}

TEST(TestActivationFuncSwish, ForwardRandUnit)
{
    PROFILE_TEST
    // Test parameters
    const auto eps = 1e-6_dt;
    const auto tensor_size = 1000U;
    // We have a significant change in the range [-5,5]
    // See https://www.wolframalpha.com/input/?i=x*sigmoid%28x%29
    const auto random_range = std::pair<raul::dtype, raul::dtype>(-5.0f, 5.0f);

    // Random generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(random_range.first, std::nextafter(random_range.second, std::numeric_limits<raul::dtype>::max()));

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    work.add<raul::DataLayer>("data", raul::DataParams{ { "in" }, 1, 1, 1 });

    const auto params = raul::BasicParams{ { "in" }, { "out" } };

    // Apply function
    raul::SwishActivation swish_activation("swish", params, networkParameters);
    TENSORS_CREATE(tensor_size);

    auto& in_tensor = memory_manager["in"];
    for (auto& val : in_tensor)
    {
        val = static_cast<raul::dtype>(dis(gen));
    }

    swish_activation.forwardCompute(raul::NetworkMode::Test);

    // Checks
    const auto& out_tensor = memory_manager["out"];

    EXPECT_EQ(out_tensor.size(), in_tensor.size());

    for (size_t i = 0; i < out_tensor.size(); ++i)
    {
        const auto in_value = in_tensor[i];
        const auto out_value = out_tensor[i];
        const auto golden_out_value = golden_swish(in_value);
        EXPECT_NEAR(out_value, golden_out_value, eps);
    }
}

TEST(TestActivationFuncSwish, BackwardRandUnit)
{
    PROFILE_TEST
    // Test parameters
    const auto eps = 1e-5_dt; // TODO(ck): try to increase precision up to 1e-6
    const auto tensor_size = 1000U;
    // We have a significant change in the range [-5,5]
    // See https://www.wolframalpha.com/input/?i=x*sigmoid%28x%29
    const auto random_range = std::pair<raul::dtype, raul::dtype>(-5.0f, 5.0f);

    // Random generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(random_range.first, std::nextafter(random_range.second, std::numeric_limits<raul::dtype>::max()));

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    work.add<raul::DataLayer>("data", raul::DataParams{ { "in" }, 1, 1, 1 });

    const auto params = raul::BasicParams{ { "in" }, { "out" } };

    // Apply function
    raul::SwishActivation swish_activation("swish", params, networkParameters);
    TENSORS_CREATE(tensor_size);

    auto& in_tensor = memory_manager["in"];
    for (auto& val : in_tensor)
    {
        val = static_cast<raul::dtype>(dis(gen));
    }

    auto& grad_tensor = memory_manager[raul::Name("out").grad()];
    for (auto& val : grad_tensor)
    {
        val = static_cast<raul::dtype>(dis(gen));
    }

    swish_activation.forwardCompute(raul::NetworkMode::Train);
    swish_activation.backwardCompute();

    // Checks
    const auto& out_tensor_grad = memory_manager[raul::Name("in").grad()];

    EXPECT_EQ(out_tensor_grad.size(), grad_tensor.size());

    for (size_t i = 0; i < out_tensor_grad.size(); ++i)
    {
        const auto in_value = in_tensor[i];
        const auto grad_value = grad_tensor[i];
        const auto out_value = out_tensor_grad[i];
        const auto golden_out_value = golden_swish_grad(in_value, grad_value);
        EXPECT_NEAR(out_value, golden_out_value, eps);
    }
}

TEST(TestActivationFuncSwish, GpuUnit)
{
    PROFILE_TEST

    GPU_ONLY_TEST

    using namespace raul;

    constexpr size_t BATCH_SIZE = 3;
    constexpr size_t WIDTH = 11;
    constexpr size_t HEIGHT = 1;
    constexpr size_t DEPTH = 23;
    constexpr dtype eps = 1.0e-6_dt;
    constexpr auto range = std::make_pair(-1.0_dt, 1.0_dt);

    WorkflowEager work(raul::CompressionMode::NONE, raul::CalculationMode::DETERMINISTIC, raul::AllocationMode::STANDARD, raul::ExecutionTarget::GPU);
    work.getKernelManager().setExecutionPolicy(raul::KernelExecutionPolicy::DefaultParams);

    work.add<DataLayer>("data", DataParams{ { "in" }, DEPTH, HEIGHT, WIDTH });
    work.add<SwishActivation>("swish", BasicParams{ { "in" }, { "out" } });

    TENSORS_CREATE(BATCH_SIZE);

    MemoryManagerGPU& memory_manager = work.getMemoryManager<MemoryManagerGPU>();

    tools::init_rand_tensor("in", range, memory_manager);
    tools::init_rand_tensor("outGradient", range, memory_manager);

    work.forwardPassTraining();
    const Tensor& in = memory_manager["in"];
    const Tensor& out = memory_manager["out"];
    EXPECT_EQ(in.size(), out.size());
    for (size_t i = 0; i < out.size(); ++i)
    {
        EXPECT_NEAR(out[i], golden_swish(in[i]), eps);
    }

    work.backwardPassTraining();
    const Tensor& inGrad = memory_manager[Name("in").grad()];
    const Tensor& outGrad = memory_manager[Name("out").grad()];
    EXPECT_EQ(in.size(), inGrad.size());
    for (size_t i = 0; i < inGrad.size(); ++i)
    {
        EXPECT_NEAR(inGrad[i], golden_swish_grad(in[i], outGrad[i]), eps);
    }
}

TEST(TestActivationFuncSwish, ToyNetTraining)
{
    PROFILE_TEST
    constexpr auto eps = 1e-6_dt;
    constexpr auto lr = 0.05_dt;
    const size_t batch_size = 50U;
    const size_t hidden_size = 500U;
    const size_t amount_of_classes = 10U;
    const size_t mnist_size = 28U;
    const size_t mnist_channels = 1U;
    const size_t print_freq = 100U;

    const auto golden_acc_before = 17.37_dt;
    const auto golden_acc_after = 90.35_dt;

    const raul::Tensor idealLosses{ 2.290527582168579_dt,  1.2226440906524658_dt,  0.7079215049743652_dt,  0.35579541325569153_dt, 0.655056893825531_dt,  0.2758506238460541_dt,
                                    0.4565277099609375_dt, 0.16564907133579254_dt, 0.24463535845279694_dt, 0.6562144756317139_dt,  0.3219243884086609_dt, 0.31373268365859985_dt };

    std::cout << "Loading MNIST..." << std::endl;
    raul::MNIST mnist;
    ASSERT_EQ(mnist.loadingData(tools::getTestAssetsDir() / "MNIST"), true);

    std::cout << "Building network..." << std::endl;
    const auto weights_path = tools::getTestAssetsDir() / "toy_network" / "784_500_10_seed_0";
    auto network = tools::toynet::buildUnique(mnist_size, mnist_size, mnist_channels, batch_size, amount_of_classes, hidden_size, tools::toynet::Nonlinearity::swish, false, weights_path);
    network->printInfo(std::cout);

    // Before train
    {
        std::cout << "Calculating acc...";
        raul::dtype testAcc = mnist.testNetwork(*network);
        std::cout << testAcc << std::endl;
        EXPECT_NEAR(testAcc, golden_acc_before, eps);
    }
    // Training
    const size_t stepsAmountTrain = mnist.getTrainImageAmount() / batch_size;
    auto sgd = std::make_shared<raul::optimizers::SGD>(lr);
    for (size_t q = 0; q < stepsAmountTrain; ++q)
    {
        raul::dtype testLoss = mnist.oneTrainIteration(*network, sgd.get(), q);
        if (q % print_freq == 0)
        {
            CHECK_NEAR(testLoss, idealLosses[q / print_freq], eps);
            printf("iteration = %d, loss = %f\n", static_cast<uint32_t>(q), testLoss);
        }
    }

    // After train
    {
        std::cout << "Calculating acc...";
        raul::dtype testAcc = mnist.testNetwork(*network);
        std::cout << testAcc << std::endl;
        EXPECT_NEAR(testAcc, golden_acc_after, eps);
    }
}

} // UT namespace