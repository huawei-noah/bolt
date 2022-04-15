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

#include <training/layers/basic/LogLayer.h>
#include <training/layers/basic/SquareLayer.h>
#include <training/network/Layers.h>
#include <training/network/Workflow.h>

namespace UT
{

namespace
{

raul::dtype golden_square_layer(const raul::dtype x)
{
    return x * x;
}

raul::dtype golden_square_layer_grad(const raul::dtype x, const raul::dtype grad)
{
    return grad * 2.0_dt * x;
}

}

TEST(TestLayerSquare, InputNumExceedsUnit)
{
    PROFILE_TEST
    // Test parameters
    const auto tensor_size = 1000U;
    const auto params = raul::ElementWiseLayerParams{ { "x", "y" }, { "x_out" } };

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);
    memory_manager.createTensor("x", tensor_size, 1, 1, 1);
    memory_manager.createTensor("y", tensor_size, 1, 1, 1);

    // Apply function
    ASSERT_THROW(raul::SquareLayer("square", params, networkParameters), raul::Exception);
}

TEST(TestLayerSquare, OutputNumExceedsUnit)
{
    PROFILE_TEST
    // Test parameters
    const auto tensor_size = 1000U;
    const auto params = raul::ElementWiseLayerParams{ { { "x" }, { "x_out", "y_out" } } };

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);
    memory_manager.createTensor("x", tensor_size, 1, 1, 1);

    // Apply function
    ASSERT_THROW(raul::SquareLayer("square", params, networkParameters), raul::Exception);
}

TEST(TestLayerSquare, ForwardUnit)
{
    PROFILE_TEST
    // Test parameters
    const auto eps_rel = TODTYPE(1e-6);
    const auto tensor_size = 1000U;
    const auto random_range = std::make_pair(-100.0_dt, 100.0_dt);

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    work.add<raul::DataLayer>("data", raul::DataParams{ { "x" }, 1, 1, 1 });

    // Apply function
    raul::SquareLayer square("square", raul::ElementWiseLayerParams{ { "x" }, { "out" } }, networkParameters);
    TENSORS_CREATE(tensor_size);
    tools::init_rand_tensor("x", random_range, memory_manager);

    square.forwardCompute(raul::NetworkMode::Train);

    // Checks
    const auto& x_tensor = memory_manager["x"];
    const auto& out_tensor = memory_manager["out"];

    EXPECT_EQ(out_tensor.size(), x_tensor.size());

    for (size_t i = 0; i < out_tensor.size(); ++i)
    {
        const auto x_value = x_tensor[i];
        const auto out_value = out_tensor[i];
        const auto golden_out_value = golden_square_layer(x_value);
        ASSERT_TRUE(tools::expect_near_relative(out_value, golden_out_value, eps_rel));
    }
}

TEST(TestLayerSquare, BackwardUnit)
{
    PROFILE_TEST
    // Test parameters
    const auto eps_rel = TODTYPE(1e-6);
    const auto tensor_size = 1000U;
    const auto random_range = std::make_pair(-100.0_dt, 100.0_dt);

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    work.add<raul::DataLayer>("data", raul::DataParams{ { "x" }, 1, 1, 1 });

    // Apply function
    raul::SquareLayer square("square", raul::ElementWiseLayerParams{ { "x" }, { "out" } }, networkParameters);
    TENSORS_CREATE(tensor_size);
    tools::init_rand_tensor("x", random_range, memory_manager);
    tools::init_rand_tensor(raul::Name("out").grad(), random_range, memory_manager);

    square.forwardCompute(raul::NetworkMode::Train);
    square.backwardCompute();

    // Checks
    const auto& x_tensor = memory_manager["x"];
    const auto& x_tensor_grad = memory_manager[raul::Name("x").grad()];
    const auto& out_tensor_grad = memory_manager[raul::Name("out").grad()];

    EXPECT_EQ(out_tensor_grad.size(), x_tensor_grad.size());

    for (size_t i = 0; i < out_tensor_grad.size(); ++i)
    {
        const auto x_value = x_tensor[i];
        const auto out_grad_value = out_tensor_grad[i];
        const auto x_grad_value = x_tensor_grad[i];
        const auto golden_out_value_x = golden_square_layer_grad(x_value, out_grad_value);
        ASSERT_TRUE(tools::expect_near_relative(x_grad_value, golden_out_value_x, eps_rel));
    }
}

TEST(TestLayerSquare, GpuUnit)
{
    PROFILE_TEST

    GPU_ONLY_TEST

    using namespace raul;

    constexpr size_t BATCH_SIZE = 1;
    constexpr size_t WIDTH = 3;
    constexpr size_t HEIGHT = 9;
    constexpr size_t DEPTH = 1;
    constexpr dtype eps = 1.0e-6_dt;
    constexpr auto range = std::make_pair(1.0_dt, 100.0_dt);

    WorkflowEager work(raul::CompressionMode::NONE, raul::CalculationMode::DETERMINISTIC, raul::AllocationMode::STANDARD, raul::ExecutionTarget::GPU);
    work.getKernelManager().setExecutionPolicy(raul::KernelExecutionPolicy::DefaultParams);

    work.add<DataLayer>("data", DataParams{ { "in" }, DEPTH, HEIGHT, WIDTH });
    work.add<SquareLayer>("square", ElementWiseLayerParams{ { "in" }, { "out" } });

    TENSORS_CREATE(BATCH_SIZE)

    MemoryManagerGPU& memory_manager = work.getMemoryManager<MemoryManagerGPU>();

    tools::init_rand_tensor("in", range, memory_manager);
    tools::init_rand_tensor("outGradient", range, memory_manager);

    work.forwardPassTraining();
    const Tensor& in = memory_manager["in"];
    const Tensor& out = memory_manager["out"];
    EXPECT_EQ(in.size(), out.size());
    for (size_t i = 0; i < out.size(); ++i)
    {
        EXPECT_NEAR(out[i], golden_square_layer(in[i]), eps);
    }

    work.backwardPassTraining();
    const Tensor& inGrad = memory_manager[Name("in").grad()];
    const Tensor& outGrad = memory_manager[Name("out").grad()];
    EXPECT_EQ(in.size(), inGrad.size());
    for (size_t i = 0; i < inGrad.size(); ++i)
    {
        EXPECT_NEAR(inGrad[i], golden_square_layer_grad(in[i], outGrad[i]), eps);
    }
}

}