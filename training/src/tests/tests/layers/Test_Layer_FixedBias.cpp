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

#include <training/base/initializers/ConstantInitializer.h>
#include <training/base/layers/basic/DataLayer.h>
#include <training/base/layers/basic/FixedBiasLayer.h>

namespace UT
{

using namespace raul;

namespace
{

template<typename T>
T golden_scale_layer(const T x, const T bias)
{
    return bias + x;
}

template<typename T>
T golden_scale_layer_grad(const T grad)
{
    return grad;
}

}

TEST(TestLayerFixedBias, InputNumExceedsUnit)
{
    // Test parameters
    const auto bias = 2.0_dt;
    const auto params = FixedBiasParams{ { "x", "y" }, { "x_out" }, bias };

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    // Apply function
    work.add<DataLayer>("input", DataParams{ { "x", "y" }, 1, 1, 1 });
    ASSERT_THROW(FixedBiasLayer("two", params, networkParameters), raul::Exception);
}

TEST(TestLayerFixedBias, OutputNumExceedsUnit)
{
    // Test parameters
    const auto bias = 2.0_dt;
    const auto params = FixedBiasParams{ { "x" }, { "x_out", "y_out" }, bias };

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    // Apply function
    work.add<DataLayer>("input", DataParams{ { "x" }, 1, 1, 1 });
    ASSERT_THROW(FixedBiasLayer("two", params, networkParameters), raul::Exception);
}

TEST(TestLayerFixedBias, TwoForwardUnit)
{
    // Test parameters
    const auto eps_rel = TODTYPE(1e-6);
    const auto tensor_size = 1000U;
    const auto bias = 2.0_dt;
    const auto params = FixedBiasParams{ { "x" }, { "out" }, bias };

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    // Apply function
    work.add<DataLayer>("input", DataParams{ { "x" }, 1, 1, 1 });
    FixedBiasLayer two("two", params, networkParameters);
    TENSORS_CREATE(tensor_size);
    initializers::ConstantInitializer initializer{ 1.0_dt };
    initializer(memory_manager["x"]);
    two.forwardCompute(NetworkMode::Test);

    // Checks
    const auto& x_tensor = memory_manager["x"];
    const auto& out_tensor = memory_manager["out"];

    EXPECT_EQ(out_tensor.size(), x_tensor.size());
    for (size_t i = 0; i < out_tensor.size(); ++i)
    {
        const auto x_value = x_tensor[i];
        const auto out_value = out_tensor[i];
        const auto golden_out_value = golden_scale_layer(x_value, bias);
        ASSERT_TRUE(tools::expect_near_relative(out_value, golden_out_value, eps_rel));
    }
}

TEST(TestLayerFixedBias, ForwardRandUnit)
{
    // Test parameters
    const auto eps_rel = 1e-6_dt;
    const auto tensor_size = random::uniform::rand<int>({ 1, 1000 });
    const auto bias = random::uniform::rand<raul::dtype>({ -100.0_dt, 100.0_dt });

    std::cout << "Run test with bias=" << bias << " and tensor shape (" << tensor_size << ",1,1,1)" << std::endl;

    const auto params = FixedBiasParams{ { "x" }, { "out" }, bias };

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    // Apply function
    work.add<DataLayer>("input", DataParams{ { "x" }, 1, 1, 1 });
    FixedBiasLayer biaser("biaser", params, networkParameters);
    TENSORS_CREATE(tensor_size);
    initializers::ConstantInitializer initializer{ 1.0_dt };
    initializer(memory_manager["x"]);
    biaser.forwardCompute(NetworkMode::Test);

    // Checks
    const auto& x_tensor = memory_manager["x"];
    const auto& out_tensor = memory_manager["out"];

    EXPECT_EQ(out_tensor.size(), x_tensor.size());
    for (size_t i = 0; i < out_tensor.size(); ++i)
    {
        const auto x_value = x_tensor[i];
        const auto out_value = out_tensor[i];
        const auto golden_out_value = golden_scale_layer(x_value, bias);
        ASSERT_TRUE(tools::expect_near_relative(out_value, golden_out_value, eps_rel));
    }
}

TEST(TestLayerFixedBias, BackwardRandUnit)
{
    // Test parameters
    const auto eps_rel = 1e-6_dt;
    const auto tensor_size = random::uniform::rand<int>({ 1, 1000 });
    const auto bias = random::uniform::rand<raul::dtype>({ -100.0_dt, 100.0_dt });

    std::cout << "Run test with bias=" << bias << " and tensor shape (" << tensor_size << ",1,1,1)" << std::endl;

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    const auto params = FixedBiasParams{ { "x" }, { "out" }, bias };
    // Apply function
    work.add<DataLayer>("input", DataParams{ { "x" }, 1, 1, 1 });
    FixedBiasLayer biaser("biaser", params, networkParameters);
    TENSORS_CREATE(tensor_size);
    tools::init_rand_tensor("x", { -100.0_dt, 100.0_dt }, memory_manager);
    biaser.forwardCompute(NetworkMode::Train);
    memory_manager[Name("out").grad()].memAllocate(nullptr);
    tools::init_rand_tensor(Name("out").grad(), { -100.0_dt, 100.0_dt }, memory_manager);
    biaser.backwardCompute();

    // Checks
    const auto& x_tensor_grad = memory_manager[Name("x").grad()];
    const auto& out_tensor_grad = memory_manager[Name("out").grad()];

    EXPECT_EQ(out_tensor_grad.size(), x_tensor_grad.size());

    for (size_t i = 0; i < out_tensor_grad.size(); ++i)
    {
        const auto out_grad_value = out_tensor_grad[i];
        const auto x_grad_value = x_tensor_grad[i];
        const auto golden_out_value_x = golden_scale_layer_grad(out_grad_value);
        ASSERT_TRUE(tools::expect_near_relative(x_grad_value, golden_out_value_x, eps_rel)) << "expected: " << golden_out_value_x << ", got: " << x_grad_value;
    }
}
#ifdef ANDROID
TEST(TestLayerFixedBias, TwoForwardFP16Unit)
{
    // Test parameters
    const auto eps_rel = 1e-6_hf;
    const auto tensor_size = 1000U;
    const auto bias = 2.0_hf;
    const auto params = FixedBiasParams{ { "x" }, { "out" }, bias };

    // Initialization
    raul::WorkflowEager work{ raul::CompressionMode::NONE, raul::CalculationMode::DETERMINISTIC, raul::AllocationMode::STANDARD, raul::ExecutionTarget::CPUFP16 };
    auto& memory_manager = work.getMemoryManager<raul::MemoryManagerFP16>();
    NETWORK_PARAMS_DEFINE(networkParameters);

    // Apply function
    work.add<DataLayer>("input", DataParams{ { "x" }, 1, 1, 1 });
    FixedBiasLayer two("two", params, networkParameters);
    TENSORS_CREATE(tensor_size);

    auto& tensorX = memory_manager["x"];
    for (auto& val : tensorX)
    {
        val = 1.0_hf;
    }

    two.forwardCompute(NetworkMode::Test);

    // Checks
    const auto& x_tensor = memory_manager["x"];
    const auto& out_tensor = memory_manager["out"];

    EXPECT_EQ(out_tensor.size(), x_tensor.size());
    for (size_t i = 0; i < out_tensor.size(); ++i)
    {
        const auto x_value = x_tensor[i];
        const auto out_value = out_tensor[i];
        const auto golden_out_value = golden_scale_layer(x_value, bias);
        ASSERT_TRUE(tools::expect_near_relative(out_value, golden_out_value, eps_rel));
    }
}
#endif // ANDROID

}