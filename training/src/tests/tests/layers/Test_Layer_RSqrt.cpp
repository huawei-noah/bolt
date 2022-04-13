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

#include <training/base/layers/basic/RSqrtLayer.h>
#include <training/compiler/Layers.h>
#include <training/compiler/Workflow.h>

namespace UT
{

namespace
{

raul::dtype golden_rsqrt_layer(const raul::dtype x)
{
    return 1.0_dt / std::sqrt(x);
}

raul::dtype golden_rsqrt_layer_grad(const raul::dtype x, const raul::dtype grad)
{
    return grad * (-0.5_dt / std::sqrt(x) / x);
}

}

TEST(TestLayerRSqrt, InputNumExceedsUnit)
{
    PROFILE_TEST

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    // Apply function
    ASSERT_THROW(raul::RSqrtLayer("rsqrt", raul::ElementWiseLayerParams{ { "x", "y" }, { "x_out" } }, networkParameters), raul::Exception);
}

TEST(TestLayerRSqrt, OutputNumExceedsUnit)
{
    PROFILE_TEST
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    // Apply function
    ASSERT_THROW(raul::RSqrtLayer("rsqrt", raul::ElementWiseLayerParams{ { { "x" }, { "x_out", "y_out" } } }, networkParameters), raul::Exception);
}

TEST(TestLayerRSqrt, ForwardBackwardFailUnit)
{
    PROFILE_TEST
    const auto tensor_size = 1000U;
    const auto random_range = std::make_pair(-100.0_dt, 0.0_dt);

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    work.add<raul::DataLayer>("data", raul::DataParams{ { "x" }, 1, 1, 4 });

    // Apply function
    raul::RSqrtLayer rsqrt("rsqrt", raul::ElementWiseLayerParams{ { "x" }, { "out" } }, networkParameters);
    TENSORS_CREATE(tensor_size);
    tools::init_rand_tensor("x", random_range, memory_manager);

    ASSERT_THROW(rsqrt.forwardCompute(raul::NetworkMode::Train), raul::Exception);
    ASSERT_THROW(rsqrt.backwardCompute(), raul::Exception);
}

TEST(TestLayerRSqrt, ForwardZeroUnit)
{
    PROFILE_TEST
    // Test parameters
    const auto eps_rel = TODTYPE(1e-6);

    const raul::Tensor x{ 3.0_dt, 2.0_dt, 1.0_dt, 0.0_dt };
    const raul::Tensor z{ 0.5773502588_dt, 0.7071067691_dt, 1.0_dt, std::numeric_limits<raul::dtype>::infinity() };

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    work.add<raul::DataLayer>("data", raul::DataParams{ { "x" }, 1, 1, 4 });

    // Apply function
    raul::RSqrtLayer rsqrt("rsqrt", raul::ElementWiseLayerParams{ { "x" }, { "out" } }, networkParameters);
    TENSORS_CREATE(1);
    memory_manager["x"] = TORANGE(x);

    rsqrt.forwardCompute(raul::NetworkMode::Train);

    // Checks
    const auto& x_tensor = memory_manager["x"];
    const auto& out_tensor = memory_manager["out"];

    EXPECT_EQ(out_tensor.size(), x_tensor.size());

    for (size_t i = 0; i < out_tensor.size(); ++i)
    {
        ASSERT_TRUE(tools::expect_near_relative(z[i], out_tensor[i], eps_rel));
    }
}

TEST(TestLayerRSqrt, BackwardZeroUnit)
{
    PROFILE_TEST
    // Test parameters
    const auto eps_rel = TODTYPE(1e-6);

    const raul::Tensor x{ 3.0_dt, 2.0_dt, 1.0_dt, 0.0_dt };
    const raul::Tensor deltas{ 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt };
    const raul::Tensor x_grad{ -0.0962250382_dt, -0.1767766774_dt, -0.5_dt, -std::numeric_limits<raul::dtype>::infinity() };

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    work.add<raul::DataLayer>("data", raul::DataParams{ { "x" }, 1, 1, 4 });

    // Apply function
    raul::RSqrtLayer rsqrt("rsqrt", raul::ElementWiseLayerParams{ { "x" }, { "out" } }, networkParameters);
    TENSORS_CREATE(1);
    memory_manager["x"] = TORANGE(x);
    memory_manager[raul::Name("out").grad()] = TORANGE(deltas);

    rsqrt.forwardCompute(raul::NetworkMode::Train);
    rsqrt.backwardCompute();

    // Checks
    const auto& x_tensor_grad = memory_manager[raul::Name("x").grad()];

    for (size_t i = 0; i < x_tensor_grad.size(); ++i)
    {
        ASSERT_TRUE(tools::expect_near_relative(x_tensor_grad[i], x_grad[i], eps_rel)) << "expected: " << x_grad[i] << ", got: " << x_tensor_grad[i];
    }
}

TEST(TestLayerRSqrt, ForwardUnit)
{
    PROFILE_TEST
    // Test parameters
    const auto eps_rel = TODTYPE(1e-6);
    const auto tensor_size = 1000U;
    const auto random_range = std::make_pair(1.0_dt, 100.0_dt);

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    work.add<raul::DataLayer>("data", raul::DataParams{ { "x" }, 1, 1, 1 });

    // Apply function
    raul::RSqrtLayer rsqrt("rsqrt", raul::ElementWiseLayerParams{ { "x" }, { "out" } }, networkParameters);
    TENSORS_CREATE(tensor_size);
    tools::init_rand_tensor("x", random_range, memory_manager);

    rsqrt.forwardCompute(raul::NetworkMode::Train);

    // Checks
    const auto& x_tensor = memory_manager["x"];
    const auto& out_tensor = memory_manager["out"];

    EXPECT_EQ(out_tensor.size(), x_tensor.size());

    for (size_t i = 0; i < out_tensor.size(); ++i)
    {
        const auto x_value = x_tensor[i];
        const auto out_value = out_tensor[i];
        const auto golden_out_value = golden_rsqrt_layer(x_value);
        ASSERT_TRUE(tools::expect_near_relative(out_value, golden_out_value, eps_rel));
    }
}

TEST(TestLayerRSqrt, BackwardUnit)
{
    PROFILE_TEST
    // Test parameters
    const auto eps_rel = TODTYPE(1e-6);
    const auto tensor_size = 1000U;
    const auto random_range = std::make_pair(1.0_dt, 100.0_dt);

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    work.add<raul::DataLayer>("data", raul::DataParams{ { "x" }, 1, 1, 1 });

    // Apply function
    raul::RSqrtLayer rsqrt("rsqrt", raul::ElementWiseLayerParams{ { "x" }, { "out" } }, networkParameters);
    TENSORS_CREATE(tensor_size);
    tools::init_rand_tensor("x", random_range, memory_manager);
    tools::init_rand_tensor(raul::Name("out").grad(), random_range, memory_manager);

    rsqrt.forwardCompute(raul::NetworkMode::Train);
    rsqrt.backwardCompute();

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
        const auto golden_out_value_x = golden_rsqrt_layer_grad(x_value, out_grad_value);
        ASSERT_TRUE(tools::expect_near_relative(x_grad_value, golden_out_value_x, eps_rel));
    }
}

}