// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

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
#include <utility>

#include <training/base/common/Common.h>
#include <training/base/common/Conversions.h>
#include <training/base/common/MemoryManager.h>
#include <training/base/layers/basic/DataLayer.h>
#include <training/base/layers/basic/TensorLayer.h>
#include <training/base/layers/basic/ElementWiseSubLayer.h>
#include <training/compiler/Workflow.h>
#include <training/system/NameGenerator.h>

namespace UT
{

namespace
{
raul::dtype golden_sub_layer(const raul::dtype x, const raul::dtype y)
{
    return x - y;
}

std::pair<raul::dtype, raul::dtype> golden_sub_layer_grad(const raul::dtype grad)
{
    const auto x_grad = grad * 1;
    const auto y_grad = grad * (-1);
    return std::make_pair(x_grad, y_grad);
}
}

TEST(TestLayerElementWiseSub, OutputNumExceedsUnit)
{
    PROFILE_TEST

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    // Apply function
    ASSERT_THROW(raul::ElementWiseSubLayer("sub", raul::ElementWiseLayerParams{ { { "x", "y" }, { "x_out", "y_out" } } }, networkParameters), raul::Exception);
}

TEST(TestLayerElementWiseSub, InputNumExceedsUnit)
{
    PROFILE_TEST

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    // Apply function
    ASSERT_THROW(raul::ElementWiseSubLayer("sub", raul::ElementWiseLayerParams{ { "x", "y", "z" }, { "out" } }, networkParameters), raul::Exception);
}

TEST(TestLayerElementWiseSub, ForwardSizeMismatchUnit)
{
    PROFILE_TEST
    // Test parameters
    const auto tensor_size = 1000U;
    const auto random_range = std::make_pair(-100.0_dt, 100.0_dt);
    const auto enable_broadcast = false;

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    work.tensorNeeded("x", "x", raul::WShape{ raul::BS(), 1u, 1u, 1u }, DEC_FORW_READ_NOMEMOPT);
    work.tensorNeeded("y", "y", raul::WShape{ raul::BS(), 2u, 2u, 2u }, DEC_FORW_READ_NOMEMOPT);

    // Apply function
    raul::ElementWiseSubLayer elementwise_sub("sub", raul::ElementWiseLayerParams{ { "x", "y" }, { "out" }, enable_broadcast }, networkParameters);
    TENSORS_CREATE(tensor_size);

    tools::init_rand_tensor("x", random_range, memory_manager);
    tools::init_rand_tensor("y", random_range, memory_manager);

    ASSERT_THROW(elementwise_sub.forwardCompute(raul::NetworkMode::Test), raul::Exception);
}

TEST(TestLayerElementWiseSub, ForwardRandUnit)
{
    PROFILE_TEST
    // Test parameters
    const auto eps_rel = TODTYPE(1e-6);
    const auto tensor_size = 1000U;
    const auto random_range = std::make_pair(-100.0_dt, 100.0_dt);
    const auto enable_broadcast = false;

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    work.tensorNeeded("x", "x", raul::WShape{ raul::BS(), 1u, 1u, 1u }, DEC_FORW_READ_NOMEMOPT);
    work.tensorNeeded("y", "y", raul::WShape{ raul::BS(), 1u, 1u, 1u }, DEC_FORW_READ_NOMEMOPT);

    // Apply function
    raul::ElementWiseSubLayer elementwise_sub("sub", raul::ElementWiseLayerParams{ { "x", "y" }, { "out" }, enable_broadcast }, networkParameters);
    TENSORS_CREATE(tensor_size);
    tools::init_rand_tensor("x", random_range, memory_manager);
    tools::init_rand_tensor("y", random_range, memory_manager);

    elementwise_sub.forwardCompute(raul::NetworkMode::Test);

    // Checks
    const auto& x_tensor = memory_manager["x"];
    const auto& y_tensor = memory_manager["y"];
    const auto& out_tensor = memory_manager["out"];

    EXPECT_EQ(out_tensor.size(), x_tensor.size());
    EXPECT_EQ(out_tensor.size(), y_tensor.size());

    for (size_t i = 0; i < out_tensor.size(); ++i)
    {
        const auto x_value = x_tensor[i];
        const auto y_value = y_tensor[i];
        const auto out_value = out_tensor[i];
        const auto golden_out_value = golden_sub_layer(x_value, y_value);
        ASSERT_TRUE(tools::expect_near_relative(out_value, golden_out_value, eps_rel));
    }
}

TEST(TestLayerElementWiseSub, BackwardRandUnit)
{
    PROFILE_TEST
    // Test parameters
    const auto eps_rel = TODTYPE(1e-6);
    const auto tensor_size = 1000U;
    const auto random_range = std::make_pair(-100.0_dt, 100.0_dt);
    const auto enable_broadcast = false;

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    work.tensorNeeded("x", "x", raul::WShape{ raul::BS(), 1u, 1u, 1u }, DEC_FORW_READ_NOMEMOPT);
    work.tensorNeeded("y", "y", raul::WShape{ raul::BS(), 1u, 1u, 1u }, DEC_FORW_READ_NOMEMOPT);

    // Apply function
    raul::ElementWiseSubLayer elementwise_sub("sub", raul::ElementWiseLayerParams{ { "x", "y" }, { "out" }, enable_broadcast }, networkParameters);
    TENSORS_CREATE(tensor_size);
    tools::init_rand_tensor("x", random_range, memory_manager);
    tools::init_rand_tensor("y", random_range, memory_manager);
    tools::init_rand_tensor(raul::Name("out").grad(), random_range, memory_manager);

    elementwise_sub.forwardCompute(raul::NetworkMode::Train);
    elementwise_sub.backwardCompute();

    // Checks
    const auto& x_tensor_grad = memory_manager[raul::Name("x").grad()];
    const auto& y_tensor_grad = memory_manager[raul::Name("y").grad()];
    const auto& out_tensor_grad = memory_manager[raul::Name("out").grad()];

    EXPECT_EQ(out_tensor_grad.size(), x_tensor_grad.size());
    EXPECT_EQ(out_tensor_grad.size(), y_tensor_grad.size());

    for (size_t i = 0; i < out_tensor_grad.size(); ++i)
    {
        const auto out_grad_value = out_tensor_grad[i];
        const auto x_grad_value = x_tensor_grad[i];
        const auto y_grad_value = y_tensor_grad[i];
        const auto [golden_out_value_x, golden_out_value_y] = golden_sub_layer_grad(out_grad_value);
        ASSERT_TRUE(tools::expect_near_relative(x_grad_value, golden_out_value_x, eps_rel));
        ASSERT_TRUE(tools::expect_near_relative(y_grad_value, golden_out_value_y, eps_rel));
    }
}

TEST(TestLayerElementWiseSub, BroadcastForwardRandUnit)
{
    PROFILE_TEST
    // Test parameters
    const auto eps_rel = TODTYPE(1e-6);
    const auto random_range = std::make_pair(-100.0_dt, 100.0_dt);
    const auto enable_broadcast = true;

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    work.tensorNeeded("x", "x", raul::WShape{ raul::BS(), 1u, 1u, 1u }, DEC_FORW_READ_NOMEMOPT);
    work.tensorNeeded("y", "y", raul::WShape{ raul::BS(), 1u, 1u, 1u }, DEC_FORW_READ_NOMEMOPT);

    // Apply function
    raul::ElementWiseSubLayer elementwise_sub("sub", raul::ElementWiseLayerParams{ { "x", "y" }, { "out" }, enable_broadcast }, networkParameters);
    TENSORS_CREATE(1);
    tools::init_rand_tensor("x", random_range, memory_manager);
    tools::init_rand_tensor("y", random_range, memory_manager);

    elementwise_sub.forwardCompute(raul::NetworkMode::Test);

    // Checks
    const auto& x_tensor = memory_manager["x"];
    const auto& y_tensor = memory_manager["y"];
    const auto& out_tensor = memory_manager["out"];

    EXPECT_EQ(out_tensor.size(), y_tensor.size());

    for (size_t i = 0; i < out_tensor.size(); ++i)
    {
        const auto x_value = x_tensor[0];
        const auto y_value = y_tensor[i];
        const auto out_value = out_tensor[i];
        const auto golden_out_value = golden_sub_layer(x_value, y_value);
        ASSERT_TRUE(tools::expect_near_relative(out_value, golden_out_value, eps_rel));
    }
}

TEST(TestLayerElementWiseSub, BroadcastBackwardRandUnit)
{
    PROFILE_TEST
    // Test parameters
    constexpr auto eps_rel = TODTYPE(1e-6);
    const auto random_range = std::make_pair(-100.0_dt, 100.0_dt);
    const auto enable_broadcast = true;

    const raul::Tensor deltas{ 1._dt, 2._dt, 3._dt, 4._dt };
    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    work.tensorNeeded("x", "x", raul::WShape{ raul::BS(), 2u, 2u, 1u }, DEC_FORW_READ_NOMEMOPT);
    work.tensorNeeded("y", "y", raul::WShape{ raul::BS(), 1u, 1u, 1u }, DEC_FORW_READ_NOMEMOPT);

    // Apply function
    raul::ElementWiseSubLayer elementwise_sub("sub", raul::ElementWiseLayerParams{ { "x", "y" }, { "out" }, enable_broadcast }, networkParameters);
    TENSORS_CREATE(1);
    tools::init_rand_tensor("x", random_range, memory_manager);
    tools::init_rand_tensor("y", random_range, memory_manager);
    memory_manager[raul::Name("out").grad()] = TORANGE(deltas);

    elementwise_sub.forwardCompute(raul::NetworkMode::Train);
    elementwise_sub.backwardCompute();

    // Checks
    const auto& out_tensor_grad = memory_manager[raul::Name("out").grad()];
    const auto& x_tensor_grad = memory_manager[raul::Name("x").grad()];
    const auto& y_tensor_grad = memory_manager[raul::Name("y").grad()];

    for (size_t i = 0; i < out_tensor_grad.size(); ++i)
    {
        const auto out_grad_value = out_tensor_grad[i];
        const auto x_grad_value = x_tensor_grad[i];
        const auto [golden_out_value_x, golden_out_value_y] = golden_sub_layer_grad(out_grad_value);
        ASSERT_TRUE(tools::expect_near_relative(x_grad_value, golden_out_value_x, eps_rel));
    }

    auto golden_out_value = 0.0_dt;
    for (size_t i = 0; i < out_tensor_grad.size(); ++i)
    {
        const auto out_grad_value = out_tensor_grad[i];
        const auto [golden_out_value_x, golden_out_value_y] = golden_sub_layer_grad(out_grad_value);
        golden_out_value += golden_out_value_y;
    }
    ASSERT_TRUE(tools::expect_near_relative(y_tensor_grad[0], golden_out_value, eps_rel));
}

} // UT namespace