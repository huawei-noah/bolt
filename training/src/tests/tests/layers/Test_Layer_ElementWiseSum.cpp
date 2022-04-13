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
#include <training/base/layers/basic/ElementWiseSumLayer.h>
#include <training/compiler/Workflow.h>
#include <training/system/NameGenerator.h>

namespace UT
{

namespace
{
raul::dtype golden_sum_layer(const raul::dtype x, const raul::dtype y)
{
    return x + y;
}

std::pair<raul::dtype, raul::dtype> golden_sum_layer_grad(const raul::dtype, const raul::dtype, const raul::dtype grad)
{
    const auto x_grad = grad;
    const auto y_grad = grad;
    return std::make_pair(x_grad, y_grad);
}
}

TEST(TestLayerElementWiseSum, BroadcastForwardRandUnit)
{
    PROFILE_TEST
    // Test parameters
    const auto eps_rel = 1e-6_dt;
    const auto random_range = std::make_pair(-100.0_dt, 100.0_dt);
    const auto enable_broadcast = true;

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    work.tensorNeeded("x", "x", raul::WShape{ raul::BS(), 1u, 1u, 1u }, DEC_FORW_READ_NOMEMOPT);
    work.tensorNeeded("y", "y", raul::WShape{ raul::BS(), 2u, 2u, 1u }, DEC_FORW_READ_NOMEMOPT);

    // Apply function
    raul::ElementWiseSumLayer elementwise_sum("sum", raul::ElementWiseLayerParams{ { "x", "y" }, { "out" }, enable_broadcast }, networkParameters);
    TENSORS_CREATE(1);
    tools::init_rand_tensor("x", random_range, memory_manager);
    tools::init_rand_tensor("y", random_range, memory_manager);

    elementwise_sum.forwardCompute(raul::NetworkMode::Test);

    // Checks
    const auto& x_tensor = memory_manager["x"];
    const auto& y_tensor = memory_manager["y"];
    const auto& out_tensor = memory_manager["out"];

    EXPECT_EQ(out_tensor.size(), y_tensor.size());

    for (size_t i = 0; i < out_tensor.size(); ++i)
    {
        // We have x tensor with shape [1,1] so there is only 1 value in x
        const auto x_value = x_tensor[0];
        const auto y_value = y_tensor[i];
        const auto out_value = out_tensor[i];
        const auto golden_out_value = golden_sum_layer(x_value, y_value);
        ASSERT_TRUE(tools::expect_near_relative(out_value, golden_out_value, eps_rel));
    }
}

TEST(TestLayerElementWiseSum, BroadcastBackwardRandUnit)
{
    PROFILE_TEST
    // Test parameters
    constexpr auto eps_rel = 1e-6_dt;
    const auto random_range = std::make_pair(-100.0_dt, 100.0_dt);
    const auto enable_broadcast = true;

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    work.tensorNeeded("x", "x", raul::WShape{ raul::BS(), 2u, 2u, 1u }, DEC_FORW_READ_NOMEMOPT);
    work.tensorNeeded("y", "y", raul::WShape{ raul::BS(), 1u, 1u, 1u }, DEC_FORW_READ_NOMEMOPT);

    // Apply function
    raul::ElementWiseSumLayer elementwise_sum("sum", raul::ElementWiseLayerParams{ { "x", "y" }, { "out" }, enable_broadcast }, networkParameters);
    TENSORS_CREATE(1);
    tools::init_rand_tensor("x", random_range, memory_manager);
    tools::init_rand_tensor("y", random_range, memory_manager);

    elementwise_sum.forwardCompute(raul::NetworkMode::Train);
    elementwise_sum.backwardCompute();

    // Checks
    const auto& x_tensor = memory_manager["x"];
    const auto& y_tensor = memory_manager["y"];

    const auto& out_tensor_grad = memory_manager[raul::Name("out").grad()];
    const auto& x_tensor_grad = memory_manager[raul::Name("x").grad()];
    const auto& y_tensor_grad = memory_manager[raul::Name("y").grad()];

    EXPECT_EQ(out_tensor_grad.size(), x_tensor.size());

    for (size_t i = 0; i < out_tensor_grad.size(); ++i)
    {
        const auto x_value = x_tensor[i];
        // We have y tensor with shape [1,1] so there is only 1 value in y
        const auto y_value = y_tensor[0];
        const auto out_grad_value = out_tensor_grad[i];
        const auto x_grad_value = x_tensor_grad[i];
        const auto [golden_out_value_x, golden_out_value_y] = golden_sum_layer_grad(x_value, y_value, out_grad_value);
        ASSERT_TRUE(tools::expect_near_relative(x_grad_value, golden_out_value_x, eps_rel));
    }

    auto golden_out_value = 0.0_dt;
    for (size_t i = 0; i < out_tensor_grad.size(); ++i)
    {
        const auto x_value = x_tensor[i];
        // We have y tensor with shape [1,1] so there is only 1 value in y
        const auto y_value = y_tensor[0];
        const auto out_grad_value = out_tensor_grad[i];
        const auto [golden_out_value_x, golden_out_value_y] = golden_sum_layer_grad(x_value, y_value, out_grad_value);
        golden_out_value += golden_out_value_y;
    }
    // We have y tensor with shape [1,1] so there is only 1 value in grad of y
    ASSERT_TRUE(tools::expect_near_relative(y_tensor_grad[0], golden_out_value, eps_rel));
}

} // UT namespace