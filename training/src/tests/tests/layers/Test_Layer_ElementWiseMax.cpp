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

#include <training/base/layers/basic/DataLayer.h>
#include <training/base/layers/basic/ElementWiseMaxLayer.h>
#include <training/compiler/Workflow.h>
#include <training/system/NameGenerator.h>

namespace UT
{

namespace
{

raul::dtype golden_max_layer(const raul::dtype x, const raul::dtype y)
{
    return std::max<raul::dtype>(x, y);
}

std::pair<raul::dtype, raul::dtype> golden_max_layer_grad(const raul::dtype x, const raul::dtype y, const raul::dtype grad)
{
    return std::make_pair((x > y) * grad, (x <= y) * grad);
}

}

TEST(TestLayerElementWiseMax, OutputNumExceedsUnit)
{
    PROFILE_TEST

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    // Apply function
    ASSERT_THROW(raul::ElementWiseMaxLayer("max", raul::ElementWiseLayerParams{ { { "x", "y" }, { "x_out", "y_out" } } }, networkParameters), raul::Exception);
}

TEST(TestLayerElementWiseMax, NoBroadcastForwardFailUnit)
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
    work.tensorNeeded("y", "y", raul::WShape{ raul::BS(), 2u, 1u, 1u }, DEC_FORW_READ_NOMEMOPT);

    // Apply function
    raul::ElementWiseMaxLayer elementwise_max("max", raul::ElementWiseLayerParams{ { "x", "y" }, { "out" }, enable_broadcast }, networkParameters);
    TENSORS_CREATE(tensor_size);
    tools::init_rand_tensor("x", random_range, memory_manager);
    tools::init_rand_tensor("y", random_range, memory_manager);

    ASSERT_THROW(elementwise_max.forwardCompute(raul::NetworkMode::Test), raul::Exception);
}

TEST(TestLayerElementWiseMax, ForwardRandUnit)
{
    PROFILE_TEST
    // Test parameters
    const auto tensor_size = 1000U;
    const auto random_range = std::make_pair(-100.0_dt, 100.0_dt);

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    work.tensorNeeded("x", "x", raul::WShape{ raul::BS(), 1u, 1u, 1u }, DEC_FORW_READ_NOMEMOPT);
    work.tensorNeeded("y", "y", raul::WShape{ raul::BS(), 1u, 1u, 1u }, DEC_FORW_READ_NOMEMOPT);

    // Apply function
    raul::ElementWiseMaxLayer elementwise_max("max", raul::ElementWiseLayerParams{ { "x", "y" }, { "out" } }, networkParameters);
    TENSORS_CREATE(tensor_size);
    tools::init_rand_tensor("x", random_range, memory_manager);
    tools::init_rand_tensor("y", random_range, memory_manager);

    elementwise_max.forwardCompute(raul::NetworkMode::Test);

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
        const auto golden_out_value = golden_max_layer(x_value, y_value);
        EXPECT_EQ(out_value, golden_out_value);
    }
}

TEST(TestLayerElementWiseMax, BackwardRandUnit)
{
    PROFILE_TEST
    // Test parameters
    const auto eps_rel = 1e-6_dt;
    const auto tensor_size = 1000U;
    const auto random_range = std::make_pair(-100.0_dt, 100.0_dt);

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    work.tensorNeeded("x", "x", raul::WShape{ raul::BS(), 1u, 1u, 1u }, DEC_FORW_READ_NOMEMOPT);
    work.tensorNeeded("y", "y", raul::WShape{ raul::BS(), 1u, 1u, 1u }, DEC_FORW_READ_NOMEMOPT);

    // Apply function
    raul::ElementWiseMaxLayer elementwise_max("max", raul::ElementWiseLayerParams{ { "x", "y" }, { "out" } }, networkParameters);
    TENSORS_CREATE(tensor_size);
    tools::init_rand_tensor("x", random_range, memory_manager);
    tools::init_rand_tensor("y", random_range, memory_manager);
    tools::init_rand_tensor(raul::Name("out").grad(), random_range, memory_manager);

    elementwise_max.forwardCompute(raul::NetworkMode::Train);
    elementwise_max.backwardCompute();

    // Checks
    const auto& x_tensor = memory_manager["x"];
    const auto& y_tensor = memory_manager["y"];
    const auto& x_tensor_grad = memory_manager[raul::Name("x").grad()];
    const auto& y_tensor_grad = memory_manager[raul::Name("y").grad()];
    const auto& out_tensor_grad = memory_manager[raul::Name("out").grad()];

    EXPECT_EQ(out_tensor_grad.size(), x_tensor_grad.size());
    EXPECT_EQ(out_tensor_grad.size(), y_tensor_grad.size());

    for (size_t i = 0; i < out_tensor_grad.size(); ++i)
    {
        const auto x_value = x_tensor[i];
        const auto y_value = y_tensor[i];
        const auto out_grad_value = out_tensor_grad[i];
        const auto x_grad_value = x_tensor_grad[i];
        const auto y_grad_value = y_tensor_grad[i];
        const auto [golden_out_value_x, golden_out_value_y] = golden_max_layer_grad(x_value, y_value, out_grad_value);
        ASSERT_TRUE(tools::expect_near_relative(x_grad_value, golden_out_value_x, eps_rel));
        ASSERT_TRUE(tools::expect_near_relative(y_grad_value, golden_out_value_y, eps_rel));
    }
}

TEST(TestLayerElementWiseMax, MultipleRandUnit)
{
    PROFILE_TEST
    // Test parameters
    const auto eps_rel = 1e-6_dt;
    const auto tensor_amount = 10U;
    const auto tensor_size = 10U;
    const auto random_range = std::make_pair(-10.0_dt, 10.0_dt);

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    auto name_gen = raul::NameGenerator("factor");
    raul::Names factor_tensor_names(tensor_amount);
    std::generate(factor_tensor_names.begin(), factor_tensor_names.end(), [&]() { return name_gen.generate(); });

    // Create and initialize tensors with random data
    for (const auto& tensor_name : factor_tensor_names)
    {
        work.tensorNeeded("x", tensor_name, raul::WShape{ raul::BS(), 1u, 1u, 1u }, DEC_FORW_READ_NOMEMOPT);
    }

    // Apply function
    raul::ElementWiseMaxLayer elementwise_max("max", raul::ElementWiseLayerParams{ factor_tensor_names, { "out" } }, networkParameters);
    TENSORS_CREATE(tensor_size);
    for (const auto& tensor_name : factor_tensor_names)
    {
        tools::init_rand_tensor(tensor_name, random_range, memory_manager);
    }
    tools::init_rand_tensor(raul::Name("out").grad(), random_range, memory_manager);

    elementwise_max.forwardCompute(raul::NetworkMode::Train);
    elementwise_max.backwardCompute();

    // Check sizes
    const auto& out_tensor = memory_manager["out"];
    const auto& out_tensor_grad = memory_manager[raul::Name("out").grad()];
    for (const auto& tensor_name : factor_tensor_names)
    {
        const auto& tensor = memory_manager[tensor_name];
        const auto& tensor_grad = memory_manager[tensor_name.grad()];
        EXPECT_EQ(out_tensor.size(), tensor.size());
        EXPECT_EQ(out_tensor_grad.size(), tensor_grad.size());
    }

    const auto skip_and_select_max = [&](const std::optional<size_t> skip_idx, const size_t axis) {
        auto out = -std::numeric_limits<raul::dtype>::infinity();
        for (size_t i = 0; i < factor_tensor_names.size(); ++i)
        {
            if (skip_idx && i == skip_idx.value()) continue;
            const auto tensor_name = factor_tensor_names[i];
            const auto& tensor = memory_manager[tensor_name];
            out = golden_max_layer(out, tensor[axis]);
        }
        return out;
    };

    // Check forward
    for (size_t i = 0; i < out_tensor.size(); ++i)
    {
        const auto golden_out = skip_and_select_max(std::nullopt, i);
        ASSERT_TRUE(tools::expect_near_relative(out_tensor[i], golden_out, eps_rel));
    }

    // Check backward
    for (size_t i = 0; i < out_tensor_grad.size(); ++i)
    {
        for (size_t j = 0; j < factor_tensor_names.size(); ++j)
        {
            const auto tensor_name = factor_tensor_names[j];
            const auto& tensor = memory_manager[tensor_name];
            const auto& tensor_grad = memory_manager[tensor_name.grad()];
            const auto skip_max_out_i = skip_and_select_max(j, i);
            const auto [_, tensor_golden_grad] = golden_max_layer_grad(skip_max_out_i, tensor[i], out_tensor_grad[i]);
            ASSERT_TRUE(tools::expect_near_relative(tensor_grad[i], tensor_golden_grad, eps_rel));
        }
    }
}

TEST(TestLayerElementWiseMax, BroadcastForwardRandUnit)
{
    PROFILE_TEST
    // Test parameters
    const auto random_range = std::make_pair(-100.0_dt, 100.0_dt);
    const auto enable_broadcast = true;

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    work.tensorNeeded("x", "x", raul::WShape{ raul::BS(), 1u, 1u, 1u }, DEC_FORW_READ_NOMEMOPT);
    work.tensorNeeded("y", "y", raul::WShape{ raul::BS(), 2u, 3u, 4u }, DEC_FORW_READ_NOMEMOPT);

    // Apply function
    raul::ElementWiseMaxLayer elementwise_max("max", raul::ElementWiseLayerParams{ { "x", "y" }, { "out" }, enable_broadcast }, networkParameters);
    TENSORS_CREATE(1);
    tools::init_rand_tensor("x", random_range, memory_manager);
    tools::init_rand_tensor("y", random_range, memory_manager);

    elementwise_max.forwardCompute(raul::NetworkMode::Test);

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
        const auto golden_out_value = golden_max_layer(x_value, y_value);
        EXPECT_EQ(out_value, golden_out_value);
    }
}

TEST(TestLayerElementWiseMax, BroadcastBackwardRandUnit)
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
    work.tensorNeeded("y", "y", raul::WShape{ raul::BS(), 2u, 3u, 4u }, DEC_FORW_READ_NOMEMOPT);

    // Apply function
    raul::ElementWiseMaxLayer elementwise_max("max", raul::ElementWiseLayerParams{ { "x", "y" }, { "out" }, enable_broadcast }, networkParameters);
    TENSORS_CREATE(1);
    tools::init_rand_tensor("x", random_range, memory_manager);
    tools::init_rand_tensor("y", random_range, memory_manager);
    tools::init_rand_tensor(raul::Name("out").grad(), random_range, memory_manager);

    elementwise_max.forwardCompute(raul::NetworkMode::Train);
    elementwise_max.backwardCompute();

    // Checks
    const auto& x_tensor = memory_manager["x"];
    const auto& y_tensor = memory_manager["y"];
    const auto& out_tensor = memory_manager["out"];

    const auto& out_tensor_grad = memory_manager[raul::Name("out").grad()];
    const auto& x_tensor_grad = memory_manager[raul::Name("x").grad()];
    const auto& y_tensor_grad = memory_manager[raul::Name("y").grad()];

    EXPECT_EQ(out_tensor.size(), y_tensor.size());

    auto golden_out_value = 0.0_dt;
    for (size_t i = 0; i < out_tensor_grad.size(); ++i)
    {
        const auto x_value = x_tensor[0];
        const auto y_value = y_tensor[i];
        const auto out_grad_value = out_tensor_grad[i];
        const auto y_grad_value = y_tensor_grad[i];
        const auto [golden_out_value_x, golden_out_value_y] = golden_max_layer_grad(x_value, y_value, out_grad_value);
        ASSERT_TRUE(tools::expect_near_relative(y_grad_value, golden_out_value_y, eps_rel));
        golden_out_value += golden_out_value_x;
    }
    ASSERT_TRUE(tools::expect_near_relative(x_tensor_grad[0], golden_out_value, eps_rel));
}

TEST(TestLayerElementWiseMax, BroadcastFuncUnit)
{
    PROFILE_TEST
    // Test parameters
    const auto enable_broadcast = true;
    const auto shape = yato::dims(3, 2, 2, 3);

    // See element-wise_min.ipynb (seed=42)
    const raul::Tensor x{ 0.4963_dt, 0.7682_dt, 0.0885_dt, 0.1320_dt, 0.3074_dt, 0.6341_dt };
    const raul::Tensor y{ 0.4901_dt, 0.8964_dt, 0.4556_dt, 0.6323_dt, 0.3489_dt, 0.4017_dt, 0.0223_dt, 0.1689_dt, 0.2939_dt,
                          0.5185_dt, 0.6977_dt, 0.8000_dt, 0.1610_dt, 0.2823_dt, 0.6816_dt, 0.9152_dt, 0.3971_dt, 0.8742_dt };
    const raul::Tensor z{ 0.4963_dt, 0.8964_dt, 0.4963_dt, 0.7682_dt, 0.8964_dt, 0.7682_dt, 0.6323_dt, 0.4963_dt, 0.4963_dt, 0.7682_dt, 0.7682_dt, 0.7682_dt,
                          0.0885_dt, 0.1689_dt, 0.2939_dt, 0.1320_dt, 0.1689_dt, 0.2939_dt, 0.5185_dt, 0.6977_dt, 0.8000_dt, 0.5185_dt, 0.6977_dt, 0.8000_dt,
                          0.3074_dt, 0.3074_dt, 0.6816_dt, 0.6341_dt, 0.6341_dt, 0.6816_dt, 0.9152_dt, 0.3971_dt, 0.8742_dt, 0.9152_dt, 0.6341_dt, 0.8742_dt };

    const raul::Tensor deltas{ 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt,
                               1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt };
    const raul::Tensor x_grad{ 4.0_dt, 5.0_dt, 1.0_dt, 1.0_dt, 2.0_dt, 3.0_dt };
    const raul::Tensor y_grad{ 0.0_dt, 2.0_dt, 0.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 2.0_dt, 2.0_dt, 2.0_dt, 2.0_dt, 2.0_dt, 0.0_dt, 0.0_dt, 2.0_dt, 2.0_dt, 1.0_dt, 2.0_dt };

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    work.tensorNeeded("x", "x", raul::WShape{ raul::BS(), 1u, 2u, 1u }, DEC_FORW_READ_NOMEMOPT);
    work.tensorNeeded("y", "y", raul::WShape{ raul::BS(), 2u, 1u, 3u }, DEC_FORW_READ_NOMEMOPT);

    // Apply function
    raul::ElementWiseMaxLayer elementwise_max("max", raul::ElementWiseLayerParams{ { "x", "y" }, { "out" }, enable_broadcast }, networkParameters);
    TENSORS_CREATE(3);
    memory_manager["x"] = TORANGE(x);
    memory_manager["y"] = TORANGE(y);
    memory_manager[raul::Name("out").grad()] = TORANGE(deltas);

    elementwise_max.forwardCompute(raul::NetworkMode::Test);
    elementwise_max.backwardCompute();

    // Forward checks
    const auto& out_tensor = memory_manager["out"];
    EXPECT_EQ(out_tensor.getShape(), shape);
    for (size_t i = 0; i < out_tensor.size(); ++i)
    {
        EXPECT_EQ(out_tensor[i], z[i]);
    }

    // Backward checks
    const auto& x_tensor_grad = memory_manager[raul::Name("x").grad()];
    const auto& y_tensor_grad = memory_manager[raul::Name("y").grad()];

    EXPECT_EQ(x_tensor_grad.getShape(), memory_manager["x"].getShape());
    EXPECT_EQ(y_tensor_grad.getShape(), memory_manager["y"].getShape());

    for (size_t i = 0; i < x_tensor_grad.size(); ++i)
    {
        EXPECT_EQ(x_tensor_grad[i], x_grad[i]);
    }
    for (size_t i = 0; i < y_tensor_grad.size(); ++i)
    {
        EXPECT_EQ(y_tensor_grad[i], y_grad[i]);
    }
}

TEST(TestLayerElementWiseMax, EqualValuesUnit)
{
    PROFILE_TEST
    // Test parameters
    const auto enable_broadcast = true;
    const auto shape = yato::dims(1, 1, 1, 5);

    // See element-wise_min.ipynb
    const raul::Tensor x{ -5.0_dt, -1.0_dt, 0.0_dt, 1.2_dt, 2.5_dt };
    const raul::Tensor y{ -5.0_dt, -1.0_dt, 0.0_dt, 1.2_dt, 2.5_dt };
    const raul::Tensor z{ -5.0_dt, -1.0_dt, 0.0_dt, 1.2_dt, 2.5_dt };
    const raul::Tensor deltas{ 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt };
    const raul::Tensor x_grad{ 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt };
    const raul::Tensor y_grad{ 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt };

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    work.tensorNeeded("x", "x", raul::WShape{ raul::BS(), 1u, 1u, 5u }, DEC_FORW_READ_NOMEMOPT);
    work.tensorNeeded("y", "y", raul::WShape{ raul::BS(), 1u, 1u, 5u }, DEC_FORW_READ_NOMEMOPT);

    // Apply function
    raul::ElementWiseMaxLayer elementwise_max("max", raul::ElementWiseLayerParams{ { "x", "y" }, { "out" }, enable_broadcast }, networkParameters);
    TENSORS_CREATE(1);
    memory_manager["x"] = TORANGE(x);
    memory_manager["y"] = TORANGE(y);
    memory_manager[raul::Name("out").grad()] = TORANGE(deltas);

    elementwise_max.forwardCompute(raul::NetworkMode::Test);
    elementwise_max.backwardCompute();

    // Forward checks
    const auto& out_tensor = memory_manager["out"];
    EXPECT_EQ(out_tensor.getShape(), shape);
    for (size_t i = 0; i < out_tensor.size(); ++i)
    {
        EXPECT_EQ(out_tensor[i], z[i]);
    }

    // Backward checks
    const auto& x_tensor_grad = memory_manager[raul::Name("x").grad()];
    const auto& y_tensor_grad = memory_manager[raul::Name("y").grad()];

    EXPECT_EQ(x_tensor_grad.getShape(), memory_manager["x"].getShape());
    EXPECT_EQ(y_tensor_grad.getShape(), memory_manager["y"].getShape());

    for (size_t i = 0; i < x_tensor_grad.size(); ++i)
    {
        EXPECT_EQ(x_tensor_grad[i], x_grad[i]);
    }
    for (size_t i = 0; i < y_tensor_grad.size(); ++i)
    {
        EXPECT_EQ(y_tensor_grad[i], y_grad[i]);
    }
}

}