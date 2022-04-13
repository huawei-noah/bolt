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
#include <training/base/layers/basic/SelectLayer.h>
#include <training/compiler/Workflow.h>

namespace UT
{

namespace
{

raul::dtype golden_select_layer(const raul::dtype cond, const raul::dtype x, const raul::dtype y)
{
    return static_cast<bool>(cond) == true ? x : y;
}

std::pair<raul::dtype, raul::dtype> golden_select_layer_grad(const raul::dtype cond, const raul::dtype grad)
{
    return std::make_pair(grad * cond, grad * (1.0_dt - cond));
}

}

TEST(TestLayerSelect, OutputNumExceedsUnit)
{
    PROFILE_TEST

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    // Apply function
    ASSERT_THROW(raul::SelectLayer("select", raul::ElementWiseLayerParams{ { { "cond", "x", "y" }, { "x_out", "y_out" } } }, networkParameters), raul::Exception);
}

TEST(TestLayerSelect, IncorrectInputNumUnit)
{
    PROFILE_TEST

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    // Apply function
    ASSERT_THROW(raul::SelectLayer("select", raul::ElementWiseLayerParams{ { { "x", "y" }, { "out" } } }, networkParameters), raul::Exception);
}

TEST(TestLayerSelect, ForwardRandUnit)
{
    PROFILE_TEST
    // Test parameters
    const auto tensor_size = 30U;
    const auto random_range = std::make_pair(-100.0_dt, 100.0_dt);

    const raul::Tensor cond{ 0.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 1.0_dt, 1.0_dt,
                             0.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 0.0_dt, 0.0_dt };

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    work.tensorNeeded("cond", "cond", raul::WShape{ raul::BS(), 1u, 1u, 1u }, DEC_FORW_READ_NOMEMOPT);
    work.tensorNeeded("x", "x", raul::WShape{ raul::BS(), 1u, 1u, 1u }, DEC_FORW_READ_NOMEMOPT);
    work.tensorNeeded("y", "y", raul::WShape{ raul::BS(), 1u, 1u, 1u }, DEC_FORW_READ_NOMEMOPT);

    // Apply function
    raul::SelectLayer select("select", raul::ElementWiseLayerParams{ { "cond", "x", "y" }, { "out" } }, networkParameters);
    TENSORS_CREATE(tensor_size);
    tools::init_rand_tensor("x", random_range, memory_manager);
    tools::init_rand_tensor("y", random_range, memory_manager);
    memory_manager["cond"] = TORANGE(cond);

    select.forwardCompute(raul::NetworkMode::Test);

    // Checks
    const auto& cond_tensor = memory_manager["cond"];
    const auto& x_tensor = memory_manager["x"];
    const auto& y_tensor = memory_manager["y"];
    const auto& out_tensor = memory_manager["out"];

    EXPECT_EQ(out_tensor.size(), x_tensor.size());
    EXPECT_EQ(out_tensor.size(), y_tensor.size());

    for (size_t i = 0; i < out_tensor.size(); ++i)
    {
        const auto golden_out_value = golden_select_layer(cond_tensor[i], x_tensor[i], y_tensor[i]);
        EXPECT_EQ(golden_out_value, out_tensor[i]);
    }
}

TEST(TestLayerSelect, BackwardRandUnit)
{
    PROFILE_TEST
    // Test parameters
    const auto tensor_size = 30U;
    const auto random_range = std::make_pair(-100.0_dt, 100.0_dt);

    const raul::Tensor cond{ 0.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 1.0_dt, 1.0_dt,
                             0.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 0.0_dt, 0.0_dt };

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    work.tensorNeeded("cond", "cond", raul::WShape{ raul::BS(), 1u, 1u, 1u }, DEC_FORW_READ_NOMEMOPT);
    work.tensorNeeded("x", "x", raul::WShape{ raul::BS(), 1u, 1u, 1u }, DEC_FORW_READ_NOMEMOPT);
    work.tensorNeeded("y", "y", raul::WShape{ raul::BS(), 1u, 1u, 1u }, DEC_FORW_READ_NOMEMOPT);

    // Apply function
    raul::SelectLayer select("select", raul::ElementWiseLayerParams{ { "cond", "x", "y" }, { "out" } }, networkParameters);
    TENSORS_CREATE(tensor_size);
    tools::init_rand_tensor("x", random_range, memory_manager);
    tools::init_rand_tensor("y", random_range, memory_manager);
    tools::init_rand_tensor(raul::Name("out").grad(), random_range, memory_manager);
    memory_manager["cond"] = TORANGE(cond);

    select.forwardCompute(raul::NetworkMode::Test);
    select.backwardCompute();

    // Checks
    const auto& cond_tensor = memory_manager["cond"];
    const auto& x_nabla_tensor = memory_manager[raul::Name("x").grad()];
    const auto& y_nabla_tensor = memory_manager[raul::Name("y").grad()];
    const auto& out_tensor_grad = memory_manager[raul::Name("out").grad()];

    for (size_t i = 0; i < x_nabla_tensor.size(); ++i)
    {
        const auto [golden_x_grad_value, golden_y_grad_value] = golden_select_layer_grad(cond_tensor[i], out_tensor_grad[i]);
        EXPECT_EQ(golden_x_grad_value, x_nabla_tensor[i]);
        EXPECT_EQ(golden_y_grad_value, y_nabla_tensor[i]);
    }
}

TEST(TestLayerSelect, BroadcastForwardUnit)
{
    PROFILE_TEST
    // Test parameters
    const bool enable_broadcast = true;
    const auto shape = yato::dims(1, 3, 4, 5);
    const size_t batch_size = 1;

    const raul::Tensor x{ 0.4963_dt, 0.7682_dt, 0.0885_dt, 0.1320_dt, 0.3074_dt, 0.6341_dt, 0.4901_dt, 0.8964_dt, 0.4556_dt, 0.6323_dt, 0.3489_dt, 0.4017_dt, 0.0223_dt, 0.1689_dt, 0.2939_dt };

    const raul::Tensor y{ 0.3051_dt, 0.9320_dt, 0.1759_dt, 0.2698_dt, 0.1507_dt, 0.0317_dt, 0.2081_dt, 0.9298_dt, 0.7231_dt, 0.7423_dt,
                          0.5263_dt, 0.2437_dt, 0.5846_dt, 0.0332_dt, 0.1387_dt, 0.2422_dt, 0.8155_dt, 0.7932_dt, 0.2783_dt, 0.4820_dt };

    const raul::Tensor cond{ 0.0_dt, 0.0_dt, 1.0_dt, 0.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 0.0_dt, 1.0_dt, 1.0_dt, 0.0_dt };

    const raul::Tensor realOutput{
        0.3051_dt, 0.9320_dt, 0.1759_dt, 0.2698_dt, 0.1507_dt, 0.0317_dt, 0.2081_dt, 0.9298_dt, 0.7231_dt, 0.7423_dt, 0.4963_dt, 0.7682_dt, 0.0885_dt, 0.1320_dt, 0.3074_dt,
        0.2422_dt, 0.8155_dt, 0.7932_dt, 0.2783_dt, 0.4820_dt, 0.6341_dt, 0.4901_dt, 0.8964_dt, 0.4556_dt, 0.6323_dt, 0.0317_dt, 0.2081_dt, 0.9298_dt, 0.7231_dt, 0.7423_dt,
        0.5263_dt, 0.2437_dt, 0.5846_dt, 0.0332_dt, 0.1387_dt, 0.6341_dt, 0.4901_dt, 0.8964_dt, 0.4556_dt, 0.6323_dt, 0.3051_dt, 0.9320_dt, 0.1759_dt, 0.2698_dt, 0.1507_dt,
        0.3489_dt, 0.4017_dt, 0.0223_dt, 0.1689_dt, 0.2939_dt, 0.3489_dt, 0.4017_dt, 0.0223_dt, 0.1689_dt, 0.2939_dt, 0.2422_dt, 0.8155_dt, 0.7932_dt, 0.2783_dt, 0.4820_dt
    };

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    work.tensorNeeded("cond", "cond", raul::WShape{ raul::BS(), 3u, 4u, 1u }, DEC_FORW_READ_NOMEMOPT);
    work.tensorNeeded("x", "x", raul::WShape{ raul::BS(), 3u, 1u, 5u }, DEC_FORW_READ_NOMEMOPT);
    work.tensorNeeded("y", "y", raul::WShape{ raul::BS(), 1u, 4u, 5u }, DEC_FORW_READ_NOMEMOPT);

    // Apply function
    raul::SelectLayer select("select", raul::ElementWiseLayerParams{ { "cond", "x", "y" }, { "out" }, enable_broadcast }, networkParameters);
    TENSORS_CREATE(batch_size);
    memory_manager["cond"] = TORANGE(cond);
    memory_manager["x"] = TORANGE(x);
    memory_manager["y"] = TORANGE(y);

    select.forwardCompute(raul::NetworkMode::Test);

    // Checks
    const auto& out_tensor = memory_manager["out"];
    EXPECT_EQ(out_tensor.getShape(), shape);
    for (size_t i = 0; i < out_tensor.size(); ++i)
    {
        EXPECT_EQ(realOutput[i], out_tensor[i]);
    }
}

TEST(TestLayerSelect, BroadcastBackwardUnit)
{
    PROFILE_TEST
    // Test parameters
    const bool enable_broadcast = true;
    const size_t batch_size = 1;

    const raul::Tensor x{ 0.4963_dt, 0.7682_dt, 0.0885_dt, 0.1320_dt, 0.3074_dt, 0.6341_dt, 0.4901_dt, 0.8964_dt, 0.4556_dt, 0.6323_dt, 0.3489_dt, 0.4017_dt, 0.0223_dt, 0.1689_dt, 0.2939_dt };

    const raul::Tensor y{ 0.3051_dt, 0.9320_dt, 0.1759_dt, 0.2698_dt, 0.1507_dt, 0.0317_dt, 0.2081_dt, 0.9298_dt, 0.7231_dt, 0.7423_dt,
                          0.5263_dt, 0.2437_dt, 0.5846_dt, 0.0332_dt, 0.1387_dt, 0.2422_dt, 0.8155_dt, 0.7932_dt, 0.2783_dt, 0.4820_dt };

    const raul::Tensor cond{ 0.0_dt, 0.0_dt, 1.0_dt, 0.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 0.0_dt, 1.0_dt, 1.0_dt, 0.0_dt };

    const raul::Tensor deltas{ 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt,
                               1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt,
                               1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt };
    const raul::Tensor x_grad{ 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 2.0_dt, 2.0_dt, 2.0_dt, 2.0_dt, 2.0_dt, 2.0_dt, 2.0_dt, 2.0_dt, 2.0_dt, 2.0_dt };
    const raul::Tensor y_grad{ 2.0_dt, 2.0_dt, 2.0_dt, 2.0_dt, 2.0_dt, 2.0_dt, 2.0_dt, 2.0_dt, 2.0_dt, 2.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 2.0_dt, 2.0_dt, 2.0_dt, 2.0_dt, 2.0_dt };
    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    work.tensorNeeded("cond", "cond", raul::WShape{ raul::BS(), 3u, 4u, 1u }, DEC_FORW_READ_NOMEMOPT);
    work.tensorNeeded("x", "x", raul::WShape{ raul::BS(), 3u, 1u, 5u }, DEC_FORW_READ_NOMEMOPT);
    work.tensorNeeded("y", "y", raul::WShape{ raul::BS(), 1u, 4u, 5u }, DEC_FORW_READ_NOMEMOPT);

    // Apply function
    raul::SelectLayer select("select", raul::ElementWiseLayerParams{ { "cond", "x", "y" }, { "out" }, enable_broadcast }, networkParameters);
    TENSORS_CREATE(batch_size);
    memory_manager["cond"] = TORANGE(cond);
    memory_manager["x"] = TORANGE(x);
    memory_manager["y"] = TORANGE(y);
    memory_manager[raul::Name("out").grad()] = TORANGE(deltas);

    select.forwardCompute(raul::NetworkMode::Test);
    select.backwardCompute();

    // Checks
    const auto& x_nabla_tensor = memory_manager[raul::Name("x").grad()];
    const auto& y_nabla_tensor = memory_manager[raul::Name("y").grad()];
    EXPECT_EQ(x_nabla_tensor.size(), x.size());
    EXPECT_EQ(y_nabla_tensor.size(), y.size());
    for (size_t i = 0; i < x_nabla_tensor.size(); ++i)
    {
        EXPECT_EQ(x_nabla_tensor[i], x_grad[i]);
    }
    for (size_t i = 0; i < y_nabla_tensor.size(); ++i)
    {
        EXPECT_EQ(y_nabla_tensor[i], y_grad[i]);
    }
}

}