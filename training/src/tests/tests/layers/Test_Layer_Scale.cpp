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

#include <training/base/common/MemoryManager.h>
#include <training/base/initializers/ConstantInitializer.h>
#include <training/base/layers/basic/DataLayer.h>
#include <training/base/layers/basic/ScaleLayer.h>
#include <training/compiler/Workflow.h>

namespace UT
{

using namespace raul;

namespace
{

dtype golden_scale_layer(const dtype x, const dtype scale)
{
    return scale * x;
}

dtype golden_scale_layer_grad(const dtype grad, const dtype scale)
{
    return grad * scale;
}

}

TEST(TestLayerScale, InputNumExceedsUnit)
{
    PROFILE_TEST
    // Test parameters
    const auto scale = 2.0_dt;
    const auto params = ScaleParams{ { "x", "y" }, { "x_out" }, scale };

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    // Apply function
    work.add<DataLayer>("input", DataParams{ { "x", "y" }, 1, 1, 1 });
    ASSERT_THROW(ScaleLayer("twice", params, networkParameters), raul::Exception);
}

TEST(TestLayerScale, OutputNumExceedsUnit)
{
    PROFILE_TEST
    // Test parameters
    const auto scale = 2.0_dt;
    const auto params = ScaleParams{ { "x" }, { "x_out", "y_out" }, scale };

    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);
    work.add<DataLayer>("input", DataParams{ { "x" }, 1, 1, 1 });

    // Apply function
    ASSERT_THROW(ScaleLayer("twice", params, networkParameters), raul::Exception);
}

TEST(TestLayerScale, TwiceForwardUnit)
{
    PROFILE_TEST
    // Test parameters
    const auto eps_rel = TODTYPE(1e-6);
    const auto tensor_size = 1000U;
    const auto scale = 2.0_dt;
    const auto params = ScaleParams{ { "x" }, { "out" }, scale };

    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    // Apply function
    work.add<DataLayer>("input", DataParams{ { "x" }, 1, 1, 1 });
    ScaleLayer twice("twice", params, networkParameters);
    TENSORS_CREATE(tensor_size);
    initializers::ConstantInitializer initializer{ 1.0_dt };
    initializer(memory_manager["x"]);
    twice.forwardCompute(NetworkMode::Test);

    // Checks
    const auto& x_tensor = memory_manager["x"];
    const auto& out_tensor = memory_manager["out"];

    EXPECT_EQ(out_tensor.size(), x_tensor.size());
    for (size_t i = 0; i < out_tensor.size(); ++i)
    {
        const auto x_value = x_tensor[i];
        const auto out_value = out_tensor[i];
        const auto golden_out_value = golden_scale_layer(x_value, scale);
        ASSERT_TRUE(tools::expect_near_relative(out_value, golden_out_value, eps_rel));
    }
}

TEST(TestLayerScale, ForwardRandUnit)
{
    PROFILE_TEST
    // Test parameters
    const auto eps_rel = 1e-6_dt;
    const auto tensor_size = random::uniform::rand<int>(1, 1000);
    const auto scale = random::uniform::rand<raul::dtype>(-100., 100.);

    std::cout << "Run test with scale=" << scale << " and tensor shape (" << tensor_size << ",1,1,1)" << std::endl;

    const auto params = ScaleParams{ { "x" }, { "out" }, scale };

    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    // Apply function
    work.add<DataLayer>("input", DataParams{ { "x" }, 1, 1, 1 });
    ScaleLayer scaler("scaler", params, networkParameters);
    TENSORS_CREATE(tensor_size);
    initializers::ConstantInitializer initializer{ 1.0_dt };
    initializer(memory_manager["x"]);
    scaler.forwardCompute(NetworkMode::Test);

    // Checks
    const auto& x_tensor = memory_manager["x"];
    const auto& out_tensor = memory_manager["out"];

    EXPECT_EQ(out_tensor.size(), x_tensor.size());
    for (size_t i = 0; i < out_tensor.size(); ++i)
    {
        const auto x_value = x_tensor[i];
        const auto out_value = out_tensor[i];
        const auto golden_out_value = golden_scale_layer(x_value, scale);
        ASSERT_TRUE(tools::expect_near_relative(out_value, golden_out_value, eps_rel));
    }
}

TEST(TestLayerScale, BackwardRandUnit)
{
    PROFILE_TEST
    // Test parameters
    const auto eps_rel = 1e-6_dt;
    const auto tensor_size = random::uniform::rand<int>(1, 1000);
    const auto scale = random::uniform::rand<raul::dtype>(-100., 100.);

    std::cout << "Run test with scale=" << scale << " and tensor shape (" << tensor_size << ",1,1,1)" << std::endl;

    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    const auto params = ScaleParams{ { "x" }, { "out" }, scale };

    // Apply function
    work.add<DataLayer>("input", DataParams{ { "x" }, 1, 1, 1 });
    ScaleLayer scaler("scaler", params, networkParameters);
    TENSORS_CREATE(tensor_size);
    tools::init_rand_tensor("x", { -100.0_dt, 100.0_dt }, memory_manager);
    scaler.forwardCompute(NetworkMode::Train);
    memory_manager[Name("out").grad()].memAllocate(nullptr);
    tools::init_rand_tensor(Name("out").grad(), { -100.0_dt, 100.0_dt }, memory_manager);
    scaler.backwardCompute();

    // Checks
    const auto& x_tensor_grad = memory_manager[Name("x").grad()];
    const auto& out_tensor_grad = memory_manager[Name("out").grad()];

    EXPECT_EQ(out_tensor_grad.size(), x_tensor_grad.size());

    for (size_t i = 0; i < out_tensor_grad.size(); ++i)
    {
        const auto out_grad_value = out_tensor_grad[i];
        const auto x_grad_value = x_tensor_grad[i];
        const auto golden_out_value_x = golden_scale_layer_grad(out_grad_value, scale);
        ASSERT_TRUE(tools::expect_near_relative(x_grad_value, golden_out_value_x, eps_rel)) << "expected: " << golden_out_value_x << ", got: " << x_grad_value;
    }
}

}