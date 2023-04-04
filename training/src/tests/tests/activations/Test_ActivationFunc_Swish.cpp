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

#include <training/base/layers/activations/SwishActivation.h>
#include <training/compiler/Layers.h>
#include <training/compiler/Workflow.h>
#include <training/base/optimizers/SGD.h>

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
    const auto eps = 1e-5_dt;
    const auto tensor_size = 1000U;
    // We have a significant change in the range [-5,5]
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

} // UT namespace
