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
#include <utility>

#include <training/common/Common.h>
#include <training/common/Conversions.h>
#include <training/common/MemoryManager.h>
#include <training/layers/basic/DataLayer.h>
#include <training/layers/basic/TensorLayer.h>
#include <training/layers/basic/ElementWiseSubLayer.h>
#include <training/network/Workflow.h>
#include <training/tools/NameGenerator.h>

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

TEST(TestLayerElementWiseSub, NoBroadcastGpuUnit)
{
    PROFILE_TEST

    GPU_ONLY_TEST

    // Test parameters
    constexpr auto eps_rel = 1e-5_dt;
    constexpr auto enable_broadcast = false;
    constexpr size_t BATCH_SIZE = 3;
    constexpr size_t WIDTH = 11;
    constexpr size_t HEIGHT = 10;
    constexpr size_t DEPTH = 21;
    constexpr auto range = std::make_pair(-1.0_dt, 1.0_dt);

    // Initialization
    raul::WorkflowEager work(raul::CompressionMode::NONE, raul::CalculationMode::DETERMINISTIC, raul::AllocationMode::STANDARD, raul::ExecutionTarget::GPU);

    work.add<raul::DataLayer>("data", raul::DataParams{ { "x", "y" }, DEPTH, HEIGHT, WIDTH });
    work.add<raul::ElementWiseSubLayer>("sub", raul::ElementWiseLayerParams{ { "x", "y" }, { "out" }, enable_broadcast });

    TENSORS_CREATE(BATCH_SIZE)

    raul::MemoryManagerGPU& memory_manager = work.getMemoryManager<raul::MemoryManagerGPU>();
    tools::init_rand_tensor("x", range, memory_manager);
    tools::init_rand_tensor("y", range, memory_manager);
    tools::init_rand_tensor(raul::Name("out").grad(), range, memory_manager);

    // Apply function
    ASSERT_NO_THROW(work.forwardPassTraining());

    // Checks
    const raul::Tensor& x = memory_manager["x"];
    const raul::Tensor& y = memory_manager["y"];
    const raul::Tensor& out = memory_manager["out"];
    for (size_t i = 0; i < out.size(); ++i)
    {
        ASSERT_TRUE(tools::expect_near_relative(out[i], golden_sub_layer(x[i], y[i]), eps_rel));
    }

    ASSERT_NO_THROW(work.backwardPassTraining());

    // Checks
    const raul::Tensor& xGrad = memory_manager[raul::Name("x").grad()];
    const raul::Tensor& yGrad = memory_manager[raul::Name("y").grad()];
    const raul::Tensor& outGrad = memory_manager[raul::Name("out").grad()];
    EXPECT_EQ(xGrad.size(), outGrad.size());
    EXPECT_EQ(yGrad.size(), outGrad.size());
    for (size_t i = 0; i < xGrad.size(); ++i)
    {
        const auto [realXGrad, realYGrad] = golden_sub_layer_grad(outGrad[i]);
        EXPECT_EQ(xGrad[i], realXGrad);
        EXPECT_EQ(yGrad[i], realYGrad);
    }
}

struct TestElementWiseSubLayerGPU : public testing::TestWithParam<tuple<size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t>>
{
    const size_t b0 = get<0>(GetParam());
    const size_t d0 = get<1>(GetParam());
    const size_t h0 = get<2>(GetParam());
    const size_t w0 = get<3>(GetParam());

    const size_t b1 = get<4>(GetParam());
    const size_t d1 = get<5>(GetParam());
    const size_t h1 = get<6>(GetParam());
    const size_t w1 = get<7>(GetParam());

    const raul::dtype eps = TODTYPE(1e-4);
    const std::pair<raul::dtype, raul::dtype> range = std::make_pair(1.0_dt, 10.0_dt);

    const bool enable_broadcast = true;

    const size_t inputNumber = 2;
};

TEST_P(TestElementWiseSubLayerGPU, BroadcastGpuUnit)
{
    PROFILE_TEST

    GPU_ONLY_TEST

    raul::WorkflowEager work{ raul::CompressionMode::NONE, raul::CalculationMode::DETERMINISTIC, raul::AllocationMode::STANDARD, raul::ExecutionTarget::GPU };

    work.add<raul::TensorLayer>("in0", raul::TensorParams{ { "in0", "realIn0Grad" }, b0, d0, h0, w0 });
    work.add<raul::TensorLayer>("in1", raul::TensorParams{ { "in1", "realIn1Grad" }, b1, d1, h1, w1 });
    work.add<raul::ElementWiseSubLayer>("sub", raul::ElementWiseLayerParams{ { "in0", "in1" }, { "out" }, enable_broadcast });
    TENSORS_CREATE(std::max(b0, b1));

    auto& memory_manager = work.getMemoryManager<raul::MemoryManagerGPU>();
    
    memory_manager["realIn0Grad"] = 0.0_dt;
    memory_manager["realIn1Grad"] = 0.0_dt;

    tools::init_rand_tensor("in0", range, memory_manager);
    tools::init_rand_tensor("in1", range, memory_manager);
    
    tools::init_rand_tensor(raul::Name("out").grad(), range, memory_manager);

    ASSERT_NO_THROW(work.forwardPassTraining());

    // Forward checks
    const raul::Tensor& in0 = memory_manager["in0"];
    const raul::Tensor& in1 = memory_manager["in1"];
    const raul::Tensor& out = memory_manager["out"];

    const auto in0Broadcasted = in0.getBroadcastedViewer(out.getShape());
    const auto in1Broadcasted = in1.getBroadcastedViewer(out.getShape());

    for (size_t i = 0; i < out.size(); ++i)
    {
        ASSERT_TRUE(tools::expect_near_relative(out[i], in0Broadcasted[i] - in1Broadcasted[i], eps));
    }

    work.backwardPassTraining();

    // Backward checks
    const raul::Tensor& deltas = memory_manager[raul::Name("out").grad()];
    for (size_t i = 0; i < inputNumber; ++i)
    {
        raul::Tensor realInGrad = memory_manager[raul::Name("realIn" + std::to_string(i) + "Grad")];
        auto realInGradBroadcasted = realInGrad.getBroadcastedViewer(out.getShape());
        for (size_t k = 0; k < out.size(); ++k)
        {
            realInGradBroadcasted[k] += (i == 0 ? 1.0_dt : -1.0_dt) * deltas[k];
        }

        // Checks
        const raul::Tensor& inGrad = memory_manager[raul::Name("in" + std::to_string(i)).grad()];
        for (size_t k = 0; k < realInGrad.size(); ++k)
        {
            ASSERT_TRUE(tools::expect_near_relative(inGrad[k], realInGrad[k], eps));
        }
    }
}

INSTANTIATE_TEST_SUITE_P(TestGpu,
                         TestElementWiseSubLayerGPU,
                         testing::Values(make_tuple(2, 11, 3, 7, 2, 1, 3, 7),
                                         make_tuple(1, 2, 4, 5, 1, 2, 1, 5),
                                         make_tuple(7, 5, 11, 4, 7, 1, 1, 4),
                                         make_tuple(17, 5, 3, 2, 1, 5, 3, 2),
                                         make_tuple(7, 8, 9, 3, 1, 1, 9, 3),
                                         make_tuple(1, 1, 1, 9, 5, 6, 7, 9),
                                         make_tuple(5, 2, 1, 11, 5, 1, 4, 11),
                                         make_tuple(2, 3, 4, 5, 2, 3, 4, 1),
                                         make_tuple(11, 23, 1, 1, 11, 23, 7, 9),
                                         make_tuple(11, 1, 7, 8, 1, 1, 1, 1)));

} // UT namespace
