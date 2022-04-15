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

#include <training/common/MemoryManager.h>
#include <training/layers/basic/ClampLayer.h>
#include <training/network/Layers.h>
#include <training/network/Workflow.h>

namespace UT
{

namespace
{

raul::dtype golden_clamp_layer(const raul::dtype x, const raul::dtype min, const raul::dtype max)
{
    return std::clamp(x, min, max);
}

raul::dtype golden_clamp_layer_grad(const raul::dtype x, const raul::dtype min, const raul::dtype max, const raul::dtype grad)
{
    return grad * static_cast<raul::dtype>(x >= min && x <= max);
}

}

TEST(TestLayerClamp, ForwardUnit)
{
    PROFILE_TEST
    // Test parameters
    const auto batch = 2;
    const auto depth = 1;
    const auto height = 1;
    const auto width = 5;

    constexpr raul::dtype min = 3.0_dt;
    constexpr raul::dtype max = 8.0_dt;

    const raul::Tensor x{ 1.0_dt, 2.0_dt, 3.0_dt, 4.0_dt, 5.0_dt, 6.0_dt, 7.0_dt, 8.0_dt, 9.0_dt, 10.0_dt };

    const raul::Tensor realOut{ 3.0_dt, 3.0_dt, 3.0_dt, 4.0_dt, 5.0_dt, 6.0_dt, 7.0_dt, 8.0_dt, 8.0_dt, 8.0_dt };

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    work.add<raul::DataLayer>("data", raul::DataParams{ { "x" }, depth, height, width });

    // Apply function
    raul::ClampLayer clamper("clamp", raul::ClampLayerParams{ { "x" }, { "out" }, min, max }, networkParameters);
    TENSORS_CREATE(batch);
    memory_manager["x"] = TORANGE(x);

    clamper.forwardCompute(raul::NetworkMode::Train);

    // Checks
    const auto& xTensor = memory_manager["x"];
    const auto& outTensor = memory_manager["out"];

    EXPECT_EQ(outTensor.size(), xTensor.size());
    EXPECT_EQ(outTensor.size(), realOut.size());
    for (size_t i = 0; i < outTensor.size(); ++i)
    {
        EXPECT_EQ(outTensor[i], realOut[i]);
    }
}

TEST(TestLayerClamp, BackwardUnit)
{
    PROFILE_TEST
    // Test parameters
    const auto batch = 2;
    const auto depth = 1;
    const auto height = 1;
    const auto width = 5;

    constexpr raul::dtype min = 3.0_dt;
    constexpr raul::dtype max = 8.0_dt;

    const raul::Tensor x{ 1.0_dt, 2.0_dt, 3.0_dt, 4.0_dt, 5.0_dt, 6.0_dt, 7.0_dt, 8.0_dt, 9.0_dt, 10.0_dt };
    const raul::Tensor deltas{ 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt };
    const raul::Tensor realGrad{ 0.0_dt, 0.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 0.0_dt, 0.0_dt };

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    work.add<raul::DataLayer>("data", raul::DataParams{ { "x" }, depth, height, width });

    // Apply function
    raul::ClampLayer clamper("clamp", raul::ClampLayerParams{ { "x" }, { "out" }, min, max }, networkParameters);
    TENSORS_CREATE(batch);
    memory_manager["x"] = TORANGE(x);
    memory_manager[raul::Name("out").grad()] = TORANGE(deltas);
    clamper.forwardCompute(raul::NetworkMode::Train);
    clamper.backwardCompute();

    // Checks
    const auto& xTensor = memory_manager["x"];
    const auto& xNablaTensor = memory_manager[raul::Name("x").grad()];

    EXPECT_EQ(xNablaTensor.size(), xTensor.size());
    EXPECT_EQ(xNablaTensor.size(), realGrad.size());
    for (size_t i = 0; i < xNablaTensor.size(); ++i)
    {
        EXPECT_EQ(xNablaTensor[i], realGrad[i]);
    }
}

TEST(TestLayerClamp, GpuUnit)
{
    PROFILE_TEST

    GPU_ONLY_TEST

    using namespace raul;

    constexpr size_t BATCH_SIZE = 7;
    constexpr size_t WIDTH = 11;
    constexpr size_t HEIGHT = 15;
    constexpr size_t DEPTH = 9;
    constexpr dtype eps = 1.0e-6_dt;
    constexpr auto range = std::make_pair(0.1_dt, 10.0_dt);
    constexpr dtype min = 3.0_dt;
    constexpr dtype max = 8.0_dt;

    WorkflowEager work(raul::CompressionMode::NONE, raul::CalculationMode::DETERMINISTIC, raul::AllocationMode::STANDARD, raul::ExecutionTarget::GPU);
    work.getKernelManager().setExecutionPolicy(raul::KernelExecutionPolicy::DefaultParams);

    work.add<DataLayer>("data", DataParams{ { "in" }, DEPTH, HEIGHT, WIDTH });
    work.add<ClampLayer>("clamp", ClampLayerParams{ { "in" }, { "out" }, min, max });

    TENSORS_CREATE(BATCH_SIZE)

    auto& memory_manager = work.getMemoryManager<MemoryManagerGPU>();

    tools::init_rand_tensor("in", range, memory_manager);
    tools::init_rand_tensor("outGradient", range, memory_manager);

    ASSERT_NO_THROW(work.forwardPassTraining());

    const Tensor& in = memory_manager["in"];
    const Tensor& out = memory_manager["out"];
    EXPECT_EQ(in.size(), out.size());
    for (size_t i = 0; i < out.size(); ++i)
    {
        EXPECT_NEAR(out[i], golden_clamp_layer(in[i], min, max), eps);
    }

    ASSERT_NO_THROW(work.backwardPassTraining());

    const Tensor& inGrad = memory_manager[Name("in").grad()];
    const Tensor& outGrad = memory_manager[Name("out").grad()];
    EXPECT_EQ(in.size(), inGrad.size());
    for (size_t i = 0; i < inGrad.size(); ++i)
    {
        EXPECT_NEAR(inGrad[i], golden_clamp_layer_grad(in[i], min, max, outGrad[i]), eps);
    }
}

}