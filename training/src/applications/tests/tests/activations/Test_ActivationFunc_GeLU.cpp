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

#include <training/common/Common.h>
#include <training/common/Conversions.h>
#include <training/common/MemoryManager.h>
#include <training/layers/activations/GeLUActivation.h>
#include <training/layers/basic/DataLayer.h>
#include <training/network/Workflow.h>

namespace UT
{

namespace
{

raul::dtype golden_gelu_tanh(const raul::dtype x)
{
    return raul::Common::GeLU_Tanh(x);
}

raul::dtype golden_gelu_tanh_grad(const raul::dtype x, const raul::dtype grad)
{
    auto th = std::tanh(RAUL_SQRT2_PI * (x + GELU_CONST * std::pow(x, 3)));
    return static_cast<raul::dtype>(grad * 0.5_dt * (1.0_dt + th + x * RAUL_SQRT2_PI * (1.0_dt + 3 * GELU_CONST * x * x) * (1.0_dt - th * th)));
}

raul::dtype golden_gelu_erf(const raul::dtype x)
{
    return raul::Common::GeLU_Erf(x);
}

raul::dtype golden_gelu_erf_grad(const raul::dtype x, const raul::dtype grad)
{
    return static_cast<raul::dtype>(grad * 0.5_dt * (1.0_dt + std::erf(x * RAUL_SQRT1_2) + x * RAUL_SQRT2_PI * exp(-0.5 * x * x)));
}

}

TEST(TestActivationFuncGeLU, GeLUErfUnit)
{
    PROFILE_TEST
    using namespace raul;

    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    constexpr auto eps = 1e-4_dt;

    size_t BATCH_SIZE = 2;
    size_t WIDTH = 2;
    size_t HEIGHT = 2;
    size_t DEPTH = 3;

    Tensor realOut = { 0.8413_dt, 0.8413_dt, 1.9545_dt, 1.9545_dt, 0.8413_dt, 0.8413_dt, 1.9545_dt, 1.9545_dt, 0.8413_dt, 0.8413_dt, 1.9545_dt, 1.9545_dt,
                       0.8413_dt, 0.8413_dt, 2.9960_dt, 2.9960_dt, 3.9999_dt, 3.9999_dt, 2.9960_dt, 2.9960_dt, 1.9545_dt, 0.8413_dt, 2.9960_dt, 7.0000_dt };

    Tensor realGrad = { 1.0833_dt, 1.0833_dt, 2.1705_dt, 2.1705_dt, 1.0833_dt, 1.0833_dt, 2.1705_dt, 2.1705_dt, 1.0833_dt, 1.0833_dt, 2.1705_dt, 2.1705_dt,
                        1.0833_dt, 1.0833_dt, 3.0358_dt, 3.0358_dt, 4.0020_dt, 4.0020_dt, 3.0358_dt, 3.0358_dt, 2.1705_dt, 1.0833_dt, 3.0358_dt, 7.0000_dt };

    Tensor raw = { 1._dt, 1._dt, 2._dt, 2._dt, 1._dt, 1._dt, 2._dt, 2._dt, 1._dt, 1._dt, 2._dt, 2._dt, 1._dt, 1._dt, 3._dt, 3._dt, 4._dt, 4._dt, 3._dt, 3._dt, 2._dt, 1._dt, 3._dt, 7._dt };

    work.add<DataLayer>("data2", DataParams{ { "in", "target", "weights" }, DEPTH, HEIGHT, WIDTH });
    GeLUErf gelu{ "gelu", { { "in" }, { "out" } }, networkParameters };

    TENSORS_CREATE(BATCH_SIZE);
    memory_manager["in"] = TORANGE(raw);
    gelu.forwardCompute(NetworkMode::Train);

    memory_manager[Name("out").grad()].memAllocate(nullptr);
    memory_manager[Name("out").grad()] = TORANGE(raw);

    const raul::Tensor& out = memory_manager["out"];

    EXPECT_EQ(out.size(), realOut.size());
    for (size_t i = 0; i < out.size(); ++i)
        EXPECT_NEAR(out[i], realOut[i], eps);

    std::cout << " - GeLU_Erf forward is Ok.\n";

    gelu.backwardCompute();

    const raul::Tensor& inGrad = memory_manager[raul::Name("in").grad()];

    EXPECT_EQ(inGrad.size(), realGrad.size());
    for (size_t i = 0; i < inGrad.size(); ++i)
        EXPECT_NEAR(inGrad[i], realGrad[i], eps);

    std::cout << " - GeLU_Erf backward is Ok.\n";
}

TEST(TestActivationFuncGeLU, GeLUATanhUnit)
{
    PROFILE_TEST
    using namespace raul;

    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    constexpr auto eps = 2e-3_dt;

    size_t BATCH_SIZE = 2;
    size_t WIDTH = 2;
    size_t HEIGHT = 2;
    size_t DEPTH = 3;

    Tensor realOut = { 0.8412_dt, 0.8412_dt, 1.9546_dt, 1.9546_dt, 0.8412_dt, 0.8412_dt, 1.9546_dt, 1.9546_dt, 0.8412_dt, 0.8412_dt, 1.9546_dt, 1.9546_dt,
                       0.8412_dt, 0.8412_dt, 2.9964_dt, 2.9964_dt, 3.9999_dt, 3.9999_dt, 2.9964_dt, 2.9964_dt, 1.9546_dt, 0.8412_dt, 2.9964_dt, 7.0000_dt };

    Tensor realGrad = { 1.0830_dt, 1.0830_dt, 2.1722_dt, 2.1722_dt, 1.0830_dt, 1.0830_dt, 2.1722_dt, 2.1722_dt, 1.0830_dt, 1.0830_dt, 2.1722_dt, 2.1722_dt,
                        1.0830_dt, 1.0830_dt, 3.0348_dt, 3.0348_dt, 4.0013_dt, 4.0013_dt, 3.0348_dt, 3.0348_dt, 2.1722_dt, 1.0830_dt, 3.0348_dt, 7.0000_dt };

    Tensor raw = { 1._dt, 1._dt, 2._dt, 2._dt, 1._dt, 1._dt, 2._dt, 2._dt, 1._dt, 1._dt, 2._dt, 2._dt, 1._dt, 1._dt, 3._dt, 3._dt, 4._dt, 4._dt, 3._dt, 3._dt, 2._dt, 1._dt, 3._dt, 7._dt };

    work.add<DataLayer>("data2", DataParams{ { "in", "target", "weights" }, DEPTH, HEIGHT, WIDTH });
    GeLUTanh gelu{ "gelu", { { "in" }, { "out" } }, networkParameters };

    TENSORS_CREATE(BATCH_SIZE);
    memory_manager["in"] = TORANGE(raw);
    gelu.forwardCompute(NetworkMode::Train);

    memory_manager[Name("out").grad()].memAllocate(nullptr);
    memory_manager[Name("out").grad()] = TORANGE(raw);

    const raul::Tensor& out = memory_manager["out"];

    EXPECT_EQ(out.size(), realOut.size());
    for (size_t i = 0; i < out.size(); ++i)
        EXPECT_NEAR(out[i], realOut[i], eps);

    std::cout << " - GeLU_Tanh forward is Ok.\n";

    gelu.backwardCompute();

    const raul::Tensor& inGrad = memory_manager[raul::Name("in").grad()];

    EXPECT_EQ(inGrad.size(), realGrad.size());
    for (size_t i = 0; i < inGrad.size(); ++i)
        EXPECT_NEAR(inGrad[i], realGrad[i], eps);

    std::cout << " - GeLU_Tanh backward is Ok.\n";
}

TEST(TestActivationFuncGeLU, TanhGpuUnit)
{
    PROFILE_TEST

    GPU_ONLY_TEST

    using namespace raul;

    constexpr size_t BATCH_SIZE = 5;
    constexpr size_t WIDTH = 10;
    constexpr size_t HEIGHT = 30;
    constexpr size_t DEPTH = 20;
    constexpr dtype eps = 1.0e-6_dt;
    constexpr auto range = std::make_pair(-1.0_dt, 1.0_dt);

    WorkflowEager work(raul::CompressionMode::NONE, raul::CalculationMode::DETERMINISTIC, raul::AllocationMode::STANDARD, raul::ExecutionTarget::GPU);
    work.getKernelManager().setExecutionPolicy(raul::KernelExecutionPolicy::DefaultParams);

    work.add<DataLayer>("data", DataParams{ { "in" }, DEPTH, HEIGHT, WIDTH });
    work.add<GeLUTanh>("gelu_tanh", BasicParams{ { "in" }, { "out" } });

    TENSORS_CREATE(BATCH_SIZE);

    MemoryManagerGPU& memory_manager = work.getMemoryManager<MemoryManagerGPU>();

    tools::init_rand_tensor("in", range, memory_manager);
    tools::init_rand_tensor("outGradient", range, memory_manager);

    work.forwardPassTraining();
    const Tensor& in = memory_manager["in"];
    const Tensor& out = memory_manager["out"];
    EXPECT_EQ(in.size(), out.size());
    for (size_t i = 0; i < out.size(); ++i)
    {
        EXPECT_NEAR(out[i], golden_gelu_tanh(in[i]), eps);
    }

    work.backwardPassTraining();
    const Tensor& inGrad = memory_manager[Name("in").grad()];
    const Tensor& outGrad = memory_manager[Name("out").grad()];
    EXPECT_EQ(in.size(), inGrad.size());
    for (size_t i = 0; i < inGrad.size(); ++i)
    {
        EXPECT_NEAR(inGrad[i], golden_gelu_tanh_grad(in[i], outGrad[i]), eps);
    }
}

TEST(TestActivationFuncGeLU, ErfGpuUnit)
{
    PROFILE_TEST

    GPU_ONLY_TEST

    using namespace raul;

    constexpr size_t BATCH_SIZE = 3;
    constexpr size_t WIDTH = 3;
    constexpr size_t HEIGHT = 100;
    constexpr size_t DEPTH = 2;
    constexpr dtype eps = 1.0e-6_dt;
    constexpr auto range = std::make_pair(-1.0_dt, 1.0_dt);

    WorkflowEager work(raul::CompressionMode::NONE, raul::CalculationMode::DETERMINISTIC, raul::AllocationMode::STANDARD, raul::ExecutionTarget::GPU);
    work.getKernelManager().setExecutionPolicy(raul::KernelExecutionPolicy::DefaultParams);

    work.add<DataLayer>("data", DataParams{ { "in" }, DEPTH, HEIGHT, WIDTH });
    work.add<GeLUErf>("gelu_erf", BasicParams{ { "in" }, { "out" } });

    TENSORS_CREATE(BATCH_SIZE);

    MemoryManagerGPU& memory_manager = work.getMemoryManager<MemoryManagerGPU>();

    tools::init_rand_tensor("in", range, memory_manager);
    tools::init_rand_tensor("outGradient", range, memory_manager);

    work.forwardPassTraining();
    const Tensor& in = memory_manager["in"];
    const Tensor& out = memory_manager["out"];
    EXPECT_EQ(in.size(), out.size());
    for (size_t i = 0; i < out.size(); ++i)
    {
        EXPECT_NEAR(out[i], golden_gelu_erf(in[i]), eps);
    }

    work.backwardPassTraining();
    const Tensor& inGrad = memory_manager[Name("in").grad()];
    const Tensor& outGrad = memory_manager[Name("out").grad()];
    EXPECT_EQ(in.size(), inGrad.size());
    for (size_t i = 0; i < inGrad.size(); ++i)
    {
        EXPECT_NEAR(inGrad[i], golden_gelu_erf_grad(in[i], outGrad[i]), eps);
    }
}

} // UT namespace