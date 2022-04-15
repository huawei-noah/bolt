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

#include <training/api/API.h>
#include <training/common/Common.h>
#include <training/common/MemoryManager.h>
#include <training/layers/basic/DataLayer.h>
#include <training/loss/CrossEntropyLoss.h>
#include <training/network/Workflow.h>

namespace UT
{

namespace
{

raul::dtype golden_ce_loss(const raul::dtype x, const raul::dtype y)
{
    return -y * std::log(x);
}

raul::dtype golden_ce_loss_grad(const raul::dtype x, const raul::dtype y, const raul::dtype g = 1.0_dt)
{
    return -y / x * g;
}

}

TEST(TestLoss, CELossGpuUnit)
{
    PROFILE_TEST

    GPU_ONLY_TEST

    using namespace raul;

    constexpr size_t BATCH_SIZE = 5;
    constexpr size_t WIDTH = 19;
    constexpr size_t HEIGHT = 2;
    constexpr size_t DEPTH = 3;
    constexpr dtype eps = 1.0e-4_dt;
    constexpr auto range = std::make_pair(1.0_dt, 10.0_dt);

    WorkflowEager work(raul::CompressionMode::NONE, raul::CalculationMode::DETERMINISTIC, raul::AllocationMode::STANDARD, raul::ExecutionTarget::GPU);
    work.getKernelManager().setExecutionPolicy(raul::KernelExecutionPolicy::DefaultParams);

    work.add<DataLayer>("data", DataParams{ { "in", "target" }, DEPTH, HEIGHT, WIDTH });
    work.add<CrossEntropyLoss>("loss", LossParams{ { "in", "target" }, { "loss" }, "none" });

    TENSORS_CREATE(BATCH_SIZE)
    MemoryManagerGPU& memory_manager = work.getMemoryManager<MemoryManagerGPU>();

    tools::init_rand_tensor("in", range, memory_manager);
    tools::init_rand_tensor("target", range, memory_manager);
    tools::init_rand_tensor("lossGradient", range, memory_manager);

    work.forwardPassTraining();

    const Tensor& loss = memory_manager["loss"];
    const Tensor& in = memory_manager["in"];
    const Tensor& target = memory_manager["target"];
    EXPECT_EQ(in.size(), loss.size());
    for (size_t i = 0; i < loss.size(); ++i)
    {
        EXPECT_NEAR(loss[i], golden_ce_loss(in[i], target[i]), eps);
    }

    work.backwardPassTraining();
    const Tensor& inGrad = memory_manager[raul::Name("in").grad()];
    const Tensor& lossGrad = memory_manager[raul::Name("loss").grad()];
    EXPECT_EQ(inGrad.size(), in.size());
    for (size_t i = 0; i < inGrad.size(); ++i)
    {
        EXPECT_NEAR(inGrad[i], golden_ce_loss_grad(in[i], target[i], lossGrad[i]), eps);
    }
}

}
