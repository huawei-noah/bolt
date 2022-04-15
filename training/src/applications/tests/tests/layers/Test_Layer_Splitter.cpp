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

#include <training/layers/basic/DataLayer.h>
#include <training/layers/basic/SplitterLayer.h>
#include <training/network/Workflow.h>

namespace UT
{

TEST(TestLayerSplitter, GpuUnit)
{
    PROFILE_TEST

    GPU_ONLY_TEST

    using namespace raul;

    constexpr size_t BATCH_SIZE = 8;
    constexpr size_t WIDTH = 15;
    constexpr size_t HEIGHT = 23;
    constexpr size_t DEPTH = 3;
    constexpr auto range = std::make_pair(1.0_dt, 100.0_dt);

    WorkflowEager work(raul::CompressionMode::NONE, raul::CalculationMode::DETERMINISTIC, raul::AllocationMode::STANDARD, raul::ExecutionTarget::GPU);
    work.getKernelManager().setExecutionPolicy(raul::KernelExecutionPolicy::DefaultParams);

    work.add<DataLayer>("data", DataParams{ { "in" }, DEPTH, HEIGHT, WIDTH });
    work.add<SplitterLayer>("splitter", BasicParams{ { "in" }, { "out1", "out2", "out3" } });

    TENSORS_CREATE(BATCH_SIZE)

    MemoryManagerGPU& memory_manager = work.getMemoryManager<MemoryManagerGPU>();
    tools::init_rand_tensor("in", range, memory_manager);
    tools::init_rand_tensor("out1Gradient", range, memory_manager);
    tools::init_rand_tensor("out2Gradient", range, memory_manager);
    tools::init_rand_tensor("out3Gradient", range, memory_manager);

    work.forwardPassTraining();
    const Tensor& input = memory_manager["in"];
    const Tensor& out1 = memory_manager["out1"];
    const Tensor& out2 = memory_manager["out2"];
    const Tensor& out3 = memory_manager["out3"];
    EXPECT_EQ(out1.size(), input.size());
    EXPECT_EQ(out2.size(), input.size());
    EXPECT_EQ(out3.size(), input.size());
    for (size_t i = 0; i < out1.size(); ++i)
    {
        EXPECT_EQ(out1[i], input[i]);
        EXPECT_EQ(out2[i], input[i]);
        EXPECT_EQ(out3[i], input[i]);
    }

    work.backwardPassTraining();
    const Tensor& inGrad = memory_manager[raul::Name("in").grad()];
    const Tensor& out1Grad = memory_manager[raul::Name("out1").grad()];
    const Tensor& out2Grad = memory_manager[raul::Name("out2").grad()];
    const Tensor& out3Grad = memory_manager[raul::Name("out3").grad()];
    EXPECT_EQ(inGrad.size(), input.size());
    for (size_t i = 0; i < inGrad.size(); ++i)
    {
        EXPECT_EQ(inGrad[i], out1Grad[i] + out2Grad[i] + out3Grad[i]);
    }
}

}