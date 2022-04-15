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

#include <training/layers/activations/SigmoidActivation.h>
#include <training/layers/basic/DataLayer.h>

namespace UT
{

namespace
{

raul::dtype golden_sigmoid(const raul::dtype x)
{
    return 1.0_dt / (1.0_dt + std::exp(-x));
}

raul::dtype golden_sigmoid_grad(const raul::dtype out, const raul::dtype grad)
{
    return grad * out * (1.0_dt - out);
}

}

TEST(TestActivationFuncSigmoid, GpuUnit)
{
    PROFILE_TEST

    GPU_ONLY_TEST

    using namespace raul;

    constexpr size_t BATCH_SIZE = 2;
    constexpr size_t WIDTH = 3;
    constexpr size_t HEIGHT = 4;
    constexpr size_t DEPTH = 5;
    constexpr dtype eps = 1.0e-6_dt;
    constexpr auto range = std::make_pair(-1.0_dt, 1.0_dt);

    WorkflowEager work(raul::CompressionMode::NONE, raul::CalculationMode::DETERMINISTIC, raul::AllocationMode::STANDARD, raul::ExecutionTarget::GPU);
    work.getKernelManager().setExecutionPolicy(raul::KernelExecutionPolicy::DefaultParams);

    work.add<DataLayer>("data", DataParams{ { "in" }, DEPTH, HEIGHT, WIDTH });
    work.add<SigmoidActivation>("sigmoid", BasicParams{ { "in" }, { "out" } });

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
        EXPECT_NEAR(out[i], golden_sigmoid(in[i]), eps);
    }

    work.backwardPassTraining();
    const Tensor& inGrad = memory_manager[Name("in").grad()];
    const Tensor& outGrad = memory_manager[Name("out").grad()];
    EXPECT_EQ(in.size(), inGrad.size());
    for (size_t i = 0; i < inGrad.size(); ++i)
    {
        EXPECT_NEAR(inGrad[i], golden_sigmoid_grad(out[i], outGrad[i]), eps);
    }
}

} // namespace UT
