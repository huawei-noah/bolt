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

#include <training/base/layers/activations/TanhActivation.h>
#include <training/base/layers/basic/DataLayer.h>
#include <training/compiler/Workflow.h>

namespace UT
{

namespace
{

raul::dtype golden_tanh_layer(const raul::dtype x)
{
    return std::tanh(x);
}

raul::dtype golden_tanh_layer_grad(const raul::dtype x, const raul::dtype grad)
{
    return grad * (1.0_dt - std::tanh(x) * std::tanh(x));
}

}

TEST(TestActivationFuncTanh, Unit)
{
    PROFILE_TEST

    using namespace raul;

    constexpr size_t BATCH_SIZE = 7;
    constexpr size_t WIDTH = 21;
    constexpr size_t HEIGHT = 3;
    constexpr size_t DEPTH = 11;
    constexpr dtype eps = 1.0e-6_dt;
    constexpr auto range = std::make_pair(-1.0_dt, 1.0_dt);

    WorkflowEager work(raul::CompressionMode::NONE, raul::CalculationMode::DETERMINISTIC, raul::AllocationMode::STANDARD, raul::ExecutionTarget::CPU);

    work.add<DataLayer>("data", DataParams{ { "in" }, DEPTH, HEIGHT, WIDTH });
    work.add<TanhActivation>("tanh", BasicParams{ { "in" }, { "out" } });

    TENSORS_CREATE(BATCH_SIZE);

    auto& memory_manager = work.getMemoryManager<MemoryManager>();

    tools::init_rand_tensor("in", range, memory_manager);
    tools::init_rand_tensor("outGradient", range, memory_manager);

    work.forwardPassTraining();
    const Tensor& in = memory_manager["in"];
    const Tensor& out = memory_manager["out"];
    EXPECT_EQ(in.size(), out.size());
    for (size_t i = 0; i < out.size(); ++i)
    {
        EXPECT_NEAR(out[i], golden_tanh_layer(in[i]), eps);
    }

    work.backwardPassTraining();
    const Tensor& inGrad = memory_manager[Name("in").grad()];
    const Tensor& outGrad = memory_manager[Name("out").grad()];
    EXPECT_EQ(in.size(), inGrad.size());
    for (size_t i = 0; i < inGrad.size(); ++i)
    {
        EXPECT_NEAR(inGrad[i], golden_tanh_layer_grad(in[i], outGrad[i]), eps);
    }
}

}