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

#include <training/base/layers/activations/SigmoidActivation.h>
#include <training/base/layers/basic/DataLayer.h>

namespace UT
{

namespace
{

raul::half golden_sigmoid_fp16(const raul::half x)
{
    return TOHTYPE(1.0_dt / (1.0_dt + std::exp(-TODTYPE(x))));
}

raul::half golden_sigmoid_grad_fp16(const raul::half out, const raul::half grad)
{
    return TOHTYPE(grad * out * (1.0_hf - out));
}

}

TEST(TestActivationFuncSigmoid, FP16Unit)
{
    PROFILE_TEST

    using namespace raul;

    constexpr size_t BATCH_SIZE = 2;
    constexpr size_t WIDTH = 50;
    constexpr size_t HEIGHT = 40;
    constexpr size_t DEPTH = 50;
    constexpr dtype eps = 3.0e-4_dt;
    const auto range = std::make_pair(0_hf, 1.0_hf);

    WorkflowEager work(raul::CompressionMode::NONE, raul::CalculationMode::DETERMINISTIC, raul::AllocationMode::STANDARD, raul::ExecutionTarget::CPUFP16);

    work.add<DataLayer>("data", DataParams{ { "in" }, DEPTH, HEIGHT, WIDTH });
    work.add<SigmoidActivation>("sigmoid", BasicParams{ { "in" }, { "out" } });

    TENSORS_CREATE(BATCH_SIZE);

    auto& memory_manager = work.getMemoryManager<MemoryManagerFP16>();

    tools::init_rand_tensor("in", range, memory_manager);
    tools::init_rand_tensor("outGradient", range, memory_manager);

    work.forwardPassTraining();
    const auto& in = memory_manager["in"];
    const auto& out = memory_manager["out"];
    EXPECT_EQ(in.size(), out.size());
    for (size_t i = 0; i < out.size(); ++i)
    {
        EXPECT_NEAR(out[i], golden_sigmoid_fp16(in[i]), eps);
    }

    work.backwardPassTraining();
    const auto& inGrad = memory_manager[Name("in").grad()];
    const auto& outGrad = memory_manager[Name("out").grad()];
    EXPECT_EQ(in.size(), inGrad.size());
    for (size_t i = 0; i < inGrad.size(); ++i)
    {
        EXPECT_NEAR(inGrad[i], golden_sigmoid_grad_fp16(out[i], outGrad[i]), eps);
    }
}

TEST(TestActivationFuncSigmoid, FP16INT8Unit)
{
    PROFILE_TEST

    using namespace raul;

    constexpr size_t BATCH_SIZE = 2;
    constexpr size_t WIDTH = 50;
    constexpr size_t HEIGHT = 40;
    constexpr size_t DEPTH = 50;
    constexpr dtype eps = 3.0e-4_dt;
    const auto range = std::make_pair(0.0_hf, 1.0_hf);

    WorkflowEager work(raul::CompressionMode::INT8, raul::CalculationMode::DETERMINISTIC, raul::AllocationMode::STANDARD, raul::ExecutionTarget::CPUFP16);

    work.add<DataLayer>("data", DataParams{ { "in" }, DEPTH, HEIGHT, WIDTH });
    work.add<SigmoidActivation>("sigmoid", BasicParams{ { "in" }, { "out" } });

    TENSORS_CREATE(BATCH_SIZE);

    auto& memory_manager = work.getMemoryManager<MemoryManagerFP16>();

    tools::init_rand_tensor("in", range, memory_manager);
    tools::init_rand_tensor("outGradient", range, memory_manager);

    work.forwardPassTraining();
    const auto& in = memory_manager["in"];
    const auto& out = memory_manager["out"];
    EXPECT_EQ(in.size(), out.size());
    for (size_t i = 0; i < out.size(); ++i)
    {
        EXPECT_NEAR(out[i], golden_sigmoid_fp16(in[i]), eps);
    }

    work.backwardPassTraining();
    const auto& inGrad = memory_manager[Name("in").grad()];
    const auto& outGrad = memory_manager[Name("out").grad()];
    EXPECT_EQ(in.size(), inGrad.size());
    for (size_t i = 0; i < inGrad.size(); ++i)
    {
        EXPECT_NEAR(inGrad[i], golden_sigmoid_grad_fp16(out[i], outGrad[i]), eps);
    }
}

TEST(TestActivationFuncSigmoid, FP16BigUnit)
{
    PROFILE_TEST

    using namespace raul;

    constexpr size_t BATCH_SIZE = 200;
    constexpr size_t WIDTH = 50;
    constexpr size_t HEIGHT = 40;
    constexpr size_t DEPTH = 50;
    const auto range = std::make_pair(0_hf, 1.0_hf);

    Workflow work(raul::CompressionMode::NONE, raul::CalculationMode::DETERMINISTIC, raul::AllocationMode::STANDARD, raul::ExecutionTarget::CPUFP16);

    work.add<DataLayer>("data", DataParams{ { "in" }, DEPTH, HEIGHT, WIDTH });
    work.add<SigmoidActivation>("sigmoid", BasicParams{ { "in" }, { "out" } });

    TENSORS_CREATE(BATCH_SIZE);

    auto& memory_manager = work.getMemoryManager<MemoryManagerFP16>();

    tools::init_rand_tensor("in", range, memory_manager);
    tools::init_rand_tensor("outGradient", range, memory_manager);

    ASSERT_NO_THROW(work.forwardPassTraining());
    ASSERT_NO_THROW(work.backwardPassTraining());
}

TEST(TestActivationFuncSigmoid, FP16INT8BigUnit)
{
    PROFILE_TEST

    using namespace raul;

    constexpr size_t BATCH_SIZE = 200;
    constexpr size_t WIDTH = 50;
    constexpr size_t HEIGHT = 40;
    constexpr size_t DEPTH = 50;
    const auto range = std::make_pair(0.0_hf, 1.0_hf);

    Workflow work(raul::CompressionMode::INT8, raul::CalculationMode::DETERMINISTIC, raul::AllocationMode::STANDARD, raul::ExecutionTarget::CPUFP16);

    work.add<DataLayer>("data", DataParams{ { "in" }, DEPTH, HEIGHT, WIDTH });
    work.add<SigmoidActivation>("sigmoid", BasicParams{ { "in" }, { "out" } });

    TENSORS_CREATE(BATCH_SIZE);

    auto& memory_manager = work.getMemoryManager<MemoryManagerFP16>();

    tools::init_rand_tensor("in", range, memory_manager);
    tools::init_rand_tensor("outGradient", range, memory_manager);

    ASSERT_NO_THROW(work.forwardPassTraining());
    ASSERT_NO_THROW(work.backwardPassTraining());
}

} // namespace UT