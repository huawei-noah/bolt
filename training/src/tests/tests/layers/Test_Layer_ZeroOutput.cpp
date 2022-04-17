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
#include <training/base/layers/composite/rnn/ZeroOutputLayer.h>
#include <training/compiler/Layers.h>
#include <training/compiler/Workflow.h>

namespace UT
{

namespace
{

raul::dtype golden_zero_output_layer(raul::dtype index, raul::dtype in, raul::dtype length)
{
    return index < length ? in : 0.0_dt;
}

}

TEST(TestZeroOutput, CpuUnit)
{
    PROFILE_TEST

    using namespace raul;

    constexpr size_t BATCH_SIZE = 3;
    constexpr size_t WIDTH = 11;
    constexpr size_t HEIGHT = 1;
    constexpr size_t DEPTH = 23;
    constexpr auto range = std::make_pair(0.0_dt, 1.0_dt);

    const Tensor realLength{ 1.0_dt, 5.0_dt, 21.0_dt };

    WorkflowEager work(raul::CompressionMode::NONE, raul::CalculationMode::DETERMINISTIC, raul::AllocationMode::STANDARD, raul::ExecutionTarget::CPU);

    work.add<DataLayer>("dataX", DataParams{ { "in" }, DEPTH, HEIGHT, WIDTH });
    work.add<DataLayer>("dataY", DataParams{ { "realLength" }, 1u, 1u, 1u });
    work.add<ZeroOutputLayer>("zerooutput", BasicParams{ { "in", "realLength" }, { "out" } });

    TENSORS_CREATE(BATCH_SIZE)

    auto& memory_manager = work.getMemoryManager();

    memory_manager["realLength"] = TORANGE(realLength);
    tools::init_rand_tensor("in", range, memory_manager);
    tools::init_rand_tensor(Name("out").grad(), range, memory_manager);

    ASSERT_NO_THROW(work.forwardPassTraining());

    const Tensor& in = memory_manager["in"];
    const Tensor& out = memory_manager["out"];
    EXPECT_EQ(in.size(), out.size());

    auto in3D = in.reshape(yato::dims(BATCH_SIZE, DEPTH * HEIGHT, WIDTH));
    auto out3D = out.reshape(yato::dims(BATCH_SIZE, DEPTH * HEIGHT, WIDTH));
    for (size_t i = 0; i < BATCH_SIZE; ++i)
    {
        for (size_t j = 0; j < DEPTH * HEIGHT; ++j)
        {
            for (size_t k = 0; k < WIDTH; ++k)
            {
                EXPECT_EQ(out3D[i][j][k], golden_zero_output_layer(static_cast<dtype>(j), in3D[i][j][k], realLength[i]));
            }
        }
    }

    ASSERT_NO_THROW(work.backwardPassTraining());

    const Tensor& inGrad = memory_manager[Name("in").grad()];
    const Tensor& outGrad = memory_manager[Name("out").grad()];
    EXPECT_EQ(in.size(), inGrad.size());

    auto inGrad3D = inGrad.reshape(yato::dims(BATCH_SIZE, DEPTH * HEIGHT, WIDTH));
    auto outGrad3D = outGrad.reshape(yato::dims(BATCH_SIZE, DEPTH * HEIGHT, WIDTH));
    for (size_t i = 0; i < BATCH_SIZE; ++i)
    {
        for (size_t j = 0; j < DEPTH * HEIGHT; ++j)
        {
            for (size_t k = 0; k < WIDTH; ++k)
            {
                EXPECT_EQ(inGrad3D[i][j][k], golden_zero_output_layer(static_cast<dtype>(j), outGrad3D[i][j][k], realLength[i]));
            }
        }
    }
}

}