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
#include <training/layers/basic/DataLayer.h>
#include <training/layers/composite/tacotron/TargetsReductionLayer.h>
#include <training/network/Workflow.h>

namespace UT
{

TEST(TestTargetsReductionLayer, CpuUnit)
{
    PROFILE_TEST

    using namespace raul;
    using namespace raul::tacotron;

    constexpr size_t BATCH_SIZE = 2u;
    constexpr size_t DEPTH = 1u;
    constexpr size_t HEIGHT = 8u;
    constexpr size_t WIDTH = 3u;

    const Tensor input{ 1.0_dt, 2.0_dt, 3.0_dt, 4.0_dt, 5.0_dt, 0.0_dt, 3.0_dt, 6.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt,  0.0_dt, 0.0_dt,  0.0_dt,  1.0_dt,
                        1.0_dt, 2.0_dt, 3.0_dt, 4.0_dt, 5.0_dt, 2.0_dt, 5.0_dt, 6.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 0.0_dt,  0.0_dt, 0.0_dt,  0.0_dt,  2.0_dt,
                        1.0_dt, 2.0_dt, 3.0_dt, 4.0_dt, 5.0_dt, 0.0_dt, 5.0_dt, 6.0_dt, 9.0_dt, 7.0_dt, 8.0_dt, 10.0_dt, 9.0_dt, 10.0_dt, 11.0_dt, 3.0_dt };
    std::array<uint32_t, 5> reductions{ 1U, 2U, 3U, 4U, 5U };

    const Tensor realOutputs[]{ { 1.0_dt, 2.0_dt, 3.0_dt, 4.0_dt, 5.0_dt, 0.0_dt, 3.0_dt, 6.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt,  0.0_dt, 0.0_dt,  0.0_dt,  1.0_dt,
                                  1.0_dt, 2.0_dt, 3.0_dt, 4.0_dt, 5.0_dt, 2.0_dt, 5.0_dt, 6.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 0.0_dt,  0.0_dt, 0.0_dt,  0.0_dt,  2.0_dt,
                                  1.0_dt, 2.0_dt, 3.0_dt, 4.0_dt, 5.0_dt, 0.0_dt, 5.0_dt, 6.0_dt, 9.0_dt, 7.0_dt, 8.0_dt, 10.0_dt, 9.0_dt, 10.0_dt, 11.0_dt, 3.0_dt },
                                { 4.0_dt, 5.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 1.0_dt, 2.0_dt, 2.0_dt,  5.0_dt,  6.0_dt,
                                  0.0_dt, 0.0_dt, 0.0_dt, 2.0_dt, 3.0_dt, 4.0_dt, 6.0_dt, 9.0_dt, 7.0_dt, 10.0_dt, 11.0_dt, 3.0_dt },
                                { 3.0_dt, 6.0_dt, 0.0_dt, 1.0_dt, 1.0_dt, 2.0_dt, 0.0_dt, 2.0_dt, 1.0_dt, 6.0_dt, 9.0_dt, 7.0_dt },
                                { 0.0_dt, 0.0_dt, 0.0_dt, 2.0_dt, 5.0_dt, 6.0_dt, 2.0_dt, 3.0_dt, 4.0_dt, 10.0_dt, 11.0_dt, 3.0_dt },
                                { 0.0_dt, 0.0_dt, 0.0_dt, 5.0_dt, 0.0_dt, 5.0_dt } };

    // Input
    for (size_t i = 0; i < reductions.size(); ++i)
    {
        MANAGERS_DEFINE
        NETWORK_PARAMS_DEFINE(networkParameters);

        TacotronParams tparams{ { "dummy_input" }, { "dummy_output" }, { "dummy_weights" } };
        tparams.outputsPerStep = reductions[i];
        work.add<DataLayer>("data", DataParams{ { "input" }, DEPTH, HEIGHT, WIDTH });
        work.add<TargetsReductionLayer>("TargetsReduction", BasicParams{ { "input" }, { "output" } }, tparams);

        TENSORS_CREATE(BATCH_SIZE);
        memory_manager["input"] = TORANGE(input);

        ASSERT_NO_THROW(work.forwardPassTraining());

        // Forward checks
        const auto& output = memory_manager["output"];
        EXPECT_EQ(output.size(), realOutputs[i].size());
        for (size_t j = 0; j < output.size(); ++j)
        {
            EXPECT_EQ(output[j], realOutputs[i][j]);
        }

        ASSERT_NO_THROW(work.backwardPassTraining());
    }
}

TEST(TestTargetsReductionLayer, GpuUnit)
{
    PROFILE_TEST

    GPU_ONLY_TEST

    using namespace raul;
    using namespace raul::tacotron;

    constexpr size_t BATCH_SIZE = 2u;
    constexpr size_t DEPTH = 1u;
    constexpr size_t HEIGHT = 8u;
    constexpr size_t WIDTH = 3u;

    const Tensor input{ 1.0_dt, 2.0_dt, 3.0_dt, 4.0_dt, 5.0_dt, 0.0_dt, 3.0_dt, 6.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt,  0.0_dt, 0.0_dt,  0.0_dt,  1.0_dt,
                        1.0_dt, 2.0_dt, 3.0_dt, 4.0_dt, 5.0_dt, 2.0_dt, 5.0_dt, 6.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 0.0_dt,  0.0_dt, 0.0_dt,  0.0_dt,  2.0_dt,
                        1.0_dt, 2.0_dt, 3.0_dt, 4.0_dt, 5.0_dt, 0.0_dt, 5.0_dt, 6.0_dt, 9.0_dt, 7.0_dt, 8.0_dt, 10.0_dt, 9.0_dt, 10.0_dt, 11.0_dt, 3.0_dt };
    std::array<uint32_t, 5> reductions{ 1U, 2U, 3U, 4U, 5U };

    const Tensor realOutputs[]{ { 1.0_dt, 2.0_dt, 3.0_dt, 4.0_dt, 5.0_dt, 0.0_dt, 3.0_dt, 6.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt,  0.0_dt, 0.0_dt,  0.0_dt,  1.0_dt,
                                  1.0_dt, 2.0_dt, 3.0_dt, 4.0_dt, 5.0_dt, 2.0_dt, 5.0_dt, 6.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 0.0_dt,  0.0_dt, 0.0_dt,  0.0_dt,  2.0_dt,
                                  1.0_dt, 2.0_dt, 3.0_dt, 4.0_dt, 5.0_dt, 0.0_dt, 5.0_dt, 6.0_dt, 9.0_dt, 7.0_dt, 8.0_dt, 10.0_dt, 9.0_dt, 10.0_dt, 11.0_dt, 3.0_dt },
                                { 4.0_dt, 5.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 1.0_dt, 2.0_dt, 2.0_dt,  5.0_dt,  6.0_dt,
                                  0.0_dt, 0.0_dt, 0.0_dt, 2.0_dt, 3.0_dt, 4.0_dt, 6.0_dt, 9.0_dt, 7.0_dt, 10.0_dt, 11.0_dt, 3.0_dt },
                                { 3.0_dt, 6.0_dt, 0.0_dt, 1.0_dt, 1.0_dt, 2.0_dt, 0.0_dt, 2.0_dt, 1.0_dt, 6.0_dt, 9.0_dt, 7.0_dt },
                                { 0.0_dt, 0.0_dt, 0.0_dt, 2.0_dt, 5.0_dt, 6.0_dt, 2.0_dt, 3.0_dt, 4.0_dt, 10.0_dt, 11.0_dt, 3.0_dt },
                                { 0.0_dt, 0.0_dt, 0.0_dt, 5.0_dt, 0.0_dt, 5.0_dt } };

    // Input
    for (size_t i = 0; i < reductions.size(); ++i)
    {
        WorkflowEager work(raul::CompressionMode::NONE, raul::CalculationMode::DETERMINISTIC, raul::AllocationMode::STANDARD, raul::ExecutionTarget::GPU);

        TacotronParams tparams{ { "dummy_input" }, { "dummy_output" }, { "dummy_weights" } };
        tparams.outputsPerStep = reductions[i];
        work.add<DataLayer>("data", DataParams{ { "input" }, DEPTH, HEIGHT, WIDTH });
        work.add<TargetsReductionLayer>("TargetsReduction", BasicParams{ { "input" }, { "output" } }, tparams);

        TENSORS_CREATE(BATCH_SIZE);
        auto& memory_manager = work.getMemoryManager<MemoryManagerGPU>();
        memory_manager["input"] = TORANGE(input);

        ASSERT_NO_THROW(work.forwardPassTraining());

        // Forward checks
        const Tensor& output = memory_manager["output"];
        EXPECT_EQ(output.size(), realOutputs[i].size());
        for (size_t j = 0; j < output.size(); ++j)
        {
            EXPECT_EQ(output[j], realOutputs[i][j]);
        }

        ASSERT_NO_THROW(work.backwardPassTraining());
    }
}

}
