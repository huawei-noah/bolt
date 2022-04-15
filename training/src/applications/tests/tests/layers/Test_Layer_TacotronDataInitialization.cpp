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

#include <training/layers/composite/tacotron/TacotronDataInitializationLayer.h>
#include <training/network/Layers.h>
#include <training/network/Workflow.h>

namespace UT
{

TEST(TestLayerTacotronDataInitialization, CpuUnit)
{
    PROFILE_TEST

    // Test parameters
    constexpr size_t BATCH = 3;
    constexpr size_t DEPTH = 4;
    constexpr size_t HEIGHT = 5;
    constexpr size_t WIDTH = 7;

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    work.add<raul::DataLayer>("data", raul::DataParams{ { "x" }, DEPTH, HEIGHT, WIDTH });
    work.add<raul::tacotron::TacotronDataInitializationLayer>("initializer",
                                                              raul::BasicParams{ { "x" }, { "out[0]", "out[1]", "out[2]", "out[3]", "out[4]", "out[5]", "out[6]", "out[7]", "out[8]" } },
                                                              raul::TacotronParams{ { "dummy_input" }, { "dummy_output" }, { "dummy_weights" } });
    TENSORS_CREATE(BATCH);

    ASSERT_NO_THROW(work.forwardPassTraining());

    // Checks
    const auto& alignments = memory_manager["out[2]"];
    for (size_t i = 0; i < alignments.size(); ++i)
    {
        if (i % HEIGHT == 0)
        {
            EXPECT_EQ(alignments[i], 1.0_dt);
        }
        else
        {
            EXPECT_EQ(alignments[i], 0.0_dt);
        }
    }

    ASSERT_NO_THROW(work.backwardPassTraining());
}

TEST(TestLayerTacotronDataInitialization, GpuUnit)
{
    PROFILE_TEST

    GPU_ONLY_TEST

    // Test parameters
    constexpr size_t BATCH = 17;
    constexpr size_t DEPTH = 4;
    constexpr size_t HEIGHT = 13;
    constexpr size_t WIDTH = 7;

    // Initialization
    raul::WorkflowEager work(raul::CompressionMode::NONE, raul::CalculationMode::DETERMINISTIC, raul::AllocationMode::STANDARD, raul::ExecutionTarget::GPU);

    work.add<raul::DataLayer>("data", raul::DataParams{ { "x" }, DEPTH, HEIGHT, WIDTH });
    work.add<raul::tacotron::TacotronDataInitializationLayer>("initializer",
                                                              raul::BasicParams{ { "x" }, { "out[0]", "out[1]", "out[2]", "out[3]", "out[4]", "out[5]", "out[6]", "out[7]", "out[8]" } },
                                                              raul::TacotronParams{ { "dummy_input" }, { "dummy_output" }, { "dummy_weights" } });
    TENSORS_CREATE(BATCH);
    auto& memory_manager = work.getMemoryManager<raul::MemoryManagerGPU>();

    work.forwardPassTraining();

    // Checks
    const raul::Tensor& alignments = memory_manager["out[2]"];
    for (size_t i = 0; i < alignments.size(); ++i)
    {
        if (i % HEIGHT == 0)
        {
            EXPECT_EQ(alignments[i], 1.0_dt);
        }
        else
        {
            EXPECT_EQ(alignments[i], 0.0_dt);
        }
    }

    ASSERT_NO_THROW(work.backwardPassTraining());
}

}