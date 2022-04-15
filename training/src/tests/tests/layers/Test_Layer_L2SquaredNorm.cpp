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

#include <training/base/layers/basic/DataLayer.h>
#include <training/base/layers/basic/L2SquaredNormLayer.h>
#include <training/compiler/Workflow.h>

namespace UT
{

TEST(TestLayerL2SquaredNorm, InputNumExceedsUnit)
{
    PROFILE_TEST

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    // Apply function
    ASSERT_THROW(raul::L2SquaredNormLayer("L2SquaredNorm", raul::BasicParams{ { "x", "y" }, { "x_out" } }, networkParameters), raul::Exception);
}

TEST(TestLayerL2SquaredNorm, OutputNumExceedsUnit)
{
    PROFILE_TEST

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    // Apply function
    ASSERT_THROW(raul::L2SquaredNormLayer("L2SquaredNorm", raul::BasicParams{ { "x" }, { "x_out", "y_out" } }, networkParameters), raul::Exception);
}

// See l2_squared_norm.py
TEST(TestLayerL2SquaredNorm, ForwardUnit)
{
    PROFILE_TEST
    // Test parameters
    const auto batch = 1;
    const auto depth = 1;
    const auto height = 2;
    const auto width = 3;

    const raul::Tensor x{ 1.0_dt, 2.0_dt, 3.0_dt, 4.0_dt, 5.0_dt, 6.0_dt };
    const raul::Tensor realOut{ 45.5_dt };
    // Initialization
    raul::WorkflowEager work(raul::CompressionMode::FP16, raul::CalculationMode::DETERMINISTIC);
    raul::MemoryManager& memory_manager = work.getMemoryManager();
    NETWORK_PARAMS_DEFINE(networkParameters);

    const auto params = raul::BasicParams{ { "x" }, { "out" } };
    memory_manager.createTensor("x", batch, depth, height, width, x);

    // Apply function
    work.add<raul::DataLayer>("data", raul::DataParams{ { "x" }, depth, height, width });
    work.add<raul::L2SquaredNormLayer>("L2SquaredNorm", raul::BasicParams{ { "x" }, { "out" } });
    TENSORS_CREATE(batch);
    memory_manager["x"] = TORANGE(x);

    work.forwardPassTraining();

    // Checks
    const auto output = memory_manager["out"];
    EXPECT_EQ(output[0], realOut[0]);
}

TEST(TestLayerL2SquaredNorm, BackwardUnit)
{
    PROFILE_TEST
    // Test parameters
    const auto batch = 1;
    const auto depth = 1;
    const auto height = 2;
    const auto width = 3;
    const raul::Tensor x{ 1.0_dt, 2.0_dt, 3.0_dt, 4.0_dt, 5.0_dt, 6.0_dt };
    const raul::Tensor realGrad{ 0.7_dt, 1.4_dt, 2.1_dt, 2.8_dt, 3.5_dt, 4.2_dt };

    // Initialization
    raul::WorkflowEager work(raul::CompressionMode::FP16, raul::CalculationMode::DETERMINISTIC);
    raul::MemoryManager& memory_manager = work.getMemoryManager();
    NETWORK_PARAMS_DEFINE(networkParameters);

    // Apply function
    work.add<raul::DataLayer>("data", raul::DataParams{ { "x" }, depth, height, width });
    work.add<raul::L2SquaredNormLayer>("L2SquaredNorm", raul::BasicParams{ { "x" }, { "out" } });
    TENSORS_CREATE(batch);
    memory_manager["x"] = TORANGE(x);
    memory_manager[raul::Name("out").grad()] = 0.7_dt;

    work.forwardPassTraining();
    work.backwardPassTraining();

    // Checks
    const auto xTensorGrad = memory_manager[raul::Name("x").grad()];
    for (size_t q = 0; q < xTensorGrad.size(); ++q)
    {
        EXPECT_EQ(xTensorGrad[q], realGrad[q]);
    }
}

}