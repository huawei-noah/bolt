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
#include <training/base/layers/basic/ReduceNonZeroLayer.h>
#include <training/compiler/Workflow.h>

#include <random>

namespace UT
{

TEST(TestLayerReduceNonZero, InputNumExceedsUnit)
{
    PROFILE_TEST

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    // Apply function
    ASSERT_THROW(raul::ReduceNonZeroLayer("nonzero", raul::BasicParamsWithDim{ { "x", "y" }, { "x_out" } }, networkParameters), raul::Exception);
}

TEST(TestLayerReduceNonZero, OutputNumExceedsUnit)
{
    PROFILE_TEST

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    // Apply function
    ASSERT_THROW(raul::ReduceNonZeroLayer("nonzero", raul::BasicParamsWithDim{ { "x", "y" }, { "x_out" } }, networkParameters), raul::Exception);
}

TEST(TestLayerReduceNonZero, ForwardUnit)
{
    PROFILE_TEST
    // Test parameters
    const auto batch = 1;
    const auto depth = 2;
    const auto height = 3;
    const auto width = 4;

    // See reduce_sum.py
    const raul::Tensor x{ 0.1_dt, 0.0_dt,        0.2_dt,        0.13203049_dt, 0.0_dt,        0.63407868_dt, 0.49009341_dt, 0.89644474_dt, 0.45562798_dt, 0.63230628_dt, 0.34889346_dt, 0.51909077_dt,
                          0.0_dt, 0.81018829_dt, 0.98009706_dt, 0.11468822_dt, 0.31676513_dt, 0.69650495_dt, 0.91427469_dt, 0.0_dt,        0.94117838_dt, 0.59950727_dt, 0.0_dt,        0.0_dt };

    std::string dimensions[] = { "default", "batch", "depth", "height", "width" };

    raul::Tensor realOutputs[] = { { 18.0_dt },
                                   { 1.0_dt, 0.0_dt, 1.0_dt, 1.0_dt, 0.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt,
                                     0.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 0.0_dt, 1.0_dt, 1.0_dt, 0.0_dt, 0.0_dt },
                                   { 1.0_dt, 1.0_dt, 2.0_dt, 2.0_dt, 1.0_dt, 2.0_dt, 2.0_dt, 1.0_dt, 2.0_dt, 2.0_dt, 1.0_dt, 1.0_dt },
                                   { 2.0_dt, 2.0_dt, 3.0_dt, 3.0_dt, 2.0_dt, 3.0_dt, 2.0_dt, 1.0_dt },
                                   { 3.0_dt, 3.0_dt, 4.0_dt, 3.0_dt, 3.0_dt, 2.0_dt } };

    for (size_t iter = 0; iter < std::size(dimensions); ++iter)
    {
        MANAGERS_DEFINE
        NETWORK_PARAMS_DEFINE(networkParameters);

        work.add<raul::DataLayer>("data", raul::DataParams{ { "x" }, depth, height, width });
        work.add<raul::ReduceNonZeroLayer>("rnonzero", raul::BasicParamsWithDim{ { "x" }, { "out" }, dimensions[iter] });
        TENSORS_CREATE(batch);
        memory_manager["x"] = TORANGE(x);

        work.forwardPassTraining();

        // Checks
        const auto& outTensor = memory_manager["out"];
        EXPECT_EQ(outTensor.size(), realOutputs[iter].size());
        for (size_t i = 0; i < outTensor.size(); ++i)
        {
            EXPECT_EQ(outTensor[i], realOutputs[iter][i]);
        }
    }
}

TEST(TestLayerReduceNonZero, BackwardUnit)
{
    PROFILE_TEST
    // Test parameters
    const auto batch = 1;
    const auto depth = 2;
    const auto height = 3;
    const auto width = 4;

    // See reduce_sum.py
    const raul::Tensor x{ 0.1_dt, 0.0_dt,        0.2_dt,        0.13203049_dt, 0.0_dt,        0.63407868_dt, 0.49009341_dt, 0.89644474_dt, 0.45562798_dt, 0.63230628_dt, 0.34889346_dt, 0.51909077_dt,
                          0.0_dt, 0.81018829_dt, 0.98009706_dt, 0.11468822_dt, 0.31676513_dt, 0.69650495_dt, 0.91427469_dt, 0.0_dt,        0.94117838_dt, 0.59950727_dt, 0.0_dt,        0.0_dt };

    std::string dimensions[] = { "default", "batch", "depth", "height", "width" };

    const raul::Tensor realGrad = raul::Tensor("realGrad", batch, depth, height, width, 1.0_dt);

    for (size_t iter = 0; iter < std::size(dimensions); ++iter)
    {
        MANAGERS_DEFINE
        NETWORK_PARAMS_DEFINE(networkParameters);

        work.add<raul::DataLayer>("data", raul::DataParams{ { "x" }, depth, height, width });
        work.add<raul::ReduceNonZeroLayer>("rnonzero", raul::BasicParamsWithDim{ { "x" }, { "out" }, dimensions[iter] });
        TENSORS_CREATE(batch);
        memory_manager["x"] = TORANGE(x);
        memory_manager[raul::Name("out").grad()] = 1.0_dt;

        work.forwardPassTraining();
        work.backwardPassTraining();

        // Checks
        const auto& x_tensor_grad = memory_manager[raul::Name("x").grad()];
        EXPECT_EQ(x_tensor_grad.getShape(), memory_manager["x"].getShape());
        for (size_t i = 0; i < x_tensor_grad.size(); ++i)
        {
            EXPECT_EQ(x_tensor_grad[i], realGrad[i]);
        }
    }
}

}