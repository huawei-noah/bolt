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
#include <training/base/layers/basic/IndexFillLayer.h>
#include <training/compiler/Workflow.h>

namespace UT
{

TEST(TestLayerIndexFill, ForwardIncorrectIndexUnit)
{
    PROFILE_TEST
    // Test parameters
    const size_t batch = 1;
    const size_t depth = 2;
    const size_t height = 1;
    const size_t width = 1;

    const raul::Tensor x{ 1.0_dt, 1.0_dt };

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    work.tensorNeeded("x", "x", raul::WShape{ raul::BS(), depth, height, width }, DEC_FORW_READ_NOMEMOPT);

    // Apply function
    raul::IndexFillLayer filler("fill", raul::IndexFillLayerParams{ { "x" }, { "out" }, raul::Dimension::Batch, std::unordered_set<size_t>{ 2 }, 1.0_dt }, networkParameters);
    TENSORS_CREATE(batch);
    memory_manager["x"] = TORANGE(x);

    ASSERT_THROW(filler.forwardCompute(raul::NetworkMode::Test), raul::Exception);
}

TEST(TestLayerIndexFill, ForwardUnit)
{
    PROFILE_TEST
    // Test parameters
    const size_t batch = 1;
    const size_t depth = 2;
    const size_t height = 3;
    const size_t width = 4;

    const raul::Tensor x{ 0.2364_dt, 0.2266_dt, 0.8005_dt, 0.1692_dt, 0.2650_dt, 0.7720_dt, 0.1282_dt, 0.7452_dt, 0.8045_dt, 0.6357_dt, 0.5896_dt, 0.6933_dt,
                          0.8782_dt, 0.5407_dt, 0.1400_dt, 0.9613_dt, 0.8666_dt, 0.4884_dt, 0.2077_dt, 0.3063_dt, 0.0585_dt, 0.8314_dt, 0.4566_dt, 0.8445_dt };

    const raul::Tensor realOut[]{ { 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt,
                                    1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt },
                                  { 0.23640_dt, 0.22660_dt, 0.80050_dt, 0.16920_dt, 0.26500_dt, 0.77200_dt, 0.12820_dt, 0.74520_dt, 0.80450_dt, 0.63570_dt, 0.58960_dt, 0.69330_dt,
                                    1.0_dt,     1.0_dt,     1.0_dt,     1.0_dt,     1.0_dt,     1.0_dt,     1.0_dt,     1.0_dt,     1.0_dt,     1.0_dt,     1.0_dt,     1.0_dt },
                                  { 0.23640_dt, 0.22660_dt, 0.80050_dt, 0.16920_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt,
                                    0.87820_dt, 0.54070_dt, 0.14000_dt, 0.96130_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt },
                                  { 0.23640_dt, 1.0_dt, 1.0_dt, 1.0_dt, 0.26500_dt, 1.0_dt, 1.0_dt, 1.0_dt, 0.80450_dt, 1.0_dt, 1.0_dt, 1.0_dt,
                                    0.87820_dt, 1.0_dt, 1.0_dt, 1.0_dt, 0.86660_dt, 1.0_dt, 1.0_dt, 1.0_dt, 0.05850_dt, 1.0_dt, 1.0_dt, 1.0_dt } };

    const raul::Dimension dimensions[]{ raul::Dimension::Batch, raul::Dimension::Depth, raul::Dimension::Height, raul::Dimension::Width };
    std::vector<std::unordered_set<size_t>> indices = { { 0 }, { 1 }, { 1, 2 }, { 1, 2, 3 } };

    for (size_t iter = 0; iter < std::size(dimensions); ++iter)
    {
        MANAGERS_DEFINE
        NETWORK_PARAMS_DEFINE(networkParameters);

        work.tensorNeeded("x", "x", raul::WShape{ raul::BS(), depth, height, width }, DEC_FORW_READ_NOMEMOPT);

        // Apply function
        raul::IndexFillLayer filler("fill", raul::IndexFillLayerParams{ { "x" }, { "out" }, dimensions[iter], indices[iter], 1.0_dt }, networkParameters);
        TENSORS_CREATE(batch);
        memory_manager["x"] = TORANGE(x);

        filler.forwardCompute(raul::NetworkMode::Test);

        // Checks
        const auto& xTensor = memory_manager["x"];
        const auto& outTensor = memory_manager["out"];

        EXPECT_EQ(outTensor.size(), xTensor.size());
        EXPECT_EQ(outTensor.size(), realOut[iter].size());
        for (size_t i = 0; i < outTensor.size(); ++i)
        {
            EXPECT_EQ(outTensor[i], realOut[iter][i]);
        }
    }
}

TEST(TestLayerIndexFill, BackwardUnit)
{
    PROFILE_TEST
    // Test parameters
    const size_t batch = 1;
    const size_t depth = 2;
    const size_t height = 3;
    const size_t width = 4;

    const raul::Tensor x{ 0.2364_dt, 0.2266_dt, 0.8005_dt, 0.1692_dt, 0.2650_dt, 0.7720_dt, 0.1282_dt, 0.7452_dt, 0.8045_dt, 0.6357_dt, 0.5896_dt, 0.6933_dt,
                          0.8782_dt, 0.5407_dt, 0.1400_dt, 0.9613_dt, 0.8666_dt, 0.4884_dt, 0.2077_dt, 0.3063_dt, 0.0585_dt, 0.8314_dt, 0.4566_dt, 0.8445_dt };

    const raul::Tensor deltas{ 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt,
                               1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt };
    const raul::Tensor realGrad[]{ { 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt,
                                     0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt },
                                   { 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt,
                                     0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt },
                                   { 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt,
                                     1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt },
                                   { 1.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 0.0_dt,
                                     1.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 0.0_dt } };

    const raul::Dimension dimensions[]{ raul::Dimension::Batch, raul::Dimension::Depth, raul::Dimension::Height, raul::Dimension::Width };
    std::vector<std::unordered_set<size_t>> indices = { { 0 }, { 1 }, { 1, 2 }, { 1, 2, 3 } };

    for (size_t iter = 0; iter < std::size(dimensions); ++iter)
    {
        MANAGERS_DEFINE
        NETWORK_PARAMS_DEFINE(networkParameters);

        work.tensorNeeded("x", "x", raul::WShape{ raul::BS(), depth, height, width }, DEC_FORW_READ_NOMEMOPT);

        // Apply function
        raul::IndexFillLayer filler("fill", raul::IndexFillLayerParams{ { "x" }, { "out" }, dimensions[iter], indices[iter], 1.0_dt }, networkParameters);
        TENSORS_CREATE(batch);
        memory_manager["x"] = TORANGE(x);
        memory_manager[raul::Name("out").grad()] = TORANGE(deltas);

        filler.forwardCompute(raul::NetworkMode::Test);
        filler.backwardCompute();

        // Checks
        const auto& xTensor = memory_manager["x"];
        const auto& xNablaTensor = memory_manager[raul::Name("x").grad()];

        EXPECT_EQ(xNablaTensor.size(), xTensor.size());
        EXPECT_EQ(xNablaTensor.size(), realGrad[iter].size());
        for (size_t i = 0; i < xNablaTensor.size(); ++i)
        {
            EXPECT_EQ(xNablaTensor[i], realGrad[iter][i]);
        }
        memory_manager.clear();
    }
}

}