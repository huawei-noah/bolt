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
#include <training/base/layers/basic/RollLayer.h>
#include <training/compiler/Layers.h>
#include <training/compiler/Workflow.h>

namespace UT
{

TEST(TestLayerRoll, ForwardNotCycledUnit)
{
    PROFILE_TEST
    // Test parameters
    const auto batch = 2;
    const auto depth = 1;
    const auto height = 4;
    const auto width = 3;

    const raul::Tensor x{ 1.0_dt,  2.0_dt,  3.0_dt,  4.0_dt,  5.0_dt,  6.0_dt,  7.0_dt,  8.0_dt,  9.0_dt,  10.0_dt, 11.0_dt, 12.0_dt,
                          13.0_dt, 14.0_dt, 15.0_dt, 16.0_dt, 17.0_dt, 18.0_dt, 19.0_dt, 20.0_dt, 21.0_dt, 22.0_dt, 23.0_dt, 24.0_dt };

    const raul::Tensor realOut[]{ { 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt,  1.0_dt,  1.0_dt,
                                    1.0_dt, 2.0_dt, 3.0_dt, 4.0_dt, 5.0_dt, 6.0_dt, 7.0_dt, 8.0_dt, 9.0_dt, 10.0_dt, 11.0_dt, 12.0_dt },
                                  { 1.0_dt,  2.0_dt,  3.0_dt,  4.0_dt,  5.0_dt,  6.0_dt,  7.0_dt,  8.0_dt,  9.0_dt,  10.0_dt, 11.0_dt, 12.0_dt,
                                    13.0_dt, 14.0_dt, 15.0_dt, 16.0_dt, 17.0_dt, 18.0_dt, 19.0_dt, 20.0_dt, 21.0_dt, 22.0_dt, 23.0_dt, 24.0_dt },
                                  { 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt,  2.0_dt,  3.0_dt,
                                    1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 13.0_dt, 14.0_dt, 15.0_dt },
                                  { 1.0_dt, 1.0_dt,  2.0_dt,  1.0_dt, 4.0_dt,  5.0_dt,  1.0_dt, 7.0_dt,  8.0_dt,  1.0_dt, 10.0_dt, 11.0_dt,
                                    1.0_dt, 13.0_dt, 14.0_dt, 1.0_dt, 16.0_dt, 17.0_dt, 1.0_dt, 19.0_dt, 20.0_dt, 1.0_dt, 22.0_dt, 23.0_dt } };

    const raul::Dimension dimensions[]{ raul::Dimension::Batch, raul::Dimension::Depth, raul::Dimension::Height, raul::Dimension::Width };
    std::vector<size_t> shifts = { 1, 2, 3, 4 };

    for (size_t iter = 0; iter < std::size(dimensions); ++iter)
    {
        // Initialization
        MANAGERS_DEFINE
        NETWORK_PARAMS_DEFINE(networkParameters);
        work.add<raul::DataLayer>("data", raul::DataParams{ { "x" }, depth, height, width });

        // Apply function
        raul::RollLayer roller("roll", raul::RollLayerParams{ { "x" }, { "out" }, dimensions[iter], shifts[iter], false }, networkParameters);
        TENSORS_CREATE(batch);
        memory_manager["x"] = TORANGE(x);

        roller.forwardCompute(raul::NetworkMode::Test);

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

TEST(TestLayerRoll, ForwardCycledUnit)
{
    PROFILE_TEST
    // Test parameters
    const auto batch = 2;
    const auto depth = 1;
    const auto height = 4;
    const auto width = 3;

    const raul::Tensor x{ 1.0_dt,  2.0_dt,  3.0_dt,  4.0_dt,  5.0_dt,  6.0_dt,  7.0_dt,  8.0_dt,  9.0_dt,  10.0_dt, 11.0_dt, 12.0_dt,
                          13.0_dt, 14.0_dt, 15.0_dt, 16.0_dt, 17.0_dt, 18.0_dt, 19.0_dt, 20.0_dt, 21.0_dt, 22.0_dt, 23.0_dt, 24.0_dt };

    const raul::Tensor realOut[]{ { 13.0_dt, 14.0_dt, 15.0_dt, 16.0_dt, 17.0_dt, 18.0_dt, 19.0_dt, 20.0_dt, 21.0_dt, 22.0_dt, 23.0_dt, 24.0_dt,
                                    1.0_dt,  2.0_dt,  3.0_dt,  4.0_dt,  5.0_dt,  6.0_dt,  7.0_dt,  8.0_dt,  9.0_dt,  10.0_dt, 11.0_dt, 12.0_dt },
                                  { 1.0_dt,  2.0_dt,  3.0_dt,  4.0_dt,  5.0_dt,  6.0_dt,  7.0_dt,  8.0_dt,  9.0_dt,  10.0_dt, 11.0_dt, 12.0_dt,
                                    13.0_dt, 14.0_dt, 15.0_dt, 16.0_dt, 17.0_dt, 18.0_dt, 19.0_dt, 20.0_dt, 21.0_dt, 22.0_dt, 23.0_dt, 24.0_dt },
                                  { 4.0_dt,  5.0_dt,  6.0_dt,  7.0_dt,  8.0_dt,  9.0_dt,  10.0_dt, 11.0_dt, 12.0_dt, 1.0_dt,  2.0_dt,  3.0_dt,
                                    16.0_dt, 17.0_dt, 18.0_dt, 19.0_dt, 20.0_dt, 21.0_dt, 22.0_dt, 23.0_dt, 24.0_dt, 13.0_dt, 14.0_dt, 15.0_dt },
                                  { 3.0_dt,  1.0_dt,  2.0_dt,  6.0_dt,  4.0_dt,  5.0_dt,  9.0_dt,  7.0_dt,  8.0_dt,  12.0_dt, 10.0_dt, 11.0_dt,
                                    15.0_dt, 13.0_dt, 14.0_dt, 18.0_dt, 16.0_dt, 17.0_dt, 21.0_dt, 19.0_dt, 20.0_dt, 24.0_dt, 22.0_dt, 23.0_dt } };

    const raul::Dimension dimensions[]{ raul::Dimension::Batch, raul::Dimension::Depth, raul::Dimension::Height, raul::Dimension::Width };
    std::vector<size_t> shifts = { 1, 2, 3, 4 };

    for (size_t iter = 0; iter < std::size(dimensions); ++iter)
    {
        // Initialization
        MANAGERS_DEFINE
        NETWORK_PARAMS_DEFINE(networkParameters);
        work.add<raul::DataLayer>("data", raul::DataParams{ { "x" }, depth, height, width });

        // Apply function
        raul::RollLayer roller("roll", raul::RollLayerParams{ { "x" }, { "out" }, dimensions[iter], shifts[iter] }, networkParameters);
        TENSORS_CREATE(batch);
        memory_manager["x"] = TORANGE(x);

        roller.forwardCompute(raul::NetworkMode::Test);

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

TEST(TestLayerRoll, BackwardUnit)
{
    PROFILE_TEST
    // Test parameters
    const auto batch = 2;
    const auto depth = 1;
    const auto height = 4;
    const auto width = 3;

    const raul::Tensor x{ 1.0_dt,  2.0_dt,  3.0_dt,  4.0_dt,  5.0_dt,  6.0_dt,  7.0_dt,  8.0_dt,  9.0_dt,  10.0_dt, 11.0_dt, 12.0_dt,
                          13.0_dt, 14.0_dt, 15.0_dt, 16.0_dt, 17.0_dt, 18.0_dt, 19.0_dt, 20.0_dt, 21.0_dt, 22.0_dt, 23.0_dt, 24.0_dt };

    const raul::Tensor deltas{ 0.6646448374_dt, 0.3886462450_dt, 0.0786520243_dt, 0.1446506977_dt, 0.1808730364_dt, 0.6439573765_dt, 0.1503965855_dt, 0.7280383706_dt,
                               0.8867152333_dt, 0.2971339822_dt, 0.7827857733_dt, 0.5104721189_dt, 0.8187109828_dt, 0.4369543791_dt, 0.1877676845_dt, 0.8780589104_dt,
                               0.1925331354_dt, 0.6161287427_dt, 0.7849457264_dt, 0.1380746961_dt, 0.0454765558_dt, 0.7794026732_dt, 0.0058631897_dt, 0.1268088222_dt };
    const raul::Tensor realGrad[]{ { 0.8187109828_dt, 0.4369543791_dt, 0.1877676845_dt, 0.8780589104_dt, 0.1925331354_dt, 0.6161287427_dt, 0.7849457264_dt, 0.1380746961_dt,
                                     0.0454765558_dt, 0.7794026732_dt, 0.0058631897_dt, 0.1268088222_dt, 0.6646448374_dt, 0.3886462450_dt, 0.0786520243_dt, 0.1446506977_dt,
                                     0.1808730364_dt, 0.6439573765_dt, 0.1503965855_dt, 0.7280383706_dt, 0.8867152333_dt, 0.2971339822_dt, 0.7827857733_dt, 0.5104721189_dt },
                                   { 0.6646448374_dt, 0.3886462450_dt, 0.0786520243_dt, 0.1446506977_dt, 0.1808730364_dt, 0.6439573765_dt, 0.1503965855_dt, 0.7280383706_dt,
                                     0.8867152333_dt, 0.2971339822_dt, 0.7827857733_dt, 0.5104721189_dt, 0.8187109828_dt, 0.4369543791_dt, 0.1877676845_dt, 0.8780589104_dt,
                                     0.1925331354_dt, 0.6161287427_dt, 0.7849457264_dt, 0.1380746961_dt, 0.0454765558_dt, 0.7794026732_dt, 0.0058631897_dt, 0.1268088222_dt },
                                   { 0.2971339822_dt, 0.7827857733_dt, 0.5104721189_dt, 0.6646448374_dt, 0.3886462450_dt, 0.0786520243_dt, 0.1446506977_dt, 0.1808730364_dt,
                                     0.6439573765_dt, 0.1503965855_dt, 0.7280383706_dt, 0.8867152333_dt, 0.7794026732_dt, 0.0058631897_dt, 0.1268088222_dt, 0.8187109828_dt,
                                     0.4369543791_dt, 0.1877676845_dt, 0.8780589104_dt, 0.1925331354_dt, 0.6161287427_dt, 0.7849457264_dt, 0.1380746961_dt, 0.0454765558_dt },
                                   { 0.3886462450_dt, 0.0786520243_dt, 0.6646448374_dt, 0.1808730364_dt, 0.6439573765_dt, 0.1446506977_dt, 0.7280383706_dt, 0.8867152333_dt,
                                     0.1503965855_dt, 0.7827857733_dt, 0.5104721189_dt, 0.2971339822_dt, 0.4369543791_dt, 0.1877676845_dt, 0.8187109828_dt, 0.1925331354_dt,
                                     0.6161287427_dt, 0.8780589104_dt, 0.1380746961_dt, 0.0454765558_dt, 0.7849457264_dt, 0.0058631897_dt, 0.1268088222_dt, 0.7794026732_dt } };

    const raul::Dimension dimensions[]{ raul::Dimension::Batch, raul::Dimension::Depth, raul::Dimension::Height, raul::Dimension::Width };
    std::vector<size_t> shifts = { 1, 2, 3, 4 };

    for (size_t iter = 0; iter < std::size(dimensions); ++iter)
    {
        // Initialization
        MANAGERS_DEFINE
        NETWORK_PARAMS_DEFINE(networkParameters);
        work.add<raul::DataLayer>("data", raul::DataParams{ { "x" }, depth, height, width });

        // Apply function
        raul::RollLayer roller("roll", raul::RollLayerParams{ { "x" }, { "out" }, dimensions[iter], shifts[iter] }, networkParameters);
        TENSORS_CREATE(batch);
        memory_manager["x"] = TORANGE(x);
        memory_manager[raul::Name("out").grad()] = TORANGE(deltas);

        roller.forwardCompute(raul::NetworkMode::Test);
        roller.backwardCompute();

        // Checks
        const auto& xTensor = memory_manager["x"];
        const auto& xNablaTensor = memory_manager[raul::Name("x").grad()];

        EXPECT_EQ(xNablaTensor.size(), xTensor.size());
        EXPECT_EQ(xNablaTensor.size(), realGrad[iter].size());
        for (size_t i = 0; i < xNablaTensor.size(); ++i)
        {
            EXPECT_EQ(xNablaTensor[i], realGrad[iter][i]);
        }
    }
}

}