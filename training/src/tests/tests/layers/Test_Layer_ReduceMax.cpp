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
#include <training/base/layers/basic/DataLayer.h>
#include <training/base/layers/basic/ReduceMaxLayer.h>
#include <training/compiler/Workflow.h>

namespace UT
{

TEST(TestLayerReduceMax, InputNumExceedsUnit)
{
    PROFILE_TEST

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    // Apply function
    ASSERT_THROW(raul::ReduceMaxLayer("rmax", raul::BasicParamsWithDim{ { "x", "y" }, { "x_out" } }, networkParameters), raul::Exception);
}

TEST(TestLayerReduceMax, OutputNumExceedsUnit)
{
    PROFILE_TEST

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    // Apply function
    ASSERT_THROW(raul::ReduceMaxLayer("rmax", raul::BasicParamsWithDim{ { "x" }, { "x_out", "y_out" } }, networkParameters), raul::Exception);
}

TEST(TestLayerReduceMax, ForwardUnit)
{
    PROFILE_TEST
    // Test parameters
    const auto batch = 2;
    const auto depth = 3;
    const auto height = 4;
    const auto width = 5;

    // See reduce_max.py
    const raul::Tensor x{ 0.6645621_dt,  0.44100678_dt, 0.3528825_dt,  0.46448255_dt, 0.03366041_dt, 0.68467236_dt,  0.74011743_dt, 0.8724445_dt,  0.22632635_dt, 0.22319686_dt,  0.3103881_dt,
                          0.7223358_dt,  0.13318717_dt, 0.5480639_dt,  0.5746088_dt,  0.8996835_dt,  0.009463668_dt, 0.5212307_dt,  0.6345445_dt,  0.1993283_dt,  0.72942245_dt,  0.54583454_dt,
                          0.10756552_dt, 0.6767061_dt,  0.6602763_dt,  0.33695042_dt, 0.60141766_dt, 0.21062577_dt,  0.8527372_dt,  0.44062173_dt, 0.9485276_dt,  0.23752594_dt,  0.81179297_dt,
                          0.5263394_dt,  0.494308_dt,   0.21612847_dt, 0.8457197_dt,  0.8718841_dt,  0.3083862_dt,   0.6868038_dt,  0.23764038_dt, 0.7817228_dt,  0.9671384_dt,   0.068701625_dt,
                          0.79873943_dt, 0.66028714_dt, 0.5871513_dt,  0.16461694_dt, 0.7381023_dt,  0.32054043_dt,  0.6073899_dt,  0.46523476_dt, 0.97803545_dt, 0.7223145_dt,   0.32347047_dt,
                          0.82577336_dt, 0.4976915_dt,  0.19483674_dt, 0.7588748_dt,  0.3380444_dt,  0.28128064_dt,  0.31513572_dt, 0.60670924_dt, 0.7498598_dt,  0.5016055_dt,   0.18282163_dt,
                          0.13179815_dt, 0.64636123_dt, 0.9559475_dt,  0.6670735_dt,  0.30755532_dt, 0.36892188_dt,  0.44735897_dt, 0.18359458_dt, 0.5288255_dt,  0.7052754_dt,   0.898633_dt,
                          0.31386292_dt, 0.62338257_dt, 0.96815526_dt, 0.11207926_dt, 0.29590535_dt, 0.9356605_dt,   0.1341263_dt,  0.31937933_dt, 0.262277_dt,   0.031487584_dt, 0.90045524_dt,
                          0.6409379_dt,  0.5821855_dt,  0.20917094_dt, 0.71736085_dt, 0.363523_dt,   0.04670918_dt,  0.14977789_dt, 0.84361756_dt, 0.9355587_dt,  0.09517312_dt,  0.08617878_dt,
                          0.6247839_dt,  0.37050653_dt, 0.5139042_dt,  0.6233207_dt,  0.8024682_dt,  0.1665138_dt,   0.22090447_dt, 0.62422717_dt, 0.08719146_dt, 0.92142665_dt,  0.9348017_dt,
                          0.60455227_dt, 0.47940433_dt, 0.14430141_dt, 0.32600033_dt, 0.92557526_dt, 0.7757342_dt,   0.636765_dt,   0.6282351_dt,  0.35401833_dt, 0.41446733_dt };

    std::string dimensions[] = { "default", "batch", "depth", "height", "width" };

    raul::Tensor realOutputs[] = {
        { 0.97803545_dt },
        { 0.6645621_dt,  0.44100678_dt, 0.60670924_dt, 0.7498598_dt,  0.5016055_dt,  0.68467236_dt, 0.74011743_dt, 0.8724445_dt,  0.9559475_dt,  0.6670735_dt,  0.3103881_dt,  0.7223358_dt,
          0.44735897_dt, 0.5480639_dt,  0.5746088_dt,  0.8996835_dt,  0.898633_dt,   0.5212307_dt,  0.6345445_dt,  0.96815526_dt, 0.72942245_dt, 0.54583454_dt, 0.9356605_dt,  0.6767061_dt,
          0.6602763_dt,  0.33695042_dt, 0.60141766_dt, 0.90045524_dt, 0.8527372_dt,  0.5821855_dt,  0.9485276_dt,  0.71736085_dt, 0.81179297_dt, 0.5263394_dt,  0.494308_dt,   0.84361756_dt,
          0.9355587_dt,  0.8718841_dt,  0.3083862_dt,  0.6868038_dt,  0.37050653_dt, 0.7817228_dt,  0.9671384_dt,  0.8024682_dt,  0.79873943_dt, 0.66028714_dt, 0.62422717_dt, 0.16461694_dt,
          0.92142665_dt, 0.9348017_dt,  0.6073899_dt,  0.47940433_dt, 0.97803545_dt, 0.7223145_dt,  0.92557526_dt, 0.82577336_dt, 0.636765_dt,   0.6282351_dt,  0.7588748_dt,  0.41446733_dt },
        { 0.72942245_dt, 0.7817228_dt,  0.9671384_dt,  0.6767061_dt,  0.79873943_dt, 0.68467236_dt, 0.74011743_dt, 0.8724445_dt,  0.8527372_dt,  0.44062173_dt,
          0.9485276_dt,  0.7223358_dt,  0.97803545_dt, 0.7223145_dt,  0.5746088_dt,  0.8996835_dt,  0.8457197_dt,  0.8718841_dt,  0.7588748_dt,  0.6868038_dt,
          0.37050653_dt, 0.5139042_dt,  0.9356605_dt,  0.8024682_dt,  0.5016055_dt,  0.262277_dt,   0.62422717_dt, 0.90045524_dt, 0.9559475_dt,  0.9348017_dt,
          0.60455227_dt, 0.71736085_dt, 0.44735897_dt, 0.32600033_dt, 0.92557526_dt, 0.84361756_dt, 0.9355587_dt,  0.6282351_dt,  0.62338257_dt, 0.96815526_dt },
        { 0.8996835_dt,  0.74011743_dt, 0.8724445_dt,  0.6345445_dt, 0.5746088_dt,  0.9485276_dt, 0.8457197_dt, 0.8718841_dt,  0.8527372_dt,  0.6868038_dt,
          0.82577336_dt, 0.7817228_dt,  0.97803545_dt, 0.7588748_dt, 0.79873943_dt, 0.7052754_dt, 0.898633_dt,  0.64636123_dt, 0.9559475_dt,  0.96815526_dt,
          0.84361756_dt, 0.9355587_dt,  0.9356605_dt,  0.6409379_dt, 0.6247839_dt,  0.7757342_dt, 0.636765_dt,  0.6282351_dt,  0.92142665_dt, 0.9348017_dt },
        { 0.6645621_dt, 0.8724445_dt, 0.7223358_dt, 0.8996835_dt,  0.72942245_dt, 0.8527372_dt,  0.9485276_dt,  0.8718841_dt, 0.9671384_dt, 0.7381023_dt, 0.97803545_dt, 0.82577336_dt,
          0.7498598_dt, 0.9559475_dt, 0.5288255_dt, 0.96815526_dt, 0.9356605_dt,  0.90045524_dt, 0.71736085_dt, 0.9355587_dt, 0.8024682_dt, 0.9348017_dt, 0.92557526_dt, 0.7757342_dt }
    };

    for (size_t iter = 0; iter < std::size(dimensions); ++iter)
    {
        MANAGERS_DEFINE
        NETWORK_PARAMS_DEFINE(networkParameters);

        work.add<raul::DataLayer>("data", raul::DataParams{ { "x" }, depth, height, width });
        work.add<raul::ReduceMaxLayer>("rmax", raul::BasicParamsWithDim{ { "x" }, { "out" }, dimensions[iter] });
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

TEST(TestLayerReduceMax, BackwardUnit)
{
    PROFILE_TEST
    // Test parameters
    const auto batch = 2;
    const auto depth = 3;
    const auto height = 4;
    const auto width = 5;

    // See reduce_max.py
    const raul::Tensor x{ 0.6645621_dt,  0.44100678_dt, 0.3528825_dt,  0.46448255_dt, 0.03366041_dt, 0.68467236_dt,  0.74011743_dt, 0.8724445_dt,  0.22632635_dt, 0.22319686_dt,  0.3103881_dt,
                          0.7223358_dt,  0.13318717_dt, 0.5480639_dt,  0.5746088_dt,  0.8996835_dt,  0.009463668_dt, 0.5212307_dt,  0.6345445_dt,  0.1993283_dt,  0.72942245_dt,  0.54583454_dt,
                          0.10756552_dt, 0.6767061_dt,  0.6602763_dt,  0.33695042_dt, 0.60141766_dt, 0.21062577_dt,  0.8527372_dt,  0.44062173_dt, 0.9485276_dt,  0.23752594_dt,  0.81179297_dt,
                          0.5263394_dt,  0.494308_dt,   0.21612847_dt, 0.8457197_dt,  0.8718841_dt,  0.3083862_dt,   0.6868038_dt,  0.23764038_dt, 0.7817228_dt,  0.9671384_dt,   0.068701625_dt,
                          0.79873943_dt, 0.66028714_dt, 0.5871513_dt,  0.16461694_dt, 0.7381023_dt,  0.32054043_dt,  0.6073899_dt,  0.46523476_dt, 0.97803545_dt, 0.7223145_dt,   0.32347047_dt,
                          0.82577336_dt, 0.4976915_dt,  0.19483674_dt, 0.7588748_dt,  0.3380444_dt,  0.28128064_dt,  0.31513572_dt, 0.60670924_dt, 0.7498598_dt,  0.5016055_dt,   0.18282163_dt,
                          0.13179815_dt, 0.64636123_dt, 0.9559475_dt,  0.6670735_dt,  0.30755532_dt, 0.36892188_dt,  0.44735897_dt, 0.18359458_dt, 0.5288255_dt,  0.7052754_dt,   0.898633_dt,
                          0.31386292_dt, 0.62338257_dt, 0.96815526_dt, 0.11207926_dt, 0.29590535_dt, 0.9356605_dt,   0.1341263_dt,  0.31937933_dt, 0.262277_dt,   0.031487584_dt, 0.90045524_dt,
                          0.6409379_dt,  0.5821855_dt,  0.20917094_dt, 0.71736085_dt, 0.363523_dt,   0.04670918_dt,  0.14977789_dt, 0.84361756_dt, 0.9355587_dt,  0.09517312_dt,  0.08617878_dt,
                          0.6247839_dt,  0.37050653_dt, 0.5139042_dt,  0.6233207_dt,  0.8024682_dt,  0.1665138_dt,   0.22090447_dt, 0.62422717_dt, 0.08719146_dt, 0.92142665_dt,  0.9348017_dt,
                          0.60455227_dt, 0.47940433_dt, 0.14430141_dt, 0.32600033_dt, 0.92557526_dt, 0.7757342_dt,   0.636765_dt,   0.6282351_dt,  0.35401833_dt, 0.41446733_dt };

    std::string dimensions[] = { "default", "batch", "depth", "height", "width" };

    raul::Tensor realGrads[] = { { 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt,
                                   0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt,
                                   0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt,
                                   0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt,
                                   0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt,
                                   0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt },
                                 { 1.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 1.0_dt, 0.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 0.0_dt, 1.0_dt, 1.0_dt, 0.0_dt,
                                   1.0_dt, 1.0_dt, 0.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 0.0_dt, 1.0_dt, 0.0_dt, 1.0_dt, 0.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 1.0_dt, 1.0_dt,
                                   0.0_dt, 1.0_dt, 1.0_dt, 0.0_dt, 1.0_dt, 1.0_dt, 0.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 0.0_dt, 1.0_dt, 1.0_dt, 0.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 0.0_dt,
                                   0.0_dt, 0.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 1.0_dt,
                                   0.0_dt, 0.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 0.0_dt, 1.0_dt, 0.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 0.0_dt,
                                   1.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 0.0_dt, 1.0_dt, 1.0_dt, 0.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 0.0_dt, 1.0_dt, 1.0_dt, 0.0_dt, 1.0_dt },
                                 { 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt,
                                   1.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 1.0_dt, 0.0_dt, 1.0_dt,
                                   0.0_dt, 1.0_dt, 1.0_dt, 0.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 0.0_dt,
                                   0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 1.0_dt,
                                   0.0_dt, 0.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 0.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 0.0_dt,
                                   1.0_dt, 1.0_dt, 0.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 0.0_dt, 0.0_dt },
                                 { 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 0.0_dt,
                                   0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 0.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 1.0_dt, 0.0_dt, 1.0_dt,
                                   0.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 0.0_dt,
                                   0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 1.0_dt,
                                   0.0_dt, 0.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 1.0_dt,
                                   0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 0.0_dt, 0.0_dt },
                                 { 1.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt,
                                   1.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 0.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 0.0_dt, 0.0_dt,
                                   0.0_dt, 0.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt,
                                   0.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 1.0_dt,
                                   0.0_dt, 0.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 0.0_dt,
                                   0.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt } };

    for (size_t iter = 0; iter < std::size(dimensions); ++iter)
    {
        MANAGERS_DEFINE
        NETWORK_PARAMS_DEFINE(networkParameters);

        work.add<raul::DataLayer>("data", raul::DataParams{ { "x" }, depth, height, width });
        work.add<raul::ReduceMaxLayer>("rmax", raul::BasicParamsWithDim{ { "x" }, { "out" }, dimensions[iter] });
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
            EXPECT_EQ(x_tensor_grad[i], realGrads[iter][i]);
        }
    }
}

TEST(TestLayerReduceMax, MaxRepeatsBackwardUnit)
{
    PROFILE_TEST
    // Test parameters
    const auto batch = 1;
    const auto depth = 1;
    const auto height = 3;
    const auto width = 7;
    const auto eps = TODTYPE(1e-6);

    // See reduce_max.py
    const raul::Tensor x{ 1.0_dt, 1.0_dt, 2.0_dt, 3.0_dt, 3.0_dt, 3.0_dt, 2.0_dt, 1.0_dt, 1.0_dt, 2.0_dt, 3.0_dt, 3.0_dt, 3.0_dt, 2.0_dt, 1.0_dt, 1.0_dt, 2.0_dt, 3.0_dt, 3.0_dt, 3.0_dt, 2.0_dt };

    std::string dimensions[] = { "default", "batch", "depth", "height", "width" };

    raul::Tensor realGrads[] = {
        { 0.0_dt,        0.0_dt,        0.0_dt, 0.11111111_dt, 0.11111111_dt, 0.11111111_dt, 0.0_dt,        0.0_dt,        0.0_dt,        0.0_dt, 0.11111111_dt,
          0.11111111_dt, 0.11111111_dt, 0.0_dt, 0.0_dt,        0.0_dt,        0.0_dt,        0.11111111_dt, 0.11111111_dt, 0.11111111_dt, 0.0_dt },
        { 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt },
        { 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt },
        { 0.33333334_dt, 0.33333334_dt, 0.33333334_dt, 0.33333334_dt, 0.33333334_dt, 0.33333334_dt, 0.33333334_dt, 0.33333334_dt, 0.33333334_dt, 0.33333334_dt, 0.33333334_dt,
          0.33333334_dt, 0.33333334_dt, 0.33333334_dt, 0.33333334_dt, 0.33333334_dt, 0.33333334_dt, 0.33333334_dt, 0.33333334_dt, 0.33333334_dt, 0.33333334_dt },
        { 0.0_dt,        0.0_dt,        0.0_dt, 0.33333334_dt, 0.33333334_dt, 0.33333334_dt, 0.0_dt,        0.0_dt,        0.0_dt,        0.0_dt, 0.33333334_dt,
          0.33333334_dt, 0.33333334_dt, 0.0_dt, 0.0_dt,        0.0_dt,        0.0_dt,        0.33333334_dt, 0.33333334_dt, 0.33333334_dt, 0.0_dt }
    };

    for (size_t iter = 0; iter < std::size(dimensions); ++iter)
    {
        MANAGERS_DEFINE
        NETWORK_PARAMS_DEFINE(networkParameters);

        work.add<raul::DataLayer>("data", raul::DataParams{ { "x" }, depth, height, width });
        work.add<raul::ReduceMaxLayer>("rmax", raul::BasicParamsWithDim{ { "x" }, { "out" }, dimensions[iter] });
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
            ASSERT_TRUE(tools::expect_near_relative(x_tensor_grad[i], realGrads[iter][i], eps));
        }
    }
}

}