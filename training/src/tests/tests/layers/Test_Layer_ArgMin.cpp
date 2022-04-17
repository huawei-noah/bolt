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

#include <training/base/layers/basic/ArgMinLayer.h>
#include <training/compiler/Workflow.h>

namespace UT
{

TEST(TestLayerArgMin, InputNumExceedsUnit)
{
    PROFILE_TEST

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    // Apply function
    ASSERT_THROW(raul::ArgMinLayer("argmin", raul::BasicParamsWithDim{ { "x", "y" }, { "x_out", "y_out" } }, networkParameters), raul::Exception);
}

TEST(TestLayerArgMin, Unit)
{
    PROFILE_TEST
    // Test parameters
    const auto batch = 2;
    const auto depth = 3;
    const auto height = 4;
    const auto width = 5;
    const raul::Tensor x{ 0.4963_dt, 0.7682_dt, 0.0885_dt, 0.1320_dt, 0.3074_dt, 0.6341_dt, 0.4901_dt, 0.8964_dt, 0.4556_dt, 0.6323_dt, 0.3489_dt, 0.4017_dt, 0.0223_dt, 0.1689_dt, 0.2939_dt,
                          0.5185_dt, 0.6977_dt, 0.8000_dt, 0.1610_dt, 0.2823_dt, 0.6816_dt, 0.9152_dt, 0.3971_dt, 0.8742_dt, 0.4194_dt, 0.5529_dt, 0.9527_dt, 0.0362_dt, 0.1852_dt, 0.3734_dt,
                          0.3051_dt, 0.9320_dt, 0.1759_dt, 0.2698_dt, 0.1507_dt, 0.0317_dt, 0.2081_dt, 0.9298_dt, 0.7231_dt, 0.7423_dt, 0.5263_dt, 0.2437_dt, 0.5846_dt, 0.0332_dt, 0.1387_dt,
                          0.2422_dt, 0.8155_dt, 0.7932_dt, 0.2783_dt, 0.4820_dt, 0.8198_dt, 0.9971_dt, 0.6984_dt, 0.5675_dt, 0.8352_dt, 0.2056_dt, 0.5932_dt, 0.1123_dt, 0.1535_dt, 0.2417_dt,
                          0.7262_dt, 0.7011_dt, 0.2038_dt, 0.6511_dt, 0.7745_dt, 0.4369_dt, 0.5191_dt, 0.6159_dt, 0.8102_dt, 0.9801_dt, 0.1147_dt, 0.3168_dt, 0.6965_dt, 0.9143_dt, 0.9351_dt,
                          0.9412_dt, 0.5995_dt, 0.0652_dt, 0.5460_dt, 0.1872_dt, 0.0340_dt, 0.9442_dt, 0.8802_dt, 0.0012_dt, 0.5936_dt, 0.4158_dt, 0.4177_dt, 0.2711_dt, 0.6923_dt, 0.2038_dt,
                          0.6833_dt, 0.7529_dt, 0.8579_dt, 0.6870_dt, 0.0051_dt, 0.1757_dt, 0.7497_dt, 0.6047_dt, 0.1100_dt, 0.2121_dt, 0.9704_dt, 0.8369_dt, 0.2820_dt, 0.3742_dt, 0.0237_dt,
                          0.4910_dt, 0.1235_dt, 0.1143_dt, 0.4725_dt, 0.5751_dt, 0.2952_dt, 0.7967_dt, 0.1957_dt, 0.9537_dt, 0.8426_dt, 0.0784_dt, 0.3756_dt, 0.5226_dt, 0.5730_dt, 0.6186_dt };

    std::string dimensions[] = { "batch", "depth", "height", "width" };

    raul::Tensor realOutputs[] = {
        { 0.4963_dt, 0.7011_dt, 0.0885_dt, 0.1320_dt, 0.3074_dt, 0.4369_dt, 0.4901_dt, 0.6159_dt, 0.4556_dt, 0.6323_dt, 0.1147_dt, 0.3168_dt, 0.0223_dt, 0.1689_dt, 0.2939_dt,
          0.5185_dt, 0.5995_dt, 0.0652_dt, 0.1610_dt, 0.1872_dt, 0.0340_dt, 0.9152_dt, 0.3971_dt, 0.0012_dt, 0.4194_dt, 0.4158_dt, 0.4177_dt, 0.0362_dt, 0.1852_dt, 0.2038_dt,
          0.3051_dt, 0.7529_dt, 0.1759_dt, 0.2698_dt, 0.0051_dt, 0.0317_dt, 0.2081_dt, 0.6047_dt, 0.1100_dt, 0.2121_dt, 0.5263_dt, 0.2437_dt, 0.2820_dt, 0.0332_dt, 0.0237_dt,
          0.2422_dt, 0.1235_dt, 0.1143_dt, 0.2783_dt, 0.4820_dt, 0.2952_dt, 0.7967_dt, 0.1957_dt, 0.5675_dt, 0.8352_dt, 0.0784_dt, 0.3756_dt, 0.1123_dt, 0.1535_dt, 0.2417_dt },
        { 0.4963_dt, 0.2437_dt, 0.0885_dt, 0.0332_dt, 0.1387_dt, 0.2422_dt, 0.4901_dt, 0.0362_dt, 0.1852_dt, 0.3734_dt, 0.3051_dt, 0.4017_dt, 0.0223_dt, 0.1689_dt,
          0.1507_dt, 0.0317_dt, 0.2081_dt, 0.1123_dt, 0.1535_dt, 0.2417_dt, 0.0340_dt, 0.7011_dt, 0.2038_dt, 0.0012_dt, 0.0237_dt, 0.4158_dt, 0.1235_dt, 0.1143_dt,
          0.4725_dt, 0.2038_dt, 0.1147_dt, 0.3168_dt, 0.1957_dt, 0.6870_dt, 0.0051_dt, 0.0784_dt, 0.3756_dt, 0.0652_dt, 0.1100_dt, 0.1872_dt },
        { 0.3489_dt, 0.4017_dt, 0.0223_dt, 0.1320_dt, 0.2823_dt, 0.0317_dt, 0.2081_dt, 0.0362_dt, 0.1852_dt, 0.1507_dt, 0.2056_dt, 0.2437_dt, 0.1123_dt, 0.0332_dt, 0.1387_dt,
          0.1147_dt, 0.3168_dt, 0.0652_dt, 0.5460_dt, 0.1872_dt, 0.0340_dt, 0.4177_dt, 0.2711_dt, 0.0012_dt, 0.0051_dt, 0.0784_dt, 0.1235_dt, 0.1143_dt, 0.3742_dt, 0.0237_dt },
        { 0.0885_dt, 0.4556_dt, 0.0223_dt, 0.1610_dt, 0.3971_dt, 0.0362_dt, 0.1507_dt, 0.0317_dt, 0.0332_dt, 0.2422_dt, 0.5675_dt, 0.1123_dt,
          0.2038_dt, 0.4369_dt, 0.1147_dt, 0.0652_dt, 0.0012_dt, 0.2038_dt, 0.0051_dt, 0.1100_dt, 0.0237_dt, 0.1143_dt, 0.1957_dt, 0.0784_dt }
    };

    raul::Tensor realIndices[] = { { 0.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 0.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 1.0_dt, 0.0_dt, 1.0_dt,
                                     1.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 0.0_dt, 1.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 0.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 1.0_dt, 1.0_dt,
                                     0.0_dt, 0.0_dt, 1.0_dt, 0.0_dt, 1.0_dt, 0.0_dt, 1.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 0.0_dt },
                                   { 0.0_dt, 2.0_dt, 0.0_dt, 2.0_dt, 2.0_dt, 2.0_dt, 0.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 2.0_dt, 2.0_dt, 2.0_dt,
                                     1.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 2.0_dt, 1.0_dt, 2.0_dt, 2.0_dt, 2.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 2.0_dt, 1.0_dt, 1.0_dt, 2.0_dt, 2.0_dt, 0.0_dt, 1.0_dt, 0.0_dt },
                                   { 2.0_dt, 2.0_dt, 2.0_dt, 0.0_dt, 3.0_dt, 3.0_dt, 3.0_dt, 1.0_dt, 1.0_dt, 2.0_dt, 3.0_dt, 0.0_dt, 3.0_dt, 0.0_dt, 0.0_dt,
                                     2.0_dt, 2.0_dt, 3.0_dt, 3.0_dt, 3.0_dt, 0.0_dt, 1.0_dt, 1.0_dt, 0.0_dt, 2.0_dt, 3.0_dt, 1.0_dt, 1.0_dt, 0.0_dt, 0.0_dt },
                                   { 2.0_dt, 3.0_dt, 2.0_dt, 3.0_dt, 2.0_dt, 2.0_dt, 4.0_dt, 0.0_dt, 3.0_dt, 0.0_dt, 3.0_dt, 2.0_dt,
                                     2.0_dt, 0.0_dt, 0.0_dt, 2.0_dt, 3.0_dt, 4.0_dt, 4.0_dt, 3.0_dt, 4.0_dt, 2.0_dt, 2.0_dt, 0.0_dt } };

    raul::Tensor deltas[] = { { 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt,
                                1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt,
                                1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt },
                              { 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt,
                                1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt },
                              { 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt,
                                1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt },
                              { 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt,
                                1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt } };

    raul::Tensor realGrads[] = { { 1.0_dt, 0.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 0.0_dt, 1.0_dt, 0.0_dt, 1.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 0.0_dt,
                                   0.0_dt, 1.0_dt, 1.0_dt, 0.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 1.0_dt, 0.0_dt, 1.0_dt, 0.0_dt, 1.0_dt, 1.0_dt, 0.0_dt, 1.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 0.0_dt,
                                   1.0_dt, 1.0_dt, 0.0_dt, 1.0_dt, 0.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 1.0_dt, 1.0_dt,
                                   0.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 0.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 1.0_dt, 0.0_dt, 1.0_dt,
                                   1.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 0.0_dt, 1.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 0.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 1.0_dt, 1.0_dt,
                                   0.0_dt, 0.0_dt, 1.0_dt, 0.0_dt, 1.0_dt, 0.0_dt, 1.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 0.0_dt },
                                 { 1.0_dt, 0.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt,
                                   0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 0.0_dt,
                                   0.0_dt, 1.0_dt, 0.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 1.0_dt, 1.0_dt,
                                   0.0_dt, 1.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 0.0_dt, 1.0_dt,
                                   1.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 0.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 0.0_dt,
                                   0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 0.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 0.0_dt },
                                 { 0.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 1.0_dt,
                                   0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 0.0_dt,
                                   0.0_dt, 1.0_dt, 0.0_dt, 1.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 0.0_dt, 1.0_dt, 0.0_dt, 0.0_dt,
                                   0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 1.0_dt, 1.0_dt,
                                   1.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt,
                                   0.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 1.0_dt, 0.0_dt, 1.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt },
                                 { 0.0_dt, 0.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 0.0_dt,
                                   0.0_dt, 0.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt,
                                   0.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 0.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 0.0_dt, 0.0_dt,
                                   0.0_dt, 0.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 0.0_dt, 0.0_dt,
                                   0.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 0.0_dt,
                                   0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt } };

    // See argmin.py
    for (size_t iter = 0; iter < std::size(dimensions); ++iter)
    {
        MANAGERS_DEFINE
        NETWORK_PARAMS_DEFINE(networkParameters);

        work.tensorNeeded("x", "x", raul::WShape{ raul::BS(), depth, height, width }, DEC_FORW_READ_NOMEMOPT);

        // Apply function
        raul::ArgMinLayer argmin("argmin", raul::BasicParamsWithDim{ { "x" }, { "ind", "val" }, dimensions[iter] }, networkParameters);
        TENSORS_CREATE(batch);
        memory_manager["x"] = TORANGE(x);
        memory_manager[raul::Name("val").grad()] = TORANGE(deltas[iter]);

        argmin.forwardCompute(raul::NetworkMode::Test);
        argmin.backwardCompute();

        // Forward checks
        const auto& out_indices = memory_manager["ind"];
        const auto& out_values = memory_manager["val"];
        EXPECT_EQ(out_values.size(), realOutputs[iter].size());
        EXPECT_EQ(out_indices.size(), realIndices[iter].size());
        for (size_t i = 0; i < out_values.size(); ++i)
        {
            EXPECT_EQ(out_values[i], realOutputs[iter][i]);
            EXPECT_EQ(out_indices[i], realIndices[iter][i]);
        }

        // Backward checks
        const auto& x_tensor_grad = memory_manager[raul::Name("x").grad()];
        EXPECT_EQ(x_tensor_grad.getShape(), memory_manager["x"].getShape());
        for (size_t i = 0; i < x_tensor_grad.size(); ++i)
        {
            EXPECT_EQ(x_tensor_grad[i], realGrads[iter][i]);
        }
        memory_manager.clear();
    }
}

}