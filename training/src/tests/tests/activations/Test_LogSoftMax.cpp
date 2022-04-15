// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <chrono>
#include <cstdio>
#include <tests/tools/TestTools.h>

#include <tests/tools/TestTools.h>

#include <training/base/common/Common.h>
#include <training/base/common/MemoryManager.h>
#include <training/base/layers/activations/LogSoftMaxActivation.h>
#include <training/base/layers/activations/SoftMaxActivation.h>
#include <training/base/layers/basic/DataLayer.h>
#include <training/base/loss/NegativeLogLikelihoodLoss.h>
#include <training/compiler/Workflow.h>

namespace UT
{

TEST(TestLogSoftMax, LogSoftMaxWithDimUnit)
{
    PROFILE_TEST
    using namespace raul;

    constexpr auto eps = 1e-4_dt;

    size_t BATCH_SIZE = 2;
    size_t WIDTH = 2;
    size_t HEIGHT = 2;
    size_t DEPTH = 3;

    Dimension dims[] = { Dimension::Depth, Dimension::Height, Dimension::Width };
    std::string names[] = { "depth", "height", "width" };

    Tensor realOut[] = { { 1.0986_dt,  -1.0986_dt, -1.0986_dt, -1.0986_dt, -1.0986_dt, -1.0986_dt, -1.0986_dt, -1.0986_dt, -1.0986_dt, -1.0986_dt, -1.0986_dt, -1.0986_dt,
                           -3.1698_dt, -3.0949_dt, -1.0986_dt, -4.0360_dt, -0.1698_dt, -0.0949_dt, -1.0986_dt, -4.0360_dt, -2.1698_dt, -3.0949_dt, -1.0986_dt, -0.0360_dt },
                         { -1.3133e+00_dt, -1.3133e+00_dt, -3.1326e-01_dt, -3.1326e-01_dt, -1.3133e+00_dt, -1.3133e+00_dt, -3.1326e-01_dt, -3.1326e-01_dt,
                           -1.3133e+00_dt, -1.3133e+00_dt, -3.1326e-01_dt, -3.1326e-01_dt, -2.1269e+00_dt, -2.1269e+00_dt, -1.2693e-01_dt, -1.2693e-01_dt,
                           -3.1326e-01_dt, -3.1326e-01_dt, -1.3133e+00_dt, -1.3133e+00_dt, -1.3133e+00_dt, -6.0025e+00_dt, -3.1326e-01_dt, -2.4757e-03_dt },

                         { -0.6931_dt, -0.6931_dt, -0.6931_dt, -0.6931_dt, -0.6931_dt, -0.6931_dt, -0.6931_dt, -0.6931_dt, -0.6931_dt, -0.6931_dt, -0.6931_dt, -0.6931_dt,
                           -0.6931_dt, -0.6931_dt, -0.6931_dt, -0.6931_dt, -0.6931_dt, -0.6931_dt, -0.6931_dt, -0.6931_dt, -0.3133_dt, -1.3133_dt, -4.0181_dt, -0.0181_dt } };

    Tensor realGrad[] = { { 5.9605e-08_dt,  5.9605e-08_dt,  1.1921e-07_dt, 1.1921e-07_dt, 5.9605e-08_dt, 5.9605e-08_dt, 1.1921e-07_dt, 1.1921e-07_dt,
                            5.9605e-08_dt,  5.9605e-08_dt,  1.1921e-07_dt, 1.1921e-07_dt, 7.0593e-01_dt, 7.2833e-01_dt, 1.7881e-07_dt, 2.7703e+00_dt,
                            -1.9066e+00_dt, -1.4567e+00_dt, 1.7881e-07_dt, 2.7703e+00_dt, 1.2006e+00_dt, 7.2833e-01_dt, 1.7881e-07_dt, -5.5406e+00_dt },
                          { 0.1932_dt, 0.1932_dt, -0.1932_dt, -0.1932_dt, 0.1932_dt,  0.1932_dt,  -0.1932_dt, -0.1932_dt, 0.1932_dt, 0.1932_dt, -0.1932_dt, -0.1932_dt,
                            0.5232_dt, 0.5232_dt, -0.5232_dt, -0.5232_dt, -1.1174_dt, -1.1174_dt, 1.1174_dt,  1.1174_dt,  0.6553_dt, 0.9802_dt, -0.6553_dt, -0.9802_dt },
                          { 0.0000_dt, 0.0000_dt, 0.0000_dt, 0.0000_dt, 0.0000_dt, 0.0000_dt, 0.0000_dt, 0.0000_dt, 0.0000_dt,  0.0000_dt, 0.0000_dt, 0.0000_dt,
                            0.0000_dt, 0.0000_dt, 0.0000_dt, 0.0000_dt, 0.0000_dt, 0.0000_dt, 0.0000_dt, 0.0000_dt, -0.1932_dt, 0.1932_dt, 2.8201_dt, -2.8201_dt } };

    Tensor raw = { 1._dt, 1._dt, 2._dt, 2._dt, 1._dt, 1._dt, 2._dt, 2._dt, 1._dt, 1._dt, 2._dt, 2._dt, 1._dt, 1._dt, 3._dt, 3._dt, 4._dt, 4._dt, 3._dt, 3._dt, 2._dt, 1._dt, 3._dt, 7._dt };

    for (size_t k = 0; k < std::size(dims); ++k)
    {
        MANAGERS_DEFINE
        NETWORK_PARAMS_DEFINE(networkParameters);

        work.add<raul::DataLayer>("data", raul::DataParams{ { "in" }, DEPTH, HEIGHT, WIDTH });

        if (k < 2)
        {
            EXPECT_THROW(work.add<LogSoftMaxActivation>("sm", BasicParamsWithDim{ { "in" }, { "out" }, dims[k] }), raul::Exception);
            continue;
        }
        else
        {
            work.add<LogSoftMaxActivation>("sm", BasicParamsWithDim{ { "in" }, { "out" }, dims[k] });
        }
        TENSORS_CREATE(BATCH_SIZE)
        memory_manager["in"] = TORANGE(raw);
        memory_manager[raul::Name("out").grad()] = TORANGE(raw);

        work.forwardPassTraining();
        const raul::Tensor& out = memory_manager["out"];

        EXPECT_EQ(out.size(), realOut[k].size());
        for (size_t i = 0; i < out.size(); ++i)
        {
            EXPECT_NEAR(out[i], realOut[k][i], eps);
        }

        std::cout << " - LogSoftMax [" << names[k] << "] forward is Ok.\n";

        work.backwardPassTraining();

        const raul::Tensor& inGrad = memory_manager[raul::Name("in").grad()];

        EXPECT_EQ(inGrad.size(), realGrad[k].size());
        for (size_t i = 0; i < inGrad.size(); ++i)
        {
            EXPECT_NEAR(inGrad[i], realGrad[k][i], eps);
        }

        std::cout << " - LogSoftMax [" << names[k] << "] backward is Ok.\n";
    }
}

TEST(TestLogSoftMax, LogSoftMaxWidthUnit)
{
    PROFILE_TEST
    using namespace raul;

    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    constexpr auto eps = 1e-4_dt;

    size_t BATCH_SIZE = 32;
    size_t WIDTH = 2;
    size_t HEIGHT = 1;
    size_t DEPTH = 1;

    const Tensor realOut(BATCH_SIZE,
                         DEPTH,
                         HEIGHT,
                         WIDTH,
                         {
                             -1.3756_dt, -0.2913_dt, -2.2145_dt, -0.1157_dt, -0.9564_dt, -0.4850_dt, -0.9344_dt, -0.4989_dt, -1.3858_dt, -0.2879_dt, -1.9440_dt, -0.1545_dt, -1.4673_dt,
                             -0.2621_dt, -1.8594_dt, -0.1693_dt, -1.9028_dt, -0.1615_dt, -1.3114_dt, -0.3140_dt, -1.7309_dt, -0.1950_dt, -1.3068_dt, -0.3157_dt, -1.3508_dt, -0.2998_dt,
                             -1.4258_dt, -0.2748_dt, -1.5950_dt, -0.2268_dt, -1.1323_dt, -0.3890_dt, -1.2286_dt, -0.3463_dt, -1.4807_dt, -0.2581_dt, -1.4699_dt, -0.2613_dt, -1.3373_dt,
                             -0.3046_dt, -1.0710_dt, -0.4195_dt, -1.8663_dt, -0.1681_dt, -1.4564_dt, -0.2653_dt, -1.9998_dt, -0.1454_dt, -1.8108_dt, -0.1786_dt, -2.0779_dt, -0.1338_dt,
                             -1.3858_dt, -0.2879_dt, -1.0112_dt, -0.4522_dt, -0.9970_dt, -0.4604_dt, -1.7337_dt, -0.1944_dt, -1.2894_dt, -0.3222_dt, -1.2220_dt, -0.3491_dt,
                         });

    const Tensor realGrad(BATCH_SIZE,
                          DEPTH,
                          HEIGHT,
                          WIDTH,
                          {
                              -0.0233_dt, 0.0233_dt, -0.0278_dt, 0.0278_dt,  0.0120_dt,  -0.0120_dt, -0.0189_dt, 0.0189_dt,  -0.0234_dt, 0.0234_dt,  -0.0267_dt, 0.0267_dt,  0.0072_dt,
                              -0.0072_dt, 0.0049_dt, -0.0049_dt, -0.0265_dt, 0.0265_dt,  0.0084_dt,  -0.0084_dt, -0.0257_dt, 0.0257_dt,  -0.0228_dt, 0.0228_dt,  -0.0231_dt, 0.0231_dt,
                              -0.0237_dt, 0.0237_dt, -0.0249_dt, 0.0249_dt,  0.0101_dt,  -0.0101_dt, -0.0221_dt, 0.0221_dt,  0.0071_dt,  -0.0071_dt, -0.0240_dt, 0.0240_dt,  0.0082_dt,
                              -0.0082_dt, 0.0107_dt, -0.0107_dt, 0.0048_dt,  -0.0048_dt, 0.0073_dt,  -0.0073_dt, 0.0042_dt,  -0.0042_dt, -0.0261_dt, 0.0261_dt,  0.0039_dt,  -0.0039_dt,
                              -0.0234_dt, 0.0234_dt, 0.0113_dt,  -0.0113_dt, 0.0115_dt,  -0.0115_dt, -0.0257_dt, 0.0257_dt,  -0.0226_dt, 0.0226_dt,  0.0092_dt,  -0.0092_dt,
                          });

    work.add<DataLayer>("data", DataParams{ { "in" }, DEPTH, HEIGHT, WIDTH });
    work.add<LogSoftMaxActivation>("sm", BasicParamsWithDim{ { "in" }, { "out" }, "width" });
    TENSORS_CREATE(BATCH_SIZE)

    memory_manager["in"] = TORANGE(Tensor({ -0.6226_dt, 0.4617_dt,  -1.4773_dt, 0.6215_dt,  -0.1313_dt, 0.3401_dt,  -0.1132_dt, 0.3223_dt,  -0.6378_dt, 0.4601_dt,  -0.9472_dt, 0.8423_dt,  -0.4721_dt,
                                            0.7331_dt,  -0.9548_dt, 0.7353_dt,  -0.7862_dt, 0.9551_dt,  -0.2192_dt, 0.7782_dt,  -0.6109_dt, 0.9250_dt,  -0.4453_dt, 0.5458_dt,  -0.7532_dt, 0.2978_dt,
                                            -0.5299_dt, 0.6211_dt,  -0.5609_dt, 0.8073_dt,  0.1008_dt,  0.8441_dt,  -0.6271_dt, 0.2552_dt,  -0.3445_dt, 0.8781_dt,  -0.5200_dt, 0.6886_dt,  -0.5070_dt,
                                            0.5257_dt,  -0.2960_dt, 0.3555_dt,  -1.1601_dt, 0.5381_dt,  -0.5758_dt, 0.6153_dt,  -1.0586_dt, 0.7958_dt,  -1.0672_dt, 0.5650_dt,  -0.7843_dt, 1.1598_dt,
                                            -0.7141_dt, 0.3838_dt,  -0.0126_dt, 0.5464_dt,  0.0409_dt,  0.5775_dt,  -0.8350_dt, 0.7043_dt,  -0.7880_dt, 0.1792_dt,  -0.1886_dt, 0.6843_dt }));

    work.forwardPassTraining();
    const raul::Tensor& out = memory_manager["out"];

    EXPECT_EQ(out.getShape(), realOut.getShape());
    for (size_t i = 0; i < out.size(); ++i)
    {
        EXPECT_NEAR(out[i], realOut[i], eps);
    }

    std::cout << " - LogSoftMax [width] forward is Ok.\n";

    memory_manager[raul::Name("out").grad()] = TORANGE(Tensor({
        -0.0312_dt, 0.0000_dt, -0.0312_dt, 0.0000_dt,  0.0000_dt,  -0.0312_dt, -0.0312_dt, 0.0000_dt,  -0.0312_dt, 0.0000_dt,  -0.0312_dt, 0.0000_dt,  0.0000_dt,  -0.0312_dt, 0.0000_dt, -0.0312_dt,
        -0.0312_dt, 0.0000_dt, 0.0000_dt,  -0.0312_dt, -0.0312_dt, 0.0000_dt,  -0.0312_dt, 0.0000_dt,  -0.0312_dt, 0.0000_dt,  -0.0312_dt, 0.0000_dt,  -0.0312_dt, 0.0000_dt,  0.0000_dt, -0.0312_dt,
        -0.0312_dt, 0.0000_dt, 0.0000_dt,  -0.0312_dt, -0.0312_dt, 0.0000_dt,  0.0000_dt,  -0.0312_dt, 0.0000_dt,  -0.0312_dt, 0.0000_dt,  -0.0312_dt, 0.0000_dt,  -0.0312_dt, 0.0000_dt, -0.0312_dt,
        -0.0312_dt, 0.0000_dt, 0.0000_dt,  -0.0312_dt, -0.0312_dt, 0.0000_dt,  0.0000_dt,  -0.0312_dt, 0.0000_dt,  -0.0312_dt, -0.0312_dt, 0.0000_dt,  -0.0312_dt, 0.0000_dt,  0.0000_dt, -0.0312_dt,
    }));

    work.backwardPassTraining();

    const raul::Tensor& inGrad = memory_manager[raul::Name("in").grad()];

    EXPECT_EQ(inGrad.getShape(), realGrad.getShape());
    for (size_t i = 0; i < inGrad.size(); ++i)
    {
        EXPECT_NEAR(inGrad[i], realGrad[i], eps);
    }

    std::cout << " - LogSoftMax [width] backward is Ok.\n";
}

}