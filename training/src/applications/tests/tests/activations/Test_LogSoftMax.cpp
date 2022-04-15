// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

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

#include <training/api/API.h>
#include <training/common/Common.h>
#include <training/common/DataLoader.h>
#include <training/common/MemoryManager.h>
#include <training/layers/activations/LogSoftMaxActivation.h>
#include <training/layers/activations/SoftMaxActivation.h>
#include <training/layers/basic/DataLayer.h>
#include <training/loss/NegativeLogLikelihoodLoss.h>
#include <training/network/Workflow.h>

namespace UT
{
TEST(TestLogSoftMax, Unit)
{
    PROFILE_TEST

    raul::DataLoader dataLoader;

    raul::dtype eps = TODTYPE(1e-4);

    raul::Tensor raw = { 1, 2, 5, -1, 3, -2, 5, -1, 3, 2, 2, -4, -1, 1, 1 };

    raul::TensorU8 classes = { 1, 0, 4 };
    size_t batch = classes.size();

    std::string reduction[] = { "none", "sum", "batch_mean" };
    raul::Tensor realOut = { -4.1872f, -3.1872f, -0.1872f, -6.1872f, -2.1872f, -7.1727f, -0.1727f, -6.1727f, -2.1727f, -3.1727f, -0.5811f, -6.5811f, -3.5811f, -1.5811f, -1.5811f };
    raul::Tensor realLoss[] = { { 0.0f, 3.1872f, 0.0f, 0.0f, 0.0f, 7.1727f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.5811f }, { 11.9411f }, { 3.9804f } };

    raul::Tensor realOutGrad[] = { { 0.0_dt },
                                   { 0.f, -1.f, 0.f, 0.f, 0.f, -1.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, -1.f },
                                   { 0.f, -0.3333f, 0.f, 0.f, 0.f, -0.3333f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, -0.3333f } };

    raul::Tensor realInGrad[] = { { 0.0_dt },
                                  { 0.0152f, -0.9587f, 0.8292f, 0.0021f, 0.1122f, -0.9992f, 0.8414f, 0.0021f, 0.1139f, 0.0419f, 0.5593f, 0.0014f, 0.0278f, 0.2057f, -0.7943f },
                                  { 0.0051f, -0.3196f, 0.2764f, 0.0007f, 0.0374f, -0.3331f, 0.2805f, 0.0007f, 0.0380f, 0.0140f, 0.1864f, 0.0005f, 0.0093f, 0.0686f, -0.2648f } };

    for (size_t iter = 0; iter < std::size(reduction); ++iter)
    {
        MANAGERS_DEFINE
        NETWORK_PARAMS_DEFINE(networkParameters);
        networkParameters.mLossReductionCoefficient = batch;

        auto& encodedClasses = dataLoader.buildOneHotVector(classes, 5);

        work.add<raul::DataLayer>("data", raul::DataParams{ { "in", "labels" }, 1u, 1u, raw.size() / batch });
        work.add<raul::LogSoftMaxActivation>("logsoftmax", raul::BasicParamsWithDim{ { "in" }, { "out" } });
        work.add<raul::NLLLoss>("loss", raul::LossParams{ { "out", "labels" }, { "loss" }, reduction[iter].c_str() });
        TENSORS_CREATE(batch);
        memory_manager["in"] = TORANGE(raw);
        memory_manager["labels"] = TORANGE(encodedClasses);

        work.forwardPassTraining();

        const raul::Tensor& out = memory_manager["out"];
        const raul::Tensor& loss = memory_manager["loss"];

        if (iter == 0)
        {
            for (size_t i = 0; i < out.size(); ++i)
            {
                EXPECT_NEAR(out[i], realOut[i], eps);
            }
            printf(" - LogSoftMax forward 5*3 is Ok.\n");
        }

        for (size_t i = 0; i < loss.size(); ++i)
        {
            EXPECT_NEAR(loss[i], realLoss[iter][i], eps);
        }
        printf(" - NLLLoss[reduction=%s] forward 5*3 is Ok.\n", reduction[iter].c_str());

        if (iter > 0)
        {
            work.backwardPassTraining();

            const raul::Tensor& out_nabla = memory_manager[raul::Name("out").grad()];
            const raul::Tensor& in_nabla = memory_manager[raul::Name("in").grad()];

            EXPECT_EQ(out_nabla.size(), realOutGrad[iter].size());
            for (size_t i = 0; i < out_nabla.size(); ++i)
            {
                EXPECT_NEAR(out_nabla[i], realOutGrad[iter][i], eps);
            }
            printf(" - NLLLoss[reduction=%s] backward 5*3 is Ok.\n", reduction[iter].c_str());

            EXPECT_EQ(in_nabla.size(), realInGrad[iter].size());
            for (size_t i = 0; i < in_nabla.size(); ++i)
            {
                EXPECT_NEAR(in_nabla[i], realInGrad[iter][i], eps);
            }
            printf(" - LogSoftMax backward 5*3 is Ok.\n");
        }
    }
}

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

TEST(TestLogSoftMax, GpuDefaultUnit)
{
    PROFILE_TEST

    GPU_ONLY_TEST

    using namespace raul;

    constexpr size_t BATCH_SIZE = 2;
    constexpr size_t WIDTH = 7;
    constexpr size_t HEIGHT = 5;
    constexpr size_t DEPTH = 3;
    constexpr dtype eps = 1.0e-5_dt;

    const Tensor in{ 0.27825248_dt, 0.48195881_dt, 0.81978035_dt, 0.99706656_dt, 0.69844109_dt, 0.56754643_dt, 0.83524317_dt, 0.20559883_dt, 0.59317201_dt, 0.11234725_dt, 0.15345693_dt, 0.24170822_dt,
                     0.72623652_dt, 0.70108020_dt, 0.20382375_dt, 0.65105355_dt, 0.77448601_dt, 0.43689132_dt, 0.51909077_dt, 0.61585236_dt, 0.81018829_dt, 0.98009706_dt, 0.11468822_dt, 0.31676513_dt,
                     0.69650495_dt, 0.91427469_dt, 0.93510365_dt, 0.94117838_dt, 0.59950727_dt, 0.06520867_dt, 0.54599625_dt, 0.18719733_dt, 0.03402293_dt, 0.94424623_dt, 0.88017988_dt, 0.00123602_dt,
                     0.59358603_dt, 0.41576999_dt, 0.41771942_dt, 0.27112156_dt, 0.69227809_dt, 0.20384824_dt, 0.68329567_dt, 0.75285405_dt, 0.85793579_dt, 0.68695557_dt, 0.00513238_dt, 0.17565155_dt,
                     0.74965751_dt, 0.60465068_dt, 0.10995799_dt, 0.21209025_dt, 0.97037464_dt, 0.83690894_dt, 0.28198743_dt, 0.37415761_dt, 0.02370095_dt, 0.49101293_dt, 0.12347054_dt, 0.11432165_dt,
                     0.47245020_dt, 0.57507253_dt, 0.29523486_dt, 0.79668880_dt, 0.19573045_dt, 0.95368505_dt, 0.84264994_dt, 0.07835853_dt, 0.37555784_dt, 0.52256131_dt, 0.57295054_dt, 0.61858714_dt,
                     0.69621414_dt, 0.52995008_dt, 0.25603563_dt, 0.73659450_dt, 0.02037555_dt, 0.20364666_dt, 0.37483507_dt, 0.25644332_dt, 0.32508332_dt, 0.09018916_dt, 0.39364243_dt, 0.60687822_dt,
                     0.17426711_dt, 0.47434032_dt, 0.85792542_dt, 0.44859987_dt, 0.51389611_dt, 0.45686555_dt, 0.60119069_dt, 0.81791973_dt, 0.97362310_dt, 0.81752795_dt, 0.97470677_dt, 0.46383917_dt,
                     0.05083925_dt, 0.26296139_dt, 0.84045261_dt, 0.49675876_dt, 0.25147682_dt, 0.11684412_dt, 0.03207397_dt, 0.07799590_dt, 0.39858162_dt, 0.77420300_dt, 0.77032053_dt, 0.01778406_dt,
                     0.81189102_dt, 0.10874528_dt, 0.39429486_dt, 0.29726368_dt, 0.40369236_dt, 0.40182865_dt, 0.05132502_dt, 0.06828105_dt, 0.42176026_dt, 0.50646609_dt, 0.27286255_dt, 0.68834960_dt,
                     0.04997081_dt, 0.46625638_dt, 0.93970972_dt, 0.29605401_dt, 0.95150155_dt, 0.68107688_dt, 0.04876953_dt, 0.81634867_dt, 0.44230276_dt, 0.27679658_dt, 0.89982665_dt, 0.09595239_dt,
                     0.55365247_dt, 0.39531565_dt, 0.85705632_dt, 0.63957226_dt, 0.74025267_dt, 0.67657948_dt, 0.37976265_dt, 0.39484727_dt, 0.08795929_dt, 0.77092206_dt, 0.89698905_dt, 0.84211242_dt,
                     0.14731085_dt, 0.52229995_dt, 0.14753294_dt, 0.22475791_dt, 0.20864725_dt, 0.67087251_dt, 0.20204341_dt, 0.48909140_dt, 0.52103406_dt, 0.82231152_dt, 0.12203997_dt, 0.15674388_dt,
                     0.20966923_dt, 0.84996670_dt, 0.32026750_dt, 0.92174435_dt, 0.68080378_dt, 0.56331301_dt, 0.49627799_dt, 0.40115923_dt, 0.56273317_dt, 0.38582766_dt, 0.49648678_dt, 0.56379652_dt,
                     0.10889745_dt, 0.23793429_dt, 0.90374637_dt, 0.09422666_dt, 0.46409690_dt, 0.99461937_dt, 0.68061852_dt, 0.51415652_dt, 0.06669503_dt, 0.74768895_dt, 0.14385962_dt, 0.35806787_dt,
                     0.33224183_dt, 0.42595631_dt, 0.50546914_dt, 0.91240376_dt, 0.56241941_dt, 0.94784641_dt, 0.80585623_dt, 0.18389302_dt, 0.72425205_dt, 0.14655197_dt, 0.28808743_dt, 0.64706135_dt,
                     0.66509604_dt, 0.87511402_dt, 0.33904207_dt, 0.50080043_dt, 0.75741178_dt, 0.01645392_dt, 0.86149031_dt, 0.08653879_dt, 0.50689125_dt, 0.41499162_dt, 0.23666352_dt, 0.56608552_dt,
                     0.91345936_dt, 0.35384023_dt, 0.20315295_dt, 0.31508058_dt, 0.00442582_dt, 0.72569698_dt };

    const Tensor realOut{
        -4.90489_dt, -4.70119_dt, -4.36337_dt, -4.18608_dt, -4.48471_dt, -4.6156_dt,  -4.3479_dt,  -4.97755_dt, -4.58997_dt, -5.0708_dt,  -5.02969_dt, -4.94144_dt, -4.45691_dt, -4.48207_dt,
        -4.97932_dt, -4.53209_dt, -4.40866_dt, -4.74625_dt, -4.66406_dt, -4.56729_dt, -4.37296_dt, -4.20305_dt, -5.06846_dt, -4.86638_dt, -4.48664_dt, -4.26887_dt, -4.24804_dt, -4.24197_dt,
        -4.58364_dt, -5.11794_dt, -4.63715_dt, -4.99595_dt, -5.14912_dt, -4.2389_dt,  -4.30297_dt, -5.18191_dt, -4.58956_dt, -4.76738_dt, -4.76543_dt, -4.91202_dt, -4.49087_dt, -4.9793_dt,
        -4.49985_dt, -4.43029_dt, -4.32521_dt, -4.49619_dt, -5.17801_dt, -5.00749_dt, -4.43349_dt, -4.5785_dt,  -5.07319_dt, -4.97106_dt, -4.21277_dt, -4.34624_dt, -4.90116_dt, -4.80899_dt,
        -5.15945_dt, -4.69213_dt, -5.05968_dt, -5.06882_dt, -4.7107_dt,  -4.60807_dt, -4.88791_dt, -4.38646_dt, -4.98742_dt, -4.22946_dt, -4.3405_dt,  -5.10479_dt, -4.80759_dt, -4.66059_dt,
        -4.6102_dt,  -4.56456_dt, -4.48693_dt, -4.6532_dt,  -4.92711_dt, -4.44655_dt, -5.16277_dt, -4.9795_dt,  -4.80831_dt, -4.9267_dt,  -4.85806_dt, -5.09296_dt, -4.7895_dt,  -4.57627_dt,
        -5.00888_dt, -4.70881_dt, -4.32522_dt, -4.73455_dt, -4.66925_dt, -4.72628_dt, -4.58196_dt, -4.36523_dt, -4.20952_dt, -4.36562_dt, -4.20844_dt, -4.71931_dt, -5.13231_dt, -4.92019_dt,
        -4.34269_dt, -4.68639_dt, -4.93167_dt, -5.0663_dt,  -5.15107_dt, -5.10515_dt, -4.78456_dt, -4.39515_dt, -4.39904_dt, -5.15157_dt, -4.35747_dt, -5.06061_dt, -4.77506_dt, -4.87209_dt,
        -4.76566_dt, -4.76753_dt, -5.11803_dt, -5.10108_dt, -4.7476_dt,  -4.66289_dt, -4.89649_dt, -4.48101_dt, -5.11939_dt, -4.7031_dt,  -4.22965_dt, -4.8733_dt,  -4.21785_dt, -4.48828_dt,
        -5.12059_dt, -4.35301_dt, -4.72705_dt, -4.89256_dt, -4.26953_dt, -5.0734_dt,  -4.6157_dt,  -4.77404_dt, -4.3123_dt,  -4.52978_dt, -4.4291_dt,  -4.49278_dt, -4.78959_dt, -4.77451_dt,
        -5.0814_dt,  -4.39843_dt, -4.27237_dt, -4.32724_dt, -5.02205_dt, -4.64706_dt, -5.02182_dt, -4.9446_dt,  -4.96071_dt, -4.49848_dt, -4.96731_dt, -4.68026_dt, -4.64832_dt, -4.34704_dt,
        -5.04732_dt, -5.01261_dt, -4.95969_dt, -4.31939_dt, -4.84909_dt, -4.24761_dt, -4.48855_dt, -4.60604_dt, -4.67308_dt, -4.7682_dt,  -4.60662_dt, -4.78353_dt, -4.67287_dt, -4.60556_dt,
        -5.06046_dt, -4.93142_dt, -4.26561_dt, -5.07513_dt, -4.70526_dt, -4.17474_dt, -4.48874_dt, -4.6552_dt,  -5.10266_dt, -4.42167_dt, -5.0255_dt,  -4.81129_dt, -4.83711_dt, -4.7434_dt,
        -4.66389_dt, -4.25695_dt, -4.60694_dt, -4.22151_dt, -4.3635_dt,  -4.98546_dt, -4.4451_dt,  -5.0228_dt,  -4.88127_dt, -4.52229_dt, -4.50426_dt, -4.29424_dt, -4.83031_dt, -4.66856_dt,
        -4.41194_dt, -5.1529_dt,  -4.30787_dt, -5.08282_dt, -4.66247_dt, -4.75436_dt, -4.93269_dt, -4.60327_dt, -4.2559_dt,  -4.81552_dt, -4.9662_dt,  -4.85428_dt, -5.16493_dt, -4.44366_dt,
    };

    WorkflowEager work(raul::CompressionMode::NONE, raul::CalculationMode::DETERMINISTIC, raul::AllocationMode::STANDARD, raul::ExecutionTarget::GPU);
    work.getKernelManager().setExecutionPolicy(raul::KernelExecutionPolicy::DefaultParams);

    work.add<DataLayer>("data", DataParams{ { "in" }, DEPTH, HEIGHT, WIDTH });
    work.add<LogSoftMaxActivation>("softmax", BasicParamsWithDim{ { "in" }, { "out" } });

    TENSORS_CREATE(BATCH_SIZE)

    MemoryManagerGPU& memory_manager = work.getMemoryManager<MemoryManagerGPU>();
    memory_manager["in"] = TORANGE(in);

    ASSERT_NO_THROW(work.forwardPassTraining());
    const Tensor& out = memory_manager["out"];
    EXPECT_EQ(out.size(), realOut.size());
    for (size_t i = 0; i < out.size(); ++i)
    {
        EXPECT_NEAR(out[i], realOut[i], eps);
    }

    ASSERT_NO_THROW(work.backwardPassTraining());
}

TEST(TestLogSoftMax, GpuWidthUnit)
{
    PROFILE_TEST

    GPU_ONLY_TEST

    using namespace raul;

    constexpr size_t BATCH_SIZE = 2;
    constexpr size_t WIDTH = 7;
    constexpr size_t HEIGHT = 5;
    constexpr size_t DEPTH = 3;
    constexpr dtype eps = 1.0e-5_dt;

    const Tensor in{ 0.27825248_dt, 0.48195881_dt, 0.81978035_dt, 0.99706656_dt, 0.69844109_dt, 0.56754643_dt, 0.83524317_dt, 0.20559883_dt, 0.59317201_dt, 0.11234725_dt, 0.15345693_dt, 0.24170822_dt,
                     0.72623652_dt, 0.70108020_dt, 0.20382375_dt, 0.65105355_dt, 0.77448601_dt, 0.43689132_dt, 0.51909077_dt, 0.61585236_dt, 0.81018829_dt, 0.98009706_dt, 0.11468822_dt, 0.31676513_dt,
                     0.69650495_dt, 0.91427469_dt, 0.93510365_dt, 0.94117838_dt, 0.59950727_dt, 0.06520867_dt, 0.54599625_dt, 0.18719733_dt, 0.03402293_dt, 0.94424623_dt, 0.88017988_dt, 0.00123602_dt,
                     0.59358603_dt, 0.41576999_dt, 0.41771942_dt, 0.27112156_dt, 0.69227809_dt, 0.20384824_dt, 0.68329567_dt, 0.75285405_dt, 0.85793579_dt, 0.68695557_dt, 0.00513238_dt, 0.17565155_dt,
                     0.74965751_dt, 0.60465068_dt, 0.10995799_dt, 0.21209025_dt, 0.97037464_dt, 0.83690894_dt, 0.28198743_dt, 0.37415761_dt, 0.02370095_dt, 0.49101293_dt, 0.12347054_dt, 0.11432165_dt,
                     0.47245020_dt, 0.57507253_dt, 0.29523486_dt, 0.79668880_dt, 0.19573045_dt, 0.95368505_dt, 0.84264994_dt, 0.07835853_dt, 0.37555784_dt, 0.52256131_dt, 0.57295054_dt, 0.61858714_dt,
                     0.69621414_dt, 0.52995008_dt, 0.25603563_dt, 0.73659450_dt, 0.02037555_dt, 0.20364666_dt, 0.37483507_dt, 0.25644332_dt, 0.32508332_dt, 0.09018916_dt, 0.39364243_dt, 0.60687822_dt,
                     0.17426711_dt, 0.47434032_dt, 0.85792542_dt, 0.44859987_dt, 0.51389611_dt, 0.45686555_dt, 0.60119069_dt, 0.81791973_dt, 0.97362310_dt, 0.81752795_dt, 0.97470677_dt, 0.46383917_dt,
                     0.05083925_dt, 0.26296139_dt, 0.84045261_dt, 0.49675876_dt, 0.25147682_dt, 0.11684412_dt, 0.03207397_dt, 0.07799590_dt, 0.39858162_dt, 0.77420300_dt, 0.77032053_dt, 0.01778406_dt,
                     0.81189102_dt, 0.10874528_dt, 0.39429486_dt, 0.29726368_dt, 0.40369236_dt, 0.40182865_dt, 0.05132502_dt, 0.06828105_dt, 0.42176026_dt, 0.50646609_dt, 0.27286255_dt, 0.68834960_dt,
                     0.04997081_dt, 0.46625638_dt, 0.93970972_dt, 0.29605401_dt, 0.95150155_dt, 0.68107688_dt, 0.04876953_dt, 0.81634867_dt, 0.44230276_dt, 0.27679658_dt, 0.89982665_dt, 0.09595239_dt,
                     0.55365247_dt, 0.39531565_dt, 0.85705632_dt, 0.63957226_dt, 0.74025267_dt, 0.67657948_dt, 0.37976265_dt, 0.39484727_dt, 0.08795929_dt, 0.77092206_dt, 0.89698905_dt, 0.84211242_dt,
                     0.14731085_dt, 0.52229995_dt, 0.14753294_dt, 0.22475791_dt, 0.20864725_dt, 0.67087251_dt, 0.20204341_dt, 0.48909140_dt, 0.52103406_dt, 0.82231152_dt, 0.12203997_dt, 0.15674388_dt,
                     0.20966923_dt, 0.84996670_dt, 0.32026750_dt, 0.92174435_dt, 0.68080378_dt, 0.56331301_dt, 0.49627799_dt, 0.40115923_dt, 0.56273317_dt, 0.38582766_dt, 0.49648678_dt, 0.56379652_dt,
                     0.10889745_dt, 0.23793429_dt, 0.90374637_dt, 0.09422666_dt, 0.46409690_dt, 0.99461937_dt, 0.68061852_dt, 0.51415652_dt, 0.06669503_dt, 0.74768895_dt, 0.14385962_dt, 0.35806787_dt,
                     0.33224183_dt, 0.42595631_dt, 0.50546914_dt, 0.91240376_dt, 0.56241941_dt, 0.94784641_dt, 0.80585623_dt, 0.18389302_dt, 0.72425205_dt, 0.14655197_dt, 0.28808743_dt, 0.64706135_dt,
                     0.66509604_dt, 0.87511402_dt, 0.33904207_dt, 0.50080043_dt, 0.75741178_dt, 0.01645392_dt, 0.86149031_dt, 0.08653879_dt, 0.50689125_dt, 0.41499162_dt, 0.23666352_dt, 0.56608552_dt,
                     0.91345936_dt, 0.35384023_dt, 0.20315295_dt, 0.31508058_dt, 0.00442582_dt, 0.72569698_dt };

    const Tensor realOut{ -2.36083_dt, -2.15712_dt, -1.8193_dt,  -1.64201_dt, -1.94064_dt, -2.07153_dt, -1.80384_dt, -2.16272_dt, -1.77514_dt, -2.25597_dt, -2.21486_dt, -2.12661_dt, -1.64208_dt,
                          -1.66723_dt, -2.33313_dt, -1.8859_dt,  -1.76247_dt, -2.10006_dt, -2.01786_dt, -1.9211_dt,  -1.72676_dt, -1.71257_dt, -2.57798_dt, -2.37591_dt, -1.99617_dt, -1.7784_dt,
                          -1.75757_dt, -1.75149_dt, -1.87177_dt, -2.40607_dt, -1.92529_dt, -2.28408_dt, -2.43726_dt, -1.52704_dt, -1.5911_dt,  -2.33879_dt, -1.74644_dt, -1.92426_dt, -1.92231_dt,
                          -2.06891_dt, -1.64775_dt, -2.13618_dt, -1.86324_dt, -1.79368_dt, -1.6886_dt,  -1.85958_dt, -2.5414_dt,  -2.37089_dt, -1.79688_dt, -1.87279_dt, -2.36749_dt, -2.26535_dt,
                          -1.50707_dt, -1.64054_dt, -2.19546_dt, -2.10329_dt, -2.24171_dt, -1.7744_dt,  -2.14194_dt, -2.15109_dt, -1.79296_dt, -1.69034_dt, -1.97018_dt, -1.73509_dt, -2.33604_dt,
                          -1.57809_dt, -1.68913_dt, -2.45342_dt, -2.15622_dt, -2.00921_dt, -1.88971_dt, -1.84407_dt, -1.76645_dt, -1.93271_dt, -2.20662_dt, -1.72606_dt, -2.44228_dt, -2.07552_dt,
                          -1.90433_dt, -2.02272_dt, -1.95408_dt, -2.18898_dt, -1.88552_dt, -1.67229_dt, -2.29357_dt, -1.9935_dt,  -1.60991_dt, -2.01924_dt, -1.95394_dt, -2.01097_dt, -1.86665_dt,
                          -1.80469_dt, -1.64899_dt, -1.80509_dt, -1.64791_dt, -2.15878_dt, -2.57178_dt, -2.35965_dt, -1.45954_dt, -1.80323_dt, -2.04851_dt, -2.18315_dt, -2.26792_dt, -2.22199_dt,
                          -1.90141_dt, -1.67195_dt, -1.67583_dt, -2.42837_dt, -1.63426_dt, -2.3374_dt,  -2.05186_dt, -2.14889_dt, -1.85942_dt, -1.86129_dt, -2.21179_dt, -2.19484_dt, -1.84136_dt,
                          -1.75665_dt, -1.99025_dt, -1.88465_dt, -2.52303_dt, -2.10674_dt, -1.63329_dt, -2.27694_dt, -1.6215_dt,  -1.89192_dt, -2.39243_dt, -1.62486_dt, -1.9989_dt,  -2.16441_dt,
                          -1.54138_dt, -2.34525_dt, -1.88755_dt, -2.14994_dt, -1.6882_dt,  -1.90568_dt, -1.805_dt,   -1.86867_dt, -2.16549_dt, -2.1504_dt,  -2.39951_dt, -1.71654_dt, -1.59048_dt,
                          -1.64535_dt, -2.34016_dt, -1.96517_dt, -2.33993_dt, -2.19599_dt, -2.2121_dt,  -1.74987_dt, -2.2187_dt,  -1.93165_dt, -1.89971_dt, -1.59843_dt, -2.34087_dt, -2.30617_dt,
                          -2.25324_dt, -1.61295_dt, -2.14265_dt, -1.54117_dt, -1.78211_dt, -1.8807_dt,  -1.94774_dt, -2.04286_dt, -1.88128_dt, -2.05819_dt, -1.94753_dt, -1.88022_dt, -2.39423_dt,
                          -2.2652_dt,  -1.59938_dt, -2.4089_dt,  -2.03903_dt, -1.50851_dt, -1.82251_dt, -1.82416_dt, -2.27162_dt, -1.59062_dt, -2.19445_dt, -1.98025_dt, -2.00607_dt, -1.91236_dt,
                          -2.13265_dt, -1.72571_dt, -2.0757_dt,  -1.69027_dt, -1.83226_dt, -2.45422_dt, -1.91386_dt, -2.32147_dt, -2.17994_dt, -1.82096_dt, -1.80293_dt, -1.59291_dt, -2.12898_dt,
                          -1.96722_dt, -1.64483_dt, -2.38579_dt, -1.54075_dt, -2.3157_dt,  -1.89535_dt, -1.98725_dt, -2.16558_dt, -1.8627_dt,  -1.51532_dt, -2.07494_dt, -2.22563_dt, -2.1137_dt,
                          -2.42436_dt, -1.70308_dt };

    WorkflowEager work(raul::CompressionMode::NONE, raul::CalculationMode::DETERMINISTIC, raul::AllocationMode::STANDARD, raul::ExecutionTarget::GPU);
    work.getKernelManager().setExecutionPolicy(raul::KernelExecutionPolicy::DefaultParams);

    work.add<DataLayer>("data", DataParams{ { "in" }, DEPTH, HEIGHT, WIDTH });
    work.add<LogSoftMaxActivation>("softmax", BasicParamsWithDim{ { "in" }, { "out" }, "width" });

    TENSORS_CREATE(BATCH_SIZE)

    MemoryManagerGPU& memory_manager = work.getMemoryManager<MemoryManagerGPU>();
    memory_manager["in"] = TORANGE(in);

    ASSERT_NO_THROW(work.forwardPassTraining());
    const Tensor& out = memory_manager["out"];
    EXPECT_EQ(out.size(), realOut.size());
    for (size_t i = 0; i < out.size(); ++i)
    {
        EXPECT_NEAR(out[i], realOut[i], eps);
    }

    ASSERT_NO_THROW(work.backwardPassTraining());
}

}
