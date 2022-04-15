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

#include <training/base/common/Common.h>
#include <training/base/common/MemoryManager.h>
#include <training/base/layers/activations/SoftMaxActivation.h>
#include <training/base/layers/basic/DataLayer.h>
#include <training/compiler/Workflow.h>

namespace UT
{
TEST(TestSoftMax, SoftMaxUnit)
{
    PROFILE_TEST
    using namespace raul;

    constexpr auto eps = 1e-4_dt;

    size_t BATCH_SIZE = 2;
    size_t WIDTH = 2;
    size_t HEIGHT = 2;
    size_t DEPTH = 3;

    Dimension dims[] = { Dimension::Batch, Dimension::Depth, Dimension::Height, Dimension::Width };
    std::string names[] = { "batch", "depth", "height", "width" };

    Tensor realOut[] = { { 0.5_dt, 0.5_dt, 0.26894143_dt, 0.26894143_dt, 0.04742587_dt, 0.04742587_dt, 0.26894143_dt, 0.26894143_dt, 0.26894143_dt, 0.5_dt, 0.26894143_dt, 0.00669285_dt,
                           0.5_dt, 0.5_dt, 0.7310586_dt,  0.7310586_dt,  0.95257413_dt, 0.95257413_dt, 0.7310586_dt,  0.7310586_dt,  0.7310586_dt,  0.5_dt, 0.7310586_dt,  0.9933072_dt },
                         { 0.3333_dt, 0.3333_dt, 0.3333_dt, 0.3333_dt, 0.3333_dt, 0.3333_dt, 0.3333_dt, 0.3333_dt, 0.3333_dt, 0.3333_dt, 0.3333_dt, 0.3333_dt,
                           0.0420_dt, 0.0453_dt, 0.3333_dt, 0.0177_dt, 0.8438_dt, 0.9094_dt, 0.3333_dt, 0.0177_dt, 0.1142_dt, 0.0453_dt, 0.3333_dt, 0.9647_dt },
                         { 0.2689_dt, 0.2689_dt, 0.7311_dt, 0.7311_dt, 0.2689_dt, 0.2689_dt, 0.7311_dt, 0.7311_dt, 0.2689_dt, 0.2689_dt, 0.7311_dt, 0.7311_dt,
                           0.1192_dt, 0.1192_dt, 0.8808_dt, 0.8808_dt, 0.7311_dt, 0.7311_dt, 0.2689_dt, 0.2689_dt, 0.2689_dt, 0.0025_dt, 0.7311_dt, 0.9975_dt },
                         { 0.5000_dt, 0.5000_dt, 0.5000_dt, 0.5000_dt, 0.5000_dt, 0.5000_dt, 0.5000_dt, 0.5000_dt, 0.5000_dt, 0.5000_dt, 0.5000_dt, 0.5000_dt,
                           0.5000_dt, 0.5000_dt, 0.5000_dt, 0.5000_dt, 0.5000_dt, 0.5000_dt, 0.5000_dt, 0.5000_dt, 0.7311_dt, 0.2689_dt, 0.0180_dt, 0.9820_dt } };

    Tensor realGrad[] = { { -0.0_dt,        0.0_dt,        -0.19661196_dt, -0.19661196_dt, -0.13552998_dt, -0.13552998_dt, -0.19661196_dt, -0.19661196_dt,
                            -0.19661196_dt, 0.0_dt,        -0.19661196_dt, -0.03324028_dt, 0.00000000_dt,  0.00000000_dt,  0.19661188_dt,  0.19661188_dt,
                            0.13552995_dt,  0.13552995_dt, 0.19661188_dt,  0.19661188_dt,  0.19661190_dt,  0.00000000,     0.19661188_dt,  0.03324006_dt },
                          { -9.9341e-09_dt, -9.9341e-09_dt, -1.9868e-08_dt, -1.9868e-08_dt, -9.9341e-09_dt, -9.9341e-09_dt, -1.9868e-08_dt, -1.9868e-08_dt,
                            -9.9341e-09_dt, -9.9341e-09_dt, -1.9868e-08_dt, -1.9868e-08_dt, -1.1114e-01_dt, -1.2353e-01_dt, 0.0000e+00_dt,  -6.8176e-02_dt,
                            2.9906e-01_dt,  2.4707e-01_dt,  0.0000e+00_dt,  -6.8176e-02_dt, -1.8792e-01_dt, -1.2353e-01_dt, 0.0000e+00_dt,  1.3635e-01_dt },
                          { -0.1966_dt, -0.1966_dt, 0.1966_dt, 0.1966_dt, -0.1966_dt, -0.1966_dt, 0.1966_dt,  0.1966_dt,  -0.1966_dt, -0.1966_dt, 0.1966_dt, 0.1966_dt,
                            -0.2100_dt, -0.2100_dt, 0.2100_dt, 0.2100_dt, 0.1966_dt,  0.1966_dt,  -0.1966_dt, -0.1966_dt, -0.1966_dt, -0.0148_dt, 0.1966_dt, 0.0148_dt },
                          { 0.0000_dt, 0.0000_dt, 0.0000_dt, 0.0000_dt, 0.0000_dt, 0.0000_dt, 0.0000_dt, 0.0000_dt, 0.0000_dt, 0.0000_dt,  0.0000_dt,  0.0000_dt,
                            0.0000_dt, 0.0000_dt, 0.0000_dt, 0.0000_dt, 0.0000_dt, 0.0000_dt, 0.0000_dt, 0.0000_dt, 0.1966_dt, -0.1966_dt, -0.0707_dt, 0.0707_dt } };

    const Tensor raw = { 1._dt, 1._dt, 2._dt, 2._dt, 1._dt, 1._dt, 2._dt, 2._dt, 1._dt, 1._dt, 2._dt, 2._dt, 1._dt, 1._dt, 3._dt, 3._dt, 4._dt, 4._dt, 3._dt, 3._dt, 2._dt, 1._dt, 3._dt, 7._dt };

    for (size_t k = 0; k < std::size(dims); ++k)
    {
        MANAGERS_DEFINE
        NETWORK_PARAMS_DEFINE(networkParameters);

        memory_manager.createTensor("in", BATCH_SIZE, DEPTH, HEIGHT, WIDTH, raw);

        work.add<DataLayer>("data", DataParams{ { "in" }, DEPTH, HEIGHT, WIDTH });
        work.add<SoftMaxActivation>("sm", BasicParamsWithDim{ { "in" }, { "out" }, dims[k] });

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

        std::cout << " - SoftMax [" << names[k] << "] forward is Ok.\n";

        work.backwardPassTraining();

        const raul::Tensor& inGrad = memory_manager[raul::Name("in").grad()];

        EXPECT_EQ(inGrad.size(), realGrad[k].size());
        for (size_t i = 0; i < inGrad.size(); ++i)
        {
            EXPECT_NEAR(inGrad[i], realGrad[k][i], eps);
        }

        std::cout << " - SoftMax [" << names[k] << "] backward is Ok.\n";
    }
}

TEST(TestSoftMax, DoubleForwardHeightUnit)
{
    PROFILE_TEST

    using namespace raul;

    constexpr size_t BATCH_SIZE = 2;
    constexpr size_t DEPTH = 1;
    constexpr size_t HEIGHT = 48;
    constexpr size_t WIDTH = 1;
    constexpr dtype eps = 1.0e-5_dt;

    const Tensor in{ 8.34375_dt, 6.94922_dt, 7.875_dt, 7.07422_dt, 8.58594_dt, 8.39062_dt, 8.53906_dt, 9.40625_dt, 9.6875_dt,
        9.48438_dt, 8.66406_dt, 9.42188_dt, 10.1172_dt, 10.1328_dt, 9.51562_dt, 10.1562_dt, 12.1641_dt, 12.75_dt, 13.0547_dt,
        12.6016_dt, 12.4688_dt, 12.3594_dt, 12.2422_dt, 12.1016_dt, 11.9375_dt, 11.8047_dt, 11.7812_dt, 11.9141_dt, 12.0625_dt,
        12.2422_dt, 12.4141_dt, 12.5547_dt, 12.7031_dt, 12.8281_dt, 12.9531_dt, 13.0625_dt, 13.1797_dt, 13.3047_dt, 13.3906_dt,
        13.5078_dt, 13.5781_dt, 13.6562_dt, 13.7422_dt, 13.8203_dt, 13.8906_dt, 13.9531_dt, 14.0156_dt, 14.0938_dt, 8.34375_dt,
        6.94922_dt, 7.875_dt, 7.07422_dt, 8.58594_dt, 8.39062_dt, 8.53906_dt, 9.40625_dt, 9.6875_dt, 9.48438_dt, 8.66406_dt,
        9.42188_dt, 10.1172_dt, 10.1328_dt, 9.51562_dt, 10.1562_dt, 12.1641_dt, 12.75_dt, 13.0547_dt, 12.6016_dt, 12.4688_dt,
        12.3594_dt, 12.2422_dt, 12.1016_dt, 11.9375_dt, 11.8047_dt, 11.7812_dt, 11.9141_dt, 12.0625_dt, 12.2422_dt, 12.4141_dt,
        12.5547_dt, 12.7031_dt, 12.8281_dt, 12.9531_dt, 13.0625_dt, 13.1797_dt, 13.3047_dt, 13.3906_dt, 13.5078_dt, 13.5781_dt,
        13.6562_dt, 13.7422_dt, 13.8203_dt, 13.8906_dt, 13.9531_dt, 14.0156_dt, 14.0938_dt };

    const Tensor realOut{ 0.0002581_dt, 6.39958e-05_dt, 0.000161515_dt, 7.25167e-05_dt, 0.000328829_dt, 0.000270485_dt,
        0.000313769_dt, 0.000746837_dt, 0.000989398_dt, 0.000807527_dt, 0.000355547_dt, 0.000758601_dt, 0.0015205_dt,
        0.00154441_dt, 0.000833153_dt, 0.00158098_dt, 0.0117746_dt, 0.0211543_dt, 0.0286898_dt, 0.0182368_dt, 0.0159689_dt,
        0.014314_dt, 0.012731_dt, 0.0110612_dt, 0.00938715_dt, 0.00821976_dt, 0.00802886_dt, 0.00917004_dt, 0.010637_dt,
        0.012731_dt, 0.0151188_dt, 0.0174012_dt, 0.020185_dt, 0.0228726_dt, 0.0259181_dt, 0.0289145_dt, 0.0325098_dt,
        0.0368384_dt, 0.0401427_dt, 0.0451343_dt, 0.0484214_dt, 0.0523547_dt, 0.0570565_dt, 0.0616912_dt, 0.0661842_dt,
        0.0704527_dt, 0.0749965_dt, 0.0810966_dt, 0.0002581_dt, 6.39958e-05_dt, 0.000161515_dt, 7.25167e-05_dt, 0.000328829_dt,
        0.000270485_dt, 0.000313769_dt, 0.000746837_dt, 0.000989398_dt, 0.000807527_dt, 0.000355547_dt, 0.000758601_dt,
        0.0015205_dt, 0.00154441_dt, 0.000833153_dt, 0.00158098_dt, 0.0117746_dt, 0.0211543_dt, 0.0286898_dt, 0.0182368_dt,
        0.0159689_dt, 0.014314_dt, 0.012731_dt, 0.0110612_dt, 0.00938715_dt, 0.00821976_dt, 0.00802886_dt, 0.00917004_dt,
        0.010637_dt, 0.012731_dt, 0.0151188_dt, 0.0174012_dt, 0.020185_dt, 0.0228726_dt, 0.0259181_dt, 0.0289145_dt, 0.0325098_dt,
        0.0368384_dt, 0.0401427_dt, 0.0451343_dt, 0.0484214_dt, 0.0523547_dt, 0.0570565_dt, 0.0616912_dt, 0.0661842_dt,
        0.0704527_dt, 0.0749965_dt, 0.0810966_dt };
    
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    work.add<DataLayer>("data", DataParams{ { "in" }, DEPTH, HEIGHT, WIDTH });
    work.add<SoftMaxActivation>("softmax", BasicParamsWithDim{ { "in" }, { "out" }, "height" });

    TENSORS_CREATE(BATCH_SIZE)
    memory_manager["in"] = TORANGE(in);

    // Step one
    ASSERT_NO_THROW(work.forwardPassTraining());
    
    const auto& out = memory_manager["out"];
    EXPECT_EQ(out.size(), realOut.size());
    for (size_t i = 0; i < out.size(); ++i)
    {
        EXPECT_NEAR(out[i], realOut[i], eps);
    }

    // Step two
    ASSERT_NO_THROW(work.forwardPassTraining());
    EXPECT_EQ(out.size(), realOut.size());
    for (size_t i = 0; i < out.size(); ++i)
    {
        EXPECT_NEAR(out[i], realOut[i], eps);
    }
}

}