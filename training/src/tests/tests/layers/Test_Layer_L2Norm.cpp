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
#include <training/base/layers/basic/L2NormLayer.h>
#include <training/compiler/Workflow.h>

namespace UT
{

TEST(TestLayerL2Norm, InputNumExceedsUnit)
{
    PROFILE_TEST

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    // Apply function
    ASSERT_THROW(raul::L2NormLayer("L2Norm", raul::BasicParams{ { "x", "y" }, { "x_out" } }, networkParameters), raul::Exception);
}

TEST(TestLayerL2Norm, OutputNumExceedsUnit)
{
    PROFILE_TEST

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    // Apply function
    ASSERT_THROW(raul::L2NormLayer("L2Norm", raul::BasicParams{ { "x" }, { "x_out", "y_out" } }, networkParameters), raul::Exception);
}

// See l2_norm.py
TEST(TestLayerL2Norm, ForwardUnit)
{
    PROFILE_TEST
    // Test parameters
    const auto batch = 2;
    const auto depth = 3;
    const auto height = 4;
    const auto width = 5;
    const auto eps = TODTYPE(1e-5);

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

    const raul::Tensor realOut{ 0.6721557_dt,  0.44604594_dt, 0.3569147_dt,   0.46978992_dt,  0.03404503_dt, 0.49951473_dt, 0.5399656_dt,    0.6365072_dt,   0.16512035_dt,  0.16283718_dt,
                                0.27581632_dt, 0.6418803_dt,  0.118352465_dt, 0.48701918_dt,  0.5106075_dt,  0.72888184_dt, 0.0076670246_dt, 0.4222769_dt,   0.5140785_dt,   0.16148654_dt,
                                0.55369675_dt, 0.41433716_dt, 0.08165184_dt,  0.51368034_dt,  0.50120866_dt, 0.2807033_dt,  0.50102305_dt,   0.17546603_dt,  0.71038985_dt,  0.3670688_dt,
                                0.6489302_dt,  0.16250214_dt, 0.5553839_dt,   0.36009237_dt,  0.33817825_dt, 0.14953724_dt, 0.5851454_dt,    0.6032483_dt,   0.21336947_dt,  0.47519302_dt,
                                0.15858118_dt, 0.521656_dt,   0.6453867_dt,   0.04584568_dt,  0.53301144_dt, 0.54732686_dt, 0.48670292_dt,   0.13645469_dt,  0.61182964_dt,  0.26570317_dt,
                                0.41248563_dt, 0.31594637_dt, 0.66419536_dt,  0.49053222_dt,  0.21967259_dt, 0.6413641_dt,  0.38654852_dt,   0.15132637_dt,  0.5894051_dt,   0.26255333_dt,
                                0.2411586_dt,  0.27018458_dt, 0.5201679_dt,   0.64289933_dt,  0.4300562_dt,  0.13524175_dt, 0.097497284_dt,  0.47814378_dt,  0.7071593_dt,   0.4934656_dt,
                                0.35651854_dt, 0.42765474_dt, 0.5185791_dt,   0.2128231_dt,   0.6130153_dt,  0.4268994_dt,  0.5439377_dt,    0.18997952_dt,  0.3773301_dt,   0.58601916_dt,
                                0.10707895_dt, 0.28270382_dt, 0.89391685_dt,  0.12814239_dt,  0.3051305_dt,  0.20540966_dt, 0.024660394_dt,  0.705217_dt,    0.5019687_dt,   0.45595506_dt,
                                0.24734943_dt, 0.8482956_dt,  0.42987427_dt,  0.055234674_dt, 0.17711578_dt, 0.5974544_dt,  0.6625676_dt,    0.06740211_dt,  0.061032265_dt, 0.44247523_dt,
                                0.30646726_dt, 0.42507973_dt, 0.5155844_dt,   0.6637676_dt,   0.13773313_dt, 0.1499963_dt,  0.42385635_dt,   0.059203856_dt, 0.62565774_dt,  0.6347395_dt,
                                0.48109287_dt, 0.38150218_dt, 0.11483272_dt,  0.25942576_dt,  0.7365578_dt,  0.5951317_dt,  0.48851663_dt,   0.4819726_dt,   0.2715976_dt,   0.31797317_dt };

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    // Apply function
    work.add<raul::DataLayer>("data", raul::DataParams{ { "x" }, depth, height, width });
    work.add<raul::L2NormLayer>("L2Norm", raul::BasicParams{ { "x" }, { "out" } });
    TENSORS_CREATE(batch);
    memory_manager["x"] = TORANGE(x);

    work.forwardPassTraining();

    // Checks
    const auto output = memory_manager["out"];
    for (size_t q = 0; q < output.size(); ++q)
    {
        ASSERT_TRUE(tools::expect_near_relative(output[q], realOut[q], eps));
    }
}

TEST(TestLayerL2Norm, BackwardUnit)
{
    PROFILE_TEST
    // Test parameters
    const auto batch = 2;
    const auto depth = 3;
    const auto height = 4;
    const auto width = 5;
    const auto eps = TODTYPE(1e-3);

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

    const raul::Tensor realGrad{
        -0.33393586_dt,  0.118637204_dt,  0.29703897_dt,   0.07111204_dt,   0.94328314_dt,   -0.00072962046_dt, -0.05986941_dt,  -0.2010144_dt,   0.4881594_dt,    0.4914974_dt,    0.3901734_dt,
        -0.27136272_dt,  0.6747357_dt,    0.008495986_dt,  -0.034131825_dt, -0.27306557_dt,  0.79875934_dt,     0.18259168_dt,   0.04616183_dt,   0.57016224_dt,   -0.10866243_dt,  0.10974151_dt,
        0.6311248_dt,    -0.045948803_dt, -0.026403248_dt, 0.35727605_dt,   -0.016167462_dt, 0.53565395_dt,     -0.37104553_dt,  0.210886_dt,     -0.23267573_dt,  0.4545588_dt,    -0.10051185_dt,
        0.17539966_dt,   0.20636037_dt,   0.48222262_dt,   -0.12854856_dt,  -0.1539309_dt,   0.3927227_dt,      0.025616884_dt,  0.4657765_dt,    0.0043483377_dt, -0.15289992_dt,  0.6090509_dt,
        -0.010083258_dt, -0.100245655_dt, 0.0026724339_dt, 0.59727055_dt,   -0.20974863_dt,  0.37785214_dt,     0.09005839_dt,   0.22792202_dt,   -0.26939768_dt,  -0.021396697_dt, 0.36540654_dt,
        -0.23513079_dt,  0.16686541_dt,   0.537951_dt,     -0.15316045_dt,  0.36247975_dt,   0.4222408_dt,      0.36986968_dt,   -0.081171215_dt, -0.30261374_dt,  0.08141583_dt,   0.5485108_dt,
        0.6018827_dt,    0.06363636_dt,   -0.26019895_dt,  0.04197079_dt,   0.27950418_dt,   0.10397804_dt,     -0.12037468_dt,  0.6340678_dt,    -0.3533926_dt,   0.056410372_dt,  -0.09407121_dt,
        0.36102915_dt,   0.12014389_dt,   -0.14817727_dt,  0.7797367_dt,    0.4916467_dt,    -0.51096964_dt,    0.74518484_dt,   0.4548586_dt,    0.47861296_dt,   0.74661386_dt,   -0.26246226_dt,
        0.038898468_dt,  0.10712385_dt,   0.66835237_dt,   -0.5808474_dt,   0.28893405_dt,   1.0677054_dt,      0.8143485_dt,    -0.06649917_dt,  -0.15092981_dt,  0.62080663_dt,   0.6290662_dt,
        0.13445854_dt,   0.307836_dt,     0.10684222_dt,   -0.046521723_dt, -0.29762423_dt,  0.593763_dt,       0.4861635_dt,    0.13406885_dt,   0.60289294_dt,   -0.12538183_dt,  -0.13705802_dt,
        0.040271282_dt,  0.1966694_dt,    0.6154494_dt,    0.38837925_dt,   -0.36091304_dt,  -0.21682405_dt,    -0.040543377_dt, -0.029723346_dt, 0.31811723_dt,   0.24143845_dt
    };

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    // Apply function
    work.add<raul::DataLayer>("data", raul::DataParams{ { "x" }, depth, height, width });
    work.add<raul::L2NormLayer>("L2Norm", raul::BasicParams{ { "x" }, { "out" } });
    TENSORS_CREATE(batch);
    memory_manager["x"] = TORANGE(x);
    memory_manager[raul::Name("out").grad()] = 1.0_dt;

    work.forwardPassTraining();
    work.backwardPassTraining();

    // Checks
    const auto xTensorGrad = memory_manager[raul::Name("x").grad()];
    for (size_t q = 0; q < xTensorGrad.size(); ++q)
    {
        ASSERT_TRUE(tools::expect_near_relative(xTensorGrad[q], realGrad[q], eps));
    }
}

}