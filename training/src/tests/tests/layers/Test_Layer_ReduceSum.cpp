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
#include <training/base/layers/basic/ReduceSumLayer.h>
#include <training/compiler/Workflow.h>

namespace UT
{

TEST(TestLayerReduceSum, InputNumExceedsUnit)
{
    PROFILE_TEST

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    // Apply function
    ASSERT_THROW(raul::ReduceSumLayer("rsum", raul::BasicParamsWithDim{ { "x", "y" }, { "x_out" } }, networkParameters), raul::Exception);
}

TEST(TestLayerReduceSum, OutputNumExceedsUnit)
{
    PROFILE_TEST

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    // Apply function
    ASSERT_THROW(raul::ReduceSumLayer("rsum", raul::BasicParamsWithDim{ { "x" }, { "x_out", "y_out" } }, networkParameters), raul::Exception);
}

TEST(TestLayerReduceSum, ForwardUnit)
{
    PROFILE_TEST
    // Test parameters
    const auto batch = 2;
    const auto depth = 3;
    const auto height = 4;
    const auto width = 5;
    const auto eps = TODTYPE(1e-6);

    // See reduce_sum.py
    const raul::Tensor x{ 0.49625659_dt, 0.76822180_dt, 0.08847743_dt, 0.13203049_dt, 0.30742282_dt, 0.63407868_dt, 0.49009341_dt, 0.89644474_dt, 0.45562798_dt, 0.63230628_dt, 0.34889346_dt,
                          0.40171731_dt, 0.02232575_dt, 0.16885895_dt, 0.29388845_dt, 0.51852179_dt, 0.69766760_dt, 0.80001140_dt, 0.16102946_dt, 0.28226858_dt, 0.68160856_dt, 0.91519397_dt,
                          0.39709991_dt, 0.87415588_dt, 0.41940832_dt, 0.55290705_dt, 0.95273811_dt, 0.03616482_dt, 0.18523103_dt, 0.37341738_dt, 0.30510002_dt, 0.93200040_dt, 0.17591017_dt,
                          0.26983356_dt, 0.15067977_dt, 0.03171951_dt, 0.20812976_dt, 0.92979902_dt, 0.72310919_dt, 0.74233627_dt, 0.52629578_dt, 0.24365824_dt, 0.58459234_dt, 0.03315264_dt,
                          0.13871688_dt, 0.24223500_dt, 0.81546897_dt, 0.79316062_dt, 0.27825248_dt, 0.48195881_dt, 0.81978035_dt, 0.99706656_dt, 0.69844109_dt, 0.56754643_dt, 0.83524317_dt,
                          0.20559883_dt, 0.59317201_dt, 0.11234725_dt, 0.15345693_dt, 0.24170822_dt, 0.72623652_dt, 0.70108020_dt, 0.20382375_dt, 0.65105355_dt, 0.77448601_dt, 0.43689132_dt,
                          0.51909077_dt, 0.61585236_dt, 0.81018829_dt, 0.98009706_dt, 0.11468822_dt, 0.31676513_dt, 0.69650495_dt, 0.91427469_dt, 0.93510365_dt, 0.94117838_dt, 0.59950727_dt,
                          0.06520867_dt, 0.54599625_dt, 0.18719733_dt, 0.03402293_dt, 0.94424623_dt, 0.88017988_dt, 0.00123602_dt, 0.59358603_dt, 0.41576999_dt, 0.41771942_dt, 0.27112156_dt,
                          0.69227809_dt, 0.20384824_dt, 0.68329567_dt, 0.75285405_dt, 0.85793579_dt, 0.68695557_dt, 0.00513238_dt, 0.17565155_dt, 0.74965751_dt, 0.60465068_dt, 0.10995799_dt,
                          0.21209025_dt, 0.97037464_dt, 0.83690894_dt, 0.28198743_dt, 0.37415761_dt, 0.02370095_dt, 0.49101293_dt, 0.12347054_dt, 0.11432165_dt, 0.47245020_dt, 0.57507253_dt,
                          0.29523486_dt, 0.79668880_dt, 0.19573045_dt, 0.95368505_dt, 0.84264994_dt, 0.07835853_dt, 0.37555784_dt, 0.52256131_dt, 0.57295054_dt, 0.61858714_dt };

    std::string dimensions[] = { "default", "batch", "depth", "height", "width" };

    raul::Tensor realOutputs[] = {
        { 58.35741806_dt },
        { 1.22249317_dt, 1.46930194_dt, 0.29230118_dt, 0.78308403_dt, 1.08190882_dt, 1.07097006_dt, 1.00918412_dt, 1.51229715_dt, 1.26581621_dt, 1.61240339_dt, 0.46358168_dt, 0.71848243_dt,
          0.71883070_dt, 1.08313370_dt, 1.22899210_dt, 1.45970011_dt, 1.29717493_dt, 0.86522007_dt, 0.70702571_dt, 0.46946591_dt, 0.71563148_dt, 1.85944021_dt, 1.27727985_dt, 0.87539190_dt,
          1.01299429_dt, 0.96867704_dt, 1.37045753_dt, 0.30728638_dt, 0.87750912_dt, 0.57726562_dt, 0.98839569_dt, 1.68485451_dt, 1.03384590_dt, 0.95678914_dt, 0.15581214_dt, 0.20737106_dt,
          0.95778728_dt, 1.53444970_dt, 0.83306718_dt, 0.95442653_dt, 1.49667048_dt, 1.08056712_dt, 0.86657977_dt, 0.40731025_dt, 0.16241783_dt, 0.73324794_dt, 0.93893951_dt, 0.90748227_dt,
          0.75070268_dt, 1.05703139_dt, 1.11501527_dt, 1.79375529_dt, 0.89417154_dt, 1.52123141_dt, 1.67789316_dt, 0.28395736_dt, 0.96872985_dt, 0.63490856_dt, 0.72640747_dt, 0.86029536_dt },
        { 1.70416093_dt, 1.92707396_dt, 1.07016969_dt, 1.03933907_dt, 0.86554801_dt, 1.42922068_dt, 2.25830054_dt, 1.72577024_dt, 0.91911149_dt, 1.48768258_dt,
          1.47377384_dt, 2.33078432_dt, 0.89667702_dt, 1.00623894_dt, 1.27981138_dt, 0.75584012_dt, 1.49896932_dt, 1.84215772_dt, 1.03759551_dt, 1.26631308_dt,
          1.73063409_dt, 2.48223543_dt, 1.36599112_dt, 1.02644718_dt, 1.39177299_dt, 1.34367418_dt, 1.06028080_dt, 1.00129557_dt, 1.97491670_dt, 1.75901783_dt,
          1.09321880_dt, 1.86630797_dt, 1.75017118_dt, 2.55491543_dt, 1.78288603_dt, 1.19518840_dt, 1.72472262_dt, 1.19242072_dt, 1.22890472_dt, 1.01787472_dt },
        { 1.99775052_dt, 2.35770011_dt, 1.80725932_dt, 0.91754687_dt, 1.51588607_dt, 1.57133508_dt, 3.00806236_dt, 1.53897393_dt, 2.05232978_dt, 1.68584180_dt,
          1.79391003_dt, 2.64936590_dt, 2.18854141_dt, 1.03240848_dt, 1.69762707_dt, 2.21899438_dt, 2.13644338_dt, 1.58138967_dt, 2.92151260_dt, 2.87688398_dt,
          1.30874014_dt, 2.86447716_dt, 2.61388803_dt, 1.49042773_dt, 1.01465690_dt, 1.83498108_dt, 2.13262606_dt, 1.11460090_dt, 2.37324333_dt, 2.06001043_dt },
        { 1.79240918_dt, 3.10855103_dt, 1.23568392_dt, 2.45949888_dt, 3.28746653_dt, 2.10045838_dt, 1.83352399_dt, 2.63509369_dt, 1.52641582_dt, 2.61107588_dt, 3.91807747_dt, 1.30628324_dt,
          3.05668020_dt, 3.36211991_dt, 2.97733665_dt, 2.33908796_dt, 2.45327091_dt, 2.00073719_dt, 2.98617339_dt, 1.85200787_dt, 2.48712969_dt, 1.77632785_dt, 3.08398914_dt, 2.16801548_dt }
    };

    for (size_t iter = 0; iter < std::size(dimensions); ++iter)
    {
        MANAGERS_DEFINE
        NETWORK_PARAMS_DEFINE(networkParameters);

        work.add<raul::DataLayer>("data", raul::DataParams{ { "x" }, depth, height, width });
        work.add<raul::ReduceSumLayer>("rsum", raul::BasicParamsWithDim{ { "x" }, { "out" }, dimensions[iter] });
        TENSORS_CREATE(batch);
        memory_manager["x"] = TORANGE(x);

        work.forwardPassTraining();

        // Checks
        const auto& outTensor = memory_manager["out"];
        EXPECT_EQ(outTensor.size(), realOutputs[iter].size());
        for (size_t i = 0; i < outTensor.size(); ++i)
        {
            ASSERT_TRUE(tools::expect_near_relative(outTensor[i], realOutputs[iter][i], eps));
        }
    }
}

TEST(TestLayerReduceSum, BackwardUnit)
{
    PROFILE_TEST
    // Test parameters
    const auto batch = 2;
    const auto depth = 3;
    const auto height = 4;
    const auto width = 5;

    const raul::Tensor x{ 0.49625659_dt, 0.76822180_dt, 0.08847743_dt, 0.13203049_dt, 0.30742282_dt, 0.63407868_dt, 0.49009341_dt, 0.89644474_dt, 0.45562798_dt, 0.63230628_dt, 0.34889346_dt,
                          0.40171731_dt, 0.02232575_dt, 0.16885895_dt, 0.29388845_dt, 0.51852179_dt, 0.69766760_dt, 0.80001140_dt, 0.16102946_dt, 0.28226858_dt, 0.68160856_dt, 0.91519397_dt,
                          0.39709991_dt, 0.87415588_dt, 0.41940832_dt, 0.55290705_dt, 0.95273811_dt, 0.03616482_dt, 0.18523103_dt, 0.37341738_dt, 0.30510002_dt, 0.93200040_dt, 0.17591017_dt,
                          0.26983356_dt, 0.15067977_dt, 0.03171951_dt, 0.20812976_dt, 0.92979902_dt, 0.72310919_dt, 0.74233627_dt, 0.52629578_dt, 0.24365824_dt, 0.58459234_dt, 0.03315264_dt,
                          0.13871688_dt, 0.24223500_dt, 0.81546897_dt, 0.79316062_dt, 0.27825248_dt, 0.48195881_dt, 0.81978035_dt, 0.99706656_dt, 0.69844109_dt, 0.56754643_dt, 0.83524317_dt,
                          0.20559883_dt, 0.59317201_dt, 0.11234725_dt, 0.15345693_dt, 0.24170822_dt, 0.72623652_dt, 0.70108020_dt, 0.20382375_dt, 0.65105355_dt, 0.77448601_dt, 0.43689132_dt,
                          0.51909077_dt, 0.61585236_dt, 0.81018829_dt, 0.98009706_dt, 0.11468822_dt, 0.31676513_dt, 0.69650495_dt, 0.91427469_dt, 0.93510365_dt, 0.94117838_dt, 0.59950727_dt,
                          0.06520867_dt, 0.54599625_dt, 0.18719733_dt, 0.03402293_dt, 0.94424623_dt, 0.88017988_dt, 0.00123602_dt, 0.59358603_dt, 0.41576999_dt, 0.41771942_dt, 0.27112156_dt,
                          0.69227809_dt, 0.20384824_dt, 0.68329567_dt, 0.75285405_dt, 0.85793579_dt, 0.68695557_dt, 0.00513238_dt, 0.17565155_dt, 0.74965751_dt, 0.60465068_dt, 0.10995799_dt,
                          0.21209025_dt, 0.97037464_dt, 0.83690894_dt, 0.28198743_dt, 0.37415761_dt, 0.02370095_dt, 0.49101293_dt, 0.12347054_dt, 0.11432165_dt, 0.47245020_dt, 0.57507253_dt,
                          0.29523486_dt, 0.79668880_dt, 0.19573045_dt, 0.95368505_dt, 0.84264994_dt, 0.07835853_dt, 0.37555784_dt, 0.52256131_dt, 0.57295054_dt, 0.61858714_dt };

    // Always one
    const raul::Tensor realGrad = raul::Tensor("realGrad", batch, depth, height, width, 1.0_dt);

    std::string dimensions[] = { "default", "batch", "depth", "height", "width" };

    for (size_t iter = 0; iter < std::size(dimensions); ++iter)
    {
        MANAGERS_DEFINE
        NETWORK_PARAMS_DEFINE(networkParameters);

        work.add<raul::DataLayer>("data", raul::DataParams{ { "x" }, depth, height, width });
        work.add<raul::ReduceSumLayer>("rsum", raul::BasicParamsWithDim{ { "x" }, { "out" }, dimensions[iter] });
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