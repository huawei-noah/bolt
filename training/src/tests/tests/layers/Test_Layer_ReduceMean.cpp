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
#include <training/base/layers/basic/ReduceMeanLayer.h>
#include <training/base/layers/basic/ReduceBatchMeanLayer.h>
#include <training/compiler/Workflow.h>

namespace UT
{

// See reduce_mean.py
TEST(TestLayerReduceMean, ForwardUnit)
{
    PROFILE_TEST
    // Test parameters
    const auto batch = 2;
    const auto depth = 3;
    const auto height = 4;
    const auto width = 5;
    const auto eps = TODTYPE(1e-6);

    // See reduce_mean.py
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
        { 0.48631182_dt },
        { 0.61124659_dt, 0.73465097_dt, 0.14615059_dt, 0.39154202_dt, 0.54095441_dt, 0.53548503_dt, 0.50459206_dt, 0.75614858_dt, 0.63290811_dt, 0.80620170_dt, 0.23179084_dt, 0.35924122_dt,
          0.35941535_dt, 0.54156685_dt, 0.61449605_dt, 0.72985005_dt, 0.64858747_dt, 0.43261003_dt, 0.35351285_dt, 0.23473296_dt, 0.35781574_dt, 0.92972010_dt, 0.63863993_dt, 0.43769595_dt,
          0.50649714_dt, 0.48433852_dt, 0.68522877_dt, 0.15364319_dt, 0.43875456_dt, 0.28863281_dt, 0.49419785_dt, 0.84242725_dt, 0.51692295_dt, 0.47839457_dt, 0.07790607_dt, 0.10368553_dt,
          0.47889364_dt, 0.76722485_dt, 0.41653359_dt, 0.47721326_dt, 0.74833524_dt, 0.54028356_dt, 0.43328989_dt, 0.20365512_dt, 0.08120891_dt, 0.36662397_dt, 0.46946976_dt, 0.45374113_dt,
          0.37535134_dt, 0.52851570_dt, 0.55750763_dt, 0.89687765_dt, 0.44708577_dt, 0.76061571_dt, 0.83894658_dt, 0.14197868_dt, 0.48436493_dt, 0.31745428_dt, 0.36320373_dt, 0.43014768_dt },
        { 0.56805366_dt, 0.64235801_dt, 0.35672322_dt, 0.34644637_dt, 0.28851601_dt, 0.47640690_dt, 0.75276685_dt, 0.57525676_dt, 0.30637050_dt, 0.49589419_dt,
          0.49125794_dt, 0.77692813_dt, 0.29889235_dt, 0.33541298_dt, 0.42660379_dt, 0.25194672_dt, 0.49965644_dt, 0.61405259_dt, 0.34586516_dt, 0.42210436_dt,
          0.57687801_dt, 0.82741183_dt, 0.45533037_dt, 0.34214905_dt, 0.46392432_dt, 0.44789138_dt, 0.35342693_dt, 0.33376518_dt, 0.65830559_dt, 0.58633929_dt,
          0.36440626_dt, 0.62210268_dt, 0.58339041_dt, 0.85163850_dt, 0.59429532_dt, 0.39839613_dt, 0.57490754_dt, 0.39747357_dt, 0.40963492_dt, 0.33929157_dt },
        { 0.49943763_dt, 0.58942503_dt, 0.45181483_dt, 0.22938672_dt, 0.37897152_dt, 0.39283377_dt, 0.75201559_dt, 0.38474348_dt, 0.51308244_dt, 0.42146045_dt,
          0.44847751_dt, 0.66234148_dt, 0.54713535_dt, 0.25810212_dt, 0.42440677_dt, 0.55474859_dt, 0.53411084_dt, 0.39534742_dt, 0.73037815_dt, 0.71922100_dt,
          0.32718503_dt, 0.71611929_dt, 0.65347201_dt, 0.37260693_dt, 0.25366423_dt, 0.45874527_dt, 0.53315651_dt, 0.27865022_dt, 0.59331083_dt, 0.51500261_dt },
        { 0.35848182_dt, 0.62171018_dt, 0.24713679_dt, 0.49189979_dt, 0.65749329_dt, 0.42009169_dt, 0.36670479_dt, 0.52701873_dt, 0.30528316_dt, 0.52221519_dt, 0.78361547_dt, 0.26125664_dt,
          0.61133605_dt, 0.67242396_dt, 0.59546733_dt, 0.46781760_dt, 0.49065417_dt, 0.40014744_dt, 0.59723467_dt, 0.37040156_dt, 0.49742594_dt, 0.35526556_dt, 0.61679780_dt, 0.43360311_dt }
    };

    for (size_t iter = 0; iter < std::size(dimensions); ++iter)
    {
        MANAGERS_DEFINE
        NETWORK_PARAMS_DEFINE(networkParameters);

        work.add<raul::DataLayer>("data", raul::DataParams{ { "x" }, depth, height, width });
        work.add<raul::ReduceMeanLayer>("rmean", raul::BasicParamsWithDim{ { "x" }, { "out" }, dimensions[iter] });
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

TEST(TestLayerReduceMean, BackwardUnit)
{
    PROFILE_TEST
    // Test parameters
    const auto batch = 2;
    const auto depth = 3;
    const auto height = 4;
    const auto width = 5;
    const auto eps = TODTYPE(1e-6);

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
    const raul::Tensor realGrads[] = { raul::Tensor("realGrad(Default)", batch, depth, height, width, 0.00833333_dt),
                                       raul::Tensor("realGrad(Batch)", batch, depth, height, width, 0.5_dt),
                                       raul::Tensor("realGrad(Depth)", batch, depth, height, width, 0.33333334_dt),
                                       raul::Tensor("realGrad(Height)", batch, depth, height, width, 0.25_dt),
                                       raul::Tensor("realGrad(Width)", batch, depth, height, width, 0.2_dt) };

    std::string dimensions[] = { "default", "batch", "depth", "height", "width" };

    for (size_t iter = 0; iter < std::size(dimensions); ++iter)
    {

        MANAGERS_DEFINE
        NETWORK_PARAMS_DEFINE(networkParameters);

        work.add<raul::DataLayer>("data", raul::DataParams{ { "x" }, depth, height, width });
        work.add<raul::ReduceMeanLayer>("rmean", raul::BasicParamsWithDim{ { "x" }, { "out" }, dimensions[iter] });
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