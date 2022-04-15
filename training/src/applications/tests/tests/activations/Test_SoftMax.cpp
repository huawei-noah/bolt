// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <tests/tools/TestTools.h>

#include <tests/tools/TestTools.h>

#include <training/api/API.h>
#include <training/common/Common.h>
#include <training/common/MemoryManager.h>
#include <training/layers/activations/SoftMaxActivation.h>
#include <training/layers/basic/DataLayer.h>
#include <training/network/Workflow.h>

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

TEST(TestSoftMax, GpuDefaultUnit)
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

    const Tensor realOut{ 0.00741023_dt, 0.00908448_dt, 0.0127354_dt,  0.0152058_dt,  0.0112802_dt,  0.00989624_dt, 0.0129339_dt,  0.00689094_dt, 0.0101531_dt,  0.0062774_dt,  0.00654084_dt,
                          0.00714432_dt, 0.0115981_dt,  0.01131_dt,    0.00687872_dt, 0.0107581_dt,  0.0121715_dt,  0.00868416_dt, 0.00942815_dt, 0.010386_dt,   0.0126139_dt,  0.0149499_dt,
                          0.00629211_dt, 0.00770118_dt, 0.0112584_dt,  0.0139976_dt,  0.0142922_dt,  0.0143793_dt,  0.0102176_dt,  0.00598836_dt, 0.00968526_dt, 0.0067653_dt,  0.00580449_dt,
                          0.0144234_dt,  0.0135284_dt,  0.00561726_dt, 0.0101573_dt,  0.00850266_dt, 0.00851925_dt, 0.00735757_dt, 0.0112109_dt,  0.00687889_dt, 0.0111107_dt,  0.011911_dt,
                          0.0132308_dt,  0.0111514_dt,  0.00563919_dt, 0.00668764_dt, 0.011873_dt,   0.0102703_dt,  0.00626242_dt, 0.00693582_dt, 0.0148053_dt,  0.0129555_dt,  0.00743796_dt,
                          0.0081561_dt,  0.00574488_dt, 0.00916711_dt, 0.00634762_dt, 0.00628981_dt, 0.00899851_dt, 0.009971_dt,   0.00753715_dt, 0.0124447_dt,  0.00682327_dt, 0.0145602_dt,
                          0.0130301_dt,  0.00606763_dt, 0.00816753_dt, 0.00946093_dt, 0.00994987_dt, 0.0104145_dt,  0.0112551_dt,  0.00953109_dt, 0.00724741_dt, 0.0117189_dt,  0.00572581_dt,
                          0.0068775_dt,  0.00816163_dt, 0.00725037_dt, 0.00776551_dt, 0.00613984_dt, 0.00831658_dt, 0.0102932_dt,  0.00667838_dt, 0.00901553_dt, 0.0132306_dt,  0.00878643_dt,
                          0.0093793_dt,  0.00885936_dt, 0.0102349_dt,  0.0127118_dt,  0.0148534_dt,  0.0127068_dt,  0.0148696_dt,  0.00892136_dt, 0.00590293_dt, 0.00729778_dt, 0.0130015_dt,
                          0.00921993_dt, 0.00721445_dt, 0.00630569_dt, 0.00579319_dt, 0.00606543_dt, 0.00835776_dt, 0.012337_dt,   0.0122892_dt,  0.00579029_dt, 0.0128108_dt,  0.00634168_dt,
                          0.00843757_dt, 0.00765732_dt, 0.00851723_dt, 0.00850137_dt, 0.0059878_dt,  0.00609019_dt, 0.00867252_dt, 0.00943914_dt, 0.00747274_dt, 0.011322_dt,   0.0059797_dt,
                          0.00906713_dt, 0.0145575_dt,  0.00764807_dt, 0.0147302_dt,  0.01124_dt,    0.00597252_dt, 0.0128681_dt,  0.00885252_dt, 0.00750219_dt, 0.0139884_dt,  0.00626107_dt,
                          0.00989522_dt, 0.00844618_dt, 0.0134027_dt,  0.010783_dt,   0.0119252_dt,  0.0111895_dt,  0.00831584_dt, 0.00844223_dt, 0.00621123_dt, 0.0122966_dt,  0.0139487_dt,
                          0.0132039_dt,  0.00659103_dt, 0.00958979_dt, 0.0065925_dt,  0.00712177_dt, 0.00700796_dt, 0.0111259_dt,  0.00696183_dt, 0.00927656_dt, 0.00957766_dt, 0.012945_dt,
                          0.00642656_dt, 0.0066535_dt,  0.00701512_dt, 0.013308_dt,   0.00783551_dt, 0.0142983_dt,  0.0112369_dt,  0.00999127_dt, 0.00934346_dt, 0.00849568_dt, 0.00998548_dt,
                          0.00836642_dt, 0.00934541_dt, 0.00999611_dt, 0.00634265_dt, 0.00721623_dt, 0.0140433_dt,  0.00625028_dt, 0.00904757_dt, 0.0153792_dt,  0.0112348_dt,  0.00951201_dt,
                          0.00608054_dt, 0.0120142_dt,  0.00656832_dt, 0.00813737_dt, 0.0079299_dt,  0.00870899_dt, 0.00942974_dt, 0.0141654_dt,  0.00998235_dt, 0.0146765_dt,  0.0127337_dt,
                          0.00683661_dt, 0.0117359_dt,  0.00658603_dt, 0.00758738_dt, 0.0108641_dt,  0.0110618_dt,  0.0136469_dt,  0.00798401_dt, 0.00938581_dt, 0.0121316_dt,  0.0057826_dt,
                          0.0134622_dt,  0.00620241_dt, 0.00944316_dt, 0.00861402_dt, 0.00720707_dt, 0.010019_dt,   0.0141804_dt,  0.00810304_dt, 0.00696956_dt, 0.00779498_dt, 0.00571346_dt,
                          0.0117529_dt };

    WorkflowEager work(raul::CompressionMode::NONE, raul::CalculationMode::DETERMINISTIC, raul::AllocationMode::STANDARD, raul::ExecutionTarget::GPU);
    work.getKernelManager().setExecutionPolicy(raul::KernelExecutionPolicy::DefaultParams);

    work.add<DataLayer>("data", DataParams{ { "in" }, DEPTH, HEIGHT, WIDTH });
    work.add<SoftMaxActivation>("softmax", BasicParamsWithDim{ { "in" }, { "out" } });

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

TEST(TestSoftMax, GpuWidthUnit)
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
        0.0943422_dt, 0.115658_dt,  0.162139_dt, 0.19359_dt,   0.143612_dt,  0.125993_dt,  0.164666_dt, 0.115012_dt,  0.169459_dt,  0.104772_dt,  0.109169_dt, 0.119241_dt,  0.193577_dt,  0.188768_dt,
        0.0969919_dt, 0.151693_dt,  0.171621_dt, 0.122449_dt,  0.13294_dt,   0.146446_dt,  0.177859_dt, 0.180401_dt,  0.075927_dt,  0.0929302_dt, 0.135855_dt, 0.168909_dt,  0.172464_dt,  0.173515_dt,
        0.15385_dt,   0.0901687_dt, 0.145834_dt, 0.101867_dt,  0.0874001_dt, 0.217178_dt,  0.203701_dt, 0.0964439_dt, 0.174393_dt,  0.145984_dt,  0.146269_dt, 0.126324_dt,  0.192482_dt,  0.118105_dt,
        0.155169_dt,  0.166346_dt,  0.184778_dt, 0.155738_dt,  0.0787557_dt, 0.093398_dt,  0.165816_dt, 0.153694_dt,  0.093716_dt,  0.103793_dt,  0.221558_dt, 0.193876_dt,  0.111308_dt,  0.122055_dt,
        0.106276_dt,  0.169585_dt,  0.117427_dt, 0.116357_dt,  0.166466_dt,  0.184457_dt,  0.139432_dt, 0.176385_dt,  0.0967094_dt, 0.206369_dt,  0.184681_dt, 0.0859993_dt, 0.115762_dt,  0.134094_dt,
        0.151116_dt,  0.158172_dt,  0.17094_dt,  0.144755_dt,  0.110072_dt,  0.177983_dt,  0.086962_dt, 0.125491_dt,  0.148922_dt,  0.132295_dt,  0.141694_dt, 0.112031_dt,  0.15175_dt,   0.187817_dt,
        0.100906_dt,  0.136218_dt,  0.199905_dt, 0.132757_dt,  0.141714_dt,  0.133859_dt,  0.154641_dt, 0.164525_dt,  0.192244_dt,  0.16446_dt,   0.192452_dt, 0.115466_dt,  0.0763998_dt, 0.094453_dt,
        0.232344_dt,  0.164765_dt,  0.128926_dt, 0.112686_dt,  0.103528_dt,  0.108393_dt,  0.149358_dt, 0.187881_dt,  0.187153_dt,  0.0881808_dt, 0.195097_dt, 0.096578_dt,  0.128496_dt,  0.116614_dt,
        0.155762_dt,  0.155472_dt,  0.109504_dt, 0.111377_dt,  0.158602_dt,  0.172622_dt,  0.136661_dt, 0.151882_dt,  0.0802163_dt, 0.121634_dt,  0.195286_dt, 0.102597_dt,  0.197603_dt,  0.150782_dt,
        0.0914069_dt, 0.19694_dt,   0.135484_dt, 0.114818_dt,  0.214086_dt,  0.0958231_dt, 0.151442_dt, 0.116492_dt,  0.184853_dt,  0.148722_dt,  0.164475_dt, 0.154328_dt,  0.114694_dt,  0.116437_dt,
        0.0907626_dt, 0.179686_dt,  0.203828_dt, 0.192944_dt,  0.0963126_dt, 0.140132_dt,  0.096334_dt, 0.111249_dt,  0.109471_dt,  0.173796_dt,  0.10875_dt,  0.144909_dt,  0.149612_dt,  0.202213_dt,
        0.0962435_dt, 0.0996422_dt, 0.105058_dt, 0.199299_dt,  0.117344_dt,  0.214131_dt,  0.168283_dt, 0.152483_dt,  0.142596_dt,  0.129658_dt,  0.152395_dt, 0.127685_dt,  0.142626_dt,  0.152557_dt,
        0.0912426_dt, 0.10381_dt,   0.202021_dt, 0.0899138_dt, 0.130154_dt,  0.221239_dt,  0.161619_dt, 0.161354_dt,  0.103145_dt,  0.203798_dt,  0.111419_dt, 0.138035_dt,  0.134516_dt,  0.147732_dt,
        0.118523_dt,  0.178046_dt,  0.125469_dt, 0.18447_dt,   0.160052_dt,  0.08593_dt,   0.147509_dt, 0.0981291_dt, 0.113049_dt,  0.16187_dt,   0.164816_dt, 0.203333_dt,  0.118958_dt,  0.139845_dt,
        0.193045_dt,  0.0920165_dt, 0.21422_dt,  0.0986968_dt, 0.150266_dt,  0.137072_dt,  0.114684_dt, 0.155253_dt,  0.219737_dt,  0.125564_dt,  0.107999_dt, 0.12079_dt,   0.0885351_dt, 0.182121_dt
    };

    WorkflowEager work(raul::CompressionMode::NONE, raul::CalculationMode::DETERMINISTIC, raul::AllocationMode::STANDARD, raul::ExecutionTarget::GPU);
    work.getKernelManager().setExecutionPolicy(raul::KernelExecutionPolicy::DefaultParams);

    work.add<DataLayer>("data", DataParams{ { "in" }, DEPTH, HEIGHT, WIDTH });
    work.add<SoftMaxActivation>("softmax", BasicParamsWithDim{ { "in" }, { "out" }, "width" });

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