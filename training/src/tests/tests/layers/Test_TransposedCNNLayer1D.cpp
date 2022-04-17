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

#include <training/base/layers/basic/trainable/TransposedConvolution1DLayer.h>
#include <training/compiler/Workflow.h>

namespace UT
{

TEST(TestTransposedCNN1DLayer, IncorrectSetupUnit)
{
    const size_t KERNEL_SIZE = 3;
    const size_t FILTERS = 2;

    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    EXPECT_THROW(raul::TransposedConvolution1DLayer cnnLayer("cnn1", raul::TransposedConvolution1DParams{ { "in" }, { "cnn1" }, KERNEL_SIZE, FILTERS }, networkParameters), raul::Exception);
}

TEST(TestTransposedCNN1DLayer, ProperSetupUnit)
{
    const size_t KERNEL_SIZE = 3;
    const size_t FILTERS = 2;
    const size_t STRIDE = 3;
    const size_t PADDING = 2;
    const size_t OUTPUT_PADDING = 1;
    const size_t DILATION = 3;

    const size_t outputHeight = 1;
    const size_t outputWidth = 10;

    const auto expectedShape = yato::dims(1, 3, FILTERS, KERNEL_SIZE);

    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    work.tensorNeeded("in", "in", raul::WShape{ raul::BS(), 3u, 1u, 3u }, DEC_FORW_READ_NOMEMOPT);

    raul::TransposedConvolution1DLayer cnnLayer(
        "cnn1", raul::TransposedConvolution1DParams{ { "in" }, { "cnn1" }, KERNEL_SIZE, FILTERS, STRIDE, PADDING, OUTPUT_PADDING, DILATION }, networkParameters);
    TENSORS_CREATE(2);

    ASSERT_NO_THROW(memory_manager["cnn1::Weights"]);
    ASSERT_NO_THROW(memory_manager["cnn1::WeightsGradient"]);
    ASSERT_NO_THROW(memory_manager["cnn1::Biases"]);
    ASSERT_NO_THROW(memory_manager["cnn1::BiasesGradient"]);

    EXPECT_EQ(memory_manager["cnn1::Weights"].getShape(), expectedShape);
    EXPECT_EQ(memory_manager["cnn1::Biases"].size(), FILTERS);
    EXPECT_EQ(memory_manager["cnn1"].getDepth(), static_cast<size_t>(FILTERS));
    EXPECT_EQ(memory_manager["cnn1"].getHeight(), outputHeight);
    EXPECT_EQ(memory_manager["cnn1"].getWidth(), outputWidth);
}

TEST(TestTransposedCNN1DLayer, SimpleUnit)
{
    const size_t KERNEL_SIZE = 2;
    const size_t FILTERS = 1;
    const size_t STRIDE = 1;
    const size_t PADDING = 0;
    const size_t OUTPUT_PADDING = 0;
    const size_t DILATION = 1;
    const size_t GROUPS = 1;
    const bool BIAS = false;

    const raul::dtype EPSILON = TODTYPE(1e-6);

    const raul::Tensor input = { 0.63434184_dt, 0.36441028_dt, 0.71042877_dt, 0.94641107_dt, 0.78902978_dt, 0.28141373_dt };

    const raul::Tensor realOutput = { 0.04903505_dt, -0.11329067_dt, -0.02634779_dt, -0.15842740_dt, 0.07315820_dt, -0.15005954_dt, -0.15420216_dt, -0.06275597_dt };

    const raul::Tensor realInputNabla = { -0.14570186_dt, -0.14570186_dt, -0.14570186_dt, -0.14570186_dt, -0.14570186_dt, -0.14570186_dt };

    const raul::Tensor realWeightsGrad = { 3.72603536_dt, 3.72603536_dt };

    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    work.tensorNeeded("in", "in", raul::WShape{ raul::BS(), 1u, 1u, 3u }, DEC_FORW_READ_NOMEMOPT);

    raul::TransposedConvolution1DLayer cnnLayer(
        "cnn1", raul::TransposedConvolution1DParams{ { "in" }, { "cnn1" }, KERNEL_SIZE, FILTERS, STRIDE, PADDING, OUTPUT_PADDING, DILATION, GROUPS, BIAS }, networkParameters);
    TENSORS_CREATE(2);
    memory_manager["in"] = TORANGE(input);
    memory_manager["cnn1::Weights"] = TORANGE((raul::Tensor{ 0.07730067_dt, -0.22300252_dt }));

    ASSERT_THROW(memory_manager["cnn1::Biases"], raul::Exception);
    ASSERT_THROW(memory_manager["cnn1::BiasesGradient"], raul::Exception);

    // Forward
    ASSERT_NO_THROW(cnnLayer.forwardCompute(raul::NetworkMode::Train));

    const auto& output = memory_manager["cnn1"];
    EXPECT_EQ(output.size(), realOutput.size());
    for (size_t i = 0; i < output.size(); ++i)
    {
        CHECK_NEAR(output[i], realOutput[i], EPSILON);
    }

    // Backward
    memory_manager[raul::Name("cnn1").grad()] = TORANGE(*memory_manager.createTensor("gradient", output.getShape(), 1.0_dt));
    cnnLayer.backwardCompute();
    const auto inputNabla = memory_manager[raul::Name("in").grad()];
    EXPECT_EQ(inputNabla.size(), realInputNabla.size());
    for (size_t i = 0; i < inputNabla.size(); ++i)
    {
        CHECK_NEAR(inputNabla[i], realInputNabla[i], EPSILON);
    }
    const auto weightsGrad = memory_manager[raul::Name("cnn1::Weights").grad()];
    EXPECT_EQ(weightsGrad.size(), realWeightsGrad.size());
    for (size_t i = 0; i < weightsGrad.size(); ++i)
    {
        CHECK_NEAR(weightsGrad[i], realWeightsGrad[i], EPSILON);
    }
}

TEST(TestTransposedCNN1DLayer, NonTrivialUnit)
{
    const size_t KERNEL_SIZE = 3;
    const size_t FILTERS = 6;
    const size_t STRIDE = 2;
    const size_t PADDING = 2;
    const size_t OUTPUT_PADDING = 1;
    const size_t DILATION = 3;
    const size_t GROUPS = 2;
    const bool BIAS = true;

    const raul::dtype EPSILON = TODTYPE(1e-6);

    const raul::Tensor input = { 0.24767512_dt, 0.65243822_dt, 0.60570377_dt, 0.37252063_dt, 0.79803473_dt, 0.83990461_dt, 0.13741332_dt, 0.23306590_dt,
                                 0.95783097_dt, 0.33128375_dt, 0.32274181_dt, 0.01620269_dt, 0.21366489_dt, 0.62490183_dt, 0.43400341_dt, 0.13705701_dt,
                                 0.51172835_dt, 0.15845925_dt, 0.07580167_dt, 0.22466868_dt, 0.06239396_dt, 0.18163097_dt, 0.99980444_dt, 0.59443748_dt };
    const raul::Tensor deltas = { 0.65407985_dt, 0.03365785_dt, 0.17161310_dt, 0.33357209_dt, 0.57818556_dt, 0.06003934_dt, 0.28456348_dt, 0.20066571_dt, 0.50138563_dt, 0.31394839_dt, 0.46535212_dt,
                                  0.16118515_dt, 0.15680242_dt, 0.20829910_dt, 0.32885128_dt, 0.10535955_dt, 0.91923493_dt, 0.40076798_dt, 0.93019837_dt, 0.65579104_dt, 0.07660151_dt, 0.84601760_dt,
                                  0.36242759_dt, 0.30833697_dt, 0.08496475_dt, 0.00291967_dt, 0.64305532_dt, 0.39077806_dt, 0.69466156_dt, 0.08966827_dt, 0.87121457_dt, 0.13297313_dt, 0.41366333_dt,
                                  0.60443485_dt, 0.75812590_dt, 0.90365517_dt, 0.95547962_dt, 0.10353893_dt, 0.62583363_dt, 0.28493702_dt, 0.44520760_dt, 0.12575495_dt, 0.95542932_dt, 0.13302475_dt,
                                  0.76722562_dt, 0.67571980_dt, 0.66247797_dt, 0.22967690_dt, 0.95447576_dt, 0.60987520_dt, 0.56432003_dt, 0.05937260_dt, 0.70989424_dt, 0.42498970_dt, 0.27093786_dt,
                                  0.92947328_dt, 0.61147439_dt, 0.22336179_dt, 0.24693054_dt, 0.47612214_dt, 0.77918065_dt, 0.37223309_dt, 0.21471256_dt, 0.32877856_dt, 0.12646258_dt, 0.67831624_dt,
                                  0.88702011_dt, 0.02927983_dt, 0.61612535_dt, 0.75829589_dt, 0.59066468_dt, 0.32193768_dt, 0.76097107_dt, 0.76275659_dt, 0.68696362_dt, 0.41213930_dt, 0.36759937_dt,
                                  0.55349046_dt, 0.41167295_dt, 0.35099947_dt, 0.81960344_dt, 0.92969978_dt, 0.45050132_dt, 0.38805157_dt, 0.50729614_dt, 0.47014588_dt, 0.62020564_dt, 0.64011681_dt,
                                  0.04587162_dt, 0.31548113_dt, 0.92106473_dt, 0.69477749_dt, 0.47513121_dt, 0.19854712_dt, 0.19409746_dt, 0.05211657_dt };

    const raul::Tensor realOutput = { 0.41258818_dt,  0.23090252_dt,  0.40937650_dt,  0.29990131_dt,  0.21662532_dt,  0.30152792_dt,  0.28267553_dt,  0.17685941_dt,  -0.05910678_dt, -0.30468705_dt,
                                      -0.03868926_dt, -0.53902435_dt, -0.15881371_dt, -0.53351986_dt, -0.20201123_dt, -0.13473484_dt, -0.14606732_dt, 0.30833948_dt,  -0.14364448_dt, 0.46278352_dt,
                                      0.18332419_dt,  0.45430458_dt,  0.17333917_dt,  0.20230797_dt,  -0.14407888_dt, -0.19817400_dt, 0.01427421_dt,  -0.20199262_dt, -0.12750518_dt, -0.15097588_dt,
                                      -0.09883307_dt, -0.07910022_dt, 0.07226747_dt,  0.20809202_dt,  -0.00218725_dt, 0.18792307_dt,  0.20290421_dt,  0.00245337_dt,  0.18286784_dt,  0.19068196_dt,
                                      -0.09972149_dt, -0.30075273_dt, 0.03266975_dt,  -0.28865686_dt, -0.29019397_dt, -0.15537907_dt, -0.25757235_dt, -0.25898933_dt, 0.36775893_dt,  0.20405143_dt,
                                      0.28224969_dt,  0.26807645_dt,  0.21223350_dt,  0.21944910_dt,  0.27967441_dt,  0.17685941_dt,  -0.12820084_dt, -0.23769802_dt, -0.18157344_dt, -0.46213004_dt,
                                      -0.15937553_dt, -0.31547189_dt, -0.20447388_dt, -0.13473484_dt, -0.07388842_dt, 0.27483535_dt,  0.05529284_dt,  0.42582422_dt,  0.20708869_dt,  0.33730060_dt,
                                      0.20356168_dt,  0.20230797_dt,  -0.33119115_dt, -0.14441624_dt, -0.23625243_dt, -0.42526352_dt, -0.10551096_dt, -0.27996942_dt, -0.27772439_dt, -0.07910022_dt,
                                      -0.07789105_dt, 0.20013525_dt,  0.04506442_dt,  0.28100497_dt,  0.19729096_dt,  0.25853509_dt,  0.26535058_dt,  0.19068196_dt,  0.08529019_dt,  -0.28183529_dt,
                                      -0.07563852_dt, -0.40614045_dt, -0.27594924_dt, -0.35438591_dt, -0.41411963_dt, -0.25898933_dt };
    const raul::Tensor realInputNabla = { 0.09306403_dt, -0.02433971_dt, -0.13310526_dt, -0.03846926_dt, 0.04001012_dt, -0.02258831_dt, 0.14178184_dt,  0.19735041_dt,
                                          0.25060210_dt, -0.18246308_dt, -0.32595548_dt, -0.21425098_dt, 0.22717594_dt, -0.04353829_dt, -0.05844389_dt, 0.02262217_dt,
                                          0.09416393_dt, -0.02489161_dt, -0.04957995_dt, -0.04627144_dt, 0.14746195_dt, -0.32425511_dt, -0.66132170_dt, -0.18858255_dt };

    const raul::Tensor realWeightsGrad = { 1.37206388_dt, 0.61419535_dt, 0.64985132_dt, 1.09826875_dt, 0.81589270_dt, 0.55404902_dt, 1.62716508_dt, 1.53189170_dt, 0.75618565_dt,
                                           1.24397111_dt, 0.51048154_dt, 0.67842019_dt, 1.14301169_dt, 0.75377727_dt, 0.53751355_dt, 1.72013032_dt, 1.61132288_dt, 0.70446956_dt,
                                           0.84956944_dt, 0.36231279_dt, 0.41886079_dt, 1.03481507_dt, 0.57983154_dt, 0.45495081_dt, 1.08667731_dt, 0.88790613_dt, 0.33945137_dt,
                                           1.20702004_dt, 1.00815487_dt, 0.98966736_dt, 1.23302817_dt, 1.32987463_dt, 1.23074257_dt, 0.75254571_dt, 0.96550834_dt, 0.74833679_dt };

    const raul::Tensor realBiasGrad = { 6.83971596_dt, 5.49397755_dt, 8.50747871_dt, 7.21682835_dt, 9.47528934_dt, 6.89160442_dt };

    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    work.tensorNeeded("in", "in", raul::WShape{ raul::BS(), 4u, 1u, 3u }, DEC_FORW_READ_NOMEMOPT);

    raul::TransposedConvolution1DLayer cnnLayer(
        "cnn1", raul::TransposedConvolution1DParams{ { "in" }, { "cnn1" }, KERNEL_SIZE, FILTERS, STRIDE, PADDING, OUTPUT_PADDING, DILATION, GROUPS, BIAS }, networkParameters);
    TENSORS_CREATE(2);
    memory_manager["in"] = TORANGE(input);
    memory_manager["cnn1::Weights"] = TORANGE((raul::Tensor{
        0.19242159_dt, 0.05964208_dt,  0.16927835_dt, -0.20316836_dt, -0.32996950_dt, -0.12878685_dt, -0.25567430_dt, 0.27351299_dt,  0.09601045_dt, 0.13807118_dt,  0.10542038_dt,  -0.00579867_dt,
        0.26086941_dt, -0.23683786_dt, 0.02098790_dt, -0.22751340_dt, 0.10278401_dt,  -0.11479411_dt, 0.10213876_dt,  -0.06944716_dt, 0.27646396_dt, -0.19756731_dt, -0.19879934_dt, -0.19881134_dt,
        0.29981425_dt, 0.11108372_dt,  0.32075027_dt, -0.27509210_dt, -0.33062539_dt, -0.26078793_dt, -0.22422969_dt, 0.13501337_dt,  0.11935863_dt, 0.27697483_dt,  -0.17214179_dt, -0.22723727_dt }));
    memory_manager["cnn1::Biases"] = TORANGE((raul::Tensor{ 0.17685941_dt, -0.13473484_dt, 0.20230797_dt, -0.07910022_dt, 0.19068196_dt, -0.25898933_dt }));

    // Forward
    ASSERT_NO_THROW(cnnLayer.forwardCompute(raul::NetworkMode::Train));

    const auto output = memory_manager["cnn1"];
    EXPECT_EQ(output.size(), realOutput.size());
    for (size_t i = 0; i < output.size(); ++i)
    {
        CHECK_NEAR(output[i], realOutput[i], EPSILON);
    }

    // Backward
    memory_manager[raul::Name("cnn1").grad()] = TORANGE(deltas);

    ASSERT_NO_THROW(cnnLayer.backwardCompute());
    const auto inputNabla = memory_manager[raul::Name("in").grad()];
    EXPECT_EQ(inputNabla.size(), realInputNabla.size());
    for (size_t i = 0; i < inputNabla.size(); ++i)
    {
        CHECK_NEAR(inputNabla[i], realInputNabla[i], EPSILON);
    }

    const auto weightsGrad = memory_manager[raul::Name("cnn1::Weights").grad()];
    EXPECT_EQ(weightsGrad.size(), realWeightsGrad.size());
    for (size_t i = 0; i < weightsGrad.size(); ++i)
    {
        CHECK_NEAR(weightsGrad[i], realWeightsGrad[i], EPSILON);
    }

    const auto biasGrad = memory_manager[raul::Name("cnn1::Biases").grad()];
    EXPECT_EQ(biasGrad.size(), realBiasGrad.size());
    for (size_t i = 0; i < biasGrad.size(); ++i)
    {
        CHECK_NEAR(biasGrad[i], realBiasGrad[i], EPSILON);
    }
}

}