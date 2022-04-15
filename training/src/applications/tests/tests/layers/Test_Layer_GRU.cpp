// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <algorithm>
#include <cstdio>
#include <gtest/gtest.h>
#include <training/api/API.h>
#include <training/network/Layers.h>
#include <tests/tools/TestTools.h>

#include <training/layers/composite/rnn/GRULayer.h>

namespace UT
{

// see gru.py
TEST(TestGRU, SimpleSeq1Unit)
{
    PROFILE_TEST
    
    // Test parameters
    const auto eps_rel = 1e-4_dt;
    const size_t input_size = 4U;
    const size_t hidden_size = 3U;
    const size_t sequence_length = 1U;
    const size_t batch_size = 2U;

    const raul::Tensor input_init{ -0.13295190_dt, -0.04689599_dt, -0.28016463_dt, 0.54008639_dt, -0.14635307_dt,
        -0.15740128_dt, -1.01898205_dt, 0.02789201_dt };
    
    const raul::Tensor output_golden{ 0.10680354_dt, 0.10680354_dt, 0.10680354_dt, 0.11834978_dt, 0.11834978_dt, 0.11834978_dt };
    const raul::Tensor hidden_golden{ 0.10680354_dt, 0.10680354_dt, 0.10680354_dt, 0.11834978_dt, 0.11834978_dt, 0.11834978_dt };

    const raul::Tensor inputs_grad_golden{ -0.25737935_dt, -0.25737935_dt, -0.25737935_dt, -0.25737935_dt, 0.81872380_dt,
        0.81872380_dt, 0.81872380_dt, 0.81872380_dt };
    const raul::Tensor ih_weights_grad_golden{ -0.00944828_dt, -0.01008253_dt, -0.06525287_dt, 0.00222383_dt, -0.00944828_dt,
        -0.01008253_dt, -0.06525287_dt, 0.00222383_dt, -0.00944828_dt, -0.01008253_dt, -0.06525287_dt, 0.00222383_dt, 0.02421623_dt,
        0.01692100_dt, 0.10731841_dt, -0.05348697_dt, 0.02421623_dt, 0.01692100_dt, 0.10731841_dt, -0.05348697_dt, 0.02421623_dt,
        0.01692100_dt, 0.10731841_dt, -0.05348697_dt, -0.04330252_dt, -0.04577118_dt, -0.29611763_dt, 0.01253940_dt, -0.04330252_dt,
        -0.04577118_dt, -0.29611763_dt, 0.01253940_dt, -0.04330252_dt, -0.04577118_dt, -0.29611763_dt, 0.01253940_dt };
    const raul::Tensor hh_weights_grad_golden{ 0._dt, 0._dt, 0._dt, 0._dt, 0._dt, 0._dt, 0._dt, 0._dt, 0._dt, 0._dt, 0._dt, 0._dt,
        0._dt, 0._dt, 0._dt, 0._dt, 0._dt, 0._dt, 0._dt, 0._dt, 0._dt, 0._dt, 0._dt, 0._dt, 0._dt, 0._dt, 0._dt };
    const raul::Tensor ih_biases_grad_golden{ 0.06463338_dt, 0.06463338_dt, 0.06463338_dt, -0.17415819_dt, -0.17415819_dt, -0.17415819_dt,
        0.29663962_dt, 0.29663962_dt, 0.29663962_dt };
    const raul::Tensor hh_biases_grad_golden{ 0.06463338_dt, 0.06463338_dt, 0.06463338_dt, -0.17415819_dt, -0.17415819_dt, -0.17415819_dt,
        0.20037875_dt, 0.20037875_dt, 0.20037875_dt };

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    work.add<raul::DataLayer>("data1", raul::DataParams{ { "in" }, sequence_length, 1, input_size });

    // Network
    const auto params = raul::GRUParams{ { "in" }, { "out" }, hidden_size };
    raul::GRULayer("gru", params, networkParameters);
    TENSORS_CREATE(batch_size)

    memory_manager["in"] = TORANGE(input_init);

    for (auto& [param, grad] : work.getTrainableParameters())
    {
        param = 1.0_dt;
    }

    // Apply
    ASSERT_NO_THROW(work.forwardPassTraining());

    // Checks
    const auto& outputTensor = memory_manager["out"];
    const auto& hiddenTensor = memory_manager["gru::hidden_state[" + Conversions::toString(sequence_length - 1) + "]"];

    EXPECT_EQ(outputTensor.size(), batch_size * hidden_size * sequence_length);
    EXPECT_EQ(hiddenTensor.size(), batch_size * hidden_size);

    for (size_t i = 0; i < outputTensor.size(); ++i)
    {
        const auto val = outputTensor[i];
        const auto golden_val = output_golden[i];
        ASSERT_TRUE(tools::expect_near_relative(val, golden_val, eps_rel)) << "at " << i << ", expected: " << golden_val << ", got: " << val;
    }

    for (size_t i = 0; i < hiddenTensor.size(); ++i)
    {
        const auto val = hiddenTensor[i];
        const auto golden_val = hidden_golden[i];
        ASSERT_TRUE(tools::expect_near_relative(val, golden_val, eps_rel)) << "at " << i << ", expected: " << golden_val << ", got: " << val;
    }

    // Apply
    memory_manager[raul::Name("out").grad()] = 1.0_dt;
    ASSERT_NO_THROW(work.backwardPassTraining());

    // Checks
    const auto& inputs_grad = memory_manager[raul::Name("in").grad()];
    const auto& inputs = memory_manager["in"];

    EXPECT_EQ(inputs.size(), inputs_grad.size());

    for (size_t i = 0; i < inputs_grad.size(); ++i)
    {
        const auto val = inputs_grad[i];
        const auto golden_val = inputs_grad_golden[i];
        ASSERT_TRUE(tools::expect_near_relative(val, golden_val, eps_rel)) << "at " << i << ", expected: " << golden_val << ", got: " << val;
    }

    const auto paramGrad = work.getTrainableParameters();
    const auto gradBiasesIH = paramGrad[0].Gradient;
    const auto gradWeightsIH = paramGrad[1].Gradient;
    const auto gradBiasesHH = paramGrad[2].Gradient;
    const auto gradWeightsHH = paramGrad[3].Gradient;

    EXPECT_EQ(ih_weights_grad_golden.size(), gradWeightsIH.size());

    for (size_t i = 0; i < gradWeightsIH.size(); ++i)
    {
        const auto val = gradWeightsIH[i];
        const auto golden_val = ih_weights_grad_golden[i];
        ASSERT_TRUE(tools::expect_near_relative(val, golden_val, eps_rel)) << "at " << i << ", expected: " << golden_val << ", got: " << val;
    }

    EXPECT_EQ(hh_weights_grad_golden.size(), gradWeightsHH.size());

    for (size_t i = 0; i < gradWeightsHH.size(); ++i)
    {
        const auto val = gradWeightsHH[i];
        const auto golden_val = hh_weights_grad_golden[i];
        ASSERT_TRUE(tools::expect_near_relative(val, golden_val, eps_rel)) << "at " << i << ", expected: " << golden_val << ", got: " << val;
    }

    EXPECT_EQ(ih_biases_grad_golden.size(), gradBiasesIH.size());

    for (size_t i = 0; i < gradBiasesIH.size(); ++i)
    {
        const auto val = gradBiasesIH[i];
        const auto golden_val = ih_biases_grad_golden[i];
        ASSERT_TRUE(tools::expect_near_relative(val, golden_val, eps_rel)) << "at " << i << ", expected: " << golden_val << ", got: " << val;
    }

    EXPECT_EQ(hh_biases_grad_golden.size(), gradBiasesHH.size());

    for (size_t i = 0; i < gradBiasesHH.size(); ++i)
    {
        const auto val = gradBiasesHH[i];
        const auto golden_val = hh_biases_grad_golden[i];
        ASSERT_TRUE(tools::expect_near_relative(val, golden_val, eps_rel)) << "at " << i << ", expected: " << golden_val << ", got: " << val;
    }
}

TEST(TestGRU, SimpleSeq3Unit)
{
    PROFILE_TEST
    
    // Test parameters
    const auto eps_rel = 1e-4_dt;
    const size_t input_size = 5U;
    const size_t hidden_size = 4U;
    const size_t sequence_length = 3U;
    const size_t batch_size = 2U;

    const raul::Tensor input_init{ 0.63761783_dt, -0.28129023_dt, -1.32987511_dt, -0.65379959_dt, 1.71982408_dt, -0.96095538_dt,
        -0.63750249_dt, 0.07472499_dt, 0.83877468_dt, 1.15289509_dt, -1.76109815_dt, -1.10703886_dt, -1.71736121_dt, 1.53456104_dt,
        -0.36151406_dt, 0.58511323_dt, -1.15600657_dt, -0.14336488_dt, -0.19474059_dt, 1.49027479_dt, -0.70052689_dt, 0.18056405_dt,
        -0.48284835_dt, -0.36609861_dt, -1.32705247_dt, 1.69527960_dt, 2.06549954_dt, 0.25783238_dt, -0.56502467_dt, 0.92781103_dt };
    
    const raul::Tensor output_golden{ 0.10574234_dt, 0.10574234_dt, 0.10574234_dt, 0.10574234_dt, 0.15240932_dt, 0.15240932_dt,
        0.15240932_dt, 0.15240932_dt, -0.61403465_dt, -0.61403465_dt, -0.61403465_dt, -0.61403465_dt, 0.06943178_dt, 0.06943178_dt,
        0.06943178_dt, 0.06943178_dt, -0.47310147_dt, -0.47310147_dt, -0.47310147_dt, -0.47310147_dt, -0.45674217_dt,
        -0.45674217_dt,-0.45674217_dt, -0.45674217_dt };
    const raul::Tensor hidden_golden{ -0.61403465_dt, -0.61403465_dt, -0.61403465_dt, -0.61403465_dt, -0.45674217_dt, -0.45674217_dt, -0.45674217_dt, -0.45674217_dt };

    const raul::Tensor inputs_grad_golden{ -0.97712630_dt, -0.97712630_dt, -0.97712630_dt, -0.97712630_dt, -0.97712630_dt,
        -0.41819614_dt, -0.41819614_dt, -0.41819614_dt, -0.41819614_dt, -0.41819614_dt, 1.25780964_dt, 1.25780964_dt, 1.25780964_dt,
        1.25780964_dt, 1.25780964_dt, -1.11048031_dt, -1.11048031_dt, -1.11048031_dt, -1.11048031_dt, -1.11048031_dt, 3.54119205_dt,
        3.54119205_dt, 3.54119205_dt, 3.54119205_dt, 3.54119205_dt, -0.06468894_dt, -0.06468894_dt, -0.06468894_dt, -0.06468894_dt,
        -0.06468894_dt };
    const raul::Tensor ih_weights_grad_golden{ -0.11048512_dt, -0.00331075_dt, -0.09028359_dt, -0.01148807_dt, -0.14872201_dt,
        -0.11048512_dt, -0.00331075_dt, -0.09028359_dt, -0.01148807_dt, -0.14872201_dt, -0.11048512_dt, -0.00331075_dt, -0.09028359_dt,
        -0.01148807_dt, -0.14872201_dt, -0.11048512_dt, -0.00331075_dt, -0.09028359_dt, -0.01148807_dt, -0.14872201_dt, -0.97238457_dt,
        0.25348625_dt, -0.22005269_dt, 0.36400220_dt, -1.66352570_dt, -0.97238457_dt, 0.25348625_dt, -0.22005269_dt, 0.36400220_dt,
        -1.66352570_dt, -0.97238457_dt, 0.25348625_dt, -0.22005269_dt, 0.36400220_dt, -1.66352570_dt, -0.97238457_dt, 0.25348625_dt,
        -0.22005269_dt, 0.36400220_dt, -1.66352570_dt, -0.33623388_dt, -0.01554246_dt, -0.30447406_dt, -0.05885433_dt, -0.44565848_dt,
        -0.33623388_dt, -0.01554246_dt, -0.30447406_dt, -0.05885433_dt, -0.44565848_dt, -0.33623388_dt, -0.01554246_dt, -0.30447406_dt,
        -0.05885433_dt, -0.44565848_dt, -0.33623388_dt, -0.01554246_dt, -0.30447406_dt, -0.05885433_dt, -0.44565848_dt };
    const raul::Tensor hh_weights_grad_golden{ 0.01068671_dt, 0.01068671_dt, 0.01068671_dt, 0.01068671_dt, 0.01068671_dt,
        0.01068671_dt, 0.01068671_dt, 0.01068671_dt, 0.01068671_dt, 0.01068671_dt, 0.01068671_dt, 0.01068671_dt, 0.01068671_dt,
        0.01068671_dt, 0.01068671_dt, 0.01068671_dt, 0.06131050_dt, 0.06131050_dt, 0.06131050_dt, 0.06131050_dt, 0.06131050_dt,
        0.06131050_dt, 0.06131050_dt, 0.06131050_dt, 0.06131050_dt, 0.06131050_dt, 0.06131050_dt, 0.06131050_dt, 0.06131050_dt,
        0.06131050_dt, 0.06131050_dt, 0.06131050_dt, 0.01282894_dt, 0.01282894_dt, 0.01282894_dt, 0.01282894_dt, 0.01282894_dt,
        0.01282894_dt, 0.01282894_dt, 0.01282894_dt, 0.01282894_dt, 0.01282894_dt, 0.01282894_dt, 0.01282894_dt, 0.01282894_dt,
        0.01282894_dt, 0.01282894_dt, 0.01282894_dt };
    const raul::Tensor ih_biases_grad_golden{ 0.13298517_dt, 0.13298517_dt, 0.13298517_dt, 0.13298517_dt, -0.02688873_dt,
        -0.02688873_dt, -0.02688873_dt, -0.02688873_dt, 0.45103103_dt, 0.45103103_dt, 0.45103103_dt, 0.45103103_dt };
    const raul::Tensor hh_biases_grad_golden{ 0.13298517_dt, 0.13298517_dt, 0.13298517_dt, 0.13298517_dt, -0.02688867_dt,
        -0.02688867_dt, -0.02688867_dt, -0.02688867_dt, 0.19064982_dt, 0.19064982_dt, 0.19064982_dt, 0.19064982_dt };

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    work.add<raul::DataLayer>("data1", raul::DataParams{ { "in" }, sequence_length, 1, input_size });

    // Network
    const auto params = raul::GRUParams{ { "in" }, { "out" }, hidden_size };
    raul::GRULayer("gru", params, networkParameters);
    TENSORS_CREATE(batch_size)

    memory_manager["in"] = TORANGE(input_init);

    for (auto& [param, grad] : work.getTrainableParameters())
    {
        param = 1.0_dt;
    }

    // Apply
    ASSERT_NO_THROW(work.forwardPassTraining());

    // Checks
    const auto& outputTensor = memory_manager["out"];
    const auto& hiddenTensor = memory_manager["gru::hidden_state[" + Conversions::toString(sequence_length - 1) + "]"];

    EXPECT_EQ(outputTensor.size(), batch_size * hidden_size * sequence_length);
    EXPECT_EQ(hiddenTensor.size(), batch_size * hidden_size);

    for (size_t i = 0; i < outputTensor.size(); ++i)
    {
        const auto val = outputTensor[i];
        const auto golden_val = output_golden[i];
        ASSERT_TRUE(tools::expect_near_relative(val, golden_val, eps_rel)) << "at " << i << ", expected: " << golden_val << ", got: " << val;
    }

    for (size_t i = 0; i < hiddenTensor.size(); ++i)
    {
        const auto val = hiddenTensor[i];
        const auto golden_val = hidden_golden[i];
        ASSERT_TRUE(tools::expect_near_relative(val, golden_val, eps_rel)) << "at " << i << ", expected: " << golden_val << ", got: " << val;
    }

    // Apply
    memory_manager[raul::Name("out").grad()] = 1.0_dt;
    ASSERT_NO_THROW(work.backwardPassTraining());

    // Checks
    const auto& inputs_grad = memory_manager[raul::Name("in").grad()];
    const auto& inputs = memory_manager["in"];

    EXPECT_EQ(inputs.size(), inputs_grad.size());

    for (size_t i = 0; i < inputs_grad.size(); ++i)
    {
        const auto val = inputs_grad[i];
        const auto golden_val = inputs_grad_golden[i];
        ASSERT_TRUE(tools::expect_near_relative(val, golden_val, eps_rel)) << "at " << i << ", expected: " << golden_val << ", got: " << val;
    }

    const auto paramGrad = work.getTrainableParameters();
    const auto gradBiasesIH = paramGrad[0].Gradient;
    const auto gradWeightsIH = paramGrad[1].Gradient;
    const auto gradBiasesHH = paramGrad[2].Gradient;
    const auto gradWeightsHH = paramGrad[3].Gradient;

    EXPECT_EQ(ih_weights_grad_golden.size(), gradWeightsIH.size());

    for (size_t i = 0; i < gradWeightsIH.size(); ++i)
    {
        const auto val = gradWeightsIH[i];
        const auto golden_val = ih_weights_grad_golden[i];
        ASSERT_TRUE(tools::expect_near_relative(val, golden_val, eps_rel)) << "at " << i << ", expected: " << golden_val << ", got: " << val;
    }

    EXPECT_EQ(hh_weights_grad_golden.size(), gradWeightsHH.size());

    for (size_t i = 0; i < gradWeightsHH.size(); ++i)
    {
        const auto val = gradWeightsHH[i];
        const auto golden_val = hh_weights_grad_golden[i];
        ASSERT_TRUE(tools::expect_near_relative(val, golden_val, eps_rel)) << "at " << i << ", expected: " << golden_val << ", got: " << val;
    }

    EXPECT_EQ(ih_biases_grad_golden.size(), gradBiasesIH.size());

    for (size_t i = 0; i < gradBiasesIH.size(); ++i)
    {
        const auto val = gradBiasesIH[i];
        const auto golden_val = ih_biases_grad_golden[i];
        ASSERT_TRUE(tools::expect_near_relative(val, golden_val, eps_rel)) << "at " << i << ", expected: " << golden_val << ", got: " << val;
    }

    EXPECT_EQ(hh_biases_grad_golden.size(), gradBiasesHH.size());

    for (size_t i = 0; i < gradBiasesHH.size(); ++i)
    {
        const auto val = gradBiasesHH[i];
        const auto golden_val = hh_biases_grad_golden[i];
        ASSERT_TRUE(tools::expect_near_relative(val, golden_val, eps_rel)) << "at " << i << ", expected: " << golden_val << ", got: " << val;
    }
}

TEST(TestGRU, SimpleSeq7FusionOffAndOnUnit)
{
    PROFILE_TEST
    
    // Test parameters
    const auto eps_rel = 1e-3_dt;
    const size_t input_size = 4U;
    const size_t hidden_size = 5U;
    const size_t sequence_length = 7U;
    const size_t batch_size = 2U;

    const raul::Tensor input_init{ -0.19567661_dt, 0.16095491_dt, 0.86689448_dt, 0.22917956_dt, 0.23287529_dt, 0.03640829_dt,
        0.40704045_dt,1.25164700_dt, -0.02822716_dt, 0.05133806_dt, -0.51514918_dt, -1.88835800_dt, -0.00439325_dt, -2.22326303_dt,
        0.74002379_dt, -0.71609747_dt, -0.65955150_dt, 0.18859509_dt, -0.41376933_dt, 0.47538298_dt, 0.10268653_dt, -0.48300776_dt,
        -0.26079562_dt, 0.47629976_dt, -1.03042293_dt, -1.62371719_dt, -0.19360958_dt, 0.80913794_dt, 0.08752438_dt, 0.70364809_dt,
        -1.34323740_dt, -0.02036794_dt, 1.06088448_dt, -0.01595338_dt, 1.27565277_dt, 0.00945723_dt, -0.36944997_dt, 1.21819782_dt,
        0.25504440_dt, 0.27399307_dt, 0.36422995_dt, 0.08813406_dt, -1.30691111_dt, -0.70636863_dt, -0.16421619_dt, -0.97146821_dt,
        -1.03084397_dt, 0.64727920_dt, -0.19061494_dt, 0.71665096_dt, -2.00018930_dt, -2.40965796_dt, 0.21942286_dt, -1.69886053_dt,
        1.30942404_dt, -1.66129482_dt };

    const raul::Tensor ih_weights{ -0.39872482_dt, 0.39215106_dt, -0.29042551_dt, -0.05087572_dt, 0.12812382_dt, 0.01424748_dt,
        -0.30092186_dt, -0.36149246_dt, 0.35646611_dt, 0.07282370_dt, 0.37101936_dt, -0.14992416_dt, 0.13172919_dt, -0.10226712_dt,
        -0.01988810_dt, -0.27237284_dt, 0.15124804_dt, 0.14139372_dt, -0.00922537_dt, -0.10057929_dt, -0.27567577_dt, 0.30926824_dt,
        -0.33291659_dt, 0.18320799_dt, -0.15037715_dt, -0.21576625_dt, 0.08033973_dt, -0.23230822_dt, 0.10304016_dt, 0.08782417_dt,
        -0.33205137_dt, 0.07446039_dt, 0.19048131_dt, 0.17702103_dt, -0.05629465_dt, -0.36662018_dt, -0.06893981_dt, 0.15532070_dt,
        -0.16317800_dt, 0.16975385_dt, 0.29782754_dt, -0.23351328_dt, 0.00441036_dt, 0.18492240_dt, 0.03505164_dt, 0.03735644_dt,
        0.05584151_dt, -0.35157594_dt, 0.03514573_dt, 0.30968189_dt, 0.40299034_dt, 0.26285022_dt, 0.05992800_dt, 0.20885509_dt,
        -0.21756031_dt, -0.37060070_dt, -0.38456839_dt, 0.44614464_dt, 0.28388643_dt, -0.30912912_dt };
    const raul::Tensor hh_weights{ 0.17496902_dt, 0.33772123_dt, 0.44704133_dt, 0.39104128_dt, 0.34648043_dt, -0.10252786_dt,
        -0.15694588_dt, 0.36718422_dt, 0.25060940_dt, -0.26912373_dt, 0.40205270_dt, 0.21607506_dt, 0.24379200_dt, -0.28030288_dt,
        0.12830549_dt, -0.15677628_dt, 0.34943330_dt, -0.08047187_dt, 0.17410582_dt, 0.07943487_dt, 0.19027519_dt, -0.15197548_dt,
        0.21804851_dt, -0.31236571_dt, 0.10100543_dt, -0.30258107_dt, -0.44119301_dt, -0.35913745_dt, 0.35306174_dt, 0.24195850_dt,
        0.41958284_dt, 0.35827231_dt, -0.39938205_dt, -0.30519247_dt, -0.07226193_dt, -0.29043496_dt, 0.31054354_dt, -0.33809698_dt,
        -0.21819615_dt, -0.43204921_dt, -0.25390354_dt, 0.36782312_dt, 0.36616063_dt, 0.32013237_dt, 0.34530580_dt, 0.39765626_dt,
        -0.11451486_dt, 0.19677490_dt, 0.39843619_dt, 0.14794666_dt, 0.44706887_dt, 0.23195308_dt, 0.27800959_dt, -0.15652126_dt,
        0.21458536_dt, 0.05138776_dt, -0.10680234_dt, -0.25209871_dt, -0.25093895_dt, -0.34412229_dt, 0.30022806_dt, 0.31793809_dt,
        -0.05089858_dt, -0.25879624_dt, 0.34565383_dt, 0.28598815_dt, 0.03324318_dt, -0.21114533_dt, 0.41101068_dt, 0.18288380_dt,
        -0.33950013_dt, 0.42802048_dt, 0.33960229_dt, -0.16300526_dt, 0.25140315_dt };
    const raul::Tensor ih_biases{ -0.25410187_dt, -0.07008478_dt, 0.37972957_dt, 0.01847848_dt, -0.31627756_dt, -0.14947352_dt,
        -0.12139395_dt, -0.08628038_dt, 0.04279861_dt, 0.41359639_dt, 0.02394396_dt, -0.27612755_dt, 0.02292162_dt, 0.21443319_dt,
        0.22183591_dt };
    const raul::Tensor hh_biases{ -0.40872574_dt, -0.08002549_dt, -0.33234432_dt, -0.19081959_dt, 0.16112810_dt, -0.31757987_dt,
        0.16624129_dt, 0.37958509_dt, 0.02933201_dt, -0.29806235_dt, -0.16022989_dt, 0.09765542_dt, -0.34091896_dt, 0.22218031_dt,
        -0.40601161_dt };
    
    const raul::Tensor output_golden{ -0.04908502_dt, -0.12932368_dt, 0.13605782_dt, 0.02979310_dt, 0.16845636_dt, 0.13728850_dt,
        -0.40618682_dt, 0.23002338_dt, -0.11854070_dt, -0.02049330_dt, -0.19374560_dt, -0.05333966_dt, -0.21428776_dt, 0.19794083_dt,
        0.18007322_dt, 0.22964041_dt, -0.05222400_dt, -0.45063496_dt, 0.11880872_dt, -0.21370764_dt, 0.01256491_dt, -0.18886641_dt,
        -0.33110374_dt, 0.21499550_dt, -0.12713812_dt, 0.09414217_dt, -0.27867889_dt, -0.31757757_dt, 0.17967044_dt, -0.28890455_dt,
        0.12870508_dt, -0.37409306_dt, -0.39582416_dt, -0.17325714_dt, -0.42465699_dt, -0.09158994_dt, -0.14073415_dt, -0.12809369_dt,
        0.27497604_dt, -0.02867714_dt, 0.18024704_dt, -0.13876417_dt, 0.05959263_dt, 0.21977210_dt, -0.05727591_dt, -0.09114996_dt,
        -0.22235252_dt, 0.18779227_dt, 0.30441296_dt, 0.17236847_dt, -0.10560454_dt, -0.16898175_dt, -0.14934486_dt, 0.45449603_dt,
        0.00756194_dt, 0.03945459_dt, -0.36106116_dt, -0.31888619_dt, 0.29281402_dt, -0.27528569_dt, -0.27271730_dt, 0.05054754_dt,
        -0.48803875_dt, 0.48465484_dt, 0.00834736_dt, 0.03663467_dt, 0.14176588_dt, -0.55479670_dt, 0.47510657_dt, 0.04648385_dt };
    const raul::Tensor hidden_golden{ 0.12870508_dt, -0.37409306_dt, -0.39582416_dt, -0.17325714_dt, -0.42465699_dt, 0.03663467_dt,
        0.14176588_dt, -0.55479670_dt, 0.47510657_dt, 0.04648385_dt };

    const raul::Tensor inputs_grad_golden{ 0.06706255_dt, 0.53410208_dt, 0.57455337_dt, -0.62969238_dt, 0.02530956_dt, 0.55312926_dt,
        0.39051867_dt, -0.72325438_dt, 0.02505900_dt, 0.49701324_dt, 0.21707325_dt, -0.10702824_dt, 0.15325701_dt, 0.54164988_dt,
        0.40254655_dt, -0.46143129_dt, -0.05772207_dt, 0.52750760_dt, 0.34876883_dt, -0.55287397_dt, 0.11948650_dt, 0.35562691_dt,
        0.24785848_dt, -0.44526634_dt, 0.12410703_dt, 0.24529704_dt, 0.06473933_dt, -0.29945928_dt, -0.13560967_dt, 0.47893506_dt,
        0.37807149_dt, -0.53091091_dt, 0.09586967_dt, 0.58233088_dt, 0.70613271_dt, -0.60545051_dt, 0.07166006_dt, 0.44349703_dt,
        0.35678184_dt, -0.45502383_dt, 0.06642039_dt, 0.35965362_dt, 0.22833532_dt, -0.32521999_dt, 0.20002100_dt, 0.38204491_dt,
        0.01235957_dt, -0.59885281_dt, -0.12039655_dt, 0.32276195_dt, 0.09595823_dt, -0.10270441_dt, 0.04946303_dt,
        0.25459915_dt, 0.28399706_dt, -0.16773076_dt };
    const raul::Tensor ih_weights_grad_golden{ -1.15681514e-02_dt, 2.45727226e-01_dt, 2.38236189e-02_dt, 1.12843558e-01_dt,
        -4.95894961e-02_dt, -2.31185928e-03_dt, -4.74320278e-02_dt, -2.92562563e-02_dt, -1.96367726e-02_dt, 1.85933888e-01_dt,
        -7.28185698e-02_dt, 1.92100853e-02_dt, -5.43801710e-02_dt, -2.92829037e-01_dt, 2.95404345e-04_dt, 1.37008205e-01_dt,
        -3.14522982e-02_dt, 2.80055493e-01_dt, 1.22947857e-01_dt, 4.58172709e-01_dt, -4.85285789e-01_dt, 1.34967923_dt, -1.05025959_dt,
        -7.19864130e-01_dt, -9.49798748e-02_dt, -2.63819635e-01_dt, 9.84790683e-01_dt, 3.07576895_dt, 1.84837170e-02_dt, -7.02335596e-01_dt,
        -1.74826765_dt, -1.88073802_dt, -3.94826569e-03_dt, -9.48365688e-01_dt, 1.22688019_dt, 1.55642343_dt, 3.03023636e-01_dt,
        -1.74265385_dt, 7.75853097e-02_dt, 9.33504462e-01_dt, 5.49203157e-01_dt, -4.20032644_dt, 8.99180770e-01_dt, -2.47827482_dt,
        3.55211198e-01_dt, -5.78275204e-01_dt, -2.30076051_dt, -1.83073068_dt, 8.96897912e-02_dt, -1.52554870_dt, 8.03956211e-01_dt,
        1.39209092e-01_dt, -5.51340461e-01_dt, -3.11690378_dt, 3.74248147e-01_dt, 1.86640453_dt, 6.53812468e-01_dt, 
        -2.70138478_dt, -1.77771389e-01_dt, -3.11907387_dt };
    const raul::Tensor hh_weights_grad_golden{ 0.01514051_dt, 0.08280170_dt, 0.09544529_dt, -0.13120307_dt, 0.01722318_dt, 0.01949469_dt,
        -0.03729708_dt, -0.05226147_dt, 0.03172657_dt, -0.03127133_dt, 0.00625432_dt, 0.10405099_dt, 0.08481736_dt, -0.14523050_dt,
        0.01143921_dt, -0.00583379_dt, -0.08468263_dt, -0.11317360_dt, 0.15005642_dt, -0.02267838_dt, 0.00314150_dt, 0.19531782_dt,
        0.18590628_dt, -0.23596179_dt, 0.05312429_dt, 0.22438687_dt, -0.14240305_dt, 0.01761173_dt, -0.04466634_dt, -0.16525964_dt,
        -0.05472863_dt, 0.29740956_dt, -0.07371731_dt, 0.04768713_dt, 0.11215933_dt, -0.07453834_dt, -0.33322003_dt, 0.09421252_dt,
        0.18262674_dt, 0.07366519_dt, -0.09912762_dt, 0.20223860_dt, -0.09617800_dt, 0.01614731_dt, 0.04994032_dt, -0.22760686_dt,
        0.02137677_dt, -0.06901166_dt, 0.19307476_dt, 0.19715972_dt, 0.05638356_dt, -0.57298666_dt, -0.25935543_dt, 0.62430120_dt,
        -0.06276698_dt, 0.01983850_dt, -1.00811970_dt, -0.29278663_dt, 0.97892523_dt, -0.05108772_dt, -0.04757062_dt, -0.46518254_dt,
        -0.33999357_dt, 0.64060187_dt, -0.00513114_dt, -0.08422026_dt, -0.48028851_dt, -0.55409992_dt, 0.79238141_dt, -0.05314661_dt,
        -0.05868209_dt, -0.66523165_dt, -0.41538826_dt, 0.73842663_dt, -0.04870981_dt };
    const raul::Tensor ih_biases_grad_golden{ -0.61284310_dt, 0.26072496_dt, -0.79627937_dt, 0.68450576_dt, -1.33463049_dt, 0.37414274_dt,
        0.00920656_dt, 1.02960610_dt, -0.72623384_dt, 0.29689837_dt, 12.5870171_dt, 12.3439045_dt, 7.91638613_dt, 8.89122868_dt, 10.4883652_dt };
    const raul::Tensor hh_biases_grad_golden{ -0.61284310_dt, 0.26072496_dt, -0.79627937_dt, 0.68450576_dt, -1.33463049_dt, 0.37414274_dt,
        0.00920656_dt, 1.02960610_dt, -0.72623384_dt, 0.29689837_dt, 3.95333195_dt, 6.15152121_dt, 3.90701532_dt, 4.00036478_dt, 4.69673634_dt };

    // Initialization
    for (size_t q = 0; q < 2; ++q)
    {
        MANAGERS_DEFINE
        NETWORK_PARAMS_DEFINE(networkParameters);

        work.add<raul::DataLayer>("data1", raul::DataParams{ { "in" }, sequence_length, 1, input_size });

        // Network
        const auto params = raul::GRUParams{ { "in" }, { "out" }, hidden_size, true, false, (q == 1) };
        raul::GRULayer("gru", params, networkParameters);
        TENSORS_CREATE(batch_size)

        memory_manager["in"] = TORANGE(input_init);

        const auto paramGrad = work.getTrainableParameters();
        paramGrad[0].Param = TORANGE(ih_biases);
        paramGrad[1].Param = TORANGE(ih_weights);
        paramGrad[2].Param = TORANGE(hh_biases);
        paramGrad[3].Param = TORANGE(hh_weights);

        // Apply
        ASSERT_NO_THROW(work.forwardPassTraining());

        // Checks
        const auto& outputTensor = memory_manager["out"];
        const auto& hiddenTensor = memory_manager["gru::hidden_state[" + Conversions::toString(sequence_length - 1) + "]"];

        EXPECT_EQ(outputTensor.size(), batch_size * hidden_size * sequence_length);
        EXPECT_EQ(hiddenTensor.size(), batch_size * hidden_size);

        for (size_t i = 0; i < outputTensor.size(); ++i)
        {
            const auto val = outputTensor[i];
            const auto golden_val = output_golden[i];
            ASSERT_TRUE(tools::expect_near_relative(val, golden_val, eps_rel)) << "at " << i << ", expected: " << golden_val << ", got: " << val;
        }

        for (size_t i = 0; i < hiddenTensor.size(); ++i)
        {
            const auto val = hiddenTensor[i];
            const auto golden_val = hidden_golden[i];
            ASSERT_TRUE(tools::expect_near_relative(val, golden_val, eps_rel)) << "at " << i << ", expected: " << golden_val << ", got: " << val;
        }

        // Apply
        memory_manager[raul::Name("out").grad()] = 1.0_dt;
        ASSERT_NO_THROW(work.backwardPassTraining());

        // Checks
        const auto& inputs_grad = memory_manager[raul::Name("in").grad()];
        const auto& inputs = memory_manager["in"];

        EXPECT_EQ(inputs.size(), inputs_grad.size());

        for (size_t i = 0; i < inputs_grad.size(); ++i)
        {
            const auto val = inputs_grad[i];
            const auto golden_val = inputs_grad_golden[i];
            ASSERT_TRUE(tools::expect_near_relative(val, golden_val, eps_rel)) << "at " << i << ", expected: " << golden_val << ", got: " << val;
        }

        const auto gradBiasesIH = paramGrad[0].Gradient;
        const auto gradWeightsIH = paramGrad[1].Gradient;
        const auto gradBiasesHH = paramGrad[2].Gradient;
        const auto gradWeightsHH = paramGrad[3].Gradient;

        EXPECT_EQ(ih_weights_grad_golden.size(), gradWeightsIH.size());

        for (size_t i = 0; i < gradWeightsIH.size(); ++i)
        {
            const auto val = gradWeightsIH[i];
            const auto golden_val = ih_weights_grad_golden[i];
            ASSERT_TRUE(tools::expect_near_relative(val, golden_val, eps_rel)) << "at " << i << ", expected: " << golden_val << ", got: " << val;
        }

        EXPECT_EQ(hh_weights_grad_golden.size(), gradWeightsHH.size());

        for (size_t i = 0; i < gradWeightsHH.size(); ++i)
        {
            const auto val = gradWeightsHH[i];
            const auto golden_val = hh_weights_grad_golden[i];
            ASSERT_TRUE(tools::expect_near_relative(val, golden_val, eps_rel)) << "at " << i << ", expected: " << golden_val << ", got: " << val;
        }

        EXPECT_EQ(ih_biases_grad_golden.size(), gradBiasesIH.size());

        for (size_t i = 0; i < gradBiasesIH.size(); ++i)
        {
            const auto val = gradBiasesIH[i];
            const auto golden_val = ih_biases_grad_golden[i];
            ASSERT_TRUE(tools::expect_near_relative(val, golden_val, eps_rel)) << "at " << i << ", expected: " << golden_val << ", got: " << val;
        }

        EXPECT_EQ(hh_biases_grad_golden.size(), gradBiasesHH.size());

        for (size_t i = 0; i < gradBiasesHH.size(); ++i)
        {
            const auto val = gradBiasesHH[i];
            const auto golden_val = hh_biases_grad_golden[i];
            ASSERT_TRUE(tools::expect_near_relative(val, golden_val, eps_rel)) << "at " << i << ", expected: " << golden_val << ", got: " << val;
        }
    }
}

TEST(TestGRU, ExternalStateSeq5Unit)
{
    PROFILE_TEST
    
    // Test parameters
    const auto eps_rel = 1e-4_dt;
    const size_t input_size = 4U;
    const size_t hidden_size = 3U;
    const size_t sequence_length = 5U;
    const size_t batch_size = 3U;

    const raul::Tensor input_init{ -0.98124963_dt, 0.88884652_dt, 1.56899524_dt, -0.08185262_dt, -0.34940010_dt, 0.20242652_dt,
        -0.28838342_dt, -0.00948004_dt, 2.41873121_dt, 1.32786930_dt, -0.26386023_dt, 0.36446592_dt, 2.54401636_dt, -2.68946719_dt,
        2.44260907_dt, 0.40346053_dt, -0.99648869_dt, 0.97850031_dt, -0.44143954_dt, -0.26104334_dt, 0.79797685_dt, -1.10714447_dt,
        2.33057547_dt, -1.04561079_dt, -0.47496739_dt, -0.49525651_dt, -0.19836050_dt, 2.21487951_dt, -0.13668862_dt, -1.01816082_dt,
        0.17840540_dt, 1.30254650_dt, -0.56442887_dt, -0.91836810_dt, -0.74956363_dt, -0.09493295_dt, 1.10086620_dt, 1.31046069_dt,
        -0.29284531_dt, 0.41807881_dt, -0.16952126_dt, -2.17486167_dt, 0.72024739_dt, 0.28544617_dt, -0.45227137_dt, 0.27532846_dt,
        0.60920399_dt, -1.24595284_dt, 2.49661326_dt, -0.70688295_dt, 1.15044856_dt, -0.54976708_dt, 0.36670256_dt, -0.30200303_dt,
        0.52035600_dt, -0.43189803_dt, -0.47292864_dt, 0.32564098_dt, -0.97362131_dt, 0.79789245_dt };
    const raul::Tensor hidden_init{ 0.19764619_dt, 1.03078353_dt, -0.11038844_dt, 0.36835346_dt, 1.21385443_dt, -1.87656891_dt,
        -0.90929747_dt, 1.49962282_dt, -0.14606902_dt };

    const raul::Tensor ih_weights{ -0.16018850_dt, -0.21523368_dt, 0.14534736_dt, 0.20478249_dt, -0.28207695_dt, 0.05103678_dt,
        0.33459508_dt, -0.05744445_dt, 0.17570728_dt, -0.13924935_dt, 0.20236105_dt, -0.41822916_dt, -0.33949858_dt, -0.29305753_dt,
        0.53059113_dt, -0.15535578_dt, -0.00157636_dt, -0.27972361_dt, 0.57636297_dt, 0.56388080_dt, -0.43542984_dt, -0.46804047_dt,
        -0.43762743_dt, -0.00278443_dt, -0.14716884_dt, -0.37790209_dt, -0.20707944_dt, 0.10907930_dt, -0.30166015_dt, 0.12793076_dt,
        -0.13239560_dt, -0.27976277_dt, 0.07931954_dt, 0.47473097_dt, -0.39033455_dt, 0.02680892_dt };
    const raul::Tensor hh_weights{ -0.21290815_dt, 0.56655741_dt, -0.54776871_dt, -0.55348331_dt, 0.56891227_dt, -0.36528373_dt,
        0.11069155_dt, -0.04982883_dt, -0.12162286_dt, -0.12894991_dt, 0.36687177_dt, 0.02759558_dt, -0.56211662_dt, -0.34084457_dt,
        -0.19681889_dt, 0.29053211_dt, -0.37362283_dt, 0.54439485_dt, -0.12859282_dt, -0.10367015_dt, 0.45238745_dt, 0.29018068_dt,
        0.48966253_dt, 0.33396500_dt, -0.17514145_dt, -0.38305598_dt, -0.04295659_dt };
    const raul::Tensor ih_biases{ 0.47783673_dt, -0.19377017_dt, -0.53541726_dt, 0.23667228_dt, 0.56203401_dt, -0.16436192_dt,
        -0.47805962_dt, -0.52369112_dt, 0.14468360_dt };
    const raul::Tensor hh_biases{ -0.04371679_dt, -0.29155451_dt, 0.11670089_dt, 0.21925384_dt, 0.45917761_dt, 0.44824445_dt,
        -0.08641994_dt, -0.50908852_dt, -0.52170706_dt };
    
    const raul::Tensor output_golden{ 0.04099482_dt, 0.72546315_dt, -0.35705012_dt, -0.18205380_dt, 0.30703565_dt, -0.16191800_dt,
        -0.65896875_dt, -0.07225198_dt, 0.53462708_dt, -0.57538640_dt, -0.09426290_dt, -0.40636504_dt, -0.60188627_dt, -0.18984750_dt,
        0.03315166_dt, 0.24364877_dt, 0.92888021_dt, -1.07731462_dt, -0.02491495_dt, 0.72756612_dt, -0.61441928_dt, -0.10913710_dt,
        0.54654652_dt, -0.58707130_dt, -0.14081354_dt, 0.21234041_dt, -0.49405333_dt, -0.56583554_dt, -0.08478528_dt, 0.36142004_dt,
        -0.80644315_dt, 1.28136015_dt, -0.50205016_dt, -0.80267113_dt, 0.76070410_dt, -0.45086491_dt, -0.79942143_dt, 0.53077704_dt,
        -0.66155320_dt, -0.76926607_dt, 0.30146921_dt, -0.50100321_dt, -0.59248662_dt, 0.08321577_dt, -0.04018950_dt };
    const raul::Tensor hidden_golden{ -0.60188627_dt, -0.18984750_dt, 0.03315166_dt, -0.56583554_dt, -0.08478528_dt, 0.36142004_dt,
        -0.59248662_dt, 0.08321577_dt, -0.04018950_dt };

    const raul::Tensor inputs_grad_golden{ -0.25887769_dt, 0.21682419_dt, -0.09057298_dt, 0.26790270_dt, -0.21655954_dt, 0.11239155_dt,
        0.10340309_dt, 0.24684711_dt, -0.08420015_dt, 0.12760904_dt, 0.17206614_dt, 0.35157475_dt, -0.23247159_dt, -0.29676440_dt,
        -0.33962688_dt, 0.06672341_dt, 0.01960248_dt, 0.19405299_dt, -0.10021859_dt, 0.01033405_dt, -0.01225911_dt, -0.12045509_dt,
        0.58137876_dt, 0.41612568_dt, 0.06256969_dt, 0.35233840_dt, 0.08686145_dt, 0.33367401_dt, -0.06411639_dt, 0.06608679_dt,
        -0.10589395_dt, 0.24795744_dt, -0.11367548_dt, 0.00650537_dt, -0.02212469_dt, 0.17822637_dt, 0.00484396_dt, 0.15573753_dt,
        0.06359266_dt, 0.10736005_dt, -0.11233012_dt, -0.22879635_dt, -0.04534333_dt, 0.32057947_dt, -0.19985782_dt, 0.17319165_dt,
        0.06864062_dt, 0.38571107_dt, -0.02268109_dt, 0.11460479_dt, -0.09249460_dt, 0.31114897_dt, 0.03686558_dt, 0.32430127_dt,
        -0.21339902_dt, 0.24955866_dt, 0.07459679_dt, 0.13102753_dt, -0.09447040_dt, 0.14053909_dt };
    const raul::Tensor ih_weights_grad_golden{ 0.01096504_dt, -0.01558308_dt, 0.05583686_dt, -0.10004991_dt, 0.03207716_dt,
        -0.05444815_dt, 0.04632650_dt, 0.02723377_dt, -0.20065905_dt, -0.08330191_dt, -0.49565664_dt, -0.22866280_dt, 0.52743018_dt,
        1.11495745_dt, 0.12717551_dt, 0.66095996_dt, 1.73225510_dt, -1.31874371_dt, 3.57361364_dt, -0.38388062_dt, 0.98578322_dt,
        -2.33836722_dt, 1.91859090_dt, -0.87441468_dt, 1.14404619_dt, -1.00529671_dt, 0.21646389_dt, 1.83426893_dt, -0.84994364_dt,
        0.66395962_dt, 0.97516519_dt, -0.77440029_dt, 1.94098234_dt, 0.54351902_dt, 2.14976430_dt, 1.99930155_dt };
    const raul::Tensor hh_weights_grad_golden{ 0.03578843_dt, -0.11866765_dt, 0.11812082_dt, 0.10448553_dt, -0.15508111_dt,
        0.15297520_dt, 0.30883890_dt, -0.99575895_dt, 0.64709747_dt, 0.44530675_dt, 0.72096628_dt, -0.95579660_dt, -1.60582161_dt,
        5.66374731_dt, -4.01859045_dt, -0.32694715_dt, -0.52114737_dt, 1.62126541_dt, -0.88528574_dt, 1.78323519_dt, -1.39898789_dt,
        -0.66307080_dt, 1.93795180_dt, -1.04120386_dt, -1.07339120_dt, 2.28095698_dt, -1.58336806_dt };
    const raul::Tensor ih_biases_grad_golden{ -0.18053019_dt, -0.29352021_dt, -1.28909278_dt, 1.08188605_dt, 6.52996635_dt, -0.93364418_dt,
        3.78647590_dt, 3.67670298_dt, 7.98870373_dt };
    const raul::Tensor hh_biases_grad_golden{ -0.18053019_dt, -0.29352021_dt, -1.28909278_dt, 1.08188605_dt, 6.52996635_dt, -0.93364418_dt,
        2.84415555_dt, 2.21476746_dt, 3.18151975_dt };

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    work.add<raul::DataLayer>("data1", raul::DataParams{ { "in" }, sequence_length, 1, input_size });
    work.add<raul::DataLayer>("data2", raul::DataParams{ { "hidden" }, 1, 1, hidden_size });

    // Network
    const auto params = raul::GRUParams{ { "in", "hidden" }, { "out", "new_hidden" } };
    raul::GRULayer("gru", params, networkParameters);
    TENSORS_CREATE(batch_size)

    memory_manager["in"] = TORANGE(input_init);
    memory_manager["hidden"] = TORANGE(hidden_init);

    const auto paramGrad = work.getTrainableParameters();
    paramGrad[0].Param = TORANGE(ih_biases);
    paramGrad[1].Param = TORANGE(ih_weights);
    paramGrad[2].Param = TORANGE(hh_biases);
    paramGrad[3].Param = TORANGE(hh_weights);

    // Apply
    ASSERT_NO_THROW(work.forwardPassTraining());

    // Checks
    const auto& outputTensor = memory_manager["out"];
    const auto& hiddenTensor = memory_manager["new_hidden"];

    EXPECT_EQ(outputTensor.size(), batch_size * hidden_size * sequence_length);
    EXPECT_EQ(hiddenTensor.size(), batch_size * hidden_size);

    for (size_t i = 0; i < outputTensor.size(); ++i)
    {
        const auto val = outputTensor[i];
        const auto golden_val = output_golden[i];
        ASSERT_TRUE(tools::expect_near_relative(val, golden_val, eps_rel)) << "at " << i << ", expected: " << golden_val << ", got: " << val;
    }

    for (size_t i = 0; i < hiddenTensor.size(); ++i)
    {
        const auto val = hiddenTensor[i];
        const auto golden_val = hidden_golden[i];
        ASSERT_TRUE(tools::expect_near_relative(val, golden_val, eps_rel)) << "at " << i << ", expected: " << golden_val << ", got: " << val;
    }

    // Apply
    memory_manager[raul::Name("out").grad()] = 1.0_dt;
    ASSERT_NO_THROW(work.backwardPassTraining());

    // Checks
    const auto& inputs_grad = memory_manager[raul::Name("in").grad()];
    const auto& inputs = memory_manager["in"];

    EXPECT_EQ(inputs.size(), inputs_grad.size());

    for (size_t i = 0; i < inputs_grad.size(); ++i)
    {
        const auto val = inputs_grad[i];
        const auto golden_val = inputs_grad_golden[i];
        ASSERT_TRUE(tools::expect_near_relative(val, golden_val, eps_rel)) << "at " << i << ", expected: " << golden_val << ", got: " << val;
    }

    const auto gradBiasesIH = paramGrad[0].Gradient;
    const auto gradWeightsIH = paramGrad[1].Gradient;
    const auto gradBiasesHH = paramGrad[2].Gradient;
    const auto gradWeightsHH = paramGrad[3].Gradient;

    EXPECT_EQ(ih_weights_grad_golden.size(), gradWeightsIH.size());

    for (size_t i = 0; i < gradWeightsIH.size(); ++i)
    {
        const auto val = gradWeightsIH[i];
        const auto golden_val = ih_weights_grad_golden[i];
        ASSERT_TRUE(tools::expect_near_relative(val, golden_val, eps_rel)) << "at " << i << ", expected: " << golden_val << ", got: " << val;
    }

    EXPECT_EQ(hh_weights_grad_golden.size(), gradWeightsHH.size());

    for (size_t i = 0; i < gradWeightsHH.size(); ++i)
    {
        const auto val = gradWeightsHH[i];
        const auto golden_val = hh_weights_grad_golden[i];
        ASSERT_TRUE(tools::expect_near_relative(val, golden_val, eps_rel)) << "at " << i << ", expected: " << golden_val << ", got: " << val;
    }

    EXPECT_EQ(ih_biases_grad_golden.size(), gradBiasesIH.size());

    for (size_t i = 0; i < gradBiasesIH.size(); ++i)
    {
        const auto val = gradBiasesIH[i];
        const auto golden_val = ih_biases_grad_golden[i];
        ASSERT_TRUE(tools::expect_near_relative(val, golden_val, eps_rel)) << "at " << i << ", expected: " << golden_val << ", got: " << val;
    }

    EXPECT_EQ(hh_biases_grad_golden.size(), gradBiasesHH.size());

    for (size_t i = 0; i < gradBiasesHH.size(); ++i)
    {
        const auto val = gradBiasesHH[i];
        const auto golden_val = hh_biases_grad_golden[i];
        ASSERT_TRUE(tools::expect_near_relative(val, golden_val, eps_rel)) << "at " << i << ", expected: " << golden_val << ", got: " << val;
    }
}

}
