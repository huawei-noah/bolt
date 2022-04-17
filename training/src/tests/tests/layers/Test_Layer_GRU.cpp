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

#include <training/compiler/Layers.h>
#include <training/base/layers/composite/rnn/GRULayer.h>

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
    const auto params = raul::GRUParams{ { "in" }, { "out" }, hidden_size, false };
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

    const auto gradBiasesIH = memory_manager[raul::Name("gru::cell::linear_ih::Biases").grad()];
    const auto gradWeightsIH = memory_manager[raul::Name("gru::cell::linear_ih::Weights").grad()];
    const auto gradBiasesHH = memory_manager[raul::Name("gru::cell::linear_hh::Biases").grad()];
    const auto gradWeightsHH = memory_manager[raul::Name("gru::cell::linear_hh::Weights").grad()];

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
    const auto params = raul::GRUParams{ { "in" }, { "out" }, hidden_size, false, true, true, false, true };
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

    const auto gradBiasesIH = memory_manager[raul::Name("gru::cell::linear_ih::Biases").grad()];
    const auto gradWeightsIH = memory_manager[raul::Name("gru::cell::linear_ih::Weights").grad()];
    const auto gradBiasesHH = memory_manager[raul::Name("gru::cell::linear_hh::Biases").grad()];
    const auto gradWeightsHH = memory_manager[raul::Name("gru::cell::linear_hh::Weights").grad()];

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

TEST(TestGRU, SimpleSeq3GlobalFusionUnit)
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
    const auto params = raul::GRUParams{ { "in" }, { "out" }, hidden_size, true };
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

    const auto gradBiasesIH = memory_manager[raul::Name("gru::cell::linear_ih::Biases").grad()];
    const auto gradWeightsIH = memory_manager[raul::Name("gru::cell::linear_ih::Weights").grad()];
    const auto gradBiasesHH = memory_manager[raul::Name("gru::cell::linear_hh::Biases").grad()];
    const auto gradWeightsHH = memory_manager[raul::Name("gru::cell::linear_hh::Weights").grad()];


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
        bool useFusion = (q == 1 ? true : false);
        const auto params = raul::GRUParams{ { "in" }, { "out" }, hidden_size, false, true, true, false, useFusion };
        raul::GRULayer("gru", params, networkParameters);
        TENSORS_CREATE(batch_size)

        memory_manager["in"] = TORANGE(input_init);

        memory_manager["gru::cell::linear_ih::Biases"] = TORANGE(ih_biases);
        memory_manager["gru::cell::linear_ih::Weights"] = TORANGE(ih_weights);
        memory_manager["gru::cell::linear_hh::Biases"] = TORANGE(hh_biases);
        memory_manager["gru::cell::linear_hh::Weights"] = TORANGE(hh_weights);

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

        const auto gradBiasesIH = memory_manager[raul::Name("gru::cell::linear_ih::Biases").grad()];
        const auto gradWeightsIH = memory_manager[raul::Name("gru::cell::linear_ih::Weights").grad()];
        const auto gradBiasesHH = memory_manager[raul::Name("gru::cell::linear_hh::Biases").grad()];
        const auto gradWeightsHH = memory_manager[raul::Name("gru::cell::linear_hh::Weights").grad()];

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
    const auto params = raul::GRUParams{ { "in", "hidden" }, { "out", "new_hidden" }, false };
    raul::GRULayer("gru", params, networkParameters);
    TENSORS_CREATE(batch_size)

    memory_manager["in"] = TORANGE(input_init);
    memory_manager["hidden"] = TORANGE(hidden_init);

    memory_manager["gru::cell::linear_ih::Biases"] = TORANGE(ih_biases);
    memory_manager["gru::cell::linear_ih::Weights"] = TORANGE(ih_weights);
    memory_manager["gru::cell::linear_hh::Biases"] = TORANGE(hh_biases);
    memory_manager["gru::cell::linear_hh::Weights"] = TORANGE(hh_weights);

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

    const auto gradBiasesIH = memory_manager[raul::Name("gru::cell::linear_ih::Biases").grad()];
    const auto gradWeightsIH = memory_manager[raul::Name("gru::cell::linear_ih::Weights").grad()];
    const auto gradBiasesHH = memory_manager[raul::Name("gru::cell::linear_hh::Biases").grad()];
    const auto gradWeightsHH = memory_manager[raul::Name("gru::cell::linear_hh::Weights").grad()];

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

TEST(TestGRU, SimpleSeq7FusionOffAndOnSeparateBiasesManagementUnit)
{
    PROFILE_TEST
    
    // Test parameters
    const auto eps_rel = 1e-3_dt;
    const size_t input_size = 4U;
    const size_t hidden_size = 5U;
    const size_t sequence_length = 7U;
    const size_t batch_size = 2U;

    const raul::Tensor input_init{ -0.49871477_dt, 0.76111090_dt, 0.61830086_dt, -0.29938447_dt, -0.18777807_dt, 1.91589761_dt,
        0.69019532_dt, -2.32170153_dt, -1.07870805_dt, 0.24081051_dt, -1.39622724_dt, 0.11355210_dt, 1.10472572_dt, -1.39517200_dt,
        0.47511867_dt, -0.81372583_dt, 0.13147806_dt, 1.57353830_dt, 0.78142995_dt, 0.98743659_dt, -1.48780751_dt, 0.58668745_dt,
        0.15829536_dt, 0.11024587_dt, -0.99438965_dt, -1.18936396_dt, -1.19594944_dt, 1.31189203_dt, -0.20983894_dt, 0.78173131_dt,
        0.98969257_dt, 0.41471335_dt, 0.87604475_dt, -0.28708741_dt, 1.02164006_dt, -0.51107377_dt, -1.71372783_dt, -0.51006478_dt,
        -0.47489312_dt, -0.63340378_dt, -2.88906121_dt, -1.50998127_dt, 1.02411532_dt, -0.34333581_dt, 1.57126486_dt, 0.19161488_dt,
        0.37994185_dt, -0.14475703_dt, 1.45025015_dt, -0.05190918_dt, -0.62843078_dt, -0.14201079_dt, -0.53414577_dt, -0.52337903_dt,
        0.86150426_dt, -0.88696319_dt };

    const raul::Tensor ih_weights{ -0.00334820_dt, 0.23990488_dt, -0.36807698_dt, -0.32912195_dt, -0.17224628_dt, 0.11992359_dt,
        -0.00886074_dt, 0.35459095_dt, -0.03968754_dt, 0.11833835_dt, -0.13515380_dt, -0.08790672_dt, -0.42724484_dt,
        -0.29618156_dt, -0.18435177_dt, 0.01656640_dt, 0.17679930_dt, 0.26833832_dt, -0.30318445_dt, -0.19474488_dt,
        0.16243565_dt, 0.37136078_dt, -0.09203663_dt, 0.33465517_dt, -0.07208338_dt, 0.04732150_dt, 0.40494126_dt,
        -0.41486681_dt, -0.28153792_dt, -0.11321893_dt, -0.17432383_dt, 0.38639289_dt, -0.28987473_dt, -0.20586711_dt,
        -0.31244153_dt, -0.41884279_dt, -0.26105666_dt, 0.38442391_dt, 0.19955492_dt, 0.21675217_dt, 0.02351967_dt, -0.22927903_dt,
        0.07566172_dt, -0.41756096_dt, -0.32314146_dt, -0.23055202_dt, 0.28216404_dt, 0.26221085_dt, -0.19833700_dt, -0.01613653_dt,
        0.28602022_dt, 0.44458985_dt, 0.17749113_dt, 0.06041539_dt, 0.29985058_dt, -0.26332039_dt, 0.08333558_dt, -0.34672716_dt,
        -0.30995756_dt, -0.23102319_dt };
    const raul::Tensor hh_weights{ 0.20235211_dt, 0.17985159_dt, -0.26490808_dt, 0.13510638_dt, 0.24550772_dt, -0.05644614_dt,
        0.01707530_dt, 0.10362148_dt, 0.27744085_dt, 0.42941183_dt, -0.34463334_dt, -0.16389024_dt, 0.17575938_dt, 0.37053853_dt,
        0.38916856_dt, 0.39460194_dt, 0.08900201_dt, -0.38888919_dt, 0.04114029_dt, -0.27977920_dt, -0.41678256_dt, 0.39734590_dt,
        0.34004325_dt, -0.44610807_dt, 0.08370590_dt, -0.07533762_dt, -0.07359397_dt, -0.20471510_dt, 0.17197877_dt, -0.26488620_dt,
        0.16394460_dt, 0.22615951_dt, 0.32014751_dt, 0.16721815_dt, -0.44262305_dt, -0.29010606_dt, 0.22330046_dt, 0.09360242_dt,
        -0.34886417_dt, -0.25751430_dt, 0.42071587_dt, 0.30134052_dt, -0.19499636_dt, -0.11255684_dt, -0.42601481_dt, -0.00803828_dt,
        -0.33677816_dt, -0.34496120_dt, -0.02464131_dt, 0.06714690_dt, -0.18314752_dt, 0.26536649_dt, -0.27214694_dt, 0.40578824_dt,
        0.30647540_dt, -0.37712759_dt, -0.11130446_dt, 0.02017945_dt, 0.06524897_dt, 0.10606754_dt, 0.17549926_dt, 0.02678818_dt,
        -0.21820837_dt, 0.21161658_dt, -0.42898914_dt, -0.26506650_dt, -0.11195090_dt, -0.21784371_dt, -0.15645024_dt, -0.36654595_dt,
        -0.09512910_dt, 0.09559476_dt, -0.29134434_dt, -0.02295071_dt, 0.32013822_dt };
    const raul::Tensor ih_biases{ -0.04597366_dt, 0.01242906_dt, -0.03858063_dt, 0.09050769_dt, 0.28435606_dt, 0.42362136_dt,
        0.28400564_dt, 0.42459065_dt, -0.03234324_dt, -0.40174159_dt, -0.21201378_dt, 0.30451006_dt, -0.00289905_dt,
        -0.22228588_dt, -0.34270504_dt };
    const raul::Tensor hh_biases{ -0.41852576_dt, -0.37745196_dt, -0.09071136_dt, 0.24525464_dt, 0.24178201_dt, -0.43130705_dt,
        0.27896380_dt, -0.34994885_dt, -0.09454554_dt, -0.18133289_dt, -0.08614016_dt, -0.08780712_dt, -0.40130711_dt,
        -0.38614115_dt, -0.06997976_dt };
    
    const raul::Tensor output_golden[] { { -8.57370198e-02_dt, 1.19639024e-01_dt, 5.53444251e-02_dt, -3.10806499e-04_dt,
        -3.10788631e-01_dt, 8.00666958e-02_dt, 3.76263261e-02_dt, -4.01700109e-01_dt, 2.31814235e-01_dt, -4.84594762e-01_dt,
        -7.17546046e-02_dt, 1.17292888e-01_dt, -2.81554103e-01_dt, -5.21355867e-03_dt, -2.93126822e-01_dt, 2.17504472e-01_dt,
        1.38011470e-01_dt, -3.11364532e-01_dt, 1.34796441e-01_dt, 1.46658793e-01_dt, 2.59505510e-02_dt, 2.19239727e-01_dt,
        6.08797073e-02_dt, -7.75245801e-02_dt, -2.07820177e-01_dt, -1.48364425e-01_dt, 3.54213387e-01_dt, 1.54362008e-01_dt,
        -2.04493836e-01_dt, -4.12495077e-01_dt, -3.18946958e-01_dt, 5.72942138e-01_dt, 1.94450945e-01_dt, -4.19023573e-01_dt,
        -1.42185301e-01_dt, -1.49107546e-01_dt, 1.89451307e-01_dt, 1.85296476e-01_dt, -1.60599183e-02_dt, -3.41232061e-01_dt,
        -2.55143046e-02_dt, 2.01914787e-01_dt, 5.86769059e-02_dt, 2.03721538e-01_dt, -3.93848389e-01_dt, 3.77948023e-02_dt,
        3.24185044e-01_dt, 4.96790037e-02_dt, 5.14572561e-02_dt, -1.99050665e-01_dt, 2.09582925e-01_dt, 4.70343411e-01_dt,
        1.89169794e-01_dt, -5.90947568e-02_dt, -2.57973611e-01_dt, 8.13183188e-02_dt, 2.35455722e-01_dt, -1.95161998e-02_dt,
        8.88638273e-02_dt, -3.67845953e-01_dt, 1.69192255e-03_dt, -3.95700634e-02_dt, -2.00530604e-01_dt, 5.09417057e-03_dt,
        -1.26955509e-01_dt, 1.74372435e-01_dt, 1.16178185e-01_dt, -1.03745155e-01_dt, 8.13597590e-02_dt, -2.04327971e-01_dt }, 
        { -0.02961647_dt, 0.01451481_dt, -0.04160413_dt, 0.00958520_dt, -0.18542275_dt, 0.28184667_dt, -0.09257030_dt,
        -0.62827367_dt, 0.26646343_dt, -0.28583765_dt, 0.01991658_dt, -0.13148597_dt, -0.39614418_dt, -0.01365161_dt,
        -0.05323233_dt, 0.44904941_dt, -0.13373131_dt, -0.46756098_dt, 0.13595393_dt, 0.37929219_dt, 0.04046041_dt,
        -0.08687550_dt, 0.11659843_dt, -0.08234733_dt, 0.07530963_dt, -0.14636600_dt, 0.09070024_dt, 0.12827668_dt,
        -0.26135188_dt, -0.11404249_dt, -0.32152492_dt, 0.37199897_dt, 0.16131368_dt, -0.50541115_dt, 0.14930819_dt,
        -0.15999079_dt, 0.08358504_dt, 0.18025717_dt, 0.00758567_dt, -0.24013162_dt, 0.15188913_dt, 0.03908857_dt,
        -0.11993092_dt, 0.25217247_dt, -0.16721363_dt, 0.25011995_dt, 0.13385345_dt, -0.14975448_dt, 0.04662204_dt,
        0.06902729_dt, 0.42518508_dt, 0.33955550_dt, 0.15296564_dt, -0.11062419_dt, 0.05563445_dt, 0.19338971_dt,
        0.02292895_dt, -0.25131604_dt, 0.08438525_dt, -0.03594999_dt, 0.11438245_dt, -0.30490211_dt, -0.47564423_dt,
        -0.00414112_dt, 0.20976135_dt, 0.39620376_dt, -0.11654735_dt, -0.28611282_dt, 0.05767488_dt, 0.16176815_dt },
        { -0.00706040_dt, 0.03178396_dt, 0.07029247_dt, 0.11342310_dt, -0.15415509_dt, 0.26933858_dt, -0.09887218_dt,
        -0.48309278_dt, 0.34790334_dt, -0.24748604_dt, 0.08931449_dt, -0.11649603_dt, -0.26981816_dt, 0.09851944_dt,
        -0.04053652_dt, 0.43648767_dt, -0.11886206_dt, -0.32755351_dt, 0.25758505_dt, 0.37187147_dt, 0.14333361_dt,
        -0.04956196_dt, 0.19021407_dt, 0.07831401_dt, 0.09686732_dt, -0.04139061_dt, 0.14461881_dt, 0.25575104_dt,
        -0.08327460_dt, -0.07595009_dt, -0.22977406_dt, 0.42075720_dt, 0.29032698_dt, -0.35370421_dt, 0.16582713_dt,
        -0.11684012_dt, 0.11565334_dt, 0.23445119_dt, 0.12515122_dt, -0.20633316_dt, 0.14084697_dt, 0.06108802_dt,
        0.03741142_dt, 0.35524768_dt, -0.13273706_dt, 0.26739243_dt, 0.17413653_dt, 0.02623897_dt, 0.18181825_dt,
        0.08243188_dt, 0.43924320_dt, 0.39940402_dt, 0.25082776_dt, 0.05352539_dt, 0.06630776_dt, 0.27124614_dt,
        0.01792890_dt, -0.08152765_dt, 0.22347879_dt, -0.00778555_dt, 0.19513480_dt, -0.33423594_dt, -0.30472487_dt,
        0.14336413_dt, 0.21643990_dt, 0.40987831_dt, -0.10091737_dt, -0.12359181_dt, 0.22056001_dt, 0.16674608_dt } };
    const raul::Tensor hidden_golden[] { { -0.31894696_dt, 0.57294214_dt, 0.19445094_dt, -0.41902357_dt, -0.14218530_dt,
        0.17437243_dt, 0.11617818_dt, -0.10374516_dt, 0.08135976_dt, -0.20432797_dt },
        { -0.32152492_dt, 0.37199897_dt, 0.16131368_dt, -0.50541115_dt, 0.14930819_dt, 0.39620376_dt, -0.11654735_dt,
        -0.28611282_dt, 0.05767488_dt, 0.16176815_dt },
        { -0.22977406_dt, 0.42075720_dt, 0.29032698_dt, -0.35370421_dt, 0.16582713_dt, 0.40987831_dt, -0.10091737_dt,
        -0.12359181_dt, 0.22056001_dt, 0.16674608_dt } };

    const raul::Tensor inputs_grad_golden[] { { -9.25021023e-02_dt, -3.01523805e-01_dt, 6.98627114e-01_dt, -1.43015325e-01_dt,
        -7.86656663e-02_dt, -3.43749017e-01_dt, 5.18770278e-01_dt, -8.68005976e-02_dt, -3.02938372e-01_dt, -7.31878459e-01_dt,
        1.37118518e-01_dt, -1.72322974e-01_dt, -3.47757265e-02_dt, -9.39048290e-01_dt, 1.94955528e-01_dt, -4.81629193e-01_dt,
        -3.52195911e-02_dt, 7.49912187e-02_dt, 8.13023865e-01_dt, -8.30701441e-02_dt, -5.15766777e-02_dt, -1.91788614e-01_dt,
        2.56899089e-01_dt, -1.37026951e-01_dt, -2.14259475e-02_dt, -3.93410474e-01_dt, -1.19292483e-01_dt, -1.93351388e-01_dt,
        -4.44839895e-02_dt, -1.23751394e-01_dt, 7.11426079e-01_dt, 1.26082888e-02_dt, -4.33393158e-02_dt, -5.82754374e-01_dt,
        7.31405020e-01_dt, -1.44216880e-01_dt, -1.71302691e-01_dt, -8.06554735e-01_dt, -9.25432905e-05_dt, -3.58552724e-01_dt,
        8.11034665e-02_dt, -5.18651307e-01_dt, -3.18620913e-02_dt, -6.58564389e-01_dt, -8.97924528e-02_dt, -4.08081114e-01_dt,
        6.63027287e-01_dt, -2.07603231e-01_dt, -1.12009682e-01_dt, -6.25381708e-01_dt, 2.95281917e-01_dt, -3.43482643e-01_dt,
        -1.62094869e-02_dt, -3.42224836e-01_dt, 1.50504053e-01_dt, -1.51037127e-01_dt },
        { -0.08421224_dt, -0.44107383_dt, 0.78805649_dt, -0.24144687_dt, 0.06028999_dt, -0.34397894_dt, 0.43945572_dt,
        -0.25038162_dt, -0.34120509_dt, -0.73727071_dt, 0.26539871_dt, -0.18698014_dt, 0.01312664_dt, -0.73127222_dt,
        0.29258260_dt, -0.35089502_dt, -0.07429407_dt, 0.02853717_dt, 0.91718560_dt, 0.04553531_dt, -0.16895452_dt,
        -0.35225326_dt, 0.41099527_dt, -0.04041295_dt, -0.11435586_dt, -0.47189313_dt, 0.03883795_dt, -0.15085733_dt,
        -0.11003456_dt, -0.26289243_dt, 0.86028969_dt, -0.01405015_dt, 0.02566952_dt, -0.64772069_dt, 0.74743372_dt,
        -0.26823974_dt, -0.28428474_dt, -0.85054433_dt, 0.14142622_dt, -0.26525578_dt, 0.03493105_dt, -0.46877751_dt,
        -0.05046972_dt, -0.41282529_dt, 0.03381753_dt, -0.43182972_dt, 0.68081868_dt, -0.37568352_dt, -0.00581875_dt,
        -0.54737091_dt, 0.38047162_dt, -0.46099150_dt, -0.06257868_dt, -0.35707968_dt, 0.29034615_dt, -0.05703910_dt },
        { -0.11691321_dt, -0.45594504_dt, 0.79155427_dt, -0.17999138_dt, -0.03996599_dt, -0.36488774_dt, 0.43468049_dt,
        -0.11774554_dt, -0.32652164_dt, -0.73080486_dt, 0.28780222_dt, -0.21729580_dt, -0.03788246_dt, -0.77943867_dt,
        0.26735383_dt, -0.29480329_dt, -0.08147183_dt, 0.01550219_dt, 0.89700401_dt, -0.03946599_dt, -0.14152449_dt,
        -0.30673307_dt, 0.38511863_dt, -0.10278194_dt, -0.10865213_dt, -0.43026522_dt, 0.02680536_dt, -0.16575882_dt,
        -0.11480118_dt, -0.26361251_dt, 0.84987438_dt, -0.01486738_dt, -0.07805163_dt, -0.67427260_dt, 0.69294155_dt,
        -0.15906250_dt, -0.27417341_dt, -0.82342595_dt, 0.17474693_dt, -0.26968765_dt, 0.06483445_dt, -0.46927723_dt,
        -0.03933090_dt, -0.50779426_dt, -0.05170135_dt, -0.44503903_dt, 0.65237957_dt, -0.30915731_dt, -0.06642064_dt,
        -0.54794633_dt, 0.37532255_dt, -0.36642486_dt, -0.09255892_dt, -0.33367574_dt, 0.25742531_dt, -0.00314695_dt } };
    const raul::Tensor hidden_grad_golden[] { { 0.65615290_dt, 1.83252537_dt, 0.13781744_dt, 1.44201040_dt, 1.00348175_dt,
        0.54186738_dt, 1.78111482_dt, 0.33433187_dt, 0.89475089_dt, 0.86535913_dt },
        { 0.04626983_dt, 1.83335853_dt, -0.09080449_dt, 1.26611650_dt, 1.11229444_dt, 0.03718096_dt, 1.77139342_dt,
        0.02294287_dt, 0.76090360_dt, 0.88440871_dt },
        { 0.24432102_dt, 1.47863686_dt, 0.04352704_dt, 1.53253007_dt, 1.44209468_dt, 0.20157242_dt, 1.33543420_dt,
        0.17899784_dt, 0.98194647_dt, 1.17850268_dt } };
    const raul::Tensor ih_weights_grad_golden [] { { -2.74017211e-02_dt, -1.49154505e-02_dt, -3.91223421e-03_dt, 2.04608459e-02_dt,
        -1.59261990e-02_dt, -2.35812552e-02_dt, 2.28142738e-02_dt, -6.13229629e-03_dt, -6.25466462e-03_dt, -1.23037528e-02_dt,
        -7.20387138e-03_dt, -8.71205553e-02_dt, 1.47865163e-02_dt, -7.40025565e-02_dt, 6.69760257e-03_dt, -1.14621691e-01_dt,
        2.85029672e-02_dt, 3.80026065e-02_dt, 2.38449834e-02_dt, 3.90766561e-02_dt, -2.40946472e-01_dt, 9.29698944e-01_dt,
        -4.58416164e-01_dt, 1.30317163e+00_dt, 2.67947698e+00_dt, 3.83054167e-01_dt, -6.44073009e-01_dt, -3.71567458e-01_dt,
        1.27126896e+00_dt, -1.02224134e-01_dt, -2.12133408e-01_dt, -1.12794220e+00_dt, -2.11958432e+00_dt, -6.71627343e-01_dt,
        -1.28821325e+00_dt, 1.34202528e+00_dt, -3.34208250e-01_dt, 1.78880835e+00_dt, 1.36162257e+00_dt, 1.37326837e-01_dt,
        -3.80962086e+00_dt, -1.06971049e+00_dt, 2.75867891e+00_dt, -3.59006214e+00_dt, -7.25637794e-01_dt, 1.69286978e+00_dt,
        1.00791526e+00_dt, -7.71707177e-01_dt, 2.50608325e-02_dt, 1.10111403e+00_dt, 2.52171850e+00_dt, -2.61387706e+00_dt,
        1.72042727e-01_dt, 3.93380070e+00_dt, 5.73299503e+00_dt, -1.88192308e+00_dt, -1.26346266e+00_dt, -2.35832357e+00_dt,
        9.52226520e-01_dt, -3.33012009e+00_dt },
        { 0.01515695_dt, -0.01556537_dt, -0.04193422_dt, 0.04416762_dt, 0.07711352_dt, -0.08050019_dt, 0.00775786_dt,
        -0.01008915_dt, 0.13275868_dt, -0.12212355_dt, -0.29995272_dt, 0.12567018_dt, -0.17704812_dt, -0.49933982_dt,
        -0.55892980_dt, 0.05801111_dt, 0.00698950_dt, -0.00710623_dt, -0.02897462_dt, 0.04966169_dt, -0.15025759_dt,
        0.68107641_dt, -0.31230891_dt, 1.11865151_dt, 2.77492499_dt, 0.90601391_dt, -0.35773870_dt, -0.49675110_dt,
        1.06987023_dt, 0.04967080_dt, -0.09080416_dt, -0.64871752_dt, -2.45233822_dt, -0.84692669_dt, -1.30428219_dt,
        1.24401033_dt, -0.19111200_dt, 1.76824546_dt, 1.23853469_dt, 0.37137318_dt, -3.33355784_dt, -0.20663369_dt,
        2.25592971_dt, -2.85118556_dt, -2.24702334_dt, 1.50240397_dt, 1.27767754_dt, -0.45343941_dt, -1.78661406_dt,
        1.10452437_dt, 2.52133369_dt, -1.82008171_dt, 0.54681885_dt, 3.67907667_dt, 5.52823734_dt, -1.88177872_dt,
        -1.36771071_dt, -0.96405947_dt, 1.84500313_dt, -3.30103755_dt },
        { -6.25821874e-02_dt, -1.98632143e-02_dt, 1.50990663e-02_dt, -2.28704382e-02_dt, 6.19283691e-03_dt, -3.26259732e-02_dt,
        2.91429684e-02_dt, -9.97227430e-03_dt, -2.77164709e-02_dt, -7.06638768e-03_dt, -3.88959534e-02_dt, -5.89128211e-02_dt,
        -4.35883403e-02_dt, -1.22362770e-01_dt, -7.91351646e-02_dt, -6.85479194e-02_dt, -3.83072160e-03_dt, -5.03487140e-03_dt,
        -8.37371871e-03_dt, 2.70785354e-02_dt, -1.38976425e-01_dt, 6.47332847e-01_dt, -4.35974658e-01_dt, 1.28151321e+00_dt,
        2.78643227e+00_dt, 8.62253726e-01_dt, -3.56498510e-01_dt, -5.02790391e-01_dt, 1.21891451e+00_dt, -7.72543699e-02_dt,
        -1.68554068e-01_dt, -8.25520217e-01_dt, -2.27432084e+00_dt, -9.89254355e-01_dt, -1.56061053e+00_dt, 1.32943869e+00_dt,
        -2.40540981e-01_dt, 1.82324803e+00_dt, 1.22914481e+00_dt, 3.93661976e-01_dt, -3.45429778e+00_dt, -3.74935627e-01_dt,
        2.14440107e+00_dt, -2.57402086e+00_dt, -2.31083727e+00_dt, 1.63542128e+00_dt, 1.42455649e+00_dt, -5.72202981e-01_dt,
        -7.52898216e-01_dt, 1.11538112e+00_dt, 2.50962377e+00_dt, -2.57560778e+00_dt, -5.40989876e-01_dt, 3.64450693e+00_dt,
        5.14214325e+00_dt, -1.40772665e+00_dt, -1.17870414e+00_dt, -9.35788572e-01_dt, 1.91784728e+00_dt, -3.43893766e+00_dt } };
    const raul::Tensor hh_weights_grad_golden[] { { 3.79499071e-03_dt, -8.80209263e-03_dt, -1.30117619e-02_dt, 5.79346623e-03_dt,
        1.08418055e-02_dt, -1.08007807e-02_dt, -2.22886261e-02_dt, 8.73351097e-03_dt, -7.37401703e-03_dt, 2.68841293e-02_dt,
        3.85046378e-03_dt, 3.33169401e-02_dt, -1.85050201e-02_dt, 1.46386158e-02_dt, -6.30537644e-02_dt, -1.57447029e-02_dt,
        1.14286914e-02_dt, -2.16874294e-04_dt, 1.29081495e-03_dt, -4.75544259e-02_dt, 3.06097232e-03_dt, -3.85432765e-02_dt,
        -1.34961978e-02_dt, -4.57143644e-03_dt, 5.61441779e-02_dt, 1.46982074e-01_dt, 8.33640173e-02_dt, -4.12352197e-02_dt,
        3.13674137e-02_dt, 4.26567793e-02_dt, 6.43069521e-02_dt, 2.35203020e-02_dt, 1.03892036e-01_dt, -6.52568787e-02_dt,
        4.15292196e-02_dt, -9.67449993e-02_dt, 1.25450827e-02_dt, 2.13597178e-01_dt, -7.01165348e-02_dt, -1.32469878e-01_dt,
        1.06343068e-01_dt, 6.08820170e-02_dt, -1.83758527e-01_dt, 1.53454050e-01_dt, -1.13714397e-01_dt, 9.07379612e-02_dt,
        3.26490030e-02_dt, -3.02950572e-03_dt, -9.16632358e-03_dt, 1.91837788e-01_dt, 6.26117736e-03_dt, 6.47648036e-01_dt,
        -1.05257429e-01_dt, 1.44780025e-01_dt, -1.00078046e+00_dt, 1.42739117e-01_dt, 5.39625704e-01_dt, -2.98494697e-01_dt,
        2.24041328e-01_dt, -8.16666842e-01_dt, 3.60179096e-02_dt, 4.87171352e-01_dt, -1.04639463e-01_dt, 1.06505051e-01_dt,
        -7.19190836e-01_dt, 1.28398418e-01_dt, 8.42316806e-01_dt, -1.79315388e-01_dt, 1.90877199e-01_dt, -1.03166974e+00_dt,
        1.40393525e-03_dt, 1.02372003e+00_dt, -1.66061044e-01_dt, 1.82061747e-01_dt, -1.64755750e+00_dt },
        { -2.06336239e-03_dt, -1.21392086e-02_dt, -1.58382282e-02_dt, 1.17725534e-02_dt, 1.08888168e-02_dt, -6.10688478e-02_dt,
        -6.53812895e-05_dt, 6.72151595e-02_dt, -2.57743038e-02_dt, 6.50642347e-03_dt, -7.05780461e-02_dt, 4.66251047e-03_dt,
        5.97687289e-02_dt, -9.72759561e-04_dt, -1.32295359e-02_dt, -1.64037853e-01_dt, -1.68339945e-02_dt, 1.10417724e-01_dt,
        -2.10126489e-02_dt, -4.57031690e-02_dt, -8.63897614e-04_dt, -1.75186172e-02_dt, -1.73618160e-02_dt, 2.01846822e-04_dt,
        2.34027933e-02_dt, 2.42396265e-01_dt, 1.29013397e-02_dt, -1.15447998e-01_dt, 3.71865220e-02_dt, 1.45322025e-01_dt,
        7.34156296e-02_dt, 1.45307451e-01_dt, 1.03126988e-01_dt, -3.61430869e-02_dt, -9.70488936e-02_dt, -1.54037058e-01_dt,
        9.09431279e-02_dt, 2.58155137e-01_dt, -7.21935108e-02_dt, -1.34414375e-01_dt, 2.54395336e-01_dt, -5.64202443e-02_dt,
        -3.37104827e-01_dt, 1.88293323e-01_dt, 1.29346848e-02_dt, 1.04515925e-01_dt, -2.76436862e-02_dt, 9.16519091e-02_dt,
        -4.92685437e-02_dt, 2.79361308e-01_dt, 4.32625651e-01_dt, 2.29690541e-02_dt, -5.05983591e-01_dt, 1.36502072e-01_dt,
        -1.73656315e-01_dt, 5.08324981e-01_dt, -5.81545755e-02_dt, -6.43786430e-01_dt, 2.12467715e-01_dt, -6.83289692e-02_dt,
        4.38951969e-01_dt, -5.58404773e-02_dt, -5.14174104e-01_dt, 1.26409635e-01_dt, -3.45334411e-02_dt, 6.83218241e-01_dt,
        -9.72456858e-03_dt, -7.07503319e-01_dt, 1.82228670e-01_dt, 1.29231066e-02_dt, 4.83273029e-01_dt, 1.34956405e-01_dt,
        -5.23505449e-01_dt, 1.13200955e-01_dt, -3.39472622e-01_dt },
        { 0.02239588_dt, -0.00480289_dt, -0.02123515_dt, 0.02128624_dt, 0.00417687_dt, -0.03775726_dt, -0.00592602_dt, 0.01690736_dt,
        -0.02578502_dt, -0.00191308_dt, 0.02369287_dt, -0.00360171_dt, -0.02650358_dt, 0.03255519_dt, -0.01456579_dt, -0.06765833_dt,
        -0.01420606_dt, 0.00493713_dt, -0.02830710_dt, -0.03792100_dt, 0.00303537_dt, -0.01701461_dt, -0.02536058_dt, -0.00497453_dt,
        0.01346707_dt, 0.21275178_dt, 0.02542560_dt, -0.06009022_dt, 0.04251155_dt, 0.14398302_dt, 0.05948129_dt, 0.17038615_dt,
        0.11483939_dt, -0.04301221_dt, -0.09474866_dt, -0.16747689_dt, 0.11515789_dt, 0.23117679_dt, -0.10968873_dt, -0.15126731_dt,
        0.22987351_dt, -0.04751486_dt, -0.20893422_dt, 0.21182910_dt, 0.02880069_dt, 0.07861820_dt, -0.02197762_dt, 0.06071748_dt,
        -0.06976791_dt, 0.26163822_dt, 0.63474494_dt, 0.11256830_dt, -0.09814491_dt, 0.61247039_dt, -0.09848526_dt, 0.76436496_dt,
        0.00617715_dt, -0.34061840_dt, 0.74200177_dt, -0.02422517_dt, 0.57431597_dt, 0.02574631_dt, -0.12581050_dt, 0.56141913_dt,
        -0.02327836_dt, 0.82180691_dt, 0.07936050_dt, -0.18451987_dt, 0.74644482_dt, 0.07475318_dt, 0.58632004_dt, 0.21516977_dt,
        0.01163810_dt, 0.60022622_dt, -0.21872407_dt } };
    const raul::Tensor ih_biases_grad_golden{ -0.02127843_dt, -0.09590528_dt, 0.19884211_dt, 0.09993608_dt, -0.15076885_dt, 0.34202248_dt,
        -0.95102441_dt, -0.38943574_dt, 0.29840440_dt, 0.80678231_dt, 8.05632210_dt, 8.07075119_dt, 6.89470148_dt, 11.67421913_dt, 9.89730263_dt };
    const raul::Tensor hh_biases_grad_golden{ -0.14693266_dt, -0.29261622_dt, -0.66793716_dt, -1.05724382_dt, -0.13950041_dt, 0.27531958_dt,
        -0.12721337_dt, -0.24831492_dt, 0.42653963_dt, 0.26813397_dt, 3.69462466_dt, 3.76037288_dt, 3.68750381_dt, 5.87739801_dt, 4.98879528_dt };

    // Initialization
    for (size_t k = 2; k < 3; ++k)
    {
        for (size_t q = 0; q < 2; ++q)
        {
            MANAGERS_DEFINE
            NETWORK_PARAMS_DEFINE(networkParameters);

            work.add<raul::DataLayer>("data1", raul::DataParams{ { "in" }, sequence_length, 1, input_size });

            // Network
            bool useFusion = (q == 1 ? true : false);
            bool useBiasForInput = (k == 0);
            bool useBiasForHidden = (k == 1);
            const auto params = raul::GRUParams{ { "in" }, { "out" }, hidden_size, false, useBiasForInput, useBiasForHidden, false, useFusion };
            raul::GRULayer("gru", params, networkParameters);
            TENSORS_CREATE(batch_size)

            const auto ihBiasesName = Name("gru::cell::linear_ih::Biases");
            const auto ihWeightsName = Name("gru::cell::linear_ih::Weights");
            const auto hhBiasesName = raul::Name("gru::cell::linear_hh::Biases");
            const auto hhWeightsName = Name("gru::cell::linear_hh::Weights");
            
            memory_manager["in"] = TORANGE(input_init);
            memory_manager[ihWeightsName] = TORANGE(ih_weights);
            memory_manager[hhWeightsName] = TORANGE(hh_weights);
            if (useBiasForInput)
            {
                memory_manager[ihBiasesName] = TORANGE(ih_biases);
            }
            if (useBiasForHidden)
            {
                memory_manager[hhBiasesName] = TORANGE(hh_biases);
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
                const auto golden_val = output_golden[k][i];
                ASSERT_TRUE(tools::expect_near_relative(val, golden_val, eps_rel)) << "at " << i << ", expected: " << golden_val << ", got: " << val;
            }

            for (size_t i = 0; i < hiddenTensor.size(); ++i)
            {
                const auto val = hiddenTensor[i];
                const auto golden_val = hidden_golden[k][i];
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
                const auto golden_val = inputs_grad_golden[k][i];
                ASSERT_TRUE(tools::expect_near_relative(val, golden_val, eps_rel)) << "at " << i << ", expected: " << golden_val << ", got: " << val;
            }

            const auto& hidden_grad = memory_manager[raul::Name("gru::hidden_state").grad()];
            const auto& hidden = memory_manager["gru::hidden_state"];

            EXPECT_EQ(hidden.size(), hidden_grad.size());

            for (size_t i = 0; i < hidden_grad.size(); ++i)
            {
                const auto val = hidden_grad[i];
                const auto golden_val = hidden_grad_golden[k][i];
                ASSERT_TRUE(tools::expect_near_relative(val, golden_val, eps_rel)) << "at " << i << ", expected: " << golden_val << ", got: " << val;
            }
            
            const auto gradWeightsIH = memory_manager[ihWeightsName.grad()];
            const auto gradWeightsHH = memory_manager[hhWeightsName.grad()];

            EXPECT_EQ(ih_weights_grad_golden[k].size(), gradWeightsIH.size());

            for (size_t i = 0; i < gradWeightsIH.size(); ++i)
            {
                const auto val = gradWeightsIH[i];
                const auto golden_val = ih_weights_grad_golden[k][i];
                ASSERT_TRUE(tools::expect_near_relative(val, golden_val, eps_rel)) << "at " << i << ", expected: " << golden_val << ", got: " << val;
            }

            EXPECT_EQ(hh_weights_grad_golden[k].size(), gradWeightsHH.size());

            for (size_t i = 0; i < gradWeightsHH.size(); ++i)
            {
                const auto val = gradWeightsHH[i];
                const auto golden_val = hh_weights_grad_golden[k][i];
                ASSERT_TRUE(tools::expect_near_relative(val, golden_val, eps_rel)) << "at " << i << ", expected: " << golden_val << ", got: " << val;
            }

            const auto ihBiasesGradName = ihBiasesName.grad();
            if (useBiasForInput)
            {
                const auto gradBiasesIH = memory_manager[ihBiasesGradName];
                EXPECT_TRUE(memory_manager.tensorExists(ihBiasesGradName));
                EXPECT_EQ(ih_biases_grad_golden.size(), gradBiasesIH.size());

                for (size_t i = 0; i < gradBiasesIH.size(); ++i)
                {
                    const auto val = gradBiasesIH[i];
                    const auto golden_val = ih_biases_grad_golden[i];
                    ASSERT_TRUE(tools::expect_near_relative(val, golden_val, eps_rel)) << "at " << i << ", expected: " << golden_val << ", got: " << val;
                }
            }
            else
            {
                EXPECT_TRUE(!memory_manager.tensorExists(ihBiasesGradName));
            }

            const auto hhBiasesGradName = hhBiasesName.grad();
            if (useBiasForHidden)
            {
                const auto gradBiasesHH = memory_manager[hhBiasesGradName];
                EXPECT_TRUE(memory_manager.tensorExists(hhBiasesGradName));
                EXPECT_EQ(hh_biases_grad_golden.size(), gradBiasesHH.size());

                for (size_t i = 0; i < gradBiasesHH.size(); ++i)
                {
                    const auto val = gradBiasesHH[i];
                    const auto golden_val = hh_biases_grad_golden[i];
                    ASSERT_TRUE(tools::expect_near_relative(val, golden_val, eps_rel)) << "at " << i << ", expected: " << golden_val << ", got: " << val;
                }
            }
            else
            {
                EXPECT_TRUE(!memory_manager.tensorExists(hhBiasesGradName));
            }
        }
    }
}

}