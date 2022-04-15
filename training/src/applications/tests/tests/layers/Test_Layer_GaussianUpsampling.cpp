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

#include <training/common/Common.h>
#include <training/common/MemoryManager.h>
#include <training/layers/basic/DataLayer.h>
#include <training/layers/basic/TensorLayer.h>
#include <training/layers/composite/GaussianUpsamplingLayer.h>
#include <training/layers/composite/tacotron/GaussianUpsamplingDistributionLayer.h>
#include <training/network/Workflow.h>

#include "../topologies/TacotronTestTools.h"

namespace UT
{

namespace
{

raul::dtype golden_gaussian_upsampling_distribution_layer(raul::dtype val, raul::dtype loc, raul::dtype scale)
{
    return static_cast<raul::dtype>(std::exp(std::log(std::exp(-0.5_dt * (val - loc) * (val - loc) / scale / scale) / std::sqrt(2.0_dt * RAUL_PI * scale * scale))));
}

raul::dtype golden_gaussian_upsampling_distribution_layer_loc_grad(raul::dtype del, raul::dtype val, raul::dtype loc, raul::dtype scale)
{
    return static_cast<dtype>(del * 0.398942_dt * (val - loc) * std::exp(-0.5_dt * (val - loc) * (val - loc) / scale / scale) / scale / scale / scale);
}

raul::dtype golden_gaussian_upsampling_distribution_layer_scale_grad(raul::dtype del, raul::dtype val, raul::dtype loc, raul::dtype scale)
{
    return static_cast<dtype>(del * (0.398942_dt * (val - loc) * (val - loc) / scale / scale - 1.0_dt / std::sqrt(2.0_dt * RAUL_PI)) * std::exp(-0.5_dt * (val - loc) * (val - loc) / scale / scale) /
                              scale / scale);
}

}

// see gaussian_upsampling.py
TEST(TestLayerGaussianUpsampling, SimpleFixedMelLenUnit)
{
    constexpr size_t BATCH = 2;
    constexpr size_t SEQLEN = 5;
    constexpr size_t INPUTDIM = 4;
    constexpr raul::dtype eps = 1.0e-4_dt;

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    // Inputs
    const raul::Tensor input{ 0.29197514_dt, 0.20656645_dt, 0.53539073_dt, 0.5612575_dt,  0.4166745_dt,  0.80782795_dt, 0.4932251_dt,  0.99812925_dt, 0.69673514_dt, 0.1253736_dt,
                              0.7098167_dt,  0.6624156_dt,  0.57225657_dt, 0.36475348_dt, 0.42051828_dt, 0.630057_dt,   0.913813_dt,   0.6616472_dt,  0.83347356_dt, 0.08395803_dt,
                              0.2797594_dt,  0.0155232_dt,  0.72637355_dt, 0.7655387_dt,  0.6798667_dt,  0.53272796_dt, 0.7565141_dt,  0.04742193_dt, 0.05037141_dt, 0.75174344_dt,
                              0.1727128_dt,  0.3119352_dt,  0.29137385_dt, 0.10051239_dt, 0.16567075_dt, 0.7696651_dt,  0.58567977_dt, 0.98200965_dt, 0.9148327_dt,  0.14166534_dt };
    const raul::Tensor durations{ 0.5554141_dt, 0.22129297_dt, 0.8649249_dt, 0.77728355_dt, 0.6451167_dt, 0.53036225_dt, 0.01444101_dt, 0.87350917_dt, 0.4697218_dt, 0.38672888_dt };
    const raul::Tensor ranges{ 0.1952138_dt, 0.7401732_dt, 0.4878018_dt, 0.8753203_dt, 0.4071133_dt, 0.01454818_dt, 0.7095418_dt, 0.36551023_dt, 0.5808557_dt, 0.9008391_dt };
    constexpr size_t mel = 3;

    // Out
    const raul::Tensor realOut{ 0.58356017_dt, 0.3899107_dt,  0.592413_dt,  0.7688811_dt, 0.6492674_dt,  0.4162626_dt,  0.5737605_dt,  0.5725301_dt,
                                0.83176386_dt, 0.59206206_dt, 0.7353294_dt, 0.2157647_dt, 0.2807103_dt,  0.61638457_dt, 0.37164274_dt, 0.31668964_dt,
                                0.42751023_dt, 0.4917453_dt,  0.5006382_dt, 0.465866_dt,  0.54192567_dt, 0.8477817_dt,  0.8017316_dt,  0.23545113_dt };

    // Gradients
    const raul::Tensor realInGrad{ 0.00148816_dt, 0.00148816_dt, 0.00148816_dt, 0.00148816_dt, 0.44603357_dt, 0.44603357_dt, 0.44603357_dt, 0.44603357_dt, 0.7377904_dt,  0.7377904_dt,
                                   0.7377904_dt,  0.7377904_dt,  0.8607639_dt,  0.8607639_dt,  0.8607639_dt,  0.8607639_dt,  0.95389736_dt, 0.95389736_dt, 0.95389736_dt, 0.95389736_dt,
                                   0._dt,         0._dt,         0._dt,         0._dt,         0.27913648_dt, 0.27913648_dt, 0.27913648_dt, 0.27913648_dt, 0.5332824_dt,  0.5332824_dt,
                                   0.5332824_dt,  0.5332824_dt,  0.841596_dt,   0.841596_dt,   0.841596_dt,   0.841596_dt,   1.3459393_dt,  1.3459393_dt,  1.3459393_dt,  1.3459393_dt };
    const raul::Tensor realRangesGrad{ -0.07153425_dt, 0.04617101_dt, 0.10229341_dt, 0.07298546_dt, 0.17718953_dt, 0._dt, -0.06634991_dt, 0.1923635_dt, -0.94127846_dt, -0.2654451_dt };
    const raul::Tensor realDurationsGrad{ 0.11475088_dt, 0.01661567_dt, -0.11619779_dt, -0.12143932_dt, -0.05121587_dt, -0.8671924_dt, -0.9167041_dt, -0.9091654_dt, -0.4165768_dt, 0.00948068_dt };

    // Inputs
    work.add<raul::DataLayer>("data_input", raul::DataParams{ { "input" }, 1U, SEQLEN, INPUTDIM });
    work.add<raul::DataLayer>("data_durations_and_ranges", raul::DataParams{ { "durations", "ranges" }, 1U, SEQLEN, 1U });
    raul::GaussianUpsamplingLayer("gaussian_Upsampling", raul::GaussianUpsamplingParams{ { "input", "durations", "ranges" }, { "u" }, mel }, networkParameters);
    TENSORS_CREATE(BATCH);
    memory_manager["input"] = TORANGE(input);
    memory_manager["durations"] = TORANGE(durations);
    memory_manager["ranges"] = TORANGE(ranges);
    memory_manager["uGradient"] = 1.0_dt;

    ASSERT_NO_THROW(work.forwardPassTraining());

    const raul::Tensor& out = memory_manager["u"];
    EXPECT_EQ(out.size(), realOut.size());
    for (size_t i = 0; i < out.size(); ++i)
    {
        ASSERT_TRUE(tools::expect_near_relative(out[i], realOut[i], eps));
    }

    ASSERT_NO_THROW(work.backwardPassTraining());

    const raul::Tensor& inGrad = memory_manager["inputGradient"];
    EXPECT_EQ(realInGrad.size(), inGrad.size());
    for (size_t i = 0; i < inGrad.size(); ++i)
    {
        ASSERT_TRUE(tools::expect_near_relative(inGrad[i], realInGrad[i], eps));
    }

    const raul::Tensor& rangesGrad = memory_manager["rangesGradient"];
    EXPECT_EQ(realRangesGrad.size(), rangesGrad.size());
    for (size_t i = 0; i < rangesGrad.size(); ++i)
    {
        ASSERT_TRUE(tools::expect_near_relative(rangesGrad[i], realRangesGrad[i], eps));
    }

    const raul::Tensor& durationsGrad = memory_manager["durationsGradient"];
    EXPECT_EQ(realDurationsGrad.size(), durationsGrad.size());
    for (size_t i = 0; i < durationsGrad.size(); ++i)
    {
        ASSERT_TRUE(tools::expect_near_relative(durationsGrad[i], realDurationsGrad[i], eps));
    }
}

// see gaussian_upsampling.py
TEST(TestLayerGaussianUpsampling, TacotronDataUnit)
{
    constexpr size_t BATCH = 12;
    constexpr size_t SEQLEN = 86;
    constexpr size_t INPUTDIM = 768;
    constexpr size_t mel = 222;
    constexpr raul::dtype eps = 1.0e-3_dt;

    auto testPath = tools::getTestAssetsDir() / "gaussian_upsampling";

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    // Inputs
    work.add<raul::DataLayer>("data_input", raul::DataParams{ { "input" }, 1U, SEQLEN, INPUTDIM });
    work.add<raul::DataLayer>("data_durations_and_ranges", raul::DataParams{ { "durations", "ranges" }, 1U, SEQLEN, 1U });
    work.add<raul::DataLayer>("data_real_out", raul::DataParams{ { "realOut" }, 1U, mel, INPUTDIM });
    raul::GaussianUpsamplingLayer("gaussian_Upsampling", raul::GaussianUpsamplingParams{ { "input", "durations", "ranges" }, { "u" }, mel }, networkParameters);

    TENSORS_CREATE(BATCH);
    EXPECT_TRUE(loadTFData(testPath / "upsampling_x.data", memory_manager["input"]));
    EXPECT_TRUE(loadTFData(testPath / "upsampling_durations.data", memory_manager["durations"]));
    EXPECT_TRUE(loadTFData(testPath / "upsampling_ranges.data", memory_manager["ranges"]));
    EXPECT_TRUE(loadTFData(testPath / "upsampling_u.data", memory_manager["realOut"]));

    ASSERT_NO_THROW(work.forwardPassTraining());

    const raul::Tensor& out = memory_manager["u"];
    const raul::Tensor& realOut = memory_manager["realOut"];
    EXPECT_EQ(out.size(), realOut.size());
    for (size_t i = 0; i < out.size(); ++i)
    {
        ASSERT_TRUE(tools::expect_near_relative(out[i], realOut[i], eps) || (out[i] == 0 && realOut[i] < eps) || (out[i] < eps && realOut[i] == 0) || (out[i] < eps && realOut[i] < eps));
    }

    ASSERT_NO_THROW(work.backwardPassTraining());
}

struct TestLayerGaussianUpsamplingDistribution : public testing::TestWithParam<tuple<size_t, size_t, size_t>>
{
    const size_t BATCH = get<0>(GetParam());
    const size_t VALUES_LEN = get<1>(GetParam());
    const size_t HEIGHT = get<2>(GetParam());
    const raul::dtype eps = 1.0e-4_dt;
    const std::pair<dtype, dtype> range = std::make_pair(1.0_dt, 100_dt);
};

TEST_P(TestLayerGaussianUpsamplingDistribution, GpuUnit)
{
    PROFILE_TEST

    GPU_ONLY_TEST

    raul::WorkflowEager work{ raul::CompressionMode::NONE, raul::CalculationMode::DETERMINISTIC, raul::AllocationMode::STANDARD, raul::ExecutionTarget::GPU };
    work.add<raul::TensorLayer>("data_values", raul::TensorParams{ { "values" }, VALUES_LEN, 1U, 1U, 1U });
    work.add<raul::DataLayer>("data_loc_and_scale", raul::DataParams{ { "loc", "scale" }, 1U, HEIGHT, 1U });
    work.add<raul::GaussianUpsamplingDistributionLayer>("gaussian_Upsampling_distribution", raul::BasicParams{ { "values", "loc", "scale" }, { "out" } });
    TENSORS_CREATE(BATCH);

    auto& memory_manager = work.getMemoryManager<MemoryManagerGPU>();
    tools::init_rand_tensor("values", range, memory_manager);
    tools::init_rand_tensor("loc", range, memory_manager);
    tools::init_rand_tensor("scale", range, memory_manager);
    tools::init_rand_tensor(raul::Name("out").grad(), range, memory_manager);

    work.forwardPassTraining();

    const raul::Tensor& out = memory_manager["out"];
    const raul::Tensor& vals = memory_manager["values"];
    const raul::Tensor& loc = memory_manager["loc"];
    const raul::Tensor& scale = memory_manager["scale"];
    auto out3D = out.reshape(yato::dims(BATCH, VALUES_LEN, HEIGHT));
    auto loc2D = loc.reshape(yato::dims(BATCH, HEIGHT));
    auto scale2D = scale.reshape(yato::dims(BATCH, HEIGHT));
    for (size_t i = 0; i < VALUES_LEN; ++i)
    {
        for (size_t j = 0; j < BATCH; ++j)
        {
            for (size_t k = 0; k < HEIGHT; ++k)
            {
                EXPECT_NEAR(out3D[j][i][k], golden_gaussian_upsampling_distribution_layer(vals[i], loc2D[j][k], scale2D[j][k]), eps);
            }
        }
    }

    work.backwardPassTraining();

    const raul::Tensor& outGrad = memory_manager[raul::Name("out").grad()];
    const raul::Tensor& locGrad = memory_manager[raul::Name("loc").grad()];
    const raul::Tensor& scaleGrad = memory_manager[raul::Name("scale").grad()];
    auto outGrad3D = outGrad.reshape(yato::dims(BATCH, VALUES_LEN, HEIGHT));
    auto locGrad2D = locGrad.reshape(yato::dims(BATCH, HEIGHT));
    auto scaleGrad2D = scaleGrad.reshape(yato::dims(BATCH, HEIGHT));

    for (size_t j = 0; j < BATCH; ++j)
    {
        for (size_t k = 0; k < HEIGHT; ++k)
        {
            auto goldenLocGrad = 0.0_dt;
            auto goldenScaleGrad = 0.0_dt;
            for (size_t i = 0; i < VALUES_LEN; ++i)
            {
                goldenLocGrad += golden_gaussian_upsampling_distribution_layer_loc_grad(outGrad3D[j][i][k], vals[i], loc2D[j][k], scale2D[j][k]);
                goldenScaleGrad += golden_gaussian_upsampling_distribution_layer_scale_grad(outGrad3D[j][i][k], vals[i], loc2D[j][k], scale2D[j][k]);
            }
            EXPECT_NEAR(locGrad2D[j][k], goldenLocGrad, eps);
            EXPECT_NEAR(scaleGrad2D[j][k], goldenScaleGrad, eps);
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
    TestGpu,
    TestLayerGaussianUpsamplingDistribution,
    testing::Values(make_tuple(2, 3, 4), make_tuple(5, 4, 3), make_tuple(101, 1, 1), make_tuple(1, 93, 1), make_tuple(2, 1, 101), make_tuple(2, 1, 1), make_tuple(1, 1, 1), make_tuple(93, 94, 95)));

}