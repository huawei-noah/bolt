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
#include <training/base/layers/composite/AdditiveAttentionLayer.h>
#include <training/compiler/Layers.h>
#include <training/compiler/Workflow.h>

namespace UT
{

TEST(TestLayerAdditiveAttention, InputNumExceedsUnit)
{
    PROFILE_TEST

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    // Apply function
    ASSERT_THROW(raul::AdditiveAttentionLayer("attn", raul::DropoutParams{ { "query", "value", "key", "mask1", "mask2" }, { "attention" }, 1.0f }, networkParameters), raul::Exception);
}

// See bahdanau_attention.py
TEST(TestLayerAdditiveAttention, ForwardUnit)
{
    PROFILE_TEST
    // Test parameters
    const raul::dtype eps = TODTYPE(1e-6);
    const raul::dtype proba = 0.0_dt;
    const size_t depth = 1;
    const size_t height = 2;
    const size_t width = 3;

    const raul::Tensor query{ 0.6645621_dt,  0.44100678_dt, 0.3528825_dt, 0.46448255_dt, 0.03366041_dt, 0.68467236_dt, 0.74011743_dt, 0.8724445_dt,  0.22632635_dt,
                              0.22319686_dt, 0.3103881_dt,  0.7223358_dt, 0.13318717_dt, 0.5480639_dt,  0.5746088_dt,  0.8996835_dt,  0.00946367_dt, 0.5212307_dt };

    const raul::Tensor value{ 0.68789124_dt, 0.48447883_dt, 0.9309944_dt,  0.252187_dt,   0.73115396_dt, 0.89256823_dt,
                              0.94674826_dt, 0.7493341_dt,  0.34925628_dt, 0.54718256_dt, 0.26160395_dt, 0.69734323_dt };

    const raul::Tensor key{
        0.7413678_dt, 0.62854624_dt, 0.01738465_dt, 0.3431449_dt, 0.51063764_dt, 0.3777541_dt, 0.07321596_dt, 0.02137029_dt, 0.2871771_dt, 0.4710616_dt, 0.6936141_dt, 0.07321334_dt
    };

    const raul::Tensor realOutput{ 0.45930246_dt, 0.613895_dt,   0.91083443_dt, 0.47764212_dt, 0.603512_dt,  0.91245186_dt, 0.45157468_dt, 0.6182701_dt, 0.91015285_dt,
                                   0.68406427_dt, 0.42868868_dt, 0.578097_dt,   0.69424564_dt, 0.4411166_dt, 0.5692273_dt,  0.69163543_dt, 0.4379304_dt, 0.57150126_dt };

    const auto expectedShape = yato::dims(2, 1, 3, 3);

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    // Topology
    work.add<raul::DataLayer>("data_q", raul::DataParams{ { "query" }, depth, width, width });
    work.add<raul::DataLayer>("data_vk", raul::DataParams{ { "value", "key" }, depth, height, width });
    raul::AdditiveAttentionLayer battn("battn", raul::DropoutParams{ { "query", "value", "key" }, { "attn" }, proba }, networkParameters);

    TENSORS_CREATE(2);
    memory_manager["query"] = TORANGE(query);
    memory_manager["value"] = TORANGE(value);
    memory_manager["key"] = TORANGE(key);

    work.forwardPassTraining();

    // Checks
    const auto& output = memory_manager["attn"];
    EXPECT_EQ(output.getShape(), expectedShape);
    for (size_t i = 0; i < output.size(); ++i)
    {
        ASSERT_TRUE(tools::expect_near_relative(output[i], realOutput[i], eps));
    }
}

TEST(TestLayerAdditiveAttention, BackwardUnit)
{
    PROFILE_TEST

    // Test parameters
    const raul::dtype eps = TODTYPE(1e-5);
    const raul::dtype proba = 0.0f;
    const size_t depth = 1;
    const size_t height = 2;
    const size_t width = 3;

    const raul::Tensor query{ 0.6645621_dt,  0.44100678_dt, 0.3528825_dt, 0.46448255_dt, 0.03366041_dt, 0.68467236_dt, 0.74011743_dt, 0.8724445_dt,  0.22632635_dt,
                              0.22319686_dt, 0.3103881_dt,  0.7223358_dt, 0.13318717_dt, 0.5480639_dt,  0.5746088_dt,  0.8996835_dt,  0.00946367_dt, 0.5212307_dt };

    const raul::Tensor value{ 0.68789124_dt, 0.48447883_dt, 0.9309944_dt,  0.252187_dt,   0.73115396_dt, 0.89256823_dt,
                              0.94674826_dt, 0.7493341_dt,  0.34925628_dt, 0.54718256_dt, 0.26160395_dt, 0.69734323_dt };

    const raul::Tensor key{
        0.7413678_dt, 0.62854624_dt, 0.01738465_dt, 0.3431449_dt, 0.51063764_dt, 0.3777541_dt, 0.07321596_dt, 0.02137029_dt, 0.2871771_dt, 0.4710616_dt, 0.6936141_dt, 0.07321334_dt
    };

    const raul::Tensor realQueryGrad{ -0.0114115_dt, -0.00423195_dt, 0.0149313_dt,   -0.0142751_dt, -0.00510888_dt, 0.01430582_dt,  -0.01027854_dt, -0.00238969_dt, 0.01323442_dt,
                                      0.03374725_dt, 0.05830509_dt,  -0.01802726_dt, 0.03138524_dt, 0.0564916_dt,   -0.02018242_dt, 0.02612901_dt,  0.04563681_dt,  -0.02028692_dt };
    const raul::Tensor realValueGrad{
        1.4504294_dt, 1.4504294_dt, 1.4504294_dt, 1.5495706_dt, 1.5495706_dt, 1.5495706_dt, 1.072158_dt, 1.072158_dt, 1.072158_dt, 1.927842_dt, 1.927842_dt, 1.927842_dt
    };
    const raul::Tensor realKeyGrad{ 0.0398373_dt, 0.06927019_dt, 0.13879874_dt, -0.07580243_dt, -0.08100071_dt, -0.0963272_dt,
                                    0.2860422_dt, 0.32549983_dt, 0.18352844_dt, -0.19478072_dt, -0.16506633_dt, -0.24202503_dt };

    const auto expectedShape = yato::dims(2, 1, 3, 3);

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    work.add<raul::DataLayer>("data_q", raul::DataParams{ { "query" }, depth, width, width });
    work.add<raul::DataLayer>("data_vk", raul::DataParams{ { "value", "key" }, depth, height, width });

    // Apply function
    raul::AdditiveAttentionLayer battn("battn", raul::DropoutParams{ { "query", "value", "key" }, { "attn" }, proba }, networkParameters);

    TENSORS_CREATE(2);

    ASSERT_EQ(memory_manager[raul::Name("attn").grad()].getShape(), expectedShape);
    std::fill(memory_manager[raul::Name("attn").grad()].begin(), memory_manager[raul::Name("attn").grad()].end(), 1_dt);

    memory_manager["query"] = TORANGE(query);
    memory_manager["value"] = TORANGE(value);
    memory_manager["key"] = TORANGE(key);

    work.forwardPassTraining();
    work.backwardPassTraining();

    // Checks
    const auto& queryNabla = memory_manager[raul::Name("query").grad()];
    const auto& valueNabla = memory_manager[raul::Name("value").grad()];
    const auto& keyNabla = memory_manager[raul::Name("key").grad()];
    for (size_t i = 0; i < queryNabla.size(); ++i)
    {
        ASSERT_TRUE(tools::expect_near_relative(queryNabla[i], realQueryGrad[i], eps));
    }
    for (size_t i = 0; i < valueNabla.size(); ++i)
    {
        ASSERT_TRUE(tools::expect_near_relative(valueNabla[i], realValueGrad[i], eps));
    }
    for (size_t i = 0; i < keyNabla.size(); ++i)
    {
        ASSERT_TRUE(tools::expect_near_relative(keyNabla[i], realKeyGrad[i], eps));
    }
}

}