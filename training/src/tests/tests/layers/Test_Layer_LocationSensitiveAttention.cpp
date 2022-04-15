// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <tests/tools/TestTools.h>
#include <tests/tools/callbacks/TensorChecker.h>

#include <training/base/common/Common.h>
#include <training/base/common/MemoryManager.h>
#include <training/base/layers/basic/DataLayer.h>
#include <training/base/layers/basic/ElementWiseMulLayer.h>
#include <training/base/layers/basic/ElementWiseSumLayer.h>
#include <training/base/layers/composite/LocationSensitiveAttentionLayer.h>
#include <training/compiler/Workflow.h>

namespace UT
{

TEST(TestLayerLocationSensitiveAttention, IncorrectParamsUnit)
{
    PROFILE_TEST
    const size_t numUnits = 4;
    const size_t queryDepth = 3;
    const size_t alignmentsSize = 2;
    const size_t anyNumber = 3;

    const raul::Name parent = "parent";

    // Wrong params
    raul::LocationSensitiveAttentionParams incorrectParams[]{
        { raul::LocationSensitiveAttentionParams{
            { "query", "state", "memory" }, { "alignment", "next_state", "max_attn" }, numUnits, raul::LocationSensitiveAttentionParams::hparams{ 1, 2, false, false }, false } },
        { raul::LocationSensitiveAttentionParams{
            { "query", "state", "memory" }, { "alignment", "values", "next_state", "max_attn" }, numUnits, raul::LocationSensitiveAttentionParams::hparams{ 1, 2, false, false }, false } },
        { raul::LocationSensitiveAttentionParams{ { "query", "state", "memory" }, { "alignment" }, numUnits, raul::LocationSensitiveAttentionParams::hparams{ 1, 2, false, false }, true } },
        { raul::LocationSensitiveAttentionParams{
            { "query", "state", "memory" }, { "alignment", "values", "next_state", "max_attn" }, numUnits, raul::LocationSensitiveAttentionParams::hparams{ 1, 2, false, false }, true } },
        { raul::LocationSensitiveAttentionParams{
            { "query", "state", "memory" }, { "alignment", "next_state", "max_attn" }, parent, numUnits, raul::LocationSensitiveAttentionParams::hparams{ 1, 2, false, false }, false } },
        { raul::LocationSensitiveAttentionParams{
            { "query", "state", "memory" }, { "alignment", "values", "next_state", "max_attn" }, parent, numUnits, raul::LocationSensitiveAttentionParams::hparams{ 1, 2, false, false }, false } },
        { raul::LocationSensitiveAttentionParams{ { "query", "state", "memory" }, { "alignment" }, parent, numUnits, raul::LocationSensitiveAttentionParams::hparams{ 1, 2, false, false }, true } },
        { raul::LocationSensitiveAttentionParams{
            { "query", "state", "memory" }, { "alignment", "values", "next_state", "max_attn" }, parent, numUnits, raul::LocationSensitiveAttentionParams::hparams{ 1, 2, false, false }, true } },
        { raul::LocationSensitiveAttentionParams{
            { "query", "state", "memory", "memory_seq_length" }, { "alignment" }, numUnits, raul::LocationSensitiveAttentionParams::hparams{ 1, 2, false, false }, false } },
        { raul::LocationSensitiveAttentionParams{ { "query", "state", "memory", "memory_seq_length" },
                                                  { "alignment", "values", "next_state", "max_attn" },
                                                  numUnits,
                                                  raul::LocationSensitiveAttentionParams::hparams{ 1, 2, false, false },
                                                  false } },
        { raul::LocationSensitiveAttentionParams{ { "query", "state", "memory", "memory_seq_length" },
                                                  { "alignment", "values", "next_state" },
                                                  parent,
                                                  numUnits,
                                                  raul::LocationSensitiveAttentionParams::hparams{ 1, 2, false, false },
                                                  false } },
        { raul::LocationSensitiveAttentionParams{ { "query", "state", "memory", "memory_seq_length" },
                                                  { "alignment", "values", "next_state", "max_attn" },
                                                  parent,
                                                  numUnits,
                                                  raul::LocationSensitiveAttentionParams::hparams{ 1, 2, false, false },
                                                  false } },
        { raul::LocationSensitiveAttentionParams{
            { "query", "state", "memory", "memory_seq_length" }, { "alignment" }, numUnits, raul::LocationSensitiveAttentionParams::hparams{ 1, 2, false, false }, true } },
        { raul::LocationSensitiveAttentionParams{
            { "query", "state", "memory", "memory_seq_length" }, { "alignment", "values" }, numUnits, raul::LocationSensitiveAttentionParams::hparams{ 1, 2, false, false }, true } },
        { raul::LocationSensitiveAttentionParams{
            { "query", "state", "memory", "memory_seq_length" }, { "alignment" }, parent, numUnits, raul::LocationSensitiveAttentionParams::hparams{ 1, 2, false, false }, true } },
        { raul::LocationSensitiveAttentionParams{ { "query", "state", "memory", "memory_seq_length" },
                                                  { "alignment", "values", "next_state", "max_attn" },
                                                  parent,
                                                  numUnits,
                                                  raul::LocationSensitiveAttentionParams::hparams{ 1, 2, false, false },
                                                  true } }
    };

    for (size_t i = 0; i < std::size(incorrectParams); ++i)
    {
        MANAGERS_DEFINE
        NETWORK_PARAMS_DEFINE(networkParameters);

        // Inputs
        work.add<raul::DataLayer>("data_query", raul::DataParams{ { "query" }, 1u, 1u, queryDepth });
        work.add<raul::DataLayer>("data_state", raul::DataParams{ { "state" }, 1u, 1u, alignmentsSize });
        work.add<raul::DataLayer>("data_memory", raul::DataParams{ { "memory" }, 1u, alignmentsSize, anyNumber });
        if (i > 1)
        {
            work.add<raul::DataLayer>("data_memory_seq_length", raul::DataParams{ { "memory_seq_legnth" }, 1u, 1u, 1u });
        }

        // Layer
        ASSERT_THROW(raul::LocationSensitiveAttentionLayer("attn", incorrectParams[i], networkParameters), raul::Exception);
    }
}

TEST(TestLayerLocationSensitiveAttention, GetTrainableParametersUnit)
{
    PROFILE_TEST
    const size_t numUnits = 4;
    const size_t queryDepth = 3;
    const size_t alignmentsSize = 2;
    const size_t anyNumber = 3;
    const size_t batchSize = 1;

    const size_t goldenTrainableParams = 7u;
    // List of trainable parameters:
    // 1. attention_variable_projection;
    // 2. attention_bias;
    // 3. memory_layer::Weights;
    // 4. query_layer::Weights;
    // 5. location_convolution::Weights;
    // 6. location_convolution::Biases;
    // 7. location_layer::Weights;
    // Optional: transition_agent_layer::Weights, transition_agent_layer::Biases.

    const raul::Tensor query{ 0.01975703_dt, 0.00704217_dt, 0.18987215_dt };
    const raul::Tensor state{ 0.01975703_dt, 0.00704217_dt };
    const raul::Tensor memory{ 0.01975703_dt, 0.00704217_dt, 0.18987215_dt, 0.7772658_dt, 0.41817415_dt, 0.7437942_dt };

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    // Inputs
    work.add<raul::DataLayer>("data_query", raul::DataParams{ { "query" }, 1u, 1u, queryDepth });
    work.add<raul::DataLayer>("data_state", raul::DataParams{ { "state" }, 1u, 1u, alignmentsSize });
    work.add<raul::DataLayer>("data_memory", raul::DataParams{ { "memory" }, 1u, alignmentsSize, anyNumber });

    // Layer
    raul::LocationSensitiveAttentionLayer(
        "attn",
        raul::LocationSensitiveAttentionParams{ { "query", "state", "memory" }, { "alignment" }, numUnits, raul::LocationSensitiveAttentionParams::hparams{ 1u, 2u, false, false }, false },
        networkParameters);

    TENSORS_CREATE(batchSize);

    memory_manager["query"] = TORANGE(query);
    memory_manager["state"] = TORANGE(state);
    memory_manager["memory"] = TORANGE(memory);

    work.printInfo(std::cout);

    // Checks
    EXPECT_EQ(work.getTrainableParameterNames().size(), goldenTrainableParams);
}

TEST(TestLayerLocationSensitiveAttention, NoMaskNotConstrainedNoTransitionAgentNoSmoothingNotCumulativeNoUseForwardUnit)
{
    PROFILE_TEST
    constexpr size_t numUnits = 4;
    constexpr size_t queryDepth = 5;
    constexpr size_t alignmentsSize = 3;
    constexpr size_t anyNumber = 5;
    constexpr size_t batchSize = 2;
    constexpr raul::dtype eps = TODTYPE(1e-5);

    raul::Name name = "attn";

    const raul::Tensor query{ 0.11943877_dt, 0.95280254_dt, 0.9744879_dt, 0.5722927_dt, 0.45100963_dt, 0.8541292_dt, 0.3453902_dt, 0.6201925_dt, 0.06198227_dt, 0.3225391_dt };
    const raul::Tensor state{ 0.15373075_dt, 0.8988131_dt, 0.92626953_dt, 0.9800353_dt, 0.52614915_dt, 0.72589886_dt };
    const raul::Tensor memory{ 0.01975703_dt, 0.00704217_dt, 0.18987215_dt, 0.7772658_dt,  0.41817415_dt, 0.7437942_dt,  0.26365364_dt, 0.4459244_dt,  0.82929873_dt, 0.52497685_dt,
                               0.55597556_dt, 0.19923508_dt, 0.46925998_dt, 0.18594062_dt, 0.23303056_dt, 0.3938471_dt,  0.9660922_dt,  0.36530995_dt, 0.28173566_dt, 0.4888971_dt,
                               0.96301997_dt, 0.45836866_dt, 0.70952535_dt, 0.477888_dt,   0.71620464_dt, 0.12221897_dt, 0.2998824_dt,  0.6689563_dt,  0.06436884_dt, 0.23358119_dt };

    // Real output
    const raul::Tensor realAlignment{ 0.2108239_dt, 0.38865748_dt, 0.40051863_dt, 0.3382085_dt, 0.32790488_dt, 0.33388662_dt };
    const raul::Tensor realMaxAttnIndices{ 2.0_dt, 0.0_dt };

    // Initialization
    raul::Workflow work;
    auto& networkParameters = work.getNetworkParameters();

    // Inputs
    work.add<raul::DataLayer>("data_query", raul::DataParams{ { "query" }, 1u, 1u, queryDepth });
    work.add<raul::DataLayer>("data_state", raul::DataParams{ { "state" }, 1u, 1u, alignmentsSize });
    work.add<raul::DataLayer>("data_memory", raul::DataParams{ { "memory" }, 1u, alignmentsSize, anyNumber });

    // Outputs
    work.add<raul::DataLayer>("output0", raul::DataParams{ { "realAlignment" }, 1u, 1u, alignmentsSize });
    work.add<raul::DataLayer>("output1", raul::DataParams{ { "realMaxAttnIndices" }, 1u, 1u, 1u });

    // Layer
    raul::LocationSensitiveAttentionLayer(
        name,
        raul::LocationSensitiveAttentionParams{ { "query", "state", "memory" }, { "alignment", "max_attn" }, numUnits, raul::LocationSensitiveAttentionParams::hparams{ 3, 3, false, false }, false },
        networkParameters);

    work.preparePipelines();
    work.setBatchSize(batchSize);
    work.prepareMemoryForTraining();

    raul::MemoryManager& memory_manager = work.getMemoryManager();

    memory_manager["query"] = TORANGE(query);
    memory_manager["state"] = TORANGE(state);
    memory_manager["memory"] = TORANGE(memory);
    memory_manager["realAlignment"] = TORANGE(realAlignment);
    memory_manager["realMaxAttnIndices"] = TORANGE(realMaxAttnIndices);

    // For result stability
    memory_manager[name / "attention_variable_projection"] = TORANGE(raul::Tensor({ -0.8425845_dt, -0.7054219_dt, -0.02346194_dt, 0.35520858_dt }));
    memory_manager[name / "query_layer" / "Weights"] =
        TORANGE(raul::Tensor({ 0.09049082_dt, 0.2369746_dt,  -0.04944408_dt, -0.813432_dt,  -0.47131312_dt, -0.45512667_dt, 0.04958135_dt, -0.18497097_dt, 0.5842254_dt, 0.26539183_dt,
                               0.59591985_dt, -0.7929145_dt, 0.63058674_dt,  -0.6403303_dt, -0.61891955_dt, 0.45280218_dt,  0.6099379_dt,  0.2233758_dt,   0.7512772_dt, -0.57287085_dt }));
    memory_manager[name / "location_convolution" / "Weights"] =
        TORANGE(raul::Tensor({ -0.43103278_dt, 0.33965617_dt, -0.0172509_dt, 0.5307831_dt, -0.1313616_dt, -0.68653256_dt, 0.29633683_dt, -0.19019729_dt, 0.11434728_dt }));
    memory_manager[name / "location_layer" / "Weights"] = TORANGE(raul::Tensor(
        { -0.6015324_dt, -0.48898125_dt, 0.18092096_dt, -0.33704627_dt, -0.50806355_dt, -0.01979709_dt, 0.65834594_dt, 0.18336487_dt, -0.7099722_dt, -0.15320659_dt, 0.60897243_dt, -0.16973108_dt }));
    memory_manager[name / "memory_layer" / "Weights"] =
        TORANGE(raul::Tensor({ -0.33970317_dt, -0.13607001_dt, 0.32126713_dt, 0.11799449_dt,  0.67575383_dt, -0.479175_dt,  0.5026809_dt,  -0.6117624_dt, -0.22085667_dt, 0.26396883_dt,
                               0.05779284_dt,  -0.0110634_dt,  0.3426292_dt,  -0.12979311_dt, 0.5445601_dt,  0.10003304_dt, 0.81344163_dt, 0.26522362_dt, 0.2123822_dt,   -0.6793937_dt }));

    tools::callbacks::TensorChecker checker({ { "alignment", "realAlignment" }, { "max_attn", "realMaxAttnIndices" } }, -1_dt, eps);
    networkParameters.mCallback = checker;

    // Forward
    ASSERT_NO_THROW(work.forwardPassTraining());

    // Backward
    ASSERT_NO_THROW(work.backwardPassTraining());
}

TEST(TestLayerLocationSensitiveAttention, NoMaskNotConstrainedNoTransitionAgentNoUseForwardUnit)
{
    PROFILE_TEST

    // Cumulative attention mode, smoothing enabled

    constexpr size_t numUnits = 5;
    constexpr size_t queryDepth = 6;
    constexpr size_t alignmentsSize = 4;
    constexpr size_t anyNumber = 7;
    constexpr size_t batchSize = 3;
    constexpr raul::dtype eps = TODTYPE(1e-5);

    raul::Name name = "attn";

    const raul::Tensor query{ 0.6910331_dt,  0.7831433_dt,  0.05777133_dt, 0.9138534_dt, 0.43685377_dt, 0.8047224_dt,  0.88028264_dt, 0.31608927_dt, 0.57692194_dt,
                              0.64140487_dt, 0.15070891_dt, 0.99759805_dt, 0.5437497_dt, 0.21818483_dt, 0.54885054_dt, 0.89362335_dt, 0.62603235_dt, 0.05133748_dt };
    const raul::Tensor state{ 0.63547623_dt, 0.44589663_dt, 0.6047574_dt,  0.82557225_dt, 0.58478403_dt, 0.04986751_dt,
                              0.9572661_dt,  0.20333457_dt, 0.11299467_dt, 0.05475962_dt, 0.2828188_dt,  0.5192108_dt };
    const raul::Tensor memory{ 0.81269646_dt, 0.07857466_dt, 0.8916855_dt,  0.16925514_dt, 0.06311357_dt, 0.54531074_dt, 0.5037316_dt,  0.9248222_dt,  0.66955376_dt, 0.9281193_dt,  0.12239242_dt,
                               0.8532245_dt,  0.90477383_dt, 0.7104306_dt,  0.40681756_dt, 0.5755513_dt,  0.8547678_dt,  0.59606934_dt, 0.77619946_dt, 0.97301054_dt, 0.06244731_dt, 0.33562684_dt,
                               0.22166848_dt, 0.32035887_dt, 0.03924382_dt, 0.06723011_dt, 0.32712245_dt, 0.49054873_dt, 0.11453211_dt, 0.34396613_dt, 0.52225363_dt, 0.30574834_dt, 0.8817626_dt,
                               0.8017194_dt,  0.9992852_dt,  0.65941477_dt, 0.1272459_dt,  0.19117236_dt, 0.65929854_dt, 0.7614676_dt,  0.75358987_dt, 0.41603255_dt, 0.94846773_dt, 0.8904344_dt,
                               0.91729546_dt, 0.26704276_dt, 0.17427123_dt, 0.04580772_dt, 0.98797727_dt, 0.03881574_dt, 0.22868955_dt, 0.0036062_dt,  0.6006421_dt,  0.25169027_dt, 0.45649374_dt,
                               0.21031535_dt, 0.13384092_dt, 0.610149_dt,   0.7017927_dt,  0.56946445_dt, 0.25802827_dt, 0.09499919_dt, 0.96377003_dt, 0.21196103_dt, 0.94442093_dt, 0.04924846_dt,
                               0.888088_dt,   0.23339641_dt, 0.4439162_dt,  0.13146889_dt, 0.9257786_dt,  0.3446467_dt,  0.9887433_dt,  0.84542334_dt, 0.7688427_dt,  0.2861563_dt,  0.47002888_dt,
                               0.83878493_dt, 0.7776841_dt,  0.35630226_dt, 0.7507192_dt,  0.3887322_dt,  0.3603543_dt,  0.0047611_dt };

    // Real output
    const raul::Tensor realAlignment{ 0.19119766_dt, 0.27237922_dt, 0.30908975_dt, 0.22733343_dt, 0.28871754_dt, 0.28436524_dt,
                                      0.16134506_dt, 0.26557225_dt, 0.24311139_dt, 0.2950311_dt,  0.23211668_dt, 0.22974089_dt };
    const raul::Tensor realNextState{ 0.82667387_dt, 0.71827585_dt, 0.9138472_dt,  1.0529057_dt,  0.87350154_dt, 0.33423275_dt,
                                      1.1186111_dt,  0.46890682_dt, 0.35610604_dt, 0.34979072_dt, 0.5149355_dt,  0.7489517_dt };
    const raul::Tensor realMaxAttnIndices{ 2.0_dt, 0.0_dt, 1.0_dt };

    // Initialization
    raul::Workflow work;
    auto& networkParameters = work.getNetworkParameters();

    // Inputs
    work.add<raul::DataLayer>("data_query", raul::DataParams{ { "query" }, 1u, 1u, queryDepth });
    work.add<raul::DataLayer>("data_state", raul::DataParams{ { "state" }, 1u, 1u, alignmentsSize });
    work.add<raul::DataLayer>("data_memory", raul::DataParams{ { "memory" }, 1u, alignmentsSize, anyNumber });

    // Outputs
    work.add<raul::DataLayer>("output0", raul::DataParams{ { "realAlignment" }, 1u, 1u, alignmentsSize });
    work.add<raul::DataLayer>("output1", raul::DataParams{ { "realNextState" }, 1u, 1u, alignmentsSize });
    work.add<raul::DataLayer>("output2", raul::DataParams{ { "realMaxAttnIndices" }, 1u, 1u, 1u });

    // Layer
    raul::LocationSensitiveAttentionLayer(
        name,
        raul::LocationSensitiveAttentionParams{
            { "query", "state", "memory" }, { "alignment", "next_state", "max_attn" }, numUnits, raul::LocationSensitiveAttentionParams::hparams{ 2, 5, false, false }, true, true },
        networkParameters);

    work.preparePipelines();
    work.setBatchSize(batchSize);
    work.prepareMemoryForTraining();

    raul::MemoryManager& memory_manager = work.getMemoryManager();

    memory_manager["query"] = TORANGE(query);
    memory_manager["state"] = TORANGE(state);
    memory_manager["memory"] = TORANGE(memory);
    memory_manager["realAlignment"] = TORANGE(realAlignment);
    memory_manager["realNextState"] = TORANGE(realNextState);
    memory_manager["realMaxAttnIndices"] = TORANGE(realMaxAttnIndices);

    // For result stability
    memory_manager[name / "attention_variable_projection"] = TORANGE(raul::Tensor({ 0.36372268_dt, -0.55428183_dt, -0.67640346_dt, -0.48148942_dt, 0.46192443_dt }));
    memory_manager[name / "query_layer" / "Weights"] =
        TORANGE(raul::Tensor({ 0.01492912_dt, -0.22769517_dt, -0.46085864_dt, -0.5776096_dt, -0.3043793_dt, -0.32275102_dt, -0.08340913_dt, -0.09443533_dt, 0.0720188_dt,  0.19336885_dt,
                               0.6329126_dt,  0.4352892_dt,   -0.13510555_dt, 0.14927697_dt, 0.07225263_dt, 0.5521658_dt,   -0.57645786_dt, -0.59377813_dt, 0.7274594_dt,  -0.06407022_dt,
                               0.0731563_dt,  0.04765564_dt,  0.68469685_dt,  0.703242_dt,   0.27867514_dt, 0.3732596_dt,   -0.32511705_dt, 0.09024948_dt,  0.27464074_dt, 0.49236614_dt }));
    memory_manager[name / "location_convolution" / "Weights"] =
        TORANGE(raul::Tensor({ 0.4165066_dt, 0.16978645_dt, 0.01862907_dt, -0.13777304_dt, 0.10233486_dt, -0.57113034_dt, -0.40776673_dt, 0.25892937_dt, -0.01023513_dt, -0.19474253_dt }));
    memory_manager[name / "location_layer" / "Weights"] =
        TORANGE(raul::Tensor({ -0.4879959_dt, 0.6891403_dt, -0.48369777_dt, -0.42060506_dt, 0.00672376_dt, 0.21907902_dt, 0.50649774_dt, -0.55083364_dt, 0.20511496_dt, 0.68524706_dt }));
    memory_manager[name / "memory_layer" / "Weights"] = TORANGE(raul::Tensor(
        { -0.4735765_dt,  0.20152837_dt,  0.1932261_dt,  -0.670368_dt,   0.4062776_dt,   0.57766455_dt,  0.57984453_dt,  0.5677802_dt,   0.67286223_dt,  0.16185504_dt,  -0.0836153_dt, -0.6228876_dt,
          -0.04033387_dt, 0.09919816_dt,  0.18522543_dt, -0.09178317_dt, 0.55060273_dt,  -0.34977636_dt, -0.60656494_dt, -0.36432242_dt, -0.00504225_dt, -0.09256577_dt, 0.22641826_dt, 0.18068236_dt,
          0.5462021_dt,   -0.27094254_dt, 0.18809599_dt, 0.13281602_dt,  -0.29424265_dt, 0.14834511_dt,  0.04521954_dt,  0.5477156_dt,   -0.35188082_dt, 0.12166631_dt,  0.05859524_dt }));

    tools::callbacks::TensorChecker checker({ { "alignment", "realAlignment" }, { "next_state", "realNextState" }, { "max_attn", "realMaxAttnIndices" } }, -1_dt, eps);
    networkParameters.mCallback = checker;

    // Forward
    ASSERT_NO_THROW(work.forwardPassTraining());

    // Backward
    ASSERT_NO_THROW(work.backwardPassTraining());
}

TEST(TestLayerLocationSensitiveAttention, UseMaskNotConstrainedNoTransitionAgentNoUseForwardUnit)
{
    PROFILE_TEST

    // Cumulative attention mode, smoothing enabled, mask memory

    constexpr size_t numUnits = 5;
    constexpr size_t queryDepth = 6;
    constexpr size_t alignmentsSize = 4;
    constexpr size_t anyNumber = 7;
    constexpr size_t batchSize = 3;
    constexpr raul::dtype eps = TODTYPE(1e-5);

    raul::Name name = "attn";

    const raul::Tensor query{ 0.6910331_dt,  0.7831433_dt,  0.05777133_dt, 0.9138534_dt, 0.43685377_dt, 0.8047224_dt,  0.88028264_dt, 0.31608927_dt, 0.57692194_dt,
                              0.64140487_dt, 0.15070891_dt, 0.99759805_dt, 0.5437497_dt, 0.21818483_dt, 0.54885054_dt, 0.89362335_dt, 0.62603235_dt, 0.05133748_dt };
    const raul::Tensor state{ 0.63547623_dt, 0.44589663_dt, 0.6047574_dt,  0.82557225_dt, 0.58478403_dt, 0.04986751_dt,
                              0.9572661_dt,  0.20333457_dt, 0.11299467_dt, 0.05475962_dt, 0.2828188_dt,  0.5192108_dt };
    const raul::Tensor memory{ 0.81269646_dt, 0.07857466_dt, 0.8916855_dt,  0.16925514_dt, 0.06311357_dt, 0.54531074_dt, 0.5037316_dt,  0.9248222_dt,  0.66955376_dt, 0.9281193_dt,  0.12239242_dt,
                               0.8532245_dt,  0.90477383_dt, 0.7104306_dt,  0.40681756_dt, 0.5755513_dt,  0.8547678_dt,  0.59606934_dt, 0.77619946_dt, 0.97301054_dt, 0.06244731_dt, 0.33562684_dt,
                               0.22166848_dt, 0.32035887_dt, 0.03924382_dt, 0.06723011_dt, 0.32712245_dt, 0.49054873_dt, 0.11453211_dt, 0.34396613_dt, 0.52225363_dt, 0.30574834_dt, 0.8817626_dt,
                               0.8017194_dt,  0.9992852_dt,  0.65941477_dt, 0.1272459_dt,  0.19117236_dt, 0.65929854_dt, 0.7614676_dt,  0.75358987_dt, 0.41603255_dt, 0.94846773_dt, 0.8904344_dt,
                               0.91729546_dt, 0.26704276_dt, 0.17427123_dt, 0.04580772_dt, 0.98797727_dt, 0.03881574_dt, 0.22868955_dt, 0.0036062_dt,  0.6006421_dt,  0.25169027_dt, 0.45649374_dt,
                               0.21031535_dt, 0.13384092_dt, 0.610149_dt,   0.7017927_dt,  0.56946445_dt, 0.25802827_dt, 0.09499919_dt, 0.96377003_dt, 0.21196103_dt, 0.94442093_dt, 0.04924846_dt,
                               0.888088_dt,   0.23339641_dt, 0.4439162_dt,  0.13146889_dt, 0.9257786_dt,  0.3446467_dt,  0.9887433_dt,  0.84542334_dt, 0.7688427_dt,  0.2861563_dt,  0.47002888_dt,
                               0.83878493_dt, 0.7776841_dt,  0.35630226_dt, 0.7507192_dt,  0.3887322_dt,  0.3603543_dt,  0.0047611_dt };
    const raul::Tensor memorySeqLength{ 2.0_dt, 3.0_dt, 4.0_dt };

    // Real output
    const raul::Tensor realAlignment{ 0.41244003_dt, 0.58756_dt, 0._dt, 0._dt, 0.393119_dt, 0.38719293_dt, 0.21968812_dt, 0._dt, 0.24311139_dt, 0.2950311_dt, 0.23211668_dt, 0.22974089_dt };
    const raul::Tensor realNextState{ 1.0479163_dt, 1.0334566_dt,  0.6047574_dt,  0.82557225_dt, 0.977903_dt,  0.43706045_dt,
                                      1.1769543_dt, 0.20333457_dt, 0.35610604_dt, 0.34979072_dt, 0.5149355_dt, 0.7489517_dt };
    const raul::Tensor realMaxAttnIndices{ 1.0_dt, 0.0_dt, 1.0_dt };
    const raul::Tensor realKeys{ 0.32252908_dt, 0.63313645_dt, 0.33559194_dt,  0.3484964_dt,  -0.02079474_dt, 1.07549_dt,     0.6181176_dt,   -0.27268583_dt, 0.33390415_dt, -0.21231672_dt,
                                 0._dt,         0._dt,         0._dt,          0._dt,         0._dt,          0._dt,          0._dt,          0._dt,          0._dt,         0._dt,
                                 1.3118237_dt,  -0.1270132_dt, -0.6617147_dt,  0.3732552_dt,  0.05422493_dt,  0.29425073_dt,  -0.02759789_dt, -0.75341266_dt, 0.35311234_dt, 0.04271942_dt,
                                 0.39864153_dt, 1.2514074_dt,  0.37823635_dt,  0.51803136_dt, 0.04289641_dt,  0._dt,          0._dt,          0._dt,          0._dt,         0._dt,
                                 0.5319779_dt,  0.48356113_dt, -0.03996772_dt, 0.6395663_dt,  0.37200555_dt,  -0.06839194_dt, 0.53928274_dt,  -0.6348988_dt,  0.7259106_dt,  0.5459642_dt,
                                 0.00554493_dt, 0.4030629_dt,  -0.18443558_dt, 0.54069936_dt, 0.07830122_dt,  -0.3060568_dt,  0.7382188_dt,   -0.34951735_dt, 0.5359505_dt,  0.2031863_dt };

    // Initialization
    raul::Workflow work;
    auto& networkParameters = work.getNetworkParameters();

    // Inputs
    work.add<raul::DataLayer>("data_query", raul::DataParams{ { "query" }, 1u, 1u, queryDepth });
    work.add<raul::DataLayer>("data_state", raul::DataParams{ { "state" }, 1u, 1u, alignmentsSize });
    work.add<raul::DataLayer>("data_memory", raul::DataParams{ { "memory" }, 1u, alignmentsSize, anyNumber });
    work.add<raul::DataLayer>("data_memory_seq_length", raul::DataParams{ { "memorySeqLength" }, 1u, 1u, 1u });

    // Outputs
    work.add<raul::DataLayer>("output0", raul::DataParams{ { "realAlignment" }, 1u, 1u, alignmentsSize });
    work.add<raul::DataLayer>("output1", raul::DataParams{ { "realNextState" }, 1u, 1u, alignmentsSize });
    work.add<raul::DataLayer>("output2", raul::DataParams{ { "realMaxAttnIndices" }, 1u, 1u, 1u });
    work.add<raul::DataLayer>("output3", raul::DataParams{ { "realKeys" }, 1u, alignmentsSize, numUnits });

    // Layer
    raul::LocationSensitiveAttentionLayer(name,
                                          raul::LocationSensitiveAttentionParams{ { "query", "state", "memory", "memorySeqLength" },
                                                                                  { "alignment", "values", "next_state", "max_attn" },
                                                                                  numUnits,
                                                                                  raul::LocationSensitiveAttentionParams::hparams{ 2, 5, false, false },
                                                                                  true,
                                                                                  true },
                                          networkParameters);

    work.preparePipelines();
    work.setBatchSize(batchSize);
    work.prepareMemoryForTraining();

    raul::MemoryManager& memory_manager = work.getMemoryManager();

    memory_manager["query"] = TORANGE(query);
    memory_manager["state"] = TORANGE(state);
    memory_manager["memory"] = TORANGE(memory);
    memory_manager["memorySeqLength"] = TORANGE(memorySeqLength);
    memory_manager["realAlignment"] = TORANGE(realAlignment);
    memory_manager["realNextState"] = TORANGE(realNextState);
    memory_manager["realMaxAttnIndices"] = TORANGE(realMaxAttnIndices);
    memory_manager["realKeys"] = TORANGE(realKeys);

    // For result stability
    memory_manager[name / "attention_variable_projection"] = TORANGE(raul::Tensor({ 0.36372268_dt, -0.55428183_dt, -0.67640346_dt, -0.48148942_dt, 0.46192443_dt }));
    memory_manager[name / "query_layer" / "Weights"] =
        TORANGE(raul::Tensor({ 0.01492912_dt, -0.22769517_dt, -0.46085864_dt, -0.5776096_dt, -0.3043793_dt, -0.32275102_dt, -0.08340913_dt, -0.09443533_dt, 0.0720188_dt,  0.19336885_dt,
                               0.6329126_dt,  0.4352892_dt,   -0.13510555_dt, 0.14927697_dt, 0.07225263_dt, 0.5521658_dt,   -0.57645786_dt, -0.59377813_dt, 0.7274594_dt,  -0.06407022_dt,
                               0.0731563_dt,  0.04765564_dt,  0.68469685_dt,  0.703242_dt,   0.27867514_dt, 0.3732596_dt,   -0.32511705_dt, 0.09024948_dt,  0.27464074_dt, 0.49236614_dt }));
    memory_manager[name / "location_convolution" / "Weights"] =
        TORANGE(raul::Tensor({ 0.4165066_dt, 0.16978645_dt, 0.01862907_dt, -0.13777304_dt, 0.10233486_dt, -0.57113034_dt, -0.40776673_dt, 0.25892937_dt, -0.01023513_dt, -0.19474253_dt }));
    memory_manager[name / "location_layer" / "Weights"] =
        TORANGE(raul::Tensor({ -0.4879959_dt, 0.6891403_dt, -0.48369777_dt, -0.42060506_dt, 0.00672376_dt, 0.21907902_dt, 0.50649774_dt, -0.55083364_dt, 0.20511496_dt, 0.68524706_dt }));
    memory_manager[name / "memory_layer" / "Weights"] = TORANGE(raul::Tensor(
        { -0.4735765_dt,  0.20152837_dt,  0.1932261_dt,  -0.670368_dt,   0.4062776_dt,   0.57766455_dt,  0.57984453_dt,  0.5677802_dt,   0.67286223_dt,  0.16185504_dt,  -0.0836153_dt, -0.6228876_dt,
          -0.04033387_dt, 0.09919816_dt,  0.18522543_dt, -0.09178317_dt, 0.55060273_dt,  -0.34977636_dt, -0.60656494_dt, -0.36432242_dt, -0.00504225_dt, -0.09256577_dt, 0.22641826_dt, 0.18068236_dt,
          0.5462021_dt,   -0.27094254_dt, 0.18809599_dt, 0.13281602_dt,  -0.29424265_dt, 0.14834511_dt,  0.04521954_dt,  0.5477156_dt,   -0.35188082_dt, 0.12166631_dt,  0.05859524_dt }));

    tools::callbacks::TensorChecker checker({ { "alignment", "realAlignment" }, { "next_state", "realNextState" }, { "max_attn", "realMaxAttnIndices" }, { name / "keys", "realKeys" } }, -1_dt, eps);
    networkParameters.mCallback = checker;

    // Forward
    ASSERT_NO_THROW(work.forwardPassTraining());

    // Backward
    ASSERT_NO_THROW(work.backwardPassTraining());
}

TEST(TestLayerLocationSensitiveAttention, UseMaskNotConstrainedNoTransitionAgentUseForwardUnit)
{
    PROFILE_TEST

    // Cumulative attention mode, smoothing enabled, use forward and mask memory

    constexpr size_t numUnits = 5;
    constexpr size_t queryDepth = 6;
    constexpr size_t alignmentsSize = 4;
    constexpr size_t anyNumber = 7;
    constexpr size_t batchSize = 3;
    constexpr raul::dtype eps = TODTYPE(1e-5);

    raul::Name name = "attn";

    const raul::Tensor query{ 0.6910331_dt,  0.7831433_dt,  0.05777133_dt, 0.9138534_dt, 0.43685377_dt, 0.8047224_dt,  0.88028264_dt, 0.31608927_dt, 0.57692194_dt,
                              0.64140487_dt, 0.15070891_dt, 0.99759805_dt, 0.5437497_dt, 0.21818483_dt, 0.54885054_dt, 0.89362335_dt, 0.62603235_dt, 0.05133748_dt };
    const raul::Tensor state{ 0.63547623_dt, 0.44589663_dt, 0.6047574_dt,  0.82557225_dt, 0.58478403_dt, 0.04986751_dt,
                              0.9572661_dt,  0.20333457_dt, 0.11299467_dt, 0.05475962_dt, 0.2828188_dt,  0.5192108_dt };
    const raul::Tensor memory{ 0.81269646_dt, 0.07857466_dt, 0.8916855_dt,  0.16925514_dt, 0.06311357_dt, 0.54531074_dt, 0.5037316_dt,  0.9248222_dt,  0.66955376_dt, 0.9281193_dt,  0.12239242_dt,
                               0.8532245_dt,  0.90477383_dt, 0.7104306_dt,  0.40681756_dt, 0.5755513_dt,  0.8547678_dt,  0.59606934_dt, 0.77619946_dt, 0.97301054_dt, 0.06244731_dt, 0.33562684_dt,
                               0.22166848_dt, 0.32035887_dt, 0.03924382_dt, 0.06723011_dt, 0.32712245_dt, 0.49054873_dt, 0.11453211_dt, 0.34396613_dt, 0.52225363_dt, 0.30574834_dt, 0.8817626_dt,
                               0.8017194_dt,  0.9992852_dt,  0.65941477_dt, 0.1272459_dt,  0.19117236_dt, 0.65929854_dt, 0.7614676_dt,  0.75358987_dt, 0.41603255_dt, 0.94846773_dt, 0.8904344_dt,
                               0.91729546_dt, 0.26704276_dt, 0.17427123_dt, 0.04580772_dt, 0.98797727_dt, 0.03881574_dt, 0.22868955_dt, 0.0036062_dt,  0.6006421_dt,  0.25169027_dt, 0.45649374_dt,
                               0.21031535_dt, 0.13384092_dt, 0.610149_dt,   0.7017927_dt,  0.56946445_dt, 0.25802827_dt, 0.09499919_dt, 0.96377003_dt, 0.21196103_dt, 0.94442093_dt, 0.04924846_dt,
                               0.888088_dt,   0.23339641_dt, 0.4439162_dt,  0.13146889_dt, 0.9257786_dt,  0.3446467_dt,  0.9887433_dt,  0.84542334_dt, 0.7688427_dt,  0.2861563_dt,  0.47002888_dt,
                               0.83878493_dt, 0.7776841_dt,  0.35630226_dt, 0.7507192_dt,  0.3887322_dt,  0.3603543_dt,  0.0047611_dt };
    const raul::Tensor memorySeqLength{ 2.0_dt, 3.0_dt, 4.0_dt };

    // Real output
    const raul::Tensor realAlignment{ 0.29203945_dt, 0.7079606_dt, 0._dt, 0._dt, 0.32988536_dt, 0.35261944_dt, 0.3174952_dt, 0._dt, 0.08089499_dt, 0.14574707_dt, 0.23074879_dt, 0.54260916_dt };
    const raul::Tensor realNextState{ 0.9275157_dt, 1.1538572_dt,  0.6047574_dt,  0.82557225_dt, 0.9146694_dt,  0.40248695_dt,
                                      1.2747613_dt, 0.20333457_dt, 0.19388966_dt, 0.20050669_dt, 0.51356757_dt, 1.06182_dt };
    const raul::Tensor realMaxAttnIndices{ 1.0_dt, 1.0_dt, 3.0_dt };

    // Initialization
    raul::Workflow work;
    auto& networkParameters = work.getNetworkParameters();

    // Inputs
    work.add<raul::DataLayer>("data_query", raul::DataParams{ { "query" }, 1u, 1u, queryDepth });
    work.add<raul::DataLayer>("data_state", raul::DataParams{ { "state" }, 1u, 1u, alignmentsSize });
    work.add<raul::DataLayer>("data_memory", raul::DataParams{ { "memory" }, 1u, alignmentsSize, anyNumber });
    work.add<raul::DataLayer>("data_memory_seq_length", raul::DataParams{ { "memorySeqLength" }, 1u, 1u, 1u });

    // Outputs
    work.add<raul::DataLayer>("output0", raul::DataParams{ { "realAlignment" }, 1u, 1u, alignmentsSize });
    work.add<raul::DataLayer>("output1", raul::DataParams{ { "realNextState" }, 1u, 1u, alignmentsSize });
    work.add<raul::DataLayer>("output2", raul::DataParams{ { "realMaxAttnIndices" }, 1u, 1u, 1u });

    // Layer
    raul::LocationSensitiveAttentionLayer(name,
                                          raul::LocationSensitiveAttentionParams{ { "query", "state", "memory", "memorySeqLength" },
                                                                                  { "alignment", "values", "next_state", "max_attn" },
                                                                                  numUnits,
                                                                                  raul::LocationSensitiveAttentionParams::hparams{ 2, 5, false, false },
                                                                                  true,
                                                                                  true,
                                                                                  0.0_dt,
                                                                                  true },
                                          networkParameters);

    work.preparePipelines();
    work.setBatchSize(batchSize);
    work.prepareMemoryForTraining();

    raul::MemoryManager& memory_manager = work.getMemoryManager();

    memory_manager["query"] = TORANGE(query);
    memory_manager["state"] = TORANGE(state);
    memory_manager["memory"] = TORANGE(memory);
    memory_manager["memorySeqLength"] = TORANGE(memorySeqLength);
    memory_manager["realAlignment"] = TORANGE(realAlignment);
    memory_manager["realNextState"] = TORANGE(realNextState);
    memory_manager["realMaxAttnIndices"] = TORANGE(realMaxAttnIndices);

    // For result stability
    memory_manager[name / "attention_variable_projection"] = TORANGE(raul::Tensor({ 0.36372268_dt, -0.55428183_dt, -0.67640346_dt, -0.48148942_dt, 0.46192443_dt }));
    memory_manager[name / "query_layer" / "Weights"] =
        TORANGE(raul::Tensor({ 0.01492912_dt, -0.22769517_dt, -0.46085864_dt, -0.5776096_dt, -0.3043793_dt, -0.32275102_dt, -0.08340913_dt, -0.09443533_dt, 0.0720188_dt,  0.19336885_dt,
                               0.6329126_dt,  0.4352892_dt,   -0.13510555_dt, 0.14927697_dt, 0.07225263_dt, 0.5521658_dt,   -0.57645786_dt, -0.59377813_dt, 0.7274594_dt,  -0.06407022_dt,
                               0.0731563_dt,  0.04765564_dt,  0.68469685_dt,  0.703242_dt,   0.27867514_dt, 0.3732596_dt,   -0.32511705_dt, 0.09024948_dt,  0.27464074_dt, 0.49236614_dt }));
    memory_manager[name / "location_convolution" / "Weights"] =
        TORANGE(raul::Tensor({ 0.4165066_dt, 0.16978645_dt, 0.01862907_dt, -0.13777304_dt, 0.10233486_dt, -0.57113034_dt, -0.40776673_dt, 0.25892937_dt, -0.01023513_dt, -0.19474253_dt }));
    memory_manager[name / "location_layer" / "Weights"] =
        TORANGE(raul::Tensor({ -0.4879959_dt, 0.6891403_dt, -0.48369777_dt, -0.42060506_dt, 0.00672376_dt, 0.21907902_dt, 0.50649774_dt, -0.55083364_dt, 0.20511496_dt, 0.68524706_dt }));
    memory_manager[name / "memory_layer" / "Weights"] = TORANGE(raul::Tensor(
        { -0.4735765_dt,  0.20152837_dt,  0.1932261_dt,  -0.670368_dt,   0.4062776_dt,   0.57766455_dt,  0.57984453_dt,  0.5677802_dt,   0.67286223_dt,  0.16185504_dt,  -0.0836153_dt, -0.6228876_dt,
          -0.04033387_dt, 0.09919816_dt,  0.18522543_dt, -0.09178317_dt, 0.55060273_dt,  -0.34977636_dt, -0.60656494_dt, -0.36432242_dt, -0.00504225_dt, -0.09256577_dt, 0.22641826_dt, 0.18068236_dt,
          0.5462021_dt,   -0.27094254_dt, 0.18809599_dt, 0.13281602_dt,  -0.29424265_dt, 0.14834511_dt,  0.04521954_dt,  0.5477156_dt,   -0.35188082_dt, 0.12166631_dt,  0.05859524_dt }));

    tools::callbacks::TensorChecker checker({ { "alignment", "realAlignment" }, { "next_state", "realNextState" }, { "max_attn", "realMaxAttnIndices" } }, -1_dt, eps);
    networkParameters.mCallback = checker;

    // Forward
    ASSERT_NO_THROW(work.forwardPassTraining());

    // Backward
    ASSERT_NO_THROW(work.backwardPassTraining());
}

TEST(TestLayerLocationSensitiveAttention, UseMaskConstrainedNoTransitionAgentNoUseForwardUnit)
{
    PROFILE_TEST

    // Cumulative attention mode, smoothing enabled, constrained and mask memory

    constexpr size_t numUnits = 5;
    constexpr size_t queryDepth = 8;
    constexpr size_t alignmentsSize = 5;
    constexpr size_t anyNumber = 4;
    constexpr size_t batchSize = 3;
    constexpr raul::dtype eps = TODTYPE(1e-5);

    raul::Name name = "attn";

    const raul::Tensor query{ 0.97862554_dt, 0.99683046_dt, 0.2748139_dt,  0.03111756_dt, 0.6671331_dt, 0.24519491_dt, 0.78158295_dt, 0.41332448_dt,
                              0.40015996_dt, 0.56719434_dt, 0.94010246_dt, 0.20412171_dt, 0.8084135_dt, 0.94816256_dt, 0.19557941_dt, 0.68961465_dt,
                              0.5837462_dt,  0.7141627_dt,  0.9072653_dt,  0.30709636_dt, 0.9239814_dt, 0.23369503_dt, 0.71944916_dt, 0.33713996_dt };
    const raul::Tensor state{ 0.83753383_dt, 0.16850388_dt, 0.03760135_dt, 0.7767941_dt,  0.49460685_dt, 0.77782524_dt, 0.16286612_dt, 0.26141143_dt,
                              0.7960582_dt,  0.24748445_dt, 0.09534061_dt, 0.36989713_dt, 0.6322192_dt,  0.9825914_dt,  0.07898891_dt };
    const raul::Tensor memory{ 0.81269646_dt, 0.07857466_dt, 0.8916855_dt,  0.16925514_dt, 0.06311357_dt, 0.54531074_dt, 0.5037316_dt,  0.9248222_dt,  0.66955376_dt, 0.9281193_dt,
                               0.12239242_dt, 0.8532245_dt,  0.90477383_dt, 0.7104306_dt,  0.40681756_dt, 0.5755513_dt,  0.8547678_dt,  0.59606934_dt, 0.77619946_dt, 0.97301054_dt,
                               0.06244731_dt, 0.33562684_dt, 0.22166848_dt, 0.32035887_dt, 0.03924382_dt, 0.06723011_dt, 0.32712245_dt, 0.49054873_dt, 0.11453211_dt, 0.34396613_dt,
                               0.52225363_dt, 0.30574834_dt, 0.8817626_dt,  0.8017194_dt,  0.9992852_dt,  0.65941477_dt, 0.1272459_dt,  0.19117236_dt, 0.65929854_dt, 0.7614676_dt,
                               0.75358987_dt, 0.41603255_dt, 0.94846773_dt, 0.8904344_dt,  0.91729546_dt, 0.26704276_dt, 0.17427123_dt, 0.04580772_dt, 0.98797727_dt, 0.03881574_dt,
                               0.22868955_dt, 0.0036062_dt,  0.6006421_dt,  0.25169027_dt, 0.45649374_dt, 0.21031535_dt, 0.13384092_dt, 0.610149_dt,   0.7017927_dt,  0.56946445_dt };
    const raul::Tensor memorySeqLength{ 3.0_dt, 1.0_dt, 2.0_dt };

    // Real output
    const raul::Tensor realAlignment{ 0.33902472_dt, 0.5806279_dt,  0.10332_dt,    0.41663912_dt, 0.63432395_dt, 0.33679605_dt, 0.51554906_dt, 0.20459965_dt,
                                      0.4847585_dt,  0.56436956_dt, 0.03594348_dt, 0.2044092_dt,  0.49786937_dt, 0.7823901_dt,  0.5946217_dt };
    const raul::Tensor realNextState{ 1.1765585_dt, 0.7491318_dt, 0.14092135_dt, 1.1934332_dt, 1.1289308_dt, 1.1146213_dt, 0.6784152_dt, 0.46601108_dt,
                                      1.2808167_dt, 0.811854_dt,  0.13128409_dt, 0.5743063_dt, 1.1300886_dt, 1.7649815_dt, 0.6736106_dt };
    const raul::Tensor realMaxAttnIndices{ 4.0_dt, 4.0_dt, 3.0_dt };

    // Initialization
    raul::Workflow work;
    auto& networkParameters = work.getNetworkParameters();

    // Inputs
    work.add<raul::DataLayer>("data_query", raul::DataParams{ { "query" }, 1u, 1u, queryDepth });
    work.add<raul::DataLayer>("data_state", raul::DataParams{ { "state" }, 1u, 1u, alignmentsSize });
    work.add<raul::DataLayer>("data_memory", raul::DataParams{ { "memory" }, 1u, alignmentsSize, anyNumber });
    work.add<raul::DataLayer>("data_memory_seq_length", raul::DataParams{ { "memorySeqLength" }, 1u, 1u, 1u });

    // Outputs
    work.add<raul::DataLayer>("output0", raul::DataParams{ { "realAlignment" }, 1u, 1u, alignmentsSize });
    work.add<raul::DataLayer>("output1", raul::DataParams{ { "realNextState" }, 1u, 1u, alignmentsSize });
    work.add<raul::DataLayer>("output2", raul::DataParams{ { "realMaxAttnIndices" }, 1u, 1u, 1u });

    // Layer
    raul::LocationSensitiveAttentionLayer(name,
                                          raul::LocationSensitiveAttentionParams{ { "query", "state", "memory", "memorySeqLength" },
                                                                                  { "alignment", "values", "next_state", "max_attn" },
                                                                                  numUnits,
                                                                                  raul::LocationSensitiveAttentionParams::hparams{ 5, 5, false, true },
                                                                                  true,
                                                                                  true },
                                          networkParameters);

    work.preparePipelines();
    work.setBatchSize(batchSize);
    work.prepareMemoryForTraining();

    raul::MemoryManager& memory_manager = work.getMemoryManager();

    memory_manager["query"] = TORANGE(query);
    memory_manager["state"] = TORANGE(state);
    memory_manager["memory"] = TORANGE(memory);
    memory_manager["memorySeqLength"] = TORANGE(memorySeqLength);
    memory_manager["realAlignment"] = TORANGE(realAlignment);
    memory_manager["realNextState"] = TORANGE(realNextState);
    memory_manager["realMaxAttnIndices"] = TORANGE(realMaxAttnIndices);

    // For result stability
    memory_manager[name / "attention_variable_projection"] = TORANGE(raul::Tensor({ 0.36372268_dt, -0.55428183_dt, -0.67640346_dt, -0.48148942_dt, 0.46192443_dt }));
    memory_manager[name / "query_layer" / "Weights"] =
        TORANGE(raul::Tensor({ 0.01373279_dt, -0.20944911_dt, -0.4239283_dt, -0.53132355_dt, -0.27998826_dt, -0.29688776_dt, 0.49039555_dt,  0.65391076_dt, -0.07672524_dt, -0.08686787_dt,
                               0.06624764_dt, 0.17787349_dt,  0.5821949_dt,  0.4004078_dt,   0.22502369_dt,  0.6567254_dt,   -0.12427902_dt, 0.13731486_dt, 0.06646276_dt,  0.5079187_dt,
                               -0.5302641_dt, -0.54619646_dt, -0.4671754_dt, -0.62143993_dt, 0.66916525_dt,  -0.058936_dt,   0.067294_dt,    0.04383683_dt, 0.6298295_dt,   0.6468886_dt,
                               0.6016828_dt,  -0.10036236_dt, 0.25634384_dt, 0.34334886_dt,  -0.2990642_dt,  0.08301741_dt,  0.25263274_dt,  0.45291102_dt, 0.41381907_dt,  0.08302146_dt }));
    memory_manager[name / "location_convolution" / "Weights"] =
        TORANGE(raul::Tensor({ 0.29451466_dt,  0.12005717_dt, 0.01317275_dt,  -0.09742025_dt, 0.07236165_dt, -0.40385014_dt, -0.2883346_dt,  0.18309075_dt,  -0.00723732_dt,
                               -0.13770375_dt, 0.00707558_dt, -0.24894215_dt, 0.3117578_dt,   0.39720428_dt, 0.02360603_dt,  -0.10541618_dt, -0.19693545_dt, -0.41700584_dt,
                               -0.06090459_dt, -0.3629806_dt, 0.4160447_dt,   0.22983533_dt,  0.15451461_dt, 0.03033999_dt,  -0.15783566_dt }));
    memory_manager[name / "location_layer" / "Weights"] =
        TORANGE(raul::Tensor({ -0.40828666_dt, 0.57657623_dt,  -0.01176476_dt, -0.45261443_dt, -0.41842088_dt, -0.4046906_dt,  -0.35190347_dt, -0.00475621_dt, -0.09430981_dt,
                               -0.29316878_dt, 0.00562549_dt,  0.18329465_dt,  -0.7279368_dt,  -0.48186913_dt, 0.54186_dt,     0.42376637_dt,  -0.4608605_dt,  0.2877854_dt,
                               0.4080819_dt,   -0.13309819_dt, 0.17161155_dt,  0.57331884_dt,  0.20383084_dt,  -0.04049957_dt, -0.45443535_dt }));
    memory_manager[name / "memory_layer" / "Weights"] =
        TORANGE(raul::Tensor({ -0.54683906_dt, 0.23270488_dt,  0.2231183_dt,   -0.7740744_dt, 0.65561616_dt, 0.77695453_dt, 0.18689406_dt,  -0.09655064_dt, 0.21387994_dt, -0.10598212_dt,
                               0.6357813_dt,   -0.40388697_dt, -0.10688573_dt, 0.26144528_dt, 0.20863402_dt, 0.6306999_dt,  -0.33976218_dt, 0.17129415_dt,  0.05221498_dt, 0.6324476_dt }));

    tools::callbacks::TensorChecker checker({ { "alignment", "realAlignment" }, { "next_state", "realNextState" }, { "max_attn", "realMaxAttnIndices" } }, -1_dt, eps);
    networkParameters.mCallback = checker;

    // Forward
    ASSERT_NO_THROW(work.forwardPassTraining());

    // Backward
    ASSERT_NO_THROW(work.backwardPassTraining());
}

TEST(TestLayerLocationSensitiveAttention, UseMaskConstrainedNoTransitionAgentUseForwardUnit)
{
    PROFILE_TEST

    // Cumulative attention mode, smoothing enabled, constrained, use forward and mask memory

    constexpr size_t numUnits = 5;
    constexpr size_t queryDepth = 8;
    constexpr size_t alignmentsSize = 5;
    constexpr size_t anyNumber = 4;
    constexpr size_t batchSize = 3;
    constexpr raul::dtype eps = TODTYPE(1e-5);

    raul::Name name = "attn";

    const raul::Tensor query{ 0.97862554_dt, 0.99683046_dt, 0.2748139_dt,  0.03111756_dt, 0.6671331_dt, 0.24519491_dt, 0.78158295_dt, 0.41332448_dt,
                              0.40015996_dt, 0.56719434_dt, 0.94010246_dt, 0.20412171_dt, 0.8084135_dt, 0.94816256_dt, 0.19557941_dt, 0.68961465_dt,
                              0.5837462_dt,  0.7141627_dt,  0.9072653_dt,  0.30709636_dt, 0.9239814_dt, 0.23369503_dt, 0.71944916_dt, 0.33713996_dt };
    const raul::Tensor state{ 0.83753383_dt, 0.16850388_dt, 0.03760135_dt, 0.7767941_dt,  0.49460685_dt, 0.77782524_dt, 0.16286612_dt, 0.26141143_dt,
                              0.7960582_dt,  0.24748445_dt, 0.09534061_dt, 0.36989713_dt, 0.6322192_dt,  0.9825914_dt,  0.07898891_dt };
    const raul::Tensor memory{ 0.81269646_dt, 0.07857466_dt, 0.8916855_dt,  0.16925514_dt, 0.06311357_dt, 0.54531074_dt, 0.5037316_dt,  0.9248222_dt,  0.66955376_dt, 0.9281193_dt,
                               0.12239242_dt, 0.8532245_dt,  0.90477383_dt, 0.7104306_dt,  0.40681756_dt, 0.5755513_dt,  0.8547678_dt,  0.59606934_dt, 0.77619946_dt, 0.97301054_dt,
                               0.06244731_dt, 0.33562684_dt, 0.22166848_dt, 0.32035887_dt, 0.03924382_dt, 0.06723011_dt, 0.32712245_dt, 0.49054873_dt, 0.11453211_dt, 0.34396613_dt,
                               0.52225363_dt, 0.30574834_dt, 0.8817626_dt,  0.8017194_dt,  0.9992852_dt,  0.65941477_dt, 0.1272459_dt,  0.19117236_dt, 0.65929854_dt, 0.7614676_dt,
                               0.75358987_dt, 0.41603255_dt, 0.94846773_dt, 0.8904344_dt,  0.91729546_dt, 0.26704276_dt, 0.17427123_dt, 0.04580772_dt, 0.98797727_dt, 0.03881574_dt,
                               0.22868955_dt, 0.0036062_dt,  0.6006421_dt,  0.25169027_dt, 0.45649374_dt, 0.21031535_dt, 0.13384092_dt, 0.610149_dt,   0.7017927_dt,  0.56946445_dt };
    const raul::Tensor memorySeqLength{ 3.0_dt, 1.0_dt, 2.0_dt };

    // Real output
    const raul::Tensor realAlignment{ 0.13951944_dt, 0.28702068_dt, 0.01046344_dt, 0.16672334_dt, 0.39627317_dt, 0.1353626_dt, 0.25059178_dt, 0.04485435_dt,
                                      0.26487622_dt, 0.30431506_dt, 0.00137509_dt, 0.03816015_dt, 0.20020191_dt, 0.5069669_dt, 0.25329596_dt };
    const raul::Tensor realNextState{ 0.9770533_dt, 0.45552456_dt, 0.04806479_dt, 0.94351745_dt, 0.89088_dt,   0.91318786_dt, 0.4134579_dt, 0.30626577_dt,
                                      1.0609344_dt, 0.55179954_dt, 0.0967157_dt,  0.40805727_dt, 0.8324211_dt, 1.4895582_dt,  0.33228487_dt };
    const raul::Tensor realMaxAttnIndices{ 4.0_dt, 4.0_dt, 3.0_dt };

    // Initialization
    raul::Workflow work;
    auto& networkParameters = work.getNetworkParameters();

    // Inputs
    work.add<raul::DataLayer>("data_query", raul::DataParams{ { "query" }, 1u, 1u, queryDepth });
    work.add<raul::DataLayer>("data_state", raul::DataParams{ { "state" }, 1u, 1u, alignmentsSize });
    work.add<raul::DataLayer>("data_memory", raul::DataParams{ { "memory" }, 1u, alignmentsSize, anyNumber });
    work.add<raul::DataLayer>("data_memory_seq_length", raul::DataParams{ { "memorySeqLength" }, 1u, 1u, 1u });

    // Outputs
    work.add<raul::DataLayer>("output0", raul::DataParams{ { "realAlignment" }, 1u, 1u, alignmentsSize });
    work.add<raul::DataLayer>("output1", raul::DataParams{ { "realNextState" }, 1u, 1u, alignmentsSize });
    work.add<raul::DataLayer>("output2", raul::DataParams{ { "realMaxAttnIndices" }, 1u, 1u, 1u });

    // Layer
    raul::LocationSensitiveAttentionLayer(name,
                                          raul::LocationSensitiveAttentionParams{ { "query", "state", "memory", "memorySeqLength" },
                                                                                  { "alignment", "values", "next_state", "max_attn" },
                                                                                  numUnits,
                                                                                  raul::LocationSensitiveAttentionParams::hparams{ 5, 5, false, true },
                                                                                  true,
                                                                                  true,
                                                                                  0.0_dt,
                                                                                  true },
                                          networkParameters);

    work.preparePipelines();
    work.setBatchSize(batchSize);
    work.prepareMemoryForTraining();

    raul::MemoryManager& memory_manager = work.getMemoryManager();

    memory_manager["query"] = TORANGE(query);
    memory_manager["state"] = TORANGE(state);
    memory_manager["memory"] = TORANGE(memory);
    memory_manager["memorySeqLength"] = TORANGE(memorySeqLength);
    memory_manager["realAlignment"] = TORANGE(realAlignment);
    memory_manager["realNextState"] = TORANGE(realNextState);
    memory_manager["realMaxAttnIndices"] = TORANGE(realMaxAttnIndices);

    // For result stability
    memory_manager[name / "attention_variable_projection"] = TORANGE(raul::Tensor({ 0.36372268_dt, -0.55428183_dt, -0.67640346_dt, -0.48148942_dt, 0.46192443_dt }));
    memory_manager[name / "query_layer" / "Weights"] =
        TORANGE(raul::Tensor({ 0.01373279_dt, -0.20944911_dt, -0.4239283_dt, -0.53132355_dt, -0.27998826_dt, -0.29688776_dt, 0.49039555_dt,  0.65391076_dt, -0.07672524_dt, -0.08686787_dt,
                               0.06624764_dt, 0.17787349_dt,  0.5821949_dt,  0.4004078_dt,   0.22502369_dt,  0.6567254_dt,   -0.12427902_dt, 0.13731486_dt, 0.06646276_dt,  0.5079187_dt,
                               -0.5302641_dt, -0.54619646_dt, -0.4671754_dt, -0.62143993_dt, 0.66916525_dt,  -0.058936_dt,   0.067294_dt,    0.04383683_dt, 0.6298295_dt,   0.6468886_dt,
                               0.6016828_dt,  -0.10036236_dt, 0.25634384_dt, 0.34334886_dt,  -0.2990642_dt,  0.08301741_dt,  0.25263274_dt,  0.45291102_dt, 0.41381907_dt,  0.08302146_dt }));
    memory_manager[name / "location_convolution" / "Weights"] =
        TORANGE(raul::Tensor({ 0.29451466_dt,  0.12005717_dt, 0.01317275_dt,  -0.09742025_dt, 0.07236165_dt, -0.40385014_dt, -0.2883346_dt,  0.18309075_dt,  -0.00723732_dt,
                               -0.13770375_dt, 0.00707558_dt, -0.24894215_dt, 0.3117578_dt,   0.39720428_dt, 0.02360603_dt,  -0.10541618_dt, -0.19693545_dt, -0.41700584_dt,
                               -0.06090459_dt, -0.3629806_dt, 0.4160447_dt,   0.22983533_dt,  0.15451461_dt, 0.03033999_dt,  -0.15783566_dt }));
    memory_manager[name / "location_layer" / "Weights"] =
        TORANGE(raul::Tensor({ -0.40828666_dt, 0.57657623_dt,  -0.01176476_dt, -0.45261443_dt, -0.41842088_dt, -0.4046906_dt,  -0.35190347_dt, -0.00475621_dt, -0.09430981_dt,
                               -0.29316878_dt, 0.00562549_dt,  0.18329465_dt,  -0.7279368_dt,  -0.48186913_dt, 0.54186_dt,     0.42376637_dt,  -0.4608605_dt,  0.2877854_dt,
                               0.4080819_dt,   -0.13309819_dt, 0.17161155_dt,  0.57331884_dt,  0.20383084_dt,  -0.04049957_dt, -0.45443535_dt }));
    memory_manager[name / "memory_layer" / "Weights"] =
        TORANGE(raul::Tensor({ -0.54683906_dt, 0.23270488_dt,  0.2231183_dt,   -0.7740744_dt, 0.65561616_dt, 0.77695453_dt, 0.18689406_dt,  -0.09655064_dt, 0.21387994_dt, -0.10598212_dt,
                               0.6357813_dt,   -0.40388697_dt, -0.10688573_dt, 0.26144528_dt, 0.20863402_dt, 0.6306999_dt,  -0.33976218_dt, 0.17129415_dt,  0.05221498_dt, 0.6324476_dt }));

    tools::callbacks::TensorChecker checker({ { "alignment", "realAlignment" }, { "next_state", "realNextState" }, { "max_attn", "realMaxAttnIndices" } }, -1_dt, eps);
    networkParameters.mCallback = checker;

    // Forward
    ASSERT_NO_THROW(work.forwardPassTraining());

    // Backward
    ASSERT_NO_THROW(work.backwardPassTraining());
}

TEST(TestLayerLocationSensitiveAttention, UseMaskNotConstrainedUseTransitionAgentNoUseForwardUnit)
{
    PROFILE_TEST

    // Cumulative attention mode, smoothing enabled, use transition agent and mask memory

    constexpr size_t numUnits = 7;
    constexpr size_t queryDepth = 4;
    constexpr size_t alignmentsSize = 3;
    constexpr size_t anyNumber = 7;
    constexpr size_t batchSize = 3;
    constexpr raul::dtype eps = TODTYPE(1e-5);

    raul::Name name = "attn";

    const raul::Tensor query{ 0.22413874_dt, 0.22268498_dt, 0.8552655_dt,  0.49562013_dt, 0.31110537_dt, 0.61050725_dt,
                              0.21236408_dt, 0.93036723_dt, 0.54842377_dt, 0.84664714_dt, 0.47629058_dt, 0.89816856_dt };
    const raul::Tensor state{ 0.7847103_dt, 0.11388123_dt, 0.59057367_dt, 0.9255452_dt, 0.6040523_dt, 0.91908026_dt, 0.34620273_dt, 0.6069509_dt, 0.18924046_dt };
    const raul::Tensor memory{ 0.81269646_dt, 0.07857466_dt, 0.8916855_dt,  0.16925514_dt, 0.06311357_dt, 0.54531074_dt, 0.5037316_dt,  0.9248222_dt,  0.66955376_dt, 0.9281193_dt,  0.12239242_dt,
                               0.8532245_dt,  0.90477383_dt, 0.7104306_dt,  0.40681756_dt, 0.5755513_dt,  0.8547678_dt,  0.59606934_dt, 0.77619946_dt, 0.97301054_dt, 0.06244731_dt, 0.33562684_dt,
                               0.22166848_dt, 0.32035887_dt, 0.03924382_dt, 0.06723011_dt, 0.32712245_dt, 0.49054873_dt, 0.11453211_dt, 0.34396613_dt, 0.52225363_dt, 0.30574834_dt, 0.8817626_dt,
                               0.8017194_dt,  0.9992852_dt,  0.65941477_dt, 0.1272459_dt,  0.19117236_dt, 0.65929854_dt, 0.7614676_dt,  0.75358987_dt, 0.41603255_dt, 0.94846773_dt, 0.8904344_dt,
                               0.91729546_dt, 0.26704276_dt, 0.17427123_dt, 0.04580772_dt, 0.98797727_dt, 0.03881574_dt, 0.22868955_dt, 0.0036062_dt,  0.6006421_dt,  0.25169027_dt, 0.45649374_dt,
                               0.21031535_dt, 0.13384092_dt, 0.610149_dt,   0.7017927_dt,  0.56946445_dt, 0.25802827_dt, 0.09499919_dt, 0.96377003_dt };
    const raul::Tensor memorySeqLength{ 2.0_dt, 2.0_dt, 2.0_dt };

    // Real output
    const raul::Tensor realAlignment{ 0.47029987_dt, 0.52970016_dt, 0._dt, 0.45886216_dt, 0.5411378_dt, 0._dt, 0.51346195_dt, 0.48653802_dt, 0._dt };
    const raul::Tensor realNextState{ 1.2550101_dt, 0.6435814_dt, 0.59057367_dt, 1.3844074_dt, 1.1451901_dt, 0.91908026_dt, 0.8596647_dt, 1.0934889_dt, 0.18924046_dt };
    const raul::Tensor realMaxAttnIndices{ 1.0_dt, 1.0_dt, 0.0_dt };
    const raul::Tensor realTransitProba{ 0.5761429_dt, 0.58464974_dt, 0.603815_dt };

    // Initialization
    raul::Workflow work;
    auto& networkParameters = work.getNetworkParameters();

    // Inputs
    work.add<raul::DataLayer>("data_query", raul::DataParams{ { "query" }, 1u, 1u, queryDepth });
    work.add<raul::DataLayer>("data_state", raul::DataParams{ { "state" }, 1u, 1u, alignmentsSize });
    work.add<raul::DataLayer>("data_memory", raul::DataParams{ { "memory" }, 1u, alignmentsSize, anyNumber });
    work.add<raul::DataLayer>("data_memory_seq_length", raul::DataParams{ { "memorySeqLength" }, 1u, 1u, 1u });

    // Outputs
    work.add<raul::DataLayer>("output0", raul::DataParams{ { "realAlignment" }, 1u, 1u, alignmentsSize });
    work.add<raul::DataLayer>("output1", raul::DataParams{ { "realNextState" }, 1u, 1u, alignmentsSize });
    work.add<raul::DataLayer>("output2", raul::DataParams{ { "realMaxAttnIndices" }, 1u, 1u, 1u });
    work.add<raul::DataLayer>("output3", raul::DataParams{ { "realTransitProba" }, 1u, 1u, 1u });

    // Layer
    raul::LocationSensitiveAttentionLayer(name,
                                          raul::LocationSensitiveAttentionParams{ { "query", "state", "memory", "memorySeqLength" },
                                                                                  { "alignment", "values", "next_state", "max_attn" },
                                                                                  numUnits,
                                                                                  raul::LocationSensitiveAttentionParams::hparams{ 5, 5, true, false },
                                                                                  true,
                                                                                  true },
                                          networkParameters);

    work.preparePipelines();
    work.setBatchSize(batchSize);
    work.prepareMemoryForTraining();

    raul::MemoryManager& memory_manager = work.getMemoryManager();

    memory_manager["query"] = TORANGE(query);
    memory_manager["state"] = TORANGE(state);
    memory_manager["memory"] = TORANGE(memory);
    memory_manager["memorySeqLength"] = TORANGE(memorySeqLength);
    memory_manager["realAlignment"] = TORANGE(realAlignment);
    memory_manager["realNextState"] = TORANGE(realNextState);
    memory_manager["realMaxAttnIndices"] = TORANGE(realMaxAttnIndices);
    memory_manager["realTransitProba"] = TORANGE(realTransitProba);

    // For result stability
    memory_manager[name / "attention_variable_projection"] = TORANGE(raul::Tensor({ -0.21917742_dt, 0.19094306_dt, 0.26173806_dt, -0.31035906_dt, 0.06741166_dt, 0.22481245_dt, -0.6266_dt }));
    memory_manager[name / "query_layer" / "Weights"] =
        TORANGE(raul::Tensor({ 0.4863749_dt,   0.3023644_dt,  0.03898406_dt,  0.37956053_dt, 0.19826788_dt,  -0.01195204_dt, -0.17408913_dt, 0.25517255_dt,  0.02175409_dt,  -0.22741026_dt,
                               -0.32522815_dt, 0.0501048_dt,  -0.16088426_dt, 0.01168489_dt, -0.68866247_dt, -0.260657_dt,   0.11950135_dt,  -0.41111445_dt, -0.10058063_dt, -0.15056169_dt,
                               -0.6669365_dt,  0.51485103_dt, -0.5994427_dt,  -0.4474305_dt, -0.47616896_dt, 0.6559612_dt,   0.6870752_dt,   0.4812215_dt }));
    memory_manager[name / "location_convolution" / "Weights"] =
        TORANGE(raul::Tensor({ -0.2357244_dt, -0.23364823_dt, 0.00324789_dt,  0.24466163_dt,  0.09907997_dt, 0.3328864_dt,   -0.20317155_dt, 0.10582519_dt, -0.26607794_dt,
                               0.33100575_dt, -0.00679237_dt, -0.00274599_dt, -0.4202745_dt,  0.16615295_dt, 0.1176818_dt,   -0.26131704_dt, -0.0544498_dt, -0.27820724_dt,
                               0.23560613_dt, -0.02338243_dt, -0.24157539_dt, -0.16926107_dt, 0.31284302_dt, -0.07684425_dt, -0.26236838_dt }));
    memory_manager[name / "location_layer" / "Weights"] = TORANGE(raul::Tensor(
        { 0.3320319_dt,   0.02018768_dt,  0.2187472_dt,   -0.40487307_dt, -0.63422394_dt, -0.50598776_dt, -0.6050468_dt,  -0.4294378_dt, 0.16675436_dt, 0.38511664_dt, -0.617469_dt,  -0.21875578_dt,
          0.360884_dt,    -0.29847285_dt, -0.5667498_dt,  -0.43953767_dt, 0.3544187_dt,   -0.0836367_dt,  0.49457508_dt,  -0.5571702_dt, 0.4216774_dt,  -0.4902776_dt, -0.4861246_dt, -0.43035218_dt,
          -0.18000495_dt, 0.28884786_dt,  -0.33423108_dt, 0.13451302_dt,  0.47462338_dt,  -0.29468757_dt, -0.12878579_dt, 0.17698961_dt, 0.610215_dt,   0.20963287_dt, -0.6354886_dt }));
    memory_manager[name / "memory_layer" / "Weights"] =
        TORANGE(raul::Tensor({ -0.43844664_dt, -0.08497471_dt, 0.04186517_dt,  -0.57668185_dt, 0.17414308_dt,  -0.07473892_dt, -0.1215468_dt,  0.5256624_dt,   0.20962256_dt,  -0.6206402_dt,
                               -0.56157005_dt, 0.1126411_dt,   -0.2717067_dt,  -0.6420031_dt,  0.17148542_dt,  0.1373409_dt,   -0.07741272_dt, -0.25084403_dt, 0.53683174_dt,  0.30630547_dt,
                               -0.04506826_dt, -0.08569926_dt, 0.17889261_dt,  -0.32382998_dt, -0.32577834_dt, 0.09183967_dt,  0.5495213_dt,   -0.26685473_dt, -0.27241576_dt, 0.14984864_dt,
                               0.50568485_dt,  0.5348134_dt,   -0.00466824_dt, 0.22064257_dt,  0.3274873_dt,   0.18657899_dt,  0.50975907_dt,  0.50708616_dt,  -0.03734189_dt, 0.12296373_dt,
                               -0.37171817_dt, 0.02866983_dt,  0.6229495_dt,   0.16727936_dt,  0.37614_dt,     -0.33729702_dt, 0.05424863_dt,  -0.41042358_dt, 0.1795525_dt }));
    memory_manager[name / "transition_agent_layer" / "Weights"] = TORANGE(
        raul::Tensor({ 0.01429349_dt, -0.07985818_dt, -0.12935376_dt, 0.6964893_dt, 0.26681113_dt, -0.21800154_dt, -0.09041494_dt, 0.1429218_dt, -0.06134254_dt, 0.3573689_dt, -0.44123855_dt }));

    tools::callbacks::TensorChecker checker(
        { { "alignment", "realAlignment" }, { "next_state", "realNextState" }, { "max_attn", "realMaxAttnIndices" }, { name / "transit_proba", "realTransitProba" } }, -1_dt, eps);
    networkParameters.mCallback = checker;

    // Forward
    ASSERT_NO_THROW(work.forwardPassTraining());

    // Backward
    ASSERT_NO_THROW(work.backwardPassTraining());
}

TEST(TestLayerLocationSensitiveAttention, UseMaskNotConstrainedUseTransitionAgentUseForwardUnit)
{
    PROFILE_TEST

    // Cumulative attention mode, smoothing enabled, use transition agent, use forward and mask memory

    constexpr size_t numUnits = 7;
    constexpr size_t queryDepth = 4;
    constexpr size_t alignmentsSize = 3;
    constexpr size_t anyNumber = 7;
    constexpr size_t batchSize = 3;
    constexpr raul::dtype eps = TODTYPE(1e-5);

    raul::Name name = "attn";

    const raul::Tensor query{ 0.22413874_dt, 0.22268498_dt, 0.8552655_dt,  0.49562013_dt, 0.31110537_dt, 0.61050725_dt,
                              0.21236408_dt, 0.93036723_dt, 0.54842377_dt, 0.84664714_dt, 0.47629058_dt, 0.89816856_dt };
    const raul::Tensor state{ 0.7847103_dt, 0.11388123_dt, 0.59057367_dt, 0.9255452_dt, 0.6040523_dt, 0.91908026_dt, 0.34620273_dt, 0.6069509_dt, 0.18924046_dt };
    const raul::Tensor memory{ 0.81269646_dt, 0.07857466_dt, 0.8916855_dt,  0.16925514_dt, 0.06311357_dt, 0.54531074_dt, 0.5037316_dt,  0.9248222_dt,  0.66955376_dt, 0.9281193_dt,  0.12239242_dt,
                               0.8532245_dt,  0.90477383_dt, 0.7104306_dt,  0.40681756_dt, 0.5755513_dt,  0.8547678_dt,  0.59606934_dt, 0.77619946_dt, 0.97301054_dt, 0.06244731_dt, 0.33562684_dt,
                               0.22166848_dt, 0.32035887_dt, 0.03924382_dt, 0.06723011_dt, 0.32712245_dt, 0.49054873_dt, 0.11453211_dt, 0.34396613_dt, 0.52225363_dt, 0.30574834_dt, 0.8817626_dt,
                               0.8017194_dt,  0.9992852_dt,  0.65941477_dt, 0.1272459_dt,  0.19117236_dt, 0.65929854_dt, 0.7614676_dt,  0.75358987_dt, 0.41603255_dt, 0.94846773_dt, 0.8904344_dt,
                               0.91729546_dt, 0.26704276_dt, 0.17427123_dt, 0.04580772_dt, 0.98797727_dt, 0.03881574_dt, 0.22868955_dt, 0.0036062_dt,  0.6006421_dt,  0.25169027_dt, 0.45649374_dt,
                               0.21031535_dt, 0.13384092_dt, 0.610149_dt,   0.7017927_dt,  0.56946445_dt, 0.25802827_dt, 0.09499919_dt, 0.96377003_dt };
    const raul::Tensor memorySeqLength{ 2.0_dt, 2.0_dt, 2.0_dt };

    // Real output
    const raul::Tensor realAlignment{ 0.37113705_dt, 0.6288629_dt, 0._dt, 0.2915739_dt, 0.70842606_dt, 0._dt, 0.24358198_dt, 0.75641805_dt, 0._dt };
    const raul::Tensor realNextState{ 1.1558473_dt, 0.74274415_dt, 0.59057367_dt, 1.2171191_dt, 1.3124783_dt, 0.91908026_dt, 0.58978474_dt, 1.363369_dt, 0.18924046_dt };
    const raul::Tensor realMaxAttnIndices{ 1.0_dt, 1.0_dt, 1.0_dt };
    const raul::Tensor realTransitProba{ 0.5761429_dt, 0.58464974_dt, 0.603815_dt };

    // Initialization
    raul::Workflow work;
    auto& networkParameters = work.getNetworkParameters();

    // Inputs
    work.add<raul::DataLayer>("data_query", raul::DataParams{ { "query" }, 1u, 1u, queryDepth });
    work.add<raul::DataLayer>("data_state", raul::DataParams{ { "state" }, 1u, 1u, alignmentsSize });
    work.add<raul::DataLayer>("data_memory", raul::DataParams{ { "memory" }, 1u, alignmentsSize, anyNumber });
    work.add<raul::DataLayer>("data_memory_seq_length", raul::DataParams{ { "memorySeqLength" }, 1u, 1u, 1u });

    // Outputs
    work.add<raul::DataLayer>("output0", raul::DataParams{ { "realAlignment" }, 1u, 1u, alignmentsSize });
    work.add<raul::DataLayer>("output1", raul::DataParams{ { "realNextState" }, 1u, 1u, alignmentsSize });
    work.add<raul::DataLayer>("output2", raul::DataParams{ { "realMaxAttnIndices" }, 1u, 1u, 1u });
    work.add<raul::DataLayer>("output3", raul::DataParams{ { "realTransitProba" }, 1u, 1u, 1u });

    // Layer
    raul::LocationSensitiveAttentionLayer(name,
                                          raul::LocationSensitiveAttentionParams{ { "query", "state", "memory", "memorySeqLength" },
                                                                                  { "alignment", "values", "next_state", "max_attn" },
                                                                                  numUnits,
                                                                                  raul::LocationSensitiveAttentionParams::hparams{ 5, 5, true, false },
                                                                                  true,
                                                                                  true,
                                                                                  0.0_dt,
                                                                                  true },
                                          networkParameters);

    work.preparePipelines();
    work.setBatchSize(batchSize);
    work.prepareMemoryForTraining();

    raul::MemoryManager& memory_manager = work.getMemoryManager();

    memory_manager["query"] = TORANGE(query);
    memory_manager["state"] = TORANGE(state);
    memory_manager["memory"] = TORANGE(memory);
    memory_manager["memorySeqLength"] = TORANGE(memorySeqLength);
    memory_manager["realAlignment"] = TORANGE(realAlignment);
    memory_manager["realNextState"] = TORANGE(realNextState);
    memory_manager["realMaxAttnIndices"] = TORANGE(realMaxAttnIndices);
    memory_manager["realTransitProba"] = TORANGE(realTransitProba);

    // For result stability
    memory_manager[name / "attention_variable_projection"] = TORANGE(raul::Tensor({ -0.21917742_dt, 0.19094306_dt, 0.26173806_dt, -0.31035906_dt, 0.06741166_dt, 0.22481245_dt, -0.6266_dt }));
    memory_manager[name / "query_layer" / "Weights"] =
        TORANGE(raul::Tensor({ 0.4863749_dt,   0.3023644_dt,  0.03898406_dt,  0.37956053_dt, 0.19826788_dt,  -0.01195204_dt, -0.17408913_dt, 0.25517255_dt,  0.02175409_dt,  -0.22741026_dt,
                               -0.32522815_dt, 0.0501048_dt,  -0.16088426_dt, 0.01168489_dt, -0.68866247_dt, -0.260657_dt,   0.11950135_dt,  -0.41111445_dt, -0.10058063_dt, -0.15056169_dt,
                               -0.6669365_dt,  0.51485103_dt, -0.5994427_dt,  -0.4474305_dt, -0.47616896_dt, 0.6559612_dt,   0.6870752_dt,   0.4812215_dt }));
    memory_manager[name / "location_convolution" / "Weights"] =
        TORANGE(raul::Tensor({ -0.2357244_dt, -0.23364823_dt, 0.00324789_dt,  0.24466163_dt,  0.09907997_dt, 0.3328864_dt,   -0.20317155_dt, 0.10582519_dt, -0.26607794_dt,
                               0.33100575_dt, -0.00679237_dt, -0.00274599_dt, -0.4202745_dt,  0.16615295_dt, 0.1176818_dt,   -0.26131704_dt, -0.0544498_dt, -0.27820724_dt,
                               0.23560613_dt, -0.02338243_dt, -0.24157539_dt, -0.16926107_dt, 0.31284302_dt, -0.07684425_dt, -0.26236838_dt }));
    memory_manager[name / "location_layer" / "Weights"] = TORANGE(raul::Tensor(
        { 0.3320319_dt,   0.02018768_dt,  0.2187472_dt,   -0.40487307_dt, -0.63422394_dt, -0.50598776_dt, -0.6050468_dt,  -0.4294378_dt, 0.16675436_dt, 0.38511664_dt, -0.617469_dt,  -0.21875578_dt,
          0.360884_dt,    -0.29847285_dt, -0.5667498_dt,  -0.43953767_dt, 0.3544187_dt,   -0.0836367_dt,  0.49457508_dt,  -0.5571702_dt, 0.4216774_dt,  -0.4902776_dt, -0.4861246_dt, -0.43035218_dt,
          -0.18000495_dt, 0.28884786_dt,  -0.33423108_dt, 0.13451302_dt,  0.47462338_dt,  -0.29468757_dt, -0.12878579_dt, 0.17698961_dt, 0.610215_dt,   0.20963287_dt, -0.6354886_dt }));
    memory_manager[name / "memory_layer" / "Weights"] =
        TORANGE(raul::Tensor({ -0.43844664_dt, -0.08497471_dt, 0.04186517_dt,  -0.57668185_dt, 0.17414308_dt,  -0.07473892_dt, -0.1215468_dt,  0.5256624_dt,   0.20962256_dt,  -0.6206402_dt,
                               -0.56157005_dt, 0.1126411_dt,   -0.2717067_dt,  -0.6420031_dt,  0.17148542_dt,  0.1373409_dt,   -0.07741272_dt, -0.25084403_dt, 0.53683174_dt,  0.30630547_dt,
                               -0.04506826_dt, -0.08569926_dt, 0.17889261_dt,  -0.32382998_dt, -0.32577834_dt, 0.09183967_dt,  0.5495213_dt,   -0.26685473_dt, -0.27241576_dt, 0.14984864_dt,
                               0.50568485_dt,  0.5348134_dt,   -0.00466824_dt, 0.22064257_dt,  0.3274873_dt,   0.18657899_dt,  0.50975907_dt,  0.50708616_dt,  -0.03734189_dt, 0.12296373_dt,
                               -0.37171817_dt, 0.02866983_dt,  0.6229495_dt,   0.16727936_dt,  0.37614_dt,     -0.33729702_dt, 0.05424863_dt,  -0.41042358_dt, 0.1795525_dt }));
    memory_manager[name / "transition_agent_layer" / "Weights"] = TORANGE(
        raul::Tensor({ 0.01429349_dt, -0.07985818_dt, -0.12935376_dt, 0.6964893_dt, 0.26681113_dt, -0.21800154_dt, -0.09041494_dt, 0.1429218_dt, -0.06134254_dt, 0.3573689_dt, -0.44123855_dt }));

    tools::callbacks::TensorChecker checker(
        { { "alignment", "realAlignment" }, { "next_state", "realNextState" }, { "max_attn", "realMaxAttnIndices" }, { name / "transit_proba", "realTransitProba" } }, -1_dt, eps);
    networkParameters.mCallback = checker;

    // Forward
    ASSERT_NO_THROW(work.forwardPassTraining());

    // Backward
    ASSERT_NO_THROW(work.backwardPassTraining());
}

TEST(TestLayerLocationSensitiveAttention, EverythingEnabledNoSmoothingNoConstrainsUnit)
{
    PROFILE_TEST

    // Cumulative attention mode, smoothing enabled, use transition agent, use forward, use constraints and mask memory

    constexpr size_t numUnits = 7;
    constexpr size_t queryDepth = 4;
    constexpr size_t alignmentsSize = 3;
    constexpr size_t anyNumber = 7;
    constexpr size_t batchSize = 3;
    constexpr raul::dtype eps = TODTYPE(1e-5);

    raul::Name name = "attn";

    const raul::Tensor query{ 0.22413874_dt, 0.22268498_dt, 0.8552655_dt,  0.49562013_dt, 0.31110537_dt, 0.61050725_dt,
                              0.21236408_dt, 0.93036723_dt, 0.54842377_dt, 0.84664714_dt, 0.47629058_dt, 0.89816856_dt };
    const raul::Tensor state{ 0.7847103_dt, 0.11388123_dt, 0.59057367_dt, 0.9255452_dt, 0.6040523_dt, 0.91908026_dt, 0.34620273_dt, 0.6069509_dt, 0.18924046_dt };
    const raul::Tensor memory{ 0.81269646_dt, 0.07857466_dt, 0.8916855_dt,  0.16925514_dt, 0.06311357_dt, 0.54531074_dt, 0.5037316_dt,  0.9248222_dt,  0.66955376_dt, 0.9281193_dt,  0.12239242_dt,
                               0.8532245_dt,  0.90477383_dt, 0.7104306_dt,  0.40681756_dt, 0.5755513_dt,  0.8547678_dt,  0.59606934_dt, 0.77619946_dt, 0.97301054_dt, 0.06244731_dt, 0.33562684_dt,
                               0.22166848_dt, 0.32035887_dt, 0.03924382_dt, 0.06723011_dt, 0.32712245_dt, 0.49054873_dt, 0.11453211_dt, 0.34396613_dt, 0.52225363_dt, 0.30574834_dt, 0.8817626_dt,
                               0.8017194_dt,  0.9992852_dt,  0.65941477_dt, 0.1272459_dt,  0.19117236_dt, 0.65929854_dt, 0.7614676_dt,  0.75358987_dt, 0.41603255_dt, 0.94846773_dt, 0.8904344_dt,
                               0.91729546_dt, 0.26704276_dt, 0.17427123_dt, 0.04580772_dt, 0.98797727_dt, 0.03881574_dt, 0.22868955_dt, 0.0036062_dt,  0.6006421_dt,  0.25169027_dt, 0.45649374_dt,
                               0.21031535_dt, 0.13384092_dt, 0.610149_dt,   0.7017927_dt,  0.56946445_dt, 0.25802827_dt, 0.09499919_dt, 0.96377003_dt };
    const raul::Tensor memorySeqLength{ 2.0_dt, 2.0_dt, 2.0_dt };

    // Real output
    const raul::Tensor realAlignment{ 0.35116285_dt, 0.64883715_dt, 0._dt, 0.26699635_dt, 0.7330037_dt, 0._dt, 0.25074086_dt, 0.7492592_dt, 0._dt };
    const raul::Tensor realNextState{ 1.1358731_dt, 0.7627184_dt, 0.59057367_dt, 1.1925416_dt, 1.3370559_dt, 0.91908026_dt, 0.5969436_dt, 1.35621_dt, 0.18924046_dt };
    const raul::Tensor realMaxAttnIndices{ 1.0_dt, 1.0_dt, 1.0_dt };
    const raul::Tensor realTransitProba{ 0.5761429_dt, 0.58464974_dt, 0.603815_dt };

    // Initialization
    raul::Workflow work;
    auto& networkParameters = work.getNetworkParameters();

    // Inputs
    work.add<raul::DataLayer>("data_query", raul::DataParams{ { "query" }, 1u, 1u, queryDepth });
    work.add<raul::DataLayer>("data_state", raul::DataParams{ { "state" }, 1u, 1u, alignmentsSize });
    work.add<raul::DataLayer>("data_memory", raul::DataParams{ { "memory" }, 1u, alignmentsSize, anyNumber });
    work.add<raul::DataLayer>("data_memory_seq_length", raul::DataParams{ { "memorySeqLength" }, 1u, 1u, 1u });

    // Outputs
    work.add<raul::DataLayer>("output0", raul::DataParams{ { "realAlignment" }, 1u, 1u, alignmentsSize });
    work.add<raul::DataLayer>("output1", raul::DataParams{ { "realNextState" }, 1u, 1u, alignmentsSize });
    work.add<raul::DataLayer>("output2", raul::DataParams{ { "realMaxAttnIndices" }, 1u, 1u, 1u });
    work.add<raul::DataLayer>("output3", raul::DataParams{ { "realTransitProba" }, 1u, 1u, 1u });

    // Layer
    raul::LocationSensitiveAttentionLayer(name,
                                          raul::LocationSensitiveAttentionParams{ { "query", "state", "memory", "memorySeqLength" },
                                                                                  { "alignment", "values", "next_state", "max_attn" },
                                                                                  numUnits,
                                                                                  raul::LocationSensitiveAttentionParams::hparams{ 5, 5, true, false },
                                                                                  true,
                                                                                  false,
                                                                                  0.0_dt,
                                                                                  true },
                                          networkParameters);

    work.preparePipelines();
    work.setBatchSize(batchSize);
    work.prepareMemoryForTraining();

    raul::MemoryManager& memory_manager = work.getMemoryManager();

    memory_manager["query"] = TORANGE(query);
    memory_manager["state"] = TORANGE(state);
    memory_manager["memory"] = TORANGE(memory);
    memory_manager["memorySeqLength"] = TORANGE(memorySeqLength);
    memory_manager["realAlignment"] = TORANGE(realAlignment);
    memory_manager["realNextState"] = TORANGE(realNextState);
    memory_manager["realMaxAttnIndices"] = TORANGE(realMaxAttnIndices);
    memory_manager["realTransitProba"] = TORANGE(realTransitProba);

    // For result stability
    memory_manager[name / "attention_variable_projection"] = TORANGE(raul::Tensor({ -0.21917742_dt, 0.19094306_dt, 0.26173806_dt, -0.31035906_dt, 0.06741166_dt, 0.22481245_dt, -0.6266_dt }));
    memory_manager[name / "query_layer" / "Weights"] =
        TORANGE(raul::Tensor({ 0.4863749_dt,   0.3023644_dt,  0.03898406_dt,  0.37956053_dt, 0.19826788_dt,  -0.01195204_dt, -0.17408913_dt, 0.25517255_dt,  0.02175409_dt,  -0.22741026_dt,
                               -0.32522815_dt, 0.0501048_dt,  -0.16088426_dt, 0.01168489_dt, -0.68866247_dt, -0.260657_dt,   0.11950135_dt,  -0.41111445_dt, -0.10058063_dt, -0.15056169_dt,
                               -0.6669365_dt,  0.51485103_dt, -0.5994427_dt,  -0.4474305_dt, -0.47616896_dt, 0.6559612_dt,   0.6870752_dt,   0.4812215_dt }));
    memory_manager[name / "location_convolution" / "Weights"] =
        TORANGE(raul::Tensor({ -0.2357244_dt, -0.23364823_dt, 0.00324789_dt,  0.24466163_dt,  0.09907997_dt, 0.3328864_dt,   -0.20317155_dt, 0.10582519_dt, -0.26607794_dt,
                               0.33100575_dt, -0.00679237_dt, -0.00274599_dt, -0.4202745_dt,  0.16615295_dt, 0.1176818_dt,   -0.26131704_dt, -0.0544498_dt, -0.27820724_dt,
                               0.23560613_dt, -0.02338243_dt, -0.24157539_dt, -0.16926107_dt, 0.31284302_dt, -0.07684425_dt, -0.26236838_dt }));
    memory_manager[name / "location_layer" / "Weights"] = TORANGE(raul::Tensor(
        { 0.3320319_dt,   0.02018768_dt,  0.2187472_dt,   -0.40487307_dt, -0.63422394_dt, -0.50598776_dt, -0.6050468_dt,  -0.4294378_dt, 0.16675436_dt, 0.38511664_dt, -0.617469_dt,  -0.21875578_dt,
          0.360884_dt,    -0.29847285_dt, -0.5667498_dt,  -0.43953767_dt, 0.3544187_dt,   -0.0836367_dt,  0.49457508_dt,  -0.5571702_dt, 0.4216774_dt,  -0.4902776_dt, -0.4861246_dt, -0.43035218_dt,
          -0.18000495_dt, 0.28884786_dt,  -0.33423108_dt, 0.13451302_dt,  0.47462338_dt,  -0.29468757_dt, -0.12878579_dt, 0.17698961_dt, 0.610215_dt,   0.20963287_dt, -0.6354886_dt }));
    memory_manager[name / "memory_layer" / "Weights"] =
        TORANGE(raul::Tensor({ -0.43844664_dt, -0.08497471_dt, 0.04186517_dt,  -0.57668185_dt, 0.17414308_dt,  -0.07473892_dt, -0.1215468_dt,  0.5256624_dt,   0.20962256_dt,  -0.6206402_dt,
                               -0.56157005_dt, 0.1126411_dt,   -0.2717067_dt,  -0.6420031_dt,  0.17148542_dt,  0.1373409_dt,   -0.07741272_dt, -0.25084403_dt, 0.53683174_dt,  0.30630547_dt,
                               -0.04506826_dt, -0.08569926_dt, 0.17889261_dt,  -0.32382998_dt, -0.32577834_dt, 0.09183967_dt,  0.5495213_dt,   -0.26685473_dt, -0.27241576_dt, 0.14984864_dt,
                               0.50568485_dt,  0.5348134_dt,   -0.00466824_dt, 0.22064257_dt,  0.3274873_dt,   0.18657899_dt,  0.50975907_dt,  0.50708616_dt,  -0.03734189_dt, 0.12296373_dt,
                               -0.37171817_dt, 0.02866983_dt,  0.6229495_dt,   0.16727936_dt,  0.37614_dt,     -0.33729702_dt, 0.05424863_dt,  -0.41042358_dt, 0.1795525_dt }));
    memory_manager[name / "transition_agent_layer" / "Weights"] = TORANGE(
        raul::Tensor({ 0.01429349_dt, -0.07985818_dt, -0.12935376_dt, 0.6964893_dt, 0.26681113_dt, -0.21800154_dt, -0.09041494_dt, 0.1429218_dt, -0.06134254_dt, 0.3573689_dt, -0.44123855_dt }));

    tools::callbacks::TensorChecker checker(
        { { "alignment", "realAlignment" }, { "next_state", "realNextState" }, { "max_attn", "realMaxAttnIndices" }, { name / "transit_proba", "realTransitProba" } }, -1_dt, eps);
    networkParameters.mCallback = checker;

    // Forward
    ASSERT_NO_THROW(work.forwardPassTraining());

    // Backward
    ASSERT_NO_THROW(work.backwardPassTraining());
}

TEST(TestLayerLocationSensitiveAttention, EverythingEnabledUnit)
{
    PROFILE_TEST

    // Cumulative attention mode, smoothing enabled, use transition agent, use forward, use constraints and mask memory

    constexpr size_t numUnits = 7;
    constexpr size_t queryDepth = 4;
    constexpr size_t alignmentsSize = 3;
    constexpr size_t anyNumber = 7;
    constexpr size_t batchSize = 3;
    constexpr raul::dtype eps = TODTYPE(1e-5);

    raul::Name name = "attn";

    const raul::Tensor query{ 0.22413874_dt, 0.22268498_dt, 0.8552655_dt,  0.49562013_dt, 0.31110537_dt, 0.61050725_dt,
                              0.21236408_dt, 0.93036723_dt, 0.54842377_dt, 0.84664714_dt, 0.47629058_dt, 0.89816856_dt };
    const raul::Tensor state{ 0.7847103_dt, 0.11388123_dt, 0.59057367_dt, 0.9255452_dt, 0.6040523_dt, 0.91908026_dt, 0.34620273_dt, 0.6069509_dt, 0.18924046_dt };
    const raul::Tensor memory{ 0.81269646_dt, 0.07857466_dt, 0.8916855_dt,  0.16925514_dt, 0.06311357_dt, 0.54531074_dt, 0.5037316_dt,  0.9248222_dt,  0.66955376_dt, 0.9281193_dt,  0.12239242_dt,
                               0.8532245_dt,  0.90477383_dt, 0.7104306_dt,  0.40681756_dt, 0.5755513_dt,  0.8547678_dt,  0.59606934_dt, 0.77619946_dt, 0.97301054_dt, 0.06244731_dt, 0.33562684_dt,
                               0.22166848_dt, 0.32035887_dt, 0.03924382_dt, 0.06723011_dt, 0.32712245_dt, 0.49054873_dt, 0.11453211_dt, 0.34396613_dt, 0.52225363_dt, 0.30574834_dt, 0.8817626_dt,
                               0.8017194_dt,  0.9992852_dt,  0.65941477_dt, 0.1272459_dt,  0.19117236_dt, 0.65929854_dt, 0.7614676_dt,  0.75358987_dt, 0.41603255_dt, 0.94846773_dt, 0.8904344_dt,
                               0.91729546_dt, 0.26704276_dt, 0.17427123_dt, 0.04580772_dt, 0.98797727_dt, 0.03881574_dt, 0.22868955_dt, 0.0036062_dt,  0.6006421_dt,  0.25169027_dt, 0.45649374_dt,
                               0.21031535_dt, 0.13384092_dt, 0.610149_dt,   0.7017927_dt,  0.56946445_dt, 0.25802827_dt, 0.09499919_dt, 0.96377003_dt };
    const raul::Tensor memorySeqLength{ 2.0_dt, 2.0_dt, 2.0_dt };

    // Real output
    const raul::Tensor realAlignment{ 0.22365978_dt, 0.56735224_dt, 0.20898801_dt, 0.10303543_dt, 0.49437153_dt, 0.40259302_dt, 0.04951486_dt, 0.48649505_dt, 0.46399006_dt };
    const raul::Tensor realNextState{ 1.00837_dt, 0.68123347_dt, 0.7995617_dt, 1.0285807_dt, 1.0984238_dt, 1.3216733_dt, 0.3957176_dt, 1.0934459_dt, 0.65323055_dt };
    const raul::Tensor realMaxAttnIndices{ 1.0_dt, 1.0_dt, 1.0_dt };
    const raul::Tensor realTransitProba{ 0.5761429_dt, 0.58464974_dt, 0.603815_dt };

    // Initialization
    raul::Workflow work;
    auto& networkParameters = work.getNetworkParameters();

    // Inputs
    work.add<raul::DataLayer>("data_query", raul::DataParams{ { "query" }, 1u, 1u, queryDepth });
    work.add<raul::DataLayer>("data_state", raul::DataParams{ { "state" }, 1u, 1u, alignmentsSize });
    work.add<raul::DataLayer>("data_memory", raul::DataParams{ { "memory" }, 1u, alignmentsSize, anyNumber });
    work.add<raul::DataLayer>("data_memory_seq_length", raul::DataParams{ { "memorySeqLength" }, 1u, 1u, 1u });

    // Outputs
    work.add<raul::DataLayer>("output0", raul::DataParams{ { "realAlignment" }, 1u, 1u, alignmentsSize });
    work.add<raul::DataLayer>("output1", raul::DataParams{ { "realNextState" }, 1u, 1u, alignmentsSize });
    work.add<raul::DataLayer>("output2", raul::DataParams{ { "realMaxAttnIndices" }, 1u, 1u, 1u });
    work.add<raul::DataLayer>("output3", raul::DataParams{ { "realTransitProba" }, 1u, 1u, 1u });

    // Layer
    raul::LocationSensitiveAttentionLayer(name,
                                          raul::LocationSensitiveAttentionParams{ { "query", "state", "memory", "memorySeqLength" },
                                                                                  { "alignment", "values", "next_state", "max_attn" },
                                                                                  numUnits,
                                                                                  raul::LocationSensitiveAttentionParams::hparams{ 5, 5, true, true },
                                                                                  true,
                                                                                  true,
                                                                                  0.0_dt,
                                                                                  true },
                                          networkParameters);

    work.preparePipelines();
    work.setBatchSize(batchSize);
    work.prepareMemoryForTraining();

    raul::MemoryManager& memory_manager = work.getMemoryManager();

    memory_manager["query"] = TORANGE(query);
    memory_manager["state"] = TORANGE(state);
    memory_manager["memory"] = TORANGE(memory);
    memory_manager["memorySeqLength"] = TORANGE(memorySeqLength);
    memory_manager["realAlignment"] = TORANGE(realAlignment);
    memory_manager["realNextState"] = TORANGE(realNextState);
    memory_manager["realMaxAttnIndices"] = TORANGE(realMaxAttnIndices);
    memory_manager["realTransitProba"] = TORANGE(realTransitProba);

    // For result stability
    memory_manager[name / "attention_variable_projection"] = TORANGE(raul::Tensor({ -0.21917742_dt, 0.19094306_dt, 0.26173806_dt, -0.31035906_dt, 0.06741166_dt, 0.22481245_dt, -0.6266_dt }));
    memory_manager[name / "query_layer" / "Weights"] =
        TORANGE(raul::Tensor({ 0.4863749_dt,   0.3023644_dt,  0.03898406_dt,  0.37956053_dt, 0.19826788_dt,  -0.01195204_dt, -0.17408913_dt, 0.25517255_dt,  0.02175409_dt,  -0.22741026_dt,
                               -0.32522815_dt, 0.0501048_dt,  -0.16088426_dt, 0.01168489_dt, -0.68866247_dt, -0.260657_dt,   0.11950135_dt,  -0.41111445_dt, -0.10058063_dt, -0.15056169_dt,
                               -0.6669365_dt,  0.51485103_dt, -0.5994427_dt,  -0.4474305_dt, -0.47616896_dt, 0.6559612_dt,   0.6870752_dt,   0.4812215_dt }));
    memory_manager[name / "location_convolution" / "Weights"] =
        TORANGE(raul::Tensor({ -0.2357244_dt, -0.23364823_dt, 0.00324789_dt,  0.24466163_dt,  0.09907997_dt, 0.3328864_dt,   -0.20317155_dt, 0.10582519_dt, -0.26607794_dt,
                               0.33100575_dt, -0.00679237_dt, -0.00274599_dt, -0.4202745_dt,  0.16615295_dt, 0.1176818_dt,   -0.26131704_dt, -0.0544498_dt, -0.27820724_dt,
                               0.23560613_dt, -0.02338243_dt, -0.24157539_dt, -0.16926107_dt, 0.31284302_dt, -0.07684425_dt, -0.26236838_dt }));
    memory_manager[name / "location_layer" / "Weights"] = TORANGE(raul::Tensor(
        { 0.3320319_dt,   0.02018768_dt,  0.2187472_dt,   -0.40487307_dt, -0.63422394_dt, -0.50598776_dt, -0.6050468_dt,  -0.4294378_dt, 0.16675436_dt, 0.38511664_dt, -0.617469_dt,  -0.21875578_dt,
          0.360884_dt,    -0.29847285_dt, -0.5667498_dt,  -0.43953767_dt, 0.3544187_dt,   -0.0836367_dt,  0.49457508_dt,  -0.5571702_dt, 0.4216774_dt,  -0.4902776_dt, -0.4861246_dt, -0.43035218_dt,
          -0.18000495_dt, 0.28884786_dt,  -0.33423108_dt, 0.13451302_dt,  0.47462338_dt,  -0.29468757_dt, -0.12878579_dt, 0.17698961_dt, 0.610215_dt,   0.20963287_dt, -0.6354886_dt }));
    memory_manager[name / "memory_layer" / "Weights"] =
        TORANGE(raul::Tensor({ -0.43844664_dt, -0.08497471_dt, 0.04186517_dt,  -0.57668185_dt, 0.17414308_dt,  -0.07473892_dt, -0.1215468_dt,  0.5256624_dt,   0.20962256_dt,  -0.6206402_dt,
                               -0.56157005_dt, 0.1126411_dt,   -0.2717067_dt,  -0.6420031_dt,  0.17148542_dt,  0.1373409_dt,   -0.07741272_dt, -0.25084403_dt, 0.53683174_dt,  0.30630547_dt,
                               -0.04506826_dt, -0.08569926_dt, 0.17889261_dt,  -0.32382998_dt, -0.32577834_dt, 0.09183967_dt,  0.5495213_dt,   -0.26685473_dt, -0.27241576_dt, 0.14984864_dt,
                               0.50568485_dt,  0.5348134_dt,   -0.00466824_dt, 0.22064257_dt,  0.3274873_dt,   0.18657899_dt,  0.50975907_dt,  0.50708616_dt,  -0.03734189_dt, 0.12296373_dt,
                               -0.37171817_dt, 0.02866983_dt,  0.6229495_dt,   0.16727936_dt,  0.37614_dt,     -0.33729702_dt, 0.05424863_dt,  -0.41042358_dt, 0.1795525_dt }));
    memory_manager[name / "transition_agent_layer" / "Weights"] = TORANGE(
        raul::Tensor({ 0.01429349_dt, -0.07985818_dt, -0.12935376_dt, 0.6964893_dt, 0.26681113_dt, -0.21800154_dt, -0.09041494_dt, 0.1429218_dt, -0.06134254_dt, 0.3573689_dt, -0.44123855_dt }));

    tools::callbacks::TensorChecker checker(
        { { "alignment", "realAlignment" }, { "next_state", "realNextState" }, { "max_attn", "realMaxAttnIndices" }, { name / "transit_proba", "realTransitProba" } }, -1_dt, eps);
    networkParameters.mCallback = checker;

    // Forward
    ASSERT_NO_THROW(work.forwardPassTraining());

    // Backward
    ASSERT_NO_THROW(work.backwardPassTraining());
}

TEST(TestLayerLocationSensitiveAttention, DoubleStepEverythingEnabledUnit)
{
    PROFILE_TEST

    // Cumulative attention mode, smoothing enabled, use transition agent, use forward, use constraints and mask memory

    constexpr size_t numUnits = 7;
    constexpr size_t queryDepth = 4;
    constexpr size_t alignmentsSize = 3;
    constexpr size_t anyNumber = 7;
    constexpr size_t batchSize = 3;
    constexpr raul::dtype eps = TODTYPE(1e-5);

    raul::Name parent = "pattn";
    raul::Name child = "cattn";

    const raul::Tensor query1{ 0.22413874_dt, 0.22268498_dt, 0.8552655_dt,  0.49562013_dt, 0.31110537_dt, 0.61050725_dt,
                               0.21236408_dt, 0.93036723_dt, 0.54842377_dt, 0.84664714_dt, 0.47629058_dt, 0.89816856_dt };
    const raul::Tensor query2{ 0.63547623_dt, 0.44589663_dt, 0.6047574_dt,  0.82557225_dt, 0.58478403_dt, 0.04986751_dt,
                               0.9572661_dt,  0.20333457_dt, 0.11299467_dt, 0.05475962_dt, 0.2828188_dt,  0.5192108_dt };
    const raul::Tensor state{ 0.7847103_dt, 0.11388123_dt, 0.59057367_dt, 0.9255452_dt, 0.6040523_dt, 0.91908026_dt, 0.34620273_dt, 0.6069509_dt, 0.18924046_dt };
    const raul::Tensor memory{ 0.81269646_dt, 0.07857466_dt, 0.8916855_dt,  0.16925514_dt, 0.06311357_dt, 0.54531074_dt, 0.5037316_dt,  0.9248222_dt,  0.66955376_dt, 0.9281193_dt,  0.12239242_dt,
                               0.8532245_dt,  0.90477383_dt, 0.7104306_dt,  0.40681756_dt, 0.5755513_dt,  0.8547678_dt,  0.59606934_dt, 0.77619946_dt, 0.97301054_dt, 0.06244731_dt, 0.33562684_dt,
                               0.22166848_dt, 0.32035887_dt, 0.03924382_dt, 0.06723011_dt, 0.32712245_dt, 0.49054873_dt, 0.11453211_dt, 0.34396613_dt, 0.52225363_dt, 0.30574834_dt, 0.8817626_dt,
                               0.8017194_dt,  0.9992852_dt,  0.65941477_dt, 0.1272459_dt,  0.19117236_dt, 0.65929854_dt, 0.7614676_dt,  0.75358987_dt, 0.41603255_dt, 0.94846773_dt, 0.8904344_dt,
                               0.91729546_dt, 0.26704276_dt, 0.17427123_dt, 0.04580772_dt, 0.98797727_dt, 0.03881574_dt, 0.22868955_dt, 0.0036062_dt,  0.6006421_dt,  0.25169027_dt, 0.45649374_dt,
                               0.21031535_dt, 0.13384092_dt, 0.610149_dt,   0.7017927_dt,  0.56946445_dt, 0.25802827_dt, 0.09499919_dt, 0.96377003_dt };
    const raul::Tensor memorySeqLength{ 2.0_dt, 2.0_dt, 2.0_dt };

    const raul::Tensor bias{ 0.6910331_dt, 0.7831433_dt, 0.05777133_dt, 0.9138534_dt, 0.43685377_dt, 0.8047224_dt, 0.88028264_dt, 0.31608927_dt, 0.57692194_dt };
    const raul::Tensor multiplier{ 0.48647678_dt, 0.69728816_dt, 0.25368452_dt, 0.09666646_dt, 0.90073335_dt, 0.17544937_dt, 0.8146868_dt, 0.1502955_dt, 0.5608002_dt };

    // Real output
    const raul::Tensor realAlignment1{ 0.22365978_dt, 0.56735224_dt, 0.20898801_dt, 0.10303543_dt, 0.49437153_dt, 0.40259302_dt, 0.04951486_dt, 0.48649505_dt, 0.46399006_dt };
    const raul::Tensor realNextState1{ 1.00837_dt, 0.68123347_dt, 0.7995617_dt, 1.0285807_dt, 1.0984238_dt, 1.3216733_dt, 0.3957176_dt, 1.0934459_dt, 0.65323055_dt };
    const raul::Tensor realMaxAttnIndices1{ 1.0_dt, 1.0_dt, 1.0_dt };

    const raul::Tensor realAlignment2{ 0.05648708_dt, 0.5683527_dt, 0.37516022_dt, 0.09937368_dt, 0.42305422_dt, 0.47757202_dt, 0.11208147_dt, 0.44273034_dt, 0.4451882_dt };
    const raul::Tensor realNextState2{ 0.97117996_dt, 1.9188483_dt, 0.64191955_dt, 1.1162626_dt, 1.3542795_dt, 1.6848874_dt, 1.0418789_dt, 1.2453146_dt, 1.4861002_dt };
    const raul::Tensor realMaxAttnIndices2{ 1.0_dt, 2.0_dt, 2.0_dt };

    const raul::Tensor realFinalResult{ 0.02747965_dt, 0.3963056_dt, 0.09517234_dt, 0.0096061_dt, 0.38105905_dt, 0.08378971_dt, 0.09131129_dt, 0.06654038_dt, 0.24966162_dt };

    // Initialization
    raul::Workflow work;
    auto& networkParameters = work.getNetworkParameters();

    // Inputs
    work.add<raul::DataLayer>("data_query", raul::DataParams{ { "query1", "query2" }, 1u, 1u, queryDepth });
    work.add<raul::DataLayer>("data_state", raul::DataParams{ { "state", "bias", "multiplier", "realFinalResult" }, 1u, 1u, alignmentsSize });
    work.add<raul::DataLayer>("data_memory", raul::DataParams{ { "memory1", "memory2" }, 1u, alignmentsSize, anyNumber });
    work.add<raul::DataLayer>("data_memory_seq_length", raul::DataParams{ { "memorySeqLength1", "memorySeqLength2" }, 1u, 1u, 1u });

    // Outputs
    work.add<raul::DataLayer>("output0", raul::DataParams{ { "realAlignment1", "realAlignment2" }, 1u, 1u, alignmentsSize });
    work.add<raul::DataLayer>("output1", raul::DataParams{ { "realNextState1", "realNextState2" }, 1u, 1u, alignmentsSize });
    work.add<raul::DataLayer>("output2", raul::DataParams{ { "realMaxAttnIndices1", "realMaxAttnIndices2" }, 1u, 1u, 1u });

    // Layer
    raul::LocationSensitiveAttentionLayer(parent,
                                          raul::LocationSensitiveAttentionParams{ { "query1", "state", "memory1", "memorySeqLength1" },
                                                                                  { "alignment1", "values", "next_state1", "max_attn1" },
                                                                                  numUnits,
                                                                                  raul::LocationSensitiveAttentionParams::hparams{ 5, 5, true, true },
                                                                                  true,
                                                                                  true,
                                                                                  0.0_dt,
                                                                                  true },
                                          networkParameters);
    work.add<raul::ElementWiseSumLayer>("calculate_biased_state", raul::ElementWiseLayerParams{ { "alignment1", "bias" }, { "biased_state" } });
    raul::LocationSensitiveAttentionLayer(child,
                                          raul::LocationSensitiveAttentionParams{ { "query2", "biased_state", "memory2", "memorySeqLength2" },
                                                                                  { "alignment2", "next_state2", "max_attn2" },
                                                                                  parent,
                                                                                  numUnits,
                                                                                  raul::LocationSensitiveAttentionParams::hparams{ 5, 5, true, true },
                                                                                  true,
                                                                                  true,
                                                                                  0.0_dt,
                                                                                  true },
                                          networkParameters);
    work.add<raul::ElementWiseMulLayer>("calculate_final_result", raul::ElementWiseLayerParams{ { "alignment2", "multiplier" }, { "final_state" } });

    work.preparePipelines();
    work.setBatchSize(batchSize);
    work.prepareMemoryForTraining();

    raul::MemoryManager& memory_manager = work.getMemoryManager();

    memory_manager["query1"] = TORANGE(query1);
    memory_manager["query2"] = TORANGE(query2);
    memory_manager["state"] = TORANGE(state);
    memory_manager["memory1"] = TORANGE(memory);
    memory_manager["memory2"] = TORANGE(memory);
    memory_manager["memorySeqLength1"] = TORANGE(memorySeqLength);
    memory_manager["memorySeqLength2"] = TORANGE(memorySeqLength);
    memory_manager["bias"] = TORANGE(bias);
    memory_manager["multiplier"] = TORANGE(multiplier);
    memory_manager["realAlignment1"] = TORANGE(realAlignment1);
    memory_manager["realNextState1"] = TORANGE(realNextState1);
    memory_manager["realMaxAttnIndices1"] = TORANGE(realMaxAttnIndices1);
    memory_manager["realAlignment2"] = TORANGE(realAlignment2);
    memory_manager["realNextState2"] = TORANGE(realNextState2);
    memory_manager["realMaxAttnIndices2"] = TORANGE(realMaxAttnIndices2);
    memory_manager["realFinalResult"] = TORANGE(realFinalResult);

    // For result stability
    memory_manager[parent / "attention_variable_projection"] = TORANGE(raul::Tensor({ -0.21917742_dt, 0.19094306_dt, 0.26173806_dt, -0.31035906_dt, 0.06741166_dt, 0.22481245_dt, -0.6266_dt }));
    memory_manager[parent / "query_layer" / "Weights"] =
        TORANGE(raul::Tensor({ 0.4863749_dt,   0.3023644_dt,  0.03898406_dt,  0.37956053_dt, 0.19826788_dt,  -0.01195204_dt, -0.17408913_dt, 0.25517255_dt,  0.02175409_dt,  -0.22741026_dt,
                               -0.32522815_dt, 0.0501048_dt,  -0.16088426_dt, 0.01168489_dt, -0.68866247_dt, -0.260657_dt,   0.11950135_dt,  -0.41111445_dt, -0.10058063_dt, -0.15056169_dt,
                               -0.6669365_dt,  0.51485103_dt, -0.5994427_dt,  -0.4474305_dt, -0.47616896_dt, 0.6559612_dt,   0.6870752_dt,   0.4812215_dt }));
    memory_manager[parent / "location_convolution" / "Weights"] =
        TORANGE(raul::Tensor({ -0.2357244_dt, -0.23364823_dt, 0.00324789_dt,  0.24466163_dt,  0.09907997_dt, 0.3328864_dt,   -0.20317155_dt, 0.10582519_dt, -0.26607794_dt,
                               0.33100575_dt, -0.00679237_dt, -0.00274599_dt, -0.4202745_dt,  0.16615295_dt, 0.1176818_dt,   -0.26131704_dt, -0.0544498_dt, -0.27820724_dt,
                               0.23560613_dt, -0.02338243_dt, -0.24157539_dt, -0.16926107_dt, 0.31284302_dt, -0.07684425_dt, -0.26236838_dt }));
    memory_manager[parent / "location_layer" / "Weights"] = TORANGE(raul::Tensor(
        { 0.3320319_dt,   0.02018768_dt,  0.2187472_dt,   -0.40487307_dt, -0.63422394_dt, -0.50598776_dt, -0.6050468_dt,  -0.4294378_dt, 0.16675436_dt, 0.38511664_dt, -0.617469_dt,  -0.21875578_dt,
          0.360884_dt,    -0.29847285_dt, -0.5667498_dt,  -0.43953767_dt, 0.3544187_dt,   -0.0836367_dt,  0.49457508_dt,  -0.5571702_dt, 0.4216774_dt,  -0.4902776_dt, -0.4861246_dt, -0.43035218_dt,
          -0.18000495_dt, 0.28884786_dt,  -0.33423108_dt, 0.13451302_dt,  0.47462338_dt,  -0.29468757_dt, -0.12878579_dt, 0.17698961_dt, 0.610215_dt,   0.20963287_dt, -0.6354886_dt }));
    memory_manager[parent / "memory_layer" / "Weights"] =
        TORANGE(raul::Tensor({ -0.43844664_dt, -0.08497471_dt, 0.04186517_dt,  -0.57668185_dt, 0.17414308_dt,  -0.07473892_dt, -0.1215468_dt,  0.5256624_dt,   0.20962256_dt,  -0.6206402_dt,
                               -0.56157005_dt, 0.1126411_dt,   -0.2717067_dt,  -0.6420031_dt,  0.17148542_dt,  0.1373409_dt,   -0.07741272_dt, -0.25084403_dt, 0.53683174_dt,  0.30630547_dt,
                               -0.04506826_dt, -0.08569926_dt, 0.17889261_dt,  -0.32382998_dt, -0.32577834_dt, 0.09183967_dt,  0.5495213_dt,   -0.26685473_dt, -0.27241576_dt, 0.14984864_dt,
                               0.50568485_dt,  0.5348134_dt,   -0.00466824_dt, 0.22064257_dt,  0.3274873_dt,   0.18657899_dt,  0.50975907_dt,  0.50708616_dt,  -0.03734189_dt, 0.12296373_dt,
                               -0.37171817_dt, 0.02866983_dt,  0.6229495_dt,   0.16727936_dt,  0.37614_dt,     -0.33729702_dt, 0.05424863_dt,  -0.41042358_dt, 0.1795525_dt }));
    memory_manager[parent / "transition_agent_layer" / "Weights"] = TORANGE(
        raul::Tensor({ 0.01429349_dt, -0.07985818_dt, -0.12935376_dt, 0.6964893_dt, 0.26681113_dt, -0.21800154_dt, -0.09041494_dt, 0.1429218_dt, -0.06134254_dt, 0.3573689_dt, -0.44123855_dt }));

    tools::callbacks::TensorChecker checker({ { "alignment1", "realAlignment1" },
                                              { "next_state1", "realNextState1" },
                                              { "max_attn1", "realMaxAttnIndices1" },
                                              { "alignment2", "realAlignment2" },
                                              { "next_state2", "realNextState2" },
                                              { "max_attn2", "realMaxAttnIndices2" },
                                              { "final_state", "realFinalResult" } },
                                            -1_dt,
                                            eps);
    networkParameters.mCallback = checker;

    // Forward
    ASSERT_NO_THROW(work.forwardPassTraining());

    // Backward
    ASSERT_NO_THROW(work.backwardPassTraining());
}

}