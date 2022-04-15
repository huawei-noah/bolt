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
#include <tests/tools/callbacks/TensorChecker.h>

#include <training/base/common/Common.h>
#include <training/base/common/MemoryManager.h>
#include <training/base/layers/basic/DataLayer.h>
#include <training/base/layers/composite/DynamicConvolutionAttentionLayer.h>
#include <training/compiler/Workflow.h>

namespace UT
{

TEST(TestLayerDynamicConvolutionAttention, IncorrectParamsUnit)
{
    PROFILE_TEST
    const size_t numUnits = 4;
    const size_t queryDepth = 3;
    const size_t alignmentsSize = 2;
    const size_t anyNumber = 3;

    const raul::Name parent = "parent";

    // Wrong params
    raul::DynamicConvolutionAttentionParams incorrectParams[]{
        { raul::DynamicConvolutionAttentionParams{
            { "query", "state", "memory" }, { "alignment", "next_state", "max_attn" }, numUnits, raul::DynamicConvolutionAttentionParams::hparams{ 1, 2, 11, 1, 0.0_dt, 0.0_dt }, false } },
        { raul::DynamicConvolutionAttentionParams{
            { "query", "state", "memory" }, { "alignment", "values", "next_state", "max_attn" }, numUnits, raul::DynamicConvolutionAttentionParams::hparams{ 1, 2, 11, 1, 0.0_dt, 0.0_dt }, false } },
        { raul::DynamicConvolutionAttentionParams{ { "query", "state", "memory" }, { "alignment" }, numUnits, raul::DynamicConvolutionAttentionParams::hparams{ 1, 2, 11, 1, 0.0_dt, 0.0_dt }, true } },
        { raul::DynamicConvolutionAttentionParams{
            { "query", "state", "memory" }, { "alignment", "values", "next_state", "max_attn" }, numUnits, raul::DynamicConvolutionAttentionParams::hparams{ 1, 2, 11, 1, 0.0_dt, 0.0_dt }, true } },
        { raul::DynamicConvolutionAttentionParams{
            { "query", "state", "memory" }, { "alignment", "next_state", "max_attn" }, parent, numUnits, raul::DynamicConvolutionAttentionParams::hparams{ 1, 2, 11, 1, 0.0_dt, 0.0_dt }, false } },
        { raul::DynamicConvolutionAttentionParams{ { "query", "state", "memory" },
                                                   { "alignment", "values", "next_state", "max_attn" },
                                                   parent,
                                                   numUnits,
                                                   raul::DynamicConvolutionAttentionParams::hparams{ 1, 2, 11, 1, 0.0_dt, 0.0_dt },
                                                   false } },
        { raul::DynamicConvolutionAttentionParams{
            { "query", "state", "memory" }, { "alignment" }, parent, numUnits, raul::DynamicConvolutionAttentionParams::hparams{ 1, 2, 11, 1, 0.0_dt, 0.0_dt }, true } },
        { raul::DynamicConvolutionAttentionParams{ { "query", "state", "memory" },
                                                   { "alignment", "values", "next_state", "max_attn" },
                                                   parent,
                                                   numUnits,
                                                   raul::DynamicConvolutionAttentionParams::hparams{ 1, 2, 11, 1, 0.0_dt, 0.0_dt },
                                                   true } },
        { raul::DynamicConvolutionAttentionParams{
            { "query", "state", "memory", "memory_seq_length" }, { "alignment" }, numUnits, raul::DynamicConvolutionAttentionParams::hparams{ 1, 2, 11, 1, 0.0_dt, 0.0_dt }, false } },
        { raul::DynamicConvolutionAttentionParams{ { "query", "state", "memory", "memory_seq_length" },
                                                   { "alignment", "values", "next_state", "max_attn" },
                                                   numUnits,
                                                   raul::DynamicConvolutionAttentionParams::hparams{ 1, 2, 11, 1, 0.0_dt, 0.0_dt },
                                                   false } },
        { raul::DynamicConvolutionAttentionParams{ { "query", "state", "memory", "memory_seq_length" },
                                                   { "alignment", "values", "next_state" },
                                                   parent,
                                                   numUnits,
                                                   raul::DynamicConvolutionAttentionParams::hparams{ 1, 2, 11, 1, 0.0_dt, 0.0_dt },
                                                   false } },
        { raul::DynamicConvolutionAttentionParams{ { "query", "state", "memory", "memory_seq_length" },
                                                   { "alignment", "values", "next_state", "max_attn" },
                                                   parent,
                                                   numUnits,
                                                   raul::DynamicConvolutionAttentionParams::hparams{ 1, 2, 11, 1, 0.0_dt, 0.0_dt },
                                                   false } },
        { raul::DynamicConvolutionAttentionParams{
            { "query", "state", "memory", "memory_seq_length" }, { "alignment" }, numUnits, raul::DynamicConvolutionAttentionParams::hparams{ 1, 2, 11, 1, 0.0_dt, 0.0_dt }, true } },
        { raul::DynamicConvolutionAttentionParams{
            { "query", "state", "memory", "memory_seq_length" }, { "alignment", "values" }, numUnits, raul::DynamicConvolutionAttentionParams::hparams{ 1, 2, 11, 1, 0.0_dt, 0.0_dt }, true } },
        { raul::DynamicConvolutionAttentionParams{
            { "query", "state", "memory", "memory_seq_length" }, { "alignment" }, parent, numUnits, raul::DynamicConvolutionAttentionParams::hparams{ 1, 2, 11, 1, 0.0_dt, 0.0_dt }, true } },
        { raul::DynamicConvolutionAttentionParams{ { "query", "state", "memory", "memory_seq_length" },
                                                   { "alignment", "values", "next_state", "max_attn" },
                                                   parent,
                                                   numUnits,
                                                   raul::DynamicConvolutionAttentionParams::hparams{ 1, 2, 11, 1, 0.0_dt, 0.0_dt },
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
        ASSERT_THROW(raul::DynamicConvolutionAttentionLayer("attn", incorrectParams[i], networkParameters), raul::Exception);
    }
}

TEST(TestLayerDynamicConvolutionAttention, GetTrainableParametersUnit)
{
    PROFILE_TEST
    const size_t numUnits = 4;
    const size_t queryDepth = 3;
    const size_t alignmentsSize = 2;
    const size_t anyNumber = 3;
    const size_t batchSize = 1;

    const size_t goldenTrainableParams = 10u;
    // List of trainable parameters:
    // 1. attention_variable_projection;
    // 2. attention_bias;
    // 3. memory_layer::Weights;
    // 4. location_convolution::Weights;
    // 5. location_convolution::Biases;
    // 6. location_layer::Weights;
    // 7. dynamic_fc1::Weights;
    // 8. dynamic_fc1::Biases;
    // 9. dynamic_fc2::Weights;
    // 10. dynamic_projection::Weights.

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
    raul::DynamicConvolutionAttentionLayer(
        "attn",
        raul::DynamicConvolutionAttentionParams{
            { "query", "state", "memory" }, { "alignment", "max_attn" }, numUnits, raul::DynamicConvolutionAttentionParams::hparams{ 1, 2, 11, 1, 0.0_dt, 0.0_dt }, false },
        networkParameters);

    TENSORS_CREATE(batchSize);

    memory_manager["query"] = TORANGE(query);
    memory_manager["state"] = TORANGE(state);
    memory_manager["memory"] = TORANGE(memory);

    work.printInfo(std::cout);

    // Checks
    EXPECT_EQ(work.getTrainableParameterNames().size(), goldenTrainableParams);
}

}