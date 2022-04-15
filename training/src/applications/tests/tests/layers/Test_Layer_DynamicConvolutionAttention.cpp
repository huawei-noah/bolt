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
#include <training/layers/composite/DynamicConvolutionAttentionLayer.h>
#include <training/network/Workflow.h>

#include <tests/tools/callbacks/TensorChecker.h>

namespace UT
{

namespace
{

using namespace std;
using namespace raul;

size_t loadDCAParams(const std::string& pathPrefix, size_t index, raul::MemoryManager& m, const raul::Name& name, bool allParamsShouldExist = true)
{
    std::map<raul::Name, std::string> maps[2];

    // Plain
    maps[0] = { { name / "location_convolution" / "Weights", "location_convolution.weights_" + std::to_string(index) },
                { name / "location_convolution" / "Biases", "location_convolution.bias_" + std::to_string(index) },
                { name / "dynamic_fc1" / "Biases", "dynamic_fc1.bias_" + std::to_string(index) } };

    // Transposed
    maps[1] = { { name / "location_layer" / "Weights", "location_layer.weights_" + std::to_string(index) },
                { name / "dynamic_fc1" / "Weights", "dynamic_fc1.weights_" + std::to_string(index) },
                { name / "dynamic_fc2" / "Weights", "dynamic_fc2.weights_" + std::to_string(index) },
                { name / "dynamic_projection" / "Weights", "dynamic_projection.weights_" + std::to_string(index) } };

    size_t loaded = 0;
    for (size_t i = 0; i < 2; ++i)
    {
        for (const auto& p : maps[i])
        {
            auto pname = p.first;
            auto file = pathPrefix + p.second + ".data";

            if (!m.tensorExists(pname))
            {
                if (allParamsShouldExist)
                {
                    std::cout << "Tensor '" + pname + "' not found" << std::endl;
                }
                continue;
            }

            if (!std::filesystem::exists(file))
            {
                std::cout << "File '" + file + "' not found" << std::endl;
                continue;
            }

            std::cout << "Loading '" + pname + "'";
            switch (i)
            {
                case 0:
                {
                    // simply load
                    raul::DataLoader::loadData(file, m[pname]);
                    break;
                }
                case 1:
                    // load and transpose height <-> width
                    raul::DataLoader::loadData(file, m[pname]);
                    raul::Common::transpose(m[pname], m[pname].getHeight());
                    break;
                    // default: Do nothing
            }
            std::cout << "   ok" << std::endl;
            ++loaded;
        }
    }

    return loaded;
}

}

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

TEST(TestLayerDynamicConvolutionAttention, DefaultModeUnit)
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

    const raul::Tensor deltas{ 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt };
    // Real output
    const raul::Tensor realAlignment{ 0.10458944_dt, 0.661409_dt, 0.23400162_dt, 0.5119926_dt, 0.23978291_dt, 0.24822456_dt };
    const raul::Tensor realMaxAttnIndices{ 1.0_dt, 0.0_dt };

    // Initialization
    raul::WorkflowEager work;
    auto& networkParameters = work.getNetworkParameters();

    // Inputs
    work.add<raul::DataLayer>("data_query", raul::DataParams{ { "query" }, 1u, 1u, queryDepth });
    work.add<raul::DataLayer>("data_state", raul::DataParams{ { "state" }, 1u, 1u, alignmentsSize });
    work.add<raul::DataLayer>("data_memory", raul::DataParams{ { "memory" }, 1u, alignmentsSize, anyNumber });

    // Outputs
    work.add<raul::DataLayer>("output0", raul::DataParams{ { "realAlignment" }, 1u, 1u, alignmentsSize });
    work.add<raul::DataLayer>("output1", raul::DataParams{ { "realMaxAttnIndices" }, 1u, 1u, 1u });

    // Layer
    raul::DynamicConvolutionAttentionLayer(
        name,
        raul::DynamicConvolutionAttentionParams{
            { "query", "state", "memory" }, { "alignment", "max_attn" }, numUnits, raul::DynamicConvolutionAttentionParams::hparams{ 3, 3, 11, 1, 0.6_dt, 0.2_dt }, false },
        networkParameters);

    work.preparePipelines();
    work.setBatchSize(batchSize);
    work.prepareMemoryForTraining();

    raul::MemoryManager& memory_manager = work.getMemoryManager();
    size_t loaded = loadDCAParams((tools::getTestAssetsDir() / "DCA/").string(), 1u, memory_manager, name, true);
    EXPECT_EQ(loaded, 7u);

    memory_manager["query"] = TORANGE(query);
    memory_manager["state"] = TORANGE(state);
    memory_manager["memory"] = TORANGE(memory);
    memory_manager["realAlignment"] = TORANGE(realAlignment);
    memory_manager["realMaxAttnIndices"] = TORANGE(realMaxAttnIndices);
    memory_manager[raul::Name("alignment").grad()] = TORANGE(deltas);

    // For result stability
    memory_manager[name / "attention_variable_projection"] = TORANGE(raul::Tensor({ -0.7740349_dt, -0.5382635_dt, 0.81799275_dt, -0.84306645_dt }));

    UT::tools::callbacks::TensorChecker checker({ { "alignment", "realAlignment" }, { "max_attn", "realMaxAttnIndices" } }, -1_dt, eps);
    networkParameters.mCallback = checker;

    // Forward
    work.forwardPassTraining();
    // Backward
    work.backwardPassTraining();
}

TEST(TestLayerDynamicConvolutionAttention, CumulativeModeUnit)
{
    PROFILE_TEST
    constexpr size_t numUnits = 5;
    constexpr size_t queryDepth = 7;
    constexpr size_t alignmentsSize = 4;
    constexpr size_t anyNumber = 6;
    constexpr size_t batchSize = 3;
    constexpr raul::dtype eps = TODTYPE(1e-5);

    raul::Name name = "attn";

    const raul::Tensor query{ 0.63547623_dt, 0.44589663_dt, 0.6047574_dt,  0.82557225_dt, 0.58478403_dt, 0.04986751_dt, 0.9572661_dt, 0.20333457_dt, 0.11299467_dt, 0.05475962_dt, 0.2828188_dt,
                              0.5192108_dt,  0.25020587_dt, 0.85186446_dt, 0.6001804_dt,  0.79308605_dt, 0.34942162_dt, 0.592427_dt,  0.22301793_dt, 0.21016634_dt, 0.30729067_dt };
    const raul::Tensor state{ 0.22413874_dt, 0.22268498_dt, 0.8552655_dt,  0.49562013_dt, 0.31110537_dt, 0.61050725_dt,
                              0.21236408_dt, 0.93036723_dt, 0.54842377_dt, 0.84664714_dt, 0.47629058_dt, 0.89816856_dt };
    const raul::Tensor memory{ 0.81269646_dt, 0.07857466_dt, 0.8916855_dt,  0.16925514_dt, 0.06311357_dt, 0.54531074_dt, 0.5037316_dt,  0.9248222_dt,  0.66955376_dt, 0.9281193_dt,  0.12239242_dt,
                               0.8532245_dt,  0.90477383_dt, 0.7104306_dt,  0.40681756_dt, 0.5755513_dt,  0.8547678_dt,  0.59606934_dt, 0.77619946_dt, 0.97301054_dt, 0.06244731_dt, 0.33562684_dt,
                               0.22166848_dt, 0.32035887_dt, 0.03924382_dt, 0.06723011_dt, 0.32712245_dt, 0.49054873_dt, 0.11453211_dt, 0.34396613_dt, 0.52225363_dt, 0.30574834_dt, 0.8817626_dt,
                               0.8017194_dt,  0.9992852_dt,  0.65941477_dt, 0.1272459_dt,  0.19117236_dt, 0.65929854_dt, 0.7614676_dt,  0.75358987_dt, 0.41603255_dt, 0.94846773_dt, 0.8904344_dt,
                               0.91729546_dt, 0.26704276_dt, 0.17427123_dt, 0.04580772_dt, 0.98797727_dt, 0.03881574_dt, 0.22868955_dt, 0.0036062_dt,  0.6006421_dt,  0.25169027_dt, 0.45649374_dt,
                               0.21031535_dt, 0.13384092_dt, 0.610149_dt,   0.7017927_dt,  0.56946445_dt, 0.25802827_dt, 0.09499919_dt, 0.96377003_dt, 0.21196103_dt, 0.94442093_dt, 0.04924846_dt,
                               0.888088_dt,   0.23339641_dt, 0.4439162_dt,  0.13146889_dt, 0.9257786_dt,  0.3446467_dt };

    const raul::Tensor deltas{ 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt };

    // Real output
    const raul::Tensor realAlignment{ 0.29262316_dt, 0.14896806_dt, 0.24737868_dt, 0.31103006_dt, 0.17908312_dt, 0.28894222_dt,
                                      0.15750135_dt, 0.3744733_dt,  0.18242854_dt, 0.21125585_dt, 0.16769765_dt, 0.43861797_dt };
    const raul::Tensor realNextState{ 0.5167619_dt,  0.37165302_dt, 1.1026442_dt, 0.80665016_dt, 0.49018848_dt, 0.89944947_dt,
                                      0.36986542_dt, 1.3048406_dt,  0.7308523_dt, 1.057903_dt,   0.64398825_dt, 1.3367865_dt };
    const raul::Tensor realMaxAttnIndices{ 3.0_dt, 3.0_dt, 3.0_dt };

    // Initialization
    raul::WorkflowEager work;
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
    raul::DynamicConvolutionAttentionLayer(
        name,
        raul::DynamicConvolutionAttentionParams{
            { "query", "state", "memory" }, { "alignment", "next_state", "max_attn" }, numUnits, raul::DynamicConvolutionAttentionParams::hparams{ 5, 3, 14, 1, 0.9_dt, 0.1_dt }, true },
        networkParameters);

    work.preparePipelines();
    work.setBatchSize(batchSize);
    work.prepareMemoryForTraining();

    raul::MemoryManager& memory_manager = work.getMemoryManager();
    size_t loaded = loadDCAParams((tools::getTestAssetsDir() / "DCA/").string(), 2u, memory_manager, name, true);
    EXPECT_EQ(loaded, 7u);

    memory_manager["query"] = TORANGE(query);
    memory_manager["state"] = TORANGE(state);
    memory_manager["memory"] = TORANGE(memory);
    memory_manager["realAlignment"] = TORANGE(realAlignment);
    memory_manager["realNextState"] = TORANGE(realNextState);
    memory_manager["realMaxAttnIndices"] = TORANGE(realMaxAttnIndices);
    memory_manager[raul::Name("alignment").grad()] = TORANGE(deltas);
    memory_manager[raul::Name("next_state").grad()] = TORANGE(deltas);

    // For result stability
    memory_manager[name / "attention_variable_projection"] = TORANGE(raul::Tensor({ 0.7624613_dt, 0.7580764_dt, 0.10920429_dt, 0.4539646_dt, -0.63475096_dt }));

    UT::tools::callbacks::TensorChecker checker({ { "alignment", "realAlignment" }, { "next_state", "realNextState" }, { "max_attn", "realMaxAttnIndices" } }, -1_dt, eps);
    networkParameters.mCallback = checker;

    // Forward
    work.forwardPassTraining();

    // Backward
    work.backwardPassTraining();
}

}