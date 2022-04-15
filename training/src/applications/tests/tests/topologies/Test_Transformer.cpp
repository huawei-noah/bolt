// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <tests/GTestExtensions.h>
#include <tests/tools/TestTools.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <set>
#include <sstream>

#include <training/api/API.h>
#include <training/common/Common.h>
#include <training/layers/activations/LogSoftMaxActivation.h>
#include <training/layers/activations/SoftMaxActivation.h>
#include <training/layers/basic/DropoutLayer.h>
#include <training/layers/basic/LabelSmoothing.h>
#include <training/layers/basic/MaskedFillLayer.h>
#include <training/layers/basic/MatMulLayer.h>
#include <training/layers/basic/PositionalEncoding.h>
#include <training/layers/basic/ReshapeLayer.h>
#include <training/layers/basic/SplitterLayer.h>
#include <training/layers/basic/TransposeLayer.h>
#include <training/layers/basic/trainable/Embedding.h>
#include <training/layers/basic/trainable/LinearLayer.h>
#include <training/layers/composite/AttentionLayer.h>
#include <training/layers/composite/MultiHeadAttention.h>
#include <training/layers/composite/Transformer.h>
#include <training/loss/KLDivLoss.h>
#include <training/loss/NegativeLogLikelihoodLoss.h>
#include <training/network/Layers.h>
#include <training/network/Workflow.h>
#include <training/optimizers/SGD.h>

namespace UT
{

using dvec = std::vector<raul::dtype>;
struct VariousEmbeddingLayerParameters : public testing::TestWithParam<std::tuple<bool, dvec, dvec, dvec>>
{
    static constexpr raul::dtype eps = 1e-4_dt;

    static constexpr size_t DICT_SIZE = 3;
    static constexpr size_t MODEL_SIZE = 6;
    static constexpr size_t BATCH_SIZE = 1;
    static constexpr size_t NUM_CLASSES = 2;

    bool enablePadding = std::get<0>(GetParam());
    raul::Tensor realEmbOut = raul::Tensor::dt_range{ std::get<1>(GetParam()).data(), std::get<1>(GetParam()).data() + std::get<1>(GetParam()).size() };
    raul::Tensor realEmbGradient = raul::Tensor::dt_range{ std::get<2>(GetParam()).data(), std::get<2>(GetParam()).data() + std::get<2>(GetParam()).size() };
    raul::Tensor lut = raul::Tensor::dt_range{ std::get<3>(GetParam()).data(), std::get<3>(GetParam()).data() + std::get<3>(GetParam()).size() };

    raul::Tensor raw = { 2.0_dt, 1.0_dt, 0.0_dt, 2.0_dt };

    std::vector<raul::dtype> gradientAfterTwoBackwardPasses;

    void SetUp() final
    {
        std::transform(realEmbGradient.begin(), realEmbGradient.end(), std::back_inserter(gradientAfterTwoBackwardPasses), [](raul::dtype v) { return v * 2; });
    }
};

// corresponds to embedding_padding.py test
TEST_P(VariousEmbeddingLayerParameters, EmbeddingPaddingNoPaddingUnit)
{
    PROFILE_TEST
    using namespace raul;

    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    work.add<raul::DataLayer>("data", raul::DataParams{ { "in" }, 1, raw.size() / BATCH_SIZE, 1 });

    EmbeddingParams layerParameter{ "in", "emb", DICT_SIZE, MODEL_SIZE, enablePadding ? 0 : -1, false, true };
    raul::Embedding emb("embedding", layerParameter, networkParameters);
    TENSORS_CREATE(BATCH_SIZE);

    memory_manager["in"] = TORANGE(raw);

    memory_manager["embedding::Weights"] = TORANGE(lut);

    emb.forwardCompute(NetworkMode::Train);

    auto& embT = memory_manager["emb"];
    ASSERT_INTERVALS_NEAR(embT.begin(), embT.end(), realEmbOut.begin(), realEmbOut.end(), eps);
    printf(" - Embedding forward is Ok.\n");

    memory_manager[Name("emb").grad()] = 1.0_dt;

    emb.backwardCompute();

    auto& embGrad = memory_manager["embedding::WeightsGradient"];
    ASSERT_INTERVALS_NEAR(embGrad.begin(), embGrad.end(), realEmbGradient.begin(), realEmbGradient.end(), eps);
}

TEST_P(VariousEmbeddingLayerParameters, ShouldAccumulateGradientsDuringBackwardPassByDefaultUnit)
{
    PROFILE_TEST
    using namespace raul;

    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    work.add<raul::DataLayer>("data", raul::DataParams{ { "in" }, 1, raw.size() / BATCH_SIZE, 1 });

    EmbeddingParams layerParameter{ "in", "emb", DICT_SIZE, MODEL_SIZE, enablePadding ? 0 : -1, false, true };
    raul::Embedding emb("embedding", layerParameter, networkParameters);
    TENSORS_CREATE(BATCH_SIZE);

    memory_manager["in"] = TORANGE(raw);

    memory_manager["embedding::Weights"] = TORANGE(lut);

    emb.forwardCompute(NetworkMode::Train);

    memory_manager[Name("emb").grad()] = 1.0_dt;

    emb.backwardCompute();
    emb.backwardCompute();

    auto& embGrad = memory_manager["embedding::WeightsGradient"];
    ASSERT_INTERVALS_NEAR(embGrad.begin(), embGrad.end(), gradientAfterTwoBackwardPasses.begin(), gradientAfterTwoBackwardPasses.end(), eps);
}

INSTANTIATE_TEST_SUITE_P(
    TestEmbedding,
    VariousEmbeddingLayerParameters,
    testing::Values(
        std::make_tuple(true,
                        dvec{ 1.389366_dt, 1.586334_dt, 0.946298_dt, -0.843677_dt, 0.931827_dt, 1.259009_dt, -0.341360_dt, 1.853006_dt, 0.468096_dt, -0.157712_dt, -0.173397_dt, 0.183478_dt,
                              0.000000_dt, 0.000000_dt, 0.000000_dt, 0.000000_dt,  0.000000_dt, 0.000000_dt, 1.389366_dt,  1.586334_dt, 0.946298_dt, -0.843677_dt, 0.931827_dt,  1.259009_dt },
                        dvec{ 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt },
                        dvec{ 0.0_dt,
                              0.0_dt,
                              0.0_dt,
                              0.0_dt,
                              0.0_dt,
                              0.0_dt,
                              -0.341360_dt,
                              1.853006_dt,
                              0.468096_dt,
                              -0.157712_dt,
                              -0.173397_dt,
                              0.183478_dt,
                              1.389366_dt,
                              1.586334_dt,
                              0.946298_dt,
                              -0.843677_dt,
                              0.931827_dt,
                              1.259009_dt }),
        std::make_tuple(false,
                        dvec{ 1.389366_dt,  1.586334_dt,  0.946298_dt, -0.843677_dt, 0.931827_dt, 1.259009_dt,  -0.341360_dt, 1.853006_dt, 0.468096_dt, -0.157712_dt, -0.173397_dt, 0.183478_dt,
                              -1.125840_dt, -1.152360_dt, 0.566651_dt, 0.793508_dt,  0.598839_dt, -1.555095_dt, 1.389366_dt,  1.586334_dt, 0.946298_dt, -0.843677_dt, 0.931827_dt,  1.259009_dt },
                        dvec{ 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt },
                        dvec{ -1.125840_dt,
                              -1.152360_dt,
                              0.566651_dt,
                              0.793508_dt,
                              0.598839_dt,
                              -1.555095_dt,
                              -0.341360_dt,
                              1.853006_dt,
                              0.468096_dt,
                              -0.157712_dt,
                              -0.173397_dt,
                              0.183478_dt,
                              1.389366_dt,
                              1.586334_dt,
                              0.946298_dt,
                              -0.843677_dt,
                              0.931827_dt,
                              1.259009_dt })));

// corresponds to embedding_positional.py test
TEST(TestTransformer, EmbeddingPositionalUnit)
{
    PROFILE_TEST
    using namespace raul;

    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    raul::DataLoader dataLoader;

    constexpr dtype eps = 1e-4_dt;

    constexpr size_t DICT_SIZE = 3;
    constexpr size_t MODEL_SIZE = 6;
    constexpr size_t BATCH_SIZE = 1;
    constexpr size_t NUM_CLASSES = 2;

    Tensor raw = { 2.0_dt, 1.0_dt, 0.0_dt, 0.0_dt };
    TensorU8 classes = { 1 };

    dtype realLoss = TODTYPE(1.513026);

    Tensor lut = { -1.125840_dt, -1.152360_dt, 0.566651_dt, 0.793508_dt, 0.598839_dt, -1.555095_dt, -0.341360_dt, 1.853006_dt, 0.468096_dt,
                   -0.157712_dt, -0.173397_dt, 0.183478_dt, 1.389366_dt, 1.586334_dt, 0.946298_dt,  -0.843677_dt, 0.931827_dt, 1.259009_dt }; // DICT_SIZE x MODEL_SIZE

    Tensor fcWeight = { 0.593586_dt, 0.472450_dt, 0.415770_dt, 0.575073_dt, 0.417719_dt, 0.295235_dt, 0.271122_dt, 0.796689_dt, 0.692278_dt, 0.195730_dt, 0.203848_dt, 0.953685_dt,
                        0.683296_dt, 0.842650_dt, 0.752854_dt, 0.078359_dt, 0.857936_dt, 0.375558_dt, 0.686956_dt, 0.522561_dt, 0.005132_dt, 0.572951_dt, 0.175652_dt, 0.618587_dt,
                        0.749658_dt, 0.696214_dt, 0.604651_dt, 0.529950_dt, 0.109958_dt, 0.256036_dt, 0.212090_dt, 0.736594_dt, 0.970375_dt, 0.020376_dt, 0.836909_dt, 0.203647_dt,
                        0.281987_dt, 0.374835_dt, 0.374158_dt, 0.256443_dt, 0.023701_dt, 0.325083_dt, 0.491013_dt, 0.090189_dt, 0.123471_dt, 0.393642_dt, 0.114322_dt, 0.606878_dt };

    Tensor realEmbOut = { 3.403238_dt,  3.885710_dt,  2.317948_dt, -2.066578_dt, 2.282500_dt, 3.083930_dt,  -0.836159_dt, 4.538919_dt,  1.146597_dt, -0.386315_dt, -0.424734_dt, 0.449427_dt,
                          -2.757733_dt, -2.822695_dt, 1.388005_dt, 1.943691_dt,  1.466851_dt, -3.809190_dt, -2.757733_dt, -2.822695_dt, 1.388005_dt, 1.943691_dt,  1.466851_dt,  -3.809190_dt };

    Tensor realEmbGrad = { -0.075262_dt, 0.367514_dt,  -0.854652_dt, -0.236231_dt, 1.298476_dt,  0.268749_dt, -0.304368_dt, 1.288292_dt, 0.921346_dt,
                           0.313994_dt,  -1.084537_dt, -0.846010_dt, 0.231370_dt,  -0.304269_dt, 0.233947_dt, -1.003837_dt, 0.948410_dt, -1.432194_dt };

    work.add<raul::DataLayer>("data1", raul::DataParams{ { "labels" }, 1, 1, NUM_CLASSES });
    work.add<raul::DataLayer>("data2", raul::DataParams{ { "in" }, 1, raw.size() / BATCH_SIZE, 1 });

    auto& encodedClasses = dataLoader.buildOneHotVector(classes, NUM_CLASSES);

    work.add<raul::Embedding>("embedding", raul::EmbeddingParams{ "in", "emb", DICT_SIZE, MODEL_SIZE });
    work.add<raul::PositionalEncoding>("pe", raul::PositionalEncodingParams{ "emb", "pe", MODEL_SIZE });
    work.add<raul::ReshapeLayer>("r", raul::ViewParams{ "pe", "pe1", 1, 1, -1 });
    work.add<raul::LinearLayer>("fc", raul::LinearParams{ { "pe1" }, { "fc" }, NUM_CLASSES });
    work.add<raul::LogSoftMaxActivation>("logsoftmax", raul::BasicParamsWithDim{ { "fc" }, { "out" } });
    work.add<raul::NLLLoss>("loss", raul::LossParams{ { "out", "labels" }, { "loss" }, "batch_mean" });
    TENSORS_CREATE(BATCH_SIZE);

    memory_manager["in"] = TORANGE(raw);

    memory_manager["labels"] = TORANGE(encodedClasses);

    memory_manager["embedding::Weights"] = TORANGE(lut);
    Common::transpose(fcWeight, NUM_CLASSES);
    memory_manager["fc::Weights"] = TORANGE(fcWeight);
    memory_manager["fc::Biases"] = TORANGE(Tensor(NUM_CLASSES, 1.0_dt));

    work.forwardPassTraining();

    const Tensor& embT = memory_manager["emb"];

    for (size_t i = 0; i < embT.size(); ++i)
    {
        EXPECT_NEAR(embT[i], realEmbOut[i], eps);
    }
    printf(" - Embedding forward is Ok.\n");

    const Tensor& loss = memory_manager["loss"];
    EXPECT_NEAR(loss[0], realLoss, eps);

    work.backwardPassTraining();

    auto& embGrad = memory_manager["embedding::WeightsGradient"];

    for (size_t i = 0; i < embGrad.size(); ++i)
    {
        EXPECT_NEAR(embGrad[i], realEmbGrad[i], eps);
    }
}

TEST(TestTransformer, EmbeddingPositionalFreezeUnit)
{
    PROFILE_TEST
    using namespace raul;

    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    raul::DataLoader dataLoader;

    constexpr dtype eps = 1e-4_dt;

    size_t DICT_SIZE = 3;
    size_t MODEL_SIZE = 6;
    size_t BATCH_SIZE = 1;
    size_t NUM_CLASSES = 2;

    Tensor raw = { 2.0_dt, 1.0_dt, 0.0_dt, 0.0_dt };
    TensorU8 classes = { 1 };

    dtype realLoss = TODTYPE(1.513026);

    Tensor lut = { -1.125840_dt, -1.152360_dt, 0.566651_dt, 0.793508_dt, 0.598839_dt, -1.555095_dt, -0.341360_dt, 1.853006_dt, 0.468096_dt,
                   -0.157712_dt, -0.173397_dt, 0.183478_dt, 1.389366_dt, 1.586334_dt, 0.946298_dt,  -0.843677_dt, 0.931827_dt, 1.259009_dt }; // DICT_SIZE x MODEL_SIZE

    Tensor fcWeight = { 0.593586_dt, 0.472450_dt, 0.415770_dt, 0.575073_dt, 0.417719_dt, 0.295235_dt, 0.271122_dt, 0.796689_dt, 0.692278_dt, 0.195730_dt, 0.203848_dt, 0.953685_dt,
                        0.683296_dt, 0.842650_dt, 0.752854_dt, 0.078359_dt, 0.857936_dt, 0.375558_dt, 0.686956_dt, 0.522561_dt, 0.005132_dt, 0.572951_dt, 0.175652_dt, 0.618587_dt,
                        0.749658_dt, 0.696214_dt, 0.604651_dt, 0.529950_dt, 0.109958_dt, 0.256036_dt, 0.212090_dt, 0.736594_dt, 0.970375_dt, 0.020376_dt, 0.836909_dt, 0.203647_dt,
                        0.281987_dt, 0.374835_dt, 0.374158_dt, 0.256443_dt, 0.023701_dt, 0.325083_dt, 0.491013_dt, 0.090189_dt, 0.123471_dt, 0.393642_dt, 0.114322_dt, 0.606878_dt };

    Tensor realEmbOut = { 3.403238_dt,  3.885710_dt,  2.317948_dt, -2.066578_dt, 2.282500_dt, 3.083930_dt,  -0.836159_dt, 4.538919_dt,  1.146597_dt, -0.386315_dt, -0.424734_dt, 0.449427_dt,
                          -2.757733_dt, -2.822695_dt, 1.388005_dt, 1.943691_dt,  1.466851_dt, -3.809190_dt, -2.757733_dt, -2.822695_dt, 1.388005_dt, 1.943691_dt,  1.466851_dt,  -3.809190_dt };

    Tensor realEmbGrad = { -0.075262_dt, 0.367514_dt,  -0.854652_dt, -0.236231_dt, 1.298476_dt,  0.268749_dt, -0.304368_dt, 1.288292_dt, 0.921346_dt,
                           0.313994_dt,  -1.084537_dt, -0.846010_dt, 0.231370_dt,  -0.304269_dt, 0.233947_dt, -1.003837_dt, 0.948410_dt, -1.432194_dt };

    work.add<raul::DataLayer>("data1", raul::DataParams{ { "labels" }, 1, 1, NUM_CLASSES });
    work.add<raul::DataLayer>("data2", raul::DataParams{ { "in" }, 1, raw.size() / BATCH_SIZE, 1 });

    auto& encodedClasses = dataLoader.buildOneHotVector(classes, NUM_CLASSES);

    work.add<raul::Embedding>("embedding", raul::EmbeddingParams{ "in", "emb", true, DICT_SIZE, MODEL_SIZE });
    work.add<raul::PositionalEncoding>("pe", raul::PositionalEncodingParams{ "emb", "pe", MODEL_SIZE });
    work.add<raul::ReshapeLayer>("r", raul::ViewParams{ "pe", "pe1", 1, 1, -1 });
    work.add<raul::LinearLayer>("fc", raul::LinearParams{ { "pe1" }, { "fc" }, NUM_CLASSES });
    work.add<raul::LogSoftMaxActivation>("logsoftmax", raul::BasicParamsWithDim{ { "fc" }, { "out" } });
    work.add<raul::NLLLoss>("loss", raul::LossParams{ { "out", "labels" }, { "loss" }, "batch_mean" });

    TENSORS_CREATE(BATCH_SIZE);

    memory_manager["in"] = TORANGE(raw);

    memory_manager["labels"] = TORANGE(encodedClasses);

    memory_manager["embedding::Weights"] = TORANGE(lut);
    Common::transpose(fcWeight, NUM_CLASSES);
    memory_manager["fc::Weights"] = TORANGE(fcWeight);
    memory_manager["fc::Biases"] = TORANGE(Tensor(NUM_CLASSES, 1.0_dt));

    EXPECT_FALSE(memory_manager.tensorExists("embedding::WeightsGradient"));

    work.forwardPassTraining();

    const Tensor& embT = memory_manager["emb"];

    for (size_t i = 0; i < embT.size(); ++i)
    {
        EXPECT_NEAR(embT[i], realEmbOut[i], eps);
    }
    printf(" - Embedding forward is Ok.\n");

    const Tensor& loss = memory_manager["loss"];
    EXPECT_NEAR(loss[0], realLoss, eps);

    work.backwardPassTraining();
}

TEST(TestTransformer, LabelSmoothingUnit)
{
    PROFILE_TEST
    using namespace raul;

    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    raul::DataLoader dataLoader;

    dtype eps = TODTYPE(1e-4);

    size_t MODEL_SIZE = 4;
    size_t BATCH_SIZE = 3;
    networkParameters.mLossReductionCoefficient = BATCH_SIZE;
    size_t NUM_CLASSES = MODEL_SIZE;
    int PADDING_CLASS_IDX = 0;

    Tensor raw = { 1.0f, 2.0f, 5.0f, -1.0f, 1.0f, 2.0f, 6.0f, -1.0f, -2.0f, 5.0f, -1.0, 3.0f };
    TensorU8 classes = { 1, 2, 0 };

    dtype realLoss = TODTYPE(0.9517);

    Tensor realSmoothingOut = { 0.0000f, 0.9000f, 0.0500f, 0.0500f, 0.0000f, 0.0500f, 0.9000f, 0.0500f, 0.0000f, 0.0000f, 0.0000f, 0.0000f };

    Tensor realInputGrad = { 0.0057f, -0.2845f, 0.2947f, -0.0159f, 0.0022f, -0.0107f, 0.0249f, -0.0164f, 0.0000f, 0.0000f, 0.0000f, 0.0000f };

    work.add<raul::DataLayer>("data1", raul::DataParams{ { "labels" }, 1, 1, NUM_CLASSES });
    work.add<raul::DataLayer>("data2", raul::DataParams{ { "in" }, 1, raw.size() / BATCH_SIZE / MODEL_SIZE, MODEL_SIZE });

    auto& encodedClasses = dataLoader.buildOneHotVector(classes, NUM_CLASSES);

    work.add<raul::LogSoftMaxActivation>("logsoftmax", raul::BasicParamsWithDim{ { "in" }, { "lsm" } });
    work.add<raul::LabelSmoothing>("smoothing", raul::LabelSmoothingParams{ { "labels" }, { "labels_sm" }, 0.1f, PADDING_CLASS_IDX });
    work.add<raul::KLDivLoss>("loss", raul::LossParams{ { "lsm", "labels_sm" }, { "loss" }, "batch_mean" });
    TENSORS_CREATE(BATCH_SIZE);

    memory_manager["in"] = TORANGE(raw);
    memory_manager["labels"] = TORANGE(encodedClasses);

    work.forwardPassTraining();

    const Tensor& smoothing = memory_manager["labels_sm"];

    for (size_t i = 0; i < smoothing.size(); ++i)
    {
        EXPECT_NEAR(smoothing[i], realSmoothingOut[i], eps);
    }
    printf(" - LabelSmoothing forward is Ok.\n");

    const Tensor& loss = memory_manager["loss"];
    EXPECT_NEAR(loss[0], realLoss, eps);

    work.backwardPassTraining();

    auto& inGrad = memory_manager[raul::Name("in").grad()];

    for (size_t i = 0; i < inGrad.size(); ++i)
    {
        EXPECT_NEAR(inGrad[i], realInputGrad[i], eps);
    }
}

// test based on based on https://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html
TEST(TestTransformer, NGramLanguageModelerUnit)
{
    PROFILE_TEST
    using namespace std;
    using namespace raul;

    size_t CONTEXT_SIZE = 2;

    Tensor idealLosses = { 517.2954f, 516.8915f, 516.4974f, 516.1131f, 515.7386f, 515.3741f, 515.0194f, 514.6746f, 514.3398f, 514.0149f };

    string sonet = "When forty winters shall besiege thy brow, "
                   "And dig deep trenches in thy beauty's field, "
                   "Thy youth's proud livery so gazed on now, "
                   "Will be a totter'd weed of small worth held: "
                   "Then being asked, where all thy beauty lies, "
                   "Where all the treasure of thy lusty days; "
                   "To say, within thine own deep sunken eyes, "
                   "Were an all - eating shame, and thriftless praise. "
                   "How much more praise deserv'd thy beauty's use, "
                   "If thou couldst answer 'This fair child of mine "
                   "Shall sum my count, and make my old excuse,' "
                   "Proving his beauty by succession thine! "
                   "This were to be new made when thou art old, "
                   "And see thy blood warm when thou feel'st it cold.";

    stringstream ss(sonet);
    vector<std::string> words;
    copy(istream_iterator<string>(ss), istream_iterator<string>(), back_inserter(words));

    vector<pair<vector<string>, string>> ngrams;
    for (size_t i = 0; i < words.size() - CONTEXT_SIZE; ++i)
        ngrams.push_back(make_pair(vector<string>(words.begin() + i, words.begin() + i + CONTEXT_SIZE), words[i + CONTEXT_SIZE]));

    set<string> vocab(words.begin(), words.end());
    vector<string> wordList(vocab.begin(), vocab.end());
    sort(wordList.begin(), wordList.end());
    map<string, size_t> word_to_idx;
    for (size_t i = 0; i < wordList.size(); ++i)
        word_to_idx[wordList[i]] = i;
}

//  output = input + dropout(sublayer({LayerNorm(input), sublayerAuxArgs}))
/*
raul::NetDef SublayerConnection(const std::string name, const raul::Name& input, const raul::Name& output, raul::NetDef& sublayer, const raul::Names& sublayerAuxArgs, raul::dtype dropout)
{
    using namespace raul;
    NetDef res;
    res.add<raul::SplitterLayer>(name + ".splitter", BasicParams{ { input }, { name + ".x1", name + ".x2" } });
    res.add<raul::LayerNormLayer>(name + ".norm", LayerNormParams{ name + ".x2", name + ".x", 1e-6f });

    Names sublayerArgs = { name + ".x" };
    std::copy(sublayerAuxArgs.begin(), sublayerAuxArgs.end(), std::back_inserter(sublayerArgs));

    res.addNetDef(sublayer, "sublayer", sublayerArgs, { name + ".y" });
    std::string out = name + ".y";
    if (dropout > 0.0_dt)
    {
        res.add<raul::DropoutLayer>(name + ".dropout", DropoutParams{ { out }, { name + ".do" }, static_cast<float>(dropout) });
        out = name + ".do";
    }
    res.add<raul::ElementWiseSumLayer>(name + ".sum", ElementWiseLayerParams{ { name + ".x1", out }, { output } });
    return res;
}*/
/*
// architecture according to https://nlp.seas.harvard.edu/2018/04/03/attention.html
TEST(TestTransformer, DISABLED_TransformerArchitectureUnit)
{
    PROFILE_TEST
    using namespace raul;
    unsigned int HEADS = 8;
    size_t LENGTH = 6;
    size_t MODEL_SIZE = 512;
    size_t SRC_VOCAB = 11;
    size_t TGT_VOCAB = 11;
    size_t FEED_FORWARD_SIZE = 2048;
    dtype DROPOUT_RATE = TODTYPE(0.1);
    float LAYER_NORM_EPSILON = 1.e-6f;

    Tensor maskData = { 0, 0, 0, 0 };

    NetDef Attention;
    AddOp(Attention, "attn", MULTI_HEAD_ATTENTION_LAYER, createParam(MultiHeadAttentionParams{ { "q", "k", "v", "mask" }, "attn", HEADS, static_cast<float>(DROPOUT_RATE) }));

    NetDef SelfAttention;
    AddOp(SelfAttention, "self_attn", MULTI_HEAD_ATTENTION_LAYER, createParam(MultiHeadAttentionParams{ { "x", "mask" }, "attn", HEADS, static_cast<float>(DROPOUT_RATE) }));

    NetDef PositionwiseFeedForward;
    Add<raul::LinearLayer>(PositionwiseFeedForward, "w_1", LinearParams{ { "x" }, { "w_1" }, FEED_FORWARD_SIZE });
    Add<raul::DropoutLayer>(PositionwiseFeedForward, "dropout", DropoutParams{ { "w_1" }, { "dropout" }, static_cast<float>(DROPOUT_RATE) });
    Add<raul::LinearLayer>(PositionwiseFeedForward, "w_2", LinearParams{ { "dropout" }, { "w_2" }, MODEL_SIZE });

    NetDef EncoderLayer;
    AddNetDef(EncoderLayer, SublayerConnection("sublayer", "x", "y", SelfAttention, { "mask" }, static_cast<float>(DROPOUT_RATE)), "l1", { "x", "mask" }, { "y" });
    AddNetDef(EncoderLayer, SublayerConnection("sublayer", "x", "y", PositionwiseFeedForward, {}, static_cast<float>(DROPOUT_RATE)), "l2", { "y" }, { "z" });

    NetDef Encoder = cloneSequential("encoder_layer", EncoderLayer, LENGTH);

    NetDef DecoderLayer;
    AddNetDef(EncoderLayer, SublayerConnection("sublayer", "x", "y", SelfAttention, { "tgt_mask" }, static_cast<float>(DROPOUT_RATE)), "l1", { "x", "tgt_mask" }, { "y" });
    Add<raul::LayerNormLayer>(EncoderLayer, "mem_norm", LayerNormParams{ "m", "m_norm", LAYER_NORM_EPSILON });
    addNetDef(EncoderLayer, SublayerConnection("sublayer", "x", "y", Attention, { "m_norm", "m_norm", "tgt_mask" }, static_cast<float>(DROPOUT_RATE)), "l2", { "y" }, { "z" });
}
*/

template <typename MM>
std::vector<typename MM::type> subsequent_mask(size_t size)
{
    std::vector<typename MM::type> v(size * size, TOMMTYPE(1.0_dt));
    raul::Common::triu(v.data(), size, size, 1);
    std::transform(v.begin(), v.end(), v.begin(), [](auto s) { return s == TOMMTYPE(0.0_dt) ? TOMMTYPE(1.0_dt) : TOMMTYPE(0.0_dt); });
    return v;
}

template<typename MM>
struct TransformerBatch
{
    TransformerBatch(const typename MM::tensor& s, const typename MM::tensor& t, MM& m, typename MM::type pad = 0.0_dt)
    {
        trg = &m["trg"];
        trg_y = &m["trg_y"];
        trg_mask = &m["trg_mask"];

        src = &m["src"];
        src_mask = &m["src_mask"];

        *src = TORANGE_MM(s);

        for (size_t i = 0; i < s.size(); ++i)
            (*src_mask)[i] = (s[i] != pad ? TOMMTYPE(1.0) : TOMMTYPE(0.0));

        if (!t.empty())
        {
            auto n = t.getBatchSize();
            auto w = t.getWidth();
            const auto t2d = t.reshape(yato::dims(n, w));
            const auto trg2d = trg->reshape(yato::dims(n, w - 1));
            const auto trg_y2d = trg_y->reshape(yato::dims(n, w - 1));
            const auto trg_mask2d = trg_mask->reshape(yato::dims(n, (w - 1) * (w - 1)));

            for (size_t b = 0; b < n; ++b)
            {
                std::copy(t2d[b].begin(), t2d[b].end() - 1, trg2d[b].begin());
                std::copy(t2d[b].begin() + 1, t2d[b].end(), trg_y2d[b].begin());
                auto sm = subsequent_mask<MM>(w - 1);
                for (size_t i = 0; i < (w - 1); ++i)
                    if (trg_y2d[b][i] != pad) ++ntokens;
                for (size_t i = 0; i < (w - 1); ++i)
                    for (size_t j = 0; j < (w - 1); ++j)
                        trg_mask2d[b][i * (w - 1) + j] = (trg2d[b][j] != pad ? TOMMTYPE(1.0) : TOMMTYPE(0.0)) * sm[i * (w - 1) + j];
            }
        }
    }

    typename MM::tensor* src = nullptr;
    typename MM::tensor* src_mask = nullptr;
    typename MM::tensor* trg = nullptr;
    typename MM::tensor* trg_y = nullptr;
    typename MM::tensor* trg_mask = nullptr;
    size_t ntokens = 0;
};

template<typename MM>
size_t load_weights(MM& memory_manager,
                    const std::filesystem::path& weights_path,
                    const raul::Names& paramNames,
                    const std::string& file_prefix = "0_",
                    const std::string& tensor_prefix = "")
{
    using namespace std;

    raul::DataLoader dataLoader;
    size_t count = 0;
    for (auto tensorName : paramNames)
    {
        auto tensorNameStr = tensorName.str().substr(tensor_prefix.size());
        auto name = tensorName;

        raul::Common::replaceAll(tensorNameStr, TENSOR_SEP, ".");
        raul::Common::replaceAll(tensorNameStr, "[", ".");
        raul::Common::replaceAll(tensorNameStr, "]", "");

        tensorNameStr = file_prefix + tensorNameStr;

        if (raul::Common::endsWith(name, "norm"))
        {
            memory_manager[name + "::Weights"] = dataLoader.loadData(weights_path / (tensorNameStr + ".a_2.data"), memory_manager[name + "::Weights"].size(), 1, 1);
            memory_manager[name + "::Biases"] = dataLoader.loadData(weights_path / (tensorNameStr + ".b_2.data"), memory_manager[name + "::Biases"].size(), 1, 1);
            count += 2;
        }
        else
        {
            if (memory_manager.tensorExists(name + "::Biases"))
            {
                memory_manager[name + "::Biases"] = dataLoader.loadData(weights_path / (tensorNameStr + ".bias.data"), memory_manager[name + "::Biases"].size(), 1, 1);
                ++count;
            }
            memory_manager[name + "::Weights"] = dataLoader.loadData(weights_path / (tensorNameStr + ".weight.data"), memory_manager[name + "::Weights"].size(), 1, 1);
            ++count;
        }
    }

    return count;
}

// transformer_test1.py
TEST(TestTransformer, TransformerModelCopyTaskTrainingUnit)
{
    PROFILE_TEST
    using namespace raul;

    raul::Workflow work;
    raul::NetworkParameters& networkParameters = work.getNetworkParameters();
    raul::MemoryManager& memory_manager = work.getMemoryManager();

    DataLoader dataLoader;

    constexpr unsigned int HEADS = 4;
    constexpr unsigned int LENGTH = 4;
    constexpr unsigned int MODEL_SIZE = 64;
    constexpr unsigned int SRC_VOCAB = 11;
    constexpr unsigned int TGT_VOCAB = 11;
    constexpr unsigned int FEED_FORWARD_SIZE = 2048;
    constexpr float DROPOUT_RATE = 0.0f;
    constexpr size_t BATCHES_TRAIN = 20;
    constexpr size_t BATCH_SIZE = 30;
    constexpr dtype LEARNING_RATE = 0.002_dt;

    constexpr size_t nEpoch = 20;
    constexpr size_t nEpochFineTune = 1;

    constexpr size_t SIZE = 10;

    constexpr float EPS = 1e-2f;

    dtype idealLoss[] = { 0.956_dt };
    (void)idealLoss;

    auto config = std::to_string(MODEL_SIZE) + "_" + std::to_string(HEADS) + "_" + std::to_string(LENGTH) + "__" + std::to_string(nEpoch);

    const auto weights_path = tools::getTestAssetsDir() / "transformer" / config;

    Tensor input("input", 1, BATCHES_TRAIN * (nEpoch + nEpochFineTune), BATCH_SIZE, SIZE);
    input = dataLoader.loadData(weights_path / "input.data", SIZE, BATCH_SIZE, BATCHES_TRAIN * (nEpoch + nEpochFineTune));

    std::vector<Tensor*> trainData;
    auto input2d = input.reshape(yato::dims(input.getDepth(), input.size() / input.getDepth()));
    for (size_t i = 0; i < input.getDepth(); ++i)
    {
        Tensor* t = new Tensor(BATCH_SIZE, 1, 1, SIZE);
        std::copy(input2d[i].begin(), input2d[i].end(), t->begin());
        trainData.push_back(t);
    }

    auto inS = trainData[0]->getShape();

    work.add<raul::DataLayer>("data1", raul::DataParams{ { "src" }, inS[1], inS[2], inS[3] });
    work.add<raul::DataLayer>("data2", raul::DataParams{ { "src_mask" }, inS[1], inS[2], inS[3] });
    work.add<raul::DataLayer>("data3", raul::DataParams{ { "trg" }, 1, 1, SIZE - 1 });
    work.add<raul::DataLayer>("data4", raul::DataParams{ { "trg_y" }, 1, 1, SIZE - 1 });
    work.add<raul::DataLayer>("data5", raul::DataParams{ { "trg_mask" }, 1, SIZE - 1, SIZE - 1 });
    work.add<raul::DataLayer>("data6", raul::DataParams{ { "target" }, 1, SIZE - 1, TGT_VOCAB });

    TransformerParams transformerParams{ { "src", "trg", "src_mask", "trg_mask" }, { "out" }, SRC_VOCAB, TGT_VOCAB, LENGTH, MODEL_SIZE, FEED_FORWARD_SIZE, HEADS, DROPOUT_RATE };

    TransformerModel("T", transformerParams, networkParameters);

    CreateGenerator(Name("T") / "generator", { { "out" }, { "gen" } }, TGT_VOCAB, networkParameters);

    work.add<ReshapeLayer>("gen_reshape", ViewParams{ "gen", "gen_view", 1, -1, (int)TGT_VOCAB });

    work.add<LabelSmoothing>("ls", LabelSmoothingParams{ "target", "ls", 0.0_dt, 0 });
    work.add<KLDivLoss>("loss", LossParams{ { "gen_view", "ls" }, { "loss" }, LossParams::Reduction::Sum });

    work.preparePipelines();
    work.setBatchSize(BATCH_SIZE);
    work.prepareMemoryForTraining();

    memory_manager["src"] = 0.0_dt;
    memory_manager["src_mask"] = 0.0_dt;
    memory_manager["trg"] = 0.0_dt;
    memory_manager["trg_y"] = 0.0_dt;
    memory_manager["trg_mask"] = 0.0_dt;

    std::vector<ParamAndGrad> trainableParams = work.getTrainableParameters();

    std::set<std::string> paramsSet;
    // auto params = work.getParameters();
    auto params = work.getTrainableParameterNames();
    for (auto& p : params)
    {
        auto s = p;
        auto pos = s.str().rfind("::");
        if (pos != std::string::npos)
        {
            s = s.str().substr(0, pos);
        }
        paramsSet.insert(s);
    }

    Names paramNames(paramsSet.begin(), paramsSet.end());

    size_t tensorsLoaded = load_weights(memory_manager, weights_path, paramNames, std::to_string(nEpoch) + "_", std::string("T") + TENSOR_SEP);

    EXPECT_EQ(tensorsLoaded, paramNames.size() * 2 - 2); // all params have weight and bias except two embeddings - they have only weights

    auto sgd = std::make_shared<optimizers::SGD>(LEARNING_RATE);
    dtype last_loss = 0;
    for (size_t epoch = nEpoch; epoch < nEpoch + nEpochFineTune; ++epoch)
    {
        size_t total_tokens = 0;
        size_t tokens = 0;
        dtype total_loss = 0;
        for (size_t batch = 0; batch < BATCHES_TRAIN; ++batch)
        {
            Tensor data(trainData[epoch * BATCHES_TRAIN + batch]->getShape(), TORANGE(*trainData[epoch * BATCHES_TRAIN + batch]));
            for (size_t q = 0; q < BATCH_SIZE; ++q)
            {
                data[q * SIZE] = 1.0_dt;
            }

            TransformerBatch B(data, data, memory_manager, 0.0_dt);

            TensorU8 classes(memory_manager["trg_y"].getShape());
            std::transform(memory_manager["trg_y"].begin(), memory_manager["trg_y"].end(), classes.begin(), [](dtype t) { return static_cast<uint8_t>(t); });
            auto& encodedClasses = dataLoader.buildOneHotVector(classes, TGT_VOCAB);
            memory_manager["target"] = TORANGE(encodedClasses);

            work.forwardPassTraining();

            Tensor& loss = memory_manager["loss"];

            total_loss += loss[0];
            loss /= (dtype)B.ntokens;
            total_tokens += B.ntokens;
            tokens += B.ntokens;

            work.backwardPassTraining();

            for (auto p : trainableParams)
            {
                sgd->operator()(memory_manager, p.Param, p.Gradient);
            }
        }
        last_loss = total_loss / static_cast<float>(total_tokens);
        std::cout << "Epoch " << epoch << ": " << last_loss << std::endl;
        EXPECT_NEAR(last_loss, idealLoss[epoch - nEpoch], EPS);
    }
}

// transformer_test1.py
TEST(TestTransformer, TransformerModelCopyTaskTraining)
{
    PROFILE_TEST
    using namespace raul;

    raul::Workflow work;
    raul::NetworkParameters& networkParameters = work.getNetworkParameters();
    raul::MemoryManager& memory_manager = work.getMemoryManager();

    DataLoader dataLoader;

    constexpr unsigned int HEADS = 4;
    constexpr unsigned int LENGTH = 4;
    constexpr unsigned int MODEL_SIZE = 64;
    constexpr unsigned int SRC_VOCAB = 11;
    constexpr unsigned int TGT_VOCAB = 11;
    constexpr unsigned int FEEED_FORWARD_SIZE = 2048;
    constexpr float DROPOUT_RATE = 0.0f;
    constexpr size_t BATCHES_TRAIN = 20;
    constexpr size_t BATCH_SIZE = 30;
    constexpr dtype LEARNING_RATE = 0.002_dt;

    constexpr size_t nEpoch = 20;
    constexpr size_t nEpochFineTune = 10;

    constexpr dtype idealFirstLoss = 0.956_dt;
    constexpr dtype EPS = 1e-2_dt;

    constexpr size_t SIZE = 10;

    auto config = std::to_string(MODEL_SIZE) + "_" + std::to_string(HEADS) + "_" + std::to_string(LENGTH) + "__" + std::to_string(nEpoch);

    const auto weights_path = tools::getTestAssetsDir() / "transformer" / config;

    Tensor input("input", 1, BATCHES_TRAIN * (nEpoch + nEpochFineTune), BATCH_SIZE, SIZE);
    DataLoader::loadData(weights_path / "input.data", input, SIZE, BATCH_SIZE, BATCHES_TRAIN * (nEpoch + nEpochFineTune));

    std::vector<Tensor*> trainData;
    auto input2d = input.reshape(yato::dims(input.getDepth(), input.size() / input.getDepth()));
    for (size_t i = 0; i < input.getDepth(); ++i)
    {
        Tensor* t = new Tensor(BATCH_SIZE, 1, 1, SIZE);
        std::copy(input2d[i].begin(), input2d[i].end(), t->begin());
        trainData.push_back(t);
    }

    auto inS = trainData[0]->getShape();

    work.add<raul::DataLayer>("data1", raul::DataParams{ { "src" }, inS[1], inS[2], inS[3] });
    work.add<raul::DataLayer>("data2", raul::DataParams{ { "src_mask" }, inS[1], inS[2], inS[3] });
    work.add<raul::DataLayer>("data3", raul::DataParams{ { "trg" }, 1, 1, SIZE - 1 });
    work.add<raul::DataLayer>("data4", raul::DataParams{ { "trg_y" }, 1, 1, SIZE - 1 });
    work.add<raul::DataLayer>("data5", raul::DataParams{ { "trg_mask" }, 1, SIZE - 1, SIZE - 1 });
    work.add<raul::DataLayer>("data6", raul::DataParams{ { "target" }, 1, SIZE - 1, TGT_VOCAB });

    TransformerParams transformerParams{ { "src", "trg", "src_mask", "trg_mask" }, { "out" }, SRC_VOCAB, TGT_VOCAB, LENGTH, MODEL_SIZE, FEEED_FORWARD_SIZE, HEADS, DROPOUT_RATE };

    TransformerModel("T", transformerParams, networkParameters);

    CreateGenerator(Name("T") / "generator", { { "out" }, { "gen" } }, TGT_VOCAB, networkParameters);

    work.add<ReshapeLayer>("gen_reshape", ViewParams{ "gen", "gen_view", 1, -1, (int)TGT_VOCAB });

    work.add<LabelSmoothing>("ls", LabelSmoothingParams{ "target", "ls", 0.0_dt, 0 });
    work.add<KLDivLoss>("loss", LossParams{ { "gen_view", "ls" }, { "loss" }, LossParams::Reduction::Sum });

    work.preparePipelines();
    work.setBatchSize(BATCH_SIZE);
    work.prepareMemoryForTraining();

    memory_manager["src"] = 0.0_dt;
    memory_manager["src_mask"] = 0.0_dt;
    memory_manager["trg"] = 0.0_dt;
    memory_manager["trg_y"] = 0.0_dt;
    memory_manager["trg_mask"] = 0.0_dt;

    std::vector<ParamAndGrad> trainableParams = work.getTrainableParameters();

    std::set<std::string> paramsSet;
    // auto params = seq.getParameters();
    auto params = work.getTrainableParameterNames();
    for (auto& p : params)
    {
        auto s = p;
        auto pos = s.str().rfind(TENSOR_SEP);
        if (pos != std::string::npos)
        {
            s = s.str().substr(0, pos);
        }
        paramsSet.insert(s);
    }
    Names paramNames(paramsSet.begin(), paramsSet.end());

    size_t tensorsLoaded = load_weights(memory_manager, weights_path, paramNames, std::to_string(nEpoch) + "_", std::string("T") + TENSOR_SEP);

    EXPECT_EQ(tensorsLoaded, paramNames.size() * 2 - 2); // all params have weight and bias except two embeddings - they have only weights

    auto sgd = std::make_shared<optimizers::SGD>(LEARNING_RATE);
    dtype last_loss = 0;
    for (size_t epoch = nEpoch; epoch < nEpoch + nEpochFineTune; ++epoch)
    {
        size_t total_tokens = 0;
        size_t tokens = 0;
        dtype total_loss = 0;
        for (size_t batch = 0; batch < BATCHES_TRAIN; ++batch)
        {
            Tensor data(trainData[epoch * BATCHES_TRAIN + batch]->getShape(), TORANGE(*trainData[epoch * BATCHES_TRAIN + batch]));
            for (size_t q = 0; q < BATCH_SIZE; ++q)
            {
                data[q * SIZE] = 1.0_dt;
            }

            TransformerBatch B(data, data, memory_manager, 0.0_dt);

            TensorU8 classes(memory_manager["trg_y"].getShape());
            std::transform(memory_manager["trg_y"].begin(), memory_manager["trg_y"].end(), classes.begin(), [](dtype t) { return static_cast<uint8_t>(t); });
            auto& encodedClasses = dataLoader.buildOneHotVector(classes, TGT_VOCAB);
            memory_manager["target"] = TORANGE(encodedClasses);

            work.forwardPassTraining();

            Tensor& loss = memory_manager["loss"];

            total_loss += loss[0];
            loss /= (dtype)B.ntokens;
            total_tokens += B.ntokens;
            tokens += B.ntokens;

            work.backwardPassTraining();

            for (auto& p : trainableParams)
            {
                sgd->operator()(memory_manager, p.Param, p.Gradient);
            }
        }
        last_loss = total_loss / static_cast<float>(total_tokens);
        std::cout << "Epoch " << epoch << ": " << last_loss << std::endl;
        if (epoch == nEpoch)
        {
            EXPECT_NEAR(last_loss, idealFirstLoss, EPS);
        }
        else
        {
            EXPECT_TRUE(last_loss < 10_dt);
        }
        ASSERT_FALSE(std::isnan(last_loss));
    }
}

#if defined(ANDROID)
TEST(TestTransformer, TransformerModelCopyTaskTrainingFP16)
{
    PROFILE_TEST
    using namespace raul;

    raul::Workflow work(CompressionMode::NONE, raul::CalculationMode::DETERMINISTIC, raul::AllocationMode::STANDARD, raul::ExecutionTarget::CPUFP16);
    raul::NetworkParameters& networkParameters = work.getNetworkParameters();
    auto& memory_manager = work.getMemoryManager<raul::MemoryManagerFP16>();

    DataLoader dataLoader;

    constexpr unsigned int HEADS = 4;
    constexpr unsigned int LENGTH = 4;
    constexpr unsigned int MODEL_SIZE = 64;
    constexpr unsigned int SRC_VOCAB = 11;
    constexpr unsigned int TGT_VOCAB = 11;
    constexpr unsigned int FEEED_FORWARD_SIZE = 2048;
    constexpr float DROPOUT_RATE = 0.0f;
    constexpr size_t BATCHES_TRAIN = 20;
    constexpr size_t BATCH_SIZE = 30;
    constexpr dtype LEARNING_RATE = 0.002_dt;

    constexpr size_t nEpoch = 20;
    constexpr size_t nEpochFineTune = 10;

    constexpr dtype idealFirstLoss = 0.956_dt;
    constexpr dtype EPS = 1e-2_dt;

    constexpr size_t SIZE = 10;

    auto config = std::to_string(MODEL_SIZE) + "_" + std::to_string(HEADS) + "_" + std::to_string(LENGTH) + "__" + std::to_string(nEpoch);

    const auto weights_path = tools::getTestAssetsDir() / "transformer" / config;

    TensorFP16 input("input", 1, BATCHES_TRAIN * (nEpoch + nEpochFineTune), BATCH_SIZE, SIZE);
    DataLoader::loadData(weights_path / "input.data", input, SIZE, BATCH_SIZE, BATCHES_TRAIN * (nEpoch + nEpochFineTune));

    std::vector<TensorFP16*> trainData;
    auto input2d = input.reshape(yato::dims(input.getDepth(), input.size() / input.getDepth()));
    for (size_t i = 0; i < input.getDepth(); ++i)
    {
        TensorFP16* t = new TensorFP16(BATCH_SIZE, 1, 1, SIZE);
        std::copy(input2d[i].begin(), input2d[i].end(), t->begin());
        trainData.push_back(t);
    }

    auto inS = trainData[0]->getShape();

    work.add<raul::DataLayer>("data1", raul::DataParams{ { "src" }, inS[1], inS[2], inS[3] });
    work.add<raul::DataLayer>("data2", raul::DataParams{ { "src_mask" }, inS[1], inS[2], inS[3] });
    work.add<raul::DataLayer>("data3", raul::DataParams{ { "trg" }, 1, 1, SIZE - 1 });
    work.add<raul::DataLayer>("data4", raul::DataParams{ { "trg_y" }, 1, 1, SIZE - 1 });
    work.add<raul::DataLayer>("data5", raul::DataParams{ { "trg_mask" }, 1, SIZE - 1, SIZE - 1 });
    work.add<raul::DataLayer>("data6", raul::DataParams{ { "target" }, 1, SIZE - 1, TGT_VOCAB });

    TransformerParams transformerParams{ { "src", "trg", "src_mask", "trg_mask" }, { "out" }, SRC_VOCAB, TGT_VOCAB, LENGTH, MODEL_SIZE, FEEED_FORWARD_SIZE, HEADS, DROPOUT_RATE };

    TransformerModel("T", transformerParams, networkParameters);

    CreateGenerator(Name("T") / "generator", { { "out" }, { "gen" } }, TGT_VOCAB, networkParameters);

    work.add<ReshapeLayer>("gen_reshape", ViewParams{ "gen", "gen_view", 1, -1, (int)TGT_VOCAB });

    work.add<LabelSmoothing>("ls", LabelSmoothingParams{ "target", "ls", 0.0_dt, 0 });
    work.add<KLDivLoss>("loss", LossParams{ { "gen_view", "ls" }, { "loss" }, LossParams::Reduction::Sum });

    work.preparePipelines();
    work.setBatchSize(BATCH_SIZE);
    work.prepareMemoryForTraining();

    memory_manager["src"] = 0.0_hf;
    memory_manager["src_mask"] = 0.0_hf;
    memory_manager["trg"] = 0.0_hf;
    memory_manager["trg_y"] = 0.0_hf;
    memory_manager["trg_mask"] = 0.0_hf;

    auto trainableParams = work.getTrainableParameters<MemoryManagerFP16>();

    std::set<std::string> paramsSet;
    // auto params = seq.getParameters();
    auto params = work.getTrainableParameterNames();
    for (auto& p : params)
    {
        auto s = p;
        auto pos = s.str().rfind(TENSOR_SEP);
        if (pos != std::string::npos)
        {
            s = s.str().substr(0, pos);
        }
        paramsSet.insert(s);
    }
    Names paramNames(paramsSet.begin(), paramsSet.end());

    size_t tensorsLoaded = load_weights(memory_manager, weights_path, paramNames, std::to_string(nEpoch) + "_", std::string("T") + TENSOR_SEP);

    EXPECT_EQ(tensorsLoaded, paramNames.size() * 2 - 2); // all params have weight and bias except two embeddings - they have only weights

    auto sgd = std::make_shared<optimizers::SGD>(LEARNING_RATE);
    dtype last_loss = 0;
    for (size_t epoch = nEpoch; epoch < nEpoch + nEpochFineTune; ++epoch)
    {
        size_t total_tokens = 0;
        size_t tokens = 0;
        dtype total_loss = 0;
        for (size_t batch = 0; batch < BATCHES_TRAIN; ++batch)
        {
            TensorFP16 data(trainData[epoch * BATCHES_TRAIN + batch]->getShape(), TORANGE_FP16(*trainData[epoch * BATCHES_TRAIN + batch]));
            for (size_t q = 0; q < BATCH_SIZE; ++q)
            {
                data[q * SIZE] = 1.0_hf;
            }

            TransformerBatch B(data, data, memory_manager, 0.0_hf);

            TensorU8 classes(memory_manager["trg_y"].getShape());
            std::transform(memory_manager["trg_y"].begin(), memory_manager["trg_y"].end(), classes.begin(), [](dtype t) { return static_cast<uint8_t>(t); });
            TensorFP16 encodedClasses("", 1, 1, 1, classes.size() * TGT_VOCAB, 0.0_hf);
            for (size_t i = 0; i < classes.size(); i++)
            {
                encodedClasses[classes[i] + TGT_VOCAB * i] = 1.0_hf;
            }
            memory_manager["target"] = TORANGE_FP16(encodedClasses);

            work.forwardPassTraining();

            TensorFP16& loss = memory_manager["loss"];

            total_loss += toFloat32(loss[0]);
            loss /= (dtype)B.ntokens;
            total_tokens += B.ntokens;
            tokens += B.ntokens;

            work.backwardPassTraining();

            for (auto& p : trainableParams)
            {
                sgd->operator()(memory_manager, p.Param, p.Gradient);
            }
        }
        last_loss = total_loss / static_cast<float>(total_tokens);
        std::cout << "Epoch " << epoch << ": " << last_loss << std::endl;
        if (epoch == nEpoch)
        {
            EXPECT_NEAR(last_loss, idealFirstLoss, EPS);
        }
        else
        {
            EXPECT_TRUE(last_loss < 10_dt);
        }
        ASSERT_FALSE(std::isnan(last_loss));
    }
}
#endif

TEST(TestTransformer, AttentionUnit)
{
    PROFILE_TEST
    using namespace raul;

    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    size_t MODEL_SIZE = 5;
    size_t BATCH_SIZE = 1;
    size_t HEIGHT = 2;
    constexpr dtype DROPOUT_RATE = 0.0_dt;
    constexpr dtype FILL_VALUE = -1e9_dt;

    constexpr dtype eps = 1e-4_dt;

    Tensor realAttn = { 0.9360_dt, 1.0640_dt, 0.9887_dt, 1.0113_dt };
    Tensor realPAttn = { 0.0640_dt, 0.9360_dt, 0.0113_dt, 0.9887_dt };

    work.add<raul::DataLayer>("data1", raul::DataParams{ { "query" }, 1, HEIGHT, MODEL_SIZE });
    work.add<raul::DataLayer>("data2", raul::DataParams{ { "key" }, 1, HEIGHT, MODEL_SIZE });
    work.add<raul::DataLayer>("data3", raul::DataParams{ { "value" }, 1, HEIGHT, HEIGHT });
    work.add<raul::DataLayer>("data4", raul::DataParams{ { "mask" }, 1, HEIGHT, HEIGHT });

    work.add<TransposeLayer>("t", raul::TransposingParams{ "key", "key_t", "width", "height" });
    work.add<MatMulLayer>("attn", raul::MatMulParams{ { "query", "key_t" }, "scores", TODTYPE(1.0_dt / sqrt(static_cast<raul::dtype>(MODEL_SIZE))) });
    work.add<MaskedFillLayer>("mfill", raul::MaskedFillParams{ { "scores", "mask" }, { "scores_masked" }, FILL_VALUE, true });
    work.add<SoftMaxActivation>("sm", raul::BasicParamsWithDim{ { "scores_masked" }, { "p_attn_" }, "width" });
    work.add<DropoutLayer>("dropout", raul::DropoutParams{ { "p_attn_" }, { "p_attn_do" }, DROPOUT_RATE });
    work.add<SplitterLayer>("splitter", raul::BasicParams{ { "p_attn_do" }, { "p_attn_do_", "p_attn" } });
    work.add<MatMulLayer>("mm", raul::MatMulParams{ { "p_attn_do_", "value" }, "res" });
    TENSORS_CREATE(BATCH_SIZE);

    memory_manager["query"] = { 1._dt, 1._dt, 2._dt, 0._dt, 5._dt, -1._dt, 2._dt, 2._dt, 0._dt, 5._dt };
    memory_manager["key"] = { -1._dt, 4._dt, 1._dt, 2._dt, 1._dt, -3._dt, 4._dt, 5._dt, 2._dt, 1._dt };
    memory_manager["value"] = { 0._dt, 2._dt, 1._dt, 1._dt };
    memory_manager["mask"] = { 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt };

    work.forwardPassTraining();

    const Tensor& p_attn = memory_manager["p_attn"];
    const Tensor& res = memory_manager["res"];

    EXPECT_EQ(p_attn.size(), realPAttn.size());
    for (size_t i = 0; i < p_attn.size(); ++i)
        EXPECT_NEAR(p_attn[i], realPAttn[i], eps);

    EXPECT_EQ(res.size(), realAttn.size());
    for (size_t i = 0; i < res.size(); ++i)
        EXPECT_NEAR(res[i], realAttn[i], eps);

    memory_manager[Name("res").grad()] = { 0.9360_dt, 1.0640_dt, 0.9887_dt, 1.0113_dt };
    memory_manager[Name("p_attn").grad()] = 0.0_dt;

    work.backwardPassTraining();

    const auto& query_nabla = memory_manager[Name("query").grad()];
    const auto& key_nabla = memory_manager[Name("key").grad()];
    const auto& value_nabla = memory_manager[Name("value").grad()];

    Tensor realQ_nabla = { 6.8549e-03_dt, 3.7253e-08_dt, -1.3710e-02_dt, 1.8626e-08_dt, 9.3132e-09_dt, 2.2586e-04_dt, -1.8816e-07_dt, -4.5167e-04_dt, -9.4078e-08_dt, -4.7039e-08_dt };
    Tensor realK_nabla = { 0.0033_dt, 0.0037_dt, 0.0071_dt, 0.0000_dt, 0.0177_dt, -0.0033_dt, -0.0037_dt, -0.0071_dt, 0.0000_dt, -0.0177_dt };
    Tensor realV_nabla = { 0.0710_dt, 0.0795_dt, 1.8537_dt, 1.9958_dt };

    EXPECT_EQ(realQ_nabla.size(), query_nabla.size());
    for (size_t i = 0; i < realQ_nabla.size(); ++i)
        EXPECT_NEAR(realQ_nabla[i], query_nabla[i], eps);

    EXPECT_EQ(realK_nabla.size(), key_nabla.size());
    for (size_t i = 0; i < realK_nabla.size(); ++i)
        EXPECT_NEAR(realK_nabla[i], key_nabla[i], eps);

    EXPECT_EQ(realV_nabla.size(), value_nabla.size());
    for (size_t i = 0; i < realV_nabla.size(); ++i)
        EXPECT_NEAR(realV_nabla[i], value_nabla[i], eps);
}

TEST(TestTransformer, AttentionLayerUnit)
{
    PROFILE_TEST
    using namespace raul;

    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    size_t MODEL_SIZE = 5;
    size_t BATCH_SIZE = 1;
    size_t HEIGHT = 2;
    constexpr dtype DROPOUT_RATE = 0.0_dt;

    constexpr dtype eps = 1e-4_dt;

    Tensor realAttn = { 0.9360_dt, 1.0640_dt, 0.9887_dt, 1.0113_dt };
    Tensor realPAttn = { 0.0640_dt, 0.9360_dt, 0.0113_dt, 0.9887_dt };

    work.add<raul::DataLayer>("data1", raul::DataParams{ { "query" }, 1, HEIGHT, MODEL_SIZE });
    work.add<raul::DataLayer>("data2", raul::DataParams{ { "key" }, 1, HEIGHT, MODEL_SIZE });
    work.add<raul::DataLayer>("data3", raul::DataParams{ { "value" }, 1, HEIGHT, HEIGHT });
    work.add<raul::DataLayer>("data4", raul::DataParams{ { "mask" }, 1, HEIGHT, HEIGHT });

    AttentionLayer("attn", { { "query", "value", "key", "mask" }, { "res", "p_attn" }, DROPOUT_RATE }, networkParameters);
    TENSORS_CREATE(BATCH_SIZE);

    memory_manager["query"] = { 1._dt, 1._dt, 2._dt, 0._dt, 5._dt, -1._dt, 2._dt, 2._dt, 0._dt, 5._dt };
    memory_manager["key"] = { -1._dt, 4._dt, 1._dt, 2._dt, 1._dt, -3._dt, 4._dt, 5._dt, 2._dt, 1._dt };
    memory_manager["value"] = { 0._dt, 2._dt, 1._dt, 1._dt };
    memory_manager["mask"] = { 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt };

    work.forwardPassTraining();

    const auto& p_attn = memory_manager["p_attn"];
    const auto& res = memory_manager["res"];

    EXPECT_EQ(p_attn.size(), realPAttn.size());
    for (size_t i = 0; i < p_attn.size(); ++i)
        EXPECT_NEAR(p_attn[i], realPAttn[i], eps);

    EXPECT_EQ(res.size(), realAttn.size());
    for (size_t i = 0; i < res.size(); ++i)
        EXPECT_NEAR(res[i], realAttn[i], eps);

    Tensor resGrad = { 0.9360_dt, 1.0640_dt, 0.9887_dt, 1.0113_dt };
    Tensor pattnGrad = { 0._dt, 0._dt, 0._dt, 0._dt };
    memory_manager[Name("res").grad()] = TORANGE(resGrad);
    memory_manager[Name("p_attn").grad()] = TORANGE(pattnGrad);

    work.backwardPassTraining();

    const auto& query_nabla = memory_manager[Name("query").grad()];
    const auto& key_nabla = memory_manager[Name("key").grad()];
    const auto& value_nabla = memory_manager[Name("value").grad()];

    Tensor realQ_nabla = { 6.8549e-03_dt, 3.7253e-08_dt, -1.3710e-02_dt, 1.8626e-08_dt, 9.3132e-09_dt, 2.2586e-04_dt, -1.8816e-07_dt, -4.5167e-04_dt, -9.4078e-08_dt, -4.7039e-08_dt };
    Tensor realK_nabla = { 0.0033_dt, 0.0037_dt, 0.0071_dt, 0.0000_dt, 0.0177_dt, -0.0033_dt, -0.0037_dt, -0.0071_dt, 0.0000_dt, -0.0177_dt };
    Tensor realV_nabla = { 0.0710_dt, 0.0795_dt, 1.8537_dt, 1.9958_dt };
    EXPECT_EQ(realQ_nabla.size(), query_nabla.size());
    for (size_t i = 0; i < realQ_nabla.size(); ++i)
        EXPECT_NEAR(realQ_nabla[i], query_nabla[i], eps);

    EXPECT_EQ(realK_nabla.size(), key_nabla.size());
    for (size_t i = 0; i < realK_nabla.size(); ++i)
        EXPECT_NEAR(realK_nabla[i], key_nabla[i], eps);

    EXPECT_EQ(realV_nabla.size(), value_nabla.size());
    for (size_t i = 0; i < realV_nabla.size(); ++i)
        EXPECT_NEAR(realV_nabla[i], value_nabla[i], eps);
}

TEST(TestTransformer, AttentionLayerNoMaskNoPAttnUnit)
{
    PROFILE_TEST
    using namespace raul;

    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    size_t MODEL_SIZE = 5;
    size_t BATCH_SIZE = 1;
    size_t HEIGHT = 2;
    constexpr dtype DROPOUT_RATE = 0.0_dt;

    constexpr dtype eps = 1e-4_dt;

    Tensor realAttn = { 0.9360_dt, 1.0640_dt, 0.9887_dt, 1.0113_dt };

    work.add<raul::DataLayer>("data1", raul::DataParams{ { "query" }, 1, HEIGHT, MODEL_SIZE });
    work.add<raul::DataLayer>("data2", raul::DataParams{ { "key" }, 1, HEIGHT, MODEL_SIZE });
    work.add<raul::DataLayer>("data3", raul::DataParams{ { "value" }, 1, HEIGHT, HEIGHT });

    AttentionLayer("attn", { { "query", "value", "key" }, { "res" }, DROPOUT_RATE }, networkParameters);
    TENSORS_CREATE(BATCH_SIZE);

    memory_manager["query"] = { 1._dt, 1._dt, 2._dt, 0._dt, 5._dt, -1._dt, 2._dt, 2._dt, 0._dt, 5._dt };
    memory_manager["key"] = { -1._dt, 4._dt, 1._dt, 2._dt, 1._dt, -3._dt, 4._dt, 5._dt, 2._dt, 1._dt };
    memory_manager["value"] = { 0._dt, 2._dt, 1._dt, 1._dt };

    work.forwardPassTraining();

    const auto& res = memory_manager["res"];

    EXPECT_EQ(res.size(), realAttn.size());
    for (size_t i = 0; i < res.size(); ++i)
        EXPECT_NEAR(res[i], realAttn[i], eps);

    Tensor resGrad = { 0.9360_dt, 1.0640_dt, 0.9887_dt, 1.0113_dt };
    memory_manager[Name("res").grad()] = TORANGE(resGrad);

    work.backwardPassTraining();

    const auto& query_nabla = memory_manager[Name("query").grad()];
    const auto& key_nabla = memory_manager[Name("key").grad()];
    const auto& value_nabla = memory_manager[Name("value").grad()];

    Tensor realQ_nabla = { 6.8549e-03_dt, 3.7253e-08_dt, -1.3710e-02_dt, 1.8626e-08_dt, 9.3132e-09_dt, 2.2586e-04_dt, -1.8816e-07_dt, -4.5167e-04_dt, -9.4078e-08_dt, -4.7039e-08_dt };
    Tensor realK_nabla = { 0.0033_dt, 0.0037_dt, 0.0071_dt, 0.0000_dt, 0.0177_dt, -0.0033_dt, -0.0037_dt, -0.0071_dt, 0.0000_dt, -0.0177_dt };
    Tensor realV_nabla = { 0.0710_dt, 0.0795_dt, 1.8537_dt, 1.9958_dt };

    EXPECT_EQ(realQ_nabla.size(), query_nabla.size());
    for (size_t i = 0; i < realQ_nabla.size(); ++i)
        EXPECT_NEAR(realQ_nabla[i], query_nabla[i], eps);

    EXPECT_EQ(realK_nabla.size(), key_nabla.size());
    for (size_t i = 0; i < realK_nabla.size(); ++i)
        EXPECT_NEAR(realK_nabla[i], key_nabla[i], eps);

    EXPECT_EQ(realV_nabla.size(), value_nabla.size());
    for (size_t i = 0; i < realV_nabla.size(); ++i)
        EXPECT_NEAR(realV_nabla[i], value_nabla[i], eps);
}

TEST(TestTransformer, SelfAttentionLayerUnit)
{
    PROFILE_TEST
    using namespace raul;

    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    size_t MODEL_SIZE = 5;
    size_t BATCH_SIZE = 1;
    size_t HEIGHT = 2;
    constexpr dtype DROPOUT_RATE = 0.0_dt;
    [[maybe_unused]] constexpr dtype FILL_VALUE = -1e9_dt;

    constexpr dtype eps = 1e-4_dt;

    Tensor realAttn = { 0.2200_dt, 1.3900_dt, 2.0000_dt, 0.0000_dt, 5.0000_dt, -0.7136_dt, 1.8568_dt, 2.0000_dt, 0.0000_dt, 5.0000_dt };
    Tensor realPAttn = { 0.6100_dt, 0.3900_dt, 0.1432_dt, 0.8568_dt };

    work.add<raul::DataLayer>("data1", raul::DataParams{ { "query" }, 1, HEIGHT, MODEL_SIZE });
    work.add<raul::DataLayer>("data4", raul::DataParams{ { "mask" }, 1, HEIGHT, HEIGHT });

    AttentionLayer("attn", { { "query", "query", "query", "mask" }, { "res", "p_attn" }, DROPOUT_RATE }, networkParameters);
    TENSORS_CREATE(BATCH_SIZE);

    memory_manager["query"] = { 1._dt, 1._dt, 2._dt, 0._dt, 5._dt, -1._dt, 2._dt, 2._dt, 0._dt, 5._dt };
    memory_manager["mask"] = { 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt };

    work.forwardPassTraining();

    const auto& p_attn = memory_manager["p_attn"];
    const auto& res = memory_manager["res"];

    EXPECT_EQ(p_attn.size(), realPAttn.size());
    for (size_t i = 0; i < p_attn.size(); ++i)
        EXPECT_NEAR(p_attn[i], realPAttn[i], eps);

    EXPECT_EQ(res.size(), realAttn.size());
    for (size_t i = 0; i < res.size(); ++i)
        EXPECT_NEAR(res[i], realAttn[i], eps);

    Tensor resGrad = { 1._dt, 1._dt, 2._dt, 0._dt, 5._dt, -1._dt, 2._dt, 2._dt, 0._dt, 5._dt };
    Tensor pattnGrad = { 0._dt, 0._dt, 0._dt, 0._dt };
    memory_manager[Name("res").grad()] = TORANGE(resGrad);
    memory_manager[Name("p_attn").grad()] = TORANGE(pattnGrad);

    work.backwardPassTraining();

    const auto& query_nabla = memory_manager[Name("query").grad()];

    Tensor realQ_nabla = { 1.0054_dt, 0.4574_dt, 1.2802_dt, 0.0000_dt, 3.2004_dt, -1.2317_dt, 2.6557_dt, 2.7198_dt, 0.0000_dt, 6.7996_dt };

    EXPECT_EQ(realQ_nabla.size(), query_nabla.size());
    for (size_t i = 0; i < realQ_nabla.size(); ++i)
        EXPECT_NEAR(realQ_nabla[i], query_nabla[i], eps);
}

TEST(TestTransformer, SelfAttentionLayerWithSplitterUnit)
{
    PROFILE_TEST
    using namespace raul;

    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    size_t MODEL_SIZE = 5;
    size_t BATCH_SIZE = 1;
    size_t HEIGHT = 2;
    constexpr dtype DROPOUT_RATE = 0.0_dt;
    constexpr dtype eps = 1e-4_dt;

    Tensor realAttn = { 0.2200_dt, 1.3900_dt, 2.0000_dt, 0.0000_dt, 5.0000_dt, -0.7136_dt, 1.8568_dt, 2.0000_dt, 0.0000_dt, 5.0000_dt };
    Tensor realPAttn = { 0.6100_dt, 0.3900_dt, 0.1432_dt, 0.8568_dt };

    work.add<raul::DataLayer>("data1", raul::DataParams{ { "query" }, 1, HEIGHT, MODEL_SIZE });
    work.add<raul::DataLayer>("data4", raul::DataParams{ { "mask" }, 1, HEIGHT, HEIGHT });

    work.add<SplitterLayer>("splitter", raul::BasicParams{ { "query" }, { "q", "k", "v" } });
    AttentionLayer("attn", { { "q", "v", "k", "mask" }, { "res", "p_attn" }, DROPOUT_RATE }, networkParameters);
    TENSORS_CREATE(BATCH_SIZE);

    memory_manager["query"] = { 1._dt, 1._dt, 2._dt, 0._dt, 5._dt, -1._dt, 2._dt, 2._dt, 0._dt, 5._dt };
    memory_manager["mask"] = { 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt };

    work.forwardPassTraining();

    const auto& p_attn = memory_manager["p_attn"];
    const auto& res = memory_manager["res"];

    EXPECT_EQ(p_attn.size(), realPAttn.size());
    for (size_t i = 0; i < p_attn.size(); ++i)
        EXPECT_NEAR(p_attn[i], realPAttn[i], eps);

    EXPECT_EQ(res.size(), realAttn.size());
    for (size_t i = 0; i < res.size(); ++i)
        EXPECT_NEAR(res[i], realAttn[i], eps);

    memory_manager[Name("res").grad()] = { 1._dt, 1._dt, 2._dt, 0._dt, 5._dt, -1._dt, 2._dt, 2._dt, 0._dt, 5._dt };
    memory_manager[Name("p_attn").grad()] = 0._dt;

    work.backwardPassTraining();

    const auto& query_nabla = memory_manager[Name("query").grad()];

    Tensor realQ_nabla = { 1.0054_dt, 0.4574_dt, 1.2802_dt, 0.0000_dt, 3.2004_dt, -1.2317_dt, 2.6557_dt, 2.7198_dt, 0.0000_dt, 6.7996_dt };

    EXPECT_EQ(realQ_nabla.size(), query_nabla.size());
    for (size_t i = 0; i < realQ_nabla.size(); ++i)
        EXPECT_NEAR(realQ_nabla[i], query_nabla[i], eps);
}

TEST(TestTransformer, MultiHeadAttentionLayerUnit)
{
    PROFILE_TEST
    using namespace raul;

    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    size_t MODEL_SIZE = 9;
    size_t BATCH_SIZE = 2;
    size_t HEIGHT = 2;
    unsigned int HEADS = 3;
    constexpr dtype DROPOUT_RATE = 0.0_dt;

    constexpr dtype eps = 1e-4_dt;

    Tensor realOut(BATCH_SIZE,
                   1,
                   HEIGHT,
                   MODEL_SIZE,
                   {
                       21.9955_dt, 21.9955_dt, 21.9955_dt, 21.9955_dt, 21.9955_dt, 21.9955_dt, 21.9955_dt, 21.9955_dt, 21.9955_dt, 22.1482_dt, 22.1482_dt, 22.1482_dt,
                       22.1482_dt, 22.1482_dt, 22.1482_dt, 22.1482_dt, 22.1482_dt, 22.1482_dt, 32.3676_dt, 32.3676_dt, 32.3676_dt, 32.3676_dt, 32.3676_dt, 32.3676_dt,
                       32.3676_dt, 32.3676_dt, 32.3676_dt, 32.4809_dt, 32.4809_dt, 32.4809_dt, 32.4809_dt, 32.4809_dt, 32.4809_dt, 32.4809_dt, 32.4809_dt, 32.4809_dt,
                   });

    Tensor realGrad(BATCH_SIZE,
                    1,
                    HEIGHT,
                    MODEL_SIZE,
                    {
                        -0.8971_dt, -0.8971_dt, -0.8971_dt, -0.8971_dt, -0.8971_dt, -0.8971_dt, -0.8971_dt, -0.8971_dt, -0.8971_dt, 26.3311_dt, 26.3311_dt, 26.3311_dt,
                        26.3311_dt, 26.3311_dt, 26.3311_dt, 26.3311_dt, 26.3311_dt, 26.3311_dt, -2.2549_dt, -2.2549_dt, -2.2549_dt, -2.2549_dt, -2.2549_dt, -2.2549_dt,
                        -2.2549_dt, -2.2549_dt, -2.2549_dt, 40.0683_dt, 40.0683_dt, 40.0683_dt, 40.0683_dt, 40.0683_dt, 40.0683_dt, 40.0683_dt, 40.0683_dt, 40.0683_dt,
                    });

    work.add<raul::DataLayer>("data", raul::DataParams{ { "in" }, 1, HEIGHT, MODEL_SIZE });

    MultiHeadAttentionLayer("attn", MultiHeadAttentionParams{ { "in" }, "out", HEADS, DROPOUT_RATE }, networkParameters);
    TENSORS_CREATE(BATCH_SIZE);

    memory_manager["in"] = { 0.1111_dt, 0.1112_dt, 0.1113_dt, 0.1114_dt, 0.1115_dt, 0.1116_dt, 0.1117_dt, 0.1118_dt, 0.1119_dt, 0.5121_dt, 0.1122_dt, 0.1123_dt,
                             0.1124_dt, 0.1125_dt, 0.1126_dt, 0.1127_dt, 0.1128_dt, 0.1129_dt, 0.2111_dt, 0.2112_dt, 0.2113_dt, 0.2114_dt, 0.2115_dt, 0.2116_dt,
                             0.2117_dt, 0.2118_dt, 0.2119_dt, 0.8121_dt, 0.2122_dt, 0.2123_dt, 0.2124_dt, 0.2125_dt, 0.2126_dt, 0.2127_dt, 0.2128_dt, 0.2129_dt };

    for (size_t i = 0; i < 4; ++i)
    {
        auto suffix = "[" + std::to_string(i) + "]";
        auto& w = memory_manager[Name("attn") / "linears" + suffix + "::Weights"];
        auto& b = memory_manager[Name("attn") / "linears" + suffix + "::Biases"];

        w = 1.0_dt;
        b = 1.0_dt;
    }

    work.forwardPassTraining();

    const auto& out = memory_manager["out"];

    EXPECT_EQ(out.size(), realOut.size());
    for (size_t i = 0; i < out.size(); ++i)
        EXPECT_NEAR(out[i], realOut[i], eps);

    memory_manager[raul::Name("out").grad()] = { 0.1111_dt, 0.1112_dt, 0.1113_dt, 0.1114_dt, 0.1115_dt, 0.1116_dt, 0.1117_dt, 0.1118_dt, 0.1119_dt, 0.8121_dt, 0.1122_dt, 0.1123_dt,
                                                 0.1124_dt, 0.1125_dt, 0.1126_dt, 0.1127_dt, 0.1128_dt, 0.1129_dt, 0.2111_dt, 0.2112_dt, 0.2113_dt, 0.2114_dt, 0.2115_dt, 0.2116_dt,
                                                 0.2117_dt, 0.2118_dt, 0.2119_dt, 0.5121_dt, 0.2122_dt, 0.2123_dt, 0.2124_dt, 0.2125_dt, 0.2126_dt, 0.2127_dt, 0.2128_dt, 0.2129_dt };

    work.backwardPassTraining();

    const auto& query_nabla = memory_manager[raul::Name("in").grad()];

    EXPECT_EQ(query_nabla.size(), realGrad.size());
    for (size_t i = 0; i < query_nabla.size(); ++i)
        EXPECT_NEAR(realGrad[i], query_nabla[i], eps);
}

}
