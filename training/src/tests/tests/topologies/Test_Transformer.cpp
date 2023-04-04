// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

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

#include <training/base/common/Common.h>
#include <training/base/layers/activations/LogSoftMaxActivation.h>
#include <training/base/layers/activations/SoftMaxActivation.h>
#include <training/base/layers/basic/DropoutLayer.h>
#include <training/base/layers/basic/LabelSmoothing.h>
#include <training/base/layers/basic/MaskedFillLayer.h>
#include <training/base/layers/basic/MatMulLayer.h>
#include <training/base/layers/basic/PositionalEncoding.h>
#include <training/base/layers/basic/ReshapeLayer.h>
#include <training/base/layers/basic/SplitterLayer.h>
#include <training/base/layers/basic/TransposeLayer.h>
#include <training/base/layers/basic/trainable/Embedding.h>
#include <training/base/layers/basic/trainable/LinearLayer.h>
#include <training/base/layers/composite/AttentionLayer.h>
#include <training/base/layers/composite/MultiHeadAttention.h>
#include <training/base/layers/composite/Transformer.h>
#include <training/base/loss/KLDivLoss.h>
#include <training/base/loss/NegativeLogLikelihoodLoss.h>
#include <training/compiler/Layers.h>
#include <training/compiler/Workflow.h>
#include <training/base/optimizers/SGD.h>

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
