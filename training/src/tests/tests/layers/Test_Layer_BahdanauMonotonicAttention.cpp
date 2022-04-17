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
#include <training/base/layers/basic/DataLayer.h>
#include <training/base/layers/basic/ElementWiseMulLayer.h>
#include <training/base/layers/basic/SplitterLayer.h>
#include <training/base/layers/basic/TensorLayer.h>
#include <training/base/layers/composite/BahdanauMonotonicAttentionLayer.h>
#include <training/compiler/Workflow.h>

namespace UT
{
using namespace raul;

TEST(TestLayerBahdanauMonotonicAttention, GetTrainableParametersUnit)
{
    PROFILE_TEST
    // See bahdanau_attention.py
    // Test parameters
    const size_t numUnits = 3;
    const size_t alignmentsSize = 5;
    const size_t anyNumber = 7;
    const size_t batchSize = 2;
    const size_t goldenTrainableParams = 6;

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    work.add<DataLayer>("data_query", DataParams{ { "query" }, 1u, 1u, numUnits });
    work.add<DataLayer>("data_state", DataParams{ { "state" }, 1u, 1u, alignmentsSize });
    work.add<DataLayer>("data_memory", DataParams{ { "memory" }, 1u, alignmentsSize, anyNumber });

    // Apply function
    BahdanauMonotonicAttentionLayer("battn", BahdanauAttentionParams{ BasicParams{ { "query", "state", "memory" }, { "attn" } }, numUnits, true, 0.0_dt, 1.7_dt }, networkParameters);
    TENSORS_CREATE(batchSize);

    work.printInfo(std::cout);

    // Checks
    EXPECT_EQ(work.getTrainableParameterNames().size(), goldenTrainableParams);
}

TEST(TestLayerBahdanauMonotonicAttention, ForwardDefaultModeUnit)
{
    PROFILE_TEST
    // See bahdanau_attention.py
    // Test parameters
    const auto eps = TODTYPE(1e-5);
    const size_t numUnits = 3;
    const size_t alignmentsSize = 5;
    const size_t anyNumber = 7;
    const size_t batchSize = 2;
    const Tensor query{ 0.8822692633_dt, 0.9150039554_dt, 0.3828637600_dt, 0.9593056440_dt, 0.3904482126_dt, 0.6008953452_dt };

    const Tensor state{ 0.2565724850_dt, 0.7936413288_dt, 0.9407714605_dt, 0.1331859231_dt, 0.9345980883_dt, 0.5935796499_dt, 0.8694044352_dt, 0.5677152872_dt, 0.7410940528_dt, 0.4294044971_dt };

    const Tensor memory{ 0.8854429126_dt, 0.5739044547_dt, 0.2665800452_dt, 0.6274491549_dt, 0.2696316838_dt, 0.4413635731_dt, 0.2969208360_dt, 0.8316854835_dt, 0.1053149104_dt, 0.2694948316_dt,
                         0.3588126302_dt, 0.1993637681_dt, 0.5471915603_dt, 0.0061604381_dt, 0.9515545368_dt, 0.0752658844_dt, 0.8860136867_dt, 0.5832095742_dt, 0.3376477361_dt, 0.8089749813_dt,
                         0.5779253840_dt, 0.9039816856_dt, 0.5546598434_dt, 0.3423134089_dt, 0.6343418360_dt, 0.3644102812_dt, 0.7104287744_dt, 0.9464110732_dt, 0.7890297771_dt, 0.2814137340_dt,
                         0.7886323333_dt, 0.5894631147_dt, 0.7539175153_dt, 0.1952474713_dt, 0.0050457716_dt, 0.3068197370_dt, 0.1164885759_dt, 0.9102694392_dt, 0.6440156698_dt, 0.7071067691_dt,
                         0.6581305861_dt, 0.4913020134_dt, 0.8913041353_dt, 0.1447432041_dt, 0.5314818621_dt, 0.1587299109_dt, 0.6541759968_dt, 0.3278088570_dt, 0.6532081366_dt, 0.3958292603_dt,
                         0.9146959186_dt, 0.2036490440_dt, 0.2018010020_dt, 0.2017830014_dt, 0.9497213960_dt, 0.6656255593_dt, 0.9811253548_dt, 0.0873618722_dt, 0.0040619373_dt, 0.1088181138_dt,
                         0.1636554599_dt, 0.7025200725_dt, 0.6790379286_dt, 0.9154621959_dt, 0.2417873144_dt, 0.1591441035_dt, 0.7652890682_dt, 0.2978977561_dt, 0.8034619093_dt, 0.3813496828_dt };

    // Mask parameter
    const Tensor memorySeqLength[]{ { 1.0_dt, 1.0_dt }, { 2.0_dt, 4.0_dt }, { 3.0_dt, 2.0_dt } };

    const Tensor realOut[]{
        { 0.2140678316_dt, 0.6273269653_dt, 0.8624919057_dt, 0.3153227270_dt, 0.7799403667_dt, 0.5078871846_dt, 0.7165711522_dt, 0.6048905253_dt, 0.7070785761_dt, 0.4987507761_dt },
        { 0.2140678316_dt, 0.6842455268_dt, 0.8197882175_dt, 0.3046578765_dt, 0.7772769332_dt, 0.5078871846_dt, 0.7979118824_dt, 0.5912402272_dt, 0.7099284530_dt, 0.4458272159_dt },
        { 0.2140678316_dt, 0.6842455268_dt, 0.9112838507_dt, 0.2360123247_dt, 0.7601333857_dt, 0.5078871846_dt, 0.7979118824_dt, 0.5438638330_dt, 0.6918377876_dt, 0.4949445128_dt }
    };
    const auto expectedShape = yato::dims(batchSize, 1, 1, alignmentsSize);

    // In order to reproduce the result
    const Tensor queryLinearLayerWeights{ -0.1448689997_dt, 0.3424179852_dt, 0.5204399824_dt, -0.3655380011_dt, 0.2678830028_dt, 0.3229590058_dt, 0.1139210016_dt, 0.1118329987_dt, -0.3971950114_dt };
    const Tensor memoryLinearLayerWeights{ -0.0409465991_dt, -0.2600440085_dt, -0.3023909926_dt, -0.3340570033_dt, -0.0308049005_dt, 0.2768029869_dt,  -0.1257040054_dt,
                                           0.0764357001_dt,  -0.2699669898_dt, 0.1572880000_dt,  0.1140609980_dt,  -0.3624039888_dt, -0.3353210092_dt, 0.3552179933_dt,
                                           0.1678149998_dt,  0.2513029873_dt,  0.3315150142_dt,  -0.2174510062_dt, -0.3773759902_dt, -0.2405180037_dt, 0.3720769882_dt };
    const Tensor attentionV{ -0.633191_dt, 0.234963_dt, -0.391515_dt };

    Name battnName = "battn";
    for (size_t iter = 0; iter < std::size(memorySeqLength); ++iter)
    {
        // Initialization
        MANAGERS_DEFINE
        NETWORK_PARAMS_DEFINE(networkParameters);

        work.add<DataLayer>("data_query", DataParams{ { "query" }, 1u, 1u, numUnits });
        work.add<DataLayer>("data_state", DataParams{ { "state" }, 1u, 1u, alignmentsSize });
        work.add<DataLayer>("data_memory", DataParams{ { "memory" }, 1u, alignmentsSize, anyNumber });
        work.add<DataLayer>("data_memory_seq_length", DataParams{ { "memorySeqLength" }, 1u, 1u, 1u });
        work.add<TensorLayer>("data_score_mask_value", TensorParams{ { "scoreMaskValue" }, 1u, 1u, 1u, 1u });

        // Apply function
        BahdanauMonotonicAttentionLayer(
            battnName, BahdanauAttentionParams{ { { "query", "state", "memory", "memorySeqLength", "scoreMaskValue" }, { "attn", "values" } }, numUnits, false, 0.0_dt, 1.7_dt }, networkParameters);

        TENSORS_CREATE(batchSize);
        memory_manager["query"] = TORANGE(query);
        memory_manager["state"] = TORANGE(state);
        memory_manager["memory"] = TORANGE(memory);
        memory_manager["memorySeqLength"] = TORANGE(memorySeqLength[iter]);
        memory_manager["scoreMaskValue"][0] = 1.1_dt;

        // In order to reproduce the result
        memory_manager[battnName / "query_layer" / "Weights"] = TORANGE(queryLinearLayerWeights);
        memory_manager[battnName / "memory_layer" / "Weights"] = TORANGE(memoryLinearLayerWeights);
        memory_manager[battnName / "attention_v"] = TORANGE(attentionV);

        ASSERT_NO_THROW(work.forwardPassTraining());

        const auto& output = memory_manager["attn"];
        EXPECT_EQ(expectedShape, output.getShape());

        for (size_t i = 0; i < output.size(); ++i)
        {
            ASSERT_TRUE(tools::expect_near_relative(output[i], realOut[iter][i], eps));
        }
    }
}

TEST(TestLayerBahdanauMonotonicAttention, BackwardDefaultModeUnit)
{
    PROFILE_TEST
    // See bahdanau_attention.py
    // Test parameters
    const auto eps = TODTYPE(1e-3);
    const size_t numUnits = 3;
    const size_t alignmentsSize = 5;
    const size_t anyNumber = 7;
    const size_t batchSize = 2;
    const Tensor query{ 0.8822692633_dt, 0.9150039554_dt, 0.3828637600_dt, 0.9593056440_dt, 0.3904482126_dt, 0.6008953452_dt };

    const Tensor state{ 0.2565724850_dt, 0.7936413288_dt, 0.9407714605_dt, 0.1331859231_dt, 0.9345980883_dt, 0.5935796499_dt, 0.8694044352_dt, 0.5677152872_dt, 0.7410940528_dt, 0.4294044971_dt };

    const Tensor memory{ 0.8854429126_dt, 0.5739044547_dt, 0.2665800452_dt, 0.6274491549_dt, 0.2696316838_dt, 0.4413635731_dt, 0.2969208360_dt, 0.8316854835_dt, 0.1053149104_dt, 0.2694948316_dt,
                         0.3588126302_dt, 0.1993637681_dt, 0.5471915603_dt, 0.0061604381_dt, 0.9515545368_dt, 0.0752658844_dt, 0.8860136867_dt, 0.5832095742_dt, 0.3376477361_dt, 0.8089749813_dt,
                         0.5779253840_dt, 0.9039816856_dt, 0.5546598434_dt, 0.3423134089_dt, 0.6343418360_dt, 0.3644102812_dt, 0.7104287744_dt, 0.9464110732_dt, 0.7890297771_dt, 0.2814137340_dt,
                         0.7886323333_dt, 0.5894631147_dt, 0.7539175153_dt, 0.1952474713_dt, 0.0050457716_dt, 0.3068197370_dt, 0.1164885759_dt, 0.9102694392_dt, 0.6440156698_dt, 0.7071067691_dt,
                         0.6581305861_dt, 0.4913020134_dt, 0.8913041353_dt, 0.1447432041_dt, 0.5314818621_dt, 0.1587299109_dt, 0.6541759968_dt, 0.3278088570_dt, 0.6532081366_dt, 0.3958292603_dt,
                         0.9146959186_dt, 0.2036490440_dt, 0.2018010020_dt, 0.2017830014_dt, 0.9497213960_dt, 0.6656255593_dt, 0.9811253548_dt, 0.0873618722_dt, 0.0040619373_dt, 0.1088181138_dt,
                         0.1636554599_dt, 0.7025200725_dt, 0.6790379286_dt, 0.9154621959_dt, 0.2417873144_dt, 0.1591441035_dt, 0.7652890682_dt, 0.2978977561_dt, 0.8034619093_dt, 0.3813496828_dt };

    // For Mask
    const Tensor memorySeqLength = { 3.0_dt, 2.0_dt };

    const Tensor queryRealGrad{ -0.0003809182_dt, -0.0020890478_dt, -0.0011571154_dt, -8.3754188381e-05_dt, -0.0004378771_dt, -0.0002263010_dt };
    const Tensor stateRealGrad{
        0.9996883869_dt, 0.9981191158_dt, 0.9896460176_dt, 0.9376283288_dt, 0.7502601147_dt, 0.9996299148_dt, 0.9974362850_dt, 0.9844242334_dt, 0.9376257658_dt, 0.7502601147_dt
    };
    const Tensor memoryRealGrad{ -1.2848134929e-06_dt,
                                 4.5413935368e-07_dt,
                                 6.8503318289e-06_dt,
                                 2.1204959921e-05_dt,
                                 5.0911908147e-06_dt,
                                 -1.0806312275e-05_dt,
                                 1.4944680515e-06_dt,
                                 -3.0462137147e-05_dt,
                                 -1.0001345800e-05_dt,
                                 0.0001099166_dt,
                                 0.0003973579_dt,
                                 0.0001046803_dt,
                                 -0.0001896893_dt,
                                 1.4684021153e-05_dt,
                                 -0.0001589781_dt,
                                 0.0001083027_dt,
                                 0.0010229523_dt,
                                 0.0029847207_dt,
                                 0.0006749138_dt,
                                 -0.0015689796_dt,
                                 0.0002623755_dt,
                                 0._dt,
                                 0._dt,
                                 0._dt,
                                 0._dt,
                                 0._dt,
                                 0._dt,
                                 0._dt,
                                 0._dt,
                                 0._dt,
                                 0._dt,
                                 0._dt,
                                 0._dt,
                                 0._dt,
                                 0._dt,
                                 -4.0988957153e-06_dt,
                                 4.1648624460e-07_dt,
                                 1.7735041183e-05_dt,
                                 5.9070152929e-05_dt,
                                 1.5281635569e-05_dt,
                                 -2.8919352189e-05_dt,
                                 2.8568676953e-06_dt,
                                 -4.1273560782e-05_dt,
                                 1.0954358913e-05_dt,
                                 0.0002079968_dt,
                                 0.0006563818_dt,
                                 0.0001604330_dt,
                                 -0.0003312167_dt,
                                 4.2701489292e-05_dt,
                                 0._dt,
                                 0._dt,
                                 0._dt,
                                 0._dt,
                                 0._dt,
                                 0._dt,
                                 0._dt,
                                 0._dt,
                                 0._dt,
                                 0._dt,
                                 0._dt,
                                 0._dt,
                                 0._dt,
                                 0._dt,
                                 0._dt,
                                 0._dt,
                                 0._dt,
                                 0._dt,
                                 0._dt,
                                 0._dt,
                                 0._dt };
    const Tensor realAttentionVNabla{ 0.0003681126_dt, 0.0008723068_dt, 0.0030059866_dt };
    const Tensor realScoreBiasNabla{ 0.0130159315_dt };

    // In order to reproduce the result
    const Tensor queryLinearLayerWeights{ -0.1448689997_dt, 0.3424179852_dt, 0.5204399824_dt, -0.3655380011_dt, 0.2678830028_dt, 0.3229590058_dt, 0.1139210016_dt, 0.1118329987_dt, -0.3971950114_dt };
    const Tensor memoryLinearLayerWeights{ -0.0409465991_dt, -0.2600440085_dt, -0.3023909926_dt, -0.3340570033_dt, -0.0308049005_dt, 0.2768029869_dt,  -0.1257040054_dt,
                                           0.0764357001_dt,  -0.2699669898_dt, 0.1572880000_dt,  0.1140609980_dt,  -0.3624039888_dt, -0.3353210092_dt, 0.3552179933_dt,
                                           0.1678149998_dt,  0.2513029873_dt,  0.3315150142_dt,  -0.2174510062_dt, -0.3773759902_dt, -0.2405180037_dt, 0.3720769882_dt };
    const Tensor attentionV{ -0.633191_dt, 0.234963_dt, -0.391515_dt };

    Name battnName = "battn";

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    work.add<DataLayer>("data_query", DataParams{ { "query" }, 1u, 1u, numUnits });
    work.add<DataLayer>("data_state", DataParams{ { "state" }, 1u, 1u, alignmentsSize });
    work.add<DataLayer>("data_memory", DataParams{ { "memory" }, 1u, alignmentsSize, anyNumber });
    work.add<DataLayer>("data_memory_seq_length", DataParams{ { "memorySeqLength" }, 1u, 1u, 1u });
    work.add<TensorLayer>("data_score_mask_value", TensorParams{ { "scoreMaskValue" }, 1u, 1u, 1u, 1u });

    // Apply function
    BahdanauMonotonicAttentionLayer(
        battnName, BahdanauAttentionParams{ { { "query", "state", "memory", "memorySeqLength", "scoreMaskValue" }, { "attn", "values" } }, numUnits, false, 0.0_dt, 1.7_dt }, networkParameters);

    TENSORS_CREATE(batchSize);
    memory_manager["query"] = TORANGE(query);
    memory_manager["state"] = TORANGE(state);
    memory_manager["memory"] = TORANGE(memory);
    memory_manager["memorySeqLength"] = TORANGE(memorySeqLength);
    memory_manager["scoreMaskValue"][0] = 1.1_dt;

    // In order to reproduce the result
    memory_manager[battnName / "query_layer" / "Weights"] = TORANGE(queryLinearLayerWeights);
    memory_manager[battnName / "memory_layer" / "Weights"] = TORANGE(memoryLinearLayerWeights);
    memory_manager[battnName / "attention_v"] = TORANGE(attentionV);

    memory_manager[Name("attn").grad()] = 1.0_dt;
    memory_manager[Name("values").grad()] = 0.0_dt;

    ASSERT_NO_THROW(work.forwardPassTraining());
    ASSERT_NO_THROW(work.backwardPassTraining());

    const auto queryNabla = memory_manager[Name("query").grad()];
    EXPECT_EQ(queryNabla.size(), queryRealGrad.size());
    for (size_t i = 0; i < queryNabla.size(); ++i)
    {
        ASSERT_TRUE(tools::expect_near_relative(queryNabla[i], queryRealGrad[i], eps) || (queryNabla[i] < eps && queryRealGrad[i] < eps));
    }
    const auto stateNabla = memory_manager[Name("state").grad()];
    EXPECT_EQ(stateNabla.size(), stateRealGrad.size());
    for (size_t i = 0; i < stateNabla.size(); ++i)
    {
        ASSERT_TRUE(tools::expect_near_relative(stateNabla[i], stateRealGrad[i], eps));
    }
    const auto memoryNabla = memory_manager[Name("memory").grad()];
    EXPECT_EQ(memoryNabla.size(), memoryRealGrad.size());
    for (size_t i = 0; i < memoryNabla.size(); ++i)
    {
        ASSERT_TRUE(tools::expect_near_relative(memoryNabla[i], memoryRealGrad[i], eps) || (memoryNabla[i] < eps && memoryRealGrad[i] < eps));
    }
    const auto attentionVNabla = memory_manager[(battnName / "attention_v").grad()];
    EXPECT_EQ(attentionVNabla.size(), realAttentionVNabla.size());
    for (size_t i = 0; i < attentionVNabla.size(); ++i)
    {
        ASSERT_TRUE(tools::expect_near_relative(attentionVNabla[i], realAttentionVNabla[i], eps));
    }

    const auto scoreBiasNabla = memory_manager[(battnName / "score_bias").grad()];
    EXPECT_EQ(scoreBiasNabla.size(), realScoreBiasNabla.size());
    for (size_t i = 0; i < scoreBiasNabla.size(); ++i)
    {
        ASSERT_TRUE(tools::expect_near_relative(scoreBiasNabla[i], realScoreBiasNabla[i], eps));
    }
}

TEST(TestLayerBahdanauMonotonicAttention, ForwardNormalizedModeUnit)
{
    PROFILE_TEST
    // See bahdanau_attention.py
    // Test parameters
    const auto eps = TODTYPE(1e-5);
    const size_t numUnits = 4;
    const size_t queryDepth = 5;
    const size_t alignmentsSize = 6;
    const size_t anyNumber = 7;
    const size_t batchSize = 3;
    const Tensor query{ 0.4962565899_dt, 0.7682217956_dt, 0.0884774327_dt, 0.1320304871_dt, 0.3074228168_dt, 0.6340786815_dt, 0.4900934100_dt, 0.8964447379_dt,
                        0.4556279778_dt, 0.6323062778_dt, 0.3488934636_dt, 0.4017173052_dt, 0.0223257542_dt, 0.1688589454_dt, 0.2938884497_dt };

    const Tensor state{ 0.5185217857_dt, 0.6976675987_dt, 0.8000113964_dt, 0.1610294580_dt, 0.2822685838_dt, 0.6816085577_dt, 0.9151939750_dt, 0.3970999122_dt, 0.8741558790_dt,
                        0.4194083214_dt, 0.5529070497_dt, 0.9527381063_dt, 0.0361648202_dt, 0.1852310300_dt, 0.3734173775_dt, 0.3051000237_dt, 0.9320003986_dt, 0.1759101748_dt };

    const Tensor memory{ 0.2698335648_dt, 0.1506797671_dt, 0.0317195058_dt, 0.2081297636_dt, 0.9297990203_dt, 0.7231091857_dt, 0.7423362732_dt, 0.5262957811_dt, 0.2436582446_dt, 0.5845923424_dt,
                         0.0331526399_dt, 0.1387168765_dt, 0.2422350049_dt, 0.8154689670_dt, 0.7931606174_dt, 0.2782524824_dt, 0.4819588065_dt, 0.8197803497_dt, 0.9970665574_dt, 0.6984410882_dt,
                         0.5675464272_dt, 0.8352431655_dt, 0.2055988312_dt, 0.5931720138_dt, 0.1123472452_dt, 0.1534569263_dt, 0.2417082191_dt, 0.7262365222_dt, 0.7010802031_dt, 0.2038237453_dt,
                         0.6510535479_dt, 0.7744860053_dt, 0.4368913174_dt, 0.5190907717_dt, 0.6158523560_dt, 0.8101882935_dt, 0.9800970554_dt, 0.1146882176_dt, 0.3167651296_dt, 0.6965049505_dt,
                         0.9142746925_dt, 0.9351036549_dt, 0.9411783814_dt, 0.5995072722_dt, 0.0652086735_dt, 0.5459962487_dt, 0.1871973276_dt, 0.0340229273_dt, 0.9442462325_dt, 0.8801798820_dt,
                         0.0012360215_dt, 0.5935860276_dt, 0.4157699943_dt, 0.4177194238_dt, 0.2711215615_dt, 0.6922780871_dt, 0.2038482428_dt, 0.6832956672_dt, 0.7528540492_dt, 0.8579357862_dt,
                         0.6869555712_dt, 0.0051323771_dt, 0.1756515503_dt, 0.7496575117_dt, 0.6046506763_dt, 0.1099579930_dt, 0.2120902538_dt, 0.9703746438_dt, 0.8369089365_dt, 0.2819874287_dt,
                         0.3741576076_dt, 0.0237009525_dt, 0.4910129309_dt, 0.1234705448_dt, 0.1143216491_dt, 0.4724501967_dt, 0.5750725269_dt, 0.2952348590_dt, 0.7966887951_dt, 0.1957304478_dt,
                         0.9536850452_dt, 0.8426499367_dt, 0.0783585310_dt, 0.3755578399_dt, 0.5225613117_dt, 0.5729505420_dt, 0.6185871363_dt, 0.6962141395_dt, 0.5299500823_dt, 0.2560356259_dt,
                         0.7365944982_dt, 0.0203755498_dt, 0.2036466599_dt, 0.3748350739_dt, 0.2564433217_dt, 0.3250833154_dt, 0.0901891589_dt, 0.3936424255_dt, 0.6068782210_dt, 0.1742671132_dt,
                         0.4743403196_dt, 0.8579254150_dt, 0.4485998750_dt, 0.5138961077_dt, 0.4568655491_dt, 0.6011906862_dt, 0.8179197311_dt, 0.9736230969_dt, 0.8175279498_dt, 0.9747067690_dt,
                         0.4638391733_dt, 0.0508392453_dt, 0.2629613876_dt, 0.8404526114_dt, 0.4967587590_dt, 0.2514768243_dt, 0.1168441176_dt, 0.0320739746_dt, 0.0779958963_dt, 0.3985816240_dt,
                         0.7742030025_dt, 0.7703205347_dt, 0.0177840590_dt, 0.8118910193_dt, 0.1087452769_dt, 0.3942948580_dt };

    // Mask parameter
    const Tensor memorySeqLength[]{ { 2, 3, 4 }, { 4, 3, 2 }, { 7, 7, 2 } };
    const Tensor scoreMaskValues[]{ { 1.0_dt }, { 1.0_dt, 2.0_dt, 3.0_dt, 4.0_dt, 5.0_dt, 6.0_dt }, { 10.0_dt, -10.0_dt, 0.013213_dt, 1.0_dt, 1.123_dt, -1.0_dt } };
    const yato::dimensionality<4> scoreShapes[]{ yato::dims(1, 1, 1, 1), yato::dims(1, 1, alignmentsSize, 1), yato::dims(1, 1, alignmentsSize, 1) };

    const Tensor realOut[]{ { 0.4272302091_dt,
                              0.6427391171_dt,
                              0.6917507052_dt,
                              0.3037623167_dt,
                              0.2880491614_dt,
                              0.5757641792_dt,
                              0.7611824274_dt,
                              0.4488304853_dt,
                              0.7797641158_dt,
                              0.4503913522_dt,
                              0.5253363252_dt,
                              0.8377920985_dt,
                              0.0297812112_dt,
                              0.1573416293_dt,
                              0.3391829133_dt,
                              0.3056303263_dt,
                              0.7310422063_dt,
                              0.3252082169_dt },
                            { 0.4272302091_dt,
                              0.6427391171_dt,
                              0.7793446183_dt,
                              0.2695223093_dt,
                              0.3383825719_dt,
                              0.6821975708_dt,
                              0.7611824274_dt,
                              0.4488304853_dt,
                              0.7797641158_dt,
                              0.6050001383_dt,
                              0.5602133870_dt,
                              0.9541477561_dt,
                              0.0297812112_dt,
                              0.1573416293_dt,
                              0.3883553147_dt,
                              0.3185997307_dt,
                              0.9315590262_dt,
                              0.1817364693_dt },
                            { 0.4272302091_dt,
                              0.6427391171_dt,
                              0.7793446183_dt,
                              0.2695223093_dt,
                              0.2784935832_dt,
                              0.6213750839_dt,
                              0.7611824274_dt,
                              0.4488304853_dt,
                              0.7797641158_dt,
                              0.5108545423_dt,
                              0.5322520733_dt,
                              0.8819122910_dt,
                              0.0297812112_dt,
                              0.1573416293_dt,
                              0.2051918805_dt,
                              0.3710842729_dt,
                              0.8062421679_dt,
                              0.1178454757_dt } };

    const auto expectedShape = yato::dims(batchSize, 1, 1, alignmentsSize);

    // In order to reproduce the result
    const Tensor queryLinearLayerWeights{ -0.1122149974_dt, 0.2652359903_dt,  0.4031310081_dt,  -0.2831450105_dt, 0.2075019926_dt,  0.2501629889_dt,  0.0882427990_dt,
                                          0.0866253972_dt,  -0.3076660037_dt, -0.0484487005_dt, -0.3076879978_dt, -0.3577930033_dt, -0.3952620029_dt, -0.0364488997_dt,
                                          0.3275179863_dt,  -0.1487360001_dt, 0.0904399976_dt,  -0.3194299936_dt, 0.1861059964_dt,  0.1349589974_dt };
    const Tensor memoryLinearLayerWeights{ -0.3624039888_dt, -0.3353210092_dt, 0.3552179933_dt,  0.1678149998_dt,  0.2513029873_dt,  0.3315150142_dt,  -0.2174510062_dt,
                                           -0.3773759902_dt, -0.2405180037_dt, 0.3720769882_dt,  -0.2393240035_dt, 0.0888077021_dt,  -0.1479790062_dt, 0.0844018012_dt,
                                           0.0187141001_dt,  -0.3726229966_dt, -0.0514447019_dt, -0.3605310023_dt, -0.1578159928_dt, 0.0187279005_dt,  0.0845528021_dt,
                                           -0.0756980032_dt, -0.2725169957_dt, -0.3426890075_dt, -0.1571239978_dt, 0.3581260145_dt,  -0.1010209993_dt, -0.2020059973_dt };
    const Tensor attentionV{ -0.076089_dt, -0.70909_dt, 0.493939_dt, 0.205051_dt };

    Name battnName = "battn";
    for (size_t iter = 1; iter < 2; ++iter)
    {
        // Initialization
        MANAGERS_DEFINE
        NETWORK_PARAMS_DEFINE(networkParameters);

        work.add<DataLayer>("data_query", DataParams{ { "query" }, 1u, 1u, queryDepth });
        work.add<DataLayer>("data_state", DataParams{ { "state" }, 1u, 1u, alignmentsSize });
        work.add<DataLayer>("data_memory", DataParams{ { "memory" }, 1u, alignmentsSize, anyNumber });
        work.add<DataLayer>("data_memory_seq_length", DataParams{ { "memorySeqLength" }, 1u, 1u, 1u });
        work.add<TensorLayer>("data_score_mask_value", TensorParams{ { "scoreMaskValue" }, scoreShapes[iter][0], scoreShapes[iter][1], scoreShapes[iter][2], scoreShapes[iter][3] });

        // Apply function
        BahdanauMonotonicAttentionLayer(
            battnName, BahdanauAttentionParams{ { { "query", "state", "memory", "memorySeqLength", "scoreMaskValue" }, { "attn", "values" } }, numUnits, true, 0.0_dt, 1.7_dt }, networkParameters);

        TENSORS_CREATE(batchSize);
        memory_manager["query"] = TORANGE(query);
        memory_manager["state"] = TORANGE(state);
        memory_manager["memory"] = TORANGE(memory);
        memory_manager["memorySeqLength"] = TORANGE(memorySeqLength[iter]);
        memory_manager["scoreMaskValue"] = TORANGE(scoreMaskValues[iter]);

        // In order to reproduce the result
        memory_manager[battnName / "query_layer" / "Weights"] = TORANGE(queryLinearLayerWeights);
        memory_manager[battnName / "memory_layer" / "Weights"] = TORANGE(memoryLinearLayerWeights);
        memory_manager[battnName / "attention_v"] = TORANGE(attentionV);

        ASSERT_NO_THROW(work.forwardPassTraining());

        const auto& output = memory_manager["attn"];
        EXPECT_EQ(expectedShape, output.getShape());

        for (size_t i = 0; i < output.size(); ++i)
        {
            ASSERT_TRUE(tools::expect_near_relative(output[i], realOut[iter][i], eps));
        }
    }
}

TEST(TestLayerBahdanauMonotonicAttention, BackwardNormalizedModeUnit)
{
    PROFILE_TEST
    // See bahdanau_attention.py
    // Test parameters
    const auto eps = TODTYPE(1e-2);
    const size_t numUnits = 4;
    const size_t queryDepth = 5;
    const size_t alignmentsSize = 6;
    const size_t anyNumber = 7;
    const size_t batchSize = 3;
    const Tensor query{ 0.4962565899_dt, 0.7682217956_dt, 0.0884774327_dt, 0.1320304871_dt, 0.3074228168_dt, 0.6340786815_dt, 0.4900934100_dt, 0.8964447379_dt,
                        0.4556279778_dt, 0.6323062778_dt, 0.3488934636_dt, 0.4017173052_dt, 0.0223257542_dt, 0.1688589454_dt, 0.2938884497_dt };

    const Tensor state{ 0.5185217857_dt, 0.6976675987_dt, 0.8000113964_dt, 0.1610294580_dt, 0.2822685838_dt, 0.6816085577_dt, 0.9151939750_dt, 0.3970999122_dt, 0.8741558790_dt,
                        0.4194083214_dt, 0.5529070497_dt, 0.9527381063_dt, 0.0361648202_dt, 0.1852310300_dt, 0.3734173775_dt, 0.3051000237_dt, 0.9320003986_dt, 0.1759101748_dt };

    const Tensor memory{ 0.2698335648_dt, 0.1506797671_dt, 0.0317195058_dt, 0.2081297636_dt, 0.9297990203_dt, 0.7231091857_dt, 0.7423362732_dt, 0.5262957811_dt, 0.2436582446_dt, 0.5845923424_dt,
                         0.0331526399_dt, 0.1387168765_dt, 0.2422350049_dt, 0.8154689670_dt, 0.7931606174_dt, 0.2782524824_dt, 0.4819588065_dt, 0.8197803497_dt, 0.9970665574_dt, 0.6984410882_dt,
                         0.5675464272_dt, 0.8352431655_dt, 0.2055988312_dt, 0.5931720138_dt, 0.1123472452_dt, 0.1534569263_dt, 0.2417082191_dt, 0.7262365222_dt, 0.7010802031_dt, 0.2038237453_dt,
                         0.6510535479_dt, 0.7744860053_dt, 0.4368913174_dt, 0.5190907717_dt, 0.6158523560_dt, 0.8101882935_dt, 0.9800970554_dt, 0.1146882176_dt, 0.3167651296_dt, 0.6965049505_dt,
                         0.9142746925_dt, 0.9351036549_dt, 0.9411783814_dt, 0.5995072722_dt, 0.0652086735_dt, 0.5459962487_dt, 0.1871973276_dt, 0.0340229273_dt, 0.9442462325_dt, 0.8801798820_dt,
                         0.0012360215_dt, 0.5935860276_dt, 0.4157699943_dt, 0.4177194238_dt, 0.2711215615_dt, 0.6922780871_dt, 0.2038482428_dt, 0.6832956672_dt, 0.7528540492_dt, 0.8579357862_dt,
                         0.6869555712_dt, 0.0051323771_dt, 0.1756515503_dt, 0.7496575117_dt, 0.6046506763_dt, 0.1099579930_dt, 0.2120902538_dt, 0.9703746438_dt, 0.8369089365_dt, 0.2819874287_dt,
                         0.3741576076_dt, 0.0237009525_dt, 0.4910129309_dt, 0.1234705448_dt, 0.1143216491_dt, 0.4724501967_dt, 0.5750725269_dt, 0.2952348590_dt, 0.7966887951_dt, 0.1957304478_dt,
                         0.9536850452_dt, 0.8426499367_dt, 0.0783585310_dt, 0.3755578399_dt, 0.5225613117_dt, 0.5729505420_dt, 0.6185871363_dt, 0.6962141395_dt, 0.5299500823_dt, 0.2560356259_dt,
                         0.7365944982_dt, 0.0203755498_dt, 0.2036466599_dt, 0.3748350739_dt, 0.2564433217_dt, 0.3250833154_dt, 0.0901891589_dt, 0.3936424255_dt, 0.6068782210_dt, 0.1742671132_dt,
                         0.4743403196_dt, 0.8579254150_dt, 0.4485998750_dt, 0.5138961077_dt, 0.4568655491_dt, 0.6011906862_dt, 0.8179197311_dt, 0.9736230969_dt, 0.8175279498_dt, 0.9747067690_dt,
                         0.4638391733_dt, 0.0508392453_dt, 0.2629613876_dt, 0.8404526114_dt, 0.4967587590_dt, 0.2514768243_dt, 0.1168441176_dt, 0.0320739746_dt, 0.0779958963_dt, 0.3985816240_dt,
                         0.7742030025_dt, 0.7703205347_dt, 0.0177840590_dt, 0.8118910193_dt, 0.1087452769_dt, 0.3942948580_dt };

    // Mask
    const Tensor memorySeqLength{ 4.0_dt, 3.0_dt, 2.0_dt };

    const Tensor queryRealGrad{ -0.0008168381_dt, -0.0005155340_dt, -0.0007735305_dt,     0.0006895980_dt,      0.0004305975_dt,      -0.0003753756_dt,    -0.0001851519_dt,   -0.0003009100_dt,
                                0.0003927155_dt,  0.0001497775_dt,  -1.9470287953e-05_dt, -1.2845145648e-05_dt, -1.9195687855e-05_dt, 1.5204145711e-05_dt, 1.1072296729e-05_dt };
    const Tensor stateRealGrad{ 0.9999361038_dt, 0.9996370077_dt, 0.9980420470_dt, 0.9889017940_dt, 0.9376558661_dt, 0.7502601147_dt, 0.9999020100_dt, 0.9994177818_dt, 0.9968632460_dt,
                                0.9844186902_dt, 0.9376075268_dt, 0.7502601743_dt, 0.9998771548_dt, 0.9993041158_dt, 0.9961088300_dt, 0.9844209552_dt, 0.9376264811_dt, 0.7502601147_dt };
    const Tensor memoryRealGrad{ 5.5376531236e-06_dt,
                                 6.9251200330e-09_dt,
                                 -7.3705064096e-06_dt,
                                 -9.2431037046e-08_dt,
                                 -1.2425049363e-06_dt,
                                 1.3704108142e-06_dt,
                                 -1.1076406281e-06_dt,
                                 3.8405527448e-05_dt,
                                 -1.0113180906e-06_dt,
                                 -4.9431029765e-05_dt,
                                 -2.8877147997e-06_dt,
                                 -1.1275419638e-05_dt,
                                 9.1915426310e-06_dt,
                                 -5.7464162637e-06_dt,
                                 0.0002307519_dt,
                                 3.2046416891e-05_dt,
                                 -0.0003045338_dt,
                                 2.7337911888e-05_dt,
                                 -3.8790916733e-05_dt,
                                 5.6512184528e-05_dt,
                                 -5.3834337450e-05_dt,
                                 0.0004872462_dt,
                                 -2.2791791707e-06_dt,
                                 -0.0006235492_dt,
                                 -2.6386958780e-05_dt,
                                 -0.0001413992_dt,
                                 0.0001175316_dt,
                                 -7.4017174484e-05_dt,
                                 0._dt,
                                 0._dt,
                                 0._dt,
                                 0._dt,
                                 0._dt,
                                 0._dt,
                                 0._dt,
                                 0._dt,
                                 0._dt,
                                 0._dt,
                                 0._dt,
                                 0._dt,
                                 0._dt,
                                 0._dt,
                                 8.8209790192e-06_dt,
                                 1.4536826711e-06_dt,
                                 -1.1051546608e-05_dt,
                                 7.8021366789e-07_dt,
                                 -2.1497403395e-06_dt,
                                 1.9545670966e-06_dt,
                                 -1.5062261127e-06_dt,
                                 3.9843456761e-05_dt,
                                 4.4484568207e-06_dt,
                                 -4.9985828809e-05_dt,
                                 2.9281600291e-06_dt,
                                 -1.0040303096e-05_dt,
                                 1.0306517652e-05_dt,
                                 -7.2083112173e-06_dt,
                                 0.0003780332_dt,
                                 0.0001253704_dt,
                                 -0.0004666141_dt,
                                 0.0001162831_dt,
                                 -5.9158242948e-05_dt,
                                 0.0001030223_dt,
                                 -9.2531699920e-05_dt,
                                 0._dt,
                                 0._dt,
                                 0._dt,
                                 0._dt,
                                 0._dt,
                                 0._dt,
                                 0._dt,
                                 0._dt,
                                 0._dt,
                                 0._dt,
                                 0._dt,
                                 0._dt,
                                 0._dt,
                                 0._dt,
                                 0._dt,
                                 0._dt,
                                 0._dt,
                                 0._dt,
                                 0._dt,
                                 0._dt,
                                 0._dt,
                                 -2.0451706106e-08_dt,
                                 -1.8770993648e-09_dt,
                                 2.5994911113e-08_dt,
                                 -7.6339734534e-10_dt,
                                 5.1734425632e-09_dt,
                                 -4.8417865273e-09_dt,
                                 3.5318492575e-09_dt,
                                 1.6743522792e-05_dt,
                                 -1.3715134628e-06_dt,
                                 -2.2303716833e-05_dt,
                                 -1.9069275368e-06_dt,
                                 -4.5294887059e-06_dt,
                                 3.9687711251e-06_dt,
                                 -2.7945375223e-06_dt,
                                 0._dt,
                                 0._dt,
                                 0._dt,
                                 0._dt,
                                 0._dt,
                                 0._dt,
                                 0._dt,
                                 0._dt,
                                 0._dt,
                                 0._dt,
                                 0._dt,
                                 0._dt,
                                 0._dt,
                                 0._dt,
                                 0._dt,
                                 0._dt,
                                 0._dt,
                                 0._dt,
                                 0._dt,
                                 0._dt,
                                 0._dt,
                                 0._dt,
                                 0._dt,
                                 0._dt,
                                 0._dt,
                                 0._dt,
                                 0._dt,
                                 0.0_dt };

    const Tensor realAttentionVNabla{ 0.0011630179_dt, -0.0016222397_dt, -0.0016261924_dt, -0.0012610571_dt };
    const Tensor realAttentionBNabla{ -0.0002754711_dt, -0.0030156649_dt, 0.0012318231_dt, 0.0007339923_dt };
    const Tensor realAttentionGNabla{ -0.0033586258_dt };
    const Tensor realScoreBiasNabla{ 0.0076629906_dt };

    // In order to reproduce the result
    const Tensor queryLinearLayerWeights{ -0.1122149974_dt, 0.2652359903_dt,  0.4031310081_dt,  -0.2831450105_dt, 0.2075019926_dt,  0.2501629889_dt,  0.0882427990_dt,
                                          0.0866253972_dt,  -0.3076660037_dt, -0.0484487005_dt, -0.3076879978_dt, -0.3577930033_dt, -0.3952620029_dt, -0.0364488997_dt,
                                          0.3275179863_dt,  -0.1487360001_dt, 0.0904399976_dt,  -0.3194299936_dt, 0.1861059964_dt,  0.1349589974_dt };
    const Tensor memoryLinearLayerWeights{ -0.3624039888_dt, -0.3353210092_dt, 0.3552179933_dt,  0.1678149998_dt,  0.2513029873_dt,  0.3315150142_dt,  -0.2174510062_dt,
                                           -0.3773759902_dt, -0.2405180037_dt, 0.3720769882_dt,  -0.2393240035_dt, 0.0888077021_dt,  -0.1479790062_dt, 0.0844018012_dt,
                                           0.0187141001_dt,  -0.3726229966_dt, -0.0514447019_dt, -0.3605310023_dt, -0.1578159928_dt, 0.0187279005_dt,  0.0845528021_dt,
                                           -0.0756980032_dt, -0.2725169957_dt, -0.3426890075_dt, -0.1571239978_dt, 0.3581260145_dt,  -0.1010209993_dt, -0.2020059973_dt };
    const Tensor attentionV{ -0.076089_dt, -0.70909_dt, 0.493939_dt, 0.205051_dt };

    Name battnName = "battn";

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    work.add<DataLayer>("data_query", DataParams{ { "query" }, 1u, 1u, queryDepth });
    work.add<DataLayer>("data_state", DataParams{ { "state" }, 1u, 1u, alignmentsSize });
    work.add<DataLayer>("data_memory", DataParams{ { "memory" }, 1u, alignmentsSize, anyNumber });
    work.add<DataLayer>("data_memory_seq_length", DataParams{ { "memorySeqLength" }, 1u, 1u, 1u });
    work.add<TensorLayer>("data_score_mask_value", TensorParams{ { "scoreMaskValue" }, 1u, 1u, 1u, 1u });

    // Apply function
    BahdanauMonotonicAttentionLayer(
        battnName, BahdanauAttentionParams{ { { "query", "state", "memory", "memorySeqLength", "scoreMaskValue" }, { "attn", "values" } }, numUnits, true, 0.0_dt, 1.7_dt }, networkParameters);

    TENSORS_CREATE(batchSize);
    memory_manager["query"] = TORANGE(query);
    memory_manager["state"] = TORANGE(state);
    memory_manager["memory"] = TORANGE(memory);
    memory_manager["memorySeqLength"] = TORANGE(memorySeqLength);
    memory_manager["scoreMaskValue"][0] = 1.1_dt;

    // In order to reproduce the result
    memory_manager[battnName / "query_layer" / "Weights"] = TORANGE(queryLinearLayerWeights);
    memory_manager[battnName / "memory_layer" / "Weights"] = TORANGE(memoryLinearLayerWeights);
    memory_manager[battnName / "attention_v"] = TORANGE(attentionV);

    memory_manager[Name("attn").grad()] = 1.0_dt;

    ASSERT_NO_THROW(work.forwardPassTraining());
    ASSERT_NO_THROW(work.backwardPassTraining());

    const auto queryNabla = memory_manager[Name("query").grad()];
    EXPECT_EQ(queryNabla.size(), queryRealGrad.size());
    for (size_t i = 0; i < queryNabla.size(); ++i)
    {
        ASSERT_TRUE(tools::expect_near_relative(queryNabla[i], queryRealGrad[i], eps) || (queryNabla[i] < eps && queryRealGrad[i] < eps));
    }
    const auto stateNabla = memory_manager[Name("state").grad()];
    EXPECT_EQ(stateNabla.size(), stateRealGrad.size());
    for (size_t i = 0; i < stateNabla.size(); ++i)
    {
        ASSERT_TRUE(tools::expect_near_relative(stateNabla[i], stateRealGrad[i], eps));
    }
    const auto memoryNabla = memory_manager[Name("memory").grad()];
    EXPECT_EQ(memoryNabla.size(), memoryRealGrad.size());
    for (size_t i = 0; i < memoryNabla.size(); ++i)
    {
        ASSERT_TRUE(tools::expect_near_relative(memoryNabla[i], memoryRealGrad[i], eps) || (memoryNabla[i] < eps && memoryRealGrad[i] < eps));
    }
    const auto attentionVNabla = memory_manager[(battnName / "attention_v").grad()];
    EXPECT_EQ(attentionVNabla.size(), realAttentionVNabla.size());
    for (size_t i = 0; i < realAttentionVNabla.size(); ++i)
    {
        ASSERT_TRUE(tools::expect_near_relative(attentionVNabla[i], realAttentionVNabla[i], eps));
    }
    const auto attentionBNabla = memory_manager[(battnName / "attention_b").grad()];
    EXPECT_EQ(attentionBNabla.size(), realAttentionBNabla.size());
    for (size_t i = 0; i < attentionBNabla.size(); ++i)
    {
        ASSERT_TRUE(tools::expect_near_relative(attentionBNabla[i], realAttentionBNabla[i], eps));
    }
    const auto attentionGNabla = memory_manager[(battnName / "attention_g").grad()];
    EXPECT_EQ(attentionGNabla.size(), realAttentionGNabla.size());
    for (size_t i = 0; i < attentionGNabla.size(); ++i)
    {
        ASSERT_TRUE(tools::expect_near_relative(attentionGNabla[i], realAttentionGNabla[i], eps));
    }
    const auto scoreBiasNabla = memory_manager[(battnName / "score_bias").grad()];
    EXPECT_EQ(scoreBiasNabla.size(), realScoreBiasNabla.size());
    for (size_t i = 0; i < scoreBiasNabla.size(); ++i)
    {
        ASSERT_TRUE(tools::expect_near_relative(scoreBiasNabla[i], realScoreBiasNabla[i], eps));
    }
}

TEST(TestLayerStepwiseBahdanauMonotonicAttention, DefaultModeUnit)
{
    PROFILE_TEST
    // See bahdanau_attention.py
    // Test parameters
    const auto eps = TODTYPE(1e-5);
    const size_t numUnits = 4;
    const size_t queryDepth = 5;
    const size_t alignmentsSize = 6;
    const size_t anyNumber = 7;
    const size_t batchSize = 3;
    const Tensor query{ 0.8995079_dt, 0.4925307_dt,  0.36408758_dt, 0.01972544_dt, 0.7798331_dt,  0.77982223_dt, 0.81948376_dt, 0.29104686_dt,
                        0.1190486_dt, 0.56751144_dt, 0.15556598_dt, 0.4869895_dt,  0.26108038_dt, 0.28730178_dt, 0.71505356_dt };

    const Tensor state{ 0.7241353_dt, 0.6742269_dt,  0.44395304_dt, 0.09891009_dt, 0.95322967_dt, 0.4661187_dt,  0.588459_dt,   0.3044685_dt, 0.03514242_dt,
                        0.2526785_dt, 0.81228757_dt, 0.77946687_dt, 0.9784529_dt,  0.07848763_dt, 0.80135584_dt, 0.20913935_dt, 0.6607659_dt, 0.21036232_dt };

    const Tensor memory{ 0.01975703_dt, 0.00704217_dt, 0.18987215_dt, 0.7772658_dt,  0.41817415_dt, 0.7437942_dt,  0.26365364_dt, 0.4459244_dt,  0.82929873_dt, 0.52497685_dt, 0.55597556_dt,
                         0.19923508_dt, 0.46925998_dt, 0.18594062_dt, 0.23303056_dt, 0.3938471_dt,  0.9660922_dt,  0.36530995_dt, 0.28173566_dt, 0.4888971_dt,  0.96301997_dt, 0.45836866_dt,
                         0.70952535_dt, 0.477888_dt,   0.71620464_dt, 0.12221897_dt, 0.2998824_dt,  0.6689563_dt,  0.06436884_dt, 0.23358119_dt, 0.8235085_dt,  0.24635303_dt, 0.87422705_dt,
                         0.97360873_dt, 0.5011089_dt,  0.4178022_dt,  0.19041097_dt, 0.05045938_dt, 0.07118928_dt, 0.17497218_dt, 0.06644797_dt, 0.7329292_dt,  0.8574884_dt,  0.4593867_dt,
                         0.28661895_dt, 0.7181833_dt,  0.30093706_dt, 0.02433372_dt, 0.42253482_dt, 0.06825948_dt, 0.48981392_dt, 0.92883205_dt, 0.9339298_dt,  0.41831005_dt, 0.8322693_dt,
                         0.22140837_dt, 0.23945987_dt, 0.7574657_dt,  0.5762696_dt,  0.5139812_dt,  0.7258351_dt,  0.86447895_dt, 0.9819726_dt,  0.24162543_dt, 0.24936235_dt, 0.72023165_dt,
                         0.3312081_dt,  0.40411353_dt, 0.59419465_dt, 0.71123624_dt, 0.8676628_dt,  0.8858366_dt,  0.82439685_dt, 0.43707013_dt, 0.92378604_dt, 0.00537562_dt, 0.63191164_dt,
                         0.5659201_dt,  0.12591887_dt, 0.5189445_dt,  0.80667794_dt, 0.34214568_dt, 0.34712052_dt, 0.5230378_dt,  0.02033377_dt, 0.9925318_dt,  0.04908013_dt, 0.5698966_dt,
                         0.4791932_dt,  0.221825_dt,   0.39972973_dt, 0.09565127_dt, 0.07026207_dt, 0.7138928_dt,  0.21078682_dt, 0.8794396_dt,  0.5082735_dt,  0.8915067_dt,  0.13851714_dt,
                         0.06712937_dt, 0.24958026_dt, 0.10923862_dt, 0.6606549_dt,  0.7950859_dt,  0.5450705_dt,  0.4209025_dt,  0.585426_dt,   0.63537335_dt, 0.40576637_dt, 0.5183171_dt,
                         0.58145976_dt, 0.7846494_dt,  0.6629163_dt,  0.77547586_dt, 0.75580096_dt, 0.2184534_dt,  0.25045693_dt, 0.22379267_dt, 0.62836266_dt, 0.10235023_dt, 0.74957764_dt,
                         0.6434492_dt,  0.6539769_dt,  0.11029541_dt, 0.10112023_dt, 0.23958611_dt };

    // Mask parameter
    const Tensor memorySeqLength{ 1.0_dt, 2.0_dt, 3.0_dt };
    const Tensor scoreMaskValues{ 1.1_dt };

    const Tensor realOut{ 0.69977564_dt, 0.5302052_dt, 0.5014616_dt, 0.18508108_dt, 0.739872_dt,   0.70417815_dt, 0.5675052_dt, 0.3154183_dt,  0.03637001_dt,
                          0.19835109_dt, 0.6725309_dt, 0.9823275_dt, 0.9505889_dt,  0.10431705_dt, 0.77667105_dt, 0.1836284_dt, 0.54797673_dt, 0.37538192_dt };
    const Tensor realIndices{ 4.0_dt, 5.0_dt, 0.0_dt };
    const auto expectedShape = yato::dims(batchSize, 1, 1, alignmentsSize);

    // In order to reproduce the result
    const Tensor queryLinearLayerWeights{ 0.09049082_dt, 0.2369746_dt,  -0.04944408_dt, -0.813432_dt,  -0.47131312_dt, -0.45512667_dt, 0.04958135_dt, -0.18497097_dt, 0.5842254_dt, 0.26539183_dt,
                                          0.59591985_dt, -0.7929145_dt, 0.63058674_dt,  -0.6403303_dt, -0.61891955_dt, 0.45280218_dt,  0.6099379_dt,  0.2233758_dt,   0.7512772_dt, -0.57287085_dt };
    const Tensor memoryLinearLayerWeights{ -0.3072731_dt,  -0.12307996_dt, 0.29059702_dt, 0.10673004_dt, 0.61124223_dt, -0.3253169_dt,  0.26568073_dt, -0.43343008_dt, 0.45469207_dt, -0.55335987_dt,
                                           -0.19977236_dt, 0.23876876_dt,  -0.7156197_dt, 0.04834241_dt, 0.0522756_dt,  -0.0100072_dt,  0.30991977_dt, -0.11740226_dt, 0.49257308_dt, 0.33437592_dt,
                                           0.37889642_dt,  0.09048331_dt,  0.73578566_dt, 0.23990375_dt, 0.1921069_dt,  -0.61453474_dt, 0.39222664_dt, -0.6685021_dt };
    const Tensor attentionV{ 0.36473823_dt, 0.41261613_dt, 0.36806035_dt, 0.03106666_dt };

    Name battnName = "battn";

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    work.add<DataLayer>("data_query", DataParams{ { "query" }, 1u, 1u, queryDepth });
    work.add<DataLayer>("data_state", DataParams{ { "state" }, 1u, 1u, alignmentsSize });
    work.add<DataLayer>("data_memory", DataParams{ { "memory" }, 1u, alignmentsSize, anyNumber });
    work.add<DataLayer>("data_memory_seq_length", DataParams{ { "memorySeqLength" }, 1u, 1u, 1u });
    work.add<TensorLayer>("data_score_mask_value", TensorParams{ { "scoreMaskValue" }, 1u, 1u, 1u, 1u });

    // Apply function
    BahdanauMonotonicAttentionLayer(
        battnName,
        BahdanauAttentionParams{ { { "query", "state", "memory", "memorySeqLength", "scoreMaskValue" }, { "attn", "values", "max" } }, numUnits, false, 0.0_dt, 3.5_dt, "parallel", true, false },
        networkParameters);

    TENSORS_CREATE(batchSize);
    memory_manager["query"] = TORANGE(query);
    memory_manager["state"] = TORANGE(state);
    memory_manager["memory"] = TORANGE(memory);
    memory_manager["memorySeqLength"] = TORANGE(memorySeqLength);
    memory_manager["scoreMaskValue"] = TORANGE(scoreMaskValues);

    // In order to reproduce the result
    memory_manager[battnName / "query_layer" / "Weights"] = TORANGE(queryLinearLayerWeights);
    memory_manager[battnName / "memory_layer" / "Weights"] = TORANGE(memoryLinearLayerWeights);
    memory_manager[battnName / "attention_v"] = TORANGE(attentionV);

    ASSERT_NO_THROW(work.forwardPassTraining());

    const auto& output = memory_manager["attn"];
    const auto& indices = memory_manager["max"];
    EXPECT_EQ(expectedShape, output.getShape());
    for (size_t i = 0; i < output.size(); ++i)
    {
        ASSERT_TRUE(tools::expect_near_relative(output[i], realOut[i], eps));
    }
    for (size_t i = 0; i < indices.size(); ++i)
    {
        EXPECT_EQ(indices[i], realIndices[i]);
    }

    ASSERT_NO_THROW(work.backwardPassTraining());
}

TEST(TestLayerStepwiseBahdanauMonotonicAttention, NormalizedModeUnit)
{
    PROFILE_TEST
    // See bahdanau_attention.py
    // Test parameters
    const auto eps = TODTYPE(1e-5);
    const size_t numUnits = 4;
    const size_t queryDepth = 5;
    const size_t alignmentsSize = 6;
    const size_t anyNumber = 7;
    const size_t batchSize = 3;
    const Tensor query{ 0.8995079_dt, 0.4925307_dt,  0.36408758_dt, 0.01972544_dt, 0.7798331_dt,  0.77982223_dt, 0.81948376_dt, 0.29104686_dt,
                        0.1190486_dt, 0.56751144_dt, 0.15556598_dt, 0.4869895_dt,  0.26108038_dt, 0.28730178_dt, 0.71505356_dt };

    const Tensor state{ 0.7241353_dt, 0.6742269_dt,  0.44395304_dt, 0.09891009_dt, 0.95322967_dt, 0.4661187_dt,  0.588459_dt,   0.3044685_dt, 0.03514242_dt,
                        0.2526785_dt, 0.81228757_dt, 0.77946687_dt, 0.9784529_dt,  0.07848763_dt, 0.80135584_dt, 0.20913935_dt, 0.6607659_dt, 0.21036232_dt };

    const Tensor memory{ 0.01975703_dt, 0.00704217_dt, 0.18987215_dt, 0.7772658_dt,  0.41817415_dt, 0.7437942_dt,  0.26365364_dt, 0.4459244_dt,  0.82929873_dt, 0.52497685_dt, 0.55597556_dt,
                         0.19923508_dt, 0.46925998_dt, 0.18594062_dt, 0.23303056_dt, 0.3938471_dt,  0.9660922_dt,  0.36530995_dt, 0.28173566_dt, 0.4888971_dt,  0.96301997_dt, 0.45836866_dt,
                         0.70952535_dt, 0.477888_dt,   0.71620464_dt, 0.12221897_dt, 0.2998824_dt,  0.6689563_dt,  0.06436884_dt, 0.23358119_dt, 0.8235085_dt,  0.24635303_dt, 0.87422705_dt,
                         0.97360873_dt, 0.5011089_dt,  0.4178022_dt,  0.19041097_dt, 0.05045938_dt, 0.07118928_dt, 0.17497218_dt, 0.06644797_dt, 0.7329292_dt,  0.8574884_dt,  0.4593867_dt,
                         0.28661895_dt, 0.7181833_dt,  0.30093706_dt, 0.02433372_dt, 0.42253482_dt, 0.06825948_dt, 0.48981392_dt, 0.92883205_dt, 0.9339298_dt,  0.41831005_dt, 0.8322693_dt,
                         0.22140837_dt, 0.23945987_dt, 0.7574657_dt,  0.5762696_dt,  0.5139812_dt,  0.7258351_dt,  0.86447895_dt, 0.9819726_dt,  0.24162543_dt, 0.24936235_dt, 0.72023165_dt,
                         0.3312081_dt,  0.40411353_dt, 0.59419465_dt, 0.71123624_dt, 0.8676628_dt,  0.8858366_dt,  0.82439685_dt, 0.43707013_dt, 0.92378604_dt, 0.00537562_dt, 0.63191164_dt,
                         0.5659201_dt,  0.12591887_dt, 0.5189445_dt,  0.80667794_dt, 0.34214568_dt, 0.34712052_dt, 0.5230378_dt,  0.02033377_dt, 0.9925318_dt,  0.04908013_dt, 0.5698966_dt,
                         0.4791932_dt,  0.221825_dt,   0.39972973_dt, 0.09565127_dt, 0.07026207_dt, 0.7138928_dt,  0.21078682_dt, 0.8794396_dt,  0.5082735_dt,  0.8915067_dt,  0.13851714_dt,
                         0.06712937_dt, 0.24958026_dt, 0.10923862_dt, 0.6606549_dt,  0.7950859_dt,  0.5450705_dt,  0.4209025_dt,  0.585426_dt,   0.63537335_dt, 0.40576637_dt, 0.5183171_dt,
                         0.58145976_dt, 0.7846494_dt,  0.6629163_dt,  0.77547586_dt, 0.75580096_dt, 0.2184534_dt,  0.25045693_dt, 0.22379267_dt, 0.62836266_dt, 0.10235023_dt, 0.74957764_dt,
                         0.6434492_dt,  0.6539769_dt,  0.11029541_dt, 0.10112023_dt, 0.23958611_dt };

    // Mask parameter
    const Tensor memorySeqLength{ 1.0_dt, 2.0_dt, 3.0_dt };
    const Tensor scoreMaskValues{ 1.1_dt };

    const Tensor realOut{ 0.70058554_dt, 0.5293953_dt, 0.5014616_dt, 0.18508108_dt, 0.739872_dt,  0.70417815_dt, 0.568482_dt,   0.31471816_dt, 0.03609333_dt,
                          0.19835109_dt, 0.6725309_dt, 0.9823275_dt, 0.9503901_dt,  0.1044533_dt, 0.77756566_dt, 0.18279624_dt, 0.54797673_dt, 0.37538192_dt };
    const Tensor realIndices{ 4.0_dt, 5.0_dt, 0.0_dt };
    const auto expectedShape = yato::dims(batchSize, 1, 1, alignmentsSize);

    // In order to reproduce the result
    const Tensor queryLinearLayerWeights{ 0.09049082_dt, 0.2369746_dt,  -0.04944408_dt, -0.813432_dt,  -0.47131312_dt, -0.45512667_dt, 0.04958135_dt, -0.18497097_dt, 0.5842254_dt, 0.26539183_dt,
                                          0.59591985_dt, -0.7929145_dt, 0.63058674_dt,  -0.6403303_dt, -0.61891955_dt, 0.45280218_dt,  0.6099379_dt,  0.2233758_dt,   0.7512772_dt, -0.57287085_dt };
    const Tensor memoryLinearLayerWeights{ -0.3072731_dt,  -0.12307996_dt, 0.29059702_dt, 0.10673004_dt, 0.61124223_dt, -0.3253169_dt,  0.26568073_dt, -0.43343008_dt, 0.45469207_dt, -0.55335987_dt,
                                           -0.19977236_dt, 0.23876876_dt,  -0.7156197_dt, 0.04834241_dt, 0.0522756_dt,  -0.0100072_dt,  0.30991977_dt, -0.11740226_dt, 0.49257308_dt, 0.33437592_dt,
                                           0.37889642_dt,  0.09048331_dt,  0.73578566_dt, 0.23990375_dt, 0.1921069_dt,  -0.61453474_dt, 0.39222664_dt, -0.6685021_dt };
    const Tensor attentionV{ 0.36473823_dt, 0.41261613_dt, 0.36806035_dt, 0.03106666_dt };

    Name battnName = "battn";

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    work.add<DataLayer>("data_query", DataParams{ { "query" }, 1u, 1u, queryDepth });
    work.add<DataLayer>("data_state", DataParams{ { "state" }, 1u, 1u, alignmentsSize });
    work.add<DataLayer>("data_memory", DataParams{ { "memory" }, 1u, alignmentsSize, anyNumber });
    work.add<DataLayer>("data_memory_seq_length", DataParams{ { "memorySeqLength" }, 1u, 1u, 1u });
    work.add<TensorLayer>("data_score_mask_value", TensorParams{ { "scoreMaskValue" }, 1u, 1u, 1u, 1u });

    // Apply function
    BahdanauMonotonicAttentionLayer(
        battnName,
        BahdanauAttentionParams{ { { "query", "state", "memory", "memorySeqLength", "scoreMaskValue" }, { "attn", "values", "max" } }, numUnits, true, 0.0_dt, 3.5_dt, "parallel", true, false },
        networkParameters);

    TENSORS_CREATE(batchSize);
    memory_manager["query"] = TORANGE(query);
    memory_manager["state"] = TORANGE(state);
    memory_manager["memory"] = TORANGE(memory);
    memory_manager["memorySeqLength"] = TORANGE(memorySeqLength);
    memory_manager["scoreMaskValue"] = TORANGE(scoreMaskValues);

    // In order to reproduce the result
    memory_manager[battnName / "query_layer" / "Weights"] = TORANGE(queryLinearLayerWeights);
    memory_manager[battnName / "memory_layer" / "Weights"] = TORANGE(memoryLinearLayerWeights);
    memory_manager[battnName / "attention_v"] = TORANGE(attentionV);

    ASSERT_NO_THROW(work.forwardPassTraining());

    const auto& output = memory_manager["attn"];
    const auto& indices = memory_manager["max"];
    EXPECT_EQ(expectedShape, output.getShape());
    for (size_t i = 0; i < output.size(); ++i)
    {
        ASSERT_TRUE(tools::expect_near_relative(output[i], realOut[i], eps));
    }
    for (size_t i = 0; i < indices.size(); ++i)
    {
        EXPECT_EQ(indices[i], realIndices[i]);
    }

    ASSERT_NO_THROW(work.backwardPassTraining());
}

TEST(TestLayerStepwiseBahdanauMonotonicAttention, OldForwardDefaultModeUnit)
{
    PROFILE_TEST
    // See bahdanau_attention.py
    // Test parameters
    const auto eps = TODTYPE(1e-5);
    const size_t numUnits = 4;
    const size_t queryDepth = 5;
    const size_t alignmentsSize = 6;
    const size_t anyNumber = 7;
    const size_t batchSize = 3;
    const Tensor query{ 0.4962565899_dt, 0.7682217956_dt, 0.0884774327_dt, 0.1320304871_dt, 0.3074228168_dt, 0.6340786815_dt, 0.4900934100_dt, 0.8964447379_dt,
                        0.4556279778_dt, 0.6323062778_dt, 0.3488934636_dt, 0.4017173052_dt, 0.0223257542_dt, 0.1688589454_dt, 0.2938884497_dt };

    const Tensor state{ 0.5185217857_dt, 0.6976675987_dt, 0.8000113964_dt, 0.1610294580_dt, 0.2822685838_dt, 0.6816085577_dt, 0.9151939750_dt, 0.3970999122_dt, 0.8741558790_dt,
                        0.4194083214_dt, 0.5529070497_dt, 0.9527381063_dt, 0.0361648202_dt, 0.1852310300_dt, 0.3734173775_dt, 0.3051000237_dt, 0.9320003986_dt, 0.1759101748_dt };

    const Tensor memory{ 0.2698335648_dt, 0.1506797671_dt, 0.0317195058_dt, 0.2081297636_dt, 0.9297990203_dt, 0.7231091857_dt, 0.7423362732_dt, 0.5262957811_dt, 0.2436582446_dt, 0.5845923424_dt,
                         0.0331526399_dt, 0.1387168765_dt, 0.2422350049_dt, 0.8154689670_dt, 0.7931606174_dt, 0.2782524824_dt, 0.4819588065_dt, 0.8197803497_dt, 0.9970665574_dt, 0.6984410882_dt,
                         0.5675464272_dt, 0.8352431655_dt, 0.2055988312_dt, 0.5931720138_dt, 0.1123472452_dt, 0.1534569263_dt, 0.2417082191_dt, 0.7262365222_dt, 0.7010802031_dt, 0.2038237453_dt,
                         0.6510535479_dt, 0.7744860053_dt, 0.4368913174_dt, 0.5190907717_dt, 0.6158523560_dt, 0.8101882935_dt, 0.9800970554_dt, 0.1146882176_dt, 0.3167651296_dt, 0.6965049505_dt,
                         0.9142746925_dt, 0.9351036549_dt, 0.9411783814_dt, 0.5995072722_dt, 0.0652086735_dt, 0.5459962487_dt, 0.1871973276_dt, 0.0340229273_dt, 0.9442462325_dt, 0.8801798820_dt,
                         0.0012360215_dt, 0.5935860276_dt, 0.4157699943_dt, 0.4177194238_dt, 0.2711215615_dt, 0.6922780871_dt, 0.2038482428_dt, 0.6832956672_dt, 0.7528540492_dt, 0.8579357862_dt,
                         0.6869555712_dt, 0.0051323771_dt, 0.1756515503_dt, 0.7496575117_dt, 0.6046506763_dt, 0.1099579930_dt, 0.2120902538_dt, 0.9703746438_dt, 0.8369089365_dt, 0.2819874287_dt,
                         0.3741576076_dt, 0.0237009525_dt, 0.4910129309_dt, 0.1234705448_dt, 0.1143216491_dt, 0.4724501967_dt, 0.5750725269_dt, 0.2952348590_dt, 0.7966887951_dt, 0.1957304478_dt,
                         0.9536850452_dt, 0.8426499367_dt, 0.0783585310_dt, 0.3755578399_dt, 0.5225613117_dt, 0.5729505420_dt, 0.6185871363_dt, 0.6962141395_dt, 0.5299500823_dt, 0.2560356259_dt,
                         0.7365944982_dt, 0.0203755498_dt, 0.2036466599_dt, 0.3748350739_dt, 0.2564433217_dt, 0.3250833154_dt, 0.0901891589_dt, 0.3936424255_dt, 0.6068782210_dt, 0.1742671132_dt,
                         0.4743403196_dt, 0.8579254150_dt, 0.4485998750_dt, 0.5138961077_dt, 0.4568655491_dt, 0.6011906862_dt, 0.8179197311_dt, 0.9736230969_dt, 0.8175279498_dt, 0.9747067690_dt,
                         0.4638391733_dt, 0.0508392453_dt, 0.2629613876_dt, 0.8404526114_dt, 0.4967587590_dt, 0.2514768243_dt, 0.1168441176_dt, 0.0320739746_dt, 0.0779958963_dt, 0.3985816240_dt,
                         0.7742030025_dt, 0.7703205347_dt, 0.0177840590_dt, 0.8118910193_dt, 0.1087452769_dt, 0.3942948580_dt };

    // Mask parameter
    const Tensor memorySeqLength{ 4.0_dt, 3.0_dt, 2.0_dt };
    const Tensor scoreMaskValues{ 1.1_dt };

    const Tensor realOut{ 0.4176315367_dt, 0.6501834393_dt, 0.7922494411_dt, 0.2852070928_dt, 0.2437335104_dt, 0.5818774104_dt, 0.7506654859_dt, 0.4769741297_dt, 0.7445594668_dt,
                          0.5289160013_dt, 0.5195670724_dt, 0.8528842926_dt, 0.0290965550_dt, 0.1552542150_dt, 0.3172052503_dt, 0.3221615851_dt, 0.7754383683_dt, 0.3647360802_dt };
    const Tensor realIndices{ 2.0_dt, 5.0_dt, 4.0_dt };
    const auto expectedShape = yato::dims(batchSize, 1, 1, alignmentsSize);

    // In order to reproduce the result
    const Tensor queryLinearLayerWeights{ -0.1122149974_dt, 0.2652359903_dt,  0.4031310081_dt,  -0.2831450105_dt, 0.2075019926_dt,  0.2501629889_dt,  0.0882427990_dt,
                                          0.0866253972_dt,  -0.3076660037_dt, -0.0484487005_dt, -0.3076879978_dt, -0.3577930033_dt, -0.3952620029_dt, -0.0364488997_dt,
                                          0.3275179863_dt,  -0.1487360001_dt, 0.0904399976_dt,  -0.3194299936_dt, 0.1861059964_dt,  0.1349589974_dt };
    const Tensor memoryLinearLayerWeights{ -0.3624039888_dt, -0.3353210092_dt, 0.3552179933_dt,  0.1678149998_dt,  0.2513029873_dt,  0.3315150142_dt,  -0.2174510062_dt,
                                           -0.3773759902_dt, -0.2405180037_dt, 0.3720769882_dt,  -0.2393240035_dt, 0.0888077021_dt,  -0.1479790062_dt, 0.0844018012_dt,
                                           0.0187141001_dt,  -0.3726229966_dt, -0.0514447019_dt, -0.3605310023_dt, -0.1578159928_dt, 0.0187279005_dt,  0.0845528021_dt,
                                           -0.0756980032_dt, -0.2725169957_dt, -0.3426890075_dt, -0.1571239978_dt, 0.3581260145_dt,  -0.1010209993_dt, -0.2020059973_dt };
    const Tensor attentionV{ -0.076089_dt, -0.70909_dt, 0.493939_dt, 0.205051_dt };

    Name battnName = "battn";

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    work.add<DataLayer>("data_query", DataParams{ { "query" }, 1u, 1u, queryDepth });
    work.add<DataLayer>("data_state", DataParams{ { "state" }, 1u, 1u, alignmentsSize });
    work.add<DataLayer>("data_memory", DataParams{ { "memory" }, 1u, alignmentsSize, anyNumber });
    work.add<DataLayer>("data_memory_seq_length", DataParams{ { "memorySeqLength" }, 1u, 1u, 1u });
    work.add<TensorLayer>("data_score_mask_value", TensorParams{ { "scoreMaskValue" }, 1u, 1u, 1u, 1u });

    // Apply function
    BahdanauMonotonicAttentionLayer(
        battnName,
        BahdanauAttentionParams{ { { "query", "state", "memory", "memorySeqLength", "scoreMaskValue" }, { "attn", "values", "max" } }, numUnits, false, 0.0_dt, 1.7_dt, "parallel", true, true },
        networkParameters);

    TENSORS_CREATE(batchSize);
    memory_manager["query"] = TORANGE(query);
    memory_manager["state"] = TORANGE(state);
    memory_manager["memory"] = TORANGE(memory);
    memory_manager["memorySeqLength"] = TORANGE(memorySeqLength);
    memory_manager["scoreMaskValue"] = TORANGE(scoreMaskValues);

    // In order to reproduce the result
    memory_manager[battnName / "query_layer" / "Weights"] = TORANGE(queryLinearLayerWeights);
    memory_manager[battnName / "memory_layer" / "Weights"] = TORANGE(memoryLinearLayerWeights);
    memory_manager[battnName / "attention_v"] = TORANGE(attentionV);

    ASSERT_NO_THROW(work.forwardPassTraining());

    const auto& output = memory_manager["attn"];
    const auto& indices = memory_manager["max"];
    EXPECT_EQ(expectedShape, output.getShape());

    for (size_t i = 0; i < output.size(); ++i)
    {
        ASSERT_TRUE(tools::expect_near_relative(output[i], realOut[i], eps));
    }
    for (size_t i = 0; i < indices.size(); ++i)
    {
        EXPECT_EQ(indices[i], realIndices[i]);
    }
}

TEST(TestLayerStepwiseBahdanauMonotonicAttention, OldBackwardDefaultModeUnit)
{
    PROFILE_TEST
    using namespace raul;

    // See bahdanau_attention.py and bahdanau_attention_scheme.py
    // Test parameters
    const auto eps = TODTYPE(1e-5);
    const size_t numUnits = 4;
    const size_t queryDepth = 5;
    const size_t alignmentsSize = 6;
    const size_t anyNumber = 7;
    const size_t batchSize = 3;
    const Tensor query{ 0.4962565899_dt, 0.7682217956_dt, 0.0884774327_dt, 0.1320304871_dt, 0.3074228168_dt, 0.6340786815_dt, 0.4900934100_dt, 0.8964447379_dt,
                        0.4556279778_dt, 0.6323062778_dt, 0.3488934636_dt, 0.4017173052_dt, 0.0223257542_dt, 0.1688589454_dt, 0.2938884497_dt };

    const Tensor state{ 0.5185217857_dt, 0.6976675987_dt, 0.8000113964_dt, 0.1610294580_dt, 0.2822685838_dt, 0.6816085577_dt, 0.9151939750_dt, 0.3970999122_dt, 0.8741558790_dt,
                        0.4194083214_dt, 0.5529070497_dt, 0.9527381063_dt, 0.0361648202_dt, 0.1852310300_dt, 0.3734173775_dt, 0.3051000237_dt, 0.9320003986_dt, 0.1759101748_dt };

    const Tensor memory{ 0.2698335648_dt, 0.1506797671_dt, 0.0317195058_dt, 0.2081297636_dt, 0.9297990203_dt, 0.7231091857_dt, 0.7423362732_dt, 0.5262957811_dt, 0.2436582446_dt, 0.5845923424_dt,
                         0.0331526399_dt, 0.1387168765_dt, 0.2422350049_dt, 0.8154689670_dt, 0.7931606174_dt, 0.2782524824_dt, 0.4819588065_dt, 0.8197803497_dt, 0.9970665574_dt, 0.6984410882_dt,
                         0.5675464272_dt, 0.8352431655_dt, 0.2055988312_dt, 0.5931720138_dt, 0.1123472452_dt, 0.1534569263_dt, 0.2417082191_dt, 0.7262365222_dt, 0.7010802031_dt, 0.2038237453_dt,
                         0.6510535479_dt, 0.7744860053_dt, 0.4368913174_dt, 0.5190907717_dt, 0.6158523560_dt, 0.8101882935_dt, 0.9800970554_dt, 0.1146882176_dt, 0.3167651296_dt, 0.6965049505_dt,
                         0.9142746925_dt, 0.9351036549_dt, 0.9411783814_dt, 0.5995072722_dt, 0.0652086735_dt, 0.5459962487_dt, 0.1871973276_dt, 0.0340229273_dt, 0.9442462325_dt, 0.8801798820_dt,
                         0.0012360215_dt, 0.5935860276_dt, 0.4157699943_dt, 0.4177194238_dt, 0.2711215615_dt, 0.6922780871_dt, 0.2038482428_dt, 0.6832956672_dt, 0.7528540492_dt, 0.8579357862_dt,
                         0.6869555712_dt, 0.0051323771_dt, 0.1756515503_dt, 0.7496575117_dt, 0.6046506763_dt, 0.1099579930_dt, 0.2120902538_dt, 0.9703746438_dt, 0.8369089365_dt, 0.2819874287_dt,
                         0.3741576076_dt, 0.0237009525_dt, 0.4910129309_dt, 0.1234705448_dt, 0.1143216491_dt, 0.4724501967_dt, 0.5750725269_dt, 0.2952348590_dt, 0.7966887951_dt, 0.1957304478_dt,
                         0.9536850452_dt, 0.8426499367_dt, 0.0783585310_dt, 0.3755578399_dt, 0.5225613117_dt, 0.5729505420_dt, 0.6185871363_dt, 0.6962141395_dt, 0.5299500823_dt, 0.2560356259_dt,
                         0.7365944982_dt, 0.0203755498_dt, 0.2036466599_dt, 0.3748350739_dt, 0.2564433217_dt, 0.3250833154_dt, 0.0901891589_dt, 0.3936424255_dt, 0.6068782210_dt, 0.1742671132_dt,
                         0.4743403196_dt, 0.8579254150_dt, 0.4485998750_dt, 0.5138961077_dt, 0.4568655491_dt, 0.6011906862_dt, 0.8179197311_dt, 0.9736230969_dt, 0.8175279498_dt, 0.9747067690_dt,
                         0.4638391733_dt, 0.0508392453_dt, 0.2629613876_dt, 0.8404526114_dt, 0.4967587590_dt, 0.2514768243_dt, 0.1168441176_dt, 0.0320739746_dt, 0.0779958963_dt, 0.3985816240_dt,
                         0.7742030025_dt, 0.7703205347_dt, 0.0177840590_dt, 0.8118910193_dt, 0.1087452769_dt, 0.3942948580_dt };

    // Mask parameter
    const Tensor memorySeqLength{ 4.0_dt, 3.0_dt, 2.0_dt };
    const Tensor scoreMaskValues{ 1.1_dt };

    const Tensor queryRealGrad{ 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt };
    const Tensor stateRealGrad{ 1.0_dt, 1.0_dt, 1.0_dt,           1.0_dt, 1.0_dt, 0.7502601147_dt, 1.0_dt, 1.0_dt, 1.0_dt,
                                1.0_dt, 1.0_dt, 0.75026011470_dt, 1.0_dt, 1.0_dt, 1.0_dt,          1.0_dt, 1.0_dt, 0.7502601147_dt };
    const Tensor memoryRealGrad{
        0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt,
        0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt,
        0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt,
        0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt,
        0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt,
        0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt
    };

    // In order to reproduce the result
    const Tensor queryLinearLayerWeights{ -0.1122149974_dt, 0.2652359903_dt,  0.4031310081_dt,  -0.2831450105_dt, 0.2075019926_dt,  0.2501629889_dt,  0.0882427990_dt,
                                          0.0866253972_dt,  -0.3076660037_dt, -0.0484487005_dt, -0.3076879978_dt, -0.3577930033_dt, -0.3952620029_dt, -0.0364488997_dt,
                                          0.3275179863_dt,  -0.1487360001_dt, 0.0904399976_dt,  -0.3194299936_dt, 0.1861059964_dt,  0.1349589974_dt };
    const Tensor memoryLinearLayerWeights{ -0.3624039888_dt, -0.3353210092_dt, 0.3552179933_dt,  0.1678149998_dt,  0.2513029873_dt,  0.3315150142_dt,  -0.2174510062_dt,
                                           -0.3773759902_dt, -0.2405180037_dt, 0.3720769882_dt,  -0.2393240035_dt, 0.0888077021_dt,  -0.1479790062_dt, 0.0844018012_dt,
                                           0.0187141001_dt,  -0.3726229966_dt, -0.0514447019_dt, -0.3605310023_dt, -0.1578159928_dt, 0.0187279005_dt,  0.0845528021_dt,
                                           -0.0756980032_dt, -0.2725169957_dt, -0.3426890075_dt, -0.1571239978_dt, 0.3581260145_dt,  -0.1010209993_dt, -0.2020059973_dt };
    const Tensor attentionV{ -0.076089_dt, -0.70909_dt, 0.493939_dt, 0.205051_dt };

    Name battnName = "battn";

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    work.add<DataLayer>("data_query", DataParams{ { "query" }, 1u, 1u, queryDepth });
    work.add<DataLayer>("data_state", DataParams{ { "state" }, 1u, 1u, alignmentsSize });
    work.add<DataLayer>("data_memory", DataParams{ { "memory" }, 1u, alignmentsSize, anyNumber });
    work.add<DataLayer>("data_memory_seq_length", DataParams{ { "memorySeqLength" }, 1u, 1u, 1u });
    work.add<TensorLayer>("data_score_mask_value", TensorParams{ { "scoreMaskValue" }, 1u, 1u, 1u, 1u });

    // Apply function
    BahdanauMonotonicAttentionLayer(
        battnName,
        BahdanauAttentionParams{ { { "query", "state", "memory", "memorySeqLength", "scoreMaskValue" }, { "attn", "values", "max" } }, numUnits, false, 0.0_dt, 1.7_dt, "parallel", true, true },
        networkParameters);

    TENSORS_CREATE(batchSize);
    memory_manager["query"] = TORANGE(query);
    memory_manager["state"] = TORANGE(state);
    memory_manager["memory"] = TORANGE(memory);
    memory_manager["memorySeqLength"] = TORANGE(memorySeqLength);
    memory_manager["scoreMaskValue"] = TORANGE(scoreMaskValues);

    // In order to reproduce the result
    memory_manager[battnName / "query_layer" / "Weights"] = TORANGE(queryLinearLayerWeights);
    memory_manager[battnName / "memory_layer" / "Weights"] = TORANGE(memoryLinearLayerWeights);
    memory_manager[battnName / "attention_v"] = TORANGE(attentionV);
    memory_manager[Name("attn").grad()] = 1.0_dt;

    ASSERT_NO_THROW(work.forwardPassTraining());
    ASSERT_NO_THROW(work.backwardPassTraining());

    const auto queryNabla = memory_manager[Name("query").grad()];
    EXPECT_EQ(queryNabla.size(), queryRealGrad.size());
    for (size_t i = 0; i < queryNabla.size(); ++i)
    {
        ASSERT_TRUE(tools::expect_near_relative(queryNabla[i], queryRealGrad[i], eps) || (queryNabla[i] < eps && queryRealGrad[i] < eps));
    }
    const auto stateNabla = memory_manager[Name("state").grad()];
    EXPECT_EQ(stateNabla.size(), stateRealGrad.size());
    for (size_t i = 0; i < stateNabla.size(); ++i)
    {
        ASSERT_TRUE(tools::expect_near_relative(stateNabla[i], stateRealGrad[i], eps));
    }
    const auto memoryNabla = memory_manager[Name("memory").grad()];
    EXPECT_EQ(memoryNabla.size(), memoryRealGrad.size());
    for (size_t i = 0; i < memoryNabla.size(); ++i)
    {
        ASSERT_TRUE(tools::expect_near_relative(memoryNabla[i], memoryRealGrad[i], eps) || (memoryNabla[i] < eps && memoryRealGrad[i] < eps));
    }
}

TEST(TestLayerBahdanauMonotonicAttention, SimpleMultipleForwardBackwardUnit)
{
    PROFILE_TEST
    using namespace raul;

    // See bahdanau_attention.py
    // Test parameters
    const auto eps = TODTYPE(1e-3);
    const size_t numUnits = 3;
    const size_t queryDepth = 2;
    const size_t alignmentsSize = 3;
    const size_t anyNumber = 2;
    const size_t batchSize = 2;
    const Tensor query1{ 0.8822692633_dt, 0.9150039554_dt, 0.3828637600_dt, 0.9593056440_dt };

    const Tensor state1{ 0.3904482126_dt, 0.6008953452_dt, 0.2565724850_dt, 0.7936413288_dt, 0.9407714605_dt, 0.1331859231_dt };

    const Tensor memory1{ 0.9345980883_dt, 0.5935796499_dt, 0.8694044352_dt, 0.5677152872_dt, 0.7410940528_dt, 0.4294044971_dt,
                          0.8854429126_dt, 0.5739044547_dt, 0.2665800452_dt, 0.6274491549_dt, 0.2696316838_dt, 0.4413635731_dt };

    const Tensor query2{ 0.2969208360_dt, 0.8316854835_dt, 0.1053149104_dt, 0.2694948316_dt };

    const Tensor state2{ 0.3588126302_dt, 0.1993637681_dt, 0.5471915603_dt, 0.0061604381_dt, 0.9515545368_dt, 0.0752658844_dt };

    const Tensor memory2{ 0.8860136867_dt, 0.5832095742_dt, 0.3376477361_dt, 0.8089749813_dt, 0.5779253840_dt, 0.9039816856_dt,
                          0.5546598434_dt, 0.3423134089_dt, 0.6343418360_dt, 0.3644102812_dt, 0.7104287744_dt, 0.9464110732_dt };

    // In order to reproduce the result
    const Tensor queryLinearLayerWeights{ -0.1448689997_dt, 0.3424179852_dt, -0.3655380011_dt, 0.2678830028_dt, 0.123214_dt, 0.565657676_dt };
    const Tensor memoryLinearLayerWeights{ -0.0409465991_dt, -0.2600440085_dt, 0.0764357001_dt, -0.2699669898_dt, 0.675676_dt, 0.234354545_dt };
    const Tensor attentionV{ -0.633191_dt, 0.234963_dt, -0.391515_dt };

    const Tensor realAttentionVGrad{ 0.0224523991_dt, -0.0126725147_dt, 0.2702962756_dt };
    const Tensor realScoreBiasGrad{ 0.3677265942_dt };

    Name battnChild = "battn_child";
    Name battnParent = "battn_parent";

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    work.add<DataLayer>("data_query", DataParams{ { "query1", "query2" }, 1u, 1u, queryDepth });
    work.add<DataLayer>("data_state", DataParams{ { "state1", "state2" }, 1u, 1u, alignmentsSize });
    work.add<DataLayer>("data_memory", DataParams{ { "memory1", "memory2" }, 1u, alignmentsSize, anyNumber });

    BahdanauMonotonicAttentionLayer(battnParent, BahdanauAttentionParams{ { { "query1", "state1", "memory1" }, { "attn1" } }, numUnits, false, 0.0_dt, 1.7_dt }, networkParameters);
    BahdanauMonotonicAttentionLayer(battnChild, BahdanauAttentionParams{ { { "query2", "state2", "memory2" }, { "attn2" } }, battnParent, numUnits, false, 0.0_dt, 1.7_dt }, networkParameters);

    TENSORS_CREATE(batchSize);
    memory_manager["query1"] = TORANGE(query1);
    memory_manager["state1"] = TORANGE(state1);
    memory_manager["memory1"] = TORANGE(memory1);
    memory_manager["query2"] = TORANGE(query2);
    memory_manager["state2"] = TORANGE(state2);
    memory_manager["memory2"] = TORANGE(memory2);

    // In order to reproduce the result
    memory_manager[battnParent / "query_layer" / "Weights"] = TORANGE(queryLinearLayerWeights);
    memory_manager[battnParent / "memory_layer" / "Weights"] = TORANGE(memoryLinearLayerWeights);
    memory_manager[battnParent / "attention_v"] = TORANGE(attentionV);
    memory_manager[Name("attn1").grad()] = 1.0_dt;
    memory_manager[Name("attn2").grad()] = 1.0_dt;

    ASSERT_NO_THROW(work.forwardPassTraining());
    ASSERT_NO_THROW(work.backwardPassTraining());

    // Checks
    const auto attentionVNabla = memory_manager[(battnParent / "attention_v").grad()];
    EXPECT_EQ(attentionVNabla.size(), realAttentionVGrad.size());
    for (size_t i = 0; i < attentionVNabla.size(); ++i)
    {
        ASSERT_TRUE(tools::expect_near_relative(attentionVNabla[i], realAttentionVGrad[i], eps));
    }
    const auto scoreBiasNabla = memory_manager[(battnParent / "score_bias").grad()];
    EXPECT_EQ(scoreBiasNabla.size(), realScoreBiasGrad.size());
    for (size_t i = 0; i < scoreBiasNabla.size(); ++i)
    {
        ASSERT_TRUE(tools::expect_near_relative(scoreBiasNabla[i], realScoreBiasGrad[i], eps));
    }
}

TEST(TestLayerBahdanauMonotonicAttention, MultipleForwardBackwardUnit)
{
    PROFILE_TEST
    using namespace raul;

    // See bahdanau_attention.py
    // Test parameters
    const auto eps = TODTYPE(1e-3);
    const size_t numUnits = 3;
    const size_t queryDepth = 2;
    const size_t alignmentsSize = 2;
    const size_t anyNumber = 2;
    const size_t batchSize = 2;
    const Tensor query1{ 0.8822692633_dt, 0.9150039554_dt, 0.3828637600_dt, 0.9593056440_dt };

    const Tensor state1{ 0.3904482126_dt, 0.6008953452_dt, 0.2565724850_dt, 0.7936413288_dt };

    const Tensor memory1{ 0.9407714605_dt, 0.1331859231_dt, 0.9345980883_dt, 0.5935796499_dt, 0.8694044352_dt, 0.5677152872_dt, 0.7410940528_dt, 0.4294044971_dt };

    const Tensor query2{ 0.8854429126_dt, 0.5739044547_dt, 0.2665800452_dt, 0.6274491549_dt };

    const Tensor state2{ 0.2696316838_dt, 0.4413635731_dt, 0.2969208360_dt, 0.8316854835_dt };

    // Mask parameter
    const Tensor memorySeqLength1{ 1.0_dt, 1.0_dt };
    const Tensor scoreMaskValues1{ 1.1_dt };
    const Tensor memorySeqLength2{ 1.0_dt, 1.0_dt };
    const Tensor scoreMaskValues2{ 1.1_dt };

    // In order to reproduce the result
    const Tensor queryLinearLayerWeights{ -0.1448689997_dt, 0.3424179852_dt, -0.3655380011_dt, 0.2678830028_dt, 0.123214_dt, 0.565657676_dt };
    const Tensor memoryLinearLayerWeights{ -0.0409465991_dt, -0.2600440085_dt, 0.0764357001_dt, -0.2699669898_dt, 0.675676_dt, 0.234354545_dt };
    const Tensor attentionV{ -0.633191_dt, 0.234963_dt, -0.391515_dt };

    const Tensor realFinalResult{ 0.2017848790_dt, 0.0496129580_dt, 0.0_dt, 0.0_dt, 0.2065062523_dt, 0.3795414865_dt, 0.0_dt, 0.0_dt };

    const Tensor realFinalValues{ 0.9407714605_dt, 0.1331859231_dt, 0.0_dt, 0.0_dt, 0.8694044352_dt, 0.5677152872_dt, 0.0_dt, 0.0_dt };

    // Gradient for internals
    const Tensor realAttentionVNabla{ 0.0076167323_dt, -0.0069890255_dt, 0.1108852476_dt };
    const Tensor realScoreBiasNabla{ 0.1327716708_dt };

    // Gradients for inputs
    const Tensor realQuery1Nabla{ -0.0004310280_dt, -0.0116593838_dt, -0.0001334163_dt, -0.0039595729_dt };
    const Tensor realState1Nabla{ 0.7582378983_dt, 0.0999240801_dt, 0.7753205299_dt, 0.4259341657_dt };
    const Tensor realMemory1Nabla{ 0.5168668628_dt, 0.8936948776_dt, 0.0_dt, 0.0_dt, 0.4381725192_dt, 1.3077256680_dt, 0.0_dt, 0.0_dt };
    const Tensor realQuery2Nabla{ -0.0003669365_dt, -0.0086477958_dt, -0.0002368438_dt, -0.0048965309_dt };
    const Tensor realState2Nabla{ 0.7688080072_dt, 0.0999240875_dt, 0.7806946039_dt, 0.4259341359_dt };

    Name battnChild = "battn_child";
    Name battnParent = "battn_parent";

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    work.add<DataLayer>("data_query", DataParams{ { "query1", "query2" }, 1u, 1u, queryDepth });
    work.add<DataLayer>("data_state", DataParams{ { "state1", "state2" }, 1u, 1u, alignmentsSize });
    work.add<DataLayer>("data_memory", DataParams{ { "memory1" }, 1u, alignmentsSize, anyNumber });
    work.add<DataLayer>("data_memory_seq_length", DataParams{ { "memorySeqLength1", "memorySeqLength2" }, 1u, 1u, 1u });
    work.add<TensorLayer>("data_score_mask_value1", TensorParams{ { "scoreMaskValues1" }, 1u, 1u, 1u, 1u });
    work.add<TensorLayer>("data_score_mask_value2", TensorParams{ { "scoreMaskValues2" }, 1u, 1u, 1u, 1u });

    BahdanauMonotonicAttentionLayer(
        battnParent, BahdanauAttentionParams{ { { "query1", "state1", "memory1", "memorySeqLength1", "scoreMaskValues1" }, { "y1", "values" } }, numUnits, false, 0.0_dt, 1.7_dt }, networkParameters);
    work.add<ElementWiseMulLayer>("mul1", ElementWiseLayerParams{ { "values", "y1" }, { "z1" } });
    BahdanauMonotonicAttentionLayer(
        battnChild, BahdanauAttentionParams{ { { "query2", "state2", "z1", "memorySeqLength2", "scoreMaskValues2" }, { "y2" } }, battnParent, numUnits, false, 0.0_dt, 1.7_dt }, networkParameters);
    work.add<ElementWiseMulLayer>("mul2", ElementWiseLayerParams{ { "values", "y2" }, { "z2" } });

    TENSORS_CREATE(batchSize);
    memory_manager["query1"] = TORANGE(query1);
    memory_manager["state1"] = TORANGE(state1);
    memory_manager["memory1"] = TORANGE(memory1);
    memory_manager["query2"] = TORANGE(query2);
    memory_manager["state2"] = TORANGE(state2);

    memory_manager["memorySeqLength1"] = TORANGE(memorySeqLength1);
    memory_manager["scoreMaskValues1"] = TORANGE(scoreMaskValues1);

    memory_manager["memorySeqLength2"] = TORANGE(memorySeqLength2);
    memory_manager["scoreMaskValues2"] = TORANGE(scoreMaskValues2);

    // In order to reproduce the result
    memory_manager[battnParent / "query_layer" / "Weights"] = TORANGE(queryLinearLayerWeights);
    memory_manager[battnParent / "memory_layer" / "Weights"] = TORANGE(memoryLinearLayerWeights);
    memory_manager[battnParent / "attention_v"] = TORANGE(attentionV);
    memory_manager[Name("z2").grad()] = 1.0_dt;
    memory_manager[Name("z1").grad()] = 1.0_dt;

    ASSERT_NO_THROW(work.forwardPassTraining());

    // Forward cheks
    const auto finalResult = memory_manager["z2"];
    const auto finalValues = memory_manager["values"];
    for (size_t i = 0; i < finalResult.size(); ++i)
    {
        ASSERT_TRUE(tools::expect_near_relative(finalResult[i], realFinalResult[i], eps));
    }
    for (size_t i = 0; i < finalValues.size(); ++i)
    {
        ASSERT_TRUE(tools::expect_near_relative(finalValues[i], realFinalValues[i], eps));
    }

    work.backwardPassTraining();

    const auto attentionVNabla = memory_manager[(battnParent / "attention_v").grad()];
    EXPECT_EQ(attentionVNabla.size(), realAttentionVNabla.size());
    for (size_t i = 0; i < attentionVNabla.size(); ++i)
    {
        ASSERT_TRUE(tools::expect_near_relative(attentionVNabla[i], realAttentionVNabla[i], eps));
    }
    const auto scoreBiasNabla = memory_manager[(battnParent / "score_bias").grad()];
    EXPECT_EQ(scoreBiasNabla.size(), realScoreBiasNabla.size());
    for (size_t i = 0; i < scoreBiasNabla.size(); ++i)
    {
        ASSERT_TRUE(tools::expect_near_relative(scoreBiasNabla[i], realScoreBiasNabla[i], eps));
    }
    const auto query1Nabla = memory_manager[Name("query1").grad()];
    EXPECT_EQ(query1Nabla.size(), realQuery1Nabla.size());
    for (size_t i = 0; i < query1Nabla.size(); ++i)
    {
        ASSERT_TRUE(tools::expect_near_relative(query1Nabla[i], realQuery1Nabla[i], eps));
    }
    const auto state1Nabla = memory_manager[Name("state1").grad()];
    EXPECT_EQ(state1Nabla.size(), realState1Nabla.size());
    for (size_t i = 0; i < state1Nabla.size(); ++i)
    {
        ASSERT_TRUE(tools::expect_near_relative(state1Nabla[i], realState1Nabla[i], eps));
    }
    const auto memory1Nabla = memory_manager[Name("memory1").grad()];
    EXPECT_EQ(memory1Nabla.size(), realMemory1Nabla.size());
    for (size_t i = 0; i < memory1Nabla.size(); ++i)
    {
        ASSERT_TRUE(tools::expect_near_relative(memory1Nabla[i], realMemory1Nabla[i], eps));
    }
    const auto query2Nabla = memory_manager[Name("query2").grad()];
    EXPECT_EQ(query2Nabla.size(), realQuery2Nabla.size());
    for (size_t i = 0; i < query2Nabla.size(); ++i)
    {
        ASSERT_TRUE(tools::expect_near_relative(query2Nabla[i], realQuery2Nabla[i], eps));
    }
    const auto state2Nabla = memory_manager[Name("state2").grad()];
    EXPECT_EQ(state2Nabla.size(), realState2Nabla.size());
    for (size_t i = 0; i < state2Nabla.size(); ++i)
    {
        ASSERT_TRUE(tools::expect_near_relative(state2Nabla[i], realState2Nabla[i], eps));
    }
}

}