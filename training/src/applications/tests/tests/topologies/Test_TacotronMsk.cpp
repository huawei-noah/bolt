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
#include <tests/tools/callbacks/LayerTypeStatistics.h>
#include <tests/tools/callbacks/TensorChecker.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <set>

#include <tests/tools/TestTools.h>

#include <training/layers/basic/DataLayer.h>
#include <training/layers/basic/PositionalEncoding.h>
#include <training/layers/basic/ConvertPrecisionLayer.h>
#include <training/layers/basic/TensorLayer.h>
#include <training/layers/basic/TileLayer.h>
#include <training/layers/composite/TacotronModel.h>
#include <training/layers/composite/tacotron/DurationPredictor.h>
#include <training/layers/composite/tacotron/TacotronLoss.h>
#include <training/postprocessing/GradientClipping.h>

#include <training/api/API.h>

#include "TacotronTestTools.h"
#include "TacotronTrainingParams.h"

namespace
{
    template<typename T> struct dname
    {
    };

    template<> struct dname<raul::MemoryManager>
    {
        [[maybe_unused]] static constexpr char const* name = "cpu";
    };

    template<> struct dname<raul::MemoryManagerGPU>
    {
        [[maybe_unused]] static constexpr char const* name = "gpu";
    };

    template<typename T>
    std::function<void(raul::BasicLayer*, T&, raul::NetworkParameters::CallbackPlace)> debugCallback()
    {
        return [](raul::BasicLayer* l, T& mm, raul::NetworkParameters::CallbackPlace p) 
        {   
           // sample debug callback
            if (p == raul::NetworkParameters::CallbackPlace::Before_Backward)
            {
                auto lname = l->getName().str();
                Common::replaceAll(lname, "::", "_");
                for (auto& i : l->getOutputs())
                {
                    if (!mm.tensorExists(i.grad()))
                    {
                        cout << "Missing tensor: " << i << endl;
                        continue;
                    }
                    /*if (mm[i.grad()].size() == 0)
                    {
                        cout << "Empty tensor: " << i << endl;
                        continue;
                    }*/
                    static size_t index = 0;
                    std::stringstream filename; 
                    filename.fill('0'); 
                    filename.width(6); 
                    filename << std::to_string(index);
                    
                    auto iname = i.grad();
                    Common::replaceAll(iname, "::", "_");
                    UT::tools::print_tensor(mm[i.grad()], "d:/"s + dname<std::remove_reference_t<T>>::name + "/" + filename.str() + "_" + iname + " (" + lname + ").txt");

                    ++index;
                }
            }
        };
    }
}

namespace UT
{
using namespace raul;
using namespace UT::tools::callbacks;

TEST(TestTacotronMsk, RangePositionalEncodingUnit)
{
    PROFILE_TEST

    // Test parameters
    constexpr auto eps_rel = 1e-3_dt;
    constexpr auto eps_abs = 1e-5_dt;
    constexpr auto input_size = 86;
    constexpr auto max_length = 150U;
    constexpr auto max_mel_length = 222U;
    constexpr auto embed_dim = 32U;
    constexpr auto batch_size = 12U;

    const auto path = tools::getTestAssetsDir() / "Tacotron_Msk_RP" / "positional_encoding";

    Workflow work;

    work.add<DataLayer>("input", DataParams{ { "durations" }, 1, 1, input_size });
    work.add<DataLayer>("golden_output", DataParams{ { "pe_golden" }, 1, max_mel_length, embed_dim });
    work.add<PositionalEncoding>("pe", PositionalEncodingParams{ "durations", "pe", embed_dim, max_length, true, max_mel_length });

    TENSORS_CREATE(batch_size);
    MemoryManager& memory_manager = work.getMemoryManager();
    // Data loading
    EXPECT_TRUE(loadTFData<Tensor>(path / "pe_durations.data", { &memory_manager["durations"] }));
    EXPECT_TRUE(loadTFData<Tensor>(path / "pe_output.data", { &memory_manager["pe_golden"] }));

    TensorChecker checker({ { "pe", "pe_golden" } }, eps_abs, eps_rel);
    work.getNetworkParameters().mCallback = checker;
    work.forwardPassTraining();
}

TEST(TestTacotronMsk, RangePositionalEncodingGpuUnit)
{
    PROFILE_TEST

    GPU_ONLY_TEST

    // Test parameters
    constexpr auto eps_abs = 1e-5_dt;
    constexpr auto input_size = 86;
    constexpr auto max_length = 150U;
    constexpr auto max_mel_length = 222U;
    constexpr auto embed_dim = 32U;
    constexpr auto batch_size = 12U;

    const auto path = tools::getTestAssetsDir() / "Tacotron_Msk_RP" / "positional_encoding";

    WorkflowEager work(CompressionMode::NONE, CalculationMode::DETERMINISTIC, AllocationMode::STANDARD, ExecutionTarget::GPU);
    auto& memory_manager = work.getMemoryManager<MemoryManagerGPU>();

    work.add<DataLayer>("input", DataParams{ { "durations" }, 1, 1, input_size });
    work.add<DataLayer>("golden_output", DataParams{ { "pe_golden" }, 1, max_mel_length, embed_dim });
    work.add<PositionalEncoding>("pe", PositionalEncodingParams{ "durations", "pe", embed_dim, max_length, true, max_mel_length });

    TENSORS_CREATE(batch_size);

    // Data loading
    EXPECT_TRUE(loadTFData(path / "pe_durations.data", memory_manager["durations"]));

    work.forwardPassTraining();
    Tensor pe = memory_manager["pe"];
    Tensor golden(batch_size, 1, max_mel_length, embed_dim);
    EXPECT_TRUE(loadTFData(path / "pe_output.data", golden));
    EXPECT_EQ(pe.size(), golden.size());
    for (size_t i = 0; i < golden.size(); ++i)
    {
        CHECK_NEAR(pe[i], golden[i], eps_abs);
    }
}

TEST(TestTacotronMsk, InputMaskTilingUnit)
{
    PROFILE_TEST

    auto testPath = tools::getTestAssetsDir() / "Tacotron_Msk_RP";
    constexpr dtype EPS = 1e-6_dt;

    WorkflowEager work(CompressionMode::NONE, CalculationMode::DETERMINISTIC, AllocationMode::STANDARD, ExecutionTarget::CPU);
    auto& memory_manager = work.getMemoryManager();
    size_t width = 768;
    size_t depth = 1;
    size_t height = 86;
    size_t batch = 12;
    work.add<DataLayer>("input", DataParams{ {"input_mask"}, depth, height, 1 });
    work.add<DataLayer>("golden", DataParams{ {"golden_tiled_mask"}, depth, height, width });
    work.add<TileLayer>("tile_mask", TilingParams{ "input_mask", "tiled_mask", width, Dimension::Width });
    TENSORS_CREATE(batch);

    EXPECT_TRUE(loadTFData(testPath / "tiling" / "input_mask.data", memory_manager["input_mask"]));
    EXPECT_TRUE(loadTFData(testPath / "tiling" / "tiled_mask.data", memory_manager["golden_tiled_mask"]));

    ASSERT_NO_THROW(work.forwardPassTraining());

    UT::tools::checkTensors({ {"tiled_mask", "golden_tiled_mask"} }, memory_manager, EPS);
}

TEST(TestTacotronMsk, InputMaskTilingGpuUnit)
{
    PROFILE_TEST

    GPU_ONLY_TEST

    auto testPath = tools::getTestAssetsDir() / "Tacotron_Msk_RP";
    constexpr dtype EPS = 1e-6_dt;

    WorkflowEager work(CompressionMode::NONE, CalculationMode::DETERMINISTIC, AllocationMode::STANDARD, ExecutionTarget::GPU);
    auto& memory_manager = work.getMemoryManager<MemoryManagerGPU>();
    size_t width = 768;
    size_t depth = 1;
    size_t height = 86;
    size_t batch = 12;
    work.add<DataLayer>("input", DataParams{ {"input_mask"}, depth, height, 1 });
    work.add<DataLayer>("golden", DataParams{ {"golden_tiled_mask"}, depth, height, width });
    work.add<TileLayer>("tile_mask", TilingParams{ "input_mask", "tiled_mask", width, Dimension::Width });
    TENSORS_CREATE(batch);

    EXPECT_TRUE(loadTFData(testPath / "tiling" / "input_mask.data", memory_manager["input_mask"]));
    EXPECT_TRUE(loadTFData(testPath / "tiling" / "tiled_mask.data", memory_manager["golden_tiled_mask"]));

    ASSERT_NO_THROW(work.forwardPassTraining());

    UT::tools::checkTensors({ {"tiled_mask", "golden_tiled_mask"} }, memory_manager, EPS);
}

TEST(TestTacotronMsk, DurationPredictorUnit)
{
    PROFILE_TEST

    WorkflowEager work;
    constexpr auto batch_size = 12U;
    constexpr size_t T_IN = 86;
    constexpr size_t encoder_output_size = 768;
    constexpr size_t DURATION_PREDICTION_PARAMS = 20;

    constexpr dtype EPS = 1e-5_dt;

    TacotronParams params({}, {}, {});
    params.maxMelFrames = 666;
    params.useDurationPrediction = true;
    params.attentionType = "";
    params.durationPredictorLstmUnits = 256;
    params.postnetKernelSize = { 5, 5, 5, 5, 3 };
    params.prenetLayers = { 256, 256 };
    params.dropoutRate = params.zoneoutRate = params.postnetDropoutRate = 0.f;
    auto testPath = tools::getTestAssetsDir() / "Tacotron_Msk_RP";

    work.add<DataLayer>("encoder_output", DataParams{ { "encoder_output" }, 1u, T_IN, encoder_output_size });
    work.add<DataLayer>("input_lengths", DataParams{ { "input_lengths" }, 1u, 1u, 1u });
    work.add<DataLayer>("duration_targets", DataParams{ { "duration_targets" }, 1u, 1u, T_IN });
    work.add<DataLayer>("input_mask", DataParams{ { "input_mask" }, 1u, T_IN, 1u });

    work.add<DataLayer>("golden_durations_output", DataParams{ { "golden_durations_output" }, 1u, params.maxMelFrames / params.outputsPerStep, encoder_output_size + params.positionalEmbeddingDim });
    work.add<DataLayer>("golden_durations", DataParams{ { "golden_durations" }, 1u, T_IN, 1u });
    work.add<DataLayer>("golden_upsampled_output", DataParams{ { "golden_upsampled_output" }, 1u, params.maxMelFrames / params.outputsPerStep, encoder_output_size });
    work.add<DataLayer>("golden_positional_embeddings", DataParams{ { "golden_positional_embeddings" }, 1u, params.maxMelFrames / params.outputsPerStep, params.positionalEmbeddingDim });

    tacotron::AddDurationPredictor(&work,
                                   Name("T") / "duration_predictor",
                                   { { "encoder_output", "input_lengths", "duration_targets", "input_mask" }, { "durations_output", "durations", "upsampled_output", "positional_embeddings" } },
                                   params);

    TENSORS_CREATE(batch_size);

    EXPECT_EQ(work.getTrainableParameterNames().size(), DURATION_PREDICTION_PARAMS);

    auto& memory_manager = work.getMemoryManager();

    // Data loading

    EXPECT_TRUE(loadTFData(testPath / "duration_predictor" / "dp_encoder_output.data", memory_manager["encoder_output"]));
    EXPECT_TRUE(loadTFData(testPath / "input_lengths.data", memory_manager["input_lengths"]));
    EXPECT_TRUE(loadTFData(testPath / "duration_targets.data", memory_manager["duration_targets"]));
    EXPECT_TRUE(loadTFData(testPath / "duration_predictor" / "input_mask.data", memory_manager["input_mask"]));

    EXPECT_TRUE(loadTFData(testPath / "duration_predictor" / "golden_durations_output.data", memory_manager["golden_durations_output"]));
    EXPECT_TRUE(loadTFData(testPath / "duration_predictor" / "golden_durations.data", memory_manager["golden_durations"]));
    EXPECT_TRUE(loadTFData(testPath / "duration_predictor" / "golden_upsampled_output.data", memory_manager["golden_upsampled_output"]));
    EXPECT_TRUE(loadTFData(testPath / "duration_predictor" / "golden_positional_embeddings.data", memory_manager["golden_positional_embeddings"]));

    size_t loaded = loadTacotronParams((testPath / "weights" / "vctk_small_250_Tacotron_model.").string(), memory_manager, "T", params, false, false);
    EXPECT_EQ(loaded, DURATION_PREDICTION_PARAMS);

    TensorChecker checker(
        {
            { "durations_output", "golden_durations_output" },
            { "durations", "golden_durations" },
            { "upsampled_output", "golden_upsampled_output" },
            { "positional_embeddings", "golden_positional_embeddings" },
        },
        EPS,
        -1_dt);
    work.getNetworkParameters().mCallback = checker;
    ASSERT_NO_THROW(work.forwardPassTraining());
    ASSERT_NO_THROW(work.backwardPassTraining());
}

TEST(TestTacotronMsk, DurationPredictorGpuUnit)
{
    PROFILE_TEST

    GPU_ONLY_TEST

    WorkflowEager work(CompressionMode::NONE, CalculationMode::DETERMINISTIC, AllocationMode::STANDARD, ExecutionTarget::GPU);
    auto& memory_manager = work.getMemoryManager<MemoryManagerGPU>();

    constexpr auto batch_size = 12U;
    constexpr size_t T_IN = 86;
    constexpr size_t encoder_output_size = 768;
    constexpr size_t DURATION_PREDICTION_PARAMS = 20;

    constexpr dtype EPS = 1e-5_dt;

    TacotronParams params({}, {}, {});
    params.maxMelFrames = 666;
    params.useDurationPrediction = true;
    params.attentionType = "";
    params.durationPredictorLstmUnits = 256;
    params.postnetKernelSize = { 5, 5, 5, 5, 3 };
    params.prenetLayers = { 256, 256 };
    params.dropoutRate = params.zoneoutRate = params.postnetDropoutRate = 0.f;
    auto testPath = tools::getTestAssetsDir() / "Tacotron_Msk_RP";

    work.add<DataLayer>("encoder_output", DataParams{ { "encoder_output" }, 1u, T_IN, encoder_output_size });
    work.add<DataLayer>("input_lengths", DataParams{ { "input_lengths" }, 1u, 1u, 1u });
    work.add<DataLayer>("duration_targets", DataParams{ { "duration_targets" }, 1u, 1u, T_IN });
    work.add<DataLayer>("input_mask", DataParams{ { "input_mask" }, 1u, T_IN, 1u });

    work.add<DataLayer>("golden_durations_output", DataParams{ { "golden_durations_output" }, 1u, params.maxMelFrames / params.outputsPerStep, encoder_output_size + params.positionalEmbeddingDim });
    work.add<DataLayer>("golden_durations", DataParams{ { "golden_durations" }, 1u, T_IN, 1u });
    work.add<DataLayer>("golden_upsampled_output", DataParams{ { "golden_upsampled_output" }, 1u, params.maxMelFrames / params.outputsPerStep, encoder_output_size });
    work.add<DataLayer>("golden_positional_embeddings", DataParams{ { "golden_positional_embeddings" }, 1u, params.maxMelFrames / params.outputsPerStep, params.positionalEmbeddingDim });

    tacotron::AddDurationPredictor(&work,
                                   Name("T") / "duration_predictor",
                                   { { "encoder_output", "input_lengths", "duration_targets", "input_mask" }, { "durations_output", "durations", "upsampled_output", "positional_embeddings" } },
                                   params);

    TENSORS_CREATE(batch_size);

    EXPECT_EQ(work.getTrainableParameterNames().size(), DURATION_PREDICTION_PARAMS);

    // Data loading

    EXPECT_TRUE(loadTFData(testPath / "duration_predictor" / "dp_encoder_output.data", memory_manager["encoder_output"]));
    EXPECT_TRUE(loadTFData(testPath / "input_lengths.data", memory_manager["input_lengths"]));
    EXPECT_TRUE(loadTFData(testPath / "duration_targets.data", memory_manager["duration_targets"]));
    EXPECT_TRUE(loadTFData(testPath / "duration_predictor" / "input_mask.data", memory_manager["input_mask"]));

    EXPECT_TRUE(loadTFData(testPath / "duration_predictor" / "golden_durations_output.data", memory_manager["golden_durations_output"]));
    EXPECT_TRUE(loadTFData(testPath / "duration_predictor" / "golden_durations.data", memory_manager["golden_durations"]));
    EXPECT_TRUE(loadTFData(testPath / "duration_predictor" / "golden_upsampled_output.data", memory_manager["golden_upsampled_output"]));
    EXPECT_TRUE(loadTFData(testPath / "duration_predictor" / "golden_positional_embeddings.data", memory_manager["golden_positional_embeddings"]));

    size_t loaded = loadTacotronParams((testPath / "weights" / "vctk_small_250_Tacotron_model.").string(), memory_manager, "T", params, false, false);
    EXPECT_EQ(loaded, DURATION_PREDICTION_PARAMS);

    ASSERT_NO_THROW(work.forwardPassTraining());

    vector<pair<Name, Name>> checks = {
        { "durations_output", "golden_durations_output" },
        { "durations", "golden_durations" },
        { "upsampled_output", "golden_upsampled_output" },
        { "positional_embeddings", "golden_positional_embeddings" },
    };
    tools::checkTensors(checks, memory_manager, EPS);

    ASSERT_NO_THROW(work.backwardPassTraining());
}

TEST(TestTacotronMsk, ParamsCountUnit)
{
    PROFILE_TEST

    constexpr size_t GOLDEN_TRAINABLE_PARAMS = 54;
    constexpr size_t GOLDEN_PARAMS = 64;

    TacotronParams params({}, {}, {});
    params.useDurationPrediction = true;
    params.attentionType = "";
    params.trainableSpeakerEmbedding = false;
    params.durationPredictorLstmUnits = 256;
    params.postnetKernelSize = { 5, 5, 5, 5, 3 };
    params.prenetLayers = { 256, 256 };

    auto count = tacotronParamsCount(params);

    EXPECT_EQ(count, std::make_pair(GOLDEN_TRAINABLE_PARAMS, GOLDEN_PARAMS));
}

TEST(TestTacotronMsk, ModelParamsCountUnit)
{
    PROFILE_TEST

    constexpr size_t GOLDEN_TRAINABLE_PARAMS = 54;

    TacotronParams params({ "encoder_output", "speaker_embedding", "duration_targets", "inputs", "input_lengths", "mel_targets" }, { "decoder_output", "mel_outputs", "duration_outputs" }, {});
    params.useDurationPrediction = true;
    params.attentionType = "";
    params.trainableSpeakerEmbedding = false;
    params.durationPredictorLstmUnits = 256;
    params.postnetKernelSize = { 5, 5, 5, 5, 3 };
    params.prenetLayers = { 256, 256 };
    params.withoutStopTokenLoss = true;
    params.lossMultipliers = { 1.f, 1.f, 2.f };

    Workflow work;
    constexpr size_t T_IN = 86;

    work.add<DataLayer>("inputs", DataParams{ { "inputs" }, 1u, T_IN, 1u });
    work.add<DataLayer>("duration_targets", DataParams{ { "duration_targets" }, 1u, 1u, T_IN });
    work.add<DataLayer>("encoder_output", DataParams{ { "encoder_output" }, 1u, T_IN, params.embeddingDim });
    work.add<DataLayer>("input_lengths", DataParams{ { "input_lengths" }, 1u, 1u, 1u });
    work.add<DataLayer>("mel_targets", DataParams{ { "mel_targets" }, 1u, params.maxMelFrames, params.numMels });
    work.add<DataLayer>("targets_lengths", DataParams{ { "targets_lengths" }, 1u, 1u, 1u });
    work.add<DataLayer>("speaker_embedding", DataParams{ { "speaker_embedding" }, 1u, 1u, params.speakerEmbeddingSize });

    AddSingleSpeakerFslTacotronModel(&work, "T", params);

    size_t trainableParamsCount = work.getTrainableParameterNames().size();
    EXPECT_EQ(trainableParamsCount, GOLDEN_TRAINABLE_PARAMS);
}

TEST(TestTacotronMsk, P225Unit)
{
    PROFILE_TEST
    using namespace tacotron;

    auto testPath = tools::getTestAssetsDir() / "Tacotron_Msk_RP";

    constexpr size_t GOLDEN_PARAMS = 64;
    constexpr size_t GOLDEN_TRAINABLE_PARAMS = 54;

    WorkflowEager work;
    constexpr size_t T_IN = 86;
    constexpr size_t BATCH_SIZE = 12;
    constexpr dtype LOSS_EPS = 1e-4_dt;
    constexpr dtype EPS = 4e-3_dt; // intermediate values on android can have rather big difference

    TacotronParams params({ "encoder_output", "speaker_embedding", "duration_targets", "inputs", "input_lengths", "mel_targets" }, { "decoder_output", "mel_outputs", "duration_outputs" }, {});
    params.maxMelFrames = 666;
    params.useDurationPrediction = true;
    params.attentionType = "";
    params.durationPredictorLstmUnits = 256;
    params.postnetKernelSize = { 5, 5, 5, 5, 3 };
    params.prenetLayers = { 256, 256 };
    params.dropoutRate = 0.f;
    params.zoneoutRate = 0.f;
    params.postnetDropoutRate = 0.f;
    params.trainableSpeakerEmbedding = false;
    params.withoutStopTokenLoss = true;
    params.lossMultipliers = { 1.f, 1.f, 2.f };
    params.useResidualRnn = true;

    work.add<DataLayer>("inputs", DataParams{ { "inputs" }, 1u, T_IN, 1u });
    work.add<DataLayer>("duration_targets", DataParams{ { "duration_targets" }, 1u, 1u, T_IN });
    work.add<DataLayer>("encoder_output", DataParams{ { "encoder_output" }, 1u, T_IN, params.embeddingDim });
    work.add<DataLayer>("input_lengths", DataParams{ { "input_lengths" }, 1u, 1u, 1u });
    work.add<DataLayer>("mel_targets", DataParams{ { "mel_targets" }, 1u, params.maxMelFrames, params.numMels });
    work.add<DataLayer>("targets_lengths", DataParams{ { "targets_lengths" }, 1u, 1u, 1u });
    work.add<DataLayer>("speaker_embedding", DataParams{ { "speaker_embedding" }, 1u, 1u, params.speakerEmbeddingSize });

    work.add<DataLayer>("golden_decoder_output", DataParams{ { "golden_decoder_output" }, 1u, params.maxMelFrames, params.numMels });
    work.add<DataLayer>("golden_mel_outputs", DataParams{ { "golden_mel_outputs" }, 1u, params.maxMelFrames, params.numMels });
    work.add<DataLayer>("golden_duration_outputs", DataParams{ { "golden_duration_outputs" }, 1u, T_IN, 1u });
    
    AddSingleSpeakerFslTacotronModel(&work, "T", params);

    tacotron::AddTacotronLoss(
        &work,
        "total_loss",
        { { "decoder_output", "mel_outputs", "duration_outputs", "mel_targets", "targets_lengths", "duration_targets" }, { "loss", "before_loss", "after_loss", "duration_loss" } },
        params);

    TENSORS_CREATE(BATCH_SIZE);

    ASSERT_EQ(work.getTrainableParameterNames().size(), GOLDEN_TRAINABLE_PARAMS);

    auto& memory_manager = work.getMemoryManager();
    memory_manager.createTensor("golden_loss", 1, 1, 1, 4, 0_dt);

    // Data loading

    size_t loaded = loadTacotronParams((testPath / "weights" / "vctk_small_250_Tacotron_model.").string(), memory_manager, "T", params, true, false);
    ASSERT_EQ(loaded, GOLDEN_PARAMS);

    EXPECT_TRUE(loadTFData(testPath / "encoder_output.data", memory_manager["encoder_output"]));
    EXPECT_TRUE(loadTFData(testPath / "input_lengths.data", memory_manager["input_lengths"]));
    EXPECT_TRUE(loadTFData(testPath / "inputs.data", memory_manager["inputs"]));
    EXPECT_TRUE(loadTFData(testPath / "duration_targets.data", memory_manager["duration_targets"]));
    EXPECT_TRUE(loadTFData(testPath / "targets_lengths.data", memory_manager["targets_lengths"]));
    EXPECT_TRUE(loadTFData(testPath / "mel_targets.data", memory_manager["mel_targets"]));
    EXPECT_TRUE(loadTFData(testPath / "speaker_embedding.data", memory_manager["speaker_embedding"]));

    EXPECT_TRUE(loadTFData(testPath / "duration_predictor" / "golden_durations.data", memory_manager["golden_duration_outputs"]));
    EXPECT_TRUE(loadTFData(testPath / "mel_outputs.data", memory_manager["golden_mel_outputs"]));
    EXPECT_TRUE(loadTFData(testPath / "decoder_output.data", memory_manager["golden_decoder_output"]));
    EXPECT_TRUE(loadTFData(testPath / "loss.data", memory_manager["golden_loss"]));

    auto idealGradNorms = loadNamedValues(testPath / "grad_norm.data", "T", params);
    EXPECT_EQ(idealGradNorms.size(), GOLDEN_TRAINABLE_PARAMS);

    TensorChecker checker(
        {
            { "duration_outputs", "golden_duration_outputs" },
            { "decoder_output", "golden_decoder_output" },
            { "mel_outputs", "golden_mel_outputs" },
        },
        EPS,
        -1_dt,
        true);
    //work.getNetworkParameters().mCallback = debugCallback<decltype(memory_manager)>();
        
        //checker;
    
    auto timeStart = chrono::steady_clock::now();
    ASSERT_NO_THROW(work.forwardPassTraining());
    auto elapsed = TODTYPE(chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - timeStart).count());
    cout << "Forward time " << elapsed << " ms" << endl;

    auto& loss = memory_manager["loss"];

    ASSERT_EQ(loss.size(), 1u);
    ASSERT_EQ(memory_manager["before_loss"].size(), 1u);
    ASSERT_EQ(memory_manager["after_loss"].size(), 1u);
    ASSERT_EQ(memory_manager["duration_loss"].size(), 1u);

    auto bloss = memory_manager["before_loss"][0];
    auto aloss = memory_manager["after_loss"][0];
    auto dloss = memory_manager["duration_loss"][0];

    EXPECT_NEAR(loss[0], memory_manager["golden_loss"][0], LOSS_EPS);
    EXPECT_NEAR(bloss, memory_manager["golden_loss"][1], LOSS_EPS);
    EXPECT_NEAR(aloss, memory_manager["golden_loss"][2], LOSS_EPS);
    EXPECT_NEAR(dloss, memory_manager["golden_loss"][3], LOSS_EPS);

    cout << "forward ok" << endl;

    ASSERT_NO_THROW(work.backwardPassTraining());
    cout << "backward ok" << endl;

    auto trainableParams = work.getTrainableParameters();
    postprocessing::GradientClipping clip;
    auto gnorm = clip.calcGlobalNorm(trainableParams, work.getNetworkParameters());
    cout << "Global norm: " << gnorm << endl;
    dtype totalGradNormDiff = 0;
    for (size_t j = 0; j < trainableParams.size(); ++j)
    {
        auto& [param, grad] = trainableParams[j];
        cout << param.getName() << ": " << TensorNorm(grad) << "( " << idealGradNorms[param.getName()] << ")" << endl;

        cout << "  diff: " << fabs(TensorNorm(grad) - idealGradNorms[param.getName()]) << std::endl;
        totalGradNormDiff += fabs(TensorNorm(grad) - idealGradNorms[param.getName()]);
    }
    cout << "Total Grad Norm Diff: " << totalGradNormDiff << std::endl;

    EXPECT_TRUE(totalGradNormDiff < 0.5_dt);
}

TEST(TestTacotronMsk, P225GpuUnit)
{
    PROFILE_TEST

    GPU_ONLY_TEST

    const bool usePool = UT::tools::getArg("use_pool", true);
    cout << "Using pool: " << (usePool ? "yes" : "no") << endl;

    using namespace tacotron;

    auto testPath = tools::getTestAssetsDir() / "Tacotron_Msk_RP";

    constexpr size_t GOLDEN_PARAMS = 64;
    constexpr size_t GOLDEN_TRAINABLE_PARAMS = 54;

    WorkflowEager work(CompressionMode::NONE, CalculationMode::DETERMINISTIC, usePool ? AllocationMode::POOL : AllocationMode::STANDARD, ExecutionTarget::GPU);
    auto& memory_manager = work.getMemoryManager<MemoryManagerGPU>();
    constexpr size_t T_IN = 86;
    constexpr size_t BATCH_SIZE = 12;
    constexpr dtype LOSS_EPS = 1e-4_dt;
    constexpr dtype EPS = 4e-3_dt; // intermediate values on android can have rather big difference

    TacotronParams params({ "encoder_output", "speaker_embedding", "duration_targets", "inputs", "input_lengths", "mel_targets" }, { "decoder_output", "mel_outputs", "duration_outputs" }, {});
    params.maxMelFrames = 666;
    params.useDurationPrediction = true;
    params.attentionType = "";
    params.durationPredictorLstmUnits = 256;
    params.postnetKernelSize = { 5, 5, 5, 5, 3 };
    params.prenetLayers = { 256, 256 };
    params.dropoutRate = 0.f;
    params.zoneoutRate = 0.f;
    params.postnetDropoutRate = 0.f;
    params.trainableSpeakerEmbedding = false;
    params.withoutStopTokenLoss = true;
    params.lossMultipliers = { 1.f, 1.f, 2.f };
    params.useResidualRnn = true;

    work.add<DataLayer>("inputs", DataParams{ { "inputs" }, 1u, T_IN, 1u });
    work.add<DataLayer>("duration_targets", DataParams{ { "duration_targets" }, 1u, 1u, T_IN });
    work.add<DataLayer>("encoder_output", DataParams{ { "encoder_output" }, 1u, T_IN, params.embeddingDim });
    work.add<DataLayer>("input_lengths", DataParams{ { "input_lengths" }, 1u, 1u, 1u });
    work.add<DataLayer>("mel_targets", DataParams{ { "mel_targets" }, 1u, params.maxMelFrames, params.numMels });
    work.add<DataLayer>("targets_lengths", DataParams{ { "targets_lengths" }, 1u, 1u, 1u });
    work.add<DataLayer>("speaker_embedding", DataParams{ { "speaker_embedding" }, 1u, 1u, params.speakerEmbeddingSize });

    work.add<DataLayer>("golden_decoder_output", DataParams{ { "golden_decoder_output" }, 1u, params.maxMelFrames, params.numMels });
    work.add<DataLayer>("golden_mel_outputs", DataParams{ { "golden_mel_outputs" }, 1u, params.maxMelFrames, params.numMels });
    work.add<DataLayer>("golden_duration_outputs", DataParams{ { "golden_duration_outputs" }, 1u, T_IN, 1u });
    memory_manager.createTensor("golden_loss", 1, 1, 1, 4, 0_dt);
    
    AddSingleSpeakerFslTacotronModel(&work, "T", params);

    tacotron::AddTacotronLoss(
        &work,
        "total_loss",
        { { "decoder_output", "mel_outputs", "duration_outputs", "mel_targets", "targets_lengths", "duration_targets" }, { "loss", "before_loss", "after_loss", "duration_loss" } },
        params);

    TENSORS_CREATE(BATCH_SIZE);

    ASSERT_EQ(work.getTrainableParameterNames().size(), GOLDEN_TRAINABLE_PARAMS);

    // Data loading

    size_t loaded = loadTacotronParams((testPath / "weights" / "vctk_small_250_Tacotron_model.").string(), memory_manager, "T", params, true, false);
    ASSERT_EQ(loaded, GOLDEN_PARAMS);

    EXPECT_TRUE(loadTFData(testPath / "encoder_output.data", memory_manager["encoder_output"]));
    EXPECT_TRUE(loadTFData(testPath / "input_lengths.data", memory_manager["input_lengths"]));
    EXPECT_TRUE(loadTFData(testPath / "inputs.data", memory_manager["inputs"]));
    EXPECT_TRUE(loadTFData(testPath / "duration_targets.data", memory_manager["duration_targets"]));
    EXPECT_TRUE(loadTFData(testPath / "targets_lengths.data", memory_manager["targets_lengths"]));
    EXPECT_TRUE(loadTFData(testPath / "mel_targets.data", memory_manager["mel_targets"]));
    EXPECT_TRUE(loadTFData(testPath / "speaker_embedding.data", memory_manager["speaker_embedding"]));

    EXPECT_TRUE(loadTFData(testPath / "duration_predictor" / "golden_durations.data", memory_manager["golden_duration_outputs"]));
    EXPECT_TRUE(loadTFData(testPath / "mel_outputs.data", memory_manager["golden_mel_outputs"]));
    EXPECT_TRUE(loadTFData(testPath / "decoder_output.data", memory_manager["golden_decoder_output"]));
    EXPECT_TRUE(loadTFData(testPath / "loss.data", memory_manager["golden_loss"]));

    auto idealGradNorms = loadNamedValues(testPath / "grad_norm.data", "T", params);
    EXPECT_EQ(idealGradNorms.size(), GOLDEN_TRAINABLE_PARAMS);

    //work.getNetworkParameters().mCallbackGPU = debugCallback<decltype(memory_manager)>();

    ASSERT_NO_THROW(work.forwardPassTraining());

    vector<pair<Name, Name>> checks = {
        { "duration_outputs", "golden_duration_outputs" },
        { "decoder_output", "golden_decoder_output" },
        { "mel_outputs", "golden_mel_outputs" },
    };

    work.getGpuCommandQueue().finish();
    auto timeStart = chrono::steady_clock::now();
    ASSERT_NO_THROW(work.forwardPassTraining());
    work.getGpuCommandQueue().finish();
    auto elapsed = TODTYPE(chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - timeStart).count());
    cout << "Forward time " << elapsed << " ms" << endl;

    Tensor loss = memory_manager["loss"];
    Tensor before_loss = memory_manager["before_loss"];
    Tensor after_loss = memory_manager["after_loss"];
    Tensor duration_loss = memory_manager["duration_loss"];
    Tensor golden_loss = memory_manager["golden_loss"];

    ASSERT_EQ(loss.size(), 1u);
    ASSERT_EQ(before_loss.size(), 1u);
    ASSERT_EQ(after_loss.size(), 1u);
    ASSERT_EQ(duration_loss.size(), 1u);

    auto bloss = before_loss[0];
    auto aloss = after_loss[0];
    auto dloss = duration_loss[0];

    EXPECT_NEAR(loss[0], golden_loss[0], LOSS_EPS);
    EXPECT_NEAR(bloss, golden_loss[1], LOSS_EPS);
    EXPECT_NEAR(aloss, golden_loss[2], LOSS_EPS);
    EXPECT_NEAR(dloss, golden_loss[3], LOSS_EPS);

    cout << "Expected loss: " << golden_loss[0] << " (before: " << golden_loss[1] << ", after: " << golden_loss[2] << ", duration: " << golden_loss[3] << ")" << endl;
    cout << "Calculated loss: " << loss[0] << " (before: " << bloss << ", after: " << aloss << ", duration: " << dloss << ")" << endl;

    UT::tools::checkTensors(checks, memory_manager, EPS);

    cout << "forward ok" << endl;

    ASSERT_NO_THROW(work.backwardPassTraining());
    cout << "backward ok" << endl;

    auto trainableParams = work.getTrainableParameters<MemoryManagerGPU>();
    postprocessing::GradientClipping clip;
    auto gnorm = clip.calcGlobalNorm(trainableParams, work.getNetworkParameters());
    cout << "Global norm: " << gnorm << endl;
    dtype totalGradNormDiff = 0;
    for (const auto& name : work.getTrainableParameterNames())
    {
        Tensor grad = memory_manager[name.grad()];
        cout << name << ": " << TensorNorm(grad) << "( " << idealGradNorms[name] << ")" << endl;

        cout << "  diff: " << fabs(TensorNorm(grad) - idealGradNorms[name]) << std::endl;
        totalGradNormDiff += fabs(TensorNorm(grad) - idealGradNorms[name]);
    }
    cout << "Total Grad Norm Diff: " << totalGradNormDiff << std::endl;

    EXPECT_TRUE(totalGradNormDiff < 0.5_dt);
}

TEST(TestTacotronMsk, DummyGpuUnit)
{
    PROFILE_TEST

    GPU_ONLY_TEST

    using namespace tacotron;

    auto testPath = tools::getTestAssetsDir() / "Tacotron_Msk_RP";

    constexpr size_t GOLDEN_PARAMS = 64;
    constexpr size_t GOLDEN_TRAINABLE_PARAMS = 54;

    WorkflowEager work(CompressionMode::NONE, CalculationMode::DETERMINISTIC, AllocationMode::POOL, ExecutionTarget::GPU);
    constexpr size_t T_IN = 32;
    constexpr size_t BATCH_SIZE = 2;

    TacotronParams params({ "encoder_output", "speaker_embedding", "duration_targets", "inputs", "input_lengths", "mel_targets" }, { "decoder_output", "mel_outputs", "duration_outputs" }, {});
    params.maxMelFrames = 120;
    params.useDurationPrediction = true;
    params.attentionType = "";
    params.durationPredictorLstmUnits = 256;
    params.postnetKernelSize = { 5, 5, 5, 5, 3 };
    params.prenetLayers = { 256, 256 };
    params.dropoutRate = 0.f;
    params.zoneoutRate = 0.f;
    params.postnetDropoutRate = 0.f;
    params.trainableSpeakerEmbedding = false;
    params.withoutStopTokenLoss = true;
    params.lossMultipliers = { 1.f, 1.f, 2.f };
    params.useResidualRnn = false;

    work.add<DataLayer>("inputs", DataParams{ { "inputs" }, 1u, T_IN, 1u });
    work.add<DataLayer>("duration_targets", DataParams{ { "duration_targets" }, 1u, 1u, T_IN });
    work.add<DataLayer>("encoder_output", DataParams{ { "encoder_output" }, 1u, T_IN, params.embeddingDim });
    work.add<DataLayer>("input_lengths", DataParams{ { "input_lengths" }, 1u, 1u, 1u });
    work.add<DataLayer>("mel_targets", DataParams{ { "mel_targets" }, 1u, params.maxMelFrames, params.numMels });
    work.add<DataLayer>("targets_lengths", DataParams{ { "targets_lengths" }, 1u, 1u, 1u });
    work.add<DataLayer>("speaker_embedding", DataParams{ { "speaker_embedding" }, 1u, 1u, params.speakerEmbeddingSize });

    work.add<DataLayer>("golden_decoder_output", DataParams{ { "golden_decoder_output" }, 1u, params.maxMelFrames, params.numMels });
    work.add<DataLayer>("golden_mel_outputs", DataParams{ { "golden_mel_outputs" }, 1u, params.maxMelFrames, params.numMels });
    work.add<DataLayer>("golden_duration_outputs", DataParams{ { "golden_duration_outputs" }, 1u, T_IN, 1u });
    work.add<TensorLayer>("golden_loss", TensorParams{ { "golden_loss" }, WShape{ 1, 1, 1, 4 }, 0_dt });

    AddSingleSpeakerFslTacotronModel(&work, "T", params);

    tacotron::AddTacotronLoss(
        &work,
        "total_loss",
        { { "decoder_output", "mel_outputs", "duration_outputs", "mel_targets", "targets_lengths", "duration_targets" }, { "loss", "before_loss", "after_loss", "duration_loss" } },
        params);

    auto paramsCount = tacotronParamsCount(params);
    ASSERT_EQ(GOLDEN_TRAINABLE_PARAMS, paramsCount.first);
    ASSERT_EQ(GOLDEN_PARAMS, paramsCount.second);

    TENSORS_CREATE(BATCH_SIZE);

    ASSERT_EQ(work.getTrainableParameterNames().size(), GOLDEN_TRAINABLE_PARAMS);

    ASSERT_NO_THROW(work.forwardPassTraining());

    cout << "forward ok" << endl;

    ASSERT_NO_THROW(work.backwardPassTraining());
    cout << "backward ok" << endl;
}

TEST(TestTacotronMsk, SamarinTrainableEmbeddingUnit)
{
    PROFILE_TEST
    using namespace tacotron;
    using namespace optimizers;

    auto testPath = tools::getTestAssetsDir() / "Tacotron_Msk_RP";

    constexpr size_t GOLDEN_PARAMS = 59;
    constexpr size_t GOLDEN_TRAINABLE_PARAMS = 51;

    bool usePool = true;

    Workflow work(CompressionMode::NONE, CalculationMode::DETERMINISTIC, usePool ? AllocationMode::POOL : AllocationMode::STANDARD);

    constexpr dtype ADAM_BETA1 = 0.9_dt;
    constexpr dtype ADAM_BETA2 = 0.999_dt;
    constexpr dtype ADAM_EPSILON = 1e-6_dt;

    string experimentName = "samarin_new";
    auto weightsPrefix = filesystem::path("weights_250000") / "250000_Tacotron_model.";

    size_t T_IN = 68;
    size_t BATCH_SIZE = 11;

    TacotronParams params({ "encoder_output", "duration_targets", "inputs", "input_lengths", "mel_targets" },
                          { "decoder_output", "mel_outputs", "duration_outputs" },
                          {}, LayerExecutionTarget::Default);
    params.maxMelFrames = 564;
    params.useDurationPrediction = true;
    params.attentionType = "None";
    params.durationPredictorLstmUnits = 256;
    params.postnetKernelSize = { 5, 3, 3, 3 };
    params.postnetChannels = 256;
    params.prenetLayers = { 128, 128 };
    params.dropoutRate = 0.0f;
    params.zoneoutRate = 0.0f;
    params.postnetDropoutRate = 0.0f;
    params.trainableSpeakerEmbedding = true;
    params.withoutStopTokenLoss = true;
    params.lossMultipliers = { 1.f, 1.f, 2.f };
    params.useResidualRnn = true;

    TacotronTrainingParams trainParams;
    trainParams.initialLearningRate = 1e-3_dt;
    trainParams.clipNorm = 1.0_dt;
    trainParams.decayLearningRate = true;
    trainParams.decaySteps = 500;
    trainParams.decayStartStep = 100;
    trainParams.decayRate = 0.5;
    trainParams.finalLearningRate = 1e-5_dt;
    trainParams.warmupLearningRate = true;
    trainParams.warmupSteps = 1000;

    size_t nEpoch = 5;

    work.add<DataLayer>("inputs", DataParams{ { "inputs" }, 1u, T_IN, 1u });
    work.add<DataLayer>("duration_targets", DataParams{ { "duration_targets" }, 1u, 1u, T_IN });
    work.add<DataLayer>("encoder_output", DataParams{ { "encoder_output" }, 1u, T_IN, params.embeddingDim });
    work.add<DataLayer>("input_lengths", DataParams{ { "input_lengths" }, 1u, 1u, 1u });
    work.add<DataLayer>("mel_targets", DataParams{ { "mel_targets" }, 1u, params.maxMelFrames, params.numMels });
    work.add<DataLayer>("targets_lengths", DataParams{ { "targets_lengths" }, 1u, 1u, 1u });

    AddSingleSpeakerFslTacotronModel(&work, "T", params);

    tacotron::AddTacotronLoss(
        &work,
        "total_loss",
        { { "decoder_output", "mel_outputs", "duration_outputs", "mel_targets", "targets_lengths", "duration_targets" }, { "loss", "before_loss", "after_loss", "duration_loss" } },
        params);

    auto paramsCount = tacotronParamsCount(params);

    ASSERT_EQ(paramsCount.first, GOLDEN_TRAINABLE_PARAMS);
    ASSERT_EQ(paramsCount.second, GOLDEN_PARAMS);

    auto timeStart = chrono::steady_clock::now();
    TENSORS_CREATE(BATCH_SIZE);
    auto elapsed = TODTYPE(chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - timeStart).count());
    cout << "Workflow preparation time " << elapsed << " ms for batch size " << BATCH_SIZE << endl;

    ASSERT_EQ(work.getTrainableParameterNames().size(), GOLDEN_TRAINABLE_PARAMS);
    cout << tools::get_size_of_trainable_params_mixed_precision(work) << " trainable weights" << endl;

    auto& memory_manager = work.getMemoryManager();
    postprocessing::GradientClipping clip(1_dt);

    // Data loading

    timeStart = chrono::steady_clock::now();
    size_t loaded = loadTacotronParams((testPath / weightsPrefix).string(), memory_manager, "T", params, true, false);
    ASSERT_EQ(loaded, GOLDEN_PARAMS - 1); // speaker embedding will be loaded separately
    ASSERT_TRUE(loadTFData(testPath / experimentName / "speaker_embedding.data", memory_manager[Name("T") / "speaker_embedding"]));
    elapsed = TODTYPE(chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - timeStart).count());
    cout << "Weights loading time " << elapsed << " ms for batch size " << BATCH_SIZE << endl;

    timeStart = chrono::steady_clock::now();
    ASSERT_TRUE(loadTFData(testPath / experimentName / "encoder_output.data", memory_manager["encoder_output"]));
    ASSERT_TRUE(loadTFData(testPath / experimentName / "input_lengths.data", memory_manager["input_lengths"]));
    ASSERT_TRUE(loadTFData(testPath / experimentName / "inputs.data", memory_manager["inputs"]));
    ASSERT_TRUE(loadTFData(testPath / experimentName / "duration_targets.data", memory_manager["duration_targets"]));
    ASSERT_TRUE(loadTFData(testPath / experimentName / "targets_lengths.data", memory_manager["targets_lengths"]));
    ASSERT_TRUE(loadTFData(testPath / experimentName / "mel_targets.data", memory_manager["mel_targets"]));
    elapsed = TODTYPE(chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - timeStart).count());
    cout << "Training data loading time " << elapsed << " ms for batch size " << BATCH_SIZE << endl;

    auto optimizer = createOptimizer(trainParams, ADAM_BETA1, ADAM_BETA2, ADAM_EPSILON);

    dtype fastestEpochTime = numeric_limits<dtype>::max();
    dtype slowestEpochRime = 0_dt;
    dtype trainingTime = 0;
    //work.getNetworkParameters().mCallback = debugCallback<MemoryManager>();
    for (size_t epoch = 0; epoch < nEpoch; ++epoch)
    {
        cout << "Epoch " << epoch << endl;
        timeStart = chrono::steady_clock::now();
        work.forwardPassTraining();

        auto gnorm_before = 1_dt;
        auto gnorm_after = 1_dt;
        if (epoch != nEpoch - 1)
        {
            work.backwardPassTraining();
            auto trainableParams = work.getTrainableParameters();
            clip.processGradients(trainableParams, work.getNetworkParameters());
            gnorm_before = clip.getGlobalNorm(work.getNetworkParameters());
            gnorm_after = clip.calcGlobalNorm(trainableParams, work.getNetworkParameters());

            if (trainParams.decayLearningRate)
            {
                optimizer->step();
            }

            for (auto p : trainableParams)
            {
                optimizer->operator()(memory_manager, p.Param, p.Gradient);
            }
        }

        auto& lossValue = memory_manager["loss"];
        elapsed = TODTYPE(chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - timeStart).count());

        trainingTime += elapsed;

        fastestEpochTime = std::min(fastestEpochTime, elapsed);
        slowestEpochRime = std::max(slowestEpochRime, elapsed);

        cout << "  Epoch time " << elapsed << " ms for batch size " << BATCH_SIZE << endl;
        cout << "  Average performance: " << trainingTime / TODTYPE(epoch + 1) << " ms/epoch (fastest: " << fastestEpochTime << ", slowest " << slowestEpochRime << ")" << endl << endl;
        cout << "  Loss = " << lossValue[0] << " (before = " << memory_manager["before_loss"][0] << ", after = " << memory_manager["after_loss"][0]
             << ", duration = " << memory_manager["duration_loss"][0] << ")" << endl;

        if (epoch != nEpoch - 1)
        {
            cout << "  Global grad norm: " << gnorm_before << " -> " << gnorm_after << endl;
        }
    }
}

TEST(TestTacotronMsk, DISABLED_SamarinTrainableEmbeddingGpuMeasureTime)
{
    PROFILE_TEST

    GPU_ONLY_TEST

    using namespace tacotron;
    using namespace optimizers;

    auto testPath = tools::getTestAssetsDir() / "Tacotron_Msk_RP";

    constexpr size_t GOLDEN_PARAMS = 59;
    constexpr size_t GOLDEN_TRAINABLE_PARAMS = 51;

    const bool usePool = UT::tools::getArg("use_pool", true);
    cout << "Using pool: " << (usePool ? "yes" : "no") << endl;

    const string kernelPolicy = UT::tools::getArg("kernel_policy", "default");
    KernelExecutionPolicy policy = KernelExecutionPolicy::DefaultParams;
    if (kernelPolicy == "skip_all")
    {
        policy = KernelExecutionPolicy::SkipAll;
    }
    else if (kernelPolicy == "skip_kernels")
    {
        policy = KernelExecutionPolicy::SkipKernels;
    }
    cout << "Kernel policy: " << kernelPolicy << endl;

    const string kernelFilter = UT::tools::getArg("kernel_filter", "");
    auto filters = Common::split(kernelFilter, ';');

    Workflow work(CompressionMode::NONE, CalculationMode::DETERMINISTIC, usePool ? AllocationMode::POOL : AllocationMode::STANDARD, ExecutionTarget::GPU);
    work.getKernelManager().setExecutionPolicy(policy);
    work.getKernelManager().setKernelFilter(filters);
    constexpr dtype ADAM_BETA1 = 0.9_dt;
    constexpr dtype ADAM_BETA2 = 0.999_dt;
    constexpr dtype ADAM_EPSILON = 1e-6_dt;

    string experimentName = "samarin_new";
    auto weightsPrefix = filesystem::path("weights_250000") / "250000_Tacotron_model.";

    size_t T_IN = 68;
    size_t BATCH_SIZE = 11;

    TacotronParams params({ "encoder_output", "duration_targets", "inputs", "input_lengths", "mel_targets" },
                          { "decoder_output", "mel_outputs", "duration_outputs" },
                          {}, LayerExecutionTarget::Default);
    params.maxMelFrames = 564;
    params.useDurationPrediction = true;
    params.attentionType = "None";
    params.durationPredictorLstmUnits = 256;
    params.postnetKernelSize = { 5, 3, 3, 3 };
    params.postnetChannels = 256;
    params.prenetLayers = { 128, 128 };
    params.dropoutRate = 0.0f;
    params.zoneoutRate = 0.0f;
    params.postnetDropoutRate = 0.0f;
    params.trainableSpeakerEmbedding = true;
    params.withoutStopTokenLoss = true;
    params.lossMultipliers = { 1.f, 1.f, 2.f };
    params.useResidualRnn = true;

    TacotronTrainingParams trainParams;
    trainParams.initialLearningRate = 1e-3_dt;
    trainParams.clipNorm = 1.0_dt;
    trainParams.decayLearningRate = true;
    trainParams.decaySteps = 500;
    trainParams.decayStartStep = 100;
    trainParams.decayRate = 0.5;
    trainParams.finalLearningRate = 1e-5_dt;
    trainParams.warmupLearningRate = true;
    trainParams.warmupSteps = 1000;

    work.add<DataLayer>("inputs", DataParams{ { "inputs" }, 1u, T_IN, 1u });
    work.add<DataLayer>("duration_targets", DataParams{ { "duration_targets" }, 1u, 1u, T_IN });
    work.add<DataLayer>("encoder_output", DataParams{ { "encoder_output" }, 1u, T_IN, params.embeddingDim });
    work.add<DataLayer>("input_lengths", DataParams{ { "input_lengths" }, 1u, 1u, 1u });
    work.add<DataLayer>("mel_targets", DataParams{ { "mel_targets" }, 1u, params.maxMelFrames, params.numMels });
    work.add<DataLayer>("targets_lengths", DataParams{ { "targets_lengths" }, 1u, 1u, 1u });

    AddSingleSpeakerFslTacotronModel(&work, "T", params);

    tacotron::AddTacotronLoss(
        &work,
        "total_loss",
        { { "decoder_output", "mel_outputs", "duration_outputs", "mel_targets", "targets_lengths", "duration_targets" }, { "loss", "before_loss", "after_loss", "duration_loss" } },
        params);

    auto paramsCount = tacotronParamsCount(params);

    ASSERT_EQ(paramsCount.first, GOLDEN_TRAINABLE_PARAMS);
    ASSERT_EQ(paramsCount.second, GOLDEN_PARAMS);

    auto timeStart = chrono::steady_clock::now();
    TENSORS_CREATE(BATCH_SIZE);
    auto elapsed = TODTYPE(chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - timeStart).count());
    cout << "Workflow preparation time " << elapsed << " ms for batch size " << BATCH_SIZE << endl;

    auto& memory_manager = work.getMemoryManager<MemoryManagerGPU>();
    postprocessing::GradientClipping clip(1_dt);

    // Data loading

    timeStart = chrono::steady_clock::now();
    size_t loaded = loadTacotronParams((testPath / weightsPrefix).string(), memory_manager, "T", params, true, false);
    ASSERT_EQ(loaded, GOLDEN_PARAMS - 1); // speaker embedding will be loaded separately
    ASSERT_TRUE(loadTFData(testPath / experimentName / "speaker_embedding.data", memory_manager[Name("T") / "speaker_embedding"]));
    elapsed = TODTYPE(chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - timeStart).count());
    cout << "Weights loading time " << elapsed << " ms for batch size " << BATCH_SIZE << endl;

    timeStart = chrono::steady_clock::now();
    ASSERT_TRUE(loadTFData(testPath / experimentName / "encoder_output.data", memory_manager["encoder_output"]));
    ASSERT_TRUE(loadTFData(testPath / experimentName / "input_lengths.data", memory_manager["input_lengths"]));
    ASSERT_TRUE(loadTFData(testPath / experimentName / "inputs.data", memory_manager["inputs"]));
    ASSERT_TRUE(loadTFData(testPath / experimentName / "duration_targets.data", memory_manager["duration_targets"]));
    ASSERT_TRUE(loadTFData(testPath / experimentName / "targets_lengths.data", memory_manager["targets_lengths"]));
    ASSERT_TRUE(loadTFData(testPath / experimentName / "mel_targets.data", memory_manager["mel_targets"]));
    elapsed = TODTYPE(chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - timeStart).count());
    cout << "Training data loading time " << elapsed << " ms for batch size " << BATCH_SIZE << endl;

    auto optimizer = createOptimizer(trainParams, ADAM_BETA1, ADAM_BETA2, ADAM_EPSILON);
    LayerTypeStatisticsGPU statCallback;
    // 0 - warmup, 1 - measure overall time, 2 - measure individual layers

    dtype forwardTime = 0;
    dtype backwardTime = 0;
    dtype clippingTime = 0;
    dtype optimizerTime = 0;

    for (size_t iter = 0; iter < 3; ++iter)
    {
        cout << "Iter " << iter + 1;
        switch (iter)
        {
            case 0:
                cout << " warmup" << endl;
                break;
            case 1:
                cout << " measure overall time" << endl;
                break;
            case 2:
                cout << " measure individual layers" << endl;
                break;
        }
        if (iter == 2)
        {
            using namespace std::placeholders;
            work.getNetworkParameters().mCallbackGPU = std::bind(&LayerTypeStatisticsGPU::operator(), &statCallback, _1, _2, _3);
        }
        timeStart = chrono::steady_clock::now();
        
        work.forwardPassTraining();
        if (iter == 1)
        {
            work.getGpuCommandQueue().finish();
            forwardTime = TODTYPE(chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - timeStart).count());
            timeStart = chrono::steady_clock::now();
        }

        work.backwardPassTraining();

        if (iter == 1)
        {
            work.getGpuCommandQueue().finish();
            backwardTime = TODTYPE(chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - timeStart).count());
        }

        auto trainableParams = work.getTrainableParameters<MemoryManagerGPU>();

        if (iter == 1)
        {
            timeStart = chrono::steady_clock::now();
        }
        clip.processGradients(trainableParams, work.getNetworkParameters());
        if (iter == 1)
        {
            work.getGpuCommandQueue().finish();
            clippingTime = TODTYPE(chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - timeStart).count());
        }

        if (trainParams.decayLearningRate)
        {
            optimizer->step();
        }

        if (iter == 1)
        {
            timeStart = chrono::steady_clock::now();
        }
        for (auto p : trainableParams)
        {
            optimizer->operator()(work.getKernelManager(), memory_manager, p.Param, p.Gradient);
        }
        if (iter == 1)
        {
            work.getGpuCommandQueue().finish();
            optimizerTime = TODTYPE(chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - timeStart).count());
        }
        work.getGpuCommandQueue().finish();
    }

    statCallback.print(forwardTime, backwardTime, 10);
    cout << "Forward time: " << forwardTime << endl;
    cout << "Backward time: " << backwardTime << endl;
    cout << "Clipping time: " << clippingTime << endl;
    cout << "Optimizer time: " << optimizerTime << endl;
}

TEST(TestTacotronMsk, DISABLED_SamarinTrainableEmbeddingCpuMeasureTime)
{
    PROFILE_TEST

    using namespace tacotron;
    using namespace optimizers;

    auto testPath = tools::getTestAssetsDir() / "Tacotron_Msk_RP";

    constexpr size_t GOLDEN_PARAMS = 59;
    constexpr size_t GOLDEN_TRAINABLE_PARAMS = 51;

    const bool usePool = UT::tools::getArg("use_pool", true);
    cout << "Using pool: " << (usePool ? "yes" : "no") << endl;

    Workflow work(CompressionMode::NONE, CalculationMode::DETERMINISTIC, usePool ? AllocationMode::POOL : AllocationMode::STANDARD, ExecutionTarget::CPU);

    constexpr dtype ADAM_BETA1 = 0.9_dt;
    constexpr dtype ADAM_BETA2 = 0.999_dt;
    constexpr dtype ADAM_EPSILON = 1e-6_dt;

    string experimentName = "samarin_new";
    auto weightsPrefix = filesystem::path("weights_250000") / "250000_Tacotron_model.";

    size_t T_IN = 68;
    size_t BATCH_SIZE = 11;

    TacotronParams params({ "encoder_output", "duration_targets", "inputs", "input_lengths", "mel_targets" },
                          { "decoder_output", "mel_outputs", "duration_outputs" },
                          {}, LayerExecutionTarget::Default);
    params.maxMelFrames = 564;
    params.useDurationPrediction = true;
    params.attentionType = "None";
    params.durationPredictorLstmUnits = 256;
    params.postnetKernelSize = { 5, 3, 3, 3 };
    params.postnetChannels = 256;
    params.prenetLayers = { 128, 128 };
    params.dropoutRate = 0.0f;
    params.zoneoutRate = 0.0f;
    params.postnetDropoutRate = 0.0f;
    params.trainableSpeakerEmbedding = true;
    params.withoutStopTokenLoss = true;
    params.lossMultipliers = { 1.f, 1.f, 2.f };
    params.useResidualRnn = true;

    TacotronTrainingParams trainParams;
    trainParams.initialLearningRate = 1e-3_dt;
    trainParams.clipNorm = 1.0_dt;
    trainParams.decayLearningRate = true;
    trainParams.decaySteps = 500;
    trainParams.decayStartStep = 100;
    trainParams.decayRate = 0.5;
    trainParams.finalLearningRate = 1e-5_dt;
    trainParams.warmupLearningRate = true;
    trainParams.warmupSteps = 1000;

    work.add<DataLayer>("inputs", DataParams{ { "inputs" }, 1u, T_IN, 1u });
    work.add<DataLayer>("duration_targets", DataParams{ { "duration_targets" }, 1u, 1u, T_IN });
    work.add<DataLayer>("encoder_output", DataParams{ { "encoder_output" }, 1u, T_IN, params.embeddingDim });
    work.add<DataLayer>("input_lengths", DataParams{ { "input_lengths" }, 1u, 1u, 1u });
    work.add<DataLayer>("mel_targets", DataParams{ { "mel_targets" }, 1u, params.maxMelFrames, params.numMels });
    work.add<DataLayer>("targets_lengths", DataParams{ { "targets_lengths" }, 1u, 1u, 1u });

    AddSingleSpeakerFslTacotronModel(&work, "T", params);

    tacotron::AddTacotronLoss(
        &work,
        "total_loss",
        { { "decoder_output", "mel_outputs", "duration_outputs", "mel_targets", "targets_lengths", "duration_targets" }, { "loss", "before_loss", "after_loss", "duration_loss" } },
        params);

    auto paramsCount = tacotronParamsCount(params);

    ASSERT_EQ(paramsCount.first, GOLDEN_TRAINABLE_PARAMS);
    ASSERT_EQ(paramsCount.second, GOLDEN_PARAMS);

    auto timeStart = chrono::steady_clock::now();
    TENSORS_CREATE(BATCH_SIZE);
    auto elapsed = TODTYPE(chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - timeStart).count());
    cout << "Workflow preparation time " << elapsed << " ms for batch size " << BATCH_SIZE << endl;

    auto& memory_manager = work.getMemoryManager();
    postprocessing::GradientClipping clip(1_dt);

    // Data loading

    timeStart = chrono::steady_clock::now();
    size_t loaded = loadTacotronParams((testPath / weightsPrefix).string(), memory_manager, "T", params, true, false);
    ASSERT_EQ(loaded, GOLDEN_PARAMS - 1); // speaker embedding will be loaded separately
    ASSERT_TRUE(loadTFData(testPath / experimentName / "speaker_embedding.data", memory_manager[Name("T") / "speaker_embedding"]));
    elapsed = TODTYPE(chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - timeStart).count());
    cout << "Weights loading time " << elapsed << " ms for batch size " << BATCH_SIZE << endl;

    timeStart = chrono::steady_clock::now();
    ASSERT_TRUE(loadTFData(testPath / experimentName / "encoder_output.data", memory_manager["encoder_output"]));
    ASSERT_TRUE(loadTFData(testPath / experimentName / "input_lengths.data", memory_manager["input_lengths"]));
    ASSERT_TRUE(loadTFData(testPath / experimentName / "inputs.data", memory_manager["inputs"]));
    ASSERT_TRUE(loadTFData(testPath / experimentName / "duration_targets.data", memory_manager["duration_targets"]));
    ASSERT_TRUE(loadTFData(testPath / experimentName / "targets_lengths.data", memory_manager["targets_lengths"]));
    ASSERT_TRUE(loadTFData(testPath / experimentName / "mel_targets.data", memory_manager["mel_targets"]));
    elapsed = TODTYPE(chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - timeStart).count());
    cout << "Training data loading time " << elapsed << " ms for batch size " << BATCH_SIZE << endl;

    auto optimizer = createOptimizer(trainParams, ADAM_BETA1, ADAM_BETA2, ADAM_EPSILON);
    LayerTypeStatistics statCallback;
    // 0 - warmup, 1 - measure overall time, 2 - measure individual layers

    dtype forwardTime = 0;
    dtype backwardTime = 0;
    dtype clippingTime = 0;
    dtype optimizerTime = 0;

    for (size_t iter = 0; iter < 3; ++iter)
    {
        if (iter == 2)
        {
            using namespace std::placeholders;
            work.getNetworkParameters().mCallback = std::bind(&LayerTypeStatistics::operator(), &statCallback, _1, _2, _3);
        }
        timeStart = chrono::steady_clock::now();
        
        work.forwardPassTraining();
        if (iter == 1)
        {
            work.getGpuCommandQueue().finish();
            forwardTime = TODTYPE(chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - timeStart).count());
            timeStart = chrono::steady_clock::now();
        }

        work.backwardPassTraining();

        if (iter == 1)
        {
            work.getGpuCommandQueue().finish();
            backwardTime = TODTYPE(chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - timeStart).count());
        }

        auto trainableParams = work.getTrainableParameters();

        if (iter == 1)
        {
            timeStart = chrono::steady_clock::now();
        }
        clip.processGradients(trainableParams, work.getNetworkParameters());
        if (iter == 1)
        {
            work.getGpuCommandQueue().finish();
            clippingTime = TODTYPE(chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - timeStart).count());
        }

        if (trainParams.decayLearningRate)
        {
            optimizer->step();
        }

        if (iter == 1)
        {
            timeStart = chrono::steady_clock::now();
        }
        for (auto p : trainableParams)
        {
            optimizer->operator()(memory_manager, p.Param, p.Gradient);
        }
        if (iter == 1)
        {
            work.getGpuCommandQueue().finish();
            optimizerTime = TODTYPE(chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - timeStart).count());
        }
        work.getGpuCommandQueue().finish();
    }

    statCallback.print(forwardTime, backwardTime);
    cout << "Forward time: " << forwardTime << endl;
    cout << "Backward time: " << backwardTime << endl;
    cout << "Clipping time: " << clippingTime << endl;
    cout << "Optimizer time: " << optimizerTime << endl;
}

TEST(TestTacotronMsk, SamarinTrainableEmbeddingGpuUnit)
{
    PROFILE_TEST

    GPU_ONLY_TEST

    using namespace tacotron;
    using namespace optimizers;

    auto testPath = tools::getTestAssetsDir() / "Tacotron_Msk_RP";

    constexpr size_t GOLDEN_PARAMS = 59;
    constexpr size_t GOLDEN_TRAINABLE_PARAMS = 51;

    const bool usePool = UT::tools::getArg("use_pool", true);
    cout << "Using pool: " << (usePool ? "yes" : "no") << endl;

    Workflow work(CompressionMode::NONE, CalculationMode::DETERMINISTIC, usePool ? AllocationMode::POOL : AllocationMode::STANDARD, ExecutionTarget::GPU);
    auto& memory_manager = work.getMemoryManager<MemoryManagerGPU>();

    constexpr dtype ADAM_BETA1 = 0.9_dt;
    constexpr dtype ADAM_BETA2 = 0.999_dt;
    constexpr dtype ADAM_EPSILON = 1e-6_dt;

    string experimentName = "samarin_new";
    auto weightsPrefix = filesystem::path("weights_250000") / "250000_Tacotron_model.";

    size_t T_IN = 68;
    size_t BATCH_SIZE = 11;

    TacotronParams params({ "encoder_output", "duration_targets", "inputs", "input_lengths", "mel_targets" },
                          { "decoder_output", "mel_outputs", "duration_outputs" },
                          {}, LayerExecutionTarget::Default);
    params.maxMelFrames = 564;
    params.useDurationPrediction = true;
    params.attentionType = "None";
    params.durationPredictorLstmUnits = 256;
    params.postnetKernelSize = { 5, 3, 3, 3 };
    params.postnetChannels = 256;
    params.prenetLayers = { 128, 128 };
    params.dropoutRate = 0.0f;
    params.zoneoutRate = 0.0f;
    params.postnetDropoutRate = 0.0f;
    params.trainableSpeakerEmbedding = true;
    params.withoutStopTokenLoss = true;
    params.lossMultipliers = { 1.f, 1.f, 2.f };
    params.useResidualRnn = true;

    TacotronTrainingParams trainParams;
    trainParams.initialLearningRate = 1e-3_dt;
    trainParams.clipNorm = 1.0_dt;
    trainParams.decayLearningRate = true;
    trainParams.decaySteps = 500;
    trainParams.decayStartStep = 100;
    trainParams.decayRate = 0.5;
    trainParams.finalLearningRate = 1e-5_dt;
    trainParams.warmupLearningRate = true;
    trainParams.warmupSteps = 1000;

    size_t nEpoch = 5;

    work.add<DataLayer>("inputs", DataParams{ { "inputs" }, 1u, T_IN, 1u });
    work.add<DataLayer>("duration_targets", DataParams{ { "duration_targets" }, 1u, 1u, T_IN });
    work.add<DataLayer>("encoder_output", DataParams{ { "encoder_output" }, 1u, T_IN, params.embeddingDim });
    work.add<DataLayer>("input_lengths", DataParams{ { "input_lengths" }, 1u, 1u, 1u });
    work.add<DataLayer>("mel_targets", DataParams{ { "mel_targets" }, 1u, params.maxMelFrames, params.numMels });
    work.add<DataLayer>("targets_lengths", DataParams{ { "targets_lengths" }, 1u, 1u, 1u });

    AddSingleSpeakerFslTacotronModel(&work, "T", params);

    tacotron::AddTacotronLoss(
        &work,
        "total_loss",
        { { "decoder_output", "mel_outputs", "duration_outputs", "mel_targets", "targets_lengths", "duration_targets" }, { "loss", "before_loss", "after_loss", "duration_loss" } },
        params);

    auto paramsCount = tacotronParamsCount(params);

    ASSERT_EQ(paramsCount.first, GOLDEN_TRAINABLE_PARAMS);
    ASSERT_EQ(paramsCount.second, GOLDEN_PARAMS);

    auto timeStart = chrono::steady_clock::now();
    TENSORS_CREATE(BATCH_SIZE);
    auto elapsed = TODTYPE(chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - timeStart).count());
    cout << "Workflow preparation time " << elapsed << " ms for batch size " << BATCH_SIZE << endl;

    ASSERT_EQ(work.getTrainableParameterNames().size(), GOLDEN_TRAINABLE_PARAMS);

    postprocessing::GradientClipping clip(1_dt);

    // Data loading

    timeStart = chrono::steady_clock::now();
    size_t loaded = loadTacotronParams((testPath / weightsPrefix).string(), memory_manager, "T", params, true, false);
    ASSERT_EQ(loaded, GOLDEN_PARAMS - 1); // speaker embedding will be loaded separately
    ASSERT_TRUE(loadTFData(testPath / experimentName / "speaker_embedding.data", memory_manager[Name("T") / "speaker_embedding"]));
    elapsed = TODTYPE(chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - timeStart).count());
    cout << "Weights loading time " << elapsed << " ms for batch size " << BATCH_SIZE << endl;

    timeStart = chrono::steady_clock::now();
    ASSERT_TRUE(loadTFData(testPath / experimentName / "encoder_output.data", memory_manager["encoder_output"]));
    ASSERT_TRUE(loadTFData(testPath / experimentName / "input_lengths.data", memory_manager["input_lengths"]));
    ASSERT_TRUE(loadTFData(testPath / experimentName / "inputs.data", memory_manager["inputs"]));
    ASSERT_TRUE(loadTFData(testPath / experimentName / "duration_targets.data", memory_manager["duration_targets"]));
    ASSERT_TRUE(loadTFData(testPath / experimentName / "targets_lengths.data", memory_manager["targets_lengths"]));
    ASSERT_TRUE(loadTFData(testPath / experimentName / "mel_targets.data", memory_manager["mel_targets"]));
    elapsed = TODTYPE(chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - timeStart).count());
    cout << "Training data loading time " << elapsed << " ms for batch size " << BATCH_SIZE << endl;

    auto optimizer = createOptimizer(trainParams, ADAM_BETA1, ADAM_BETA2, ADAM_EPSILON);

    dtype fastestEpochTime = numeric_limits<dtype>::max();
    dtype slowestEpochRime = 0_dt;
    dtype trainingTime = 0;
    //work.getNetworkParameters().mCallbackGPU = debugCallback<MemoryManagerGPU>();

    for (size_t epoch = 0; epoch < nEpoch; ++epoch)
    {
        cout << "Epoch " << epoch << endl;
        timeStart = chrono::steady_clock::now();
        work.forwardPassTraining();
        
        work.backwardPassTraining();
        
        auto trainableParams = work.getTrainableParameters<MemoryManagerGPU>();

        clip.processGradients(trainableParams, work.getNetworkParameters());
        auto gnorm_before = clip.getGlobalNorm(work.getNetworkParameters());
        auto gnorm_after = clip.calcGlobalNorm(trainableParams, work.getNetworkParameters());

        if (trainParams.decayLearningRate)
        {
            optimizer->step();
        }

        for (auto p : trainableParams)
        {
            optimizer->operator()(work.getKernelManager(), memory_manager, p.Param, p.Gradient);
        }
        const Tensor lossValue = memory_manager["loss"];
        const Tensor bLoss = memory_manager["before_loss"];
        const Tensor aLoss = memory_manager["after_loss"];
        const Tensor dLoss = memory_manager["duration_loss"];

        elapsed = TODTYPE(chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - timeStart).count());

        trainingTime += elapsed;

        fastestEpochTime = std::min(fastestEpochTime, elapsed);
        slowestEpochRime = std::max(slowestEpochRime, elapsed);


        cout << "  Epoch time " << elapsed << " ms for batch size " << BATCH_SIZE << endl;
        cout << "  Average performance: " << trainingTime / TODTYPE(epoch + 1) << " ms/epoch (fastest: " << fastestEpochTime << ", slowest " << slowestEpochRime << ")" << endl << endl;
        cout << "  Loss = " << lossValue[0] << " (before = " << bLoss[0] << ", after = " << aLoss[0]
             << ", duration = " << dLoss[0] << ")" << endl;
        cout << "  Global grad norm: " << gnorm_before << " -> " << gnorm_after << endl;
    }
}

TEST(TestTacotronMsk, P225TrainableEmbeddingUnit)
{
    PROFILE_TEST
    using namespace tacotron;

    auto testPath = tools::getTestAssetsDir() / "Tacotron_Msk_RP";

    constexpr size_t GOLDEN_PARAMS = 59;
    constexpr size_t GOLDEN_TRAINABLE_PARAMS = 51;

    Workflow work;
    constexpr size_t T_IN = 86;
    constexpr size_t BATCH_SIZE = 12;
    constexpr dtype EPS = 1e-3_dt;

    TacotronParams params({ "encoder_output", "duration_targets", "inputs", "input_lengths", "mel_targets" }, { "decoder_output", "mel_outputs", "duration_outputs" }, {});
    params.maxMelFrames = 666;
    params.useDurationPrediction = true;
    params.attentionType = "None";
    params.durationPredictorLstmUnits = 256;
    params.postnetKernelSize = { 5, 3, 3, 3 };
    params.postnetChannels = 256;
    params.prenetLayers = { 128, 128 };
    params.dropoutRate = 0.f;
    params.zoneoutRate = 0.f;
    params.postnetDropoutRate = 0.f;
    params.trainableSpeakerEmbedding = true;
    params.withoutStopTokenLoss = true;
    params.lossMultipliers = { 1.f, 1.f, 2.f };
    params.useResidualRnn = true;

    work.add<DataLayer>("inputs", DataParams{ { "inputs" }, 1u, T_IN, 1u });
    work.add<DataLayer>("duration_targets", DataParams{ { "duration_targets" }, 1u, 1u, T_IN });
    work.add<DataLayer>("encoder_output", DataParams{ { "encoder_output" }, 1u, T_IN, params.embeddingDim });
    work.add<DataLayer>("input_lengths", DataParams{ { "input_lengths" }, 1u, 1u, 1u });
    work.add<DataLayer>("mel_targets", DataParams{ { "mel_targets" }, 1u, params.maxMelFrames, params.numMels });
    work.add<DataLayer>("targets_lengths", DataParams{ { "targets_lengths" }, 1u, 1u, 1u });

    work.add<DataLayer>("golden_decoder_output", DataParams{ { "golden_decoder_output" }, 1u, params.maxMelFrames, params.numMels });
    work.add<DataLayer>("golden_mel_outputs", DataParams{ { "golden_mel_outputs" }, 1u, params.maxMelFrames, params.numMels });
    work.add<DataLayer>("golden_duration_outputs", DataParams{ { "golden_duration_outputs" }, 1u, T_IN, 1u });
    work.add<TensorLayer>("golden_loss", TensorParams{ { "golden_loss" }, WShape{ 1, 1, 1, 4 } });

    AddSingleSpeakerFslTacotronModel(&work, "T", params);

    tacotron::AddTacotronLoss(
        &work,
        "total_loss",
        { { "decoder_output", "mel_outputs", "duration_outputs", "mel_targets", "targets_lengths", "duration_targets" }, { "loss", "before_loss", "after_loss", "duration_loss" } },
        params);

    auto paramsCount = tacotronParamsCount(params);

    ASSERT_EQ(paramsCount.first, GOLDEN_TRAINABLE_PARAMS);
    ASSERT_EQ(paramsCount.second, GOLDEN_PARAMS);

    TENSORS_CREATE(BATCH_SIZE);

    ASSERT_EQ(work.getTrainableParameterNames().size(), GOLDEN_TRAINABLE_PARAMS);

    auto& memory_manager = work.getMemoryManager();

    // Data loading

    size_t loaded = loadTacotronParams((testPath / "weights" / "small_v2_230_Tacotron_model.").string(), memory_manager, "T", params, true, false);
    ASSERT_EQ(loaded, GOLDEN_PARAMS - 1); // speaker embedding will be loaded separately

    EXPECT_TRUE(loadTFData(testPath / "p225_trainable" / "encoder_output.data", memory_manager["encoder_output"]));
    EXPECT_TRUE(loadTFData(testPath / "p225_trainable" / "input_lengths.data", memory_manager["input_lengths"]));
    EXPECT_TRUE(loadTFData(testPath / "p225_trainable" / "inputs.data", memory_manager["inputs"]));
    EXPECT_TRUE(loadTFData(testPath / "p225_trainable" / "duration_targets.data", memory_manager["duration_targets"]));
    EXPECT_TRUE(loadTFData(testPath / "p225_trainable" / "targets_lengths.data", memory_manager["targets_lengths"]));
    EXPECT_TRUE(loadTFData(testPath / "p225_trainable" / "mel_targets.data", memory_manager["mel_targets"]));

    EXPECT_TRUE(loadTFData(testPath / "p225_trainable" / "speaker_embedding.data", memory_manager[Name("T") / "speaker_embedding"]));

    EXPECT_TRUE(loadTFData(testPath / "p225_trainable" / "duration_outputs.data", memory_manager["golden_duration_outputs"]));
    EXPECT_TRUE(loadTFData(testPath / "p225_trainable" / "mel_outputs.data", memory_manager["golden_mel_outputs"]));
    EXPECT_TRUE(loadTFData(testPath / "p225_trainable" / "decoder_output.data", memory_manager["golden_decoder_output"]));
    EXPECT_TRUE(loadTFData(testPath / "p225_trainable" / "loss.data", memory_manager["golden_loss"]));

    auto idealGradNorms = loadNamedValues(testPath / "p225_trainable" / "grad_norm.data", "T", params);
    EXPECT_EQ(idealGradNorms.size(), GOLDEN_TRAINABLE_PARAMS);

    TensorChecker checker(
        {
            { "duration_outputs", "golden_duration_outputs" },
            { "decoder_output", "golden_decoder_output" },
            { "mel_outputs", "golden_mel_outputs" },
        },
        EPS,
        -1_dt,
        true);
    work.getNetworkParameters().mCallback = checker;
    ASSERT_NO_THROW(work.forwardPassTraining());

    auto& loss = memory_manager["loss"];

    ASSERT_EQ(loss.size(), 1u);
    ASSERT_EQ(memory_manager["before_loss"].size(), 1u);
    ASSERT_EQ(memory_manager["after_loss"].size(), 1u);
    ASSERT_EQ(memory_manager["duration_loss"].size(), 1u);

    auto bloss = memory_manager["before_loss"][0];
    auto aloss = memory_manager["after_loss"][0];
    auto dloss = memory_manager["duration_loss"][0];

    EXPECT_NEAR(loss[0], memory_manager["golden_loss"][0], EPS);
    EXPECT_NEAR(bloss, memory_manager["golden_loss"][1], EPS);
    EXPECT_NEAR(aloss, memory_manager["golden_loss"][2], EPS);
    EXPECT_NEAR(dloss, memory_manager["golden_loss"][3], EPS);

    cout << "forward ok" << endl;

    ASSERT_NO_THROW(work.backwardPassTraining());
    cout << "backward ok" << endl;

    auto trainableParams = work.getTrainableParameters();
    postprocessing::GradientClipping clip;
    auto gnorm = clip.calcGlobalNorm(trainableParams, work.getNetworkParameters());
    cout << "Global norm: " << gnorm << endl;
    dtype totalGradNormDiff = 0;
    for (size_t j = 0; j < trainableParams.size(); ++j)
    {
        auto& [param, grad] = trainableParams[j];
        cout << param.getName() << ": " << TensorNorm(grad) << "( " << idealGradNorms[param.getName()] << ")" << endl;

        cout << "  diff: " << fabs(TensorNorm(grad) - idealGradNorms[param.getName()]) << std::endl;
        totalGradNormDiff += fabs(TensorNorm(grad) - idealGradNorms[param.getName()]);
    }
    cout << "Total Grad Norm Diff: " << totalGradNormDiff << std::endl;

    EXPECT_TRUE(totalGradNormDiff < 0.05_dt);
}

TEST(TestTacotronMsk, P225TrainableEmbeddingGpuUnit)
{
    PROFILE_TEST

    GPU_ONLY_TEST

    using namespace tacotron;

    auto testPath = tools::getTestAssetsDir() / "Tacotron_Msk_RP";

    const bool usePool = UT::tools::getArg("use_pool", true);
    cout << "Using pool: " << (usePool ? "yes" : "no") << endl;

    constexpr size_t GOLDEN_PARAMS = 59;
    constexpr size_t GOLDEN_TRAINABLE_PARAMS = 51;

    WorkflowEager work(CompressionMode::NONE, CalculationMode::DETERMINISTIC, usePool ? AllocationMode::POOL : AllocationMode::STANDARD, ExecutionTarget::GPU);
    auto& memory_manager = work.getMemoryManager<MemoryManagerGPU>();
    constexpr size_t T_IN = 86;
    constexpr size_t BATCH_SIZE = 12;
    constexpr dtype EPS = 1e-3_dt;

    TacotronParams params({ "encoder_output", "duration_targets", "inputs", "input_lengths", "mel_targets" }, { "decoder_output", "mel_outputs", "duration_outputs" }, {});
    params.maxMelFrames = 666;
    params.useDurationPrediction = true;
    params.attentionType = "None";
    params.durationPredictorLstmUnits = 256;
    params.postnetKernelSize = { 5, 3, 3, 3 };
    params.postnetChannels = 256;
    params.prenetLayers = { 128, 128 };
    params.dropoutRate = 0.f;
    params.zoneoutRate = 0.f;
    params.postnetDropoutRate = 0.f;
    params.trainableSpeakerEmbedding = true;
    params.withoutStopTokenLoss = true;
    params.lossMultipliers = { 1.f, 1.f, 2.f };
    params.useResidualRnn = true;

    work.add<DataLayer>("inputs", DataParams{ { "inputs" }, 1u, T_IN, 1u });
    work.add<DataLayer>("duration_targets", DataParams{ { "duration_targets" }, 1u, 1u, T_IN });
    work.add<DataLayer>("encoder_output", DataParams{ { "encoder_output" }, 1u, T_IN, params.embeddingDim });
    work.add<DataLayer>("input_lengths", DataParams{ { "input_lengths" }, 1u, 1u, 1u });
    work.add<DataLayer>("mel_targets", DataParams{ { "mel_targets" }, 1u, params.maxMelFrames, params.numMels });
    work.add<DataLayer>("targets_lengths", DataParams{ { "targets_lengths" }, 1u, 1u, 1u });

    work.add<DataLayer>("golden_decoder_output", DataParams{ { "golden_decoder_output" }, 1u, params.maxMelFrames, params.numMels });
    work.add<DataLayer>("golden_mel_outputs", DataParams{ { "golden_mel_outputs" }, 1u, params.maxMelFrames, params.numMels });
    work.add<DataLayer>("golden_duration_outputs", DataParams{ { "golden_duration_outputs" }, 1u, T_IN, 1u });
    work.add<TensorLayer>("golden_loss", TensorParams{ { "golden_loss" }, WShape{ 1, 1, 1, 4 } });

    AddSingleSpeakerFslTacotronModel(&work, "T", params);

    tacotron::AddTacotronLoss(
        &work,
        "total_loss",
        { { "decoder_output", "mel_outputs", "duration_outputs", "mel_targets", "targets_lengths", "duration_targets" }, { "loss", "before_loss", "after_loss", "duration_loss" } },
        params);

    auto paramsCount = tacotronParamsCount(params);

    ASSERT_EQ(paramsCount.first, GOLDEN_TRAINABLE_PARAMS);
    ASSERT_EQ(paramsCount.second, GOLDEN_PARAMS);

    TENSORS_CREATE(BATCH_SIZE);

    ASSERT_EQ(work.getTrainableParameterNames().size(), GOLDEN_TRAINABLE_PARAMS);

    // Data loading

    size_t loaded = loadTacotronParams((testPath / "weights" / "small_v2_230_Tacotron_model.").string(), memory_manager, "T", params, true, false);
    ASSERT_EQ(loaded, GOLDEN_PARAMS - 1); // speaker embedding will be loaded separately

    EXPECT_TRUE(loadTFData(testPath / "p225_trainable" / "encoder_output.data", memory_manager["encoder_output"]));
    EXPECT_TRUE(loadTFData(testPath / "p225_trainable" / "input_lengths.data", memory_manager["input_lengths"]));
    EXPECT_TRUE(loadTFData(testPath / "p225_trainable" / "inputs.data", memory_manager["inputs"]));
    EXPECT_TRUE(loadTFData(testPath / "p225_trainable" / "duration_targets.data", memory_manager["duration_targets"]));
    EXPECT_TRUE(loadTFData(testPath / "p225_trainable" / "targets_lengths.data", memory_manager["targets_lengths"]));
    EXPECT_TRUE(loadTFData(testPath / "p225_trainable" / "mel_targets.data", memory_manager["mel_targets"]));

    EXPECT_TRUE(loadTFData(testPath / "p225_trainable" / "speaker_embedding.data", memory_manager[Name("T") / "speaker_embedding"]));

    EXPECT_TRUE(loadTFData(testPath / "p225_trainable" / "duration_outputs.data", memory_manager["golden_duration_outputs"]));
    EXPECT_TRUE(loadTFData(testPath / "p225_trainable" / "mel_outputs.data", memory_manager["golden_mel_outputs"]));
    EXPECT_TRUE(loadTFData(testPath / "p225_trainable" / "decoder_output.data", memory_manager["golden_decoder_output"]));
    EXPECT_TRUE(loadTFData(testPath / "p225_trainable" / "loss.data", memory_manager["golden_loss"]));

    auto idealGradNorms = loadNamedValues(testPath / "p225_trainable" / "grad_norm.data", "T", params);
    EXPECT_EQ(idealGradNorms.size(), GOLDEN_TRAINABLE_PARAMS);

    vector<pair<Name, Name>> checks =
        {
            { "duration_outputs", "golden_duration_outputs" },
            { "decoder_output", "golden_decoder_output" },
            { "mel_outputs", "golden_mel_outputs" },
        };

    ASSERT_NO_THROW(work.forwardPassTraining());

    Tensor loss = memory_manager["loss"];
    Tensor before_loss = memory_manager["before_loss"];
    Tensor after_loss = memory_manager["after_loss"];
    Tensor duration_loss = memory_manager["duration_loss"];
    Tensor golden_loss = memory_manager["golden_loss"];

    ASSERT_EQ(loss.size(), 1u);
    ASSERT_EQ(before_loss.size(), 1u);
    ASSERT_EQ(after_loss.size(), 1u);
    ASSERT_EQ(duration_loss.size(), 1u);

    auto bloss = before_loss[0];
    auto aloss = after_loss[0];
    auto dloss = duration_loss[0];

    EXPECT_NEAR(loss[0], golden_loss[0], EPS);
    EXPECT_NEAR(bloss, golden_loss[1], EPS);
    EXPECT_NEAR(aloss, golden_loss[2], EPS);
    EXPECT_NEAR(dloss, golden_loss[3], EPS);

    UT::tools::checkTensors(checks, memory_manager, EPS);

    cout << "forward ok" << endl;

    ASSERT_NO_THROW(work.backwardPassTraining());
    cout << "backward ok" << endl;

    auto trainableParams = work.getTrainableParameters<MemoryManagerGPU>();
    postprocessing::GradientClipping clip;
    auto gnorm = clip.calcGlobalNorm(trainableParams, work.getNetworkParameters());
    cout << "Global norm: " << gnorm << endl;
    dtype totalGradNormDiff = 0;
    for (size_t j = 0; j < trainableParams.size(); ++j)
    {
        auto& [param, gradGPU] = trainableParams[j];
        Tensor grad = memory_manager[gradGPU.getName()];
        cout << param.getName() << ": " << TensorNorm(grad) << "( " << idealGradNorms[param.getName()] << ")" << endl;

        cout << "  diff: " << fabs(TensorNorm(grad) - idealGradNorms[param.getName()]) << std::endl;
        totalGradNormDiff += fabs(TensorNorm(grad) - idealGradNorms[param.getName()]);
    }
    cout << "Total Grad Norm Diff: " << totalGradNormDiff << std::endl;

    EXPECT_TRUE(totalGradNormDiff < 0.05_dt);
}

TEST(TestTacotronMsk, DISABLED_SamarinTrainableEmbeddingTraining)
{
    PROFILE_TEST
    using namespace tacotron;
    using namespace optimizers;

    random::setGlobalSeed(42);

    auto testPath = tools::getTestAssetsDir() / "Tacotron_Msk_RP";

    constexpr size_t GOLDEN_PARAMS = 59;
    constexpr size_t GOLDEN_TRAINABLE_PARAMS = 51;

    bool mixedPrecision = UT::tools::getArg("target", "fp32") == "fp16";
    cout << "Target: " << (mixedPrecision ? "FP16" : "FP32") << endl;

    bool usePool = true;

    Workflow work(CompressionMode::NONE, CalculationMode::DETERMINISTIC, usePool ? AllocationMode::POOL : AllocationMode::STANDARD);

    constexpr dtype ADAM_BETA1 = 0.9_dt;
    constexpr dtype ADAM_BETA2 = 0.999_dt;
    constexpr dtype ADAM_EPSILON = 1e-6_dt;

    string experimentName = "samarin";
    auto outputPrefix = std::filesystem::path("Tacotron_Msk_RP") / experimentName;

    constexpr size_t T_IN = 68;
    constexpr size_t BATCH_SIZE = 11;

    TacotronParams params({ "encoder_output", "duration_targets", "inputs", "input_lengths", "mel_targets" },
                          { "decoder_output", "mel_outputs", "duration_outputs" },
                          {},
                          mixedPrecision ? LayerExecutionTarget::CPUFP16 : LayerExecutionTarget::Default);
    params.maxMelFrames = 564;
    params.useDurationPrediction = true;
    params.attentionType = "None";
    params.durationPredictorLstmUnits = 256;
    params.postnetKernelSize = { 5, 3, 3, 3 };
    params.postnetChannels = 256;
    params.prenetLayers = { 128, 128 };
    params.dropoutRate = 0.5f;
    params.zoneoutRate = 0.1f;
    params.postnetDropoutRate = 0.1f;
    params.trainableSpeakerEmbedding = true;
    params.withoutStopTokenLoss = true;
    params.lossMultipliers = { 1.f, 1.f, 2.f };
    params.useResidualRnn = true;

    TacotronTrainingParams trainParams;
    trainParams.initialLearningRate = 1e-3_dt;
    trainParams.clipNorm = 1.0_dt;
    trainParams.decayLearningRate = true;
    trainParams.decaySteps = 500;
    trainParams.decayStartStep = 100;
    trainParams.decayRate = 0.5;
    trainParams.finalLearningRate = 1e-5_dt;
    trainParams.warmupLearningRate = true;
    trainParams.warmupSteps = 1000;

    size_t nEpoch = 400;
    const size_t nEpochToSave = 50;

    work.add<DataLayer>("inputs", DataParams{ { "inputs" }, 1u, T_IN, 1u });
    work.add<DataLayer>("duration_targets", DataParams{ { "duration_targets" }, 1u, 1u, T_IN });
    work.add<DataLayer>("encoder_output", DataParams{ { "encoder_output" }, 1u, T_IN, params.embeddingDim });
    work.add<DataLayer>("input_lengths", DataParams{ { "input_lengths" }, 1u, 1u, 1u });
    work.add<DataLayer>("mel_targets", DataParams{ { "mel_targets" }, 1u, params.maxMelFrames, params.numMels });
    work.add<DataLayer>("targets_lengths", DataParams{ { "targets_lengths" }, 1u, 1u, 1u });

    AddSingleSpeakerFslTacotronModel(&work, "T", params);

    tacotron::AddTacotronLoss(
        &work,
        "total_loss",
        { { "decoder_output", "mel_outputs", "duration_outputs", "mel_targets", "targets_lengths", "duration_targets" }, { "loss", "before_loss", "after_loss", "duration_loss" } },
        params);

    auto paramsCount = tacotronParamsCount(params);

    ASSERT_EQ(paramsCount.first, GOLDEN_TRAINABLE_PARAMS);
    ASSERT_EQ(paramsCount.second, GOLDEN_PARAMS);

    auto timeStart = chrono::steady_clock::now();
    TENSORS_CREATE(BATCH_SIZE);
    auto elapsed = TODTYPE(chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - timeStart).count());
    cout << "Workflow preparation time " << elapsed << " ms for batch size " << BATCH_SIZE << endl;

    ASSERT_EQ(work.getTrainableParameterNames().size(), GOLDEN_TRAINABLE_PARAMS);
    cout << tools::get_size_of_trainable_params_mixed_precision(work) << " trainable weights" << endl;

    auto& memory_manager = work.getMemoryManager();
    // auto& memory_managerFP16 = work.getMemoryManager<MemoryManagerFP16>();
    postprocessing::GradientClipping clip(1_dt);

//    std::ofstream outfile("SamarinTrainableEmbeddingTraining.txt");
//    const auto layerSet = work.getSetOfLayers();
//    for (auto& l : layerSet)
//    {
//        outfile << l << std::endl;
//    }
//    outfile.close();

    // Data loading

    timeStart = chrono::steady_clock::now();
    size_t loaded = loadTacotronParams((testPath / "weights" / "small_v2_230_Tacotron_model.").string(), memory_manager, "T", params, true, false);
    // loaded += loadTacotronParams((testPath / "weights" / "small_v2_230_Tacotron_model.").string(), memory_managerFP16, "T", params, true, false);
    ASSERT_EQ(loaded, GOLDEN_PARAMS - 1); // speaker embedding will be loaded separately
    ASSERT_TRUE(loadTFData(testPath / "samarin" / "speaker_embedding.data", memory_manager[Name("T") / "speaker_embedding"]));
    elapsed = TODTYPE(chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - timeStart).count());
    cout << "Weights loading time " << elapsed << " ms for batch size " << BATCH_SIZE << endl;

    timeStart = chrono::steady_clock::now();
    ASSERT_TRUE(loadTFData(testPath / "samarin" / "encoder_output.data", memory_manager["encoder_output"]));
    ASSERT_TRUE(loadTFData(testPath / "samarin" / "input_lengths.data", memory_manager["input_lengths"]));
    ASSERT_TRUE(loadTFData(testPath / "samarin" / "inputs.data", memory_manager["inputs"]));
    ASSERT_TRUE(loadTFData(testPath / "samarin" / "duration_targets.data", memory_manager["duration_targets"]));
    ASSERT_TRUE(loadTFData(testPath / "samarin" / "targets_lengths.data", memory_manager["targets_lengths"]));
    ASSERT_TRUE(loadTFData(testPath / "samarin" / "mel_targets.data", memory_manager["mel_targets"]));
    elapsed = TODTYPE(chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - timeStart).count());
    cout << "Training data loading time " << elapsed << " ms for batch size " << BATCH_SIZE << endl;

    auto optimizer = createOptimizer(trainParams, ADAM_BETA1, ADAM_BETA2, ADAM_EPSILON);

    dtype fastestEpochTime = numeric_limits<dtype>::max();
    dtype slowestEpochRime = 0_dt;
    dtype trainingTime = 0;

    for (size_t epoch = 0; epoch < nEpoch; ++epoch)
    {
        cout << "Epoch " << epoch << endl;
        timeStart = chrono::steady_clock::now();
        work.forwardPassTraining();

        work.backwardPassTraining();

        auto names = work.getTrainableParameterNames();

        std::vector<ParamAndGrad> trainableParams;
        // std::vector<ParamAndGradImpl<TensorFP16>> trainableParamsFP16;

        for (auto& name : names)
        {
            if (memory_manager.tensorExists(name) && memory_manager.tensorExists(name.grad()))
            {
                trainableParams.push_back(ParamAndGrad{ memory_manager[name], memory_manager[name.grad()] });
            }

            // if (memory_managerFP16.tensorExists(name) && memory_managerFP16.tensorExists(name.grad()))
            {
                // trainableParamsFP16.push_back(ParamAndGradImpl<TensorFP16>{ memory_managerFP16[name], memory_managerFP16[name.grad()] });
            }
        }

        // clip.processGradientsMixedPrecision(trainableParams, trainableParamsFP16, work.getNetworkParameters());
        clip.processGradients(trainableParams, work.getNetworkParameters());
        auto gnorm_before = clip.getGlobalNorm(work.getNetworkParameters());
        auto gnorm_after = clip.calcGlobalNorm(trainableParams, work.getNetworkParameters());
        // auto gnorm_after = clip.calcGlobalNormMixedPrecision(trainableParams, trainableParamsFP16, work.getNetworkParameters());

        if (trainParams.decayLearningRate)
        {
            optimizer->step();
        }

        for (auto name : names)
        {
            if (memory_manager.tensorExists(name)) optimizer->operator()(memory_manager, memory_manager[name], memory_manager[name.grad()]);
            // else if(memory_managerFP16.tensorExists(name)) optimizer->operator()(memory_managerFP16, memory_managerFP16[name], memory_managerFP16[name.grad()]);
        }

        auto& lossValue = memory_manager["loss"];
        elapsed = TODTYPE(chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - timeStart).count());

        trainingTime += elapsed;

        fastestEpochTime = std::min(fastestEpochTime, elapsed);
        slowestEpochRime = std::max(slowestEpochRime, elapsed);

        cout << "  Epoch time " << elapsed << " ms for batch size " << BATCH_SIZE << endl;
        cout << "  Average performance: " << trainingTime / TODTYPE(epoch + 1) << " ms/epoch (fastest: " << fastestEpochTime << ", slowest " << slowestEpochRime << ")" << endl << endl;
        cout << "  Loss = " << lossValue[0] << " (before = " << memory_manager["before_loss"][0] << ", after = " << memory_manager["after_loss"][0]
             << ", duration = " << memory_manager["duration_loss"][0] << ")" << endl;

        cout << "  Global grad norm: " << gnorm_before << " -> " << gnorm_after << endl;

        if (nEpochToSave > 0 && ((epoch + 1) % nEpochToSave == 0 || epoch == nEpoch - 1))
        {
            [[maybe_unused]] size_t saved = saveTacotronParams(outputPrefix / to_string(epoch + 1), "230000_Tacotron_model.", memory_manager, "T", params, false);
        }
    }
}

#if defined(ANDROID)
TEST(TestTacotronMsk, SamarinTrainableEmbeddingTrainingFP16)
{
    PROFILE_TEST
    using namespace tacotron;
    using namespace optimizers;

    random::setGlobalSeed(42);

    auto testPath = tools::getTestAssetsDir() / "Tacotron_Msk_RP";

    constexpr size_t GOLDEN_PARAMS = 59;
    constexpr size_t GOLDEN_TRAINABLE_PARAMS = 51;

    bool usePool = false;
    bool masterWeightsEnabled = true;

    Workflow work(CompressionMode::NONE, CalculationMode::DETERMINISTIC, usePool ? AllocationMode::POOL : AllocationMode::STANDARD, ExecutionTarget::CPUFP16);

    constexpr dtype ADAM_BETA1 = 0.9_dt;
    constexpr dtype ADAM_BETA2 = 0.999_dt;
    constexpr dtype ADAM_EPSILON = 1e-6_dt;

    string experimentName = "samarin";
    auto outputPrefix = std::filesystem::path("Tacotron_Msk_RP") / experimentName;

    constexpr size_t T_IN = 68;
    constexpr size_t BATCH_SIZE = 11;

    TacotronParams params({ "encoder_output", "duration_targets", "inputs", "input_lengths", "mel_targets" },
                          { "decoder_output", "mel_outputs", "duration_outputs" },
                          {},
                          masterWeightsEnabled ? LayerExecutionTarget::CPU : LayerExecutionTarget::Default);
    params.maxMelFrames = 564;
    params.useDurationPrediction = true;
    params.attentionType = "None";
    params.durationPredictorLstmUnits = 256;
    params.postnetKernelSize = { 5, 3, 3, 3 };
    params.postnetChannels = 256;
    params.prenetLayers = { 128, 128 };
    params.dropoutRate = 0.5f;
    params.zoneoutRate = 0.1f;
    params.postnetDropoutRate = 0.1f;
    params.trainableSpeakerEmbedding = true;
    params.withoutStopTokenLoss = true;
    params.lossMultipliers = { 1.f, 1.f, 2.f };
    params.useResidualRnn = true;
    //params.useFusion = true;

    TacotronTrainingParams trainParams;
    trainParams.initialLearningRate = 1e-3_dt;
    trainParams.clipNorm = 1.0_dt;
    trainParams.decayLearningRate = true;
    trainParams.decaySteps = 500;
    trainParams.decayStartStep = 100;
    trainParams.decayRate = 0.5;
    trainParams.finalLearningRate = 1e-5_dt;
    trainParams.warmupLearningRate = true;
    trainParams.warmupSteps = 1000;

    size_t nEpoch = 400;
    const size_t nEpochToSave = 50;

    work.add<DataLayer>("inputs", DataParams{ { "inputs" }, 1u, T_IN, 1u });
    work.add<DataLayer>("duration_targets", DataParams{ { "duration_targets" }, 1u, 1u, T_IN });
    work.add<DataLayer>("encoder_output", DataParams{ { "encoder_output" }, 1u, T_IN, params.embeddingDim });
    work.add<DataLayer>("input_lengths", DataParams{ { "input_lengths" }, 1u, 1u, 1u });
    work.add<DataLayer>("mel_targets", DataParams{ { "mel_targets" }, 1u, params.maxMelFrames, params.numMels });
    work.add<DataLayer>("targets_lengths", DataParams{ { "targets_lengths" }, 1u, 1u, 1u });

    AddSingleSpeakerFslTacotronModel(&work, "T", params);

    work.overrideLayerExecutionTarget(raul::LayerExecutionTarget::CPU);
    work.add<raul::ConvertPrecisionLayer>("c1", raul::ConvertPrecisionParams{ { "decoder_output"}, { "decoder_output_fp32" }, false });
    work.add<raul::ConvertPrecisionLayer>("c2", raul::ConvertPrecisionParams{ { "mel_outputs"}, { "mel_outputs_fp32" }, false });
    work.add<raul::ConvertPrecisionLayer>("c3", raul::ConvertPrecisionParams{ { "duration_outputs"}, { "duration_outputs_fp32" }, false });
    work.add<raul::ConvertPrecisionLayer>("c4", raul::ConvertPrecisionParams{ { "mel_targets"}, { "mel_targets_fp32" }, false, false });
    work.add<raul::ConvertPrecisionLayer>("c5", raul::ConvertPrecisionParams{ { "targets_lengths"}, { "targets_lengths_fp32" }, false, false });
    work.add<raul::ConvertPrecisionLayer>("c6", raul::ConvertPrecisionParams{ { "duration_targets"}, { "duration_targets_fp32" }, false, false });
    tacotron::AddTacotronLoss(
        &work,
        "total_loss",
        { { "decoder_output_fp32", "mel_outputs_fp32", "duration_outputs_fp32", "mel_targets_fp32", "targets_lengths_fp32", "duration_targets_fp32" }, { "loss", "before_loss", "after_loss", "duration_loss" } },
        params);
    work.resetLayerExecutionTargetOverride();

    auto paramsCount = tacotronParamsCount(params);

    ASSERT_EQ(paramsCount.first, GOLDEN_TRAINABLE_PARAMS);
    ASSERT_EQ(paramsCount.second, GOLDEN_PARAMS);

    auto timeStart = chrono::steady_clock::now();
    TENSORS_CREATE(BATCH_SIZE);
    auto elapsed = TODTYPE(chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - timeStart).count());
    cout << "Workflow preparation time " << elapsed << " ms for batch size " << BATCH_SIZE << endl;

    //due to master weights usage amount of trainable params are larger than expected
    //ASSERT_EQ(work.getTrainableParametersSafe<MemoryManagerFP16>().size() + work.getTrainableParametersSafe().size(), GOLDEN_TRAINABLE_PARAMS);
    cout << "FP32 parameters: " << work.getTrainableParametersSafe().size() << endl;
    cout << "FP16 parameters: " << work.getTrainableParametersSafe<MemoryManagerFP16>().size() << endl;
    cout << tools::get_size_of_trainable_params_mixed_precision(work) << " trainable weights" << endl;

    auto& memory_manager = work.getMemoryManager<MemoryManagerFP16>();
    auto& memory_manager_fp32 = work.getMemoryManager<MemoryManager>();
    postprocessing::GradientClipping clip(1_dt);

    // Data loading

    timeStart = chrono::steady_clock::now();
    size_t loaded_fp16 = loadTacotronParams((testPath / "weights" / "small_v2_230_Tacotron_model.").string(), memory_manager, "T", params, true, false);
    size_t loaded_fp32 = loadTacotronParams((testPath / "weights" / "small_v2_230_Tacotron_model.").string(), memory_manager_fp32, "T", params, true, false);
    cout << "Loaded " << loaded_fp16 << " fp16 parameters and " << loaded_fp32 << " fp32 parameters" << endl;
    ASSERT_EQ(loaded_fp16 + loaded_fp32, GOLDEN_PARAMS - 1); // speaker embedding will be loaded separately
    ASSERT_TRUE(loadTFData(testPath / "samarin" / "speaker_embedding.data", memory_manager[Name("T") / "speaker_embedding"]));
    elapsed = TODTYPE(chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - timeStart).count());
    cout << "Weights loading time " << elapsed << " ms for batch size " << BATCH_SIZE << endl;

    timeStart = chrono::steady_clock::now();
    ASSERT_TRUE(loadTFData(testPath / "samarin" / "encoder_output.data", memory_manager["encoder_output"]));
    ASSERT_TRUE(loadTFData(testPath / "samarin" / "input_lengths.data", memory_manager["input_lengths"]));
    ASSERT_TRUE(loadTFData(testPath / "samarin" / "inputs.data", memory_manager["inputs"]));
    ASSERT_TRUE(loadTFData(testPath / "samarin" / "duration_targets.data", memory_manager["duration_targets"]));
    ASSERT_TRUE(loadTFData(testPath / "samarin" / "targets_lengths.data", memory_manager["targets_lengths"]));
    ASSERT_TRUE(loadTFData(testPath / "samarin" / "mel_targets.data", memory_manager["mel_targets"]));
    elapsed = TODTYPE(chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - timeStart).count());
    cout << "Training data loading time " << elapsed << " ms for batch size " << BATCH_SIZE << endl;

    auto optimizer = createOptimizer(trainParams, ADAM_BETA1, ADAM_BETA2, ADAM_EPSILON);

    dtype fastestEpochTime = numeric_limits<dtype>::max();
    dtype slowestEpochRime = 0_dt;
    dtype trainingTime = 0;

    for (size_t epoch = 0; epoch < nEpoch; ++epoch)
    {
        cout << "Epoch " << epoch << endl;
        timeStart = chrono::steady_clock::now();
        work.forwardPassTraining();

        work.backwardPassTraining();

        auto trainableParamsFP16 = work.getTrainableParametersSafe<MemoryManagerFP16>();
        auto trainableParamsFP32 = work.getTrainableParametersSafe();

        clip.processGradientsMixedPrecision(trainableParamsFP32, trainableParamsFP16, work.getNetworkParameters());

        auto gnorm_before = clip.getGlobalNorm(work.getNetworkParameters());
        auto gnorm_after = clip.calcGlobalNormMixedPrecision(trainableParamsFP32, trainableParamsFP16, work.getNetworkParameters());

        if (trainParams.decayLearningRate)
        {
            optimizer->step();
        }

        for (auto& p : trainableParamsFP16)
        {
            optimizer->operator()(memory_manager, p.Param, p.Gradient);
        }

        for (auto& p : trainableParamsFP32)
        {
            optimizer->operator()(memory_manager_fp32, p.Param, p.Gradient);
        }

        auto& lossValue = memory_manager_fp32["loss"];
        elapsed = TODTYPE(chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - timeStart).count());

        trainingTime += elapsed;

        fastestEpochTime = std::min(fastestEpochTime, elapsed);
        slowestEpochRime = std::max(slowestEpochRime, elapsed);

        cout << "  Epoch time " << elapsed << " ms for batch size " << BATCH_SIZE << endl;
        cout << "  Average performance: " << trainingTime / TODTYPE(epoch + 1) << " ms/epoch (fastest: " << fastestEpochTime << ", slowest " << slowestEpochRime << ")" << endl << endl;
        cout << "  Loss = " << lossValue[0] << " (before = " << memory_manager_fp32["before_loss"][0] << ", after = " << memory_manager_fp32["after_loss"][0]
             << ", duration = " << memory_manager_fp32["duration_loss"][0] << ")" << endl;

        cout << "  Global grad norm: " << gnorm_before << " -> " << gnorm_after << endl;

        if (nEpochToSave > 0 && ((epoch + 1) % nEpochToSave == 0 || epoch == nEpoch - 1))
        {
            //[[maybe_unused]] size_t saved = saveTacotronParams(outputPrefix / to_string(epoch + 1), "230000_Tacotron_model.", memory_manager, "T", params, false);
        }
    }
}

TEST(TestTacotronMsk, SamarinTrainableEmbeddingMicrobatchingTrainingFP16)
{
    PROFILE_TEST
    using namespace tacotron;
    using namespace optimizers;

    random::setGlobalSeed(42);

    auto testPath = tools::getTestAssetsDir() / "Tacotron_Msk_RP";

    constexpr size_t GOLDEN_PARAMS = 59;
    constexpr size_t GOLDEN_TRAINABLE_PARAMS = 51;

    bool usePool = false;
    bool masterWeightsEnabled = true;

    Workflow work(CompressionMode::NONE, CalculationMode::DETERMINISTIC, usePool ? AllocationMode::POOL : AllocationMode::STANDARD, ExecutionTarget::CPUFP16);

    constexpr dtype ADAM_BETA1 = 0.9_dt;
    constexpr dtype ADAM_BETA2 = 0.999_dt;
    constexpr dtype ADAM_EPSILON = 1e-6_dt;

    string experimentName = "samarin";
    auto outputPrefix = std::filesystem::path("Tacotron_Msk_RP") / experimentName;

    constexpr size_t T_IN = 68;
    constexpr size_t BATCH_SIZE = 11;
    constexpr size_t MICRO_BATCH_SIZE = 6;
    const size_t numberOfMicroBatches = static_cast<size_t>(static_cast<float>(BATCH_SIZE) / static_cast<float>(MICRO_BATCH_SIZE) + 0.5f);

    TacotronParams params({ "encoder_output", "duration_targets", "inputs", "input_lengths", "mel_targets" },
                          { "decoder_output", "mel_outputs", "duration_outputs" },
                          {},
                          masterWeightsEnabled ? LayerExecutionTarget::CPU : LayerExecutionTarget::Default);
    params.maxMelFrames = 564;
    params.useDurationPrediction = true;
    params.attentionType = "None";
    params.durationPredictorLstmUnits = 256;
    params.postnetKernelSize = { 5, 3, 3, 3 };
    params.postnetChannels = 256;
    params.prenetLayers = { 128, 128 };
    params.dropoutRate = 0.5f;
    params.zoneoutRate = 0.1f;
    params.postnetDropoutRate = 0.1f;
    params.trainableSpeakerEmbedding = true;
    params.withoutStopTokenLoss = true;
    params.lossMultipliers = { 1.f, 1.f, 2.f };
    params.useResidualRnn = true;
    //params.useFusion = true;

    TacotronTrainingParams trainParams;
    trainParams.initialLearningRate = 1e-3_dt;
    trainParams.clipNorm = 1.0_dt;
    trainParams.decayLearningRate = true;
    trainParams.decaySteps = 500;
    trainParams.decayStartStep = 100;
    trainParams.decayRate = 0.5;
    trainParams.finalLearningRate = 1e-5_dt;
    trainParams.warmupLearningRate = true;
    trainParams.warmupSteps = 1000;

    size_t nEpoch = 400;
    const size_t nEpochToSave = 50;

    work.add<DataLayer>("inputs", DataParams{ { "inputs" }, 1u, T_IN, 1u });
    work.add<DataLayer>("duration_targets", DataParams{ { "duration_targets" }, 1u, 1u, T_IN });
    work.add<DataLayer>("encoder_output", DataParams{ { "encoder_output" }, 1u, T_IN, params.embeddingDim });
    work.add<DataLayer>("input_lengths", DataParams{ { "input_lengths" }, 1u, 1u, 1u });
    work.add<DataLayer>("mel_targets", DataParams{ { "mel_targets" }, 1u, params.maxMelFrames, params.numMels });
    work.add<DataLayer>("targets_lengths", DataParams{ { "targets_lengths" }, 1u, 1u, 1u });

    AddSingleSpeakerFslTacotronModel(&work, "T", params);

    work.overrideLayerExecutionTarget(raul::LayerExecutionTarget::CPU);
    work.add<raul::ConvertPrecisionLayer>("c1", raul::ConvertPrecisionParams{ { "decoder_output"}, { "decoder_output_fp32" }, false });
    work.add<raul::ConvertPrecisionLayer>("c2", raul::ConvertPrecisionParams{ { "mel_outputs"}, { "mel_outputs_fp32" }, false });
    work.add<raul::ConvertPrecisionLayer>("c3", raul::ConvertPrecisionParams{ { "duration_outputs"}, { "duration_outputs_fp32" }, false });
    work.add<raul::ConvertPrecisionLayer>("c4", raul::ConvertPrecisionParams{ { "mel_targets"}, { "mel_targets_fp32" }, false, false });
    work.add<raul::ConvertPrecisionLayer>("c5", raul::ConvertPrecisionParams{ { "targets_lengths"}, { "targets_lengths_fp32" }, false, false });
    work.add<raul::ConvertPrecisionLayer>("c6", raul::ConvertPrecisionParams{ { "duration_targets"}, { "duration_targets_fp32" }, false, false });
    tacotron::AddTacotronLoss(
        &work,
        "total_loss",
        { { "decoder_output_fp32", "mel_outputs_fp32", "duration_outputs_fp32", "mel_targets_fp32", "targets_lengths_fp32", "duration_targets_fp32" }, { "loss", "before_loss", "after_loss", "duration_loss" } },
        params);
    work.resetLayerExecutionTargetOverride();

    auto paramsCount = tacotronParamsCount(params);

    ASSERT_EQ(paramsCount.first, GOLDEN_TRAINABLE_PARAMS);
    ASSERT_EQ(paramsCount.second, GOLDEN_PARAMS);

    auto timeStart = chrono::steady_clock::now();
    TENSORS_CREATE(MICRO_BATCH_SIZE);
    auto elapsed = TODTYPE(chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - timeStart).count());
    cout << "Workflow preparation time " << elapsed << " ms for micro batch size " << MICRO_BATCH_SIZE << endl;

    //due to master weights usage amount of trainable params are larger than expected
    //ASSERT_EQ(work.getTrainableParametersSafe<MemoryManagerFP16>().size() + work.getTrainableParametersSafe().size(), GOLDEN_TRAINABLE_PARAMS);
    cout << "FP32 parameters: " << work.getTrainableParametersSafe().size() << endl;
    cout << "FP16 parameters: " << work.getTrainableParametersSafe<MemoryManagerFP16>().size() << endl;
    cout << tools::get_size_of_trainable_params_mixed_precision(work) << " trainable weights" << endl;

    auto& memory_manager = work.getMemoryManager<MemoryManagerFP16>();
    auto& memory_manager_fp32 = work.getMemoryManager<MemoryManager>();
    postprocessing::GradientClipping clip(1_dt);

    // Data loading

    timeStart = chrono::steady_clock::now();
    size_t loaded_fp16 = loadTacotronParams((testPath / "weights" / "small_v2_230_Tacotron_model.").string(), memory_manager, "T", params, true, false);
    size_t loaded_fp32 = loadTacotronParams((testPath / "weights" / "small_v2_230_Tacotron_model.").string(), memory_manager_fp32, "T", params, true, false);
    cout << "Loaded " << loaded_fp16 << " fp16 parameters and " << loaded_fp32 << " fp32 parameters" << endl;
    ASSERT_EQ(loaded_fp16 + loaded_fp32, GOLDEN_PARAMS - 1); // speaker embedding will be loaded separately
    ASSERT_TRUE(loadTFData(testPath / "samarin" / "speaker_embedding.data", memory_manager[Name("T") / "speaker_embedding"]));
    elapsed = TODTYPE(chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - timeStart).count());
    cout << "Weights loading time " << elapsed << " ms for batch size " << BATCH_SIZE << endl;

    timeStart = chrono::steady_clock::now();
    raul::TensorFP16 encoder_output(BATCH_SIZE, memory_manager["encoder_output"].getDepth(), memory_manager["encoder_output"].getHeight(), memory_manager["encoder_output"].getWidth());
    raul::TensorFP16 input_lengths(BATCH_SIZE, memory_manager["input_lengths"].getDepth(), memory_manager["input_lengths"].getHeight(), memory_manager["input_lengths"].getWidth());
    raul::TensorFP16 inputs(BATCH_SIZE, memory_manager["inputs"].getDepth(), memory_manager["inputs"].getHeight(), memory_manager["inputs"].getWidth());
    raul::TensorFP16 duration_targets(BATCH_SIZE, memory_manager["duration_targets"].getDepth(), memory_manager["duration_targets"].getHeight(), memory_manager["duration_targets"].getWidth());
    raul::TensorFP16 targets_lengths(BATCH_SIZE, memory_manager["targets_lengths"].getDepth(), memory_manager["targets_lengths"].getHeight(), memory_manager["targets_lengths"].getWidth());
    raul::TensorFP16 mel_targets(BATCH_SIZE, memory_manager["mel_targets"].getDepth(), memory_manager["mel_targets"].getHeight(), memory_manager["mel_targets"].getWidth());

    ASSERT_TRUE(loadTFData(testPath / "samarin" / "encoder_output.data", encoder_output));
    ASSERT_TRUE(loadTFData(testPath / "samarin" / "input_lengths.data", input_lengths));
    ASSERT_TRUE(loadTFData(testPath / "samarin" / "inputs.data", inputs));
    ASSERT_TRUE(loadTFData(testPath / "samarin" / "duration_targets.data", duration_targets));
    ASSERT_TRUE(loadTFData(testPath / "samarin" / "targets_lengths.data", targets_lengths));
    ASSERT_TRUE(loadTFData(testPath / "samarin" / "mel_targets.data", mel_targets));
    elapsed = TODTYPE(chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - timeStart).count());
    cout << "Training data loading time " << elapsed << " ms for batch size " << BATCH_SIZE << endl;

    auto optimizer = createOptimizer(trainParams, ADAM_BETA1, ADAM_BETA2, ADAM_EPSILON);

    dtype fastestEpochTime = numeric_limits<dtype>::max();
    dtype slowestEpochRime = 0_dt;
    dtype trainingTime = 0;

    for (size_t epoch = 0; epoch < nEpoch; ++epoch)
    {
        cout << "Epoch " << epoch << endl;
        timeStart = chrono::steady_clock::now();

        dtype lossValue = 0_dt;
        dtype beforeLossValue = 0_dt;
        dtype afterLossValue = 0_dt;
        dtype durationLossValue = 0_dt;

        std::fill(memory_manager["encoder_output"].begin(), memory_manager["encoder_output"].end(), 0_hf);
        std::fill(memory_manager["input_lengths"].begin(), memory_manager["input_lengths"].end(), 0_hf);
        std::fill(memory_manager["inputs"].begin(), memory_manager["inputs"].end(), 0_hf);
        std::fill(memory_manager["duration_targets"].begin(), memory_manager["duration_targets"].end(), 0_hf);
        std::fill(memory_manager["targets_lengths"].begin(), memory_manager["targets_lengths"].end(), 0_hf);
        std::fill(memory_manager["mel_targets"].begin(), memory_manager["mel_targets"].end(), 0_hf);

        for (size_t microBatchIdx = 0; microBatchIdx < numberOfMicroBatches; ++microBatchIdx)
        {
            size_t mbs = MICRO_BATCH_SIZE;
            if(microBatchIdx == numberOfMicroBatches - 1) mbs = MICRO_BATCH_SIZE - (MICRO_BATCH_SIZE * numberOfMicroBatches - BATCH_SIZE);

            {
                const size_t elementsInSample = encoder_output.size() / encoder_output.getBatchSize();
                const size_t elementsInMicroBatch = elementsInSample * mbs;
                std::copy(encoder_output.begin() + microBatchIdx * elementsInMicroBatch, encoder_output.begin() + (microBatchIdx + 1) * elementsInMicroBatch, memory_manager["encoder_output"].begin());
            }

            {
                const size_t elementsInSample = input_lengths.size() / input_lengths.getBatchSize();
                const size_t elementsInMicroBatch = elementsInSample * mbs;
                std::copy(input_lengths.begin() + microBatchIdx * elementsInMicroBatch, input_lengths.begin() + (microBatchIdx + 1) * elementsInMicroBatch, memory_manager["input_lengths"].begin());
            }

            {
                const size_t elementsInSample = inputs.size() / inputs.getBatchSize();
                const size_t elementsInMicroBatch = elementsInSample * mbs;
                std::copy(inputs.begin() + microBatchIdx * elementsInMicroBatch, inputs.begin() + (microBatchIdx + 1) * elementsInMicroBatch, memory_manager["inputs"].begin());
            }

            {
                const size_t elementsInSample = duration_targets.size() / duration_targets.getBatchSize();
                const size_t elementsInMicroBatch = elementsInSample * mbs;
                std::copy(duration_targets.begin() + microBatchIdx * elementsInMicroBatch, duration_targets.begin() + (microBatchIdx + 1) * elementsInMicroBatch, memory_manager["duration_targets"].begin());
            }

            {
                const size_t elementsInSample = targets_lengths.size() / targets_lengths.getBatchSize();
                const size_t elementsInMicroBatch = elementsInSample * mbs;
                std::copy(targets_lengths.begin() + microBatchIdx * elementsInMicroBatch, targets_lengths.begin() + (microBatchIdx + 1) * elementsInMicroBatch, memory_manager["targets_lengths"].begin());
            }

            {
                const size_t elementsInSample = mel_targets.size() / mel_targets.getBatchSize();
                const size_t elementsInMicroBatch = elementsInSample * mbs;
                std::copy(mel_targets.begin() + microBatchIdx * elementsInMicroBatch, mel_targets.begin() + (microBatchIdx + 1) * elementsInMicroBatch, memory_manager["mel_targets"].begin());
            }

            if (microBatchIdx == 0)
            {
                work.forwardPassTraining(true); // zero gradients
            }
            else
            {
                work.forwardPassTraining(false); // keep gradients to accumulate result
            }

            work.backwardPassTraining();

            lossValue += memory_manager_fp32["loss"][0];
            beforeLossValue += memory_manager_fp32["before_loss"][0];
            afterLossValue += memory_manager_fp32["after_loss"][0];
            durationLossValue += memory_manager_fp32["duration_loss"][0];

        }

        lossValue /= TODTYPE(numberOfMicroBatches);
        beforeLossValue /= TODTYPE(numberOfMicroBatches);
        afterLossValue /= TODTYPE(numberOfMicroBatches);
        durationLossValue /= TODTYPE(numberOfMicroBatches);

        auto trainableParamsFP16 = work.getTrainableParametersSafe<MemoryManagerFP16>();
        auto trainableParamsFP32 = work.getTrainableParametersSafe();

        clip.processGradientsMixedPrecision(trainableParamsFP32, trainableParamsFP16, work.getNetworkParameters());

        auto gnorm_before = clip.getGlobalNorm(work.getNetworkParameters());
        auto gnorm_after = clip.calcGlobalNormMixedPrecision(trainableParamsFP32, trainableParamsFP16, work.getNetworkParameters());

        if (trainParams.decayLearningRate)
        {
            optimizer->step();
        }

        for (auto& p : trainableParamsFP16)
        {
            optimizer->operator()(memory_manager, p.Param, p.Gradient);
        }

        for (auto& p : trainableParamsFP32)
        {
            optimizer->operator()(memory_manager_fp32, p.Param, p.Gradient);
        }

        elapsed = TODTYPE(chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - timeStart).count());

        trainingTime += elapsed;

        fastestEpochTime = std::min(fastestEpochTime, elapsed);
        slowestEpochRime = std::max(slowestEpochRime, elapsed);

        cout << "  Epoch time " << elapsed << " ms for micro batch size " << MICRO_BATCH_SIZE << endl;
        cout << "  Average performance: " << trainingTime / TODTYPE(epoch + 1) << " ms/epoch (fastest: " << fastestEpochTime << ", slowest " << slowestEpochRime << ")" << endl << endl;
        cout << "  Loss = " << lossValue << " (before = " << beforeLossValue << ", after = " << afterLossValue << ", duration = " << durationLossValue << ")" << endl;

        cout << "  Global grad norm: " << gnorm_before << " -> " << gnorm_after << endl;

        if (nEpochToSave > 0 && ((epoch + 1) % nEpochToSave == 0 || epoch == nEpoch - 1))
        {
            //[[maybe_unused]] size_t saved = saveTacotronParams(outputPrefix / to_string(epoch + 1), "230000_Tacotron_model.", memory_manager, "T", params, false);
        }
    }
}
#endif

TEST(TestTacotronMsk, DISABLED_SamarinTrainableEmbeddingGpuTraining)
{
    PROFILE_TEST

    GPU_ONLY_TEST

    using namespace tacotron;
    using namespace optimizers;

    random::setGlobalSeed(42);

    auto testPath = tools::getTestAssetsDir() / "Tacotron_Msk_RP";

    constexpr size_t GOLDEN_PARAMS = 59;
    constexpr size_t GOLDEN_TRAINABLE_PARAMS = 51;

    bool mixedPrecision = false;

    const bool usePool = UT::tools::getArg("use_pool", true);
    cout << "Using pool: " << (usePool ? "yes" : "no") << endl;

    Workflow work(CompressionMode::NONE, CalculationMode::DETERMINISTIC, usePool ? AllocationMode::POOL : AllocationMode::STANDARD, ExecutionTarget::GPU);

    constexpr dtype ADAM_BETA1 = 0.9_dt;
    constexpr dtype ADAM_BETA2 = 0.999_dt;
    constexpr dtype ADAM_EPSILON = 1e-6_dt;

    string experimentName = "samarin_gpu";
    auto outputPrefix = std::filesystem::path("Tacotron_Msk_RP") / experimentName;
    auto weightsPrefix = filesystem::path("weights_250000") / "250000_Tacotron_model.";

    constexpr size_t T_IN = 68;
    constexpr size_t BATCH_SIZE = 11;

    TacotronParams params({ "encoder_output", "duration_targets", "inputs", "input_lengths", "mel_targets" },
                          { "decoder_output", "mel_outputs", "duration_outputs" },
                          {},
                          mixedPrecision ? LayerExecutionTarget::CPUFP16 : LayerExecutionTarget::Default);
    params.maxMelFrames = 564;
    params.useDurationPrediction = true;
    params.attentionType = "None";
    params.durationPredictorLstmUnits = 256;
    params.postnetKernelSize = { 5, 3, 3, 3 };
    params.postnetChannels = 256;
    params.prenetLayers = { 128, 128 };
    params.dropoutRate = 0.5f;
    params.zoneoutRate = 0.1f;
    params.postnetDropoutRate = 0.1f;
    params.trainableSpeakerEmbedding = true;
    params.withoutStopTokenLoss = true;
    params.lossMultipliers = { 1.f, 1.f, 2.f };
    params.useResidualRnn = true;

    TacotronTrainingParams trainParams;
    trainParams.initialLearningRate = 1e-3_dt;
    trainParams.clipNorm = 1.0_dt;
    trainParams.decayLearningRate = true;
    trainParams.decaySteps = 500;
    trainParams.decayStartStep = 100;
    trainParams.decayRate = 0.5;
    trainParams.finalLearningRate = 1e-5_dt;
    trainParams.warmupLearningRate = true;
    trainParams.warmupSteps = 1000;

    size_t nEpoch = 400;
    const size_t nEpochToSave = 50;

    work.add<DataLayer>("inputs", DataParams{ { "inputs" }, 1u, T_IN, 1u });
    work.add<DataLayer>("duration_targets", DataParams{ { "duration_targets" }, 1u, 1u, T_IN });
    work.add<DataLayer>("encoder_output", DataParams{ { "encoder_output" }, 1u, T_IN, params.embeddingDim });
    work.add<DataLayer>("input_lengths", DataParams{ { "input_lengths" }, 1u, 1u, 1u });
    work.add<DataLayer>("mel_targets", DataParams{ { "mel_targets" }, 1u, params.maxMelFrames, params.numMels });
    work.add<DataLayer>("targets_lengths", DataParams{ { "targets_lengths" }, 1u, 1u, 1u });

    AddSingleSpeakerFslTacotronModel(&work, "T", params);

    tacotron::AddTacotronLoss(
        &work,
        "total_loss",
        { { "decoder_output", "mel_outputs", "duration_outputs", "mel_targets", "targets_lengths", "duration_targets" }, { "loss", "before_loss", "after_loss", "duration_loss" } },
        params);

    auto paramsCount = tacotronParamsCount(params);

    ASSERT_EQ(paramsCount.first, GOLDEN_TRAINABLE_PARAMS);
    ASSERT_EQ(paramsCount.second, GOLDEN_PARAMS);

    auto timeStart = chrono::steady_clock::now();
    TENSORS_CREATE(BATCH_SIZE);
    auto elapsed = TODTYPE(chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - timeStart).count());
    cout << "Workflow preparation time " << elapsed << " ms for batch size " << BATCH_SIZE << endl;

    ASSERT_EQ(work.getTrainableParameterNames().size(), GOLDEN_TRAINABLE_PARAMS);

    auto& memory_manager = work.getMemoryManager<MemoryManagerGPU>();
    postprocessing::GradientClipping clip(1_dt);

    // Data loading

    timeStart = chrono::steady_clock::now();
    size_t loaded = loadTacotronParams((testPath / weightsPrefix).string(), memory_manager, "T", params, true, false);
    ASSERT_EQ(loaded, GOLDEN_PARAMS - 1); // speaker embedding will be loaded separately
    ASSERT_TRUE(loadTFData(testPath / "samarin" / "speaker_embedding.data", memory_manager[Name("T") / "speaker_embedding"]));
    elapsed = TODTYPE(chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - timeStart).count());
    cout << "Weights loading time " << elapsed << " ms for batch size " << BATCH_SIZE << endl;

    timeStart = chrono::steady_clock::now();
    ASSERT_TRUE(loadTFData(testPath / "samarin" / "encoder_output.data", memory_manager["encoder_output"]));
    ASSERT_TRUE(loadTFData(testPath / "samarin" / "input_lengths.data", memory_manager["input_lengths"]));
    ASSERT_TRUE(loadTFData(testPath / "samarin" / "inputs.data", memory_manager["inputs"]));
    ASSERT_TRUE(loadTFData(testPath / "samarin" / "duration_targets.data", memory_manager["duration_targets"]));
    ASSERT_TRUE(loadTFData(testPath / "samarin" / "targets_lengths.data", memory_manager["targets_lengths"]));
    ASSERT_TRUE(loadTFData(testPath / "samarin" / "mel_targets.data", memory_manager["mel_targets"]));
    elapsed = TODTYPE(chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - timeStart).count());
    cout << "Training data loading time " << elapsed << " ms for batch size " << BATCH_SIZE << endl;

    auto optimizer = createOptimizer(trainParams, ADAM_BETA1, ADAM_BETA2, ADAM_EPSILON);

    dtype fastestEpochTime = numeric_limits<dtype>::max();
    dtype slowestEpochRime = 0_dt;
    dtype trainingTime = 0;

    for (size_t epoch = 0; epoch < nEpoch; ++epoch)
    {
        cout << "Epoch " << epoch << endl;
        timeStart = chrono::steady_clock::now();
        work.forwardPassTraining();

        work.backwardPassTraining();

        auto trainableParams = work.getTrainableParameters<MemoryManagerGPU>();
        
        clip.processGradients(trainableParams, work.getNetworkParameters());
        auto gnorm_before = clip.getGlobalNorm(work.getNetworkParameters());
        auto gnorm_after = clip.calcGlobalNorm(trainableParams, work.getNetworkParameters());
        
        if (trainParams.decayLearningRate)
        {
            optimizer->step();
        }

        for (auto p : trainableParams)
        {
            optimizer->operator()(work.getKernelManager(), memory_manager, p.Param, p.Gradient);
        }

        const Tensor lossValue = memory_manager["loss"];
        const Tensor bLoss = memory_manager["before_loss"];
        const Tensor aLoss = memory_manager["after_loss"];
        const Tensor dLoss = memory_manager["duration_loss"];
        elapsed = TODTYPE(chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - timeStart).count());

        trainingTime += elapsed;

        fastestEpochTime = std::min(fastestEpochTime, elapsed);
        slowestEpochRime = std::max(slowestEpochRime, elapsed);

        cout << "  Epoch time " << elapsed << " ms for batch size " << BATCH_SIZE << endl;
        cout << "  Average performance: " << trainingTime / TODTYPE(epoch + 1) << " ms/epoch (fastest: " << fastestEpochTime << ", slowest " << slowestEpochRime << ")" << endl << endl;
        cout << "  Loss = " << lossValue[0] << " (before = " << bLoss[0] << ", after = " << aLoss[0]
             << ", duration = " << dLoss[0] << ")" << endl;

        cout << "  Global grad norm: " << gnorm_before << " -> " << gnorm_after << endl;

        if (nEpochToSave > 0 && ((epoch + 1) % nEpochToSave == 0 || epoch == nEpoch - 1))
        {
            [[maybe_unused]] size_t saved = saveTacotronParams(outputPrefix / to_string(epoch + 1), "250000_Tacotron_model.", memory_manager, "T", params, false);
        }
    }
}

TEST(TestTacotronMsk, DISABLED_SamarinTrainableEmbeddingMicrobatchingTraining)
{
    PROFILE_TEST

    using namespace tacotron;
    using namespace optimizers;

    random::setGlobalSeed(42);

    auto testPath = tools::getTestAssetsDir() / "Tacotron_Msk_RP";

    constexpr size_t GOLDEN_PARAMS = 59;
    constexpr size_t GOLDEN_TRAINABLE_PARAMS = 51;

    bool mixedPrecision = false;

    bool usePool = true;

    Workflow work(CompressionMode::NONE, CalculationMode::DETERMINISTIC, usePool ? AllocationMode::POOL : AllocationMode::STANDARD);


    constexpr dtype ADAM_BETA1 = 0.9_dt;
    constexpr dtype ADAM_BETA2 = 0.999_dt;
    constexpr dtype ADAM_EPSILON = 1e-6_dt;

    string experimentName = "samarin";
    auto outputPrefix = std::filesystem::path("Tacotron_Msk_RP") / experimentName;

    constexpr size_t T_IN = 68;
    constexpr size_t BATCH_SIZE = 11;
    constexpr size_t MICRO_BATCH_SIZE = 6;
    const size_t numberOfMicroBatches = static_cast<size_t>(static_cast<float>(BATCH_SIZE) / static_cast<float>(MICRO_BATCH_SIZE) + 0.5f);

    TacotronParams params({ "encoder_output", "duration_targets", "inputs", "input_lengths", "mel_targets" },
                          { "decoder_output", "mel_outputs", "duration_outputs" },
                          {},
                          mixedPrecision ? LayerExecutionTarget::CPUFP16 : LayerExecutionTarget::Default);
    params.maxMelFrames = 564;
    params.useDurationPrediction = true;
    params.attentionType = "None";
    params.durationPredictorLstmUnits = 256;
    params.postnetKernelSize = { 5, 3, 3, 3 };
    params.postnetChannels = 256;
    params.prenetLayers = { 128, 128 };
    params.dropoutRate = 0.5f;
    params.zoneoutRate = 0.1f;
    params.postnetDropoutRate = 0.1f;
    params.trainableSpeakerEmbedding = true;
    params.withoutStopTokenLoss = true;
    params.lossMultipliers = { 1.f, 1.f, 2.f };
    params.useResidualRnn = true;

    TacotronTrainingParams trainParams;
    trainParams.initialLearningRate = 1e-3_dt;
    trainParams.clipNorm = 1.0_dt;
    trainParams.decayLearningRate = true;
    trainParams.decaySteps = 500;
    trainParams.decayStartStep = 100;
    trainParams.decayRate = 0.5;
    trainParams.finalLearningRate = 1e-5_dt;
    trainParams.warmupLearningRate = true;
    trainParams.warmupSteps = 1000;

    size_t nEpoch = 400;
    const size_t nEpochToSave = 50;

    work.add<DataLayer>("inputs", DataParams{ { "inputs" }, 1u, T_IN, 1u });
    work.add<DataLayer>("duration_targets", DataParams{ { "duration_targets" }, 1u, 1u, T_IN });
    work.add<DataLayer>("encoder_output", DataParams{ { "encoder_output" }, 1u, T_IN, params.embeddingDim });
    work.add<DataLayer>("input_lengths", DataParams{ { "input_lengths" }, 1u, 1u, 1u });
    work.add<DataLayer>("mel_targets", DataParams{ { "mel_targets" }, 1u, params.maxMelFrames, params.numMels });
    work.add<DataLayer>("targets_lengths", DataParams{ { "targets_lengths" }, 1u, 1u, 1u });

    AddSingleSpeakerFslTacotronModel(&work, "T", params);

    tacotron::AddTacotronLoss(
        &work,
        "total_loss",
        { { "decoder_output", "mel_outputs", "duration_outputs", "mel_targets", "targets_lengths", "duration_targets" }, { "loss", "before_loss", "after_loss", "duration_loss" } },
        params);

    auto paramsCount = tacotronParamsCount(params);

    ASSERT_EQ(paramsCount.first, GOLDEN_TRAINABLE_PARAMS);
    ASSERT_EQ(paramsCount.second, GOLDEN_PARAMS);

    auto timeStart = chrono::steady_clock::now();
    TENSORS_CREATE(MICRO_BATCH_SIZE);
    auto elapsed = TODTYPE(chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - timeStart).count());
    cout << "Workflow preparation time " << elapsed << " ms for micro batch size " << MICRO_BATCH_SIZE << endl;

    ASSERT_EQ(work.getTrainableParameterNames().size(), GOLDEN_TRAINABLE_PARAMS);
    cout << tools::get_size_of_trainable_params_mixed_precision(work) << " trainable weights" << endl;

    auto& memory_manager = work.getMemoryManager();
    postprocessing::GradientClipping clip(1_dt);

    // Data loading

    timeStart = chrono::steady_clock::now();
    size_t loaded = loadTacotronParams((testPath / "weights" / "small_v2_230_Tacotron_model.").string(), memory_manager, "T", params, true, false);
    ASSERT_EQ(loaded, GOLDEN_PARAMS - 1); // speaker embedding will be loaded separately
    ASSERT_TRUE(loadTFData(testPath / "samarin" / "speaker_embedding.data", memory_manager[Name("T") / "speaker_embedding"]));
    elapsed = TODTYPE(chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - timeStart).count());
    cout << "Weights loading time " << elapsed << " ms for micro batch size " << MICRO_BATCH_SIZE << endl;

    timeStart = chrono::steady_clock::now();
    raul::Tensor encoder_output(BATCH_SIZE, memory_manager["encoder_output"].getDepth(), memory_manager["encoder_output"].getHeight(), memory_manager["encoder_output"].getWidth());
    raul::Tensor input_lengths(BATCH_SIZE, memory_manager["input_lengths"].getDepth(), memory_manager["input_lengths"].getHeight(), memory_manager["input_lengths"].getWidth());
    raul::Tensor inputs(BATCH_SIZE, memory_manager["inputs"].getDepth(), memory_manager["inputs"].getHeight(), memory_manager["inputs"].getWidth());
    raul::Tensor duration_targets(BATCH_SIZE, memory_manager["duration_targets"].getDepth(), memory_manager["duration_targets"].getHeight(), memory_manager["duration_targets"].getWidth());
    raul::Tensor targets_lengths(BATCH_SIZE, memory_manager["targets_lengths"].getDepth(), memory_manager["targets_lengths"].getHeight(), memory_manager["targets_lengths"].getWidth());
    raul::Tensor mel_targets(BATCH_SIZE, memory_manager["mel_targets"].getDepth(), memory_manager["mel_targets"].getHeight(), memory_manager["mel_targets"].getWidth());

    ASSERT_TRUE(loadTFData(testPath / "samarin" / "encoder_output.data", encoder_output));
    ASSERT_TRUE(loadTFData(testPath / "samarin" / "input_lengths.data", input_lengths));
    ASSERT_TRUE(loadTFData(testPath / "samarin" / "inputs.data", inputs));
    ASSERT_TRUE(loadTFData(testPath / "samarin" / "duration_targets.data", duration_targets));
    ASSERT_TRUE(loadTFData(testPath / "samarin" / "targets_lengths.data", targets_lengths));
    ASSERT_TRUE(loadTFData(testPath / "samarin" / "mel_targets.data", mel_targets));
    elapsed = TODTYPE(chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - timeStart).count());
    cout << "Training data loading time " << elapsed << " ms for micro batch size " << MICRO_BATCH_SIZE << endl;

    auto optimizer = createOptimizer(trainParams, ADAM_BETA1, ADAM_BETA2, ADAM_EPSILON);

    dtype fastestEpochTime = numeric_limits<dtype>::max();
    dtype slowestEpochRime = 0_dt;
    dtype trainingTime = 0;

    for (size_t epoch = 0; epoch < nEpoch; ++epoch)
    {
        cout << "Epoch " << epoch << endl;
        timeStart = chrono::steady_clock::now();

        dtype lossValue = 0_dt;
        dtype beforeLossValue = 0_dt;
        dtype afterLossValue = 0_dt;
        dtype durationLossValue = 0_dt;

        std::fill(memory_manager["encoder_output"].begin(), memory_manager["encoder_output"].end(), 0_dt);
        std::fill(memory_manager["input_lengths"].begin(), memory_manager["input_lengths"].end(), 0_dt);
        std::fill(memory_manager["inputs"].begin(), memory_manager["inputs"].end(), 0_dt);
        std::fill(memory_manager["duration_targets"].begin(), memory_manager["duration_targets"].end(), 0_dt);
        std::fill(memory_manager["targets_lengths"].begin(), memory_manager["targets_lengths"].end(), 0_dt);
        std::fill(memory_manager["mel_targets"].begin(), memory_manager["mel_targets"].end(), 0_dt);

        for (size_t microBatchIdx = 0; microBatchIdx < numberOfMicroBatches; ++microBatchIdx)
        {
            size_t mbs = MICRO_BATCH_SIZE;
            if(microBatchIdx == numberOfMicroBatches - 1) mbs = MICRO_BATCH_SIZE - (MICRO_BATCH_SIZE * numberOfMicroBatches - BATCH_SIZE);

            {
                const size_t elementsInSample = encoder_output.size() / encoder_output.getBatchSize();
                const size_t elementsInMicroBatch = elementsInSample * mbs;
                std::copy(encoder_output.begin() + microBatchIdx * elementsInMicroBatch, encoder_output.begin() + (microBatchIdx + 1) * elementsInMicroBatch, memory_manager["encoder_output"].begin());
            }

            {
                const size_t elementsInSample = input_lengths.size() / input_lengths.getBatchSize();
                const size_t elementsInMicroBatch = elementsInSample * mbs;
                std::copy(input_lengths.begin() + microBatchIdx * elementsInMicroBatch, input_lengths.begin() + (microBatchIdx + 1) * elementsInMicroBatch, memory_manager["input_lengths"].begin());
            }

            {
                const size_t elementsInSample = inputs.size() / inputs.getBatchSize();
                const size_t elementsInMicroBatch = elementsInSample * mbs;
                std::copy(inputs.begin() + microBatchIdx * elementsInMicroBatch, inputs.begin() + (microBatchIdx + 1) * elementsInMicroBatch, memory_manager["inputs"].begin());
            }

            {
                const size_t elementsInSample = duration_targets.size() / duration_targets.getBatchSize();
                const size_t elementsInMicroBatch = elementsInSample * mbs;
                std::copy(duration_targets.begin() + microBatchIdx * elementsInMicroBatch, duration_targets.begin() + (microBatchIdx + 1) * elementsInMicroBatch, memory_manager["duration_targets"].begin());
            }

            {
                const size_t elementsInSample = targets_lengths.size() / targets_lengths.getBatchSize();
                const size_t elementsInMicroBatch = elementsInSample * mbs;
                std::copy(targets_lengths.begin() + microBatchIdx * elementsInMicroBatch, targets_lengths.begin() + (microBatchIdx + 1) * elementsInMicroBatch, memory_manager["targets_lengths"].begin());
            }

            {
                const size_t elementsInSample = mel_targets.size() / mel_targets.getBatchSize();
                const size_t elementsInMicroBatch = elementsInSample * mbs;
                std::copy(mel_targets.begin() + microBatchIdx * elementsInMicroBatch, mel_targets.begin() + (microBatchIdx + 1) * elementsInMicroBatch, memory_manager["mel_targets"].begin());
            }

            if (microBatchIdx == 0)
            {
                work.forwardPassTraining(true); // zero gradients
            }
            else
            {
                work.forwardPassTraining(false); // keep gradients to accumulate result
            }

            work.backwardPassTraining();

            lossValue += memory_manager["loss"][0];
            beforeLossValue += memory_manager["before_loss"][0];
            afterLossValue += memory_manager["after_loss"][0];
            durationLossValue += memory_manager["duration_loss"][0];

        }

        lossValue /= TODTYPE(numberOfMicroBatches);
        beforeLossValue /= TODTYPE(numberOfMicroBatches);
        afterLossValue /= TODTYPE(numberOfMicroBatches);
        durationLossValue /= TODTYPE(numberOfMicroBatches);

        auto names = work.getTrainableParameterNames();

        std::vector<ParamAndGrad> trainableParams;

        for (auto& name : names)
        {
            if (memory_manager.tensorExists(name) && memory_manager.tensorExists(name.grad()))
            {
                trainableParams.push_back(raul::ParamAndGrad{ memory_manager[name], memory_manager[name.grad()] });
            }
        }

        clip.processGradients(trainableParams, work.getNetworkParameters());
        auto gnorm_before = clip.getGlobalNorm(work.getNetworkParameters());
        auto gnorm_after = clip.calcGlobalNorm(trainableParams, work.getNetworkParameters());

        if (trainParams.decayLearningRate)
        {
            optimizer->step();
        }

        for (auto name : names)
        {
            optimizer->operator()(memory_manager, memory_manager[name], memory_manager[name.grad()]);
        }

        elapsed = TODTYPE(chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - timeStart).count());

        trainingTime += elapsed;

        fastestEpochTime = std::min(fastestEpochTime, elapsed);
        slowestEpochRime = std::max(slowestEpochRime, elapsed);

        cout << "  Epoch time " << elapsed << " ms for micro batch size " << MICRO_BATCH_SIZE << endl;
        cout << "  Average performance: " << trainingTime / TODTYPE(epoch + 1) << " ms/epoch (fastest: " << fastestEpochTime << ", slowest " << slowestEpochRime << ")" << endl << endl;
        cout << "  Loss = " << lossValue << " (before = " << beforeLossValue << ", after = " << afterLossValue << ", duration = " << durationLossValue << ")" << endl;


        cout << "  Global grad norm: " << gnorm_before << " -> " << gnorm_after << endl;

        if (nEpochToSave > 0 && ((epoch + 1) % nEpochToSave == 0 || epoch == nEpoch - 1))
        {
            [[maybe_unused]] size_t saved = saveTacotronParams(outputPrefix / to_string(epoch + 1), "230000_Tacotron_model.", memory_manager, "T", params, false);
        }
    }
}

TEST(TestTacotronMsk, TacotronLossUnit)
{
    PROFILE_TEST
    using namespace tacotron;

    auto testPath = tools::getTestAssetsDir() / "Tacotron_Msk_RP";
    constexpr size_t BATCH_SIZE = 12;
    constexpr dtype EPS = 1e-4_dt;

    TacotronParams params({ "encoder_output", "speaker_embedding", "duration_targets", "inputs", "input_lengths", "mel_targets" }, { "decoder_output", "mel_output" }, {});
    params.useDurationPrediction = true;
    params.attentionType = "";
    params.trainableSpeakerEmbedding = false;
    params.durationPredictorLstmUnits = 256;
    params.postnetKernelSize = { 5, 5, 5, 5, 3 };
    params.prenetLayers = { 256, 256 };
    params.withoutStopTokenLoss = true;
    params.lossMultipliers = { 1.f, 1.f, 2.f };
    params.maxMelFrames = 666;

    Workflow work;
    constexpr size_t T_IN = 86;

    work.add<DataLayer>("decoder_output", DataParams{ { "decoder_output" }, 1u, params.maxMelFrames, params.numMels });
    work.add<DataLayer>("mel_outputs", DataParams{ { "mel_outputs" }, 1u, params.maxMelFrames, params.numMels });
    work.add<DataLayer>("duration_outputs", DataParams{ { "duration_outputs" }, 1u, 1u, T_IN });
    work.add<DataLayer>("mel_targets", DataParams{ { "mel_targets" }, 1u, params.maxMelFrames, params.numMels });
    work.add<DataLayer>("duration_targets", DataParams{ { "duration_targets" }, 1u, 1u, T_IN });
    work.add<DataLayer>("targets_lengths", DataParams{ { "targets_lengths" }, 1u, 1u, 1u });
    work.add<TensorLayer>("golden_loss", TensorParams{ { "golden_loss" }, WShape{ 1, 1, 1, 4 } });

    tacotron::AddTacotronLoss(
        &work,
        "total_loss",
        { { "decoder_output", "mel_outputs", "duration_outputs", "mel_targets", "targets_lengths", "duration_targets" }, { "loss", "before_loss", "after_loss", "duration_loss" } },
        params);

    TENSORS_CREATE(BATCH_SIZE);
    MemoryManager& memory_manager = work.getMemoryManager();

    EXPECT_TRUE(loadTFData(testPath / "duration_targets.data", memory_manager["duration_targets"]));
    EXPECT_TRUE(loadTFData(testPath / "targets_lengths.data", memory_manager["targets_lengths"]));
    EXPECT_TRUE(loadTFData(testPath / "duration_predictor" / "golden_durations.data", memory_manager["duration_outputs"]));
    EXPECT_TRUE(loadTFData(testPath / "mel_targets.data", memory_manager["mel_targets"]));
    EXPECT_TRUE(loadTFData(testPath / "mel_outputs.data", memory_manager["mel_outputs"]));
    EXPECT_TRUE(loadTFData(testPath / "decoder_output.data", memory_manager["decoder_output"]));
    EXPECT_TRUE(loadTFData(testPath / "loss.data", memory_manager["golden_loss"]));

    ASSERT_NO_THROW(work.forwardPassTraining());

    auto& loss = memory_manager["loss"];

    ASSERT_EQ(loss.size(), 1u);
    ASSERT_EQ(memory_manager["before_loss"].size(), 1u);
    ASSERT_EQ(memory_manager["after_loss"].size(), 1u);
    ASSERT_EQ(memory_manager["duration_loss"].size(), 1u);

    auto bloss = memory_manager["before_loss"][0];
    auto aloss = memory_manager["after_loss"][0];
    auto dloss = memory_manager["duration_loss"][0];

    EXPECT_NEAR(loss[0], memory_manager["golden_loss"][0], EPS);
    EXPECT_NEAR(bloss, memory_manager["golden_loss"][1], EPS);
    EXPECT_NEAR(aloss, memory_manager["golden_loss"][2], EPS);
    EXPECT_NEAR(dloss, memory_manager["golden_loss"][3], EPS);

    ASSERT_NO_THROW(work.backwardPassTraining());
}

TEST(TestTacotronMsk, TacotronLossGpuUnit)
{
    PROFILE_TEST

    GPU_ONLY_TEST

    using namespace tacotron;

    auto testPath = tools::getTestAssetsDir() / "Tacotron_Msk_RP";
    constexpr size_t BATCH_SIZE = 12;
    constexpr dtype EPS = 1e-4_dt;

    TacotronParams params({ "encoder_output", "speaker_embedding", "duration_targets", "inputs", "input_lengths", "mel_targets" }, { "decoder_output", "mel_output" }, {});
    params.useDurationPrediction = true;
    params.attentionType = "";
    params.trainableSpeakerEmbedding = false;
    params.durationPredictorLstmUnits = 256;
    params.postnetKernelSize = { 5, 5, 5, 5, 3 };
    params.prenetLayers = { 256, 256 };
    params.withoutStopTokenLoss = true;
    params.lossMultipliers = { 1.f, 1.f, 2.f };
    params.maxMelFrames = 666;

    WorkflowEager work(CompressionMode::NONE, CalculationMode::DETERMINISTIC, AllocationMode::STANDARD, ExecutionTarget::GPU);
    auto& memory_manager = work.getMemoryManager<MemoryManagerGPU>();
    constexpr size_t T_IN = 86;

    work.add<DataLayer>("decoder_output", DataParams{ { "decoder_output" }, 1u, params.maxMelFrames, params.numMels });
    work.add<DataLayer>("mel_outputs", DataParams{ { "mel_outputs" }, 1u, params.maxMelFrames, params.numMels });
    work.add<DataLayer>("duration_outputs", DataParams{ { "duration_outputs" }, 1u, 1u, T_IN });
    work.add<DataLayer>("mel_targets", DataParams{ { "mel_targets" }, 1u, params.maxMelFrames, params.numMels });
    work.add<DataLayer>("duration_targets", DataParams{ { "duration_targets" }, 1u, 1u, T_IN });
    work.add<DataLayer>("targets_lengths", DataParams{ { "targets_lengths" }, 1u, 1u, 1u });
    work.add<TensorLayer>("golden_loss", TensorParams{ { "golden_loss" }, WShape{ 1, 1, 1, 4 } });

    tacotron::AddTacotronLoss(
        &work,
        "total_loss",
        { { "decoder_output", "mel_outputs", "duration_outputs", "mel_targets", "targets_lengths", "duration_targets" }, { "loss", "before_loss", "after_loss", "duration_loss" } },
        params);

    TENSORS_CREATE(BATCH_SIZE);

    EXPECT_TRUE(loadTFData(testPath / "duration_targets.data", memory_manager["duration_targets"]));
    EXPECT_TRUE(loadTFData(testPath / "targets_lengths.data", memory_manager["targets_lengths"]));
    EXPECT_TRUE(loadTFData(testPath / "duration_predictor" / "golden_durations.data", memory_manager["duration_outputs"]));
    EXPECT_TRUE(loadTFData(testPath / "mel_targets.data", memory_manager["mel_targets"]));
    EXPECT_TRUE(loadTFData(testPath / "mel_outputs.data", memory_manager["mel_outputs"]));
    EXPECT_TRUE(loadTFData(testPath / "decoder_output.data", memory_manager["decoder_output"]));
    EXPECT_TRUE(loadTFData(testPath / "loss.data", memory_manager["golden_loss"]));

    ASSERT_NO_THROW(work.forwardPassTraining());

    Tensor loss = memory_manager["loss"];
    Tensor before_loss = memory_manager["before_loss"];
    Tensor after_loss = memory_manager["after_loss"];
    Tensor duration_loss = memory_manager["duration_loss"];
    Tensor golden_loss = memory_manager["golden_loss"];

    ASSERT_EQ(loss.size(), 1u);
    ASSERT_EQ(before_loss.size(), 1u);
    ASSERT_EQ(after_loss.size(), 1u);
    ASSERT_EQ(duration_loss.size(), 1u);

    auto bloss = before_loss[0];
    auto aloss = after_loss[0];
    auto dloss = duration_loss[0];

    EXPECT_NEAR(loss[0], golden_loss[0], EPS);
    EXPECT_NEAR(bloss, golden_loss[1], EPS);
    EXPECT_NEAR(aloss, golden_loss[2], EPS);
    EXPECT_NEAR(dloss, golden_loss[3], EPS);

    ASSERT_NO_THROW(work.backwardPassTraining());
}

}
