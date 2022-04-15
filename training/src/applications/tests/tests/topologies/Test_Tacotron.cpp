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
#include <tests/tools/callbacks/TensorChecker.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <set>

#include <tests/tools/TestTools.h>

#include <training/layers/basic/DataLayer.h>
#include <training/layers/basic/ElementWiseSumLayer.h>
#include <training/layers/basic/SlicerLayer.h>
#include <training/layers/basic/TensorLayer.h>
#include <training/layers/basic/trainable/LinearLayer.h>
#include <training/layers/composite/BahdanauMonotonicAttentionLayer.h>
#include <training/layers/composite/TacotronModel.h>
#include <training/layers/composite/rnn/LSTMCellLayer.h>
#include <training/layers/composite/tacotron/MaskedCrossEntropy.h>
#include <training/layers/composite/tacotron/MaskedLoss.h>
#include <training/layers/composite/tacotron/PostNet.h>
#include <training/layers/composite/tacotron/SequenceLoss.h>
#include <training/layers/composite/tacotron/TacotronDecoderCell.h>
#include <training/layers/composite/tacotron/TacotronDecoderRNN.h>
#include <training/layers/composite/tacotron/TacotronLoss.h>
#include <training/optimizers/SGD.h>
#include <training/postprocessing/GradientClipping.h>

#include <training/api/API.h>

#include "TacotronTestTools.h"
#include "TacotronTrainingParams.h"

namespace UT
{
using namespace raul;
using namespace UT::tools::callbacks;

TEST(TestTacotron, PostNetUnit)
{
    PROFILE_TEST

    using namespace tacotron;

    Workflow work;

    constexpr size_t BATCH_SIZE = 8;
    constexpr size_t POSTNET_PARAMS = 30;
    constexpr size_t POSTNET_TRAINABLE_PARAMS = 20;
    constexpr dtype EPS = 1e-4_dt;

    TacotronParams params({}, {}, {});
    params.postnetDropoutRate = 0.f;

    work.add<DataLayer>("data_input", DataParams{ { "input" }, 1u, params.maxMelFrames, params.numMels });
    work.add<DataLayer>("data_target", DataParams{ { "target" }, 1u, params.maxMelFrames, params.postnetChannels });
    tacotron::AddPostNet(&work, Name("T") / "postnet_convolutions", BasicParams{ { "input" }, { "output" } }, params);
    work.add<TensorLayer>(
        "grad",
        TensorParams{ { Name("output").grad() }, WShape{ BS(), 1u, params.maxMelFrames, params.postnetChannels }, 1_dt, Workflow::Usage::ForwardAndBackward, Workflow::Mode::Read, true, true });

    TENSORS_CREATE(BATCH_SIZE);
    MemoryManager& memory_manager = work.getMemoryManager();
    size_t loaded = loadTacotronParams((tools::getTestAssetsDir() / "Tacotron" / "pretrained" / "300000_Tacotron_model.").string(), memory_manager, "T", params, false);

    EXPECT_EQ(loaded, POSTNET_PARAMS);
    EXPECT_EQ(work.getTrainableParameterNames().size(), POSTNET_TRAINABLE_PARAMS);

    bool ok = loadTFData(tools::getTestAssetsDir() / "Tacotron" / "decoder_output.data", memory_manager["input"]);
    EXPECT_TRUE(ok);
    ok = loadTFData(tools::getTestAssetsDir() / "Tacotron" / "postnet_output.data", memory_manager["target"]);
    EXPECT_TRUE(ok);

    TensorChecker checker({ { "output", "target" } }, EPS);
    work.getNetworkParameters().mCallback = checker;

    vector<pair<string, size_t>> remainingGradients;

    for (const auto& t : memory_manager.getTensorCollection().tensors)
    {
        if (!t.second->empty() && Common::endsWith(t.first.str(), TENSOR_GRADIENT_POSTFIX))
        {
            remainingGradients.emplace_back(t.first.str(), t.second->size());
        }
    }

    ASSERT_NO_THROW(work.forwardPassTraining());

    remainingGradients.clear();

    for (const auto& t : memory_manager.getTensorCollection().tensors)
    {
        if (!t.second->empty() && Common::endsWith(t.first.str(), TENSOR_GRADIENT_POSTFIX))
        {
            remainingGradients.emplace_back(t.first.str(), t.second->size());
        }
    }

    ASSERT_NO_THROW(work.backwardPassTraining());

    remainingGradients.clear();

    for (const auto& t : memory_manager.getTensorCollection().tensors)
    {
        if (!t.second->empty() && Common::endsWith(t.first.str(), TENSOR_GRADIENT_POSTFIX))
        {
            remainingGradients.emplace_back(t.first.str(), t.second->size());
        }
    }
}

TEST(TestTacotron, PostNetGpuUnit)
{
    PROFILE_TEST

    GPU_ONLY_TEST

    using namespace tacotron;

    WorkflowEager work(CompressionMode::NONE, CalculationMode::DETERMINISTIC, AllocationMode::POOL, ExecutionTarget::GPU);
    auto& memory_manager = work.getMemoryManager<MemoryManagerGPU>();

    constexpr size_t BATCH_SIZE = 8;
    constexpr size_t POSTNET_PARAMS = 30;
    constexpr size_t POSTNET_TRAINABLE_PARAMS = 20;
    constexpr dtype EPS = 1e-4_dt;

    TacotronParams params({}, {}, {});
    params.postnetDropoutRate = 0.f;

    work.add<DataLayer>("data_input", DataParams{ { "input" }, 1u, params.maxMelFrames, params.numMels });
    work.add<DataLayer>("data_target", DataParams{ { "target" }, 1u, params.maxMelFrames, params.postnetChannels });
    tacotron::AddPostNet(&work, Name("T") / "postnet_convolutions", BasicParams{ { "input" }, { "output" } }, params);
    work.add<TensorLayer>(
        "grad",
        TensorParams{ { Name("output").grad() }, WShape{ BS(), 1u, params.maxMelFrames, params.postnetChannels }, 1_dt, Workflow::Usage::ForwardAndBackward, Workflow::Mode::Read, true, true });

    TENSORS_CREATE(BATCH_SIZE);
    size_t loaded = loadTacotronParams((tools::getTestAssetsDir() / "Tacotron" / "pretrained" / "300000_Tacotron_model.").string(), memory_manager, "T", params, false);

    EXPECT_EQ(loaded, POSTNET_PARAMS);
    EXPECT_EQ(work.getTrainableParameterNames().size(), POSTNET_TRAINABLE_PARAMS);

    bool ok = loadTFData(tools::getTestAssetsDir() / "Tacotron" / "decoder_output.data", memory_manager["input"]);
    EXPECT_TRUE(ok);
    ok = loadTFData(tools::getTestAssetsDir() / "Tacotron" / "postnet_output.data", memory_manager["target"]);
    EXPECT_TRUE(ok);    

    ASSERT_NO_THROW(work.forwardPassTraining());

    UT::tools::checkTensors({ { "output", "target" } }, memory_manager, EPS);

    ASSERT_NO_THROW(work.backwardPassTraining());
}

TEST(TestTacotron, BahdanauMonotonicAttentionUnit)
{
    PROFILE_TEST

    using namespace tacotron;

    Name battnParent = Name("T") / "decoder" / "_cell" / "attention_mechanism";
    Name battnChild = Name("T") / "decoder" / "_cell[1]" / "attention_mechanism";

    Workflow work;

    constexpr size_t BATCH_SIZE = 8;
    constexpr size_t DECODER_OUTPUT_SIZE = 512u;
    constexpr size_t ALIGNMENTS_SIZE = 230u;
    constexpr size_t EMBEDDED_ENCODER_OUTPUT_SIZE = 768u;
    constexpr size_t ATTENTION_TRAINABLE_PARAMS = 6;
    constexpr dtype EPS = 1e-6_dt;

    [[maybe_unused]] const auto expectedShape = yato::dims(BATCH_SIZE, 1, 1, ALIGNMENTS_SIZE);
    TacotronParams params({}, {}, {});

    // Query
    work.add<DataLayer>("data_query", DataParams{ { "query1", "query2" }, 1u, 1u, DECODER_OUTPUT_SIZE });
    // State
    work.add<DataLayer>("data_state", DataParams{ { "state1", "state2" }, 1u, 1u, ALIGNMENTS_SIZE });
    // Memory
    work.add<DataLayer>("data_memory", DataParams{ { "memory1", "memory2" }, 1u, ALIGNMENTS_SIZE, EMBEDDED_ENCODER_OUTPUT_SIZE });
    // Memory seq length
    work.add<DataLayer>("data_memory_seq_length", DataParams{ { "memory_seq_length" }, 1u, 1u, 1u });
    // Targets
    work.add<DataLayer>("data_attention_output_state", DataParams{ { "attention_output_state1", "attention_output_state2" }, 1u, 1u, ALIGNMENTS_SIZE });
    work.add<DataLayer>("data_attention_values", DataParams{ { "attention_values1", "attention_values2" }, 1u, ALIGNMENTS_SIZE, EMBEDDED_ENCODER_OUTPUT_SIZE });

    // Parameters for attention
    BahdanauAttentionParams attentionParams1 = { { { "query1", "state1", "memory1", "memory_seq_length" }, { "attn1", "values1" } },
                                                 params.attentionDim,
                                                 params.attentionNormalizeEnergy,
                                                 0.0,
                                                 params.attentionScoreBiasInit,
                                                 params.attentionMode,
                                                 true };

    BahdanauAttentionParams attentionParams2 = { { { "query2", "state2", "memory2", "memory_seq_length" }, { "attn2" }, battnParent },
                                                 params.attentionDim,
                                                 params.attentionNormalizeEnergy,
                                                 0.0,
                                                 params.attentionScoreBiasInit,
                                                 params.attentionMode,
                                                 true };

    // Apply function
    auto& networkParameters = work.getNetworkParameters();
    BahdanauMonotonicAttentionLayer(battnParent, attentionParams1, networkParameters);
    BahdanauMonotonicAttentionLayer(battnChild, attentionParams2, networkParameters);

    work.add<TensorLayer>("grad1", TensorParams{ { Name("attn1").grad() }, WShape{ BS(), 1u, 1u, ALIGNMENTS_SIZE }, 1_dt, Workflow::Usage::ForwardAndBackward, Workflow::Mode::Read, true, true });
    work.add<TensorLayer>("grad2", TensorParams{ { Name("attn2").grad() }, WShape{ BS(), 1u, 1u, ALIGNMENTS_SIZE }, 1_dt, Workflow::Usage::ForwardAndBackward, Workflow::Mode::Read, true, true });

    work.preparePipelines();
    work.setBatchSize(BATCH_SIZE);
    work.prepareMemoryForTraining();
    MemoryManager& memory_manager = work.getMemoryManager();
    size_t loaded = loadTacotronParams((tools::getTestAssetsDir() / "Tacotron" / "pretrained" / "300000_Tacotron_model.").string(), memory_manager, "T", params, false);
    EXPECT_EQ(loaded, ATTENTION_TRAINABLE_PARAMS);

    // Load data
    bool ok = loadTFData<Tensor>(tools::getTestAssetsDir() / "Tacotron" / "attention_input.data", { &memory_manager["query1"], &memory_manager["query2"] });
    EXPECT_TRUE(ok);
    ok = loadTFData<Tensor>(tools::getTestAssetsDir() / "Tacotron" / "attention_input_state.data", { &memory_manager["state1"], &memory_manager["state2"] });
    EXPECT_TRUE(ok);
    ok = loadTFData<Tensor>(tools::getTestAssetsDir() / "Tacotron" / "attention_memory.data", { &memory_manager["memory1"], &memory_manager["memory2"] });
    EXPECT_TRUE(ok);
    ok = loadTFData(tools::getTestAssetsDir() / "Tacotron" / "input_lengths.data", memory_manager["memory_seq_length"]);
    EXPECT_TRUE(ok);

    // Load real outputs
    ok = loadTFData<Tensor>(tools::getTestAssetsDir() / "Tacotron" / "attention_output_state.data", { &memory_manager["attention_output_state1"], &memory_manager["attention_output_state2"] });
    EXPECT_TRUE(ok);
    ok = loadTFData<Tensor>(tools::getTestAssetsDir() / "Tacotron" / "attention_values.data", { &memory_manager["attention_values1"], &memory_manager["attention_values2"] });
    EXPECT_TRUE(ok);

    TensorChecker checker({ { "attn1", "attention_output_state1" }, { "values1", "attention_values1" }, { "attn2", "attention_output_state2" } }, EPS);
    work.getNetworkParameters().mCallback = checker;

    ASSERT_NO_THROW(work.forwardPassTraining());

    ASSERT_NO_THROW(work.backwardPassTraining());

    vector<pair<string, size_t>> remainingGradients;

    for (const auto& t : memory_manager.getTensorCollection().tensors)
    {
        if (!t.second->empty() && Common::endsWith(t.first.str(), TENSOR_GRADIENT_POSTFIX))
        {
            remainingGradients.emplace_back(t.first.str(), t.second->size());
        }
    }
}

TEST(TestTacotron, PostprocessingUnit)
{
    PROFILE_TEST

    using namespace tacotron;

    Workflow work;

    constexpr size_t BATCH_SIZE = 8;
    constexpr size_t POSTPROCESSING_PARAMS = 32;
    constexpr size_t POSTPROCESSING_TRAINABLE_PARAMS = 22;
    constexpr dtype EPS = 1e-4_dt;

    Name mName = "T";

    TacotronParams params({}, {}, {});
    params.postnetDropoutRate = 0.f;

    work.add<DataLayer>("data_decoder", DataParams{ { "decoder_output", "target_mel_outputs" }, 1u, params.maxMelFrames, params.numMels });
    work.add<DataLayer>("data_stop_token", DataParams{ { "stop_token_prediction" }, 1u, params.maxMelFrames, 1u });
    // postprocessing
    tacotron::AddPostNet(&work, mName / "postnet_convolutions", BasicParams{ { "decoder_output" }, { mName / "residual" } }, params);
    work.add<LinearLayer>(mName / "postnet_projection", LinearParams{ { mName / "residual" }, { mName / "projected_residual" }, params.numMels });
    work.add<ElementWiseSumLayer>(mName / "mel_outputs_sum", ElementWiseLayerParams{ { "decoder_output", mName / "projected_residual" }, { mName / "mel_outputs" } });

    work.add<TensorLayer>(
        "grad",
        TensorParams{ { (mName / "mel_outputs").grad() }, WShape{ BS(), 1u, params.maxMelFrames, params.numMels }, 1_dt, Workflow::Usage::ForwardAndBackward, Workflow::Mode::Read, true, true });

    work.preparePipelines();
    work.setBatchSize(BATCH_SIZE);
    work.prepareMemoryForTraining();
    MemoryManager& memory_manager = work.getMemoryManager();

    size_t loaded = loadTacotronParams((tools::getTestAssetsDir() / "Tacotron" / "pretrained" / "300000_Tacotron_model.").string(), memory_manager, "T", params, false);
    EXPECT_EQ(loaded, POSTPROCESSING_PARAMS);
    EXPECT_EQ(work.getTrainableParameterNames().size(), POSTPROCESSING_TRAINABLE_PARAMS);

    EXPECT_TRUE(loadTFData(tools::getTestAssetsDir() / "Tacotron" / "decoder_output.data", memory_manager["decoder_output"]));
    EXPECT_TRUE(loadTFData(tools::getTestAssetsDir() / "Tacotron" / "stop_token_prediction.data", memory_manager["stop_token_prediction"]));
    EXPECT_TRUE(loadTFData(tools::getTestAssetsDir() / "Tacotron" / "mel_outputs.data", memory_manager["target_mel_outputs"]));

    TensorChecker checker({ { mName / "mel_outputs", "target_mel_outputs" } }, EPS);
    work.getNetworkParameters().mCallback = checker;

    ASSERT_NO_THROW(work.forwardPassTraining());
    ASSERT_NO_THROW(work.backwardPassTraining());
}

TEST(TestTacotron, PostprocessingGpuUnit)
{
    PROFILE_TEST

    GPU_ONLY_TEST

    using namespace tacotron;

    WorkflowEager work(CompressionMode::NONE, CalculationMode::DETERMINISTIC, AllocationMode::POOL, ExecutionTarget::GPU);
    auto& memory_manager = work.getMemoryManager<MemoryManagerGPU>();

    constexpr size_t BATCH_SIZE = 8;
    constexpr size_t POSTPROCESSING_PARAMS = 32;
    constexpr size_t POSTPROCESSING_TRAINABLE_PARAMS = 22;
    constexpr dtype EPS = 1e-4_dt;

    Name mName = "T";

    TacotronParams params({}, {}, {});
    params.postnetDropoutRate = 0.f;

    work.add<DataLayer>("data_decoder", DataParams{ { "decoder_output", "target_mel_outputs" }, 1u, params.maxMelFrames, params.numMels });
    work.add<DataLayer>("data_stop_token", DataParams{ { "stop_token_prediction" }, 1u, params.maxMelFrames, 1u });
    // postprocessing
    tacotron::AddPostNet(&work, mName / "postnet_convolutions", BasicParams{ { "decoder_output" }, { mName / "residual" } }, params);
    work.add<LinearLayer>(mName / "postnet_projection", LinearParams{ { mName / "residual" }, { mName / "projected_residual" }, params.numMels });
    work.add<ElementWiseSumLayer>(mName / "mel_outputs_sum", ElementWiseLayerParams{ { "decoder_output", mName / "projected_residual" }, { mName / "mel_outputs" } });

    work.add<TensorLayer>(
        "grad",
        TensorParams{ { (mName / "mel_outputs").grad() }, WShape{ BS(), 1u, params.maxMelFrames, params.numMels }, 1_dt, Workflow::Usage::ForwardAndBackward, Workflow::Mode::Read, true, true });

    work.preparePipelines();
    work.setBatchSize(BATCH_SIZE);
    work.prepareMemoryForTraining();

    size_t loaded = loadTacotronParams((tools::getTestAssetsDir() / "Tacotron" / "pretrained" / "300000_Tacotron_model.").string(), memory_manager, "T", params, false);
    EXPECT_EQ(loaded, POSTPROCESSING_PARAMS);
    EXPECT_EQ(work.getTrainableParameterNames().size(), POSTPROCESSING_TRAINABLE_PARAMS);

    EXPECT_TRUE(loadTFData(tools::getTestAssetsDir() / "Tacotron" / "decoder_output.data", memory_manager["decoder_output"]));
    EXPECT_TRUE(loadTFData(tools::getTestAssetsDir() / "Tacotron" / "stop_token_prediction.data", memory_manager["stop_token_prediction"]));
    EXPECT_TRUE(loadTFData(tools::getTestAssetsDir() / "Tacotron" / "mel_outputs.data", memory_manager["target_mel_outputs"]));

    ASSERT_NO_THROW(work.forwardPassTraining());
    UT::tools::checkTensors({ { mName / "mel_outputs", "target_mel_outputs" } }, memory_manager, EPS);
    ASSERT_NO_THROW(work.backwardPassTraining());
}

TEST(TestTacotron, Alice07PostprocessingUnit)
{
    PROFILE_TEST

    using namespace tacotron;

    Workflow work;

    constexpr size_t BATCH_SIZE = 1;
    constexpr size_t POSTPROCESSING_PARAMS = 32;
    constexpr size_t POSTPROCESSING_TRAINABLE_PARAMS = 22;
    constexpr dtype EPS = 2e-6_dt;

    constexpr dtype REAL_BEFORE_LOSS = 0.449890822_dt;
    constexpr dtype REAL_AFTER_LOSS = 0.415826172_dt;
    constexpr dtype REAL_STOP_TOKEN_LOSS = 18.4135742_dt;

    Name mName = "T";

    TacotronParams params({}, {}, {});
    params.postnetDropoutRate = 0.f;
    params.maxMelFrames = 264;

    work.add<DataLayer>("data_decoder", DataParams{ { "decoder_output" }, 1u, params.maxMelFrames, params.numMels });
    work.add<DataLayer>("data_stop_token", DataParams{ { "stop_token_prediction", "stop_token_targets" }, 1u, params.maxMelFrames, 1u });
    work.add<DataLayer>("data_mel_targets", DataParams{ { "mel_targets", "target_mel_outputs" }, 1u, params.maxMelFrames, params.numMels });
    work.add<DataLayer>("data_targets_lengths", DataParams{ { "targets_lengths" }, 1u, 1u, 1u });

    // postprocessing
    AddPostNet(&work, mName / "postnet_convolutions", BasicParams{ { "decoder_output" }, { mName / "residual" } }, params);
    work.add<LinearLayer>(mName / "postnet_projection", LinearParams{ { mName / "residual" }, { mName / "projected_residual" }, params.numMels });
    work.add<ElementWiseSumLayer>(mName / "mel_outputs_sum", ElementWiseLayerParams{ { "decoder_output", mName / "projected_residual" }, { mName / "mel_outputs" } });

    AddTacotronLoss(
        &work,
        "total_loss",
        { { "decoder_output", mName / "mel_outputs", "stop_token_prediction", "mel_targets", "targets_lengths", "stop_token_targets" }, { "loss", "before_loss", "after_loss", "stop_token_loss" } },
        params);

    TENSORS_CREATE(BATCH_SIZE)

    MemoryManager& memory_manager = work.getMemoryManager();

    size_t loaded = loadTacotronParams((tools::getTestAssetsDir() / "Tacotron" / "pretrained" / "300000_Tacotron_model.").string(), memory_manager, "T", params, false, false);
    EXPECT_EQ(loaded, POSTPROCESSING_PARAMS);
    EXPECT_EQ(work.getTrainableParameterNames().size(), POSTPROCESSING_TRAINABLE_PARAMS);

    EXPECT_TRUE(loadTFData(tools::getTestAssetsDir() / "Tacotron" / "Alice07" / "decoder_output.data", memory_manager["decoder_output"]));
    EXPECT_TRUE(loadTFData(tools::getTestAssetsDir() / "Tacotron" / "Alice07" / "mel_targets_1.data", memory_manager["mel_targets"]));
    EXPECT_TRUE(loadTFData(tools::getTestAssetsDir() / "Tacotron" / "Alice07" / "mel_outputs.data", memory_manager["target_mel_outputs"]));
    EXPECT_TRUE(loadTFData(tools::getTestAssetsDir() / "Tacotron" / "Alice07" / "stop_token_targets_1.data", memory_manager["stop_token_targets"]));
    EXPECT_TRUE(loadTFData(tools::getTestAssetsDir() / "Tacotron" / "Alice07" / "stop_token_prediction.data", memory_manager["stop_token_prediction"]));
    EXPECT_TRUE(loadTFData(tools::getTestAssetsDir() / "Tacotron" / "Alice07" / "targets_lengths_1.data", memory_manager["targets_lengths"]));

    TensorChecker checker({ { mName / "mel_outputs", "target_mel_outputs" } }, EPS);
    work.getNetworkParameters().mCallback = checker;
    ASSERT_NO_THROW(work.forwardPassTraining());

    auto ab = memory_manager["before_loss"][0];
    auto aa = memory_manager["after_loss"][0];
    auto as = memory_manager["stop_token_loss"][0];

    EXPECT_NEAR(ab, REAL_BEFORE_LOSS, EPS);
    EXPECT_NEAR(aa, REAL_AFTER_LOSS, EPS);
    EXPECT_NEAR(as, REAL_STOP_TOKEN_LOSS, EPS);

    ASSERT_NO_THROW(work.backwardPassTraining());
}

TEST(TestTacotron, LossUnit)
{
    PROFILE_TEST

    using namespace tacotron;

    Workflow work;

    constexpr size_t BATCH_SIZE = 8;
    constexpr dtype EPS = 1e-4_dt;
    constexpr dtype realBeforeLoss = 0.411359072_dt;
    constexpr dtype realAfterLoss = 0.371690124_dt;
    constexpr dtype realStopTokenLoss = 0.801208_dt;
    constexpr dtype realLoss = 1.58425725_dt;
    TacotronParams params({}, {}, {});

    work.add<DataLayer>("decoder_output", DataParams{ { "decoder_output", "mel_outputs", "mel_targets" }, 1, params.maxMelFrames, params.numMels });
    work.add<DataLayer>("targets_lengths", DataParams{ { "targets_lengths" }, 1, 1, 1 });
    work.add<DataLayer>("stop_token_targets", DataParams{ { "stop_token_targets", "stop_token_prediction" }, 1, params.maxMelFrames, 1 });

    AddMaskedLoss(&work, "before_loss", { { "decoder_output", "mel_targets", "targets_lengths" }, { "before_loss" } }, params.outputsPerStep, MaskedLossType::L1, false);
    AddMaskedLoss(&work, "after_loss", { { "mel_outputs", "mel_targets", "targets_lengths" }, { "after_loss" } }, params.outputsPerStep, MaskedLossType::L1, false);
    AddMaskedCrossEntropy(
        &work, "stop_token_loss", { { "stop_token_prediction", "stop_token_targets", "targets_lengths" }, { "stop_token_loss" } }, params.outputsPerStep, params.maskedSigmoidEpsilon, false);

    work.add<ElementWiseSumLayer>("total_loss", ElementWiseLayerParams{ { "before_loss", "after_loss", "stop_token_loss" }, { "loss" } });

    TENSORS_CREATE(BATCH_SIZE);

    MemoryManager& memory_manager = work.getMemoryManager();

    EXPECT_TRUE(loadTFData(tools::getTestAssetsDir() / "Tacotron" / "stop_token_targets.data", memory_manager["stop_token_targets"]));
    EXPECT_TRUE(loadTFData(tools::getTestAssetsDir() / "Tacotron" / "stop_token_prediction.data", memory_manager["stop_token_prediction"]));
    EXPECT_TRUE(loadTFData(tools::getTestAssetsDir() / "Tacotron" / "targets_lengths.data", memory_manager["targets_lengths"]));
    EXPECT_TRUE(loadTFData(tools::getTestAssetsDir() / "Tacotron" / "mel_targets.data", memory_manager["mel_targets"]));
    EXPECT_TRUE(loadTFData(tools::getTestAssetsDir() / "Tacotron" / "mel_outputs.data", memory_manager["mel_outputs"]));
    EXPECT_TRUE(loadTFData(tools::getTestAssetsDir() / "Tacotron" / "decoder_output.data", memory_manager["decoder_output"]));

    memory_manager.createTensor("real_before", 1u, 1u, 1u, 1u, { realBeforeLoss });
    memory_manager.createTensor("real_after", 1u, 1u, 1u, 1u, { realAfterLoss });
    memory_manager.createTensor("real_stop_token", 1u, 1u, 1u, 1u, { realStopTokenLoss });
    memory_manager.createTensor("real_loss", 1u, 1u, 1u, 1u, { realLoss });

    TensorChecker checker({ { "before_loss", "real_before" }, { "after_loss", "real_after" }, { "stop_token_loss", "real_stop_token" }, { "loss", "real_loss" } }, EPS);
    work.getNetworkParameters().mCallback = checker;
    ASSERT_NO_THROW(work.forwardPassTraining());
    ASSERT_NO_THROW(work.backwardPassTraining());
}

TEST(TestTacotron, TacotronLossUnit)
{
    PROFILE_TEST
    using namespace tacotron;

    Workflow work;

    constexpr size_t BATCH_SIZE = 8;
    constexpr dtype EPS = 1e-4_dt;
    constexpr dtype realBeforeLoss = 0.411359072_dt;
    constexpr dtype realAfterLoss = 0.371690124_dt;
    constexpr dtype realStopTokenLoss = 0.801208_dt;
    constexpr dtype realLoss = 1.58425725_dt;
    TacotronParams params({}, {}, {});

    work.add<DataLayer>("decoder_output", DataParams{ { "decoder_output" }, 1, params.maxMelFrames, params.numMels });
    work.add<DataLayer>("mel_outputs", DataParams{ { "mel_outputs" }, 1, params.maxMelFrames, params.numMels });
    work.add<DataLayer>("mel_targets", DataParams{ { "mel_targets" }, 1, params.maxMelFrames, params.numMels });
    work.add<DataLayer>("targets_lengths", DataParams{ { "targets_lengths" }, 1, 1, 1 });
    work.add<DataLayer>("stop_token_targets", DataParams{ { "stop_token_targets" }, 1, params.maxMelFrames, 1 });
    work.add<DataLayer>("stop_token_prediction", DataParams{ { "stop_token_prediction" }, 1, params.maxMelFrames, 1 });

    AddTacotronLoss(
        &work,
        "total_loss",
        { { "decoder_output", "mel_outputs", "stop_token_prediction", "mel_targets", "targets_lengths", "stop_token_targets" }, { "loss", "before_loss", "after_loss", "stop_token_loss" } },
        params);

    TENSORS_CREATE(BATCH_SIZE);
    MemoryManager& memory_manager = work.getMemoryManager();

    EXPECT_TRUE(loadTFData(tools::getTestAssetsDir() / "Tacotron" / "stop_token_targets.data", memory_manager["stop_token_targets"]));
    EXPECT_TRUE(loadTFData(tools::getTestAssetsDir() / "Tacotron" / "stop_token_prediction.data", memory_manager["stop_token_prediction"]));
    EXPECT_TRUE(loadTFData(tools::getTestAssetsDir() / "Tacotron" / "targets_lengths.data", memory_manager["targets_lengths"]));
    EXPECT_TRUE(loadTFData(tools::getTestAssetsDir() / "Tacotron" / "mel_targets.data", memory_manager["mel_targets"]));
    EXPECT_TRUE(loadTFData(tools::getTestAssetsDir() / "Tacotron" / "mel_outputs.data", memory_manager["mel_outputs"]));
    EXPECT_TRUE(loadTFData(tools::getTestAssetsDir() / "Tacotron" / "decoder_output.data", memory_manager["decoder_output"]));

    ASSERT_NO_THROW(work.forwardPassTraining());

    vector<pair<string, size_t>> remainingGradients;

    for (const auto& t : memory_manager.getTensorCollection().tensors)
    {
        if (!t.second->empty() && Common::endsWith(t.first.str(), TENSOR_GRADIENT_POSTFIX))
        {
            remainingGradients.emplace_back(t.first.str(), t.second->size());
        }
    }

    auto& loss = memory_manager["loss"];
    ASSERT_EQ(loss.size(), 1u);
    EXPECT_NEAR(loss[0], realLoss, EPS);

    ASSERT_EQ(memory_manager["before_loss"].size(), 1u);
    EXPECT_NEAR(memory_manager["before_loss"][0], realBeforeLoss, EPS);

    ASSERT_EQ(memory_manager["after_loss"].size(), 1u);
    EXPECT_NEAR(memory_manager["after_loss"][0], realAfterLoss, EPS);

    ASSERT_EQ(memory_manager["stop_token_loss"].size(), 1u);
    EXPECT_NEAR(memory_manager["stop_token_loss"][0], realStopTokenLoss, EPS);

    ASSERT_NO_THROW(work.backwardPassTraining());

    remainingGradients.clear();
    for (const auto& t : memory_manager.getTensorCollection().tensors)
    {
        if (!t.second->empty() && Common::endsWith(t.first.str(), TENSOR_GRADIENT_POSTFIX))
        {
            remainingGradients.emplace_back(t.first.str(), t.second->size());
        }
    }
}

TEST(TestTacotron, TacotronWithLossUnit)
{
    PROFILE_TEST
    using namespace std;

    bool usePool = false;
    Workflow work(CompressionMode::NONE, raul::CalculationMode::DETERMINISTIC, usePool ? raul::AllocationMode::POOL : raul::AllocationMode::STANDARD);

    constexpr size_t TACOTRON_GOLDEN_PARAMS = 53;
    constexpr size_t TACOTRON_GOLDEN_TRAINABLE_PARAMS = 43;

    constexpr size_t BATCH_SIZE = 8;
    TacotronParams params({ "encoder_output", "input_lengths", "mel_targets" }, { "decoder_output", "mel_outputs", "stop_token_predictions" }, {});
    params.dropoutRate = 0.0f;
    params.postnetDropoutRate = 0.0f;
    params.zoneoutRate = 0.1f;
    params.attentionSigmoidNoise = 0.0f;

    size_t totalMemory = 0;
    vector<pair<string, size_t>> tensorSizes;

    size_t T_IN = 230; // number of steps in the input time series

    work.add<DataLayer>("encoder_output", DataParams{ { "encoder_output" }, 1u, T_IN, params.embeddingDim });
    work.add<DataLayer>("input_lengths", DataParams{ { "input_lengths" }, 1u, 1u, 1u });
    work.add<DataLayer>("mel_targets", DataParams{ { "mel_targets" }, 1u, params.maxMelFrames, params.numMels });

    // labels
    work.add<DataLayer>("targets_lengths", DataParams{ { "targets_lengths" }, 1u, 1u, 1u });
    work.add<DataLayer>("stop_token_targets", DataParams{ { "stop_token_targets" }, 1u, params.maxMelFrames, 1u });

    AddSingleSpeakerFslTacotronModel(&work, "T", params);
    tacotron::AddTacotronLoss(&work,
                              "total_loss",
                              { { "decoder_output", "mel_outputs", "stop_token_predictions", "mel_targets", "targets_lengths", "stop_token_targets" },
                                params.withoutStopTokenLoss ? Names{ "loss", "bloss", "aloss" } : Names{ "loss", "bloss", "aloss", "sloss" } },
                              params);

    TENSORS_CREATE(BATCH_SIZE);

    MemoryManager& memory_manager = work.getMemoryManager();
    createAdamTensors(work.getTrainableParameterNames(), memory_manager);

    auto [TACOTRON_TRAINABLE_PARAMS, TACOTRON_PARAMS] = tacotronParamsCount(params);

    EXPECT_EQ(TACOTRON_GOLDEN_PARAMS, TACOTRON_PARAMS);
    EXPECT_EQ(TACOTRON_GOLDEN_TRAINABLE_PARAMS, TACOTRON_TRAINABLE_PARAMS);

    size_t loaded = loadTacotronParams((tools::getTestAssetsDir() / "Tacotron" / "pretrained" / "300000_Tacotron_model.").string(), memory_manager, "T", params, true);
    EXPECT_EQ(loaded, TACOTRON_PARAMS + 2 * TACOTRON_TRAINABLE_PARAMS - 3); // -3 because speaker_embedding will be loaded separately and doesn't have Adam params

    totalMemory = 0;
    for (const auto& t : memory_manager.getTensorCollection().tensors)
    {
        totalMemory += t.second->size();
    }
    cout << "Total memory before forward: " << sizeof(dtype) * totalMemory << " bytes" << endl;

    // targets
    ASSERT_TRUE(loadTFData(tools::getTestAssetsDir() / "Tacotron" / "stop_token_targets.data", memory_manager["stop_token_targets"]));
    ASSERT_TRUE(loadTFData(tools::getTestAssetsDir() / "Tacotron" / "targets_lengths.data", memory_manager["targets_lengths"]));
    ASSERT_TRUE(loadTFData(tools::getTestAssetsDir() / "Tacotron" / "mel_targets.data", memory_manager["mel_targets"]));

    // training data
    ASSERT_TRUE(loadTFData(tools::getTestAssetsDir() / "Tacotron" / "input_lengths.data", memory_manager["input_lengths"]));
    ASSERT_TRUE(loadTFData(tools::getTestAssetsDir() / "Tacotron" / "encoder_output.data", memory_manager["encoder_output"]));
    // speaker embedding
    ASSERT_TRUE(loadTFData(tools::getTestAssetsDir() / "Tacotron" / "speaker_embedding.data", memory_manager[Name("T") / "speaker_embedding"]));

    work.forwardPassTraining();

    vector<pair<string, size_t>> remainingGradients;

    for (const auto& t : memory_manager.getTensorCollection().tensors)
    {
        if (!t.second->empty() && Common::endsWith(t.first.str(), TENSOR_GRADIENT_POSTFIX))
        {
            remainingGradients.emplace_back(t.first.str(), t.second->size());
        }
    }

    auto& lossValue = memory_manager["loss"];
    auto& blossValue = memory_manager["bloss"];
    auto& alossValue = memory_manager["aloss"];
    auto& slossValue = memory_manager["sloss"];
    std::cout << "loss = " << lossValue[0] << " (" << blossValue[0] << ", " << alossValue[0] << ", " << slossValue[0] << ")" << std::endl;
    EXPECT_EQ(lossValue.size(), 1U);

    tensorSizes.clear();
    tensorSizes.reserve(memory_manager.getTensorCollection().tensors.size());
    totalMemory = 0;
    for (const auto& t : memory_manager.getTensorCollection().tensors)
    {
        if (!t.second->empty())
        {
            tensorSizes.emplace_back(t.first.str(), t.second->size());
            totalMemory += t.second->size();
        }
    }
    cout << "Total memory after forward: " << sizeof(dtype) * totalMemory << " bytes" << endl;
    sort(tensorSizes.begin(), tensorSizes.end(), [](auto& t1, auto& t2) { return t1.second > t2.second; });

    ASSERT_NO_THROW(work.backwardPassTraining(););

    tensorSizes.clear();
    tensorSizes.reserve(memory_manager.getTensorCollection().tensors.size());
    totalMemory = 0;
    remainingGradients.clear();
    for (const auto& t : memory_manager.getTensorCollection().tensors)
    {
        if (!t.second->empty())
        {
            tensorSizes.emplace_back(t.first.str(), t.second->size());
            totalMemory += t.second->size();

            if (Common::endsWith(t.first.str(), TENSOR_GRADIENT_POSTFIX))
            {
                remainingGradients.emplace_back(t.first.str(), t.second->size());
            }
        }
    }
    cout << "Total memory after backward: " << sizeof(dtype) * totalMemory << " bytes" << endl;
    sort(tensorSizes.begin(), tensorSizes.end(), [](auto& t1, auto& t2) { return t1.second > t2.second; });
}

TEST(TestTacotron, DISABLED_TacotronSingleBatchTraining)
{
    PROFILE_TEST
    using namespace std;
    using namespace optimizers;

    random::setGlobalSeed(42);

    bool usePool = false;
    Workflow work(CompressionMode::NONE, raul::CalculationMode::DETERMINISTIC, usePool ? raul::AllocationMode::POOL : raul::AllocationMode::STANDARD);

    constexpr size_t TACOTRON_GOLDEN_PARAMS = 53;
    constexpr size_t TACOTRON_GOLDEN_TRAINABLE_PARAMS = 43;

    constexpr size_t BATCH_SIZE = 10;
    constexpr size_t nEpoch = 1000;

    auto suffix = BATCH_SIZE == 8 ? "" : "_" + to_string(BATCH_SIZE);

    size_t nEpochToSave = 100;

    constexpr dtype ADAM_BETA1 = 0.9_dt;
    constexpr dtype ADAM_BETA2 = 0.999_dt;
    constexpr dtype ADAM_EPSILON = 1e-6_dt;

    TacotronParams params({ "encoder_output", "input_lengths", "mel_targets" }, { "decoder_output", "mel_outputs", "stop_token_predictions" }, {});
    params.dropoutRate = 0.5f;
    params.postnetDropoutRate = 0.5f;
    params.zoneoutRate = 0.1f;
    params.attentionSigmoidNoise = 0.2f;

    TacotronTrainingParams trainParams;
    trainParams.initialLearningRate = 1e-2_dt;
    trainParams.clipNorm = 1.0_dt;
    trainParams.decayLearningRate = true;
    trainParams.decaySteps = 1000;
    trainParams.decayStartStep = 200;
    trainParams.decayRate = 0.5;
    trainParams.finalLearningRate = 1e-4_dt;
    trainParams.warmupLearningRate = true;
    trainParams.warmupSteps = 400;

    size_t T_IN = 230; // number of steps in the input time series

    work.add<DataLayer>("encoder_output", DataParams{ { "encoder_output" }, 1u, T_IN, params.embeddingDim });
    work.add<DataLayer>("input_lengths", DataParams{ { "input_lengths" }, 1u, 1u, 1u });
    work.add<DataLayer>("mel_targets", DataParams{ { "mel_targets" }, 1u, params.maxMelFrames, params.numMels });

    // labels
    work.add<DataLayer>("targets_lengths", DataParams{ { "targets_lengths" }, 1u, 1u, 1u });
    work.add<DataLayer>("stop_token_targets", DataParams{ { "stop_token_targets" }, 1u, params.maxMelFrames, 1u });

    AddSingleSpeakerFslTacotronModel(&work, "T", params);
    tacotron::AddTacotronLoss(&work,
                              "total_loss",
                              { { "decoder_output", "mel_outputs", "stop_token_predictions", "mel_targets", "targets_lengths", "stop_token_targets" },
                                params.withoutStopTokenLoss ? Names{ "loss", "before_loss", "after_loss" } : Names{ "loss", "before_loss", "after_loss", "stop_token_loss" } },
                              params);

    TENSORS_CREATE(BATCH_SIZE);
    MemoryManager& memory_manager = work.getMemoryManager();
    createAdamTensors(work.getTrainableParameterNames(), memory_manager);

    auto [TACOTRON_TRAINABLE_PARAMS, TACOTRON_PARAMS] = tacotronParamsCount(params);

    EXPECT_EQ(TACOTRON_GOLDEN_PARAMS, TACOTRON_PARAMS);
    EXPECT_EQ(TACOTRON_GOLDEN_TRAINABLE_PARAMS, TACOTRON_TRAINABLE_PARAMS);

    size_t loaded = loadTacotronParams((tools::getTestAssetsDir() / "Tacotron" / "pretrained" / "300000_Tacotron_model.").string(), memory_manager, "T", params, true);
    EXPECT_EQ(loaded, TACOTRON_PARAMS + 2 * TACOTRON_TRAINABLE_PARAMS - 3); // -3 because speaker_embedding will be loaded separately and doesn't have Adam params

    // targets
    ASSERT_TRUE(loadTFData(tools::getTestAssetsDir() / "Tacotron" / ("stop_token_targets" + suffix + ".data"), memory_manager["stop_token_targets"]));
    ASSERT_TRUE(loadTFData(tools::getTestAssetsDir() / "Tacotron" / ("targets_lengths" + suffix + ".data"), memory_manager["targets_lengths"]));
    ASSERT_TRUE(loadTFData(tools::getTestAssetsDir() / "Tacotron" / ("mel_targets" + suffix + ".data"), memory_manager["mel_targets"]));

    // training data
    ASSERT_TRUE(loadTFData(tools::getTestAssetsDir() / "Tacotron" / ("input_lengths" + suffix + ".data"), memory_manager["input_lengths"]));
    ASSERT_TRUE(loadTFData(tools::getTestAssetsDir() / "Tacotron" / ("encoder_output" + suffix + ".data"), memory_manager["encoder_output"]));

    // speaker embedding
    ASSERT_TRUE(loadTFData(tools::getTestAssetsDir() / "Tacotron" / "speaker_embedding.data", memory_manager[Name("T") / "speaker_embedding"]));

    auto optimizer = createOptimizer(trainParams, ADAM_BETA1, ADAM_BETA2, ADAM_EPSILON);

    dtype trainingTime = 0;

    postprocessing::GradientClipping clip(trainParams.clipNorm);

    for (size_t epoch = 0; epoch < nEpoch; ++epoch)
    {
        auto epochStart = chrono::steady_clock::now();
        cout << "Epoch " << epoch << endl;
        auto timeStart = chrono::steady_clock::now();
        work.forwardPassTraining();
        auto totalTime = TODTYPE(chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - timeStart).count());
        cout << "  loss = " << memory_manager["loss"][0] << " (before = " << memory_manager["before_loss"][0] << ", after = " << memory_manager["after_loss"][0]
             << ", stop_token = " << memory_manager["stop_token_loss"][0] << ")" << endl;

        cout << "  forward: " << totalTime << " ms" << endl;

        if (trainParams.decayLearningRate)
        {
            optimizer->step();
        }

        timeStart = chrono::steady_clock::now();
        work.backwardPassTraining();
        totalTime = TODTYPE(chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - timeStart).count());
        cout << "  backward: " << totalTime << " ms" << endl;

        timeStart = chrono::steady_clock::now();
        auto trainableParams = work.getTrainableParameters();
        clip.processGradients(trainableParams, work.getNetworkParameters());
        totalTime = TODTYPE(chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - timeStart).count());
        cout << "  gradient clipping: (norm=" << clip.getGlobalNorm(work.getNetworkParameters()) << "): " << totalTime << " ms" << endl;

        timeStart = chrono::steady_clock::now();
        for (auto& p : trainableParams)
        {
            optimizer->operator()(memory_manager, p.Param, p.Gradient);
        }
        totalTime = TODTYPE(chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - timeStart).count());
        cout << "  optimizer (lr=" << optimizer->getLearningRate() << "): " << totalTime << " ms" << endl;
        auto epochTime = TODTYPE(chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - epochStart).count());
        trainingTime += epochTime;
        cout << "  epoch: " << epochTime << " ms" << endl;
        cout << "Average performance: " << trainingTime / TODTYPE(epoch + 1) << " ms/epoch" << endl << endl;
        if ((epoch + 1) % nEpochToSave == 0)
        {
            size_t saved = saveTacotronParams(std::filesystem::path("Tacotron") / "finetuned" / to_string(epoch + 1), "300000_Tacotron_model.", memory_manager, "T", params);
            EXPECT_EQ(saved, TACOTRON_PARAMS + 2 * (TACOTRON_TRAINABLE_PARAMS - 1)); // we do not save adam tensors for speaker_embedding
        }
    }
}

#if defined(ANDROID)
TEST(TestTacotron, DISABLED_TacotronSingleBatchTrainingFP16)
{
    PROFILE_TEST
    using namespace std;
    using namespace optimizers;

    random::setGlobalSeed(42);

    bool usePool = false;
    Workflow work(CompressionMode::NONE, raul::CalculationMode::DETERMINISTIC, usePool ? raul::AllocationMode::POOL : raul::AllocationMode::STANDARD, raul::ExecutionTarget::CPUFP16);

    constexpr size_t TACOTRON_GOLDEN_PARAMS = 53;
    constexpr size_t TACOTRON_GOLDEN_TRAINABLE_PARAMS = 43;

    constexpr size_t BATCH_SIZE = 10;
    constexpr size_t nEpoch = 1000;

    auto suffix = BATCH_SIZE == 8 ? "" : "_" + to_string(BATCH_SIZE);

    size_t nEpochToSave = 100;

    constexpr dtype ADAM_BETA1 = 0.9_dt;
    constexpr dtype ADAM_BETA2 = 0.999_dt;
    constexpr dtype ADAM_EPSILON = 1e-6_dt;

    TacotronParams params({ "encoder_output", "input_lengths", "mel_targets" }, { "decoder_output", "mel_outputs", "stop_token_predictions" }, {});
    params.dropoutRate = 0.5f;
    params.postnetDropoutRate = 0.5f;
    params.zoneoutRate = 0.1f;
    params.attentionSigmoidNoise = 0.2f;

    TacotronTrainingParams trainParams;
    trainParams.initialLearningRate = 1e-2_dt;
    trainParams.clipNorm = 1.0_dt;
    trainParams.decayLearningRate = true;
    trainParams.decaySteps = 1000;
    trainParams.decayStartStep = 200;
    trainParams.decayRate = 0.5;
    trainParams.finalLearningRate = 1e-4_dt;
    trainParams.warmupLearningRate = true;
    trainParams.warmupSteps = 400;

    size_t T_IN = 230; // number of steps in the input time series

    work.add<DataLayer>("encoder_output", DataParams{ { "encoder_output" }, 1u, T_IN, params.embeddingDim });
    work.add<DataLayer>("input_lengths", DataParams{ { "input_lengths" }, 1u, 1u, 1u });
    work.add<DataLayer>("mel_targets", DataParams{ { "mel_targets" }, 1u, params.maxMelFrames, params.numMels });

    // labels
    work.add<DataLayer>("targets_lengths", DataParams{ { "targets_lengths" }, 1u, 1u, 1u });
    work.add<DataLayer>("stop_token_targets", DataParams{ { "stop_token_targets" }, 1u, params.maxMelFrames, 1u });

    AddSingleSpeakerFslTacotronModel(&work, "T", params);
    tacotron::AddTacotronLoss(&work,
                              "total_loss",
                              { { "decoder_output", "mel_outputs", "stop_token_predictions", "mel_targets", "targets_lengths", "stop_token_targets" },
                                params.withoutStopTokenLoss ? Names{ "loss", "before_loss", "after_loss" } : Names{ "loss", "before_loss", "after_loss", "stop_token_loss" } },
                              params);

    TENSORS_CREATE(BATCH_SIZE);
    auto& memory_manager = work.getMemoryManager<raul::MemoryManagerFP16>();
    createAdamTensors(work.getTrainableParameterNames(), memory_manager);

    auto [TACOTRON_TRAINABLE_PARAMS, TACOTRON_PARAMS] = tacotronParamsCount(params);

    EXPECT_EQ(TACOTRON_GOLDEN_PARAMS, TACOTRON_PARAMS);
    EXPECT_EQ(TACOTRON_GOLDEN_TRAINABLE_PARAMS, TACOTRON_TRAINABLE_PARAMS);

    size_t loaded = loadTacotronParams((tools::getTestAssetsDir() / "Tacotron" / "pretrained" / "300000_Tacotron_model.").string(), memory_manager, "T", params, true);
    EXPECT_EQ(loaded, TACOTRON_PARAMS + 2 * TACOTRON_TRAINABLE_PARAMS - 3); // -3 because speaker_embedding will be loaded separately and doesn't have Adam params

    // targets
    ASSERT_TRUE(loadTFData(tools::getTestAssetsDir() / "Tacotron" / ("stop_token_targets" + suffix + ".data"), memory_manager["stop_token_targets"]));
    ASSERT_TRUE(loadTFData(tools::getTestAssetsDir() / "Tacotron" / ("targets_lengths" + suffix + ".data"), memory_manager["targets_lengths"]));
    ASSERT_TRUE(loadTFData(tools::getTestAssetsDir() / "Tacotron" / ("mel_targets" + suffix + ".data"), memory_manager["mel_targets"]));

    // training data
    ASSERT_TRUE(loadTFData(tools::getTestAssetsDir() / "Tacotron" / ("input_lengths" + suffix + ".data"), memory_manager["input_lengths"]));
    ASSERT_TRUE(loadTFData(tools::getTestAssetsDir() / "Tacotron" / ("encoder_output" + suffix + ".data"), memory_manager["encoder_output"]));

    // speaker embedding
    ASSERT_TRUE(loadTFData(tools::getTestAssetsDir() / "Tacotron" / "speaker_embedding.data", memory_manager[Name("T") / "speaker_embedding"]));

    auto optimizer = createOptimizer(trainParams, ADAM_BETA1, ADAM_BETA2, ADAM_EPSILON);

    dtype trainingTime = 0;

    postprocessing::GradientClipping clip(trainParams.clipNorm);

    for (size_t epoch = 0; epoch < nEpoch; ++epoch)
    {
        auto epochStart = chrono::steady_clock::now();
        cout << "Epoch " << epoch << endl;
        auto timeStart = chrono::steady_clock::now();
        work.forwardPassTraining();
        auto totalTime = TODTYPE(chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - timeStart).count());
        cout << "  loss = " << toFloat32(memory_manager["loss"][0]) << " (before = " << toFloat32(memory_manager["before_loss"][0]) << ", after = " << toFloat32(memory_manager["after_loss"][0])
             << ", stop_token = " << toFloat32(memory_manager["stop_token_loss"][0]) << ")" << endl;

        cout << "  forward: " << totalTime << " ms" << endl;

        if (trainParams.decayLearningRate)
        {
            optimizer->step();
        }

        timeStart = chrono::steady_clock::now();
        work.backwardPassTraining();
        totalTime = TODTYPE(chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - timeStart).count());
        cout << "  backward: " << totalTime << " ms" << endl;

        timeStart = chrono::steady_clock::now();
        auto trainableParams = work.getTrainableParameters<MemoryManagerFP16>();
        clip.processGradients(trainableParams, work.getNetworkParameters());
        totalTime = TODTYPE(chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - timeStart).count());
        cout << "  gradient clipping: (norm=" << clip.getGlobalNorm(work.getNetworkParameters()) << "): " << totalTime << " ms" << endl;

        timeStart = chrono::steady_clock::now();
        for (auto& p : trainableParams)
        {
            optimizer->operator()(memory_manager, p.Param, p.Gradient);
        }
        totalTime = TODTYPE(chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - timeStart).count());
        cout << "  optimizer (lr=" << optimizer->getLearningRate() << "): " << totalTime << " ms" << endl;
        auto epochTime = TODTYPE(chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - epochStart).count());
        trainingTime += epochTime;
        cout << "  epoch: " << epochTime << " ms" << endl;
        cout << "Average performance: " << trainingTime / TODTYPE(epoch + 1) << " ms/epoch" << endl << endl;
        if ((epoch + 1) % nEpochToSave == 0)
        {
            // size_t saved = saveTacotronParams(std::filesystem::path("Tacotron") / "finetuned" / to_string(epoch + 1), "300000_Tacotron_model.", memory_manager, "T", params);
            // EXPECT_EQ(saved, TACOTRON_PARAMS + 2 * (TACOTRON_TRAINABLE_PARAMS - 1)); // we do not save adam tensors for speaker_embedding
        }
    }
}
#endif

TEST(TestTacotron, AliceUnit)
{
    PROFILE_TEST
    using namespace raul;
    using namespace std;
    using namespace optimizers;

    constexpr size_t TACOTRON_GOLDEN_PARAMS = 53;
    constexpr size_t TACOTRON_GOLDEN_TRAINABLE_PARAMS = 43;

    bool usePool = false;
    Workflow work(CompressionMode::NONE, raul::CalculationMode::DETERMINISTIC, usePool ? raul::AllocationMode::POOL : raul::AllocationMode::STANDARD);

    constexpr size_t BATCH_SIZE = 15;

    TacotronParams params({ "encoder_output", "input_lengths", "mel_targets" }, { "decoder_output", "mel_outputs", "stop_token_predictions" }, {});
    params.dropoutRate = 0.f;
    params.postnetDropoutRate = 0.f;
    params.zoneoutRate = 0.f;
    params.attentionSigmoidNoise = 0.f;

    params.maxMelFrames = 546; // @todo: read from file

    size_t decoder_iterations = params.maxMelFrames / params.outputsPerStep;

    work.add<DataLayer>("golden_decoder_output", DataParams{ { "golden_decoder_output" }, 1u, params.maxMelFrames, params.numMels });
    work.add<DataLayer>("golden_stop_token", DataParams{ { "golden_stop_token" }, 1u, params.maxMelFrames, 1u });

    TacotronTrainingParams trainParams;
    trainParams.initialLearningRate = 1e-5_dt;

    size_t T_IN = 94; // number of steps in the input time series

    work.add<DataLayer>("encoder_output", DataParams{ { "encoder_output" }, 1u, T_IN, params.embeddingDim });
    work.add<DataLayer>("input_lengths", DataParams{ { "input_lengths" }, 1u, 1u, 1u });
    work.add<DataLayer>("mel_targets", DataParams{ { "mel_targets" }, 1u, params.maxMelFrames, params.numMels });

    // labels
    work.add<DataLayer>("targets_lengths", DataParams{ { "targets_lengths" }, 1u, 1u, 1u });
    work.add<DataLayer>("stop_token_targets", DataParams{ { "stop_token_targets" }, 1u, params.maxMelFrames, 1u });

    AddSingleSpeakerFslTacotronModel(&work, "T", params);
    tacotron::AddTacotronLoss(&work,
                              "total_loss",
                              { { "decoder_output", "mel_outputs", "stop_token_predictions", "mel_targets", "targets_lengths", "stop_token_targets" },
                                params.withoutStopTokenLoss ? Names{ "loss", "bloss", "aloss" } : Names{ "loss", "bloss", "aloss", "sloss" } },
                              params);

    vector<dtype> idealLosses;
    ifstream f(tools::getTestAssetsDir() / "Tacotron" / "Alice" / "loss.data");
    std::copy(istream_iterator<dtype>(f), istream_iterator<dtype>(), back_inserter(idealLosses));
    EXPECT_TRUE(idealLosses.size() == 3);

    auto idealGradNorms = loadNamedValues(tools::getTestAssetsDir() / "Tacotron" / "Alice" / "grad_norm.data", "T", params);
    EXPECT_EQ(idealGradNorms.size(), TACOTRON_GOLDEN_TRAINABLE_PARAMS);

    TENSORS_CREATE(BATCH_SIZE);
    MemoryManager& memory_manager = work.getMemoryManager();

    EXPECT_TRUE(loadTFData(tools::getTestAssetsDir() / "Tacotron" / "Alice" / "decoder_output.data", memory_manager["golden_decoder_output"]));
    EXPECT_TRUE(loadTFData(tools::getTestAssetsDir() / "Tacotron" / "Alice" / "stop_token_prediction.data", memory_manager["golden_stop_token"]));

    auto [TACOTRON_TRAINABLE_PARAMS, TACOTRON_PARAMS] = tacotronParamsCount(params);
    EXPECT_EQ(TACOTRON_GOLDEN_PARAMS, TACOTRON_PARAMS);
    EXPECT_EQ(TACOTRON_GOLDEN_TRAINABLE_PARAMS, TACOTRON_TRAINABLE_PARAMS);
    size_t loaded = loadTacotronParams((tools::getTestAssetsDir() / "Tacotron" / "pretrained" / "300000_Tacotron_model.").string(), memory_manager, "T", params, true, false);
    EXPECT_EQ(loaded, TACOTRON_PARAMS - 1); // -1 because speaker_embedding will be loaded separately and doesn't have Adam params
    postprocessing::GradientClipping clip;

    // targets
    EXPECT_TRUE(loadTFData(tools::getTestAssetsDir() / "Tacotron" / "Alice" / "stop_token_targets_15.data", memory_manager["stop_token_targets"]));
    EXPECT_TRUE(loadTFData(tools::getTestAssetsDir() / "Tacotron" / "Alice" / "targets_lengths_15.data", memory_manager["targets_lengths"]));
    EXPECT_TRUE(loadTFData(tools::getTestAssetsDir() / "Tacotron" / "Alice" / "mel_targets_15.data", memory_manager["mel_targets"]));

    // training data
    EXPECT_TRUE(loadTFData(tools::getTestAssetsDir() / "Tacotron" / "Alice" / "input_lengths_15.data", memory_manager["input_lengths"]));
    EXPECT_TRUE(loadTFData(tools::getTestAssetsDir() / "Tacotron" / "Alice" / "encoder_output_15.data", memory_manager["encoder_output"]));

    // speaker embedding
    EXPECT_TRUE(loadTFData(tools::getTestAssetsDir() / "Tacotron" / "Alice" / "speaker_embedding.data", memory_manager[Name("T") / "speaker_embedding"]));

    auto timeStart = chrono::steady_clock::now();
    work.forwardPassTraining();
    auto& lossValue = memory_manager["loss"];
    auto totalTime = TODTYPE(chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - timeStart).count());
    cout << "Test time " << totalTime << " ms for batch size " << BATCH_SIZE << endl;
    cout << "Initial loss = " << lossValue[0] << " (before = " << memory_manager["bloss"][0] << ", after = " << memory_manager["aloss"][0] << ", stop_token = " << memory_manager["sloss"][0] << ")"
         << endl;
    cout << "Ideal loss = " << (idealLosses[0] + idealLosses[1] + idealLosses[2]) << " (before = " << idealLosses[0] << ", after = " << idealLosses[1] << ", stop_token = " << idealLosses[2] << ")"
         << endl;

    auto& goldenf = memory_manager["golden_decoder_output"];
    auto& goldens = memory_manager["golden_stop_token"];
    auto& calculatedf = memory_manager["decoder_output"];
    auto& calculateds = memory_manager["stop_token_predictions"];
    for (size_t i = 0; i < decoder_iterations; ++i)
    {
        size_t sz = params.numMels * params.outputsPerStep;
        Tensor fr(Tensor::dt_range(&*(calculatedf.begin() + i * sz), &*(calculatedf.begin() + (i + 1) * sz)));
        Tensor st(Tensor::dt_range(&*(calculateds.begin() + i * params.outputsPerStep), &*(calculateds.begin() + (i + 1) * params.outputsPerStep)));

        Tensor gf(Tensor::dt_range(&*(goldenf.begin() + i * sz), &*(goldenf.begin() + (i + 1) * sz)));
        Tensor gs(Tensor::dt_range(&*(goldens.begin() + i * params.outputsPerStep), &*(goldens.begin() + (i + 1) * params.outputsPerStep)));

        cout << "Iter " << i << ". frame: " << TensorDiff(fr, gf) << "(" << TensorNorm(fr) << ", " << TensorNorm(gf) << ")"
             << ", stop: " << TensorDiff(st, gs) << "(" << TensorNorm(st) << ", " << TensorNorm(gs) << ")" << endl;
    }

    work.backwardPassTraining();
    auto trainableParams = work.getTrainableParameters();
    clip.processGradients(trainableParams, work.getNetworkParameters());
    cout << "Global norm: " << clip.getGlobalNorm(work.getNetworkParameters()) << endl;
    dtype totalGradNormDiff = 0;
    for (size_t j = 0; j < trainableParams.size(); ++j)
    {
        auto& [param, grad] = trainableParams[j];
        cout << param.getName() << ": " << TensorNorm(grad) << "( " << idealGradNorms[param.getName()] << ")" << endl;

        cout << "  diff: " << fabs(TensorNorm(grad) - idealGradNorms[param.getName()]) << std::endl;
        totalGradNormDiff += fabs(TensorNorm(grad) - idealGradNorms[param.getName()]);
    }
    cout << "Total Grad Norm Diff: " << totalGradNormDiff << std::endl;

    EXPECT_TRUE(totalGradNormDiff < 4_dt); // such a big value because of stop_toke_loss
}

TEST(TestTacotron, AliceWithoutStopTokenUnit)
{
    PROFILE_TEST
    using namespace raul;
    using namespace std;
    using namespace optimizers;

    constexpr size_t TACOTRON_GOLDEN_PARAMS = 53;
    constexpr size_t TACOTRON_GOLDEN_TRAINABLE_PARAMS = 43;

    bool usePool = false;
    Workflow work(CompressionMode::NONE, raul::CalculationMode::DETERMINISTIC, usePool ? raul::AllocationMode::POOL : raul::AllocationMode::STANDARD);

    constexpr size_t BATCH_SIZE = 15;

    TacotronParams params({ "encoder_output", "input_lengths", "mel_targets" }, { "decoder_output", "mel_outputs", "stop_token_predictions" }, {});
    params.dropoutRate = 0.f;
    params.postnetDropoutRate = 0.f;
    params.zoneoutRate = 0.f;
    params.attentionSigmoidNoise = 0.f;
    params.withoutStopTokenLoss = false;
    params.lossMultipliers = { 1.f, 1.f, 0.f };

    params.maxMelFrames = 546; // @todo: read from file

    size_t decoder_iterations = params.maxMelFrames / params.outputsPerStep;

    work.add<DataLayer>("golden_decoder_output", DataParams{ { "golden_decoder_output" }, 1u, params.maxMelFrames, params.numMels });
    work.add<DataLayer>("golden_stop_token", DataParams{ { "golden_stop_token" }, 1u, params.maxMelFrames, 1u });

    size_t T_IN = 94; // number of steps in the input time series

    work.add<DataLayer>("encoder_output", DataParams{ { "encoder_output" }, 1u, T_IN, params.embeddingDim });
    work.add<DataLayer>("input_lengths", DataParams{ { "input_lengths" }, 1u, 1u, 1u });
    work.add<DataLayer>("mel_targets", DataParams{ { "mel_targets" }, 1u, params.maxMelFrames, params.numMels });

    // labels
    work.add<DataLayer>("targets_lengths", DataParams{ { "targets_lengths" }, 1u, 1u, 1u });
    work.add<DataLayer>("stop_token_targets", DataParams{ { "stop_token_targets" }, 1u, params.maxMelFrames, 1u });

    AddSingleSpeakerFslTacotronModel(&work, "T", params);
    tacotron::AddTacotronLoss(&work,
                              "total_loss",
                              { { "decoder_output", "mel_outputs", "stop_token_predictions", "mel_targets", "targets_lengths", "stop_token_targets" },
                                params.withoutStopTokenLoss ? Names{ "loss", "before_loss", "after_loss" } : Names{ "loss", "before_loss", "after_loss", "stop_token_loss" } },
                              params);

    vector<dtype> idealLosses;
    ifstream f(tools::getTestAssetsDir() / "Tacotron" / "Alice" / "loss.data");
    std::copy(istream_iterator<dtype>(f), istream_iterator<dtype>(), back_inserter(idealLosses));
    EXPECT_TRUE(idealLosses.size() == 3);

    auto idealGradNorms = loadNamedValues(tools::getTestAssetsDir() / "Tacotron" / "Alice" / "without_stop_token" / "grad_norm.data", "T", params);
    EXPECT_EQ(idealGradNorms.size(), TACOTRON_GOLDEN_TRAINABLE_PARAMS);

    TENSORS_CREATE(BATCH_SIZE);
    MemoryManager& memory_manager = work.getMemoryManager();

    ASSERT_TRUE(loadTFData(tools::getTestAssetsDir() / "Tacotron" / "Alice" / "decoder_output.data", memory_manager["golden_decoder_output"]));
    ASSERT_TRUE(loadTFData(tools::getTestAssetsDir() / "Tacotron" / "Alice" / "stop_token_prediction.data", memory_manager["golden_stop_token"]));

    auto [TACOTRON_TRAINABLE_PARAMS, TACOTRON_PARAMS] = tacotronParamsCount(params);

    EXPECT_EQ(TACOTRON_GOLDEN_PARAMS, TACOTRON_PARAMS);
    EXPECT_EQ(TACOTRON_GOLDEN_TRAINABLE_PARAMS, TACOTRON_TRAINABLE_PARAMS);
    size_t loaded = loadTacotronParams((tools::getTestAssetsDir() / "Tacotron" / "pretrained" / "300000_Tacotron_model.").string(), memory_manager, "T", params, true, false);
    EXPECT_EQ(loaded, TACOTRON_PARAMS - 1); // -1 because speaker_embedding will be loaded separately and doesn't have Adam params
    postprocessing::GradientClipping clip;

    // targets
    ASSERT_TRUE(loadTFData(tools::getTestAssetsDir() / "Tacotron" / "Alice" / "stop_token_targets_15.data", memory_manager["stop_token_targets"]));
    ASSERT_TRUE(loadTFData(tools::getTestAssetsDir() / "Tacotron" / "Alice" / "targets_lengths_15.data", memory_manager["targets_lengths"]));
    ASSERT_TRUE(loadTFData(tools::getTestAssetsDir() / "Tacotron" / "Alice" / "mel_targets_15.data", memory_manager["mel_targets"]));

    // training data
    ASSERT_TRUE(loadTFData(tools::getTestAssetsDir() / "Tacotron" / "Alice" / "input_lengths_15.data", memory_manager["input_lengths"]));
    ASSERT_TRUE(loadTFData(tools::getTestAssetsDir() / "Tacotron" / "Alice" / "encoder_output_15.data", memory_manager["encoder_output"]));

    // speaker embedding
    ASSERT_TRUE(loadTFData(tools::getTestAssetsDir() / "Tacotron" / "Alice" / "speaker_embedding.data", memory_manager[Name("T") / "speaker_embedding"]));

    auto timeStart = chrono::steady_clock::now();
    work.forwardPassTraining();
    auto& lossValue = memory_manager["loss"];
    auto totalTime = TODTYPE(chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - timeStart).count());
    cout << "Test time " << totalTime << " ms for batch size " << BATCH_SIZE << endl;
    cout << "Initial loss = " << lossValue[0] << " (before = " << memory_manager["before_loss"][0] << ", after = " << memory_manager["after_loss"][0] << ")" << endl;
    cout << "Ideal loss = " << (idealLosses[0] + idealLosses[1]) << " (before = " << idealLosses[0] << ", after = " << idealLosses[1] << ", stop_token = " << idealLosses[2] << ")" << endl;

    auto& goldenf = memory_manager["golden_decoder_output"];
    auto& goldens = memory_manager["golden_stop_token"];
    auto& calculatedf = memory_manager["decoder_output"];
    auto& calculateds = memory_manager["stop_token_predictions"];
    for (size_t i = 0; i < decoder_iterations; ++i)
    {
        size_t sz = params.numMels * params.outputsPerStep;
        Tensor fr(Tensor::dt_range(&*(calculatedf.begin() + i * sz), &*(calculatedf.begin() + (i + 1) * sz)));
        Tensor st(Tensor::dt_range(&*(calculateds.begin() + i * params.outputsPerStep), &*(calculateds.begin() + (i + 1) * params.outputsPerStep)));

        Tensor gf(Tensor::dt_range(&*(goldenf.begin() + i * sz), &*(goldenf.begin() + (i + 1) * sz)));
        Tensor gs(Tensor::dt_range(&*(goldens.begin() + i * params.outputsPerStep), &*(goldens.begin() + (i + 1) * params.outputsPerStep)));

        cout << "Iter " << i << ". frame: " << TensorDiff(fr, gf) << "(" << TensorNorm(fr) << ", " << TensorNorm(gf) << ")"
             << ", stop: " << TensorDiff(st, gs) << "(" << TensorNorm(st) << ", " << TensorNorm(gs) << ")" << endl;
    }

    work.backwardPassTraining();
    auto tParams = work.getTrainableParameters();

    cout << "Global norm: " << clip.calcGlobalNorm(tParams, work.getNetworkParameters()) << endl;
    dtype totalGradNormDiff = 0;
    for (size_t j = 0; j < tParams.size(); ++j)
    {
        auto& [param, grad] = tParams[j];
        cout << param.getName() << ": " << TensorNorm(grad) << "( " << idealGradNorms[param.getName()] << ")" << endl;

        cout << "  diff: " << fabs(TensorNorm(grad) - idealGradNorms[param.getName()]) << std::endl;
        totalGradNormDiff += fabs(TensorNorm(grad) - idealGradNorms[param.getName()]);
    }
    cout << "Total Grad Norm Diff: " << sqrt(totalGradNormDiff) << std::endl;

    EXPECT_TRUE(totalGradNormDiff < 0.05_dt);
}

TEST(TestTacotron, Alice07DeterministicUnit)
{
    PROFILE_TEST

    using namespace optimizers;

    constexpr size_t TACOTRON_PARAMS = 53;
    constexpr size_t TACOTRON_TRAINABLE_PARAMS = 43;

    Workflow work;

    constexpr size_t BATCH_SIZE = 1;

    TacotronParams params({ "encoder_output", "input_lengths", "mel_targets" }, { "decoder_output", "mel_outputs", "stop_token_prediction" }, {});
    params.dropoutRate = 0.f;
    params.postnetDropoutRate = 0.f;
    params.zoneoutRate = 0.f;
    params.attentionSigmoidNoise = 0.f;

    params.maxMelFrames = 264; // @todo: read from file

    size_t decoder_iterations = params.maxMelFrames / params.outputsPerStep;

    TacotronTrainingParams trainParams;
    trainParams.initialLearningRate = 1e-5_dt;

    size_t T_IN = 35; // number of steps in the input time series

    work.add<DataLayer>("encoder_output", DataParams{ { "encoder_output" }, 1u, T_IN, params.embeddingDim });
    work.add<DataLayer>("input_lengths", DataParams{ { "input_lengths" }, 1u, 1u, 1u });
    work.add<DataLayer>("mel_targets", DataParams{ { "mel_targets" }, 1u, params.maxMelFrames, params.numMels });

    work.add<DataLayer>("golden_decoder_output", DataParams{ { "golden_decoder_output" }, 1u, params.maxMelFrames, params.numMels });
    work.add<DataLayer>("golden_stop_token", DataParams{ { "golden_stop_token" }, 1u, params.maxMelFrames, 1u });

    work.add<DataLayer>("targets_lengths", DataParams{ { "targets_lengths" }, 1u, 1u, 1u });
    work.add<DataLayer>("stop_token_targets", DataParams{ { "stop_token_targets" }, 1u, params.maxMelFrames, 1u });

    AddSingleSpeakerFslTacotronModel(&work, "T", params);
    tacotron::AddTacotronLoss(
        &work,
        "total_loss",
        { { "decoder_output", "mel_outputs", "stop_token_prediction", "mel_targets", "targets_lengths", "stop_token_targets" }, { "loss", "before_loss", "after_loss", "stop_token_loss" } },
        params);

    vector<dtype> idealLosses;
    ifstream f(tools::getTestAssetsDir() / "Tacotron" / "Alice07" / "loss.data");
    std::copy(istream_iterator<dtype>(f), istream_iterator<dtype>(), back_inserter(idealLosses));
    ASSERT_TRUE(idealLosses.size() >= 3);
    idealLosses.resize(3);

    auto idealGradNorms = loadNamedValues(tools::getTestAssetsDir() / "Tacotron" / "Alice07" / "grad_norm.data", "T", params);
    EXPECT_EQ(idealGradNorms.size(), TACOTRON_TRAINABLE_PARAMS);

    TENSORS_CREATE(BATCH_SIZE);
    MemoryManager& memory_manager = work.getMemoryManager();
    EXPECT_TRUE(loadTFData(tools::getTestAssetsDir() / "Tacotron" / "Alice07" / "decoder_output.data", memory_manager["golden_decoder_output"]));
    EXPECT_TRUE(loadTFData(tools::getTestAssetsDir() / "Tacotron" / "Alice07" / "stop_token_prediction.data", memory_manager["golden_stop_token"]));

    size_t loaded = loadTacotronParams((tools::getTestAssetsDir() / "Tacotron" / "pretrained" / "300000_Tacotron_model.").string(), memory_manager, "T", params, true, false);
    EXPECT_EQ(loaded, TACOTRON_PARAMS - 1); // -1 because speaker_embedding will be loaded separately and doesn't have Adam params
    postprocessing::GradientClipping clip;

    // targets
    EXPECT_TRUE(loadTFData(tools::getTestAssetsDir() / "Tacotron" / "Alice07" / "stop_token_targets_1.data", memory_manager["stop_token_targets"]));
    EXPECT_TRUE(loadTFData(tools::getTestAssetsDir() / "Tacotron" / "Alice07" / "targets_lengths_1.data", memory_manager["targets_lengths"]));
    EXPECT_TRUE(loadTFData(tools::getTestAssetsDir() / "Tacotron" / "Alice07" / "mel_targets_1.data", memory_manager["mel_targets"]));

    // training data
    EXPECT_TRUE(loadTFData(tools::getTestAssetsDir() / "Tacotron" / "Alice07" / "input_lengths_1.data", memory_manager["input_lengths"]));
    EXPECT_TRUE(loadTFData(tools::getTestAssetsDir() / "Tacotron" / "Alice07" / "encoder_output_1.data", memory_manager["encoder_output"]));

    // speaker embedding
    EXPECT_TRUE(loadTFData(tools::getTestAssetsDir() / "Tacotron" / "Alice07" / "speaker_embedding.data", memory_manager[Name("T") / "speaker_embedding"]));

    auto callback = [&](BasicLayer* layer, const MemoryManager& memory_manager, NetworkParameters::CallbackPlace place) {
        if (place != NetworkParameters::CallbackPlace::Before_Forward)
        {
            return;
        }
        if (!Common::startsWith(layer->getName(), Name("total_loss") / "before" / "loss"))
        {
            return;
        }

        auto& goldenf = memory_manager["golden_decoder_output"];
        auto& goldens = memory_manager["golden_stop_token"];
        auto& calculatedf = memory_manager["decoder_output"];
        auto& calculateds = memory_manager["stop_token_prediction"];
        for (size_t i = 0; i < decoder_iterations; ++i)
        {
            size_t sz = params.numMels * params.outputsPerStep;
            Tensor fr(Tensor::dt_range(&*(calculatedf.begin() + i * sz), &*(calculatedf.begin() + (i + 1) * sz)));
            Tensor st(Tensor::dt_range(&*(calculateds.begin() + i * params.outputsPerStep), &*(calculateds.begin() + (i + 1) * params.outputsPerStep)));

            Tensor gf(Tensor::dt_range(&*(goldenf.begin() + i * sz), &*(goldenf.begin() + (i + 1) * sz)));
            Tensor gs(Tensor::dt_range(&*(goldens.begin() + i * params.outputsPerStep), &*(goldens.begin() + (i + 1) * params.outputsPerStep)));

            cout << "Iter " << i << ". frame: " << TensorDiff(fr, gf) << "(" << TensorNorm(fr) << ", " << TensorNorm(gf) << ")"
                 << ", stop: " << TensorDiff(st, gs) << "(" << TensorNorm(st) << ", " << TensorNorm(gs) << ")" << endl;
        }
    };

    work.getNetworkParameters().mCallback = callback;
    auto timeStart = chrono::steady_clock::now();
    work.forwardPassTraining();

    auto& lossValue = memory_manager["loss"];
    auto totalTime = TODTYPE(chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - timeStart).count());
    cout << "Test time " << totalTime << " ms for batch size " << BATCH_SIZE << endl;
    cout << "Initial loss = " << lossValue[0] << " (before = " << memory_manager["before_loss"][0] << ", after = " << memory_manager["after_loss"][0]
         << ", stop_token = " << memory_manager["stop_token_loss"][0] << ")" << endl;
    cout << "Ideal loss = " << (idealLosses[0] + idealLosses[1] + idealLosses[2]) << " (before = " << idealLosses[0] << ", after = " << idealLosses[1] << ", stop_token = " << idealLosses[2] << ")"
         << endl;

    work.backwardPassTraining();

    auto tParams = work.getTrainableParameters();
    clip.processGradients(tParams, work.getNetworkParameters());
    cout << "Gradients." << endl;
    cout << "Global norm: " << clip.getGlobalNorm(work.getNetworkParameters()) << endl;
    dtype totalGradNormDiff = 0;
    for (size_t j = 0; j < tParams.size(); ++j)
    {
        auto& [param, grad] = tParams[j];
        cout << param.getName() << ": " << TensorNorm(grad) << " (" << idealGradNorms[param.getName()] << ")" << endl;

        cout << "  diff: " << fabs(TensorNorm(grad) - idealGradNorms[param.getName()]) << std::endl;
        totalGradNormDiff += fabs(TensorNorm(grad) - idealGradNorms[param.getName()]);
    }
    cout << "Total Grad Norm Diff: " << totalGradNormDiff << std::endl;

    EXPECT_TRUE(totalGradNormDiff < 0.5_dt);
}

TEST(TestTacotron, Alice07DeterministicAdamUnit)
{
    PROFILE_TEST
    using namespace std;
    using namespace optimizers;

    constexpr size_t TACOTRON_PARAMS = 53;
    constexpr size_t TACOTRON_TRAINABLE_PARAMS = 43;

    Workflow work;

    constexpr size_t BATCH_SIZE = 1;

    TacotronParams params({ "encoder_output", "input_lengths", "mel_targets" }, { "decoder_output", "mel_outputs", "stop_token_predictions" }, {});
    params.dropoutRate = 0.f;
    params.postnetDropoutRate = 0.f;
    params.zoneoutRate = 0.f;
    params.attentionSigmoidNoise = 0.f;

    params.withoutStopTokenLoss = false;
    params.maxMelFrames = 264; // @todo: read from file

    constexpr dtype ADAM_BETA1 = 0_dt;
    constexpr dtype ADAM_BETA2 = 0_dt;
    constexpr dtype ADAM_EPSILON = 1e-6_dt;

    TacotronTrainingParams trainParams;
    trainParams.initialLearningRate = 1e-2_dt;
    trainParams.clipNorm = 1.0_dt;
    trainParams.clipGradients = false;
    trainParams.decayLearningRate = true;
    trainParams.decaySteps = 1000;
    trainParams.decayStartStep = 200;
    trainParams.decayRate = 0.5;
    trainParams.finalLearningRate = 1e-4_dt;
    trainParams.warmupLearningRate = true;
    trainParams.warmupSteps = 400;

    size_t decoder_iterations = params.maxMelFrames / params.outputsPerStep;

    work.add<DataLayer>("golden_decoder_output", DataParams{ { "golden_decoder_output" }, 1u, params.maxMelFrames, params.numMels });
    work.add<DataLayer>("golden_stop_token", DataParams{ { "golden_stop_token" }, 1u, params.maxMelFrames, 1u });

    size_t T_IN = 35; // number of steps in the input time series

    work.add<DataLayer>("encoder_output", DataParams{ { "encoder_output" }, 1u, T_IN, params.embeddingDim });
    work.add<DataLayer>("input_lengths", DataParams{ { "input_lengths" }, 1u, 1u, 1u });
    work.add<DataLayer>("mel_targets", DataParams{ { "mel_targets" }, 1u, params.maxMelFrames, params.numMels });

    // labels
    work.add<DataLayer>("targets_lengths", DataParams{ { "targets_lengths" }, 1u, 1u, 1u });
    work.add<DataLayer>("stop_token_targets", DataParams{ { "stop_token_targets" }, 1u, params.maxMelFrames, 1u });

    AddSingleSpeakerFslTacotronModel(&work, "T", params);
    tacotron::AddTacotronLoss(
        &work,
        "total_loss",
        { { "decoder_output", "mel_outputs", "stop_token_predictions", "mel_targets", "targets_lengths", "stop_token_targets" }, { "loss", "before_loss", "after_loss", "stop_token_loss" } },
        params);

    vector<dtype> idealLosses;
    ifstream f(tools::getTestAssetsDir() / "Tacotron" / "Alice07" / "adam" / "loss.data");
    std::copy(istream_iterator<dtype>(f), istream_iterator<dtype>(), back_inserter(idealLosses));
    ASSERT_TRUE(idealLosses.size() >= 3);
    idealLosses.resize(3);

    auto idealGradNorms = loadNamedValues(tools::getTestAssetsDir() / "Tacotron" / "Alice07" / "grad_norm.data", "T", params);
    auto varNormsBefore = loadNamedValues(tools::getTestAssetsDir() / "Tacotron" / "Alice07" / "adam" / "var_norm0.data", "T", params);
    auto varNormsAfter = loadNamedValues(tools::getTestAssetsDir() / "Tacotron" / "Alice07" / "adam" / "var_norm.data", "T", params);

    auto varAdamMNorms = loadNamedValues(tools::getTestAssetsDir() / "Tacotron" / "Alice07" / "adam" / "adam_m_norm.data", "T", params);
    EXPECT_EQ(idealGradNorms.size(), TACOTRON_TRAINABLE_PARAMS);

    TENSORS_CREATE(BATCH_SIZE);
    MemoryManager& memory_manager = work.getMemoryManager();

    EXPECT_TRUE(loadTFData(tools::getTestAssetsDir() / "Tacotron" / "Alice07" / "decoder_output.data", memory_manager["golden_decoder_output"]));
    EXPECT_TRUE(loadTFData(tools::getTestAssetsDir() / "Tacotron" / "Alice07" / "stop_token_prediction.data", memory_manager["golden_stop_token"]));

    size_t loaded = loadTacotronParams((tools::getTestAssetsDir() / "Tacotron" / "pretrained" / "300000_Tacotron_model.").string(), memory_manager, "T", params, true, false);
    EXPECT_EQ(loaded, TACOTRON_PARAMS - 1); // -1 because speaker_embedding will be loaded separately and doesn't have Adam params
    postprocessing::GradientClipping clip;

    // targets
    EXPECT_TRUE(loadTFData(tools::getTestAssetsDir() / "Tacotron" / "Alice07" / "stop_token_targets_1.data", memory_manager["stop_token_targets"]));
    EXPECT_TRUE(loadTFData(tools::getTestAssetsDir() / "Tacotron" / "Alice07" / "targets_lengths_1.data", memory_manager["targets_lengths"]));
    EXPECT_TRUE(loadTFData(tools::getTestAssetsDir() / "Tacotron" / "Alice07" / "mel_targets_1.data", memory_manager["mel_targets"]));

    // training data
    EXPECT_TRUE(loadTFData(tools::getTestAssetsDir() / "Tacotron" / "Alice07" / "input_lengths_1.data", memory_manager["input_lengths"]));
    EXPECT_TRUE(loadTFData(tools::getTestAssetsDir() / "Tacotron" / "Alice07" / "encoder_output_1.data", memory_manager["encoder_output"]));

    // speaker embedding
    EXPECT_TRUE(loadTFData(tools::getTestAssetsDir() / "Tacotron" / "Alice07" / "speaker_embedding.data", memory_manager[Name("T") / "speaker_embedding"]));

    auto timeStart = chrono::steady_clock::now();
    work.forwardPassTraining();
    auto& lossValue = memory_manager["loss"];
    auto totalTime = TODTYPE(chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - timeStart).count());
    cout << "Test time " << totalTime << " ms for batch size " << BATCH_SIZE << endl;
    cout << "Initial loss = " << lossValue[0] << " (before = " << memory_manager["before_loss"][0] << ", after = " << memory_manager["after_loss"][0];
    if (!params.withoutStopTokenLoss)
    {
        cout << ", stop_token = " << memory_manager["stop_token_loss"][0];
    }
    cout << ")" << endl;
    cout << "Ideal loss = " << (idealLosses[0] + idealLosses[1]) << " (before = " << idealLosses[0] << ", after = " << idealLosses[1] << ", stop_token = " << idealLosses[2] << ")" << endl;

    auto& goldenf = memory_manager["golden_decoder_output"];
    auto& goldens = memory_manager["golden_stop_token"];
    auto& calculatedf = memory_manager["decoder_output"];
    auto& calculateds = memory_manager["stop_token_predictions"];
    for (size_t i = 0; i < decoder_iterations; ++i)
    {
        size_t sz = params.numMels * params.outputsPerStep;
        Tensor fr(Tensor::dt_range(&*(calculatedf.begin() + i * sz), &*(calculatedf.begin() + (i + 1) * sz)));
        Tensor st(Tensor::dt_range(&*(calculateds.begin() + i * params.outputsPerStep), &*(calculateds.begin() + (i + 1) * params.outputsPerStep)));

        Tensor gf(Tensor::dt_range(&*(goldenf.begin() + i * sz), &*(goldenf.begin() + (i + 1) * sz)));
        Tensor gs(Tensor::dt_range(&*(goldens.begin() + i * params.outputsPerStep), &*(goldens.begin() + (i + 1) * params.outputsPerStep)));

        cout << "Iter " << i << ". frame: " << TensorDiff(fr, gf) << "(" << TensorNorm(fr) << ", " << TensorNorm(gf) << ")"
             << ", stop: " << TensorDiff(st, gs) << "(" << TensorNorm(st) << ", " << TensorNorm(gs) << ")" << endl;
    }

    work.backwardPassTraining();
    auto tParams = work.getTrainableParameters();

    auto optimizer = createOptimizer(trainParams, ADAM_BETA1, ADAM_BETA2, ADAM_EPSILON);

    optimizer->step();

    cout << "Global norm: " << clip.calcGlobalNorm(tParams, work.getNetworkParameters()) << endl;
    dtype totalGradNormDiff = 0;
    for (size_t j = 0; j < tParams.size(); ++j)
    {
        auto& [param, grad] = tParams[j];
        cout << param.getName() << endl;
        cout << "  grad: " << TensorNorm(grad) << "( " << idealGradNorms[param.getName()] << ")" << endl;
        cout << "    diff: " << fabs(TensorNorm(grad) - idealGradNorms[param.getName()]) << std::endl;
        cout << "  before: " << TensorNorm(param) << "( " << varNormsBefore[param.getName()] << ")" << endl;
        cout << "    diff: " << fabs(TensorNorm(param) - varNormsBefore[param.getName()]) << std::endl;

        totalGradNormDiff += fabs(TensorNorm(grad) - idealGradNorms[param.getName()]);

        optimizer->operator()(memory_manager, param, grad);

        cout << "  after: " << TensorNorm(param) << "( " << varNormsAfter[param.getName()] << ")" << endl;
        cout << "    diff: " << fabs(TensorNorm(param) - varNormsAfter[param.getName()]) << std::endl;
    }

    cout << "Total Grad Norm Diff: " << sqrt(totalGradNormDiff) << std::endl;

    EXPECT_TRUE(totalGradNormDiff < 0.5_dt);
}

TEST(TestTacotron, AdamUnit)
{
    PROFILE_TEST

    using namespace optimizers;

    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    size_t h = 64;
    size_t w = 20;

    constexpr dtype IDEAL_GRAD_NORM = 0.0116356285_dt;

    constexpr auto GRAD_NORM_EPS = 1e-6_dt;
    constexpr auto WEIGHTS_EPS = 1e-3_dt;

    Tensor weights0("w", 1u, 1u, h, w);
    Tensor grad(1u, 1u, h, w);
    Tensor weights1(1u, 1u, h, w);

    EXPECT_TRUE(loadTFData<Tensor>(tools::getTestAssetsDir() / "Tacotron" / "Alice07" / "adam" / "residual_projection_weights.data", { &weights0, &weights1 }));
    EXPECT_TRUE(loadTFData<Tensor>(tools::getTestAssetsDir() / "Tacotron" / "Alice07" / "adam" / "gradient.data", grad));

    postprocessing::GradientClipping clip;
    ParamAndGrad params = { weights0, grad };
    auto gnorm = clip.calcGlobalNorm({ params }, networkParameters);

    auto* m = networkParameters.mMemoryManager.createTensor(Name("Adam") / "w" / "m", weights0.getShape());
    auto* v = networkParameters.mMemoryManager.createTensor(Name("Adam") / "w" / "v", weights0.getShape());

    DataLoader::loadData(tools::getTestAssetsDir() / "Tacotron" / "Alice07" / "adam" / "kernel.Adam.0.data", *m);
    DataLoader::loadData(tools::getTestAssetsDir() / "Tacotron" / "Alice07" / "adam" / "kernel.Adam_1.0.data", *v);

    cout << "Grad Norm Diff: " << fabs(gnorm - IDEAL_GRAD_NORM) << " (" << gnorm << ", " << IDEAL_GRAD_NORM << ")" << std::endl;
    EXPECT_NEAR(gnorm, IDEAL_GRAD_NORM, GRAD_NORM_EPS);
    constexpr dtype ADAM_BETA1 = 0.9_dt;
    constexpr dtype ADAM_BETA2 = 0.999_dt;
    constexpr dtype ADAM_EPSILON = 1e-6_dt;

    TacotronTrainingParams trainParams;
    trainParams.initialLearningRate = 1e-2_dt;
    trainParams.clipNorm = 1.0_dt;
    trainParams.clipGradients = false;
    trainParams.decayLearningRate = true;
    trainParams.decaySteps = 1000;
    trainParams.decayStartStep = 200;
    trainParams.decayRate = 0.5;
    trainParams.finalLearningRate = 1e-4_dt;
    trainParams.warmupLearningRate = true;
    trainParams.warmupSteps = 400;

    auto optimizer = createOptimizer(trainParams, ADAM_BETA1, ADAM_BETA2, ADAM_EPSILON);

    optimizer->step();
    optimizer->operator()(networkParameters.mMemoryManager, weights0, grad);

    cout << "Weights diff: " << TensorDiff(weights0, weights1) << endl;
    EXPECT_TRUE(TensorDiff(weights0, weights1) < WEIGHTS_EPS);
}

static Dataset GetAliceDataset(const std::filesystem::path& path)
{
    Dataset dataset;
    dataset.describePart("encoder_output", 15, 1, 94, 512);
    dataset.setDataSourceFor("encoder_output", std::make_unique<LoadDataInCustomNumpyFormat>(path / "encoder_output_15.data"));
    dataset.describePart("input_lengths", 15, 1, 1, 1);
    dataset.setDataSourceFor("input_lengths", std::make_unique<LoadDataInCustomNumpyFormat>(path / "input_lengths_15.data"));
    dataset.describePart("stop_token_targets", 15, 1, 546, 1);
    dataset.setDataSourceFor("stop_token_targets", std::make_unique<LoadDataInCustomNumpyFormat>(path / "stop_token_targets_15.data"));
    dataset.describePart("targets_lengths", 15, 1, 1, 1);
    dataset.setDataSourceFor("targets_lengths", std::make_unique<LoadDataInCustomNumpyFormat>(path / "targets_lengths_15.data"));
    dataset.describePart("mel_targets", 15, 1, 546, 20);
    dataset.setDataSourceFor("mel_targets", std::make_unique<LoadDataInCustomNumpyFormat>(path / "mel_targets_15.data"));

    return dataset;
}

TEST(TestTacotron, DISABLED_AliceTrainingWithMicroBatches)
{
    PROFILE_TEST
    using namespace raul;
    using namespace std;

    raul::Workflow work;

    constexpr size_t TACOTRON_GOLDEN_PARAMS = 53;
    constexpr size_t TACOTRON_GOLDEN_TRAINABLE_PARAMS = 43;

    bool useMicroBatches = false;
    constexpr size_t BATCH_SIZE = 15;
    constexpr size_t MICRO_BATCH_SIZE = 5;
    constexpr size_t nEpoch = 1000;

    auto suffix = BATCH_SIZE == 8 ? "" : "_" + to_string(BATCH_SIZE);

    [[maybe_unused]] constexpr size_t nEpochToSave = 100;

    constexpr dtype ADAM_BETA1 = 0.9_dt;
    constexpr dtype ADAM_BETA2 = 0.999_dt;
    constexpr dtype ADAM_EPSILON = 1e-6_dt;

    TacotronParams params({ "encoder_output", "input_lengths", "mel_targets" }, { "decoder_output", "mel_outputs", "stop_token_predictions" }, {});
    params.dropoutRate = 0.5f;
    params.postnetDropoutRate = 0.5f;
    params.zoneoutRate = 0.1f;
    params.attentionSigmoidNoise = 0.2f;

    params.maxMelFrames = 546; // @todo: read from file

    TacotronTrainingParams trainParams;
    trainParams.initialLearningRate = 1e-2_dt;
    trainParams.clipNorm = 1.0_dt;
    trainParams.decayLearningRate = true;
    trainParams.decaySteps = 1000;
    trainParams.decayStartStep = 200;
    trainParams.decayRate = 0.5;
    trainParams.finalLearningRate = 1e-4_dt;
    trainParams.warmupLearningRate = true;
    trainParams.warmupSteps = 400;

    auto optimizer = createOptimizer(trainParams, ADAM_BETA1, ADAM_BETA2, ADAM_EPSILON);

    size_t T_IN = 94; // number of steps in the input time series

    work.add<DataLayer>("encoder_output", DataParams{ { "encoder_output" }, 1u, T_IN, params.embeddingDim });
    work.add<DataLayer>("input_lengths", DataParams{ { "input_lengths" }, 1u, 1u, 1u });
    work.add<DataLayer>("mel_targets", DataParams{ { "mel_targets" }, 1u, params.maxMelFrames, params.numMels });

    // labels
    work.add<DataLayer>("targets_lengths", DataParams{ { "targets_lengths" }, 1u, 1u, 1u });
    work.add<DataLayer>("stop_token_targets", DataParams{ { "stop_token_targets" }, 1u, params.maxMelFrames, 1u });

    AddSingleSpeakerFslTacotronModel(&work, "T", params);
    tacotron::AddTacotronLoss(&work, "total_loss", { { "decoder_output", "mel_outputs", "stop_token_predictions", "mel_targets", "targets_lengths", "stop_token_targets" }, { "loss" } }, params);

    TENSORS_CREATE(BATCH_SIZE);
    MemoryManager& memory_manager = work.getMemoryManager();
    createAdamTensors(work.getTrainableParameterNames(), memory_manager);

    auto [TACOTRON_TRAINABLE_PARAMS, TACOTRON_PARAMS] = tacotronParamsCount(params);

    EXPECT_EQ(TACOTRON_GOLDEN_PARAMS, TACOTRON_PARAMS);
    EXPECT_EQ(TACOTRON_GOLDEN_TRAINABLE_PARAMS, TACOTRON_TRAINABLE_PARAMS);

    size_t loaded = loadTacotronParams((tools::getTestAssetsDir() / "Tacotron" / "pretrained" / "300000_Tacotron_model.").string(), memory_manager, "T", params, true);
    EXPECT_EQ(loaded, TACOTRON_PARAMS + 2 * TACOTRON_TRAINABLE_PARAMS - 3); // -3 because speaker_embedding will be loaded separately and doesn't have Adam params

    // speaker embedding
    ASSERT_TRUE(loadTFData(tools::getTestAssetsDir() / "Tacotron" / "Alice" / ("speaker_embedding.data"), memory_manager[Name("T") / "speaker_embedding"]));

    Dataset trainData = GetAliceDataset(tools::getTestAssetsDir() / "Tacotron" / "Alice");

    Train train(work,
                trainData,
                { { { "stop_token_targets", "stop_token_targets" },
                    { "targets_lengths", "targets_lengths" },
                    { "mel_targets", "mel_targets" },
                    { "input_lengths", "input_lengths" },
                    { "encoder_output", "encoder_output" } },
                  "loss",
                  { std::make_shared<postprocessing::GradientClipping>(trainParams.clipNorm) } });
    useMicroBatches ? train.useMicroBatches(BATCH_SIZE, MICRO_BATCH_SIZE) : train.useBatches(BATCH_SIZE);

    dtype trainingTime = 0;
    for (size_t epoch = 0, step = 1; epoch < nEpoch; ++epoch, ++step)
    {
        auto epochStart = chrono::steady_clock::now();
        if (trainParams.decayLearningRate)
        {
            optimizer->step();
        }
        auto lossValue = train.oneIteration(*optimizer);
        auto epochTime = TODTYPE(chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - epochStart).count());
        trainingTime += epochTime;

        cout << "Epoch " << epoch << endl;
        cout << "  loss = " << lossValue << endl;
        cout << "  epoch: " << epochTime << " ms" << endl;
        cout << "Average performance: " << trainingTime / static_cast<float>(epoch + 1) << " ms/epoch" << endl;

        if ((epoch + 1) % nEpochToSave == 0)
        {
            [[maybe_unused]] size_t saved =
                saveTacotronParams(std::filesystem::path("Tacotron") / (useMicroBatches ? "finetuned_mbs" : "finetuned_bs") / to_string(epoch), "300000_Tacotron_model.", memory_manager, "T", params);
        }
    }
}

// No randomness, no gradient clipping, SGD
TEST(TestTacotron, DISABLED_Alice07DeterministicSGDTraining)
{
    PROFILE_TEST
    using namespace raul;
    using namespace std;
    using namespace optimizers;

    bool usePool = false;
    Workflow work(CompressionMode::NONE, raul::CalculationMode::DETERMINISTIC, usePool ? raul::AllocationMode::POOL : raul::AllocationMode::STANDARD);

    constexpr size_t TACOTRON_GOLDEN_PARAMS = 53;
    constexpr size_t TACOTRON_GOLDEN_TRAINABLE_PARAMS = 43;

    constexpr auto LOSS_EPS_19 = 2e-5_dt;
    constexpr auto LOSS_EPS_20 = 2e-2_dt;

    constexpr size_t BATCH_SIZE = 1;
    constexpr size_t nEpoch = 100;

    TacotronParams params({ "encoder_output", "input_lengths", "mel_targets" }, { "decoder_output", "mel_outputs", "stop_token_predictions" }, {});
    params.dropoutRate = 0.f;
    params.postnetDropoutRate = 0.f;
    params.zoneoutRate = 0.f;
    params.attentionSigmoidNoise = 0.f;

    params.maxMelFrames = 264; // @todo: read from file

    TacotronTrainingParams trainParams;
    trainParams.initialLearningRate = 1e-3_dt;

    size_t T_IN = 35; // number of steps in the input time series

    work.add<DataLayer>("encoder_output", DataParams{ { "encoder_output" }, 1u, T_IN, params.embeddingDim });
    work.add<DataLayer>("input_lengths", DataParams{ { "input_lengths" }, 1u, 1u, 1u });
    work.add<DataLayer>("mel_targets", DataParams{ { "mel_targets" }, 1u, params.maxMelFrames, params.numMels });

    // labels
    work.add<DataLayer>("targets_lengths", DataParams{ { "targets_lengths" }, 1u, 1u, 1u });
    work.add<DataLayer>("stop_token_targets", DataParams{ { "stop_token_targets" }, 1u, params.maxMelFrames, 1u });

    AddSingleSpeakerFslTacotronModel(&work, "T", params);
    tacotron::AddTacotronLoss(&work,
                              "total_loss",
                              { { "decoder_output", "mel_outputs", "stop_token_predictions", "mel_targets", "targets_lengths", "stop_token_targets" },
                                params.withoutStopTokenLoss ? Names{ "loss", "before_loss", "after_loss" } : Names{ "loss", "before_loss", "after_loss", "stop_token_loss" } },
                              params);

    vector<dtype> idealLosses;
    vector<dtype> idealGradNorm;
    {
        ifstream f(tools::getTestAssetsDir() / "Tacotron" / "Alice07" / "loss.data");
        std::copy(istream_iterator<dtype>(f), istream_iterator<dtype>(), back_inserter(idealLosses));
    }
    {
        ifstream f(tools::getTestAssetsDir() / "Tacotron" / "Alice07" / "global_grad_norm.data");
        std::copy(istream_iterator<dtype>(f), istream_iterator<dtype>(), back_inserter(idealGradNorm));
    }

    EXPECT_EQ(idealLosses.size(), nEpoch * 3);
    EXPECT_EQ(idealGradNorm.size(), nEpoch);

    TENSORS_CREATE(BATCH_SIZE);

    MemoryManager& memory_manager = work.getMemoryManager();

    auto [TACOTRON_TRAINABLE_PARAMS, TACOTRON_PARAMS] = tacotronParamsCount(params);

    EXPECT_EQ(TACOTRON_GOLDEN_PARAMS, TACOTRON_PARAMS);
    EXPECT_EQ(TACOTRON_GOLDEN_TRAINABLE_PARAMS, TACOTRON_TRAINABLE_PARAMS);

    size_t loaded = loadTacotronParams((tools::getTestAssetsDir() / "Tacotron" / "pretrained" / "300000_Tacotron_model.").string(), memory_manager, "T", params, true, false);
    EXPECT_EQ(loaded, TACOTRON_PARAMS - 1); // -1 because speaker_embedding will be loaded separately and doesn't have Adam params

    // targets
    ASSERT_TRUE(loadTFData(tools::getTestAssetsDir() / "Tacotron" / "Alice07" / "stop_token_targets_1.data", memory_manager["stop_token_targets"]));
    ASSERT_TRUE(loadTFData(tools::getTestAssetsDir() / "Tacotron" / "Alice07" / "targets_lengths_1.data", memory_manager["targets_lengths"]));
    ASSERT_TRUE(loadTFData(tools::getTestAssetsDir() / "Tacotron" / "Alice07" / "mel_targets_1.data", memory_manager["mel_targets"]));
    // training data
    ASSERT_TRUE(loadTFData(tools::getTestAssetsDir() / "Tacotron" / "Alice07" / "input_lengths_1.data", memory_manager["input_lengths"]));
    ASSERT_TRUE(loadTFData(tools::getTestAssetsDir() / "Tacotron" / "Alice07" / "encoder_output_1.data", memory_manager["encoder_output"]));
    // speaker embedding
    ASSERT_TRUE(loadTFData(tools::getTestAssetsDir() / "Tacotron" / "Alice07" / "speaker_embedding.data", memory_manager[Name("T") / "speaker_embedding"]));

    auto optimizer = std::make_unique<optimizers::SGD>(trainParams.initialLearningRate);

    dtype trainingTime = 0;

    postprocessing::GradientClipping clip;

    for (size_t epoch = 0; epoch < nEpoch; ++epoch)
    {
        auto epochStart = chrono::steady_clock::now();
        cout << "Epoch " << epoch << endl;
        auto timeStart = chrono::steady_clock::now();
        work.forwardPassTraining();

        auto& lossValue = memory_manager["loss"];
        auto ab = memory_manager["before_loss"][0];
        auto aa = memory_manager["after_loss"][0];
        auto as = memory_manager["stop_token_loss"][0];

        cout << "  Actual loss: " << lossValue[0] << " (before: " << ab << ", after: " << aa << ", stop_token: " << as << ")" << endl;

        auto tb = idealLosses[3 * epoch];
        auto ta = idealLosses[3 * epoch + 1];
        auto ts = idealLosses[3 * epoch + 2];

        cout << "  Target loss: " << (tb + ta + ts) << " (before: " << tb << ", after: " << ta << ", stop_token: " << ts << ")" << endl;

        cout << "    Loss Difference: " << fabs((tb + ta + ts) - lossValue[0]) << " (before: " << fabs(tb - ab) << ", after: " << fabs(ta - aa) << ", stop_token: " << fabs(ts - as) << ")" << endl;

        timeStart = chrono::steady_clock::now();
        work.backwardPassTraining();
        auto trainableParams = work.getTrainableParameters();
        auto gradNorm = clip.calcGlobalNorm(trainableParams, work.getNetworkParameters());
        cout << "  Global grad norm: " << gradNorm << ". Ideal grad norm: " << idealGradNorm[epoch] << endl;
        cout << "    Grad Difference: " << fabs(gradNorm - idealGradNorm[epoch]) << endl;

        if (epoch < 19)
        {
            EXPECT_NEAR(ab, tb, LOSS_EPS_19);
            EXPECT_NEAR(aa, ta, LOSS_EPS_19);
            EXPECT_NEAR(as, ts, LOSS_EPS_19);
        }
        else
        {
            EXPECT_NEAR(ab, tb, LOSS_EPS_20);
            EXPECT_NEAR(aa, ta, LOSS_EPS_20);
        }

        timeStart = chrono::steady_clock::now();

        for (auto& p : trainableParams)
        {
            optimizer->operator()(memory_manager, p.Param, p.Gradient);
        }

        auto epochTime = TODTYPE(chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - epochStart).count());
        trainingTime += epochTime;
        cout << "Average performance: " << trainingTime / TODTYPE(epoch + 1) << " ms/epoch" << endl << endl;
    }
}

TEST(TestTacotron, Alice07DeterministicAdamTraining)
{
    PROFILE_TEST
    using namespace std;
    using namespace optimizers;

    constexpr size_t TACOTRON_GOLDEN_PARAMS = 53;
    constexpr size_t TACOTRON_GOLDEN_TRAINABLE_PARAMS = 43;

    bool usePool = false;
    Workflow work(CompressionMode::NONE, raul::CalculationMode::DETERMINISTIC, usePool ? raul::AllocationMode::POOL : raul::AllocationMode::STANDARD);

    constexpr size_t BATCH_SIZE = 1;
    constexpr size_t nEpoch = 20;

    TacotronParams params({ "encoder_output", "input_lengths", "mel_targets" }, { "decoder_output", "mel_outputs", "stop_token_predictions" }, {});
    params.dropoutRate = 0.f;
    params.postnetDropoutRate = 0.f;
    params.zoneoutRate = 0.f;
    params.attentionSigmoidNoise = 0.f;

    params.maxMelFrames = 264; // @todo: read from file

    params.withoutStopTokenLoss = true;

    constexpr dtype ADAM_BETA1 = 0.9_dt;
    constexpr dtype ADAM_BETA2 = 0.999_dt;
    constexpr dtype ADAM_EPSILON = 1e-6_dt;

    TacotronTrainingParams trainParams;
    trainParams.initialLearningRate = 1e-2_dt;
    trainParams.clipNorm = 1.0_dt;
    trainParams.clipGradients = true;
    trainParams.decayLearningRate = true;
    trainParams.decaySteps = 1000;
    trainParams.decayStartStep = 200;
    trainParams.decayRate = 0.5;
    trainParams.finalLearningRate = 1e-4_dt;
    trainParams.warmupLearningRate = true;
    trainParams.warmupSteps = 400;

    size_t T_IN = 35; // number of steps in the input time series

    work.add<DataLayer>("encoder_output", DataParams{ { "encoder_output" }, 1u, T_IN, params.embeddingDim });
    work.add<DataLayer>("input_lengths", DataParams{ { "input_lengths" }, 1u, 1u, 1u });
    work.add<DataLayer>("mel_targets", DataParams{ { "mel_targets" }, 1u, params.maxMelFrames, params.numMels });

    // labels
    work.add<DataLayer>("targets_lengths", DataParams{ { "targets_lengths" }, 1u, 1u, 1u });
    work.add<DataLayer>("stop_token_targets", DataParams{ { "stop_token_targets" }, 1u, params.maxMelFrames, 1u });

    AddSingleSpeakerFslTacotronModel(&work, "T", params);
    tacotron::AddTacotronLoss(&work,
                              "total_loss",
                              { { "decoder_output", "mel_outputs", "stop_token_predictions", "mel_targets", "targets_lengths", "stop_token_targets" },
                                params.withoutStopTokenLoss ? Names{ "loss", "before_loss", "after_loss" } : Names{ "loss", "before_loss", "after_loss", "stop_token_loss" } },
                              params);

    vector<dtype> idealLosses;
    vector<dtype> idealGradNorm;
    vector<dtype> idealLearningRate;
    {
        ifstream f(tools::getTestAssetsDir() / "Tacotron" / "Alice07" / "adam" / "loss.data");
        std::copy(istream_iterator<dtype>(f), istream_iterator<dtype>(), back_inserter(idealLosses));
        EXPECT_TRUE(idealLosses.size() == 3 * nEpoch);
    }
    {
        ifstream f(tools::getTestAssetsDir() / "Tacotron" / "Alice07" / "adam" / "global_grad_norm.data");
        std::copy(istream_iterator<dtype>(f), istream_iterator<dtype>(), back_inserter(idealGradNorm));
        EXPECT_TRUE(idealGradNorm.size() == nEpoch);
    }

    {
        ifstream f(tools::getTestAssetsDir() / "Tacotron" / "Alice07" / "adam" / "learning_rate.data");
        std::copy(istream_iterator<dtype>(f), istream_iterator<dtype>(), back_inserter(idealLearningRate));
        EXPECT_TRUE(idealLearningRate.size() == nEpoch);
    }

    TENSORS_CREATE(BATCH_SIZE);
    MemoryManager& memory_manager = work.getMemoryManager();
    createAdamTensors(work.getTrainableParameterNames(), memory_manager);

    auto [TACOTRON_TRAINABLE_PARAMS, TACOTRON_PARAMS] = tacotronParamsCount(params);

    EXPECT_EQ(TACOTRON_GOLDEN_PARAMS, TACOTRON_PARAMS);
    EXPECT_EQ(TACOTRON_GOLDEN_TRAINABLE_PARAMS, TACOTRON_TRAINABLE_PARAMS);

    size_t loaded = loadTacotronParams((tools::getTestAssetsDir() / "Tacotron" / "pretrained" / "300000_Tacotron_model.").string(), memory_manager, "T", params, true);
    EXPECT_EQ(loaded, TACOTRON_PARAMS + 2 * TACOTRON_TRAINABLE_PARAMS - 3); // -1 because speaker_embedding will be loaded separately and doesn't have Adam params

    // targets
    ASSERT_TRUE(loadTFData(tools::getTestAssetsDir() / "Tacotron" / "Alice07" / "stop_token_targets_1.data", memory_manager["stop_token_targets"]));
    ASSERT_TRUE(loadTFData(tools::getTestAssetsDir() / "Tacotron" / "Alice07" / "targets_lengths_1.data", memory_manager["targets_lengths"]));
    ASSERT_TRUE(loadTFData(tools::getTestAssetsDir() / "Tacotron" / "Alice07" / "mel_targets_1.data", memory_manager["mel_targets"]));

    // training data
    ASSERT_TRUE(loadTFData(tools::getTestAssetsDir() / "Tacotron" / "Alice07" / "input_lengths_1.data", memory_manager["input_lengths"]));
    ASSERT_TRUE(loadTFData(tools::getTestAssetsDir() / "Tacotron" / "Alice07" / "encoder_output_1.data", memory_manager["encoder_output"]));

    // speaker embedding
    ASSERT_TRUE(loadTFData(tools::getTestAssetsDir() / "Tacotron" / "Alice07" / "speaker_embedding.data", memory_manager[Name("T") / "speaker_embedding"]));

    auto optimizer = createOptimizer(trainParams, ADAM_BETA1, ADAM_BETA2, ADAM_EPSILON);

    postprocessing::GradientClipping clip(trainParams.clipNorm);
    ofstream fLoss("loss_output.data"); // before target_before after target_after stop_token target_stop_token
    for (size_t epoch = 0; epoch < nEpoch; ++epoch)
    {
        cout << "Epoch " << epoch << endl;
        work.forwardPassTraining();

        auto ab = memory_manager["before_loss"][0];
        auto aa = memory_manager["after_loss"][0];
        auto as = params.withoutStopTokenLoss ? 0_dt : memory_manager["stop_token_loss"][0];

        cout << "  Actual loss: " << memory_manager["loss"][0] << " (before: " << ab << ", after: " << aa << ", stop_token: " << as << ")" << endl;

        auto tb = idealLosses[3 * epoch];
        auto ta = idealLosses[3 * epoch + 1];
        auto ts = params.withoutStopTokenLoss ? 0_dt : idealLosses[3 * epoch + 2];

        auto targetLoss = tb + ta + ts;
        cout << "  Target loss: " << targetLoss << " (before: " << tb << ", after: " << ta << ", stop_token: " << ts << ")" << endl;
        cout << "    Loss Difference: " << fabs(targetLoss - memory_manager["loss"][0]) << " (before: " << fabs(tb - ab) << ", after: " << fabs(ta - aa) << ", stop_token: " << fabs(ts - as) << ")"
             << endl;

        fLoss << ab << " " << tb << " " << aa << " " << ta << " " << as << " " << ts << endl;
        fLoss << ab << " " << aa << " " << as << endl;
        fLoss.flush();

        work.backwardPassTraining();

        auto trainableParams = work.getTrainableParameters();
        auto gnorm = clip.calcGlobalNorm(trainableParams, work.getNetworkParameters());

        if (trainParams.clipGradients)
        {
            clip.processGradients(trainableParams, work.getNetworkParameters());
        }

        if (trainParams.decayLearningRate)
        {
            optimizer->step();
        }

        cout << "  Global grad norm: " << gnorm << ". Ideal grad norm: " << idealGradNorm[epoch] << endl;
        cout << "    Grad Difference: " << fabs(gnorm - idealGradNorm[epoch]) << endl;
        cout << "  Learning rate: " << optimizer->getLearningRate() << ". Ideal grad norm: " << idealLearningRate[epoch] << endl;
        cout << "    Learning rate Difference: " << fabs(optimizer->getLearningRate() - idealLearningRate[epoch]) << endl;

        for (auto& p : trainableParams)
        {
            optimizer->operator()(memory_manager, p.Param, p.Gradient);
        }
    }
}

TEST(TestTacotron, Alice07StopTokenLossUnit)
{
    PROFILE_TEST

    Workflow work;

    constexpr size_t BATCH_SIZE = 1;
    constexpr size_t nEpoch = 25;

    TacotronParams params({}, {}, {});
    params.maxMelFrames = 264; // @todo: read from file

    // labels

    work.add<DataLayer>("targets_lengths", DataParams{ { "targets_lengths" }, 1u, 1u, 1u });
    work.add<DataLayer>("stop_token", DataParams{ { "stop_token_targets", "stop_token_predictions" }, 1u, params.maxMelFrames, 1u });

    Names goldenStopToken;
    for (size_t i = 0; i < nEpoch; ++i)
    {
        goldenStopToken.push_back("golden_stop_token_predictions[" + to_string(i) + "]");
    }
    work.add<DataLayer>("golden_stop_token", DataParams{ goldenStopToken, 1u, params.maxMelFrames, 1u });

    tacotron::AddMaskedCrossEntropy(
        &work, "loss", { { "stop_token_predictions", "stop_token_targets", "targets_lengths" }, { "stop_token_loss" } }, params.outputsPerStep, params.maskedSigmoidEpsilon, true);

    vector<dtype> idealLosses;
    {
        ifstream f(tools::getTestAssetsDir() / "Tacotron" / "Alice07" / "stop_token_loss_test" / "loss.data");
        std::copy(istream_iterator<dtype>(f), istream_iterator<dtype>(), back_inserter(idealLosses));
    }

    EXPECT_TRUE(idealLosses.size() >= nEpoch * 3);

    TENSORS_CREATE(BATCH_SIZE);
    MemoryManager& memory_manager = work.getMemoryManager();
    // targets
    EXPECT_TRUE(loadTFData(tools::getTestAssetsDir() / "Tacotron" / "Alice07" / "stop_token_targets_1.data", memory_manager["stop_token_targets"]));
    EXPECT_TRUE(loadTFData(tools::getTestAssetsDir() / "Tacotron" / "Alice07" / "stop_token_loss_test" / "stop_token_prediction.data", memory_manager, goldenStopToken));
    EXPECT_TRUE(loadTFData(tools::getTestAssetsDir() / "Tacotron" / "Alice07" / "targets_lengths_1.data", memory_manager["targets_lengths"]));

    for (size_t epoch = 0; epoch < nEpoch; ++epoch)
    {
        cout << "Epoch " << epoch << endl;

        memory_manager["stop_token_predictions"] = TORANGE(memory_manager[goldenStopToken[epoch]]);

        work.forwardPassTraining();

        auto as = memory_manager["stop_token_loss"][0];

        cout << "  Actual loss: " << as << endl;

        auto ts = idealLosses[3 * epoch + 2];

        cout << "  Target loss: " << ts << endl;
        cout << "  Loss Difference: " << fabs(ts - as) << endl;

        work.backwardPassTraining();
    }
}

TEST(TestTacotron, AliceStopTokenLossUnit)
{
    PROFILE_TEST

    Workflow work;

    constexpr size_t BATCH_SIZE = 15;
    constexpr size_t nEpoch = 1;

    TacotronParams params({}, {}, {});
    params.maxMelFrames = 546; // @todo: read from file

    // labels
    work.add<DataLayer>("targets_lengths", DataParams{ { "targets_lengths" }, 1u, 1u, 1u });
    work.add<DataLayer>("stop_token", DataParams{ { "stop_token_targets", "stop_token_predictions" }, 1u, params.maxMelFrames, 1u });

    Names goldenStopToken;
    for (size_t i = 0; i < nEpoch; ++i)
    {
        goldenStopToken.push_back("golden_stop_token_predictions[" + to_string(i) + "]");
    }
    work.add<DataLayer>("golden_stop_token", DataParams{ goldenStopToken, 1u, params.maxMelFrames, 1u });

    tacotron::AddMaskedCrossEntropy(
        &work, "loss", { { "stop_token_predictions", "stop_token_targets", "targets_lengths" }, { "stop_token_loss" } }, params.outputsPerStep, params.maskedSigmoidEpsilon, true);

    vector<dtype> idealLosses;
    {
        ifstream f(tools::getTestAssetsDir() / "Tacotron" / "Alice" / "loss.data");
        std::copy(istream_iterator<dtype>(f), istream_iterator<dtype>(), back_inserter(idealLosses));
    }

    EXPECT_TRUE(idealLosses.size() == 3);

    TENSORS_CREATE(BATCH_SIZE);
    MemoryManager& memory_manager = work.getMemoryManager();
    // targets
    EXPECT_TRUE(loadTFData(tools::getTestAssetsDir() / "Tacotron" / "Alice" / "stop_token_targets_15_.data", memory_manager["stop_token_targets"]));
    EXPECT_TRUE(loadTFData(tools::getTestAssetsDir() / "Tacotron" / "Alice" / "stop_token_prediction_.data", memory_manager, goldenStopToken));
    EXPECT_TRUE(loadTFData(tools::getTestAssetsDir() / "Tacotron" / "Alice" / "targets_lengths_15_.data", memory_manager["targets_lengths"]));

    for (size_t epoch = 0; epoch < nEpoch; ++epoch)
    {
        cout << "Epoch " << epoch << endl;

        memory_manager["stop_token_predictions"] = TORANGE(memory_manager[goldenStopToken[epoch]]);

        work.forwardPassTraining();

        auto as = memory_manager["stop_token_loss"][0];

        cout << "  Actual loss: " << as << endl;

        auto ts = idealLosses[3 * epoch + 2];

        cout << "  Target loss: " << ts << endl;
        cout << "  Loss Difference: " << fabs(ts - as) << endl;
    }
}

TEST(TestTacotron, LearningRateUnit)
{
    PROFILE_TEST

    using namespace optimizers;

    constexpr dtype ADAM_BETA1 = 0.9_dt;
    constexpr dtype ADAM_BETA2 = 0.999_dt;
    constexpr dtype ADAM_EPSILON = 1e-6_dt;

    constexpr size_t nEpoch = 499;

    constexpr dtype EPS = 1e-6_dt;

    TacotronTrainingParams trainParams;
    trainParams.initialLearningRate = 1e-2_dt;
    trainParams.clipNorm = 1.0_dt;
    trainParams.decayLearningRate = true;
    trainParams.decaySteps = 1000;
    trainParams.decayStartStep = 200;
    trainParams.decayRate = 0.5;
    trainParams.finalLearningRate = 1e-4_dt;
    trainParams.warmupLearningRate = true;
    trainParams.warmupSteps = 400;

    auto optimizer = createOptimizer(trainParams, ADAM_BETA1, ADAM_BETA2, ADAM_EPSILON);

    vector<dtype> lr;
    ifstream f(tools::getTestAssetsDir() / "Tacotron" / "learning_rate.data");
    std::copy(istream_iterator<dtype>(f), istream_iterator<dtype>(), back_inserter(lr));
    EXPECT_TRUE(lr.size() == nEpoch);

    for (size_t i = 0; i < nEpoch; ++i)
    {
        optimizer->step();
        ASSERT_TRUE(tools::expect_near_relative(optimizer->getLearningRate(), lr[i], EPS));
    }
}

TEST(TestTacotron, DISABLED_AliceTraining)
{
    PROFILE_TEST

    using namespace optimizers;

    random::setGlobalSeed(42);

    bool usePool = false;
    Workflow work(CompressionMode::NONE, raul::CalculationMode::DETERMINISTIC, usePool ? raul::AllocationMode::POOL : raul::AllocationMode::STANDARD);

    auto testPath = tools::getTestAssetsDir() / "Tacotron";
    string experimentName = "general";
    auto outputPrefix = testPath / "Alice" / experimentName;

    constexpr size_t TACOTRON_GOLDEN_PARAMS = 53;
    constexpr size_t TACOTRON_GOLDEN_TRAINABLE_PARAMS = 43;

    constexpr size_t BATCH_SIZE = 15;
    constexpr size_t nEpoch = 300;

    auto suffix = BATCH_SIZE == 8 ? "" : "_" + to_string(BATCH_SIZE);

    const size_t nEpochToSave = 100;

    constexpr dtype ADAM_BETA1 = 0.9_dt;
    constexpr dtype ADAM_BETA2 = 0.999_dt;
    constexpr dtype ADAM_EPSILON = 1e-6_dt;

    TacotronParams params({ "encoder_output", "input_lengths", "mel_targets" }, { "decoder_output", "mel_outputs", "stop_token_predictions" }, {});
    params.dropoutRate = 0.5f;
    params.postnetDropoutRate = 0.5f;
    params.zoneoutRate = 0.1f;
    params.attentionSigmoidNoise = 0.2f;

    params.maxMelFrames = 546; // @todo: read from file

    TacotronTrainingParams trainParams;
    trainParams.initialLearningRate = 3e-3_dt;
    trainParams.clipNorm = 1.0_dt;
    trainParams.decayLearningRate = true;
    trainParams.decaySteps = 800;
    trainParams.decayStartStep = 0;
    trainParams.decayRate = 0.5;
    trainParams.finalLearningRate = 5e-5_dt;
    trainParams.warmupLearningRate = true;
    trainParams.warmupSteps = 200;

    size_t T_IN = 94; // number of steps in the input time series

    work.add<DataLayer>("encoder_output", DataParams{ { "encoder_output" }, 1u, T_IN, params.embeddingDim });
    work.add<DataLayer>("input_lengths", DataParams{ { "input_lengths" }, 1u, 1u, 1u });
    work.add<DataLayer>("mel_targets", DataParams{ { "mel_targets" }, 1u, params.maxMelFrames, params.numMels });

    // labels
    work.add<DataLayer>("targets_lengths", DataParams{ { "targets_lengths" }, 1u, 1u, 1u });
    work.add<DataLayer>("stop_token_targets", DataParams{ { "stop_token_targets" }, 1u, params.maxMelFrames, 1u });

    AddSingleSpeakerFslTacotronModel(&work, "T", params);
    tacotron::AddTacotronLoss(&work,
                              "total_loss",
                              { { "decoder_output", "mel_outputs", "stop_token_predictions", "mel_targets", "targets_lengths", "stop_token_targets" },
                                params.withoutStopTokenLoss ? Names{ "loss", "before_loss", "after_loss" } : Names{ "loss", "before_loss", "after_loss", "stop_token_loss" } },
                              params);

    auto timeStart = chrono::steady_clock::now();
    TENSORS_CREATE(BATCH_SIZE);

    MemoryManager& memory_manager = work.getMemoryManager();
    createAdamTensors(work.getTrainableParameterNames(), memory_manager);

    auto epochTime = TODTYPE(chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - timeStart).count());
    cout << "Workflow preparation time " << epochTime << " ms for batch size " << BATCH_SIZE << endl;
    cout << memory_manager.size() << " tensors" << endl;
    cout << work.getTrainableParameterNames().size() << " trainable params" << endl;
    cout << tools::get_size_of_trainable_params(work) << " trainable weights" << endl;

    auto [TACOTRON_TRAINABLE_PARAMS, TACOTRON_PARAMS] = tacotronParamsCount(params);

    EXPECT_EQ(TACOTRON_GOLDEN_PARAMS, TACOTRON_PARAMS);
    EXPECT_EQ(TACOTRON_GOLDEN_TRAINABLE_PARAMS, TACOTRON_TRAINABLE_PARAMS);
    size_t loaded = loadTacotronParams((tools::getTestAssetsDir() / "Tacotron" / "pretrained" / "300000_Tacotron_model.").string(), memory_manager, "T", params, true);
    EXPECT_EQ(loaded, TACOTRON_PARAMS + 2 * TACOTRON_TRAINABLE_PARAMS - 3); // -3 because speaker_embedding will be loaded separately and doesn't have Adam params

    // targets
    ASSERT_TRUE(loadTFData(testPath / "Alice" / ("stop_token_targets" + suffix + ".data"), memory_manager["stop_token_targets"]));
    ASSERT_TRUE(loadTFData(testPath / "Alice" / ("targets_lengths" + suffix + ".data"), memory_manager["targets_lengths"]));
    ASSERT_TRUE(loadTFData(testPath / "Alice" / ("mel_targets" + suffix + ".data"), memory_manager["mel_targets"]));

    // training data
    ASSERT_TRUE(loadTFData(testPath / "Alice" / ("input_lengths" + suffix + ".data"), memory_manager["input_lengths"]));
    ASSERT_TRUE(loadTFData(testPath / "Alice" / ("encoder_output" + suffix + ".data"), memory_manager["encoder_output"]));

    // speaker embedding
    ASSERT_TRUE(loadTFData(testPath / "Alice" / ("speaker_embedding.data"), memory_manager[Name("T") / "speaker_embedding"]));

    auto optimizer = createOptimizer(trainParams, ADAM_BETA1, ADAM_BETA2, ADAM_EPSILON);

    postprocessing::GradientClipping clip(trainParams.clipNorm);

    ofstream fLoss("loss_" + experimentName + ".data");      // before target_before after target_after stop_token target_stop_token
    ofstream fLrGrad("lr_grad_" + experimentName + ".data"); // before target_before after target_after stop_token target_stop_token

    dtype fastestEpochTime = 1e+6_dt;
    dtype slowestEpochRime = 0;
    dtype trainingTime = 0;

    for (size_t epoch = 0; epoch < nEpoch; ++epoch)
    {
        cout << "Epoch " << epoch << endl;
        timeStart = chrono::steady_clock::now();

        work.forwardPassTraining();
        work.backwardPassTraining();

        auto trainableParams = work.getTrainableParameters();
        clip.processGradients(trainableParams, work.getNetworkParameters());
        auto gnorm_before = clip.getGlobalNorm(work.getNetworkParameters());
        auto gnorm_after = clip.calcGlobalNorm(trainableParams, work.getNetworkParameters());

        if (trainParams.decayLearningRate)
        {
            optimizer->step();
        }

        for (auto p : trainableParams)
        {
            optimizer->operator()(memory_manager, p.Param, p.Gradient);
        }

        auto& lossValue = memory_manager["loss"];
        epochTime = TODTYPE(chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - timeStart).count());

        trainingTime += epochTime;

        fastestEpochTime = std::min(fastestEpochTime, epochTime);
        slowestEpochRime = std::max(slowestEpochRime, epochTime);

        cout << "  Epoch time " << epochTime << " ms for batch size " << BATCH_SIZE << endl;
        cout << "  Average performance: " << trainingTime / TODTYPE(epoch + 1) << " ms/epoch (fastest: " << fastestEpochTime << ", slowest " << slowestEpochRime << ")" << endl << endl;
        cout << "  Loss = " << lossValue[0] << " (before = " << memory_manager["before_loss"][0] << ", after = " << memory_manager["after_loss"][0]
             << ", stop_token = " << (params.withoutStopTokenLoss ? 0_dt : memory_manager["stop_token_loss"][0]) << ")" << endl;

        cout << "  Global grad norm: " << gnorm_before << " -> " << gnorm_after << endl;

        fLoss << epoch + 1 << " " << memory_manager["before_loss"][0] << " " << memory_manager["after_loss"][0] << " " << (params.withoutStopTokenLoss ? 0_dt : memory_manager["stop_token_loss"][0])
              << " " << lossValue[0] << endl;

        auto lr = optimizer->getLearningRate();
        fLrGrad << epoch + 1 << " " << lr << " " << gnorm_before << endl;

        cout << "  learning rate: " << lr << endl;

        if (nEpochToSave > 0 && ((epoch + 1) % nEpochToSave == 0 || epoch == nEpoch - 1))
        {
            [[maybe_unused]] size_t saved = saveTacotronParams(outputPrefix / to_string(epoch + 1), "300000_Tacotron_model.", memory_manager, "T", params, true, false);
        }
    }
}

TEST(TestTacotron, DISABLED_AliceDeterministicTraining)
{
    PROFILE_TEST

    using namespace optimizers;

    constexpr size_t TACOTRON_GOLDEN_PARAMS = 53;
    constexpr size_t TACOTRON_GOLDEN_TRAINABLE_PARAMS = 43;

    constexpr auto EPS = 1e-2_dt;
    constexpr auto EPS_REL_GRAD = 5e-4_dt;

    random::setGlobalSeed(42);

    bool usePool = false;
    Workflow work(CompressionMode::NONE, raul::CalculationMode::DETERMINISTIC, usePool ? raul::AllocationMode::POOL : raul::AllocationMode::STANDARD);

    constexpr size_t BATCH_SIZE = 15;
    constexpr size_t nEpoch = 50;

    auto suffix = BATCH_SIZE == 8 ? "" : "_" + to_string(BATCH_SIZE);
    auto testPath = tools::getTestAssetsDir() / "Tacotron";

    constexpr dtype ADAM_BETA1 = 0.9_dt;
    constexpr dtype ADAM_BETA2 = 0.999_dt;
    constexpr dtype ADAM_EPSILON = 1e-6_dt;

    vector<dtype> idealLosses;
    vector<dtype> idealLrGrad;
    {
        ifstream f(tools::getTestAssetsDir() / "Tacotron" / "Alice" / "deterministic" / "loss.data");
        std::copy(istream_iterator<dtype>(f), istream_iterator<dtype>(), back_inserter(idealLosses));
    }
    {
        ifstream f(tools::getTestAssetsDir() / "Tacotron" / "Alice" / "deterministic" / "lr_grad.data");
        std::copy(istream_iterator<dtype>(f), istream_iterator<dtype>(), back_inserter(idealLrGrad));
    }
    EXPECT_TRUE(idealLosses.size() == 3 * nEpoch);
    EXPECT_TRUE(idealLrGrad.size() == 2 * nEpoch);

    TacotronParams params({ "encoder_output", "input_lengths", "mel_targets" }, { "decoder_output", "mel_outputs", "stop_token_predictions" }, {});
    params.dropoutRate = 0.0f;
    params.postnetDropoutRate = 0.0f;
    params.zoneoutRate = 0.0f;
    params.attentionSigmoidNoise = 0.0f;

    params.withoutStopTokenLoss = true;
    params.maxMelFrames = 546; // @todo: read from file

    TacotronTrainingParams trainParams;
    trainParams.initialLearningRate = 1e-2_dt;
    trainParams.clipNorm = 1.0_dt;
    trainParams.decayLearningRate = true;
    trainParams.decaySteps = 1000;
    trainParams.decayStartStep = 200;
    trainParams.decayRate = 0.5;
    trainParams.finalLearningRate = 1e-4_dt;
    trainParams.warmupLearningRate = true;
    trainParams.warmupSteps = 400;

    size_t T_IN = 94; // number of steps in the input time series

    work.add<DataLayer>("encoder_output", DataParams{ { "encoder_output" }, 1u, T_IN, params.embeddingDim });
    work.add<DataLayer>("input_lengths", DataParams{ { "input_lengths" }, 1u, 1u, 1u });
    work.add<DataLayer>("mel_targets", DataParams{ { "mel_targets" }, 1u, params.maxMelFrames, params.numMels });

    // labels
    work.add<DataLayer>("targets_lengths", DataParams{ { "targets_lengths" }, 1u, 1u, 1u });
    work.add<DataLayer>("stop_token_targets", DataParams{ { "stop_token_targets" }, 1u, params.maxMelFrames, 1u });

    AddSingleSpeakerFslTacotronModel(&work, "T", params);
    tacotron::AddTacotronLoss(&work,
                              "total_loss",
                              { { "decoder_output", "mel_outputs", "stop_token_predictions", "mel_targets", "targets_lengths", "stop_token_targets" },
                                params.withoutStopTokenLoss ? Names{ "loss", "before_loss", "after_loss" } : Names{ "loss", "before_loss", "after_loss", "stop_token_loss" } },
                              params);

    TENSORS_CREATE(BATCH_SIZE);
    MemoryManager& memory_manager = work.getMemoryManager();
    createAdamTensors(work.getTrainableParameterNames(), memory_manager);

    auto [TACOTRON_TRAINABLE_PARAMS, TACOTRON_PARAMS] = tacotronParamsCount(params);

    EXPECT_EQ(TACOTRON_GOLDEN_PARAMS, TACOTRON_PARAMS);
    EXPECT_EQ(TACOTRON_GOLDEN_TRAINABLE_PARAMS, TACOTRON_TRAINABLE_PARAMS);

    size_t loaded = loadTacotronParams((tools::getTestAssetsDir() / "Tacotron" / "pretrained" / "300000_Tacotron_model.").string(), memory_manager, "T", params, true);
    EXPECT_EQ(loaded, TACOTRON_PARAMS + 2 * TACOTRON_TRAINABLE_PARAMS - 3); // -3 because speaker_embedding will be loaded separately and doesn't have Adam params

    // targets
    ASSERT_TRUE(loadTFData(testPath / "Alice" / ("stop_token_targets" + suffix + ".data"), memory_manager["stop_token_targets"]));
    ASSERT_TRUE(loadTFData(testPath / "Alice" / ("targets_lengths" + suffix + ".data"), memory_manager["targets_lengths"]));
    ASSERT_TRUE(loadTFData(testPath / "Alice" / ("mel_targets" + suffix + ".data"), memory_manager["mel_targets"]));

    // training data
    ASSERT_TRUE(loadTFData(testPath / "Alice" / ("input_lengths" + suffix + ".data"), memory_manager["input_lengths"]));
    ASSERT_TRUE(loadTFData(testPath / "Alice" / ("encoder_output" + suffix + ".data"), memory_manager["encoder_output"]));

    // speaker embedding
    ASSERT_TRUE(loadTFData(testPath / "Alice" / ("speaker_embedding.data"), memory_manager[Name("T") / "speaker_embedding"]));

    auto optimizer = createOptimizer(trainParams, ADAM_BETA1, ADAM_BETA2, ADAM_EPSILON);
    postprocessing::GradientClipping clip(trainParams.clipNorm);

    dtype trainingTime = 0;

    for (size_t epoch = 0; epoch < nEpoch; ++epoch)
    {
        auto epochStart = chrono::steady_clock::now();
        cout << "Epoch " << epoch << endl;

        auto timeStart = chrono::steady_clock::now();
        work.forwardPassTraining();
        auto totalTime = TODTYPE(chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - timeStart).count());

        auto ab = memory_manager["before_loss"][0];
        auto aa = memory_manager["after_loss"][0];
        auto as = params.withoutStopTokenLoss ? 0_dt : memory_manager["stop_token_loss"][0];

        auto idealLoss = idealLosses[3 * epoch] + idealLosses[3 * epoch + 1] + idealLosses[3 * epoch + 2];

        cout << "  loss = " << memory_manager["loss"][0] << " (before = " << ab << ", after = " << aa << ", stop_token = " << as << ")" << endl;
        cout << "  diff = " << fabs(memory_manager["loss"][0] - idealLoss) << " (before = " << fabs(ab - idealLosses[3 * epoch]) << ", after = " << fabs(aa - idealLosses[3 * epoch + 1])
             << ", stop_token = " << fabs(as - idealLosses[3 * epoch + 2]) << ")" << endl;
        cout << "  forward: " << totalTime << " ms" << endl;

        EXPECT_NEAR(ab, idealLosses[3 * epoch], EPS);
        EXPECT_NEAR(aa, idealLosses[3 * epoch + 1], EPS);
        EXPECT_NEAR(as, idealLosses[3 * epoch + 2], EPS);

        timeStart = chrono::steady_clock::now();
        work.backwardPassTraining();
        totalTime = TODTYPE(chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - timeStart).count());

        cout << "  backward: " << totalTime << " ms" << endl;

        timeStart = chrono::steady_clock::now();
        auto trainableParams = work.getTrainableParameters();
        clip.processGradients(trainableParams, work.getNetworkParameters());
        totalTime = TODTYPE(chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - timeStart).count());

        auto gnorm = clip.getGlobalNorm(work.getNetworkParameters());
        auto new_norm = clip.calcGlobalNorm(trainableParams, work.getNetworkParameters());

        if (trainParams.decayLearningRate)
        {
            optimizer->step();
        }

        auto lr = optimizer->getLearningRate();

        EXPECT_NEAR(lr, idealLrGrad[2 * epoch], 1e-6);
        tools::expect_near_relative(gnorm, idealLrGrad[2 * epoch + 1], EPS_REL_GRAD);

        cout << "  gradient clipping: (" << gnorm << " ->" << new_norm << "): " << totalTime << " ms" << endl;

        timeStart = chrono::steady_clock::now();

        for (auto& p : trainableParams)
        {
            optimizer->operator()(memory_manager, p.Param, p.Gradient);
        }

        totalTime = TODTYPE(chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - timeStart).count());

        cout << "  optimizer (lr=" << lr << "): " << totalTime << " ms" << endl;

        auto epochTime = TODTYPE(chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - epochStart).count());
        trainingTime += epochTime;

        cout << "  epoch: " << epochTime << " ms" << endl;
        cout << "Average performance: " << trainingTime / TODTYPE(epoch + 1) << " ms/epoch" << endl << endl;
    }
}

TEST(TestTacotron, DecoderRNNUnit)
{
    PROFILE_TEST

    // Test parameters
    constexpr auto eps_rel = 1e-4_dt;
    constexpr auto eps_abs = 1e-6_dt;
    constexpr auto input_size = 896U;
    constexpr auto hidden_size = 512U;
    constexpr auto batch_size = 8U;

    Workflow work;

    Names rnn_cell_state;
    Names next_cell_state;

    TacotronParams params({}, {}, {});
    params.dropoutRate = 0.f;
    params.postnetDropoutRate = 0.f;
    params.zoneoutRate = 0.f;

    for (size_t i = 0; i < params.decoderLstmUnits.size(); ++i)
    {
        rnn_cell_state.push_back("hidden" + std::to_string(i));
        rnn_cell_state.push_back("cell" + std::to_string(i));
        next_cell_state.push_back("next_hidden" + std::to_string(i));
        next_cell_state.push_back("next_cell" + std::to_string(i));

        work.add<DataLayer>("data[" + to_string(i) + "]",
                            DataParams{ { "hidden" + std::to_string(i), "cell" + std::to_string(i), "hidden_golden" + std::to_string(i), "cell_golden" + std::to_string(i) }, 1u, 1u, hidden_size });
    }
    Names inputs = rnn_cell_state;
    inputs.insert(inputs.begin(), "input");
    const size_t DECODER_RNN_TRAINABLE_PARAMS = params.decoderLstmUnits.size() * 2;
    work.add<DataLayer>("input", DataParams{ { "input" }, 1u, 1u, input_size });
    tacotron::AddDecoderRNN(&work, Name("T") / "decoder" / "_cell" / "decoder_LSTM", { inputs, { "output" } }, next_cell_state, params);

    TENSORS_CREATE(batch_size);
    MemoryManager& memory_manager = work.getMemoryManager();
    size_t loaded = loadTacotronParams((tools::getTestAssetsDir() / "Tacotron" / "pretrained" / "300000_Tacotron_model.").string(), memory_manager, "T", params, false);
    EXPECT_EQ(loaded, DECODER_RNN_TRAINABLE_PARAMS);
    EXPECT_EQ(work.getTrainableParameterNames().size(), DECODER_RNN_TRAINABLE_PARAMS);

    // Data loading
    EXPECT_TRUE(loadTFData<Tensor>(tools::getTestAssetsDir() / "Tacotron" / "lstm_input.data", { &memory_manager["input"] }));
    for (size_t i = 0; i < params.decoderLstmUnits.size(); ++i)
    {
        EXPECT_TRUE(loadTFData<Tensor>((tools::getTestAssetsDir() / "Tacotron" / "lstm_h_").string() + std::to_string(i) + ".data", { &memory_manager["hidden_golden" + std::to_string(i)] }));
        EXPECT_TRUE(loadTFData<Tensor>((tools::getTestAssetsDir() / "Tacotron" / "lstm_c_").string() + std::to_string(i) + ".data", { &memory_manager["cell_golden" + std::to_string(i)] }));
    }

    vector<pair<Name, Name>> checked;
    for (size_t j = 0; j < params.decoderLstmUnits.size(); ++j)
    {
        checked.emplace_back(make_pair(Name("next_hidden" + std::to_string(j)), Name("hidden_golden" + std::to_string(j))));
        checked.emplace_back(make_pair(Name("next_cell" + std::to_string(j)), Name("cell_golden" + std::to_string(j))));
    }
    TensorChecker checker(checked, eps_abs, eps_rel);
    work.getNetworkParameters().mCallback = checker;
    work.forwardPassTraining();
}

TEST(TestTacotron, DecoderRNNGpuUnit)
{
    PROFILE_TEST

    GPU_ONLY_TEST

    // Test parameters
    //constexpr auto eps_rel = 1e-4_dt;
    constexpr auto eps_abs = 1e-6_dt;
    constexpr auto input_size = 896U;
    constexpr auto hidden_size = 512U;
    constexpr auto batch_size = 8U;

    WorkflowEager work(CompressionMode::NONE, CalculationMode::DETERMINISTIC, AllocationMode::POOL, ExecutionTarget::GPU);
    auto& memory_manager = work.getMemoryManager<MemoryManagerGPU>();

    Names rnn_cell_state;
    Names next_cell_state;

    TacotronParams params({}, {}, {});
    params.dropoutRate = 0.f;
    params.postnetDropoutRate = 0.f;
    params.zoneoutRate = 0.f;

    for (size_t i = 0; i < params.decoderLstmUnits.size(); ++i)
    {
        rnn_cell_state.push_back("hidden" + std::to_string(i));
        rnn_cell_state.push_back("cell" + std::to_string(i));
        next_cell_state.push_back("next_hidden" + std::to_string(i));
        next_cell_state.push_back("next_cell" + std::to_string(i));

        work.add<TensorLayer>("data[" + to_string(i) + "]",
            TensorParams{ { "hidden" + std::to_string(i), "cell" + std::to_string(i) }, WShape{ BS(), 1u, 1u, hidden_size }, 0_dt});
        work.add<DataLayer>("data_golden[" + to_string(i) + "]",
            DataParams{ { "hidden_golden" + std::to_string(i), "cell_golden" + std::to_string(i) }, 1u, 1u, hidden_size });
    
    }
    Names inputs = rnn_cell_state;
    inputs.insert(inputs.begin(), "input");
    const size_t DECODER_RNN_TRAINABLE_PARAMS = params.decoderLstmUnits.size() * 2;
    work.add<DataLayer>("input", DataParams{ { "input" }, 1u, 1u, input_size });
    tacotron::AddDecoderRNN(&work, Name("T") / "decoder" / "_cell" / "decoder_LSTM", { inputs, { "output" } }, next_cell_state, params);

    TENSORS_CREATE(batch_size);
    size_t loaded = loadTacotronParams((tools::getTestAssetsDir() / "Tacotron" / "pretrained" / "300000_Tacotron_model.").string(), memory_manager, "T", params, false);
    EXPECT_EQ(loaded, DECODER_RNN_TRAINABLE_PARAMS);
    EXPECT_EQ(work.getTrainableParameterNames().size(), DECODER_RNN_TRAINABLE_PARAMS);

    // Data loading
    EXPECT_TRUE(loadTFData(tools::getTestAssetsDir() / "Tacotron" / "lstm_input.data", memory_manager["input"]));
    for (size_t i = 0; i < params.decoderLstmUnits.size(); ++i)
    {
        EXPECT_TRUE(loadTFData((tools::getTestAssetsDir() / "Tacotron" / "lstm_h_").string() + std::to_string(i) + ".data", memory_manager["hidden_golden" + std::to_string(i)]));
        EXPECT_TRUE(loadTFData((tools::getTestAssetsDir() / "Tacotron" / "lstm_c_").string() + std::to_string(i) + ".data", memory_manager["cell_golden" + std::to_string(i)]));
    }

    vector<pair<Name, Name>> checked;
    for (size_t j = 0; j < params.decoderLstmUnits.size(); ++j)
    {
        checked.emplace_back(make_pair(Name("next_hidden" + std::to_string(j)), Name("hidden_golden" + std::to_string(j))));
        checked.emplace_back(make_pair(Name("next_cell" + std::to_string(j)), Name("cell_golden" + std::to_string(j))));
    }
    ASSERT_NO_THROW(work.forwardPassTraining());
    UT::tools::checkTensors(checked, memory_manager, eps_abs);
}

TEST(TestTacotron, LSTMCellZeroStateUnit)
{
    PROFILE_TEST

    // Test parameters
    const auto eps_rel = 1e-4_dt;
    const auto input_size = 896U;
    const auto hidden_size = 512U;
    const auto batch_size = 8U;

    Workflow work;

    work.add<DataLayer>("input", DataParams{ { "in" }, 1, 1, input_size });
    work.add<DataLayer>("data", DataParams{ { "hidden", "cell", "hidden_golden", "cell_golden" }, 1, 1, hidden_size });

    const auto params = LSTMCellParams{ "in", "hidden", "cell", "new_hidden", "new_cell", {}, true, 0.0_dt, true, 1.0_dt };
    LSTMCellLayer("lstm", params, work.getNetworkParameters());
    TENSORS_CREATE(batch_size);
    MemoryManager& memory_manager = work.getMemoryManager();
    // Data loading
    EXPECT_TRUE(loadTFData<Tensor>(tools::getTestAssetsDir() / "Tacotron" / "lstm_input.data", { &memory_manager["in"] }));
    EXPECT_TRUE(loadTFData<Tensor>(tools::getTestAssetsDir() / "Tacotron" / "vanilla_lstm_h.data", { &memory_manager["hidden_golden"] }));
    EXPECT_TRUE(loadTFData<Tensor>(tools::getTestAssetsDir() / "Tacotron" / "vanilla_lstm_c.data", { &memory_manager["cell_golden"] }));

    auto& bias = memory_manager[Name("lstm") / "linear" / "Biases"];
    auto& weights = memory_manager[Name("lstm") / "linear" / "Weights"];
    DataLoader::loadData(tools::getTestAssetsDir() / "Tacotron" / "pretrained" / "300000_Tacotron_model.inference.decoder.decoder_LSTM.multi_rnn_cell.cell_0.decoder_LSTM_1.bias.0.data", bias);
    DataLoader::loadData(tools::getTestAssetsDir() / "Tacotron" / "pretrained" / "300000_Tacotron_model.inference.decoder.decoder_LSTM.multi_rnn_cell.cell_0.decoder_LSTM_1.kernel.0.data", weights);

    TensorChecker checker({ { "new_hidden", "hidden_golden" }, { "new_cell", "cell_golden" } }, -1_dt, eps_rel);
    work.getNetworkParameters().mCallback = checker;
    work.forwardPassTraining();
}

TEST(TestTacotron, LSTMCellZeroInputUnit)
{
    PROFILE_TEST

    // Test parameters
    const auto eps_rel = 1e-5_dt;
    const auto input_size = 896U;
    const auto hidden_size = 512U;
    const auto batch_size = 8U;

    Workflow work;

    work.add<DataLayer>("input", DataParams{ { "in" }, 1, 1, input_size });
    work.add<DataLayer>("data", DataParams{ { "hidden", "cell", "hidden_golden", "cell_golden" }, 1, 1, hidden_size });

    const auto params = LSTMCellParams{ "in", "hidden", "cell", "new_hidden", "new_cell", {}, true, 0.0_dt, true, 1.0_dt };
    LSTMCellLayer("lstm", params, work.getNetworkParameters());
    TENSORS_CREATE(batch_size);
    MemoryManager& memory_manager = work.getMemoryManager();
    // Data loading
    EXPECT_TRUE(loadTFData<Tensor>(tools::getTestAssetsDir() / "Tacotron" / "vanilla_lstm_h_zeros.data", { &memory_manager["hidden_golden"] }));
    EXPECT_TRUE(loadTFData<Tensor>(tools::getTestAssetsDir() / "Tacotron" / "vanilla_lstm_c_zeros.data", { &memory_manager["cell_golden"] }));

    auto& bias = memory_manager[Name("lstm") / "linear" / "Biases"];
    auto& weights = memory_manager[Name("lstm") / "linear" / "Weights"];
    DataLoader::loadData(tools::getTestAssetsDir() / "Tacotron" / "pretrained" / "300000_Tacotron_model.inference.decoder.decoder_LSTM.multi_rnn_cell.cell_0.decoder_LSTM_1.bias.0.data", bias);
    DataLoader::loadData(tools::getTestAssetsDir() / "Tacotron" / "pretrained" / "300000_Tacotron_model.inference.decoder.decoder_LSTM.multi_rnn_cell.cell_0.decoder_LSTM_1.kernel.0.data", weights);

    TensorChecker checker({ { "new_hidden", "hidden_golden" }, { "new_cell", "cell_golden" } }, -1_dt, eps_rel);
    work.getNetworkParameters().mCallback = checker;
    work.forwardPassTraining();
}

TEST(TestTacotron, LSTMCellOnesInputUnit)
{
    PROFILE_TEST

    // Test parameters
    const auto eps_rel = 1e-3_dt;
    const auto eps_abs = 1e-5_dt;
    const auto input_size = 896U;
    const auto hidden_size = 512U;
    const auto batch_size = 8U;

    Workflow work;

    work.add<DataLayer>("input", DataParams{ { "in" }, 1, 1, input_size });
    work.add<DataLayer>("data", DataParams{ { "hidden", "cell", "hidden_golden", "cell_golden" }, 1, 1, hidden_size });

    const auto params = LSTMCellParams{ "in", "hidden", "cell", "new_hidden", "new_cell", {}, true, 0.0_dt, true, 1.0_dt };
    LSTMCellLayer("lstm", params, work.getNetworkParameters());
    TENSORS_CREATE(batch_size);
    MemoryManager& memory_manager = work.getMemoryManager();
    // Data loading
    EXPECT_TRUE(loadTFData<Tensor>(tools::getTestAssetsDir() / "Tacotron" / "vanilla_lstm_h_ones.data", { &memory_manager["hidden_golden"] }));
    EXPECT_TRUE(loadTFData<Tensor>(tools::getTestAssetsDir() / "Tacotron" / "vanilla_lstm_c_ones.data", { &memory_manager["cell_golden"] }));
    memory_manager["in"] = 1_dt;
    auto& bias = memory_manager[Name("lstm") / "linear" / "Biases"];
    auto& weights = memory_manager[Name("lstm") / "linear" / "Weights"];
    DataLoader::loadData(tools::getTestAssetsDir() / "Tacotron" / "pretrained" / "300000_Tacotron_model.inference.decoder.decoder_LSTM.multi_rnn_cell.cell_0.decoder_LSTM_1.bias.0.data", bias);
    DataLoader::loadData(tools::getTestAssetsDir() / "Tacotron" / "pretrained" / "300000_Tacotron_model.inference.decoder.decoder_LSTM.multi_rnn_cell.cell_0.decoder_LSTM_1.kernel.0.data", weights);

    TensorChecker checker({ { "new_hidden", "hidden_golden" }, { "new_cell", "cell_golden" } }, eps_abs, eps_rel);
    work.getNetworkParameters().mCallback = checker;
    work.forwardPassTraining();
}

TEST(TestTacotron, Alice07DecoderCellUnit)
{
    PROFILE_TEST

    using namespace optimizers;

    constexpr size_t DECODER_CELL_PARAMS = 20;

    Workflow work;

    constexpr size_t BATCH_SIZE = 1;

    constexpr dtype FRAME_PREDICTION_EPS = 5e-5_dt;
    constexpr dtype STOP_TOKEN_EPS = 1e-4_dt;

    TacotronParams params({}, {}, {});
    params.dropoutRate = 0.f;
    params.postnetDropoutRate = 0.f;
    params.zoneoutRate = 0.f;
    params.attentionSigmoidNoise = 0.f;
    params.maxMelFrames = 264; // @todo: read from file
    size_t T_IN = 35;          // number of steps in the input time series

    size_t decoder_iterations = params.maxMelFrames / params.outputsPerStep;

    vector<shared_ptr<BasicLayer>> cells, refinedCells;
    Names initialRnnState;
    for (size_t i = 0; i < params.decoderLstmUnits.size(); ++i)
    {
        initialRnnState.emplace_back("initial_rnn_state_h[" + to_string(i) + "]");
        initialRnnState.emplace_back("initial_rnn_state_c[" + to_string(i) + "]");
    }

    Names initialData = { "zero_input", "initial_attention", "initial_alignments" };
    copy(initialRnnState.begin(), initialRnnState.end(), back_inserter(initialData));
    Names stepInputs = { "embedded_encoder_output", "input_lengths" };
    copy(initialData.begin(), initialData.end(), back_inserter(stepInputs));

    auto mainCellName = Name("T") / "decoder" / "_cell";
    auto cellName = mainCellName;
    Names mels;
    Names goldenAttentions;
    Name goldenAttentionValues = "golden_attention_values";
    Names goldenAlignments;
    vector<Names> goldenRnnState(params.decoderLstmUnits.size() * 2);

    for (size_t time = 0; time < decoder_iterations; ++time)
    {
        auto suffix = "[" + to_string(time) + "]";
        mels.push_back("mels" + suffix);

        goldenAttentions.push_back("golden_attention" + suffix);
        goldenAlignments.push_back("golden_alignments" + suffix);

        for (size_t i = 0; i < params.decoderLstmUnits.size(); ++i)
        {
            goldenRnnState[i].emplace_back("golden_rnn_cell_state_h[" + to_string(i) + "]" + suffix);
            goldenRnnState[i + params.decoderLstmUnits.size()].emplace_back("golden_rnn_cell_state_c[" + to_string(i) + "]" + suffix);
        }
    }

    work.add<DataLayer>("initial_alignments", DataParams{ { "initial_alignments" }, 1u, 1u, T_IN });
    work.add<DataLayer>("initial_attention", DataParams{ { "initial_attention" }, 1u, 1u, params.embeddingDim + params.speakerEmbeddingSize });
    work.add<DataLayer>("zero_input", DataParams{ { "zero_input" }, 1u, 1u, params.numMels });

    work.add<DataLayer>("golden_decoder_output", DataParams{ { "golden_decoder_output" }, 1u, params.maxMelFrames, params.numMels });
    work.add<DataLayer>("golden_stop_token", DataParams{ { "golden_stop_token" }, 1u, params.maxMelFrames, 1u });

    work.add<DataLayer>("embedded_encoder_output", DataParams{ { "embedded_encoder_output" }, 1u, T_IN, params.embeddingDim + params.speakerEmbeddingSize });
    work.add<DataLayer>("input_lengths", DataParams{ { "input_lengths" }, 1u, 1u, 1u });
    work.add<DataLayer>("mel_targets", DataParams{ { "mel_targets" }, 1u, params.maxMelFrames, params.numMels });

    work.add<TensorLayer>("reduced_mel_targets",
                          TensorParams{ { "reduced_mel_targets" }, WShape{ BS(), 1u, decoder_iterations, params.numMels } });

    work.add<DataLayer>(goldenAttentionValues, DataParams{ { goldenAttentionValues }, 1u, T_IN, params.embeddingDim + params.speakerEmbeddingSize });

    work.add<DataLayer>("golden_attentions", DataParams{ goldenAttentions, 1u, 1u, params.embeddingDim + params.speakerEmbeddingSize });
    work.add<DataLayer>("golden_alignments", DataParams{ goldenAlignments, 1u, 1u, T_IN });
    for (size_t i = 0; i < 2 * params.decoderLstmUnits.size(); ++i)
    {
        work.add<DataLayer>("golden_rnn_state[" + to_string(i) + "]", DataParams{ goldenRnnState[i], 1u, 1u, params.decoderLstmUnits[i / 2] });
    }
    work.add<DataLayer>("initial_rnn_state", DataParams{ initialRnnState, 1u, 1u, params.decoderLstmUnits[0] });

    work.add<SlicerLayer>("mel_extractor", SlicingParams{ "reduced_mel_targets", { mels }, "height" });

    for (size_t time = 0; time < decoder_iterations; ++time)
    {
        Names stepOutputs = {
            /**
             *   - decoder_cell_output:
             *    - frames_prediction
             *    - stop_token
             *  - decoder_state:
             *    - attention
             *    - alignments
             *    - rnn cell state
             *      - h
             *      - c
             */
            cellName / "frames_prediction",
            cellName / "stop_token",
            cellName / "state" / "attention",
            cellName / "state" / "alignments",
        };

        for (size_t i = 0; i < params.decoderLstmUnits.size(); ++i)
        {
            stepOutputs.emplace_back(cellName / "state" / "rnn_cell_state_h[" + to_string(i) + "]");
            stepOutputs.emplace_back(cellName / "state" / "rnn_cell_state_c[" + to_string(i) + "]");
        }

        TacotronParams stepParams = params;

        stepParams.getInputs() = stepInputs;
        stepParams.getOutputs() = stepOutputs;

        if (time == 0)
        {
            tacotron::AddDecoderCell(&work, cellName, { stepInputs, stepOutputs }, params);
        }
        else
        {
            tacotron::AddDecoderCell(&work, cellName, { stepInputs, stepOutputs, mainCellName }, params);
        }

        stepInputs = {
            /**
             * inputs:
             *   - encoder_output
             *   - input_lengths: int32 Tensor with shape [N] where N is batch size and values are the lengths
             *    of each sequence in inputs.
             *   - [previous_step_output]: encoder output or output from previous decoding step
             *   - [state]: optional previous state tensors (will be initialized if not provided):
             *     - attention
             *     - alignments
             *     - rnn cell state
             *       - h
             *       - c
             */
            "embedded_encoder_output",
            "input_lengths",
            mels[time], // teacher_forcing = 1
        };
        std::copy(stepOutputs.begin() + 2, stepOutputs.end(), back_inserter(stepInputs));

        cellName = Name("T") / "decoder" / ("_cell[" + to_string(time + 1) + "]");
    }

    TENSORS_CREATE(BATCH_SIZE);
    auto& memory_manager = work.getMemoryManager();
    EXPECT_TRUE(loadTFData(tools::getTestAssetsDir() / "Tacotron" / "Alice07" / "mel_targets_1.data", memory_manager["mel_targets"]));

    const auto& targets = memory_manager["mel_targets"];
    auto& reducedTargets = memory_manager["reduced_mel_targets"];

    auto targets3D = targets.reshape(yato::dims(targets.getBatchSize(), targets.getHeight(), targets.getWidth()));
    auto reducedTargets3D = reducedTargets.reshape(yato::dims(targets.getBatchSize(), reducedTargets.getHeight(), targets.getWidth()));

    for (size_t q = 0; q < targets.getBatchSize(); ++q)
    {
        for (size_t i = 0; i < reducedTargets.getHeight(); ++i)
        {
            auto src = targets3D[q][(params.outputsPerStep - 1) + i * params.outputsPerStep];
            auto tgt = reducedTargets3D[q][i];
            std::copy(src.begin(), src.end(), tgt.begin());
        }
    }

    memory_manager["initial_alignments"][0] = 1_dt;

    EXPECT_TRUE(loadTFData(tools::getTestAssetsDir() / "Tacotron" / "Alice07" / "decoder_output.data", memory_manager["golden_decoder_output"]));
    EXPECT_TRUE(loadTFData(tools::getTestAssetsDir() / "Tacotron" / "Alice07" / "stop_token_prediction.data", memory_manager["golden_stop_token"]));
    size_t loaded = loadTacotronParams((tools::getTestAssetsDir() / "Tacotron" / "pretrained" / "300000_Tacotron_model.").string(), memory_manager, "T", params, false, false);
    EXPECT_EQ(loaded, DECODER_CELL_PARAMS);

    // training data
    EXPECT_TRUE(loadTFData(tools::getTestAssetsDir() / "Tacotron" / "Alice07" / "input_lengths_1.data", memory_manager["input_lengths"]));
    EXPECT_TRUE(loadTFData(tools::getTestAssetsDir() / "Tacotron" / "Alice07" / "encoder_fin_outputs.data", memory_manager["embedded_encoder_output"]));

    EXPECT_TRUE(loadTFData(tools::getTestAssetsDir() / "Tacotron" / "Alice07" / "decoder_cell_test" / "state_attention.data", memory_manager, goldenAttentions));
    EXPECT_TRUE(loadTFData(tools::getTestAssetsDir() / "Tacotron" / "Alice07" / "decoder_cell_test" / "attention_values.data", memory_manager, { goldenAttentionValues }));
    EXPECT_TRUE(loadTFData(tools::getTestAssetsDir() / "Tacotron" / "Alice07" / "decoder_cell_test" / "state_alignments.data", memory_manager, goldenAlignments));

    for (size_t i = 0; i < params.decoderLstmUnits.size(); ++i)
    {
        EXPECT_TRUE(loadTFData(tools::getTestAssetsDir() / "Tacotron" / "Alice07" / "decoder_cell_test" / ("rnn" + to_string(i) + "_h.data"), memory_manager, goldenRnnState[i]));
        EXPECT_TRUE(loadTFData(
            tools::getTestAssetsDir() / "Tacotron" / "Alice07" / "decoder_cell_test" / ("rnn" + to_string(i) + "_c.data"), memory_manager, goldenRnnState[i + params.decoderLstmUnits.size()]));
    }

    auto& goldenf = memory_manager["golden_decoder_output"];
    auto& goldens = memory_manager["golden_stop_token"];

    auto callback = [&](BasicLayer* layer, const MemoryManager& memory_manager, NetworkParameters::CallbackPlace place) {
        if (place != NetworkParameters::CallbackPlace::After_Forward)
        {
            return;
        }

        if (!Common::endsWith(layer->getName(), "stop_projection_dense") && !Common::endsWith(layer->getName(), "frame_projection"))
        {
            return;
        }
        static size_t iter = 0;
        static dtype frDiff = 0_dt;
        static dtype frNorm = 0_dt;
        static dtype frGoldenNorm = 0_dt;
        size_t sz = params.numMels * params.outputsPerStep;

        size_t i = iter;
        auto cellName = i == 0 ? Name("T") / "decoder" / "_cell" : Name("T") / "decoder" / "_cell[" + to_string(i) + "]";

        if (Common::endsWith(layer->getName(), "frame_projection"))
        {
            const auto& fr = memory_manager[cellName / "frames_prediction"];
            Tensor gf(Tensor::dt_range(&*(goldenf.begin() + i * sz), &*(goldenf.begin() + (i + 1) * sz)));
            frDiff = TensorDiff(fr, gf);
            frNorm = TensorNorm(fr);
            frGoldenNorm = TensorNorm(gf);
            return;
        }

        ++iter;

        const auto& st = memory_manager[cellName / "stop_token"];
        Tensor gs(Tensor::dt_range(&*(goldens.begin() + i * params.outputsPerStep), &*(goldens.begin() + (i + 1) * params.outputsPerStep)));
        auto stDiff = TensorDiff(st, gs);
        cout << "Iter " << i << ". frame: " << frDiff << "(" << frNorm << ", " << frGoldenNorm << ")"
             << ", stop: " << TensorDiff(st, gs) << "(" << TensorNorm(st) << ", " << TensorNorm(gs) << ")" << endl;
        EXPECT_LE(frDiff, FRAME_PREDICTION_EPS);
        EXPECT_LE(stDiff, STOP_TOKEN_EPS);
    };

    work.getNetworkParameters().mCallback = callback;
    work.forwardPassTraining();
}

TEST(TestTacotron, Alice07DecoderRnnUnit)
{
    PROFILE_TEST

    Workflow work;

    TacotronParams params({}, {}, {});
    params.dropoutRate = 0.f;
    params.postnetDropoutRate = 0.f;
    params.zoneoutRate = 0.f;
    params.attentionSigmoidNoise = 0.f;

    params.maxMelFrames = 264; // @todo: read from file

    constexpr size_t BATCH_SIZE = 1;
    const size_t DECODER_RNN_TRAINABLE_PARAMS = params.decoderLstmUnits.size() * 2;
    constexpr auto input_size = 896U;
    constexpr auto hidden_size = 512U;
    constexpr auto EPS = 1e-5_dt;

    size_t decoder_iterations = params.maxMelFrames / params.outputsPerStep;

    vector<Names> goldenRnnState(params.decoderLstmUnits.size() * 2);
    Names goldenInput;
    Names goldenOutput;

    for (size_t time = 0; time < decoder_iterations; ++time)
    {
        auto suffix = "[" + to_string(time) + "]";
        goldenInput.push_back("golden_input" + suffix);
        goldenOutput.push_back("golden_output" + suffix);

        for (size_t i = 0; i < params.decoderLstmUnits.size(); ++i)
        {
            goldenRnnState[2 * i].emplace_back("golden_rnn_cell_state_h[" + to_string(i) + "]" + suffix);
            goldenRnnState[2 * i + 1].emplace_back("golden_rnn_cell_state_c[" + to_string(i) + "]" + suffix);
        }
    }

    work.add<DataLayer>("golden_input", DataParams{ goldenInput, 1u, 1u, input_size });
    work.add<DataLayer>("golden_output", DataParams{ goldenOutput, 1u, 1u, hidden_size });
    for (size_t i = 0; i < 2 * params.decoderLstmUnits.size(); ++i)
    {
        work.add<DataLayer>("golden_rnn[" + to_string(i) + "]", DataParams{ goldenRnnState[i], 1u, 1u, params.decoderLstmUnits[0] });
    }

    work.add<DataLayer>("input", DataParams{ { "input", "targets_lengths" }, 1u, 1u, input_size });

    Names rnn_cell_state;
    Names next_cell_state;

    for (size_t i = 0; i < params.decoderLstmUnits.size(); ++i)
    {
        rnn_cell_state.push_back("hidden" + std::to_string(i));
        rnn_cell_state.push_back("cell" + std::to_string(i));
        next_cell_state.push_back("next_hidden" + std::to_string(i));
        next_cell_state.push_back("next_cell" + std::to_string(i));
    }
    Names inputs = rnn_cell_state;
    inputs.insert(inputs.begin(), "input");
    work.add<TensorLayer>("rnn_state", TensorParams{ rnn_cell_state, WShape{ BS(), 1u, 1u, hidden_size } });

    tacotron::AddDecoderRNN(&work, Name("T") / "decoder" / "_cell" / "decoder_LSTM", { inputs, { "output" } }, next_cell_state, params);

    TENSORS_CREATE(BATCH_SIZE);
    MemoryManager& memory_manager = work.getMemoryManager();
    size_t loaded = loadTacotronParams((tools::getTestAssetsDir() / "Tacotron" / "pretrained" / "300000_Tacotron_model.").string(), memory_manager, "T", params, false);
    EXPECT_EQ(loaded, DECODER_RNN_TRAINABLE_PARAMS);
    EXPECT_EQ(work.getTrainableParameterNames().size(), DECODER_RNN_TRAINABLE_PARAMS);
    // training data
    bool ok = loadTFData(tools::getTestAssetsDir() / "Tacotron" / "Alice07" / "decoder_cell_test" / "lstm_input.data", memory_manager, goldenInput);
    EXPECT_TRUE(ok);
    ok = loadTFData(tools::getTestAssetsDir() / "Tacotron" / "Alice07" / "decoder_cell_test" / "lstm_output.data", memory_manager, goldenOutput);
    EXPECT_TRUE(ok);

    for (size_t i = 0; i < params.decoderLstmUnits.size(); ++i)
    {
        ok = loadTFData(tools::getTestAssetsDir() / "Tacotron" / "Alice07" / "decoder_cell_test" / ("rnn" + to_string(i) + "_h.data"), memory_manager, goldenRnnState[2 * i]);
        EXPECT_TRUE(ok);
        ok = loadTFData(tools::getTestAssetsDir() / "Tacotron" / "Alice07" / "decoder_cell_test" / ("rnn" + to_string(i) + "_c.data"), memory_manager, goldenRnnState[2 * i + 1]);
        EXPECT_TRUE(ok);
    }

    memory_manager.createTensor("output_export", memory_manager["output"].getShape(), 0_dt);
    for (size_t i = 0; i < 2 * params.decoderLstmUnits.size(); ++i)
    {
        memory_manager.createTensor(next_cell_state[i] + "_export", memory_manager[next_cell_state[i]].getShape(), 0_dt);
    }

    auto check = [&](BasicLayer*, const MemoryManager& memory_manager, NetworkParameters::CallbackPlace place) {
        if (place != NetworkParameters::CallbackPlace::After_Forward)
        {
            return;
        }
        if (!memory_manager["output"].empty())
        {
            const_cast<Tensor&>(memory_manager["output_export"]) = TORANGE(memory_manager["output"]);
        }
        for (size_t i = 0; i < 2 * params.decoderLstmUnits.size(); ++i)
        {
            if (!memory_manager[next_cell_state[i]].empty())
            {
                const_cast<Tensor&>(memory_manager[next_cell_state[i] + "_export"]) = TORANGE(memory_manager[next_cell_state[i]]);
            }
        }
    };

    work.getNetworkParameters().mCallback = check;

    for (size_t time = 0; time < decoder_iterations; ++time)
    {
        // set input and state
        memory_manager["input"] = TORANGE(memory_manager[goldenInput[time]]);
        for (size_t i = 0; i < 2 * params.decoderLstmUnits.size(); ++i)
        {
            memory_manager[rnn_cell_state[i]] = TORANGE(memory_manager[goldenRnnState[i][time]]);
        }
        ASSERT_NO_THROW(work.forwardPassTraining());
        auto diff = TensorDiff(memory_manager["output_export"], memory_manager[goldenOutput[time]]);
        cout << "Epoch " << time << ". Output diff = " << diff << endl;
        EXPECT_TRUE(diff < EPS);
        if (time < decoder_iterations - 1)
        {
            for (size_t i = 0; i < 2 * params.decoderLstmUnits.size(); ++i)
            {
                cout << "  " << next_cell_state[i] << " " << TensorDiff(memory_manager[next_cell_state[i] + "_export"], memory_manager[goldenRnnState[i][time + 1]]) << endl;
            }
        }
        ASSERT_NO_THROW(work.backwardPassTraining());
    }
}

TEST(TestTacotron, SequenceLossVanillaUnit)
{
    constexpr size_t numDecoderSymbols = 4u;
    constexpr size_t sequenceLength = 3u;
    constexpr size_t batchSize = 2u;
    constexpr dtype EPS = 1e-5_dt;

    // Inputs
    const raul::Tensor logits{ 0.81269646_dt, 0.07857466_dt, 0.8916855_dt,  0.16925514_dt, 0.06311357_dt, 0.54531074_dt, 0.5037316_dt,  0.9248222_dt,
                               0.66955376_dt, 0.9281193_dt,  0.12239242_dt, 0.8532245_dt,  0.90477383_dt, 0.7104306_dt,  0.40681756_dt, 0.5755513_dt,
                               0.8547678_dt,  0.59606934_dt, 0.77619946_dt, 0.97301054_dt, 0.06244731_dt, 0.33562684_dt, 0.22166848_dt, 0.32035887_dt };
    const raul::Tensor targets{ 1.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 2.0_dt, 0.0_dt, 1.0_dt, 1.0_dt,
                                0.0_dt, 0.0_dt, 1.0_dt, 2.0_dt, 2.0_dt, 2.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 2.0_dt, 2.0_dt, 2.0_dt };
    const raul::Tensor weights{ 0.67922187_dt, 0.8561305_dt, 0.33027768_dt, 0.06890488_dt, 0.7387967_dt, 0.4994563_dt };

    // Possible reductions
    const size_t numOfModes = 7;
    bool reductions[numOfModes][4]{ { false, false, false, false }, { false, false, true, false }, { false, false, false, true }, { false, false, true, true },
                                    { true, false, false, false },  { false, true, false, false }, { true, true, false, false } };

    // Outputs
    const raul::WShape expectedShapes[]{ { batchSize, 1u, sequenceLength, 1u }, { batchSize, 1u, 1u, 1u }, { 1u, 1u, sequenceLength, 1u }, { 1u, 1u, 1u, 1u }, { batchSize, 1u, 1u, 1u },
                                         { 1u, 1u, sequenceLength, 1u },        { 1u, 1u, 1u, 1u } };
    const raul::Tensor realOutputs[]{ { 1.9685409_dt, 0._dt, 1.9761338_dt, 0.31690788_dt, 4.3444977_dt, 3.999257_dt },
                                      { 1.3148916_dt, 2.8868876_dt },
                                      { 1.1427244_dt, 2.1722488_dt, 2.9876955_dt },
                                      { 2.1008894_dt },
                                      { 2.1143928_dt, 6.6255674_dt },
                                      { 3.0548952_dt, 2.7239473_dt, 7.2015743_dt },
                                      { 3.9729526_dt } };
    const raul::Tensor realLogitsGrads[]{
        { -0.4592381_dt, 0.10557608_dt, 0.23806475_dt, -0.56362456_dt, 0.13089682_dt, 0.21200432_dt, 0.2033701_dt,  0.30985922_dt, -0.5795112_dt, 0.10495793_dt, -0.28338635_dt, -0.2328934_dt,
          0.02187019_dt, 0.01800738_dt, -0.0556128_dt, -0.12207452_dt, -1.2842929_dt, -1.3283546_dt, 0.17869447_dt, 0.21756303_dt, 0.10446783_dt, -0.8616276_dt, -0.8764139_dt,  -0.8637077_dt },
        { -0.15307938_dt, 0.03519203_dt, 0.07935491_dt, -0.18787487_dt, 0.04363228_dt,  0.07066811_dt,  0.06779004_dt, 0.10328642_dt, -0.19317041_dt, 0.03498598_dt, -0.09446212_dt, -0.07763114_dt,
          0.00729006_dt,  0.00600246_dt, -0.0185376_dt, -0.04069151_dt, -0.42809764_dt, -0.44278488_dt, 0.05956483_dt, 0.07252101_dt, 0.03482261_dt,  -0.2872092_dt, -0.29213798_dt, -0.2879026_dt },
        { -0.22961906_dt, 0.05278804_dt, 0.11903238_dt, -0.28181228_dt, 0.06544841_dt,  0.10600216_dt, 0.10168505_dt, 0.15492961_dt, -0.2897556_dt, 0.05247897_dt, -0.14169317_dt, -0.1164467_dt,
          0.0109351_dt,   0.00900369_dt, -0.0278064_dt, -0.06103726_dt, -0.64214647_dt, -0.6641773_dt, 0.08934724_dt, 0.10878152_dt, 0.05223392_dt, -0.4308138_dt, -0.43820694_dt, -0.43185386_dt },
        { -0.07653969_dt, 0.01759601_dt, 0.03967746_dt, -0.09393743_dt, 0.02181614_dt,  0.03533405_dt,  0.03389502_dt, 0.05164321_dt, -0.09658521_dt, 0.01749299_dt, -0.04723106_dt, -0.03881557_dt,
          0.00364503_dt,  0.00300123_dt, -0.0092688_dt, -0.02034575_dt, -0.21404882_dt, -0.22139244_dt, 0.02978241_dt, 0.0362605_dt,  0.0174113_dt,   -0.1436046_dt, -0.14606899_dt, -0.1439513_dt },
        { -0.24615711_dt, 0.05659004_dt, 0.12760556_dt,  -0.3021095_dt,  0.07016226_dt, 0.11363684_dt, 0.1090088_dt, 0.16608824_dt, -0.31062493_dt, 0.0562587_dt,  -0.15189847_dt, -0.12483364_dt,
          0.0167311_dt,   0.01377598_dt, -0.04254482_dt, -0.09338927_dt, -0.9825079_dt, -1.0162159_dt, 0.1367046_dt, 0.16643976_dt, 0.07991983_dt,  -0.6591611_dt, -0.67047286_dt, -0.6607524_dt },
        { -0.6138507_dt, 0.14112058_dt, 0.31821448_dt,  -0.7533812_dt,  0.08207072_dt, 0.13292414_dt,  0.12751059_dt, 0.19427797_dt, -0.6984301_dt, 0.12649588_dt, -0.3415388_dt, -0.2806844_dt,
          0.02923327_dt, 0.02406995_dt, -0.07433607_dt, -0.16317359_dt, -0.8052361_dt, -0.83286226_dt, 0.11203927_dt, 0.13640939_dt, 0.12590522_dt, -1.0384383_dt, -1.0562589_dt, -1.0409454_dt },
        { -0.14474277_dt, 0.03327549_dt, 0.0750333_dt,   -0.17764331_dt, 0.04125609_dt, 0.06681956_dt,  0.06409823_dt, 0.09766149_dt, -0.18265046_dt, 0.03308066_dt,  -0.08931777_dt, -0.07340339_dt,
          0.00689305_dt,  0.00567557_dt, -0.01752805_dt, -0.03847547_dt, -0.4047837_dt, -0.41867107_dt, 0.05632096_dt, 0.06857156_dt, 0.03292619_dt,  -0.27156797_dt, -0.27622834_dt, -0.2722236_dt }
    };

    TensorChecker checker({ { "loss", "realLoss" } }, { { raul::Name("logits").grad(), "realLogitsGrad" } }, EPS);

    // Topology
    for (size_t i = 0; i < numOfModes; ++i)
    {
        MANAGERS_DEFINE
        NETWORK_PARAMS_DEFINE(networkParameters);
        networkParameters.mCallback = checker;

        work.add<DataLayer>("logits_and_targets", DataParams{ { "logits", "targets", "realLogitsGrad" }, 1u, sequenceLength, numDecoderSymbols });
        work.add<DataLayer>("weights", DataParams{ { "weights" }, 1u, sequenceLength, 1u });
        work.add<TensorLayer>("realOutput", TensorParams{ { "realLoss" }, expectedShapes[i] });

        raul::tacotron::AddSequenceLoss(&work, "SeqLoss", raul::BasicParams{ { "logits", "targets", "weights" }, { "loss" } }, reductions[i][0], reductions[i][1], reductions[i][2], reductions[i][3]);

        TENSORS_CREATE(batchSize);
        memory_manager["logits"] = TORANGE(logits);
        memory_manager["targets"] = TORANGE(targets);
        memory_manager["weights"] = TORANGE(weights);
        memory_manager["realLoss"] = TORANGE(realOutputs[i]);
        memory_manager["realLogitsGrad"] = TORANGE(realLogitsGrads[i]);
        memory_manager[raul::Name("loss").grad()] = 1.0_dt;

        ASSERT_NO_THROW(work.forwardPassTraining());
        ASSERT_NO_THROW(work.backwardPassTraining());
    }
}

}
