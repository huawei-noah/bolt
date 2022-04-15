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
#include <training/layers/basic/TensorLayer.h>
#include <training/layers/basic/trainable/Convolution1DLayer.h>
#include <training/layers/basic/trainable/LinearLayer.h>
#include <training/layers/composite/BahdanauMonotonicAttentionLayer.h>
#include <training/layers/composite/DynamicConvolutionAttentionLayer.h>
#include <training/layers/composite/TacotronModel.h>
#include <training/layers/composite/rnn/BidirectionalLSTMFunc.h>
#include <training/layers/composite/tacotron/MaskedCrossEntropy.h>
#include <training/layers/composite/tacotron/TacotronDecoderRNN.h>
#include <training/layers/composite/tacotron/TacotronLoss.h>
#include <training/postprocessing/GradientClipping.h>

#include <training/api/API.h>

#include "TacotronTestTools.h"
#include "TacotronTrainingParams.h"

namespace UT
{
using namespace raul;
using namespace UT::tools;
using namespace std;

TEST(TestTacotronSpB, TacotronParamsCountUnit)
{
    PROFILE_TEST
    {
        TacotronParams p{ {}, {}, {} };
        pair<size_t, size_t> goldenCnt = { 43, 53 };
        auto cnt = tacotronParamsCount(p);
        EXPECT_EQ(goldenCnt.first, cnt.first);
        EXPECT_EQ(goldenCnt.second, cnt.second);
    }
    {
        TacotronParams p{ {}, {}, {} };
        p.useAttentionRnn = true;
        p.attentionType = "DynamicConvolutional";

        pair<size_t, size_t> goldenCnt = { 49, 59 };
        auto cnt = tacotronParamsCount(p);
        EXPECT_EQ(goldenCnt.first, cnt.first);
        EXPECT_EQ(goldenCnt.second, cnt.second);
    }
}

TEST(TestTacotronSpB, DecoderRNNUnit)
{
    PROFILE_TEST

    // Test parameters
    constexpr auto eps_rel = 1e-4_dt;
    constexpr auto eps_abs = 1e-6_dt;
    constexpr auto input_size = 1280U;
    constexpr auto hidden_size = 512U;
    constexpr auto batch_size = 1U;

    constexpr size_t ITERS = 3;

    Workflow work;

    Names rnn_cell_state[ITERS];
    Names next_cell_state[ITERS];

    TacotronParams params({}, {}, {});
    params.dropoutRate = 0.f;
    params.postnetDropoutRate = 0.f;
    params.zoneoutRate = 0.f;
    params.useResidualRnn = false;

    vector<Names> hidden_golden(params.decoderLstmUnits.size());
    vector<Names> cell_golden(params.decoderLstmUnits.size());

    const size_t DECODER_RNN_TRAINABLE_PARAMS = params.decoderLstmUnits.size() * 2;

    auto testPath = tools::getTestAssetsDir() / "Tacotron_SpB";
    Names inputs;
    Names outputs;
    Names golden_outputs;
    for (size_t iter = 0; iter < ITERS; ++iter)
    {
        inputs.push_back("input_" + to_string(iter));
        outputs.push_back("outputs_" + to_string(iter));
        golden_outputs.push_back("golden_outputs_" + to_string(iter));
        for (size_t i = 0; i < params.decoderLstmUnits.size(); ++i)
        {
            auto suffix = "_" + to_string(iter) + "_" + to_string(i);
            rnn_cell_state[iter].push_back("hidden_golden" + suffix);
            rnn_cell_state[iter].push_back("cell_golden" + suffix);
            next_cell_state[iter].push_back("next_hidden" + suffix);
            next_cell_state[iter].push_back("next_cell" + suffix);

            hidden_golden[i].push_back("hidden_golden" + suffix);
            cell_golden[i].push_back("cell_golden" + suffix);

            work.add<DataLayer>("data" + suffix, DataParams{ { "hidden_golden" + suffix, "cell_golden" + suffix }, 1u, 1u, hidden_size });
        }
    }
    work.add<DataLayer>("input", DataParams{ inputs, 1u, 1u, input_size });
    work.add<DataLayer>("golden_output", DataParams{ golden_outputs, 1u, 1u, hidden_size });

    for (size_t iter = 0; iter < ITERS; ++iter)
    {
        Names decoder_inputs = rnn_cell_state[iter];
        decoder_inputs.insert(decoder_inputs.begin(), inputs[iter]);
        auto name = Name("T") / "decoder" / "_cell" / "decoder_LSTM";
        tacotron::AddDecoderRNN(&work, iter == 0 ? name : name / to_string(iter), { decoder_inputs, { outputs[iter] }, iter == 0 ? Name() : name }, next_cell_state[iter], params);
    }

    TENSORS_CREATE(batch_size);
    MemoryManager& memory_manager = work.getMemoryManager();

    // Data loading
    EXPECT_TRUE(loadTFData(testPath / "non_residual_decoder" / "decoder_rnn_input.data", memory_manager, inputs));
    [[maybe_unused]] auto& tt = memory_manager[inputs[0]];
    EXPECT_TRUE(loadTFData(testPath / "non_residual_decoder" / "decoder_rnn_output.data", memory_manager, golden_outputs));
    for (size_t i = 0; i < params.decoderLstmUnits.size(); ++i)
    {
        EXPECT_TRUE(loadTFData((testPath / "non_residual_decoder" / "decoder_rnn_input_h").string() + to_string(i) + ".data", memory_manager, hidden_golden[i]));
        EXPECT_TRUE(loadTFData((testPath / "non_residual_decoder" / "decoder_rnn_input_c").string() + to_string(i) + ".data", memory_manager, cell_golden[i]));
    }

    size_t loaded = loadTacotronParams((testPath / "weights" / "200000_Tacotron_model.").string(), memory_manager, "T", params, false);
    EXPECT_EQ(loaded, DECODER_RNN_TRAINABLE_PARAMS);
    EXPECT_EQ(work.getTrainableParameterNames().size(), DECODER_RNN_TRAINABLE_PARAMS);

    vector<pair<Name, Name>> checked;
    for (size_t iter = 1; iter < 3; ++iter)
    {
        checked.emplace_back(make_pair(outputs[iter], golden_outputs[iter]));
        if (iter > 0)
        {
            for (size_t j = 0; j < params.decoderLstmUnits.size(); ++j)
            {
                auto suffix1 = "_" + to_string(iter - 1) + "_" + to_string(j);
                auto suffix2 = "_" + to_string(iter) + "_" + to_string(j);
                checked.emplace_back(make_pair(Name("next_hidden" + suffix1), Name("hidden_golden" + suffix2)));
                checked.emplace_back(make_pair(Name("next_cell" + suffix1), Name("cell_golden" + suffix2)));
            }
        }
    }
    callbacks::TensorChecker checker(checked, eps_abs, eps_rel);
    work.getNetworkParameters().mCallback = checker;
    work.forwardPassTraining();
}

TEST(TestTacotronSpB, TacotronLossUnit)
{
    PROFILE_TEST
    using namespace tacotron;

    Workflow work;

    auto testPath = tools::getTestAssetsDir() / "Tacotron_SpB";
    constexpr size_t BATCH_SIZE = 2;
    constexpr dtype EPS = 1e-4_dt;
    constexpr dtype realBeforeLoss = 0.565931261_dt;
    constexpr dtype realAfterLoss = 0.348040164_dt;
    constexpr dtype realStopTokenLoss = 0.95668_dt;
    constexpr dtype realLoss = realBeforeLoss + realAfterLoss + realStopTokenLoss;

    TacotronParams params({}, {}, {});
    params.maskUseSquaredError = true;
    params.maxMelFrames = 984; // @todo: read from file
    params.maskedSigmoidEpsilon = 1e-5f;

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

    EXPECT_TRUE(loadTFData(testPath / "unit" / "stop_token_targets.data", memory_manager["stop_token_targets"]));
    EXPECT_TRUE(loadTFData(testPath / "unit" / "stop_token_prediction.data", memory_manager["stop_token_prediction"]));
    EXPECT_TRUE(loadTFData(testPath / "unit" / "targets_lengths.data", memory_manager["targets_lengths"]));
    EXPECT_TRUE(loadTFData(testPath / "unit" / "mel_targets.data", memory_manager["mel_targets"]));
    EXPECT_TRUE(loadTFData(testPath / "unit" / "mel_outputs.data", memory_manager["mel_outputs"]));
    EXPECT_TRUE(loadTFData(testPath / "unit" / "decoder_output.data", memory_manager["decoder_output"]));

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
    ASSERT_EQ(memory_manager["before_loss"].size(), 1u);
    ASSERT_EQ(memory_manager["after_loss"].size(), 1u);
    ASSERT_EQ(memory_manager["stop_token_loss"].size(), 1u);

    auto bloss = memory_manager["before_loss"][0];
    auto aloss = memory_manager["after_loss"][0];
    auto sloss = memory_manager["stop_token_loss"][0];

    EXPECT_NEAR(loss[0], realLoss, EPS);
    EXPECT_NEAR(bloss, realBeforeLoss, EPS);
    EXPECT_NEAR(aloss, realAfterLoss, EPS);
    EXPECT_NEAR(sloss, realStopTokenLoss, EPS);

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

TEST(TestTacotronSpB, StopTokenLossUnit)
{
    PROFILE_TEST
    using namespace tacotron;

    Workflow work;

    auto testPath = tools::getTestAssetsDir() / "Tacotron_SpB";
    constexpr size_t BATCH_SIZE = 1;
    constexpr dtype EPS = 1e-4_dt;
    constexpr dtype realStopTokenLoss = 11.232508_dt;

    TacotronParams params({}, {}, {});
    params.maxMelFrames = 255; // @todo: read from file
    params.maskedSigmoidEpsilon = 1e-5f;

    work.add<DataLayer>("targets_lengths", DataParams{ { "targets_lengths" }, 1, 1, 1 });
    work.add<DataLayer>("stop_token_targets", DataParams{ { "stop_token_targets" }, 1, params.maxMelFrames, 1 });
    work.add<DataLayer>("stop_token_prediction", DataParams{ { "stop_token_prediction" }, 1, params.maxMelFrames, 1 });

    AddMaskedCrossEntropy(
        &work, "stop_token_loss", { { "stop_token_prediction", "stop_token_targets", "targets_lengths" }, { "stop_token_loss" } }, params.outputsPerStep, params.maskedSigmoidEpsilon, true);

    TENSORS_CREATE(BATCH_SIZE);
    MemoryManager& memory_manager = work.getMemoryManager();

    EXPECT_TRUE(loadTFData(testPath / "unit_8_0" / "stop_token_targets.data", memory_manager["stop_token_targets"]));
    EXPECT_TRUE(loadTFData(testPath / "unit_8_0" / "stop_token_prediction.data", memory_manager["stop_token_prediction"]));
    EXPECT_TRUE(loadTFData(testPath / "unit_8_0" / "targets_lengths.data", memory_manager["targets_lengths"]));

    ASSERT_NO_THROW(work.forwardPassTraining());

    ASSERT_EQ(memory_manager["stop_token_loss"].size(), 1u);

    auto sloss = memory_manager["stop_token_loss"][0];

    EXPECT_NEAR(sloss, realStopTokenLoss, EPS);

    ASSERT_NO_THROW(work.backwardPassTraining());
}

TEST(TestTacotronSpB, Adam2DeterministicUnit)
{
    PROFILE_TEST

    using namespace optimizers;

    bool usePool = false;
    Workflow work(CompressionMode::NONE, CalculationMode::DETERMINISTIC, usePool ? AllocationMode::POOL : AllocationMode::STANDARD);

    constexpr size_t BATCH_SIZE = 2;

    constexpr dtype FRAME_PREDICTION_EPS = 1e-5_dt;
    constexpr dtype STOP_TOKEN_EPS = 1e-4_dt;

    auto testPath = tools::getTestAssetsDir() / "Tacotron_SpB";

    TacotronParams params({ "encoder_output", "input_lengths", "mel_targets" }, { "decoder_output", "mel_outputs", "stop_token_prediction" }, {});
    params.dropoutRate = 0.f;
    params.postnetDropoutRate = 0.f;
    params.zoneoutRate = 0.f;
    params.attentionSigmoidNoise = 0.f;
    params.useAttentionRnn = true;
    params.useResidualRnn = true;
    params.attentionType = "DynamicConvolutional";

    params.postnetChannels = 512;
    params.maskUseSquaredError = true;

    params.maxMelFrames = 984; // @todo: read from file
    params.maskedSigmoidEpsilon = 1e-5f;

    size_t decoder_iterations = params.maxMelFrames / params.outputsPerStep;

    size_t T_IN = 197; // number of steps in the input time series

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
    ifstream f(testPath / "unit" / "loss.data");
    copy(istream_iterator<dtype>(f), istream_iterator<dtype>(), back_inserter(idealLosses));
    ASSERT_TRUE(idealLosses.size() >= 3);
    idealLosses.resize(3);

    auto timeStart = chrono::steady_clock::now();
    TENSORS_CREATE(BATCH_SIZE);
    auto totalTime = TODTYPE(chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - timeStart).count());
    cout << "Workflow preparation time " << totalTime << " ms for batch size " << BATCH_SIZE << endl;
    MemoryManager& memory_manager = work.getMemoryManager();
    EXPECT_TRUE(loadTFData(testPath / "unit" / "decoder_output.data", memory_manager["golden_decoder_output"]));
    EXPECT_TRUE(loadTFData(testPath / "unit" / "stop_token_prediction.data", memory_manager["golden_stop_token"]));

    auto [TACOTRON_TRAINABLE_PARAMS, TACOTRON_PARAMS] = tacotronParamsCount(params);

    auto idealGradNorms = loadNamedValues(testPath / "unit" / "grad_norm.data", "T", params);
    EXPECT_EQ(idealGradNorms.size(), TACOTRON_TRAINABLE_PARAMS);
    size_t loaded = loadTacotronParams((testPath / "weights" / "200000_Tacotron_model.").string(), memory_manager, "T", params, true, false);
    EXPECT_EQ(loaded, TACOTRON_PARAMS - 1); // -1 because speaker_embedding will be loaded separately and doesn't have Adam params
    postprocessing::GradientClipping clip;

    // targets
    EXPECT_TRUE(loadTFData(testPath / "unit" / "stop_token_targets.data", memory_manager["stop_token_targets"]));
    EXPECT_TRUE(loadTFData(testPath / "unit" / "targets_lengths.data", memory_manager["targets_lengths"]));
    EXPECT_TRUE(loadTFData(testPath / "unit" / "mel_targets.data", memory_manager["mel_targets"]));

    // training data
    EXPECT_TRUE(loadTFData(testPath / "unit" / "input_lengths.data", memory_manager["input_lengths"]));
    EXPECT_TRUE(loadTFData(testPath / "unit" / "encoder_output.data", memory_manager["encoder_output"]));

    // speaker embedding
    EXPECT_TRUE(loadTFData(testPath / "unit" / "speaker_embedding.data", memory_manager[Name("T") / "speaker_embedding"]));

    auto callback = [&](BasicLayer* layer, const MemoryManager& memory_manager, NetworkParameters::CallbackPlace place) {
        if (place != NetworkParameters::CallbackPlace::Before_Forward)
        {
            return;
        }
        if (!Common::startsWith(layer->getName(), Name("total_loss") / "before" / "loss" / "mul"))
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

            auto frDiff = TensorDiff(fr, gf);
            auto stDiff = TensorDiff(st, gs);

            cout << "Iter " << i << ". frame: " << frDiff << "(" << TensorNorm(fr) << ", " << TensorNorm(gf) << ")"
                 << ", stop: " << stDiff << "(" << TensorNorm(st) << ", " << TensorNorm(gs) << ")" << endl;

            EXPECT_LE(frDiff, FRAME_PREDICTION_EPS);
            EXPECT_LE(stDiff, STOP_TOKEN_EPS);
        }
    };

    work.getNetworkParameters().mCallback = callback;

    timeStart = chrono::steady_clock::now();
    work.forwardPassTraining();

    auto& lossValue = memory_manager["loss"];
    totalTime = TODTYPE(chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - timeStart).count());
    cout << "Epoch time " << totalTime << " ms for batch size " << BATCH_SIZE << endl;
    cout << "Initial loss = " << lossValue[0] << " (before = " << memory_manager["before_loss"][0] << ", after = " << memory_manager["after_loss"][0]
         << ", stop_token = " << memory_manager["stop_token_loss"][0] << ")" << endl;
    cout << "Ideal loss = " << (idealLosses[0] + idealLosses[1] + idealLosses[2]) << " (before = " << idealLosses[0] << ", after = " << idealLosses[1] << ", stop_token = " << idealLosses[2] << ")"
         << endl;

    work.backwardPassTraining();

    auto tParams = work.getTrainableParameters();
    // clip.processGradients(tParams, work.getNetworkParameters());
    cout << "Gradients." << endl;
    cout << "Global norm: " << clip.calcGlobalNorm(tParams, work.getNetworkParameters()) << endl;
    dtype totalGradNormDiff = 0;
    for (size_t j = 0; j < tParams.size(); ++j)
    {
        auto& [param, grad] = tParams[j];
        cout << param.getName() << ": " << TensorNorm(grad) << " (" << idealGradNorms[param.getName()] << ")" << endl;

        cout << "  diff: " << fabs(TensorNorm(grad) - idealGradNorms[param.getName()]) << endl;
        totalGradNormDiff += fabs(TensorNorm(grad) - idealGradNorms[param.getName()]);
    }
    cout << "Total Grad Norm Diff: " << totalGradNormDiff << endl;

    EXPECT_TRUE(totalGradNormDiff < 0.5_dt);
}

TEST(TestTacotronSpB, AliceMskModelUnit)
{
    PROFILE_TEST

    using namespace optimizers;

    bool usePool = false;
    Workflow work(CompressionMode::NONE, CalculationMode::DETERMINISTIC, usePool ? AllocationMode::POOL : AllocationMode::STANDARD);

    constexpr size_t BATCH_SIZE = 15;

    constexpr dtype FRAME_PREDICTION_EPS = 1e-5_dt;
    constexpr dtype STOP_TOKEN_EPS = 1e-4_dt;

    auto testPath = tools::getTestAssetsDir() / "Tacotron_Msk";

    TacotronParams params({ "encoder_output", "input_lengths", "mel_targets" }, { "decoder_output", "mel_outputs", "stop_token_prediction" }, {});
    params.dropoutRate = 0.f;
    params.postnetDropoutRate = 0.f;
    params.zoneoutRate = 0.f;
    params.attentionSigmoidNoise = 0.f;
    params.useAttentionRnn = true;
    params.useResidualRnn = true;
    params.attentionType = "DynamicConvolutional";

    params.postnetChannels = 512;
    params.decoderLstmUnits = { 512, 512 };
    params.maskUseSquaredError = true;

    params.attentionPriorBeta = 5.9f;

    params.maxMelFrames = 546; // @todo: read from file
    params.maskedSigmoidEpsilon = 1e-5f;
    params.withoutStopTokenLoss = true;

    size_t decoder_iterations = params.maxMelFrames / params.outputsPerStep;

    size_t T_IN = 91; // number of steps in the input time series

    work.add<DataLayer>("encoder_output", DataParams{ { "encoder_output" }, 1u, T_IN, params.embeddingDim });
    work.add<DataLayer>("input_lengths", DataParams{ { "input_lengths" }, 1u, 1u, 1u });
    work.add<DataLayer>("mel_targets", DataParams{ { "mel_targets" }, 1u, params.maxMelFrames, params.numMels });

    work.add<DataLayer>("golden_decoder_output", DataParams{ { "golden_decoder_output" }, 1u, params.maxMelFrames, params.numMels });
    work.add<DataLayer>("golden_stop_token", DataParams{ { "golden_stop_token" }, 1u, params.maxMelFrames, 1u });

    work.add<DataLayer>("targets_lengths", DataParams{ { "targets_lengths" }, 1u, 1u, 1u });
    work.add<DataLayer>("stop_token_targets", DataParams{ { "stop_token_targets" }, 1u, params.maxMelFrames, 1u });

    AddSingleSpeakerFslTacotronModel(&work, "T", params);
    tacotron::AddTacotronLoss(&work,
                              "total_loss",
                              { { "decoder_output", "mel_outputs", "stop_token_prediction", "mel_targets", "targets_lengths", "stop_token_targets" },
                                params.withoutStopTokenLoss ? Names{ "loss", "before_loss", "after_loss" } : Names{ "loss", "before_loss", "after_loss", "stop_token_loss" } },
                              params);

    vector<dtype> idealLosses;
    ifstream f(testPath / "unit" / "loss.data");
    copy(istream_iterator<dtype>(f), istream_iterator<dtype>(), back_inserter(idealLosses));
    ASSERT_TRUE(idealLosses.size() >= 3);
    idealLosses.resize(3);

    auto timeStart = chrono::steady_clock::now();
    TENSORS_CREATE(BATCH_SIZE);
    auto totalTime = TODTYPE(chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - timeStart).count());
    cout << "Workflow preparation time " << totalTime << " ms for batch size " << BATCH_SIZE << endl;
    MemoryManager& memory_manager = work.getMemoryManager();
    EXPECT_TRUE(loadTFData(testPath / "unit" / "decoder_output.data", memory_manager["golden_decoder_output"]));
    EXPECT_TRUE(loadTFData(testPath / "unit" / "stop_token_prediction.data", memory_manager["golden_stop_token"]));

    auto [TACOTRON_TRAINABLE_PARAMS, TACOTRON_PARAMS] = tacotronParamsCount(params);

    auto idealGradNorms = loadNamedValues(testPath / "unit" / "grad_norm.data", "T", params);
    EXPECT_EQ(idealGradNorms.size(), TACOTRON_TRAINABLE_PARAMS);
    size_t loaded = loadTacotronParams((testPath / "weights" / "184000_Tacotron_model.").string(), memory_manager, "T", params, true, false);
    EXPECT_EQ(loaded, TACOTRON_PARAMS - 1); // -3 because speaker_embedding will be loaded separately
    postprocessing::GradientClipping clip(1_dt);

    // targets
    EXPECT_TRUE(loadTFData(testPath / "stop_token_targets.data", memory_manager["stop_token_targets"]));
    EXPECT_TRUE(loadTFData(testPath / "targets_lengths.data", memory_manager["targets_lengths"]));
    EXPECT_TRUE(loadTFData(testPath / "mel_targets.data", memory_manager["mel_targets"]));

    // training data
    EXPECT_TRUE(loadTFData(testPath / "input_lengths.data", memory_manager["input_lengths"]));
    EXPECT_TRUE(loadTFData(testPath / "encoder_output.data", memory_manager["encoder_output"]));

    // speaker embedding
    EXPECT_TRUE(loadTFData(testPath / "speaker_embedding.data", memory_manager[Name("T") / "speaker_embedding"]));

    auto callback = [&](BasicLayer* layer, const MemoryManager& memory_manager, NetworkParameters::CallbackPlace place) {
        if (place != NetworkParameters::CallbackPlace::Before_Forward)
        {
            return;
        }
        if (!Common::startsWith(layer->getName(), Name("total_loss") / "before" / "loss" / "mul"))
        {
            return;
        }

        auto& goldenf = memory_manager["golden_decoder_output"];
        auto& goldens = memory_manager["golden_stop_token"];
        auto& calculatedf = memory_manager["decoder_output"];

        for (size_t i = 0; i < decoder_iterations; ++i)
        {
            size_t sz = params.numMels * params.outputsPerStep;
            Tensor fr(Tensor::dt_range(&*(calculatedf.begin() + i * sz), &*(calculatedf.begin() + (i + 1) * sz)));

            Tensor gf(Tensor::dt_range(&*(goldenf.begin() + i * sz), &*(goldenf.begin() + (i + 1) * sz)));

            auto frDiff = TensorDiff(fr, gf);

            cout << "Iter " << i << ". frame: " << frDiff << "(" << TensorNorm(fr) << ", " << TensorNorm(gf) << ")";
            if (!params.withoutStopTokenLoss)
            {
                auto& calculateds = memory_manager["stop_token_prediction"];
                Tensor st(Tensor::dt_range(&*(calculateds.begin() + i * params.outputsPerStep), &*(calculateds.begin() + (i + 1) * params.outputsPerStep)));
                Tensor gs(Tensor::dt_range(&*(goldens.begin() + i * params.outputsPerStep), &*(goldens.begin() + (i + 1) * params.outputsPerStep)));

                auto stDiff = TensorDiff(st, gs);
                cout << ", stop: " << stDiff << "(" << TensorNorm(st) << ", " << TensorNorm(gs) << ")";

                if (!params.withoutStopTokenLoss)
                {
                    EXPECT_LE(stDiff, STOP_TOKEN_EPS);
                }
            }
            cout << endl;

            EXPECT_LE(frDiff, FRAME_PREDICTION_EPS);
        }
    };

    work.getNetworkParameters().mCallback = callback;

    timeStart = chrono::steady_clock::now();
    work.forwardPassTraining();

    auto& lossValue = memory_manager["loss"];
    totalTime = TODTYPE(chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - timeStart).count());
    cout << "Epoch time " << totalTime << " ms for batch size " << BATCH_SIZE << endl;
    cout << "Initial loss = " << lossValue[0] << " (before = " << memory_manager["before_loss"][0] << ", after = " << memory_manager["after_loss"][0];
    if (!params.withoutStopTokenLoss)
    {
        cout << ", stop_token = " << memory_manager["stop_token_loss"][0];
    }
    cout << ")" << endl;
    cout << "Ideal loss = " << (idealLosses[0] + idealLosses[1] + (params.withoutStopTokenLoss ? 0_dt : idealLosses[2])) << " (before = " << idealLosses[0] << ", after = " << idealLosses[1];
    if (!params.withoutStopTokenLoss)
    {
        cout << ", stop_token = " << idealLosses[2];
    }
    cout << ")" << endl;

    work.backwardPassTraining();

    auto tParams = work.getTrainableParameters();
    cout << "Gradients." << endl;
    cout << "Global norm: " << clip.calcGlobalNorm(tParams, work.getNetworkParameters()) << endl;
    dtype totalGradNormDiff = 0;
    for (size_t j = 0; j < tParams.size(); ++j)
    {
        auto& [param, grad] = tParams[j];
        cout << param.getName() << ": " << TensorNorm(grad) << " (" << idealGradNorms[param.getName()] << ")" << endl;

        cout << "  diff: " << fabs(TensorNorm(grad) - idealGradNorms[param.getName()]) << endl;
        totalGradNormDiff += fabs(TensorNorm(grad) - idealGradNorms[param.getName()]);
    }
    cout << "Total Grad Norm Diff: " << totalGradNormDiff << endl;

    EXPECT_TRUE(totalGradNormDiff < 0.5_dt);
}

TEST(TestTacotronSpB, MultiSpeakerMskSMAModelUnit)
{
    PROFILE_TEST

    using namespace optimizers;

    bool usePool = false;
    Workflow work(CompressionMode::NONE, CalculationMode::DETERMINISTIC, usePool ? AllocationMode::POOL : AllocationMode::STANDARD);

    constexpr size_t BATCH_SIZE = 40;

    constexpr dtype FRAME_PREDICTION_EPS = 2e-5_dt;
    constexpr dtype STOP_TOKEN_EPS = 1e-4_dt;

    auto testPath = tools::getTestAssetsDir() / "Tacotron_Msk_M4";

    TacotronParams params({ "encoder_output", "speaker_embedding", "input_lengths", "mel_targets" }, { "decoder_output", "mel_outputs", "stop_token_prediction" }, {});
    params.dropoutRate = 0.f;           // 0.5 for training
    params.postnetDropoutRate = 0.f;    // 0.1 for training
    params.zoneoutRate = 0.f;           // 0.1 for training
    params.attentionSigmoidNoise = 0.f; // 1.0 for training
    params.attentionScoreBiasInit = -1.5;
    params.useAttentionRnn = false;
    params.useResidualRnn = false;
    params.trainableSpeakerEmbedding = false;
    params.attentionType = "StepwiseMonotonic";

    params.postnetChannels = 512;
    params.decoderLstmUnits = { 512, 512, 512 };
    params.maskUseSquaredError = true;

    params.maxMelFrames = 984; // @todo: read from file
    params.maskedSigmoidEpsilon = 5e-7f;
    params.withoutStopTokenLoss = true; // false for training

    size_t decoder_iterations = params.maxMelFrames / params.outputsPerStep;

    size_t T_IN = 197; // number of steps in the input time series

    work.add<DataLayer>("encoder_output", DataParams{ { "encoder_output" }, 1u, T_IN, params.embeddingDim });
    work.add<DataLayer>("speaker_embedding", DataParams{ { "speaker_embedding" }, 1u, 1u, params.speakerEmbeddingSize });
    work.add<DataLayer>("input_lengths", DataParams{ { "input_lengths" }, 1u, 1u, 1u });
    work.add<DataLayer>("mel_targets", DataParams{ { "mel_targets" }, 1u, params.maxMelFrames, params.numMels });

    work.add<DataLayer>("golden_decoder_output", DataParams{ { "golden_decoder_output" }, 1u, params.maxMelFrames, params.numMels });
    work.add<DataLayer>("golden_stop_token", DataParams{ { "golden_stop_token" }, 1u, params.maxMelFrames, 1u });

    work.add<DataLayer>("targets_lengths", DataParams{ { "targets_lengths" }, 1u, 1u, 1u });
    work.add<DataLayer>("stop_token_targets", DataParams{ { "stop_token_targets" }, 1u, params.maxMelFrames, 1u });

    AddSingleSpeakerFslTacotronModel(&work, "T", params);
    tacotron::AddTacotronLoss(&work,
                              "total_loss",
                              { { "decoder_output", "mel_outputs", "stop_token_prediction", "mel_targets", "targets_lengths", "stop_token_targets" },
                                params.withoutStopTokenLoss ? Names{ "loss", "before_loss", "after_loss" } : Names{ "loss", "before_loss", "after_loss", "stop_token_loss" } },
                              params);

    vector<dtype> idealLosses;
    ifstream f(testPath / "unit" / "loss.data");
    copy(istream_iterator<dtype>(f), istream_iterator<dtype>(), back_inserter(idealLosses));
    ASSERT_TRUE(idealLosses.size() >= 3);
    idealLosses.resize(3);

    auto timeStart = chrono::steady_clock::now();
    TENSORS_CREATE(BATCH_SIZE);
    auto totalTime = TODTYPE(chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - timeStart).count());
    cout << "Workflow preparation time " << totalTime << " ms for batch size " << BATCH_SIZE << endl;
    MemoryManager& memory_manager = work.getMemoryManager();
    EXPECT_TRUE(loadTFData(testPath / "unit" / "decoder_output.data", memory_manager["golden_decoder_output"]));
    EXPECT_TRUE(loadTFData(testPath / "unit" / "stop_token_prediction.data", memory_manager["golden_stop_token"]));

    auto [TACOTRON_TRAINABLE_PARAMS, TACOTRON_PARAMS] = tacotronParamsCount(params);

    auto idealGradNorms = loadNamedValues(testPath / "unit" / "grad_norm.data", "T", params);
    EXPECT_EQ(idealGradNorms.size(), TACOTRON_TRAINABLE_PARAMS);
    size_t loaded = loadTacotronParams((testPath / "weights" / "160000_Tacotron_model.").string(), memory_manager, "T", params, true, false);
    EXPECT_EQ(loaded, TACOTRON_PARAMS);
    postprocessing::GradientClipping clip(1_dt);

    // targets
    EXPECT_TRUE(loadTFData(testPath / "stop_token_targets.data", memory_manager["stop_token_targets"]));
    EXPECT_TRUE(loadTFData(testPath / "targets_lengths.data", memory_manager["targets_lengths"]));
    EXPECT_TRUE(loadTFData(testPath / "mel_targets.data", memory_manager["mel_targets"]));

    // training data
    EXPECT_TRUE(loadTFData(testPath / "input_lengths.data", memory_manager["input_lengths"]));
    EXPECT_TRUE(loadTFData(testPath / "encoder_output.data", memory_manager["encoder_output"]));

    // speaker embedding
    EXPECT_TRUE(loadTFData(testPath / "speaker_embedding.data", memory_manager["speaker_embedding"]));

    auto callback = [&](BasicLayer* layer, const MemoryManager& memory_manager, NetworkParameters::CallbackPlace place) {
        if (place != NetworkParameters::CallbackPlace::Before_Forward)
        {
            return;
        }
        if (!Common::startsWith(layer->getName(), Name("total_loss") / "before" / "loss" / "mul"))
        {
            return;
        }

        auto& goldenf = memory_manager["golden_decoder_output"];
        auto& goldens = memory_manager["golden_stop_token"];
        auto& calculatedf = memory_manager["decoder_output"];

        for (size_t i = 0; i < decoder_iterations; ++i)
        {
            size_t sz = params.numMels * params.outputsPerStep;
            Tensor fr(Tensor::dt_range(&*(calculatedf.begin() + i * sz), &*(calculatedf.begin() + (i + 1) * sz)));

            Tensor gf(Tensor::dt_range(&*(goldenf.begin() + i * sz), &*(goldenf.begin() + (i + 1) * sz)));

            auto frDiff = TensorDiff(fr, gf);

            cout << "Iter " << i << ". frame: " << frDiff << "(" << TensorNorm(fr) << ", " << TensorNorm(gf) << ")";
            if (!params.withoutStopTokenLoss)
            {
                auto& calculateds = memory_manager["stop_token_prediction"];
                Tensor st(Tensor::dt_range(&*(calculateds.begin() + i * params.outputsPerStep), &*(calculateds.begin() + (i + 1) * params.outputsPerStep)));
                Tensor gs(Tensor::dt_range(&*(goldens.begin() + i * params.outputsPerStep), &*(goldens.begin() + (i + 1) * params.outputsPerStep)));

                auto stDiff = TensorDiff(st, gs);
                cout << ", stop: " << stDiff << "(" << TensorNorm(st) << ", " << TensorNorm(gs) << ")";

                if (!params.withoutStopTokenLoss)
                {
                    EXPECT_LE(stDiff, STOP_TOKEN_EPS);
                }
            }
            cout << endl;

            EXPECT_LE(frDiff, FRAME_PREDICTION_EPS);
        }
    };

    work.getNetworkParameters().mCallback = callback;

    timeStart = chrono::steady_clock::now();
    work.forwardPassTraining();

    auto& lossValue = memory_manager["loss"];
    totalTime = TODTYPE(chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - timeStart).count());
    cout << "Epoch time " << totalTime << " ms for batch size " << BATCH_SIZE << endl;
    cout << "Initial loss = " << lossValue[0] << " (before = " << memory_manager["before_loss"][0] << ", after = " << memory_manager["after_loss"][0];
    if (!params.withoutStopTokenLoss)
    {
        cout << ", stop_token = " << memory_manager["stop_token_loss"][0];
    }
    cout << ")" << endl;
    cout << "Ideal loss = " << (idealLosses[0] + idealLosses[1] + (params.withoutStopTokenLoss ? 0_dt : idealLosses[2])) << " (before = " << idealLosses[0] << ", after = " << idealLosses[1];
    if (!params.withoutStopTokenLoss)
    {
        cout << ", stop_token = " << idealLosses[2];
    }
    cout << ")" << endl;

    work.backwardPassTraining();

    auto tParams = work.getTrainableParameters();
    // clip.processGradients(tParams, work.getNetworkParameters());
    cout << "Gradients." << endl;
    cout << "Global norm: " << clip.calcGlobalNorm(tParams, work.getNetworkParameters()) << endl;
    dtype totalGradNormDiff = 0;
    for (size_t j = 0; j < tParams.size(); ++j)
    {
        auto& [param, grad] = tParams[j];
        cout << param.getName() << ": " << TensorNorm(grad) << " (" << idealGradNorms[param.getName()] << ")" << endl;

        cout << "  diff: " << fabs(TensorNorm(grad) - idealGradNorms[param.getName()]) << endl;
        totalGradNormDiff += fabs(TensorNorm(grad) - idealGradNorms[param.getName()]);
    }
    cout << "Total Grad Norm Diff: " << totalGradNormDiff << endl;

    EXPECT_TRUE(totalGradNormDiff < 0.5_dt);
}

TEST(TestTacotronSpB, DISABLED_MultiSpeakerMskSMAModelTraining)
{
    PROFILE_TEST

    using namespace optimizers;

    random::setGlobalSeed(42);

    bool usePool = true;
    Workflow work(CompressionMode::NONE, CalculationMode::DETERMINISTIC, usePool ? AllocationMode::POOL : AllocationMode::STANDARD);

    constexpr size_t BATCH_SIZE = 40;

    constexpr dtype ADAM_BETA1 = 0.9_dt;
    constexpr dtype ADAM_BETA2 = 0.999_dt;
    constexpr dtype ADAM_EPSILON = 1e-6_dt;

    auto testPath = tools::getTestAssetsDir() / "Tacotron_Msk_M4";
    string experimentName = "multispeaker";
    auto outputPrefix = std::filesystem::path("Tacotron_Msk") / "Adam_p297" / experimentName;

    TacotronParams params({ "encoder_output", "speaker_embedding", "input_lengths", "mel_targets" }, { "decoder_output", "mel_outputs", "stop_token_prediction" }, {});
    params.dropoutRate = 0.5f;           // 0.5 for training
    params.postnetDropoutRate = 0.1f;    // 0.1 for training
    params.zoneoutRate = 0.1f;           // 0.1 for training
    params.attentionSigmoidNoise = 1.0f; // 1.0 for training
    params.attentionScoreBiasInit = -1.5;
    params.useAttentionRnn = false;
    params.useResidualRnn = false;
    params.trainableSpeakerEmbedding = false;
    params.attentionType = "StepwiseMonotonic";

    params.postnetChannels = 512;
    params.decoderLstmUnits = { 512, 512, 512 };
    params.maskUseSquaredError = true;

    params.maxMelFrames = 984; // @todo: read from file
    params.maskedSigmoidEpsilon = 5e-7f;
    params.withoutStopTokenLoss = false; // false for training

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

    size_t nEpoch = 1200;
    const size_t nEpochToSave = 200;

    size_t T_IN = 197; // number of steps in the input time series

    work.add<DataLayer>("encoder_output", DataParams{ { "encoder_output" }, 1u, T_IN, params.embeddingDim });
    work.add<DataLayer>("speaker_embedding", DataParams{ { "speaker_embedding" }, 1u, 1u, params.speakerEmbeddingSize });
    work.add<DataLayer>("input_lengths", DataParams{ { "input_lengths" }, 1u, 1u, 1u });
    work.add<DataLayer>("mel_targets", DataParams{ { "mel_targets" }, 1u, params.maxMelFrames, params.numMels });

    work.add<DataLayer>("targets_lengths", DataParams{ { "targets_lengths" }, 1u, 1u, 1u });
    work.add<DataLayer>("stop_token_targets", DataParams{ { "stop_token_targets" }, 1u, params.maxMelFrames, 1u });

    AddSingleSpeakerFslTacotronModel(&work, "T", params);
    tacotron::AddTacotronLoss(&work,
                              "total_loss",
                              { { "decoder_output", "mel_outputs", "stop_token_prediction", "mel_targets", "targets_lengths", "stop_token_targets" },
                                params.withoutStopTokenLoss ? Names{ "loss", "before_loss", "after_loss" } : Names{ "loss", "before_loss", "after_loss", "stop_token_loss" } },
                              params);

    auto timeStart = chrono::steady_clock::now();
    TENSORS_CREATE(BATCH_SIZE);
    auto totalTime = TODTYPE(chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - timeStart).count());
    cout << "Workflow preparation time " << totalTime << " ms for batch size " << BATCH_SIZE << endl;
    MemoryManager& memory_manager = work.getMemoryManager();
    cout << memory_manager.size() << " tensors" << endl;

    auto [TACOTRON_TRAINABLE_PARAMS, TACOTRON_PARAMS] = tacotronParamsCount(params);

    size_t loaded = loadTacotronParams((testPath / "weights" / "160000_Tacotron_model.").string(), memory_manager, "T", params, true, false);
    EXPECT_EQ(loaded, TACOTRON_PARAMS);
    postprocessing::GradientClipping clip(1_dt);

    // targets
    EXPECT_TRUE(loadTFData(testPath / "stop_token_targets.data", memory_manager["stop_token_targets"]));
    EXPECT_TRUE(loadTFData(testPath / "targets_lengths.data", memory_manager["targets_lengths"]));
    EXPECT_TRUE(loadTFData(testPath / "mel_targets.data", memory_manager["mel_targets"]));

    // training data
    EXPECT_TRUE(loadTFData(testPath / "input_lengths.data", memory_manager["input_lengths"]));
    EXPECT_TRUE(loadTFData(testPath / "encoder_output.data", memory_manager["encoder_output"]));

    // speaker embedding
    EXPECT_TRUE(loadTFData(testPath / "speaker_embedding.data", memory_manager["speaker_embedding"]));

    auto optimizer = createOptimizer(trainParams, ADAM_BETA1, ADAM_BETA2, ADAM_EPSILON);

    dtype fastestEpochTime = 1e+6_dt;
    dtype slowestEpochRime = 0_dt;
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
        auto epochTime = TODTYPE(chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - timeStart).count());

        trainingTime += epochTime;

        fastestEpochTime = std::min(fastestEpochTime, epochTime);
        slowestEpochRime = std::max(slowestEpochRime, epochTime);

        cout << "  Epoch time " << epochTime << " ms for batch size " << BATCH_SIZE << endl;
        cout << "  Average performance: " << trainingTime / TODTYPE(epoch + 1) << " ms/epoch (fastest: " << fastestEpochTime << ", slowest " << slowestEpochRime << ")" << endl << endl;
        cout << "  Loss = " << lossValue[0] << " (before = " << memory_manager["before_loss"][0] << ", after = " << memory_manager["after_loss"][0]
             << ", stop_token = " << (params.withoutStopTokenLoss ? 0_dt : memory_manager["stop_token_loss"][0]) << ")" << endl;

        cout << "  Global grad norm: " << gnorm_before << " -> " << gnorm_after << endl;

        if (nEpochToSave > 0 && ((epoch + 1) % nEpochToSave == 0 || epoch == nEpoch - 1))
        {
            [[maybe_unused]] size_t saved = saveTacotronParams(outputPrefix / to_string(epoch + 1), "160000_Tacotron_model.", memory_manager, "T", params, false);
        }
    }
}

TEST(TestTacotronSpB, DISABLED_MultiSpeakerMskSMAModelShortTraining)
{
    PROFILE_TEST

    using namespace optimizers;

    random::setGlobalSeed(42);

    bool usePool = false;
    Workflow work(CompressionMode::NONE, CalculationMode::DETERMINISTIC, usePool ? AllocationMode::POOL : AllocationMode::STANDARD);

    constexpr size_t BATCH_SIZE = 16;

    constexpr dtype ADAM_BETA1 = 0.9_dt;
    constexpr dtype ADAM_BETA2 = 0.999_dt;
    constexpr dtype ADAM_EPSILON = 1e-6_dt;

    auto testPath = tools::getTestAssetsDir() / "Tacotron_Msk_M4";
    string experimentName = "multispeaker";
    auto outputPrefix = std::filesystem::path("Tacotron_Msk") / "Adam_p297_short" / experimentName;

    TacotronParams params({ "encoder_output", "speaker_embedding", "input_lengths", "mel_targets" }, { "decoder_output", "mel_outputs", "stop_token_prediction" }, {});
    params.dropoutRate = 0.5f;           // 0.5 for training
    params.postnetDropoutRate = 0.1f;    // 0.1 for training
    params.zoneoutRate = 0.1f;           // 0.1 for training
    params.attentionSigmoidNoise = 1.0f; // 1.0 for training
    params.attentionScoreBiasInit = -1.5;
    params.useAttentionRnn = false;
    params.useResidualRnn = false;
    params.trainableSpeakerEmbedding = false;
    params.attentionType = "StepwiseMonotonic";

    params.postnetChannels = 512;
    params.decoderLstmUnits = { 512, 512, 512 };
    params.maskUseSquaredError = true;

    params.maxMelFrames = 546; // @todo: read from file
    params.maskedSigmoidEpsilon = 5e-7f;
    params.withoutStopTokenLoss = false; // false for training

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

    constexpr size_t nEpoch = 200;
    constexpr size_t nEpochToSave = 0;

    size_t T_IN = 88; // number of steps in the input time series

    work.add<DataLayer>("encoder_output", DataParams{ { "encoder_output" }, 1u, T_IN, params.embeddingDim });
    work.add<DataLayer>("speaker_embedding", DataParams{ { "speaker_embedding" }, 1u, 1u, params.speakerEmbeddingSize });
    work.add<DataLayer>("input_lengths", DataParams{ { "input_lengths" }, 1u, 1u, 1u });
    work.add<DataLayer>("mel_targets", DataParams{ { "mel_targets" }, 1u, params.maxMelFrames, params.numMels });

    work.add<DataLayer>("targets_lengths", DataParams{ { "targets_lengths" }, 1u, 1u, 1u });
    work.add<DataLayer>("stop_token_targets", DataParams{ { "stop_token_targets" }, 1u, params.maxMelFrames, 1u });

    AddSingleSpeakerFslTacotronModel(&work, "T", params);
    tacotron::AddTacotronLoss(&work,
                              "total_loss",
                              { { "decoder_output", "mel_outputs", "stop_token_prediction", "mel_targets", "targets_lengths", "stop_token_targets" },
                                params.withoutStopTokenLoss ? Names{ "loss", "before_loss", "after_loss" } : Names{ "loss", "before_loss", "after_loss", "stop_token_loss" } },
                              params);

    auto timeStart = chrono::steady_clock::now();
    TENSORS_CREATE(BATCH_SIZE);
    auto totalTime = TODTYPE(chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - timeStart).count());
    cout << "Workflow preparation time " << totalTime << " ms for batch size " << BATCH_SIZE << endl;
    MemoryManager& memory_manager = work.getMemoryManager();
    cout << memory_manager.size() << " tensors" << endl;

    auto [TACOTRON_TRAINABLE_PARAMS, TACOTRON_PARAMS] = tacotronParamsCount(params);

    size_t loaded = loadTacotronParams((testPath / "weights" / "160000_Tacotron_model.").string(), memory_manager, "T", params, true, false);
    EXPECT_EQ(loaded, TACOTRON_PARAMS);
    postprocessing::GradientClipping clip(1_dt);

    // targets
    EXPECT_TRUE(loadTFData(testPath / "scarlett_p297" / "stop_token_targets.data", memory_manager["stop_token_targets"]));
    EXPECT_TRUE(loadTFData(testPath / "scarlett_p297" / "targets_lengths.data", memory_manager["targets_lengths"]));
    EXPECT_TRUE(loadTFData(testPath / "scarlett_p297" / "mel_targets.data", memory_manager["mel_targets"]));

    // training data
    EXPECT_TRUE(loadTFData(testPath / "scarlett_p297" / "input_lengths.data", memory_manager["input_lengths"]));
    EXPECT_TRUE(loadTFData(testPath / "scarlett_p297" / "encoder_output.data", memory_manager["encoder_output"]));

    // speaker embedding
    EXPECT_TRUE(loadTFData(testPath / "scarlett_p297" / "speaker_embedding.data", memory_manager["speaker_embedding"]));

    auto optimizer = createOptimizer(trainParams, ADAM_BETA1, ADAM_BETA2, ADAM_EPSILON);

    dtype firstEpochTime = 1e+6_dt;
    dtype fastestEpochTime = 1e+6_dt;
    dtype slowestEpochRime = 0_dt;
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
        auto epochTime = TODTYPE(chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - timeStart).count());

        trainingTime += epochTime;

        fastestEpochTime = std::min(fastestEpochTime, epochTime);
        slowestEpochRime = std::max(slowestEpochRime, epochTime);
        if (epoch == 0)
        {
            firstEpochTime = epochTime;
        }

        auto averageTime = epoch == 0 ? firstEpochTime : (trainingTime - firstEpochTime) / TODTYPE(epoch);

        cout << "  Epoch time " << epochTime << " ms for batch size " << BATCH_SIZE << endl;
        cout << "  Average performance: " << averageTime << " ms/epoch (first: " << firstEpochTime << ", fastest: " << fastestEpochTime << ", slowest " << slowestEpochRime << ")" << endl << endl;
        cout << "  Loss = " << lossValue[0] << " (before = " << memory_manager["before_loss"][0] << ", after = " << memory_manager["after_loss"][0]
             << ", stop_token = " << (params.withoutStopTokenLoss ? 0_dt : memory_manager["stop_token_loss"][0]) << ")" << endl;

        cout << "  Global grad norm: " << gnorm_before << " -> " << gnorm_after << endl;

        if (epoch == 0)
        {
            slowestEpochRime = 0_dt;
        }

        if constexpr (nEpochToSave > 0)
        {
            if ((epoch + 1) % nEpochToSave == 0 || epoch == nEpoch - 1)
            {
                [[maybe_unused]] size_t saved = saveTacotronParams(outputPrefix / to_string(epoch + 1), "160000_Tacotron_model.", memory_manager, "T", params, false);
            }
        }
    }
}

TEST(TestTacotronSpB, DISABLED_MultiSpeakerMskSMAModelShirshova24Training)
{
    PROFILE_TEST

    using namespace optimizers;

    random::setGlobalSeed(42);

    bool usePool = true;
    Workflow work(CompressionMode::NONE, CalculationMode::DETERMINISTIC, usePool ? AllocationMode::POOL : AllocationMode::STANDARD);

    constexpr size_t BATCH_SIZE = 8;
    constexpr size_t TOTAL_SIZE = 24;
    constexpr size_t nIters = TOTAL_SIZE / BATCH_SIZE;

    constexpr dtype ADAM_BETA1 = 0.9_dt;
    constexpr dtype ADAM_BETA2 = 0.999_dt;
    constexpr dtype ADAM_EPSILON = 1e-6_dt;

    auto testPath = tools::getTestAssetsDir() / "Tacotron_Msk_M4";
    string experimentName = "Shirshova_24";
    auto outputPrefix = std::filesystem::path("Tacotron_Msk") / experimentName;

    TacotronParams params({ "encoder_output", "speaker_embedding", "input_lengths", "mel_targets" }, { "decoder_output", "mel_outputs", "stop_token_prediction" }, {});
    params.dropoutRate = 0.5f;           // 0.5 for training
    params.postnetDropoutRate = 0.1f;    // 0.1 for training
    params.zoneoutRate = 0.1f;           // 0.1 for training
    params.attentionSigmoidNoise = 1.0f; // 1.0 for training
    params.attentionScoreBiasInit = -1.5;
    params.useAttentionRnn = false;
    params.useResidualRnn = false;
    params.trainableSpeakerEmbedding = false;
    params.attentionType = "StepwiseMonotonic";

    params.postnetChannels = 512;
    params.decoderLstmUnits = { 512, 512, 512 };
    params.maskUseSquaredError = true;

    params.maxMelFrames = 744; // @todo: read from file
    params.maskedSigmoidEpsilon = 5e-7f;
    params.withoutStopTokenLoss = false; // false for training

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

    constexpr size_t nEpoch = 1200 / nIters;
    constexpr size_t nEpochToSave = 300 / nIters;

    size_t T_IN = 103; // number of steps in the input time series

    work.add<DataLayer>("encoder_output", DataParams{ { "encoder_output" }, 1u, T_IN, params.embeddingDim });
    work.add<DataLayer>("speaker_embedding", DataParams{ { "speaker_embedding" }, 1u, 1u, params.speakerEmbeddingSize });
    work.add<DataLayer>("input_lengths", DataParams{ { "input_lengths" }, 1u, 1u, 1u });
    work.add<DataLayer>("mel_targets", DataParams{ { "mel_targets" }, 1u, params.maxMelFrames, params.numMels });

    work.add<DataLayer>("targets_lengths", DataParams{ { "targets_lengths" }, 1u, 1u, 1u });
    work.add<DataLayer>("stop_token_targets", DataParams{ { "stop_token_targets" }, 1u, params.maxMelFrames, 1u });

    AddSingleSpeakerFslTacotronModel(&work, "T", params);
    tacotron::AddTacotronLoss(&work,
                              "total_loss",
                              { { "decoder_output", "mel_outputs", "stop_token_prediction", "mel_targets", "targets_lengths", "stop_token_targets" },
                                params.withoutStopTokenLoss ? Names{ "loss", "before_loss", "after_loss" } : Names{ "loss", "before_loss", "after_loss", "stop_token_loss" } },
                              params);

    auto timeStart = chrono::steady_clock::now();
    TENSORS_CREATE(BATCH_SIZE);
    auto totalTime = TODTYPE(chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - timeStart).count());
    cout << "Workflow preparation time " << totalTime << " ms for batch size " << BATCH_SIZE << endl;
    MemoryManager& memory_manager = work.getMemoryManager();
    cout << memory_manager.size() << " tensors" << endl;

    memory_manager.createTensor("encoder_output_full", TOTAL_SIZE, 1u, T_IN, params.embeddingDim);
    memory_manager.createTensor("speaker_embedding_full", TOTAL_SIZE, 1u, 1u, params.speakerEmbeddingSize);
    memory_manager.createTensor("input_lengths_full", TOTAL_SIZE, 1u, 1u, 1u);
    memory_manager.createTensor("mel_targets_full", TOTAL_SIZE, 1u, params.maxMelFrames, params.numMels);
    memory_manager.createTensor("targets_lengths_full", TOTAL_SIZE, 1u, 1u, 1u);
    memory_manager.createTensor("stop_token_targets_full", TOTAL_SIZE, 1u, params.maxMelFrames, 1u);

    auto [TACOTRON_TRAINABLE_PARAMS, TACOTRON_PARAMS] = tacotronParamsCount(params);

    size_t loaded = loadTacotronParams((testPath / "weights" / "160000_Tacotron_model.").string(), memory_manager, "T", params, true, false);
    EXPECT_EQ(loaded, TACOTRON_PARAMS);
    postprocessing::GradientClipping clip(1_dt);

    // targets
    EXPECT_TRUE(loadTFData(testPath / "shirshova_p297_24" / "stop_token_targets.data", memory_manager["stop_token_targets_full"]));
    EXPECT_TRUE(loadTFData(testPath / "shirshova_p297_24" / "targets_lengths.data", memory_manager["targets_lengths_full"]));
    EXPECT_TRUE(loadTFData(testPath / "shirshova_p297_24" / "mel_targets.data", memory_manager["mel_targets_full"]));

    // training data
    EXPECT_TRUE(loadTFData(testPath / "shirshova_p297_24" / "input_lengths.data", memory_manager["input_lengths_full"]));
    EXPECT_TRUE(loadTFData(testPath / "shirshova_p297_24" / "encoder_output.data", memory_manager["encoder_output_full"]));

    // speaker embedding
    EXPECT_TRUE(loadTFData(testPath / "shirshova_p297_24" / "speaker_embedding.data", memory_manager["speaker_embedding_full"]));

    auto optimizer = createOptimizer(trainParams, ADAM_BETA1, ADAM_BETA2, ADAM_EPSILON);

    dtype firstEpochTime = 1e+6_dt;
    dtype fastestEpochTime = 1e+6_dt;
    dtype slowestEpochRime = 0_dt;
    dtype trainingTime = 0;

    Names batchTensors = { "stop_token_targets", "targets_lengths", "mel_targets", "input_lengths", "encoder_output", "speaker_embedding" };

    for (size_t epoch = 0; epoch < nEpoch; ++epoch)
    {
        cout << "Epoch " << epoch << endl;
        timeStart = chrono::steady_clock::now();

        for (size_t iter = 0; iter < nIters; ++iter)
        {
            for (const auto& tName : batchTensors)
            {
                auto& tfull = memory_manager[tName + "_full"];
                auto& tbatch = memory_manager[tName];
                copy_n(tfull.begin() + iter * tbatch.size(), tbatch.size(), tbatch.begin());
            }

            work.forwardPassTraining();

            work.backwardPassTraining();

            auto trainableParams = work.getTrainableParameters();

            clip.processGradients(trainableParams, work.getNetworkParameters());

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
        auto epochTime = TODTYPE(chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - timeStart).count());

        trainingTime += epochTime;

        fastestEpochTime = std::min(fastestEpochTime, epochTime);
        slowestEpochRime = std::max(slowestEpochRime, epochTime);
        if (epoch == 0)
        {
            firstEpochTime = epochTime;
        }

        auto averageTime = epoch == 0 ? firstEpochTime : (trainingTime - firstEpochTime) / TODTYPE(epoch);

        cout << "  Epoch time " << epochTime << " ms for " << nIters << " batches of size " << BATCH_SIZE << endl;
        cout << "  Average performance: " << averageTime << " ms/epoch (first: " << firstEpochTime << ", fastest: " << fastestEpochTime << ", slowest " << slowestEpochRime << ")" << endl << endl;
        cout << "  Loss = " << lossValue[0] << " (before = " << memory_manager["before_loss"][0] << ", after = " << memory_manager["after_loss"][0]
             << ", stop_token = " << (params.withoutStopTokenLoss ? 0_dt : memory_manager["stop_token_loss"][0]) << ")" << endl;

        if (epoch == 0)
        {
            slowestEpochRime = 0_dt;
        }

        if constexpr (nEpochToSave > 0)
        {
            if ((epoch + 1) % nEpochToSave == 0 || epoch == nEpoch - 1)
            {
                [[maybe_unused]] size_t saved = saveTacotronParams(outputPrefix / to_string(epoch + 1), "160000_Tacotron_model.", memory_manager, "T", params, false);
            }
        }
    }
}

TEST(TestTacotronSpB, DISABLED_MultiSpeakerMskSMAModelShirshova32Training)
{
    PROFILE_TEST

    using namespace optimizers;

    random::setGlobalSeed(42);

    bool usePool = true;
    Workflow work(CompressionMode::NONE, CalculationMode::DETERMINISTIC, usePool ? AllocationMode::POOL : AllocationMode::STANDARD);

    constexpr size_t BATCH_SIZE = 16;
    constexpr size_t TOTAL_SIZE = 32;
    constexpr size_t nIters = TOTAL_SIZE / BATCH_SIZE;

    constexpr dtype ADAM_BETA1 = 0.9_dt;
    constexpr dtype ADAM_BETA2 = 0.999_dt;
    constexpr dtype ADAM_EPSILON = 1e-6_dt;

    auto testPath = tools::getTestAssetsDir() / "Tacotron_Msk_M4";
    string experimentName = "Shirshova_32";
    auto outputPrefix = std::filesystem::path("Tacotron_Msk") / experimentName;

    TacotronParams params({ "encoder_output", "speaker_embedding", "input_lengths", "mel_targets" }, { "decoder_output", "mel_outputs", "stop_token_prediction" }, {});
    params.dropoutRate = 0.5f;           // 0.5 for training
    params.postnetDropoutRate = 0.1f;    // 0.1 for training
    params.zoneoutRate = 0.1f;           // 0.1 for training
    params.attentionSigmoidNoise = 1.0f; // 1.0 for training
    params.attentionScoreBiasInit = -1.5;
    params.useAttentionRnn = false;
    params.useResidualRnn = false;
    params.trainableSpeakerEmbedding = false;
    params.attentionType = "StepwiseMonotonic";

    params.postnetChannels = 512;
    params.decoderLstmUnits = { 512, 512, 512 };
    params.maskUseSquaredError = true;

    params.maxMelFrames = 744; // @todo: read from file
    params.maskedSigmoidEpsilon = 5e-7f;
    params.withoutStopTokenLoss = false; // false for training

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

    constexpr size_t nEpoch = 1200 / nIters;
    constexpr size_t nEpochToSave = 400 / nIters;

    size_t T_IN = 103; // number of steps in the input time series

    work.add<DataLayer>("encoder_output", DataParams{ { "encoder_output" }, 1u, T_IN, params.embeddingDim });
    work.add<DataLayer>("speaker_embedding", DataParams{ { "speaker_embedding" }, 1u, 1u, params.speakerEmbeddingSize });
    work.add<DataLayer>("input_lengths", DataParams{ { "input_lengths" }, 1u, 1u, 1u });
    work.add<DataLayer>("mel_targets", DataParams{ { "mel_targets" }, 1u, params.maxMelFrames, params.numMels });

    work.add<DataLayer>("targets_lengths", DataParams{ { "targets_lengths" }, 1u, 1u, 1u });
    work.add<DataLayer>("stop_token_targets", DataParams{ { "stop_token_targets" }, 1u, params.maxMelFrames, 1u });

    AddSingleSpeakerFslTacotronModel(&work, "T", params);
    tacotron::AddTacotronLoss(&work,
                              "total_loss",
                              { { "decoder_output", "mel_outputs", "stop_token_prediction", "mel_targets", "targets_lengths", "stop_token_targets" },
                                params.withoutStopTokenLoss ? Names{ "loss", "before_loss", "after_loss" } : Names{ "loss", "before_loss", "after_loss", "stop_token_loss" } },
                              params);

    auto timeStart = chrono::steady_clock::now();
    TENSORS_CREATE(BATCH_SIZE);
    auto totalTime = TODTYPE(chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - timeStart).count());
    cout << "Workflow preparation time " << totalTime << " ms for batch size " << BATCH_SIZE << endl;
    MemoryManager& memory_manager = work.getMemoryManager();
    cout << memory_manager.size() << " tensors" << endl;
    cout << tools::get_size_of_trainable_params(work) << " total trainable weights" << endl;

    memory_manager.createTensor("encoder_output_full", TOTAL_SIZE, 1u, T_IN, params.embeddingDim);
    memory_manager.createTensor("speaker_embedding_full", TOTAL_SIZE, 1u, 1u, params.speakerEmbeddingSize);
    memory_manager.createTensor("input_lengths_full", TOTAL_SIZE, 1u, 1u, 1u);
    memory_manager.createTensor("mel_targets_full", TOTAL_SIZE, 1u, params.maxMelFrames, params.numMels);
    memory_manager.createTensor("targets_lengths_full", TOTAL_SIZE, 1u, 1u, 1u);
    memory_manager.createTensor("stop_token_targets_full", TOTAL_SIZE, 1u, params.maxMelFrames, 1u);

    auto [TACOTRON_TRAINABLE_PARAMS, TACOTRON_PARAMS] = tacotronParamsCount(params);

    size_t loaded = loadTacotronParams((testPath / "weights" / "160000_Tacotron_model.").string(), memory_manager, "T", params, true, false);
    EXPECT_EQ(loaded, TACOTRON_PARAMS);
    postprocessing::GradientClipping clip(1_dt);

    // targets
    EXPECT_TRUE(loadTFData(testPath / "shirshova_p297_32" / "stop_token_targets.data", memory_manager["stop_token_targets_full"]));
    EXPECT_TRUE(loadTFData(testPath / "shirshova_p297_32" / "targets_lengths.data", memory_manager["targets_lengths_full"]));
    EXPECT_TRUE(loadTFData(testPath / "shirshova_p297_32" / "mel_targets.data", memory_manager["mel_targets_full"]));

    // training data
    EXPECT_TRUE(loadTFData(testPath / "shirshova_p297_32" / "input_lengths.data", memory_manager["input_lengths_full"]));
    EXPECT_TRUE(loadTFData(testPath / "shirshova_p297_32" / "encoder_output.data", memory_manager["encoder_output_full"]));

    // speaker embedding
    EXPECT_TRUE(loadTFData(testPath / "shirshova_p297_32" / "speaker_embedding.data", memory_manager["speaker_embedding_full"]));

    auto optimizer = createOptimizer(trainParams, ADAM_BETA1, ADAM_BETA2, ADAM_EPSILON);

    dtype firstEpochTime = 1e+6_dt;
    dtype fastestEpochTime = 1e+6_dt;
    dtype slowestEpochRime = 0_dt;
    dtype trainingTime = 0;

    Names batchTensors = { "stop_token_targets", "targets_lengths", "mel_targets", "input_lengths", "encoder_output", "speaker_embedding" };

    for (size_t epoch = 0; epoch < nEpoch; ++epoch)
    {
        cout << "Epoch " << epoch << endl;
        timeStart = chrono::steady_clock::now();

        for (size_t iter = 0; iter < nIters; ++iter)
        {
            for (const auto& tName : batchTensors)
            {
                auto& tfull = memory_manager[tName + "_full"];
                auto& tbatch = memory_manager[tName];
                copy_n(tfull.begin() + iter * tbatch.size(), tbatch.size(), tbatch.begin());
            }

            work.forwardPassTraining();

            work.backwardPassTraining();

            auto trainableParams = work.getTrainableParameters();

            clip.processGradients(trainableParams, work.getNetworkParameters());

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
        auto epochTime = TODTYPE(chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - timeStart).count());

        trainingTime += epochTime;

        fastestEpochTime = std::min(fastestEpochTime, epochTime);
        slowestEpochRime = std::max(slowestEpochRime, epochTime);
        if (epoch == 0)
        {
            firstEpochTime = epochTime;
        }

        auto averageTime = epoch == 0 ? firstEpochTime : (trainingTime - firstEpochTime) / TODTYPE(epoch);

        cout << "  Epoch time " << epochTime << " ms for " << nIters << " batches of size " << BATCH_SIZE << endl;
        cout << "  Average performance: " << averageTime << " ms/epoch (first: " << firstEpochTime << ", fastest: " << fastestEpochTime << ", slowest " << slowestEpochRime << ")" << endl << endl;
        cout << "  Loss = " << lossValue[0] << " (before = " << memory_manager["before_loss"][0] << ", after = " << memory_manager["after_loss"][0]
             << ", stop_token = " << (params.withoutStopTokenLoss ? 0_dt : memory_manager["stop_token_loss"][0]) << ")" << endl;

        if (epoch == 0)
        {
            slowestEpochRime = 0_dt;
        }

        if constexpr (nEpochToSave > 0)
        {
            if ((epoch + 1) % nEpochToSave == 0 || epoch == nEpoch - 1)
            {
                [[maybe_unused]] size_t saved = saveTacotronParams(outputPrefix / to_string(epoch + 1), "160000_Tacotron_model.", memory_manager, "T", params, false);
            }
        }
    }
}

TEST(TestTacotronSpB, DISABLED_MultiSpeakerMskSMAModelWithPoolShortTraining)
{
    PROFILE_TEST

    using namespace optimizers;

    random::setGlobalSeed(42);

    bool usePool = true;
    Workflow work(CompressionMode::NONE, CalculationMode::DETERMINISTIC, usePool ? AllocationMode::POOL : AllocationMode::STANDARD);

    constexpr size_t BATCH_SIZE = 16;

    constexpr dtype ADAM_BETA1 = 0.9_dt;
    constexpr dtype ADAM_BETA2 = 0.999_dt;
    constexpr dtype ADAM_EPSILON = 1e-6_dt;

    auto testPath = tools::getTestAssetsDir() / "Tacotron_Msk_M4";
    string experimentName = "multispeaker";
    auto outputPrefix = std::filesystem::path("Tacotron_Msk") / "Adam_p297_short" / experimentName;

    TacotronParams params({ "encoder_output", "speaker_embedding", "input_lengths", "mel_targets" }, { "decoder_output", "mel_outputs", "stop_token_prediction" }, {});
    params.dropoutRate = 0.5f;           // 0.5 for training
    params.postnetDropoutRate = 0.1f;    // 0.1 for training
    params.zoneoutRate = 0.1f;           // 0.1 for training
    params.attentionSigmoidNoise = 1.0f; // 1.0 for training
    params.attentionScoreBiasInit = -1.5;
    params.useAttentionRnn = false;
    params.useResidualRnn = false;
    params.trainableSpeakerEmbedding = false;
    params.attentionType = "StepwiseMonotonic";

    params.postnetChannels = 512;
    params.decoderLstmUnits = { 512, 512, 512 };
    params.maskUseSquaredError = true;

    params.maxMelFrames = 546; // @todo: read from file
    params.maskedSigmoidEpsilon = 5e-7f;
    params.withoutStopTokenLoss = false; // false for training

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

    constexpr size_t nEpoch = 200;
    constexpr size_t nEpochToSave = 0;

    size_t T_IN = 88; // number of steps in the input time series

    work.add<DataLayer>("encoder_output", DataParams{ { "encoder_output" }, 1u, T_IN, params.embeddingDim });
    work.add<DataLayer>("speaker_embedding", DataParams{ { "speaker_embedding" }, 1u, 1u, params.speakerEmbeddingSize });
    work.add<DataLayer>("input_lengths", DataParams{ { "input_lengths" }, 1u, 1u, 1u });
    work.add<DataLayer>("mel_targets", DataParams{ { "mel_targets" }, 1u, params.maxMelFrames, params.numMels });

    work.add<DataLayer>("targets_lengths", DataParams{ { "targets_lengths" }, 1u, 1u, 1u });
    work.add<DataLayer>("stop_token_targets", DataParams{ { "stop_token_targets" }, 1u, params.maxMelFrames, 1u });

    AddSingleSpeakerFslTacotronModel(&work, "T", params);
    tacotron::AddTacotronLoss(&work,
                              "total_loss",
                              { { "decoder_output", "mel_outputs", "stop_token_prediction", "mel_targets", "targets_lengths", "stop_token_targets" },
                                params.withoutStopTokenLoss ? Names{ "loss", "before_loss", "after_loss" } : Names{ "loss", "before_loss", "after_loss", "stop_token_loss" } },
                              params);

    auto timeStart = chrono::steady_clock::now();
    TENSORS_CREATE(BATCH_SIZE);
    auto totalTime = TODTYPE(chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - timeStart).count());
    cout << "Workflow preparation time " << totalTime << " ms for batch size " << BATCH_SIZE << endl;
    MemoryManager& memory_manager = work.getMemoryManager();
    cout << memory_manager.size() << " tensors" << endl;

    auto [TACOTRON_TRAINABLE_PARAMS, TACOTRON_PARAMS] = tacotronParamsCount(params);

    size_t loaded = loadTacotronParams((testPath / "weights" / "160000_Tacotron_model.").string(), memory_manager, "T", params, true, false);
    EXPECT_EQ(loaded, TACOTRON_PARAMS);
    postprocessing::GradientClipping clip(1_dt);

    // targets
    EXPECT_TRUE(loadTFData(testPath / "scarlett_p297" / "stop_token_targets.data", memory_manager["stop_token_targets"]));
    EXPECT_TRUE(loadTFData(testPath / "scarlett_p297" / "targets_lengths.data", memory_manager["targets_lengths"]));
    EXPECT_TRUE(loadTFData(testPath / "scarlett_p297" / "mel_targets.data", memory_manager["mel_targets"]));

    // training data
    EXPECT_TRUE(loadTFData(testPath / "scarlett_p297" / "input_lengths.data", memory_manager["input_lengths"]));
    EXPECT_TRUE(loadTFData(testPath / "scarlett_p297" / "encoder_output.data", memory_manager["encoder_output"]));

    // speaker embedding
    EXPECT_TRUE(loadTFData(testPath / "scarlett_p297" / "speaker_embedding.data", memory_manager["speaker_embedding"]));

    auto optimizer = createOptimizer(trainParams, ADAM_BETA1, ADAM_BETA2, ADAM_EPSILON);

    dtype firstEpochTime = 1e+6_dt;
    dtype fastestEpochTime = 1e+6_dt;
    dtype slowestEpochRime = 0_dt;
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
        auto epochTime = TODTYPE(chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - timeStart).count());

        trainingTime += epochTime;

        fastestEpochTime = std::min(fastestEpochTime, epochTime);
        slowestEpochRime = std::max(slowestEpochRime, epochTime);
        if (epoch == 0)
        {
            firstEpochTime = epochTime;
        }

        auto averageTime = epoch == 0 ? firstEpochTime : (trainingTime - firstEpochTime) / TODTYPE(epoch);

        cout << "  Epoch time " << epochTime << " ms for batch size " << BATCH_SIZE << endl;
        cout << "  Average performance: " << averageTime << " ms/epoch (first: " << firstEpochTime << ", fastest: " << fastestEpochTime << ", slowest " << slowestEpochRime << ")" << endl << endl;
        cout << "  Loss = " << lossValue[0] << " (before = " << memory_manager["before_loss"][0] << ", after = " << memory_manager["after_loss"][0]
             << ", stop_token = " << (params.withoutStopTokenLoss ? 0_dt : memory_manager["stop_token_loss"][0]) << ")" << endl;

        cout << "  Global grad norm: " << gnorm_before << " -> " << gnorm_after << endl;

        if (epoch == 0)
        {
            slowestEpochRime = 0_dt;
        }

        if constexpr (nEpochToSave > 0)
        {
            if ((epoch + 1) % nEpochToSave == 0 || epoch == nEpoch - 1)
            {
                [[maybe_unused]] size_t saved = saveTacotronParams(outputPrefix / to_string(epoch + 1), "160000_Tacotron_model.", memory_manager, "T", params, false);
            }
        }
    }
}

TEST(TestTacotronSpB, DISABLED_MultiSpeakerMskSMAModelWithPoolBSUpdateShortTraining)
{
    PROFILE_TEST

    using namespace optimizers;

    random::setGlobalSeed(42);

    bool usePool = true;
    Workflow work(CompressionMode::NONE, CalculationMode::DETERMINISTIC, usePool ? AllocationMode::POOL : AllocationMode::STANDARD);

    constexpr size_t BATCH_SIZE = 16;

    constexpr dtype ADAM_BETA1 = 0.9_dt;
    constexpr dtype ADAM_BETA2 = 0.999_dt;
    constexpr dtype ADAM_EPSILON = 1e-6_dt;

    auto testPath = tools::getTestAssetsDir() / "Tacotron_Msk_M4";
    string experimentName = "multispeaker";
    auto outputPrefix = std::filesystem::path("Tacotron_Msk") / "Adam_p297_short" / experimentName;

    TacotronParams params({ "encoder_output", "speaker_embedding", "input_lengths", "mel_targets" }, { "decoder_output", "mel_outputs", "stop_token_prediction" }, {});
    params.dropoutRate = 0.5f;           // 0.5 for training
    params.postnetDropoutRate = 0.1f;    // 0.1 for training
    params.zoneoutRate = 0.1f;           // 0.1 for training
    params.attentionSigmoidNoise = 1.0f; // 1.0 for training
    params.attentionScoreBiasInit = -1.5;
    params.useAttentionRnn = false;
    params.useResidualRnn = false;
    params.trainableSpeakerEmbedding = false;
    params.attentionType = "StepwiseMonotonic";

    params.postnetChannels = 512;
    params.decoderLstmUnits = { 512, 512, 512 };
    params.maskUseSquaredError = true;

    params.maxMelFrames = 546; // @todo: read from file
    params.maskedSigmoidEpsilon = 5e-7f;
    params.withoutStopTokenLoss = false; // false for training

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

    constexpr size_t nEpoch = 200;

    size_t T_IN = 88; // number of steps in the input time series

    work.add<DataLayer>("encoder_output", DataParams{ { "encoder_output" }, 1u, T_IN, params.embeddingDim });
    work.add<DataLayer>("speaker_embedding", DataParams{ { "speaker_embedding" }, 1u, 1u, params.speakerEmbeddingSize });
    work.add<DataLayer>("input_lengths", DataParams{ { "input_lengths" }, 1u, 1u, 1u });
    work.add<DataLayer>("mel_targets", DataParams{ { "mel_targets" }, 1u, params.maxMelFrames, params.numMels });

    work.add<DataLayer>("targets_lengths", DataParams{ { "targets_lengths" }, 1u, 1u, 1u });
    work.add<DataLayer>("stop_token_targets", DataParams{ { "stop_token_targets" }, 1u, params.maxMelFrames, 1u });

    AddSingleSpeakerFslTacotronModel(&work, "T", params);
    tacotron::AddTacotronLoss(&work,
                              "total_loss",
                              { { "decoder_output", "mel_outputs", "stop_token_prediction", "mel_targets", "targets_lengths", "stop_token_targets" },
                                params.withoutStopTokenLoss ? Names{ "loss", "before_loss", "after_loss" } : Names{ "loss", "before_loss", "after_loss", "stop_token_loss" } },
                              params);

    auto timeStart = chrono::steady_clock::now();
    TENSORS_CREATE(BATCH_SIZE);
    auto totalTime = TODTYPE(chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - timeStart).count());
    cout << "Workflow preparation time " << totalTime << " ms for batch size " << BATCH_SIZE << endl;
    MemoryManager& memory_manager = work.getMemoryManager();

    Names dataTensors = { "encoder_output", "speaker_embedding", "input_lengths", "mel_targets", "targets_lengths", "stop_token_targets" };

    cout << memory_manager.size() << " tensors" << endl;

    auto [TACOTRON_TRAINABLE_PARAMS, TACOTRON_PARAMS] = tacotronParamsCount(params);

    size_t loaded = loadTacotronParams((testPath / "weights" / "160000_Tacotron_model.").string(), memory_manager, "T", params, true, false);
    EXPECT_EQ(loaded, TACOTRON_PARAMS);
    postprocessing::GradientClipping clip(1_dt);

    // targets
    EXPECT_TRUE(loadTFData(testPath / "scarlett_p297" / "stop_token_targets.data", memory_manager["stop_token_targets"]));
    EXPECT_TRUE(loadTFData(testPath / "scarlett_p297" / "targets_lengths.data", memory_manager["targets_lengths"]));
    EXPECT_TRUE(loadTFData(testPath / "scarlett_p297" / "mel_targets.data", memory_manager["mel_targets"]));

    // training data
    EXPECT_TRUE(loadTFData(testPath / "scarlett_p297" / "input_lengths.data", memory_manager["input_lengths"]));
    EXPECT_TRUE(loadTFData(testPath / "scarlett_p297" / "encoder_output.data", memory_manager["encoder_output"]));

    // speaker embedding
    EXPECT_TRUE(loadTFData(testPath / "scarlett_p297" / "speaker_embedding.data", memory_manager["speaker_embedding"]));

    for (const auto& name : dataTensors)
    {
        const auto& t = memory_manager[name];
        memory_manager.createTensor(name + "_p", t.getShape());
        memory_manager[name + "_p"] = TORANGE(t);
    }

    auto optimizer = createOptimizer(trainParams, ADAM_BETA1, ADAM_BETA2, ADAM_EPSILON);

    dtype firstEpochTime = 1e+6_dt;
    dtype fastestEpochTime = 1e+6_dt;
    dtype slowestEpochRime = 0_dt;
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

        work.setBatchSize(BATCH_SIZE);
        for (const auto& name : dataTensors)
        {
            const auto& p = memory_manager[name + "_p"];
            memory_manager[name] = TORANGE(p);
        }

        auto& lossValue = memory_manager["loss"];
        auto epochTime = TODTYPE(chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - timeStart).count());

        trainingTime += epochTime;

        fastestEpochTime = std::min(fastestEpochTime, epochTime);
        slowestEpochRime = std::max(slowestEpochRime, epochTime);
        if (epoch == 0)
        {
            firstEpochTime = epochTime;
        }

        auto averageTime = epoch == 0 ? firstEpochTime : (trainingTime - firstEpochTime) / TODTYPE(epoch);

        cout << "  Epoch time " << epochTime << " ms for batch size " << BATCH_SIZE << endl;
        cout << "  Average performance: " << averageTime << " ms/epoch (first: " << firstEpochTime << ", fastest: " << fastestEpochTime << ", slowest " << slowestEpochRime << ")" << endl << endl;
        cout << "  Loss = " << lossValue[0] << " (before = " << memory_manager["before_loss"][0] << ", after = " << memory_manager["after_loss"][0]
             << ", stop_token = " << (params.withoutStopTokenLoss ? 0_dt : memory_manager["stop_token_loss"][0]) << ")" << endl;

        cout << "  Global grad norm: " << gnorm_before << " -> " << gnorm_after << endl;

        if (epoch == 0)
        {
            slowestEpochRime = 0_dt;
        }
    }
}

TEST(TestTacotronSpB, Adam_8_0_DeterministicUnit)
{
    PROFILE_TEST

    using namespace optimizers;

    Workflow work;

    constexpr size_t BATCH_SIZE = 1;

    constexpr dtype FRAME_PREDICTION_EPS = 1e-5_dt;
    constexpr dtype STOP_TOKEN_EPS = 1e-4_dt;

    auto testPath = tools::getTestAssetsDir() / "Tacotron_SpB";

    TacotronParams params({ "encoder_output", "input_lengths", "mel_targets" }, { "decoder_output", "mel_outputs", "stop_token_prediction" }, {});
    params.dropoutRate = 0.f;
    params.postnetDropoutRate = 0.f;
    params.zoneoutRate = 0.f;
    params.attentionSigmoidNoise = 0.f;
    params.useAttentionRnn = true;
    params.useResidualRnn = true;
    params.attentionType = "DynamicConvolutional";

    params.postnetChannels = 512;
    params.maskUseSquaredError = true;

    params.maxMelFrames = 255; // @todo: read from file
    params.maskedSigmoidEpsilon = 1e-5f;
    params.withoutStopTokenLoss = false;

    size_t decoder_iterations = params.maxMelFrames / params.outputsPerStep;

    size_t T_IN = 66; // number of steps in the input time series

    work.add<DataLayer>("encoder_output", DataParams{ { "encoder_output" }, 1u, T_IN, params.embeddingDim });
    work.add<DataLayer>("input_lengths", DataParams{ { "input_lengths" }, 1u, 1u, 1u });
    work.add<DataLayer>("mel_targets", DataParams{ { "mel_targets" }, 1u, params.maxMelFrames, params.numMels });

    work.add<DataLayer>("golden_decoder_output", DataParams{ { "golden_decoder_output" }, 1u, params.maxMelFrames, params.numMels });
    work.add<DataLayer>("golden_stop_token", DataParams{ { "golden_stop_token" }, 1u, params.maxMelFrames, 1u });

    work.add<DataLayer>("targets_lengths", DataParams{ { "targets_lengths" }, 1u, 1u, 1u });
    work.add<DataLayer>("stop_token_targets", DataParams{ { "stop_token_targets" }, 1u, params.maxMelFrames, 1u });

    AddSingleSpeakerFslTacotronModel(&work, "T", params);
    tacotron::AddTacotronLoss(&work,
                              "total_loss",
                              { { "decoder_output", "mel_outputs", "stop_token_prediction", "mel_targets", "targets_lengths", "stop_token_targets" },
                                params.withoutStopTokenLoss ? Names{ "loss", "before_loss", "after_loss" } : Names{ "loss", "before_loss", "after_loss", "stop_token_loss" } },
                              params);

    vector<dtype> idealLosses;
    ifstream f(testPath / "unit_8_0" / "loss.data");
    copy(istream_iterator<dtype>(f), istream_iterator<dtype>(), back_inserter(idealLosses));
    ASSERT_TRUE(idealLosses.size() >= 3);
    idealLosses.resize(3);

    auto timeStart = chrono::steady_clock::now();
    TENSORS_CREATE(BATCH_SIZE);
    auto totalTime = TODTYPE(chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - timeStart).count());
    cout << "Workflow preparation time " << totalTime << " ms for batch size " << BATCH_SIZE << endl;
    MemoryManager& memory_manager = work.getMemoryManager();
    EXPECT_TRUE(loadTFData(testPath / "unit_8_0" / "decoder_output.data", memory_manager["golden_decoder_output"]));
    EXPECT_TRUE(loadTFData(testPath / "unit_8_0" / "stop_token_prediction.data", memory_manager["golden_stop_token"]));

    auto [TACOTRON_TRAINABLE_PARAMS, TACOTRON_PARAMS] = tacotronParamsCount(params);

    auto idealGradNorms = loadNamedValues(testPath / "unit_8_0" / "grad_norm.data", "T", params);
    EXPECT_EQ(idealGradNorms.size(), TACOTRON_TRAINABLE_PARAMS);
    size_t loaded = loadTacotronParams((testPath / "weights" / "200000_Tacotron_model.").string(), memory_manager, "T", params, true, false);
    EXPECT_EQ(loaded, TACOTRON_PARAMS - 1); // -1 because speaker_embedding will be loaded separately and doesn't have Adam params

    postprocessing::GradientClipping clip;

    // targets
    EXPECT_TRUE(loadTFData(testPath / "unit_8_0" / "stop_token_targets.data", memory_manager["stop_token_targets"]));
    EXPECT_TRUE(loadTFData(testPath / "unit_8_0" / "targets_lengths.data", memory_manager["targets_lengths"]));
    EXPECT_TRUE(loadTFData(testPath / "unit_8_0" / "mel_targets.data", memory_manager["mel_targets"]));

    // training data
    EXPECT_TRUE(loadTFData(testPath / "unit_8_0" / "input_lengths.data", memory_manager["input_lengths"]));
    EXPECT_TRUE(loadTFData(testPath / "unit_8_0" / "encoder_output.data", memory_manager["encoder_output"]));

    // speaker embedding
    EXPECT_TRUE(loadTFData(testPath / "unit_8_0" / "speaker_embedding.data", memory_manager[Name("T") / "speaker_embedding"]));

    auto callback = [&](BasicLayer* layer, const MemoryManager& memory_manager, NetworkParameters::CallbackPlace place) {
        if (place != NetworkParameters::CallbackPlace::Before_Forward)
        {
            return;
        }
        if (!Common::startsWith(layer->getName(), Name("total_loss") / "before" / "loss" / "mul"))
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

            Tensor gf(Tensor::dt_range(&*(goldenf.begin() + i * sz), &*(goldenf.begin() + (i + 1) * sz)));

            auto frDiff = TensorDiff(fr, gf);
            dtype stDiff = 0_dt;
            dtype stNorm = 0_dt;
            dtype stGNorm = 0_dt;

            if (!params.withoutStopTokenLoss)
            {
                Tensor st(Tensor::dt_range(&*(calculateds.begin() + i * params.outputsPerStep), &*(calculateds.begin() + (i + 1) * params.outputsPerStep)));
                Tensor gs(Tensor::dt_range(&*(goldens.begin() + i * params.outputsPerStep), &*(goldens.begin() + (i + 1) * params.outputsPerStep)));

                stDiff = TensorDiff(st, gs);
                stNorm = TensorNorm(st);
                stGNorm = TensorNorm(gs);
            }

            cout << "Iter " << i << ". frame: " << frDiff << "(" << TensorNorm(fr) << ", " << TensorNorm(gf) << ")"
                 << ", stop: " << stDiff << "(" << stNorm << ", " << stGNorm << ")" << endl;

            EXPECT_LE(frDiff, FRAME_PREDICTION_EPS);
            EXPECT_LE(stDiff, STOP_TOKEN_EPS);
        }
    };

    work.getNetworkParameters().mCallback = callback;

    callbacks::LayerTypeStatistics statCallback;
    using namespace std::placeholders;
    work.getNetworkParameters().mCallback = std::bind(&callbacks::LayerTypeStatistics::operator(), &statCallback, _1, _2, _3);

    timeStart = chrono::steady_clock::now();
    work.forwardPassTraining();

    auto& lossValue = memory_manager["loss"];
    auto forwardTime = TODTYPE(chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - timeStart).count());

    timeStart = chrono::steady_clock::now();
    work.backwardPassTraining();
    auto backwardTime = TODTYPE(chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - timeStart).count());
    auto epochTime = forwardTime + backwardTime;
    cout << "Epoch time " << epochTime << " ms (forward " << forwardTime << "ms, backward " << backwardTime << "ms) for batch size " << BATCH_SIZE << endl;
    cout << "Initial loss = " << lossValue[0] << " (before = " << memory_manager["before_loss"][0] << ", after = " << memory_manager["after_loss"][0]
         << ", stop_token = " << (params.withoutStopTokenLoss ? 0_dt : memory_manager["stop_token_loss"][0]) << ")" << endl;
    cout << "Ideal loss = " << (idealLosses[0] + idealLosses[1] + idealLosses[2]) << " (before = " << idealLosses[0] << ", after = " << idealLosses[1] << ", stop_token = " << idealLosses[2] << ")"
         << endl;

    statCallback.print(forwardTime, backwardTime);

    auto tParams = work.getTrainableParameters();
    // clip.processGradients(tParams, work.getNetworkParameters());
    cout << "Gradients." << endl;
    cout << "Global norm: " << clip.calcGlobalNorm(tParams, work.getNetworkParameters()) << endl;
    dtype totalGradNormDiff = 0;
    for (size_t j = 0; j < tParams.size(); ++j)
    {
        auto& [param, grad] = tParams[j];
        cout << param.getName() << ": " << TensorNorm(grad) << " (" << idealGradNorms[param.getName()] << ")" << endl;

        cout << "  diff: " << fabs(TensorNorm(grad) - idealGradNorms[param.getName()]) << endl;
        totalGradNormDiff += fabs(TensorNorm(grad) - idealGradNorms[param.getName()]);
    }
    cout << "Total Grad Norm Diff: " << totalGradNormDiff << endl;

    EXPECT_TRUE(totalGradNormDiff < 0.5_dt);
}

TEST(TestTacotronSpB, DISABLED_Adam2Training)
{
    PROFILE_TEST

    using namespace optimizers;

    random::setGlobalSeed(42);

    bool usePool = true;
    Workflow work(CompressionMode::NONE, CalculationMode::DETERMINISTIC, usePool ? AllocationMode::POOL : AllocationMode::STANDARD);
    constexpr size_t BATCH_SIZE = 2;

    constexpr dtype ADAM_BETA1 = 0.9_dt;
    constexpr dtype ADAM_BETA2 = 0.999_dt;
    constexpr dtype ADAM_EPSILON = 1e-6_dt;

    auto testPath = tools::getTestAssetsDir() / "Tacotron_SpB";
    string experimentName = "general";
    auto outputPrefix = std::filesystem::path("Tacotron_SpB") / "Adam2" / experimentName;

    TacotronParams params({ "encoder_output", "input_lengths", "mel_targets" }, { "decoder_output", "mel_outputs", "stop_token_prediction" }, {});
    params.dropoutRate = 0.5f;
    params.postnetDropoutRate = 0.1f;
    params.zoneoutRate = 0.1f;
    params.useAttentionRnn = true;
    params.useResidualRnn = true;
    params.attentionType = "DynamicConvolutional";

    params.postnetChannels = 512;
    params.maskUseSquaredError = true;

    params.maxMelFrames = 984; // @todo: read from file
    params.maskedSigmoidEpsilon = 5e-7f;
    params.withoutStopTokenLoss = false;

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

    size_t nEpoch = 1400;
    const size_t nEpochToSave = 200;

    size_t T_IN = 197; // number of steps in the input time series

    work.add<DataLayer>("encoder_output", DataParams{ { "encoder_output" }, 1u, T_IN, params.embeddingDim });
    work.add<DataLayer>("input_lengths", DataParams{ { "input_lengths" }, 1u, 1u, 1u });
    work.add<DataLayer>("mel_targets", DataParams{ { "mel_targets" }, 1u, params.maxMelFrames, params.numMels });

    work.add<DataLayer>("targets_lengths", DataParams{ { "targets_lengths" }, 1u, 1u, 1u });
    work.add<DataLayer>("stop_token_targets", DataParams{ { "stop_token_targets" }, 1u, params.maxMelFrames, 1u });

    AddSingleSpeakerFslTacotronModel(&work, "T", params);
    tacotron::AddTacotronLoss(&work,
                              "total_loss",
                              { { "decoder_output", "mel_outputs", "stop_token_prediction", "mel_targets", "targets_lengths", "stop_token_targets" },
                                params.withoutStopTokenLoss ? Names{ "loss", "before_loss", "after_loss" } : Names{ "loss", "before_loss", "after_loss", "stop_token_loss" } },
                              params);

    auto timeStart = chrono::steady_clock::now();
    TENSORS_CREATE(BATCH_SIZE);

    auto epochTime = TODTYPE(chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - timeStart).count());
    cout << "Workflow preparation time " << epochTime << " ms for batch size " << BATCH_SIZE << endl;
    MemoryManager& memory_manager = work.getMemoryManager();

    cout << memory_manager.size() << "  tensors" << endl;

    auto [TACOTRON_TRAINABLE_PARAMS, TACOTRON_PARAMS] = tacotronParamsCount(params);

    size_t loaded = loadTacotronParams((testPath / "weights" / "200000_Tacotron_model.").string(), memory_manager, "T", params, true, false);
    EXPECT_EQ(loaded, TACOTRON_PARAMS - 1); // -1 because speaker_embedding will be loaded separately and doesn't have Adam params
    postprocessing::GradientClipping clip(1_dt);

    // targets
    EXPECT_TRUE(loadTFData(testPath / "unit" / "stop_token_targets.data", memory_manager["stop_token_targets"]));
    EXPECT_TRUE(loadTFData(testPath / "unit" / "targets_lengths.data", memory_manager["targets_lengths"]));
    EXPECT_TRUE(loadTFData(testPath / "unit" / "mel_targets.data", memory_manager["mel_targets"]));

    // training data
    EXPECT_TRUE(loadTFData(testPath / "unit" / "input_lengths.data", memory_manager["input_lengths"]));
    EXPECT_TRUE(loadTFData(testPath / "unit" / "encoder_output.data", memory_manager["encoder_output"]));

    // speaker embedding
    EXPECT_TRUE(loadTFData(testPath / "unit" / "speaker_embedding.data", memory_manager[Name("T") / "speaker_embedding"]));

    auto optimizer = createOptimizer(trainParams, ADAM_BETA1, ADAM_BETA2, ADAM_EPSILON);

    dtype fastestEpochTime = 1e+5_dt;
    dtype slowestEpochRime = 0_dt;
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

        if (nEpochToSave > 0 && ((epoch + 1) % nEpochToSave == 0 || epoch == nEpoch - 1))
        {
            [[maybe_unused]] size_t saved = saveTacotronParams(outputPrefix / to_string(epoch + 1), "200000_Tacotron_model.", memory_manager, "T", params, false);
        }
    }
}

TEST(TestTacotronSpB, DISABLED_AliceMskModelTraining)
{
    PROFILE_TEST

    using namespace optimizers;

    random::setGlobalSeed(42);

    bool usePool = true;
    Workflow work(CompressionMode::NONE, CalculationMode::DETERMINISTIC, usePool ? AllocationMode::POOL : AllocationMode::STANDARD);
    constexpr size_t BATCH_SIZE = 15;

    constexpr dtype ADAM_BETA1 = 0.9_dt;
    constexpr dtype ADAM_BETA2 = 0.999_dt;
    constexpr dtype ADAM_EPSILON = 1e-6_dt;

    auto testPath = tools::getTestAssetsDir() / "Tacotron_Msk";
    string experimentName = "general";
    auto outputPrefix = std::filesystem::path("Tacotron_Msk") / "Alice" / experimentName;

    TacotronParams params({ "encoder_output", "input_lengths", "mel_targets" }, { "decoder_output", "mel_outputs", "stop_token_prediction" }, {});
    params.dropoutRate = 0.5f;
    params.postnetDropoutRate = 0.1f;
    params.zoneoutRate = 0.1f;
    params.useAttentionRnn = true;
    params.useResidualRnn = true;
    params.attentionType = "DynamicConvolutional";

    params.postnetChannels = 512;
    params.decoderLstmUnits = { 512, 512 };
    params.maskUseSquaredError = true;

    params.attentionPriorBeta = 5.9f;

    params.maxMelFrames = 546; // @todo: read from file
    params.maskedSigmoidEpsilon = 5e-7f;
    params.withoutStopTokenLoss = false;

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

    size_t nEpoch = 1200;
    const size_t nEpochToSave = 200;

    size_t T_IN = 91; // number of steps in the input time series

    work.add<DataLayer>("encoder_output", DataParams{ { "encoder_output" }, 1u, T_IN, params.embeddingDim });
    work.add<DataLayer>("input_lengths", DataParams{ { "input_lengths" }, 1u, 1u, 1u });
    work.add<DataLayer>("mel_targets", DataParams{ { "mel_targets" }, 1u, params.maxMelFrames, params.numMels });

    work.add<DataLayer>("targets_lengths", DataParams{ { "targets_lengths" }, 1u, 1u, 1u });
    work.add<DataLayer>("stop_token_targets", DataParams{ { "stop_token_targets" }, 1u, params.maxMelFrames, 1u });

    AddSingleSpeakerFslTacotronModel(&work, "T", params);
    tacotron::AddTacotronLoss(&work,
                              "total_loss",
                              { { "decoder_output", "mel_outputs", "stop_token_prediction", "mel_targets", "targets_lengths", "stop_token_targets" },
                                params.withoutStopTokenLoss ? Names{ "loss", "before_loss", "after_loss" } : Names{ "loss", "before_loss", "after_loss", "stop_token_loss" } },
                              params);

    auto timeStart = chrono::steady_clock::now();
    TENSORS_CREATE(BATCH_SIZE);

    auto epochTime = TODTYPE(chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - timeStart).count());
    cout << "Workflow preparation time " << epochTime << " ms for batch size " << BATCH_SIZE << endl;
    MemoryManager& memory_manager = work.getMemoryManager();
    cout << memory_manager.size() << " tensors" << endl;

    auto [TACOTRON_TRAINABLE_PARAMS, TACOTRON_PARAMS] = tacotronParamsCount(params);

    size_t loaded = loadTacotronParams((testPath / "weights" / "184000_Tacotron_model.").string(), memory_manager, "T", params, true, false);
    EXPECT_EQ(loaded, TACOTRON_PARAMS - 1); // -1 because speaker_embedding will be loaded separately
    postprocessing::GradientClipping clip(1_dt);

    // targets
    EXPECT_TRUE(loadTFData(testPath / "stop_token_targets.data", memory_manager["stop_token_targets"]));
    EXPECT_TRUE(loadTFData(testPath / "targets_lengths.data", memory_manager["targets_lengths"]));
    EXPECT_TRUE(loadTFData(testPath / "mel_targets.data", memory_manager["mel_targets"]));

    // training data
    EXPECT_TRUE(loadTFData(testPath / "input_lengths.data", memory_manager["input_lengths"]));
    EXPECT_TRUE(loadTFData(testPath / "encoder_output.data", memory_manager["encoder_output"]));

    // speaker embedding
    EXPECT_TRUE(loadTFData(testPath / "speaker_embedding.data", memory_manager[Name("T") / "speaker_embedding"]));

    auto optimizer = createOptimizer(trainParams, ADAM_BETA1, ADAM_BETA2, ADAM_EPSILON);

    dtype fastestEpochTime = 1e+5_dt;
    dtype slowestEpochRime = 0_dt;
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

        if (nEpochToSave > 0 && ((epoch + 1) % nEpochToSave == 0 || epoch == nEpoch - 1))
        {
            [[maybe_unused]] size_t saved = saveTacotronParams(outputPrefix / to_string(epoch + 1), "184000_Tacotron_model.", memory_manager, "T", params, false);
        }
    }
}

TEST(TestTacotronSpB, Adam_8_0_DeterministicTraining)
{
    PROFILE_TEST

    using namespace optimizers;

    Workflow work;

    constexpr size_t BATCH_SIZE = 1;

    constexpr dtype LR_EPS = 1e-6_dt;

    constexpr dtype ADAM_BETA1 = 0.9_dt;
    constexpr dtype ADAM_BETA2 = 0.999_dt;
    constexpr dtype ADAM_EPSILON = 1e-6_dt;

    auto testPath = tools::getTestAssetsDir() / "Tacotron_SpB";

    TacotronParams params({ "encoder_output", "input_lengths", "mel_targets" }, { "decoder_output", "mel_outputs", "stop_token_prediction" }, {});
    params.dropoutRate = 0.f;
    params.postnetDropoutRate = 0.f;
    params.zoneoutRate = 0.f;
    params.attentionSigmoidNoise = 0.f;
    params.useAttentionRnn = true;
    params.useResidualRnn = true;
    params.attentionType = "DynamicConvolutional";

    params.postnetChannels = 512;
    params.maskUseSquaredError = true;

    params.maxMelFrames = 255; // @todo: read from file
    params.maskedSigmoidEpsilon = 1e-5f;
    params.withoutStopTokenLoss = false;

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

    size_t nEpoch = 200;

    size_t T_IN = 66; // number of steps in the input time series

    work.add<DataLayer>("encoder_output", DataParams{ { "encoder_output" }, 1u, T_IN, params.embeddingDim });
    work.add<DataLayer>("input_lengths", DataParams{ { "input_lengths" }, 1u, 1u, 1u });
    work.add<DataLayer>("mel_targets", DataParams{ { "mel_targets" }, 1u, params.maxMelFrames, params.numMels });

    work.add<DataLayer>("targets_lengths", DataParams{ { "targets_lengths" }, 1u, 1u, 1u });
    work.add<DataLayer>("stop_token_targets", DataParams{ { "stop_token_targets" }, 1u, params.maxMelFrames, 1u });

    AddSingleSpeakerFslTacotronModel(&work, "T", params);
    tacotron::AddTacotronLoss(&work,
                              "total_loss",
                              { { "decoder_output", "mel_outputs", "stop_token_prediction", "mel_targets", "targets_lengths", "stop_token_targets" },
                                params.withoutStopTokenLoss ? Names{ "loss", "before_loss", "after_loss" } : Names{ "loss", "before_loss", "after_loss", "stop_token_loss" } },
                              params);

    vector<dtype> idealLosses;
    vector<dtype> idealGradNorm;
    vector<dtype> idealLearningRate;

    {
        ifstream f(testPath / "unit_8_0" / "loss.data");
        copy(istream_iterator<dtype>(f), istream_iterator<dtype>(), back_inserter(idealLosses));
        ASSERT_TRUE(idealLosses.size() >= 3 * nEpoch);
        idealLosses.resize(3 * nEpoch);
    }
    {
        ifstream f(testPath / "unit_8_0" / "global_grad_norm.data");
        std::copy(istream_iterator<dtype>(f), istream_iterator<dtype>(), back_inserter(idealGradNorm));
        EXPECT_TRUE(idealGradNorm.size() >= 2 * nEpoch);
        idealGradNorm.resize(2 * nEpoch);
    }

    {
        ifstream f(testPath / "unit_8_0" / "learning_rate.data");
        std::copy(istream_iterator<dtype>(f), istream_iterator<dtype>(), back_inserter(idealLearningRate));
        EXPECT_TRUE(idealLearningRate.size() >= nEpoch);
        idealLearningRate.resize(nEpoch);
    }

    auto timeStart = chrono::steady_clock::now();
    TENSORS_CREATE(BATCH_SIZE);
    auto totalTime = TODTYPE(chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - timeStart).count());
    cout << "Workflow preparation time " << totalTime << " ms for batch size " << BATCH_SIZE << endl;
    MemoryManager& memory_manager = work.getMemoryManager();

    auto [TACOTRON_TRAINABLE_PARAMS, TACOTRON_PARAMS] = tacotronParamsCount(params);

    size_t loaded = loadTacotronParams((testPath / "weights" / "200000_Tacotron_model.").string(), memory_manager, "T", params, true, false);
    EXPECT_EQ(loaded, TACOTRON_PARAMS - 1); // -1 because speaker_embedding will be loaded separately and doesn't have Adam params
    postprocessing::GradientClipping clip(1_dt);

    // targets
    EXPECT_TRUE(loadTFData(testPath / "unit_8_0" / "stop_token_targets.data", memory_manager["stop_token_targets"]));
    EXPECT_TRUE(loadTFData(testPath / "unit_8_0" / "targets_lengths.data", memory_manager["targets_lengths"]));
    EXPECT_TRUE(loadTFData(testPath / "unit_8_0" / "mel_targets.data", memory_manager["mel_targets"]));

    // training data
    EXPECT_TRUE(loadTFData(testPath / "unit_8_0" / "input_lengths.data", memory_manager["input_lengths"]));
    EXPECT_TRUE(loadTFData(testPath / "unit_8_0" / "encoder_output.data", memory_manager["encoder_output"]));

    // speaker embedding
    EXPECT_TRUE(loadTFData(testPath / "unit_8_0" / "speaker_embedding.data", memory_manager[Name("T") / "speaker_embedding"]));

    auto optimizer = createOptimizer(trainParams, ADAM_BETA1, ADAM_BETA2, ADAM_EPSILON);

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
        totalTime = TODTYPE(chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - timeStart).count());
        cout << "  Epoch time " << totalTime << " ms for batch size " << BATCH_SIZE << endl;
        cout << "  Loss = " << lossValue[0] << " (before = " << memory_manager["before_loss"][0] << ", after = " << memory_manager["after_loss"][0]
             << ", stop_token = " << (params.withoutStopTokenLoss ? 0_dt : memory_manager["stop_token_loss"][0]) << ")" << endl;
        cout << "  Ideal loss = " << (idealLosses[3 * epoch] + idealLosses[3 * epoch + 1] + idealLosses[3 * epoch + 2]) << " (before = " << idealLosses[3 * epoch]
             << ", after = " << idealLosses[3 * epoch + 1] << ", stop_token = " << idealLosses[3 * epoch + 2] << ")" << endl;

        cout << "  Global grad norm: " << gnorm_before << " -> " << gnorm_after;
        cout << " (" << idealGradNorm[2 * epoch] << " -> " << idealGradNorm[2 * epoch + 1] << ")" << endl;

        cout << "  Learning rate: " << optimizer->getLearningRate() << " (" << idealLearningRate[epoch] << ")" << endl;
        EXPECT_NEAR(optimizer->getLearningRate(), idealLearningRate[epoch], LR_EPS);
    }
}

TEST(TestTacotronSpB, DecoderRNNResidualUnit)
{
    PROFILE_TEST

    // Test parameters
    constexpr auto eps_rel = 1e-4_dt;
    constexpr auto eps_abs = 1e-6_dt;
    constexpr auto input_size = 1280U;
    constexpr auto hidden_size = 512U;
    constexpr auto batch_size = 1U;

    constexpr size_t ITERS = 3;

    Workflow work;

    Names rnn_cell_state[ITERS];
    Names next_cell_state[ITERS];

    TacotronParams params({}, {}, {});
    params.dropoutRate = 0.f;
    params.postnetDropoutRate = 0.f;
    params.zoneoutRate = 0.f;
    params.useResidualRnn = true;

    vector<Names> hidden_golden(params.decoderLstmUnits.size());
    vector<Names> cell_golden(params.decoderLstmUnits.size());

    const size_t DECODER_RNN_TRAINABLE_PARAMS = params.decoderLstmUnits.size() * 2;

    auto testPath = tools::getTestAssetsDir() / "Tacotron_SpB";
    Names inputs;
    Names outputs;
    Names golden_outputs;
    for (size_t iter = 0; iter < ITERS; ++iter)
    {
        inputs.push_back("input_" + to_string(iter));
        outputs.push_back("outputs_" + to_string(iter));
        golden_outputs.push_back("golden_outputs_" + to_string(iter));
        for (size_t i = 0; i < params.decoderLstmUnits.size(); ++i)
        {
            auto suffix = "_" + to_string(iter) + "_" + to_string(i);
            rnn_cell_state[iter].push_back("hidden_golden" + suffix);
            rnn_cell_state[iter].push_back("cell_golden" + suffix);
            next_cell_state[iter].push_back("next_hidden" + suffix);
            next_cell_state[iter].push_back("next_cell" + suffix);

            hidden_golden[i].push_back("hidden_golden" + suffix);
            cell_golden[i].push_back("cell_golden" + suffix);

            work.add<DataLayer>("data" + suffix, DataParams{ { "hidden_golden" + suffix, "cell_golden" + suffix }, 1u, 1u, hidden_size });
        }
    }
    work.add<DataLayer>("input", DataParams{ inputs, 1u, 1u, input_size });
    work.add<DataLayer>("golden_output", DataParams{ golden_outputs, 1u, 1u, hidden_size });

    for (size_t iter = 0; iter < ITERS; ++iter)
    {
        Names decoder_inputs = rnn_cell_state[iter];
        decoder_inputs.insert(decoder_inputs.begin(), inputs[iter]);
        auto name = Name("T") / "decoder" / "_cell" / "decoder_LSTM";
        tacotron::AddDecoderRNN(&work, iter == 0 ? name : name / to_string(iter), { decoder_inputs, { outputs[iter] }, iter == 0 ? Name() : name }, next_cell_state[iter], params);
    }

    TENSORS_CREATE(batch_size);
    MemoryManager& memory_manager = work.getMemoryManager();

    // Data loading
    EXPECT_TRUE(loadTFData(testPath / "residual_decoder" / "decoder_rnn_input.data", memory_manager, inputs));
    [[maybe_unused]] auto& tt = memory_manager[inputs[0]];
    EXPECT_TRUE(loadTFData(testPath / "residual_decoder" / "decoder_rnn_output.data", memory_manager, golden_outputs));
    for (size_t i = 0; i < params.decoderLstmUnits.size(); ++i)
    {
        EXPECT_TRUE(loadTFData((testPath / "residual_decoder" / "decoder_rnn_input_h").string() + to_string(i) + ".data", memory_manager, hidden_golden[i]));
        EXPECT_TRUE(loadTFData((testPath / "residual_decoder" / "decoder_rnn_input_c").string() + to_string(i) + ".data", memory_manager, cell_golden[i]));
    }

    size_t loaded = loadTacotronParams((testPath / "weights" / "200000_Tacotron_model.").string(), memory_manager, "T", params, false);
    EXPECT_EQ(loaded, DECODER_RNN_TRAINABLE_PARAMS);
    EXPECT_EQ(work.getTrainableParameterNames().size(), DECODER_RNN_TRAINABLE_PARAMS);

    vector<pair<Name, Name>> checked;
    for (size_t iter = 1; iter < 3; ++iter)
    {
        checked.emplace_back(make_pair(outputs[iter], golden_outputs[iter]));
        if (iter > 0)
        {
            for (size_t j = 0; j < params.decoderLstmUnits.size(); ++j)
            {
                auto suffix1 = "_" + to_string(iter - 1) + "_" + to_string(j);
                auto suffix2 = "_" + to_string(iter) + "_" + to_string(j);
                checked.emplace_back(make_pair(Name("next_hidden" + suffix1), Name("hidden_golden" + suffix2)));
                checked.emplace_back(make_pair(Name("next_cell" + suffix1), Name("cell_golden" + suffix2)));
            }
        }
    }
    callbacks::TensorChecker checker(checked, eps_abs, eps_rel);
    work.getNetworkParameters().mCallback = checker;
    work.forwardPassTraining();
}

TEST(TestTacotronSpB, DynamicConvolutionAttentionUnit)
{
    PROFILE_TEST

    using namespace tacotron;

    Name attnParent = Name("T") / "decoder" / "_cell" / "attention_mechanism";
    Name attnChild = Name("T") / "decoder" / "_cell[1]" / "attention_mechanism";

    Workflow work;
    auto& networkParameters = work.getNetworkParameters();

    constexpr size_t BATCH_SIZE = 2u;
    constexpr size_t DECODER_OUTPUT_SIZE = 512u;
    constexpr size_t ALIGNMENTS_SIZE = 197u;
    constexpr size_t ENCODER_OUTPUT_SIZE = 768u;
    constexpr size_t ATTENTION_TRAINABLE_PARAMS = 10;
    constexpr dtype EPS = 1e-6_dt;

    TacotronParams params({ "encoder_output", "input_lengths", "mel_targets" }, { "decoder_output", "mel_output", "stop_token_prediction" }, {});
    params.attentionType = "DynamicConvolutional";

    // Query
    work.add<DataLayer>("data_query", DataParams{ { "query1", "query2" }, 1u, 1u, DECODER_OUTPUT_SIZE });
    // State
    work.add<DataLayer>("data_state", DataParams{ { "state1", "state2" }, 1u, 1u, ALIGNMENTS_SIZE });
    // Memory
    work.add<DataLayer>("data_memory", DataParams{ { "memory" }, 1u, ALIGNMENTS_SIZE, ENCODER_OUTPUT_SIZE });
    // Memory Seq Length
    work.add<DataLayer>("data_memory_seq_length", DataParams{ { "memory_seq_length" }, 1u, 1u, 1u });

    // Targets
    work.add<DataLayer>("data_attention_output_state", DataParams{ { "attention_output_state1", "attention_output_state2" }, 1u, 1u, ALIGNMENTS_SIZE });
    work.add<DataLayer>("data_attention_values", DataParams{ { "attention_values1", "attention_values2" }, 1u, ALIGNMENTS_SIZE, ENCODER_OUTPUT_SIZE });

    // Parameters for attention
    DynamicConvolutionAttentionParams attentionParams1 = {
        { "query1", "state1", "memory", "memory_seq_length" },
        { "attn1", "values" },
        params.attentionDim,
        DynamicConvolutionAttentionParams::hparams{
            params.attentionFilters, params.attentionKernelSize, params.attentionPriorFilterSize, params.attentionWindowSize, params.attentionPriorAlpha, params.attentionPriorBeta },
        false,
        false
    };

    DynamicConvolutionAttentionParams attentionParams2 = {
        { "query2", "state2", "memory", "memory_seq_length" },
        { "attn2", "max_ind2" },
        attnParent,
        params.attentionDim,
        DynamicConvolutionAttentionParams::hparams{
            params.attentionFilters, params.attentionKernelSize, params.attentionPriorFilterSize, params.attentionWindowSize, params.attentionPriorAlpha, params.attentionPriorBeta },
        false,
        false
    };

    // expected attn1 and attn2 shape is [BATCH_SIZE, 1, 1, ALIGNMENTS_SIZE]

    // Apply function
    DynamicConvolutionAttentionLayer(attnParent, attentionParams1, networkParameters);
    DynamicConvolutionAttentionLayer(attnChild, attentionParams2, networkParameters);

    work.add<TensorLayer>("grad1", TensorParams{ { Name("attn1").grad() }, WShape{ BS(), 1u, 1u, ALIGNMENTS_SIZE }, 1_dt, Workflow::Usage::ForwardAndBackward, Workflow::Mode::Read, true, true });
    work.add<TensorLayer>("grad2", TensorParams{ { Name("attn2").grad() }, WShape{ BS(), 1u, 1u, ALIGNMENTS_SIZE }, 1_dt, Workflow::Usage::ForwardAndBackward, Workflow::Mode::Read, true, true });

    work.preparePipelines();
    work.setBatchSize(BATCH_SIZE);
    work.prepareMemoryForTraining();

    MemoryManager& memory_manager = work.getMemoryManager();
    size_t loaded = loadTacotronParams((tools::getTestAssetsDir() / "Tacotron_SpB" / "weights" / "200000_Tacotron_model.").string(), memory_manager, "T", params, false, false);
    EXPECT_EQ(loaded, ATTENTION_TRAINABLE_PARAMS);

    // Load data
    EXPECT_TRUE(loadTFData<Tensor>(tools::getTestAssetsDir() / "Tacotron_SpB" / "dca" / "query.data", { &memory_manager["query1"], &memory_manager["query2"] }));
    EXPECT_TRUE(loadTFData<Tensor>(tools::getTestAssetsDir() / "Tacotron_SpB" / "dca" / "state.data", { &memory_manager["state1"], &memory_manager["state2"] }));
    EXPECT_TRUE(loadTFData<Tensor>(tools::getTestAssetsDir() / "Tacotron_SpB" / "dca" / "memory.data", { &memory_manager["memory"] }));
    EXPECT_TRUE(loadTFData<Tensor>(tools::getTestAssetsDir() / "Tacotron_SpB" / "dca" / "memory_sequence_length.data", { &memory_manager["memory_seq_length"] }));

    // Load real outputs
    EXPECT_TRUE(loadTFData<Tensor>(tools::getTestAssetsDir() / "Tacotron_SpB" / "dca" / "attention.data", { &memory_manager["attention_output_state1"], &memory_manager["attention_output_state2"] }));
    EXPECT_TRUE(loadTFData<Tensor>(tools::getTestAssetsDir() / "Tacotron_SpB" / "dca" / "values.data", { &memory_manager["attention_values1"], &memory_manager["attention_values2"] }));

    callbacks::TensorChecker checker({ { "attn1", "attention_output_state1" }, { "attn2", "attention_output_state2" }, { "values", "attention_values1" }, { "values", "attention_values2" } }, EPS);
    networkParameters.mCallback = checker;

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

TEST(TestTacotronSpB, BidirectionalLSTMDepthUnit)
{
    PROFILE_TEST

    using namespace tacotron;

    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    constexpr size_t BATCH_SIZE = 12u;
    constexpr size_t DEPTH = 86u;
    constexpr size_t INPUT_SIZE = 769u;
    constexpr size_t HIDDEN_SIZE = 256u;
    constexpr dtype EPS = 1e-4_dt;

    // Input
    work.add<DataLayer>("data_input", DataParams{ { "input" }, DEPTH, 1u, INPUT_SIZE });
    // Sequence length
    work.add<DataLayer>("data_length", DataParams{ { "length" }, 1u, 1u, 1u });
    // Real output
    work.add<DataLayer>("data_output", DataParams{ { "realOut" }, DEPTH, 1u, HIDDEN_SIZE * 2 });

    BidirectionalLSTMFunc("bidirectional_lstm", LSTMParams{ { "input", "length" }, { "out" }, HIDDEN_SIZE, true, false, false, 0.0_dt, true, 1.0_dt }, networkParameters);

    TENSORS_CREATE(BATCH_SIZE);
    // Load data
    EXPECT_TRUE(loadTFData(tools::getTestAssetsDir() / "Tacotron_SpB" / "bidirectional_lstm" / "rp_rnn_encoder_outputs.data", { &memory_manager["input"] }));
    EXPECT_TRUE(loadTFData(tools::getTestAssetsDir() / "Tacotron_SpB" / "bidirectional_lstm" / "rp_rnn_input_lengths.data", { &memory_manager["length"] }));
    DataLoader::loadData(tools::getTestAssetsDir() / "Tacotron_SpB" / "bidirectional_lstm" / "bidirectional_rnn.fw.range_predictor_fw_LSTM.kernel.data",
                         memory_manager["bidirectional_lstm::direct::cell::linear::Weights"]);
    DataLoader::loadData(tools::getTestAssetsDir() / "Tacotron_SpB" / "bidirectional_lstm" / "bidirectional_rnn.fw.range_predictor_fw_LSTM.bias.data",
                         memory_manager["bidirectional_lstm::direct::cell::linear::Biases"]);
    DataLoader::loadData(tools::getTestAssetsDir() / "Tacotron_SpB" / "bidirectional_lstm" / "bidirectional_rnn.bw.range_predictor_bw_LSTM.kernel.data",
                         memory_manager["bidirectional_lstm::reversed::cell::linear::Weights"]);
    DataLoader::loadData(tools::getTestAssetsDir() / "Tacotron_SpB" / "bidirectional_lstm" / "bidirectional_rnn.bw.range_predictor_bw_LSTM.bias.data",
                         memory_manager["bidirectional_lstm::reversed::cell::linear::Biases"]);

    // Load real output
    EXPECT_TRUE(loadTFData(tools::getTestAssetsDir() / "Tacotron_SpB" / "bidirectional_lstm" / "rp_rnn_outputs_after_concat.data", { &memory_manager["realOut"] }));

    ASSERT_NO_THROW(work.forwardPassTraining());

    const auto& out = memory_manager["out"];
    const auto& realOut = memory_manager["realOut"];
    for (size_t i = 0; i < out.size(); ++i)
    {
        EXPECT_NEAR(out[i], realOut[i], EPS);
    }
}

TEST(TestTacotronSpB, BidirectionalLSTMHeightUnit)
{
    PROFILE_TEST

    using namespace tacotron;

    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    constexpr size_t BATCH_SIZE = 12u;
    constexpr size_t HEIGHT = 86u;
    constexpr size_t INPUT_SIZE = 769u;
    constexpr size_t HIDDEN_SIZE = 256u;
    constexpr dtype EPS = 1e-4_dt;

    // Input
    work.add<DataLayer>("data_input", DataParams{ { "input" }, 1u, HEIGHT, INPUT_SIZE });
    // Sequence length
    work.add<DataLayer>("data_length", DataParams{ { "length" }, 1u, 1u, 1u });
    // Real output
    work.add<DataLayer>("data_output", DataParams{ { "realOut" }, 1u, HEIGHT, HIDDEN_SIZE * 2 });

    BidirectionalLSTMFunc("bidirectional_lstm", LSTMParams{ { "input", "length" }, { "out" }, HIDDEN_SIZE, true, false, false, 0.0_dt, true, 1.0_dt }, networkParameters);

    TENSORS_CREATE(BATCH_SIZE);
    // Load data
    EXPECT_TRUE(loadTFData(tools::getTestAssetsDir() / "Tacotron_SpB" / "bidirectional_lstm" / "rp_rnn_encoder_outputs.data", { &memory_manager["input"] }));
    EXPECT_TRUE(loadTFData(tools::getTestAssetsDir() / "Tacotron_SpB" / "bidirectional_lstm" / "rp_rnn_input_lengths.data", { &memory_manager["length"] }));
    DataLoader::loadData(tools::getTestAssetsDir() / "Tacotron_SpB" / "bidirectional_lstm" / "bidirectional_rnn.fw.range_predictor_fw_LSTM.kernel.data",
                         memory_manager["bidirectional_lstm::direct::cell::linear::Weights"]);
    DataLoader::loadData(tools::getTestAssetsDir() / "Tacotron_SpB" / "bidirectional_lstm" / "bidirectional_rnn.fw.range_predictor_fw_LSTM.bias.data",
                         memory_manager["bidirectional_lstm::direct::cell::linear::Biases"]);
    DataLoader::loadData(tools::getTestAssetsDir() / "Tacotron_SpB" / "bidirectional_lstm" / "bidirectional_rnn.bw.range_predictor_bw_LSTM.kernel.data",
                         memory_manager["bidirectional_lstm::reversed::cell::linear::Weights"]);
    DataLoader::loadData(tools::getTestAssetsDir() / "Tacotron_SpB" / "bidirectional_lstm" / "bidirectional_rnn.bw.range_predictor_bw_LSTM.bias.data",
                         memory_manager["bidirectional_lstm::reversed::cell::linear::Biases"]);

    // Load real output
    EXPECT_TRUE(loadTFData(tools::getTestAssetsDir() / "Tacotron_SpB" / "bidirectional_lstm" / "rp_rnn_outputs_after_concat.data", { &memory_manager["realOut"] }));

    ASSERT_NO_THROW(work.forwardPassTraining());

    const auto& out = memory_manager["out"];
    const auto& realOut = memory_manager["realOut"];
    for (size_t i = 0; i < out.size(); ++i)
    {
        EXPECT_NEAR(out[i], realOut[i], EPS);
    }
}

}
