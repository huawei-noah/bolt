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

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <set>

#include <tests/tools/TestTools.h>

#include <training/api/API.h>
#include <training/common/Common.h>
#include <training/common/MemoryManager.h>
#include <training/initializers/ConstantInitializer.h>
#include <training/layers/activations/LogSoftMaxActivation.h>
#include <training/layers/basic/DropoutLayer.h>
#include <training/layers/basic/trainable/LinearLayer.h>
#include <training/layers/composite/BERT.h>
#include <training/loss/NegativeLogLikelihoodLoss.h>
#include <training/network/Layers.h>
#include <training/network/Workflow.h>
#include <training/optimizers/Adam.h>
#include <training/tools/Profiler.h>

namespace UT
{

struct BertDatasetEntry
{
    raul::Tensor* input_ids;
    raul::Tensor* token_type_ids;
    raul::Tensor* attention_mask;
    raul::Tensor* label_ids;
    raul::Tensor* label; // one-hot encoded labels
};

typedef std::vector<BertDatasetEntry> BertDataset;

BertDataset readDataset(const std::string& path, size_t batch_size, size_t num_labels, raul::MemoryManager& mm, size_t maxBatches = 0)
{
    using namespace std;

    BertDataset dataset;

    ifstream f(path);
    size_t cnt, len;
    f >> cnt >> len;

    size_t batches = (cnt % batch_size == 0 ? cnt / batch_size : cnt / batch_size + 1);
    size_t N = batches;
    if (maxBatches > 0) N = std::min(N, maxBatches);

    for (size_t i = 0; i < N; ++i)
    {
        if (i == batches - 1 && cnt % batch_size != 0) batch_size = cnt % batch_size;

        BertDatasetEntry e{ mm.createTensor(batch_size, 1, 1, len),
                            mm.createTensor(batch_size, 1, 1, len),
                            mm.createTensor(batch_size, 1, 1, len),
                            mm.createTensor(batch_size, 1, 1, 1, 0._dt),
                            mm.createTensor(batch_size, 1, 1, num_labels, 0._dt) };

        raul::Tensor* tensors[] = { e.input_ids, e.token_type_ids, e.attention_mask, e.label_ids };
        for (size_t k = 0; k < batch_size; ++k)
        {
            for (size_t j = 0; j < 4; ++j)
            {
                auto w = tensors[j]->getWidth();
                for (size_t q = 0; q < w; ++q)
                    f >> (*tensors[j])[k * w + q];
            }
            size_t lbl = static_cast<size_t>((*e.label_ids)[k]);
            if (lbl >= num_labels) throw std::runtime_error("Bad label " + std::to_string(lbl) + "in " + path);
            (*e.label)[k * num_labels + lbl] = 1._dt;
        }
        dataset.push_back(e);
    }

    return dataset;
}

namespace
{
size_t load_weights(raul::MemoryManager& memory_manager,
                    const std::filesystem::path& weights_path,
                    const raul::Names& paramNames,
                    const std::string& file_prefix = "0_",
                    const std::string& tensor_prefix = "")
{
    using namespace std;

    raul::DataLoader dataLoader;
    // cout << "Loading trainableParams..." << endl;
    size_t count = 0;
    for (auto tensorName : paramNames)
    {
        tensorName = tensorName.str().substr(tensor_prefix.size());
        auto name = tensor_prefix + tensorName;
        // cout << tensorName << endl;

        if (memory_manager.tensorExists(name + "::Biases"))
        {
            memory_manager[name + "::Biases"] = dataLoader.loadData(weights_path / (file_prefix + tensorName.str() + ".bias.data"), memory_manager[name + "::Biases"].size(), 1, 1);
            ++count;
        }
        memory_manager[name + "::Weights"] = dataLoader.loadData(weights_path / (file_prefix + tensorName.str() + ".weight.data"), memory_manager[name + "::Weights"].size(), 1, 1);
        ++count;
    }

    // cout << "done" << endl;
    return count;
}
}

// corresponds to bert_simple.py test
TEST(TestBERT, BertForSequenceClassificationUnit)
{
    PROFILE_TEST

    using namespace raul;
    using namespace std;

    constexpr size_t BATCH_SIZE = 16;
    constexpr size_t TESTSET_SIZE = 872;
    constexpr size_t nBatchesTest = (TESTSET_SIZE % BATCH_SIZE == 0 ? TESTSET_SIZE / BATCH_SIZE : TESTSET_SIZE / BATCH_SIZE + 1);

    float ATTENTION_DROPOUT = 0.0f;
    string ACTIVATION = "gelu";
    float HIDDEN_DROPOUT = 0.0f;
    uint32_t HIDDEN_SIZE = 384;
    uint32_t INTERMEDIATE_SIZE = 1536;
    uint32_t MAX_POSITION_EMBEDDINGS = 512;
    uint32_t NUM_ATTENTION_HEADS = 12;
    uint32_t NUM_HIDDEN_LAYERS = 4;
    uint32_t TYPE_VOCAB_SIZE = 2;
    uint32_t VOCAB_SIZE = 30522;

    constexpr dtype ERROR_EPS = 1e-6_dt;

    size_t LENGTH = 64;
    size_t NUM_LABELS = 2;

    raul::Workflow work;
    raul::NetworkParameters& networkParameters = work.getNetworkParameters();
    raul::MemoryManager& memory_manager = work.getMemoryManager();
    networkParameters.mLossReductionCoefficient = BATCH_SIZE;

    const auto test_path = tools::getTestAssetsDir() / "BERT" / "dataset" / "test.data";
    ASSERT_TRUE(std::filesystem::exists(test_path));
    auto testDataset = readDataset(test_path.string(), BATCH_SIZE, NUM_LABELS, memory_manager);
    EXPECT_EQ(testDataset.size(), nBatchesTest);

    work.add<raul::DataLayer>("data1", raul::DataParams{ { "input_ids" }, 1, 1, LENGTH });
    work.add<raul::DataLayer>("data2", raul::DataParams{ { "token_type_ids" }, 1, 1, LENGTH });
    work.add<raul::DataLayer>("data3", raul::DataParams{ { "attention_mask" }, 1, 1, LENGTH });
    work.add<raul::DataLayer>("data4", raul::DataParams{ { "labels" }, 1, 1, NUM_LABELS });

    BERTParams params{ { "input_ids", "token_type_ids", "attention_mask" },
                       { "hidden_state", "pooled_output" },
                       VOCAB_SIZE,
                       TYPE_VOCAB_SIZE,
                       NUM_HIDDEN_LAYERS,
                       HIDDEN_SIZE,
                       INTERMEDIATE_SIZE,
                       NUM_ATTENTION_HEADS,
                       MAX_POSITION_EMBEDDINGS,
                       ACTIVATION,
                       HIDDEN_DROPOUT,
                       ATTENTION_DROPOUT };

    BERTModel("bert", params, networkParameters);

    string out_name = "pooled_output";

    work.add<LinearLayer>("classifier", LinearParams{ { out_name }, { "logits" }, NUM_LABELS });
    work.add<LogSoftMaxActivation>("softmax", BasicParamsWithDim{ { "logits" }, { "softmax" } });
    work.add<NLLLoss>("loss_fct", LossParams{ { "softmax", "labels" }, { "loss" }, "batch_mean" });

    work.preparePipelines();
    work.setBatchSize(BATCH_SIZE);
    work.prepareMemoryForTraining();

    initializers::ConstantInitializer(1._dt)(memory_manager["classifier::Weights"]);
    initializers::ConstantInitializer(0._dt)(memory_manager["classifier::Biases"]);

    set<string> paramsSet;
    auto model_params = work.getTrainableParameterNames();
    for (auto& p : model_params)
    {
        auto s = p;
        auto pos = s.str().find("::");
        if (pos != string::npos)
        {
            s = s.str().substr(0, pos);
        }
        paramsSet.insert(s);
    }
    paramsSet.erase("classifier");

    raul::Names paramNames(paramsSet.begin(), paramsSet.end());
    const auto weights_path = tools::getTestAssetsDir() / "BERT" / "pretrained";
    size_t tensorsLoaded = load_weights(memory_manager, weights_path, paramNames, "0_");
    EXPECT_EQ(tensorsLoaded, 71u);

    // only 1 batch of testset

    memory_manager["input_ids"] = TORANGE(*(testDataset[0].input_ids));
    memory_manager["token_type_ids"] = TORANGE(*(testDataset[0].token_type_ids));
    memory_manager["attention_mask"] = TORANGE(*(testDataset[0].attention_mask));
    memory_manager["labels"] = TORANGE(*(testDataset[0].label));

    auto timeStart = chrono::steady_clock::now();

    work.forwardPassTraining();

    auto loss_value = memory_manager["loss"][0];

    EXPECT_NEAR(loss_value, 0.693147_dt, ERROR_EPS);

    work.backwardPassTraining();

    auto totalTime = TODTYPE(chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - timeStart).count());
    std::cout << "Iteration time " << totalTime << " ms for batch size " << BATCH_SIZE << std::endl;
}

raul::dtype flat_accuracy(raul::Tensor& logits, raul::Tensor& labels)
{
    size_t N = logits.getBatchSize();
    auto logits2d = logits.reshape(yato::dims(N, logits.getWidth()));
    size_t guesses = 0;
    for (size_t i = 0; i < N; ++i)
    {
        size_t argmax = std::max_element(logits2d[i].begin(), logits2d[i].end()) - logits2d[i].begin();
        size_t label = static_cast<size_t>(labels[i]);
        if (argmax == label) ++guesses;
    }

    return static_cast<raul::dtype>(guesses) / static_cast<raul::dtype>(N);
}

// corresponds to bert.py test
TEST(TestBERT, BertForSequenceClassificationTraining)
{
    PROFILE_TEST
    using namespace raul;
    using namespace std;

    constexpr size_t BATCH_SIZE = 32;
    constexpr size_t TEST_BATCH_SIZE = 8;
    constexpr size_t TESTSET_SIZE = 872;
    constexpr size_t TRAINSET_SIZE = 67349;
    constexpr size_t LAST_BATCH = TRAINSET_SIZE % BATCH_SIZE;
    constexpr size_t nBatchesTrain = LAST_BATCH > 0 ? TRAINSET_SIZE / BATCH_SIZE + 1 : TRAINSET_SIZE / BATCH_SIZE;
    constexpr size_t nBatchesTest = TESTSET_SIZE / TEST_BATCH_SIZE;

    float ATTENTION_DROPOUT = 0.1f;
    string ACTIVATION = "gelu";
    float HIDDEN_DROPOUT = 0.1f;
    uint32_t HIDDEN_SIZE = 384;
    uint32_t INTERMEDIATE_SIZE = 1536;
    uint32_t MAX_POSITION_EMBEDDINGS = 512;
    uint32_t NUM_ATTENTION_HEADS = 12;
    uint32_t NUM_HIDDEN_LAYERS = 4;
    uint32_t TYPE_VOCAB_SIZE = 2;
    uint32_t VOCAB_SIZE = 30522;

    constexpr dtype ERROR_EPS = 5e-2_dt;

    constexpr size_t nEpoch = 0;
    constexpr size_t nEpochFineTune = 1;

    // Adam optimizer params
    constexpr dtype LEARNING_RATE = 2e-5_dt;

    size_t LENGTH = 64;
    size_t NUM_LABELS = 2;

    raul::Workflow work;
    raul::NetworkParameters& networkParameters = work.getNetworkParameters();
    raul::MemoryManager& memory_manager = work.getMemoryManager();

    const auto train_path = tools::getTestAssetsDir() / "BERT" / "dataset" / "train.data";
    const auto test_path = tools::getTestAssetsDir() / "BERT" / "dataset" / "test.data";

    ASSERT_TRUE(std::filesystem::exists(train_path));
    ASSERT_TRUE(std::filesystem::exists(test_path));

    MemoryManager datasetMM;
    auto testDataset = readDataset(test_path.string(), TEST_BATCH_SIZE, NUM_LABELS, datasetMM);
    EXPECT_EQ(testDataset.size(), nBatchesTest);
    auto trainDataset = readDataset(train_path.string(), BATCH_SIZE, NUM_LABELS, datasetMM);
    EXPECT_EQ(trainDataset.size(), nBatchesTrain);

    BERTParams params{ { "input_ids", "token_type_ids", "attention_mask" },
                       { "hidden_state", "pooled_output" },
                       VOCAB_SIZE,
                       TYPE_VOCAB_SIZE,
                       NUM_HIDDEN_LAYERS,
                       HIDDEN_SIZE,
                       INTERMEDIATE_SIZE,
                       NUM_ATTENTION_HEADS,
                       MAX_POSITION_EMBEDDINGS,
                       ACTIVATION,
                       HIDDEN_DROPOUT,
                       ATTENTION_DROPOUT };

    work.add<raul::DataLayer>("data1", raul::DataParams{ { "input_ids" }, 1, 1, LENGTH });
    work.add<raul::DataLayer>("data2", raul::DataParams{ { "token_type_ids" }, 1, 1, LENGTH });
    work.add<raul::DataLayer>("data3", raul::DataParams{ { "attention_mask" }, 1, 1, LENGTH });
    work.add<raul::DataLayer>("data4", raul::DataParams{ { "labels" }, 1, 1, NUM_LABELS });

    BERTModel("bert", params, networkParameters);

    string out_name = "pooled_output";
    if (params.hiddenDropout > 0)
    {
        work.add<DropoutLayer>("dropout", DropoutParams{ { "pooled_output" }, { "pooled_output_do" }, params.hiddenDropout });
        out_name = "pooled_output_do";
    }

    work.add<LinearLayer>("classifier", LinearParams{ { out_name }, { "logits" }, NUM_LABELS });
    work.add<LogSoftMaxActivation>("softmax", BasicParamsWithDim{ { "logits" }, { "softmax" } });
    work.add<NLLLoss>("loss_fct", LossParams{ { "softmax", "labels" }, { "loss" }, "mean" });

    work.preparePipelines();
    work.setBatchSize(TEST_BATCH_SIZE);
    work.prepareMemoryForTraining();

    vector<ParamAndGrad> trainableParams = work.getTrainableParameters();

    set<string> paramsSet;
    auto model_params = work.getTrainableParameterNames();
    for (auto& p : model_params)
    {
        auto s = p;
        auto pos = s.str().find("::");
        if (pos != string::npos)
        {
            s = s.str().substr(0, pos);
        }
        paramsSet.insert(s);
    }

    raul::Names paramNames(paramsSet.begin(), paramsSet.end());
    const auto weights_path = tools::getTestAssetsDir() / "BERT" / "pretrained";
    size_t tensorsLoaded = load_weights(memory_manager, weights_path, paramNames, to_string(nEpoch) + "_");
    EXPECT_EQ(tensorsLoaded, 73u);

    dtype total_eval_accuracy = 0_dt;
    for (size_t q = 0; q < nBatchesTest; ++q)
    {
        memory_manager["input_ids"] = TORANGE(*(testDataset[q].input_ids));
        memory_manager["token_type_ids"] = TORANGE(*(testDataset[q].token_type_ids));
        memory_manager["attention_mask"] = TORANGE(*(testDataset[q].attention_mask));
        memory_manager["labels"] = TORANGE(*(testDataset[q].label));

        work.forwardPassTesting();

        total_eval_accuracy += flat_accuracy(memory_manager["softmax"], *testDataset[q].label_ids);
        cout << "Test batch " << q << " / " << nBatchesTest << " acc = " << total_eval_accuracy / TODTYPE(q + 1) << endl;
    }
    total_eval_accuracy /= TODTYPE(nBatchesTest);
    cout << "Accuracy before epoch " << nEpoch << ": " << total_eval_accuracy << endl;

    auto adam = std::make_shared<optimizers::Adam>(LEARNING_RATE);

    for (size_t epoch = nEpoch; epoch < nEpoch + nEpochFineTune; ++epoch)
    {
        work.setBatchSize(BATCH_SIZE);

        cout << "Epoch " << epoch << endl;
        dtype total_loss = 0_dt;
        auto timeStart = chrono::steady_clock::now();

        for (size_t batch = 0; batch < nBatchesTrain; ++batch)
        {
            if (batch == nBatchesTrain - 1 && LAST_BATCH > 0)
            {
                work.setBatchSize(LAST_BATCH);
            }

            memory_manager["input_ids"] = TORANGE(*(trainDataset[batch].input_ids));
            memory_manager["token_type_ids"] = TORANGE(*(trainDataset[batch].token_type_ids));
            memory_manager["attention_mask"] = TORANGE(*(trainDataset[batch].attention_mask));
            memory_manager["labels"] = TORANGE(*(trainDataset[batch].label));

            work.forwardPassTraining();

            auto loss_value = memory_manager["loss"][0];
            total_loss += loss_value;

            work.backwardPassTraining();

            trainableParams = work.getTrainableParameters();

            for (auto p : trainableParams)
            {
                adam->operator()(memory_manager, p.Param, p.Gradient);
            }

            if (batch > 0 && batch % 40 == 0)
            {
                auto totalTime = TODTYPE(chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - timeStart).count());
                cout << "  Batch " << batch << " of " << nBatchesTrain << ". Loss " << loss_value << " Elapsed " << totalTime << endl;
            }
        }
        auto epochTime = TODTYPE(chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - timeStart).count());
        // calc accuracy on testset
        work.setBatchSize(TEST_BATCH_SIZE);

        total_eval_accuracy = 0_dt;
        for (size_t q = 0; q < nBatchesTest; ++q)
        {
            memory_manager["input_ids"] = TORANGE(*(testDataset[q].input_ids));
            memory_manager["token_type_ids"] = TORANGE(*(testDataset[q].token_type_ids));
            memory_manager["attention_mask"] = TORANGE(*(testDataset[q].attention_mask));
            memory_manager["labels"] = TORANGE(*(testDataset[q].label));

            work.forwardPassTesting();

            total_eval_accuracy += flat_accuracy(memory_manager["softmax"], *testDataset[q].label_ids);
        }
        total_eval_accuracy /= TODTYPE(nBatchesTest);
        cout << "  Accuracy " << total_eval_accuracy;

        cout << "  Epoch time " << epochTime << endl;
    }

    EXPECT_NEAR(total_eval_accuracy, 0.87_dt, ERROR_EPS);
}

// corresponds to bert.py test
TEST(TestBERT, DISABLED_BertForSequenceClassificationFineTuning)
{
    PROFILE_TEST
    using namespace raul;
    using namespace std;

    constexpr size_t BATCH_SIZE = 32;
    constexpr size_t TEST_BATCH_SIZE = 8;
    constexpr size_t TESTSET_SIZE = 872;
    constexpr size_t TRAINSET_SIZE = 67349;
    constexpr size_t LAST_BATCH = TRAINSET_SIZE % BATCH_SIZE;
    constexpr size_t nBatchesTrain = LAST_BATCH > 0 ? TRAINSET_SIZE / BATCH_SIZE + 1 : TRAINSET_SIZE / BATCH_SIZE;
    constexpr size_t nBatchesTest = TESTSET_SIZE / TEST_BATCH_SIZE;

    float ATTENTION_DROPOUT = 0.1f;
    string ACTIVATION = "gelu";
    float HIDDEN_DROPOUT = 0.1f;
    uint32_t HIDDEN_SIZE = 384;
    uint32_t INTERMEDIATE_SIZE = 1536;
    uint32_t MAX_POSITION_EMBEDDINGS = 512;
    uint32_t NUM_ATTENTION_HEADS = 12;
    uint32_t NUM_HIDDEN_LAYERS = 4;
    uint32_t TYPE_VOCAB_SIZE = 2;
    uint32_t VOCAB_SIZE = 30522;

    [[maybe_unused]] constexpr dtype ERROR_EPS = 1e-4_dt;

    constexpr size_t nEpoch = 1;
    constexpr size_t nEpochFineTune = 1;

    // Adam optimizer params
    constexpr dtype LEARNING_RATE = 2e-5_dt;

    size_t LENGTH = 64;
    size_t NUM_LABELS = 2;

    raul::Workflow work;
    raul::NetworkParameters& networkParameters = work.getNetworkParameters();
    raul::MemoryManager& memory_manager = work.getMemoryManager();

    const auto train_path = tools::getTestAssetsDir() / "BERT" / "dataset" / "train.data";
    const auto test_path = tools::getTestAssetsDir() / "BERT" / "dataset" / "test.data";

    ASSERT_TRUE(std::filesystem::exists(train_path));
    ASSERT_TRUE(std::filesystem::exists(test_path));

    MemoryManager datasetMM;
    auto testDataset = readDataset(test_path.string(), TEST_BATCH_SIZE, NUM_LABELS, datasetMM);
    EXPECT_EQ(testDataset.size(), nBatchesTest);
    auto trainDataset = readDataset(train_path.string(), BATCH_SIZE, NUM_LABELS, datasetMM);
    EXPECT_EQ(trainDataset.size(), nBatchesTrain);

    BERTParams params{ { "input_ids", "token_type_ids", "attention_mask" },
                       { "hidden_state", "pooled_output" },
                       VOCAB_SIZE,
                       TYPE_VOCAB_SIZE,
                       NUM_HIDDEN_LAYERS,
                       HIDDEN_SIZE,
                       INTERMEDIATE_SIZE,
                       NUM_ATTENTION_HEADS,
                       MAX_POSITION_EMBEDDINGS,
                       ACTIVATION,
                       HIDDEN_DROPOUT,
                       ATTENTION_DROPOUT };

    work.add<raul::DataLayer>("data1", raul::DataParams{ { "input_ids" }, 1, 1, LENGTH });
    work.add<raul::DataLayer>("data2", raul::DataParams{ { "token_type_ids" }, 1, 1, LENGTH });
    work.add<raul::DataLayer>("data3", raul::DataParams{ { "attention_mask" }, 1, 1, LENGTH });
    work.add<raul::DataLayer>("data4", raul::DataParams{ { "labels" }, 1, 1, NUM_LABELS });

    BERTModel("bert", params, networkParameters);

    string out_name = "pooled_output";
    if (params.hiddenDropout > 0)
    {
        work.add<DropoutLayer>("dropout", DropoutParams{ { "pooled_output" }, { "pooled_output_do" }, params.hiddenDropout });
        out_name = "pooled_output_do";
    }
    work.add<LinearLayer>("classifier", LinearParams{ { out_name }, { "logits" }, NUM_LABELS });
    work.add<LogSoftMaxActivation>("softmax", BasicParamsWithDim{ { "logits" }, { "softmax" } });
    work.add<NLLLoss>("loss_fct", LossParams{ { "softmax", "labels" }, { "loss" }, "mean" });

    work.preparePipelines();
    work.setBatchSize(TEST_BATCH_SIZE);
    work.prepareMemoryForTraining();

    vector<ParamAndGrad> trainableParams = work.getTrainableParameters();

    set<string> paramsSet;
    auto model_params = work.getTrainableParameterNames();
    for (auto& p : model_params)
    {
        auto s = p;
        auto pos = s.str().find("::");
        if (pos != string::npos)
        {
            s = s.str().substr(0, pos);
        }
        paramsSet.insert(s);
    }

    raul::Names paramNames(paramsSet.begin(), paramsSet.end());
    const auto weights_path = tools::getTestAssetsDir() / "BERT" / "pretrained";
    size_t tensorsLoaded = load_weights(memory_manager, weights_path, paramNames, to_string(nEpoch) + "_");
    EXPECT_EQ(tensorsLoaded, 73u);

    dtype total_eval_accuracy = 0_dt;
    for (size_t q = 0; q < nBatchesTest; ++q)
    {
        memory_manager["input_ids"] = TORANGE(*(testDataset[q].input_ids));
        memory_manager["token_type_ids"] = TORANGE(*(testDataset[q].token_type_ids));
        memory_manager["attention_mask"] = TORANGE(*(testDataset[q].attention_mask));
        memory_manager["labels"] = TORANGE(*(testDataset[q].label));

        work.forwardPassTesting();

        total_eval_accuracy += flat_accuracy(memory_manager["logits"], *testDataset[q].label_ids);
        cout << "Test batch " << q << " / " << nBatchesTest << " acc = " << total_eval_accuracy / TODTYPE(q + 1) << endl;
    }
    total_eval_accuracy /= TODTYPE(nBatchesTest);
    cout << "Accuracy before epoch " << nEpoch << ": " << total_eval_accuracy << endl;

    auto adam = std::make_shared<optimizers::Adam>(LEARNING_RATE);

    for (size_t epoch = nEpoch; epoch < nEpoch + nEpochFineTune; ++epoch)
    {
        work.setBatchSize(BATCH_SIZE);

        cout << "Epoch " << epoch << endl;
        dtype total_loss = 0_dt;
        auto timeStart = chrono::steady_clock::now();

        for (size_t batch = 0; batch < nBatchesTrain; ++batch)
        {
            if (batch == nBatchesTrain - 1 && LAST_BATCH > 0)
            {
                work.setBatchSize(LAST_BATCH);
            }

            memory_manager["input_ids"] = TORANGE(*(trainDataset[batch].input_ids));
            memory_manager["token_type_ids"] = TORANGE(*(trainDataset[batch].token_type_ids));
            memory_manager["attention_mask"] = TORANGE(*(trainDataset[batch].attention_mask));
            memory_manager["labels"] = TORANGE(*(trainDataset[batch].label));

            work.forwardPassTraining();

            auto loss_value = memory_manager["loss"][0];
            total_loss += loss_value;

            work.backwardPassTraining();

            trainableParams = work.getTrainableParameters();

            for (auto p : trainableParams)
                adam->operator()(memory_manager, p.Param, p.Gradient);

            if (batch > 0 && batch % 40 == 0)
            {
                auto totalTime = TODTYPE(chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - timeStart).count());
                cout << "  Batch " << batch << " of " << nBatchesTrain << ". Loss " << loss_value << " Elapsed " << totalTime << endl;
            }
        }
        auto epochTime = TODTYPE(chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - timeStart).count());
        // calc accuracy on testset
        work.setBatchSize(TEST_BATCH_SIZE);

        total_eval_accuracy = 0_dt;
        for (size_t q = 0; q < nBatchesTest; ++q)
        {
            memory_manager["input_ids"] = TORANGE(*(testDataset[q].input_ids));
            memory_manager["token_type_ids"] = TORANGE(*(testDataset[q].token_type_ids));
            memory_manager["attention_mask"] = TORANGE(*(testDataset[q].attention_mask));
            memory_manager["labels"] = TORANGE(*(testDataset[q].label));

            work.forwardPassTesting();

            total_eval_accuracy += flat_accuracy(memory_manager["logits"], *testDataset[q].label_ids);
        }
        total_eval_accuracy /= TODTYPE(nBatchesTest);
        cout << "  Accuracy " << total_eval_accuracy;

        cout << "  Epoch time " << epochTime << endl;
    }
}

// corresponds to bert.py test
TEST(TestBERT, BertForSequenceClassificationMemoryTest)
{
    PROFILE_TEST
    using namespace raul;
    using namespace std;

    constexpr size_t BATCH_SIZE = 16;

    float ATTENTION_DROPOUT = 0.1f;
    string ACTIVATION = "gelu";
    float HIDDEN_DROPOUT = 0.1f;
    uint32_t HIDDEN_SIZE = 384;
    uint32_t INTERMEDIATE_SIZE = 1536;
    uint32_t MAX_POSITION_EMBEDDINGS = 512;
    uint32_t NUM_ATTENTION_HEADS = 12;
    uint32_t NUM_HIDDEN_LAYERS = 4;
    uint32_t TYPE_VOCAB_SIZE = 2;
    uint32_t VOCAB_SIZE = 30522;

    auto COMPRESSION = CompressionMode::FP16;

    [[maybe_unused]] constexpr dtype ERROR_EPS = 1e-4_dt;

    size_t LENGTH = 64;
    size_t NUM_LABELS = 2;

    raul::Workflow work(COMPRESSION, CalculationMode::DETERMINISTIC);
    raul::MemoryManager& memory_manager = work.getMemoryManager();
    raul::NetworkParameters& networkParameters = work.getNetworkParameters();

    const auto test_path = tools::getTestAssetsDir() / "BERT" / "dataset" / "test.data";
    ASSERT_TRUE(std::filesystem::exists(test_path));

    auto dataset = readDataset(test_path.string(), BATCH_SIZE, NUM_LABELS, memory_manager, 1u);

    BERTParams params{ { "input_ids", "token_type_ids", "attention_mask" },
                       { "hidden_state", "pooled_output" },
                       VOCAB_SIZE,
                       TYPE_VOCAB_SIZE,
                       NUM_HIDDEN_LAYERS,
                       HIDDEN_SIZE,
                       INTERMEDIATE_SIZE,
                       NUM_ATTENTION_HEADS,
                       MAX_POSITION_EMBEDDINGS,
                       ACTIVATION,
                       HIDDEN_DROPOUT,
                       ATTENTION_DROPOUT };

    work.add<raul::DataLayer>("data1", raul::DataParams{ { "input_ids" }, 1, 1, LENGTH });
    work.add<raul::DataLayer>("data2", raul::DataParams{ { "token_type_ids" }, 1, 1, LENGTH });
    work.add<raul::DataLayer>("data3", raul::DataParams{ { "attention_mask" }, 1, 1, LENGTH });
    work.add<raul::DataLayer>("data4", raul::DataParams{ { "labels" }, 1, 1, NUM_LABELS });

    BERTModel("bert", params, networkParameters);
    string out_name = "pooled_output";
    if (params.hiddenDropout > 0)
    {
        work.add<DropoutLayer>("dropout", DropoutParams{ { "pooled_output" }, { "pooled_output_do" }, params.hiddenDropout });
        out_name = "pooled_output_do";
    }
    work.add<LinearLayer>("classifier", LinearParams{ { out_name }, { "logits" }, NUM_LABELS });
    work.add<LogSoftMaxActivation>("softmax", BasicParamsWithDim{ { "logits" }, { "softmax" }, "width" });
    work.add<NLLLoss>("loss_fct", LossParams{ { "softmax", "labels" }, { "loss" }, "mean" });

    work.preparePipelines();
    work.setBatchSize(BATCH_SIZE);
    work.prepareMemoryForTraining();

    initializers::ConstantInitializer(1._dt)(memory_manager["classifier::Weights"]);
    initializers::ConstantInitializer(0._dt)(memory_manager["classifier::Biases"]);

    vector<ParamAndGrad> trainableParams = work.getTrainableParameters();

    size_t K = 0;
    for (auto p : trainableParams)
    {
        K += p.Param.size();
    }
    cout << "Trainable params: " << K << endl;

    auto adam = std::make_shared<optimizers::Adam>(2e-5_dt);
    for (size_t ii = 0; ii < 5; ++ii)
    {
        cout << "Epoch " << ii << ". Batch size " << BATCH_SIZE << endl;
        memory_manager["input_ids"] = TORANGE(*(dataset[0].input_ids));
        memory_manager["token_type_ids"] = TORANGE(*(dataset[0].token_type_ids));
        memory_manager["attention_mask"] = TORANGE(*(dataset[0].attention_mask));
        memory_manager["labels"] = TORANGE(*(dataset[0].label));

        EXPECT_NO_THROW(work.forwardPassTraining());

        EXPECT_NO_THROW(work.backwardPassTraining());

        trainableParams = work.getTrainableParameters();

        /*for (auto p : trainableParams)
        {
            adam->operator()(memory_manager, p.Param, p.Gradient);
        }*/
    }
}

}
