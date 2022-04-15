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

#include <training/api/API.h>
#include <training/common/Train.h>
#include <training/initializers/ConstantInitializer.h>
#include <training/network/Layers.h>

#include <tests/tools/TestTools.h>
using namespace raul;

namespace
{

class TrainableLayerStub : public TrainableLayer
{
  public:
    static std::map<std::string, std::vector<std::unique_ptr<Tensor>>> InputData;
    static std::map<std::string, std::unique_ptr<Tensor>> GradientCopies;
    static std::map<std::string, size_t> CallForwardInTestModeCounter;
    static std::map<std::string, size_t> CallForwardInTrainModeCounter;
    static std::map<std::string, size_t> CallBackwardCounter;

  public:
    static std::unique_ptr<Tensor> averageGradientOf(const std::string& layerName, const std::string& parametersName)
    {
        const Tensor& gradient = *GradientCopies[layerName + "::" + parametersName + "Gradient"];
        std::unique_ptr<Tensor> result = std::make_unique<Tensor>(gradient.getBatchSize(), gradient.getDepth(), gradient.getHeight(), gradient.getWidth());
        std::transform(gradient.begin(), gradient.end(), result->begin(), [&layerName](dtype v) { return (v / static_cast<dtype>(TrainableLayerStub::CallBackwardCounter[layerName])); });

        return result;
    }

  public:
    TrainableLayerStub(const Name& name, const TrainableParams& layerParameters, NetworkParameters& networkParameters)
        : TrainableLayer(name, "TrainableLayerStub", layerParameters, networkParameters)
    {
        InputData[mName].clear();
        CallForwardInTrainModeCounter[mName] = 0;
        CallForwardInTestModeCounter[mName] = 0;
        CallBackwardCounter[mName] = 0;

        mNetworkParams.mWorkflow.copyDeclaration(name, mInputs[0], raul::Workflow::Usage::Forward, raul::Workflow::Mode::Read);
        mNetworkParams.mWorkflow.copyDeclaration(mName, mInputs[0], mOutputs[0], DEC_FORW_WRIT_NOMEMOPT);

        mNetworkParams.mWorkflow.tensorNeeded(mName, mBiasesName, WShape{ 1u, 1u, 1u, mNetworkParams.mWorkflow.getWidth(mInputs[0]) }, DEC_TRAINABLE);
        mNetworkParams.mWorkflow.copyDeclaration(mName, mBiasesName, mBiasesName.grad(), DEC_TRAINABLE_GRAD);

        mNetworkParams.mWorkflow.tensorNeeded(mName, mWeightsName, WShape{ 1u, 1u, mNetworkParams.mWorkflow.getHeight(mInputs[0]), mNetworkParams.mWorkflow.getWidth(mInputs[0]) }, DEC_TRAINABLE);
        mNetworkParams.mWorkflow.copyDeclaration(mName, mWeightsName, mWeightsName.grad(), DEC_TRAINABLE_GRAD);

        GradientCopies[mWeightsName + "Gradient"] = std::make_unique<Tensor>(1, 1, mNetworkParams.mWorkflow.getHeight(mInputs[0]), mNetworkParams.mWorkflow.getWidth(mInputs[0]));
        GradientCopies[mBiasesName + "Gradient"] = std::make_unique<Tensor>(1, 1, 1, mNetworkParams.mWorkflow.getWidth(mInputs[0]));
    }

    void forwardComputeImpl(NetworkMode mode) final
    {
        if (mode == NetworkMode::Train)
        {
            ++CallForwardInTrainModeCounter[mName];
            const Tensor& input = mNetworkParams.mMemoryManager[mInputs[0]];
            auto inputCopy = std::make_unique<Tensor>(input.getBatchSize(), input.getDepth(), input.getHeight(), input.getWidth());
            std::copy(input.begin(), input.end(), inputCopy->begin());
            InputData[mName].emplace_back(std::move(inputCopy));
        }
        else
        {
            ++CallForwardInTestModeCounter[mName];
        }
    }
    void backwardComputeImpl() final
    {
        std::random_device rd;
        std::mt19937 g(rd());
        std::uniform_int_distribution<uint16_t> distrib(1, 10);
        ++CallBackwardCounter[mName];
        auto& weightsGradient = mNetworkParams.mMemoryManager[mWeightsName + "Gradient"];
        std::transform(weightsGradient.begin(), weightsGradient.end(), weightsGradient.begin(), [&distrib, &g](dtype v) { return v + static_cast<dtype>(distrib(g)); });
        std::copy(weightsGradient.begin(), weightsGradient.end(), GradientCopies[mWeightsName + "Gradient"]->begin());
        auto& biasesGradient = mNetworkParams.mMemoryManager[mBiasesName + "Gradient"];
        std::transform(biasesGradient.begin(), biasesGradient.end(), biasesGradient.begin(), [&distrib, &g](dtype v) { return v + static_cast<dtype>(distrib(g)); });
        std::copy(biasesGradient.begin(), biasesGradient.end(), GradientCopies[mBiasesName + "Gradient"]->begin());
    }

  private:
};
std::map<std::string, std::vector<std::unique_ptr<Tensor>>> TrainableLayerStub::InputData;
std::map<std::string, std::unique_ptr<Tensor>> TrainableLayerStub::GradientCopies;
std::map<std::string, size_t> TrainableLayerStub::CallForwardInTestModeCounter;
std::map<std::string, size_t> TrainableLayerStub::CallForwardInTrainModeCounter;
std::map<std::string, size_t> TrainableLayerStub::CallBackwardCounter;

class LossStub : public BasicLayer
{
  public:
    static std::vector<dtype> LossAfterBackwardCall;
    static size_t CallForwardInTestModeCounter;
    static size_t CallForwardInTrainModeCounter;
    static size_t CallBackwardCounter;

  public:
    LossStub(const Name& name, const BasicParams& layerParameters, NetworkParameters& networkParameters)
        : BasicLayer(name, "LossStub", layerParameters, networkParameters)
    {
        LossAfterBackwardCall.clear();
        CallForwardInTrainModeCounter = 0;
        CallForwardInTestModeCounter = 0;
        CallBackwardCounter = 0;

        mNetworkParams.mWorkflow.copyDeclaration(name, mInputs[0], raul::Workflow::Usage::Forward, raul::Workflow::Mode::Read);
        mNetworkParams.mWorkflow.tensorNeeded(mName, mOutputs[0], raul::WShape{ 1u, 1u, 1u, 1u }, DEC_FORW_WRIT_NOMEMOPT);
    }

    void forwardComputeImpl(NetworkMode mode) final
    {
        if (mode == NetworkMode::Train)
        {
            ++CallForwardInTrainModeCounter;
            std::random_device rd;
            std::mt19937 g(rd());
            std::uniform_int_distribution<uint16_t> distrib(1, 10);
            LossAfterBackwardCall.push_back(distrib(g));
            mNetworkParams.mMemoryManager[mOutputs[0]][0] = LossAfterBackwardCall.back();
        }
        else
        {
            ++CallForwardInTestModeCounter;
        }
    }
    void backwardComputeImpl() final { ++CallBackwardCounter; }
};
std::vector<dtype> LossStub::LossAfterBackwardCall;
size_t LossStub::CallForwardInTestModeCounter = 0;
size_t LossStub::CallForwardInTrainModeCounter = 0;
size_t LossStub::CallBackwardCounter = 0;

struct OptimizerStub : public optimizers::Optimizer
{
    std::string mCallLog;
    std::map<std::string, std::unique_ptr<Tensor>> mGradientsCopy;

    void optimize(MemoryManager&, Tensor& params, const Tensor& gradients) final
    {
        appendCallLog(params, gradients);
        mGradientsCopy[gradients.getName()] = std::make_unique<Tensor>(gradients.getBatchSize(), gradients.getDepth(), gradients.getHeight(), gradients.getWidth());
        std::copy(gradients.begin(), gradients.end(), mGradientsCopy[gradients.getName()]->begin());
    }
    std::ostream& as_ostream(std::ostream& out) const final { return out; }

  private:
    void appendCallLog(Tensor& params, const Tensor& gradients)
    {
        mCallLog += "processed param: ";
        mCallLog += params.getName() + ", ";
        mCallLog += "processed gradient: ";
        mCallLog += gradients.getName() + ", ";
    }
};

class LoadingDataStub : public LoadData
{
  public:
    static std::map<std::string, std::vector<std::unique_ptr<Tensor>>> GeneratedData;

  public:
    explicit LoadingDataStub(const char* name)
        : mName(name)
    {
        GeneratedData[mName].clear();
    }

    std::unique_ptr<Tensor> operator()(size_t numberOfSamples, size_t sampleDepth, size_t sampleHeight, size_t sampleWidth) final
    {
        std::random_device rd;
        std::mt19937 g(rd());
        std::uniform_int_distribution<uint16_t> distrib(1, 10);
        auto result = std::make_unique<Tensor>(numberOfSamples, sampleDepth, sampleHeight, sampleWidth);
        std::generate(result->begin(), result->end(), [&distrib, &g]() { return static_cast<dtype>(distrib(g)); });
        auto resultCopy = std::make_unique<Tensor>(numberOfSamples, sampleDepth, sampleHeight, sampleWidth);
        std::copy(result->begin(), result->end(), resultCopy->begin());
        GeneratedData[mName].emplace_back(std::move(resultCopy));
        return result;
    }

    void prepareToReloadData() final {}

  private:
    std::string mName;
};
std::map<std::string, std::vector<std::unique_ptr<Tensor>>> LoadingDataStub::GeneratedData;

} // !namespace

struct TestTrain : testing::Test
{
    const char* nameOfFirstPartOfDataset = "dataset_part1";
    const char* nameOfSecondPartOfDataset = "dataset_part2";

    static constexpr size_t numberOfSamplesInDataset = 7;
    static constexpr size_t batchSize = 2;
    static constexpr size_t microbatchSize = 1;

    const char* nameOfFirstNetworkInput = "data1";
    static constexpr size_t firstNetInputDepth = 2;
    static constexpr size_t firstNetInputHeight = 2;
    static constexpr size_t firstNetInputWidth = 2;
    static constexpr size_t sizeOfFirstNetInput = batchSize * firstNetInputDepth * firstNetInputHeight * firstNetInputWidth;

    const char* nameOfSecondNetworkInput = "data2";
    static constexpr size_t secondNetInputDepth = 1;
    static constexpr size_t secondNetInputHeight = 1;
    static constexpr size_t secondNetInputWidth = 1;
    static constexpr size_t sizeOfSecondNetInput = batchSize * secondNetInputDepth * secondNetInputHeight * secondNetInputWidth;

    const char* nameOfLossTensor = "loss";

    std::unique_ptr<Dataset> datasetStub;
    std::unique_ptr<Workflow> networkStub;
    OptimizerStub optimizerStub;

    void SetUp() final
    {
        networkStub = std::make_unique<Workflow>();
        networkStub->add<raul::DataLayer>("in1", DataParams{ { nameOfFirstNetworkInput }, firstNetInputDepth, firstNetInputHeight, firstNetInputWidth });
        networkStub->add<raul::DataLayer>("in2", DataParams{ { nameOfSecondNetworkInput }, secondNetInputDepth, secondNetInputHeight, secondNetInputWidth });
        networkStub->add<TrainableLayerStub>("tl1", LinearParams{ { nameOfFirstNetworkInput }, { "first_network_input_copy" }, 0 });
        networkStub->add<TrainableLayerStub>("tl2", LinearParams{ { nameOfSecondNetworkInput }, { "second_network_input_copy" }, 0 });
        networkStub->add<LossStub>("loss", BasicParams{ { "first_network_input_copy", "second_network_input_copy" }, { nameOfLossTensor } });

        networkStub->preparePipelines();
        networkStub->setBatchSize(batchSize);
        networkStub->prepareMemoryForTraining();

        datasetStub = std::make_unique<Dataset>();
        datasetStub->describePart(nameOfFirstPartOfDataset, numberOfSamplesInDataset, firstNetInputDepth, firstNetInputHeight, firstNetInputWidth);
        datasetStub->describePart(nameOfSecondPartOfDataset, numberOfSamplesInDataset, secondNetInputDepth, secondNetInputHeight, secondNetInputWidth);
        datasetStub->setDataSourceFor(nameOfFirstPartOfDataset, std::make_unique<LoadingDataStub>(nameOfFirstPartOfDataset));
        datasetStub->setDataSourceFor(nameOfSecondPartOfDataset, std::make_unique<LoadingDataStub>(nameOfSecondPartOfDataset));
    }

    void TearDown() final {}
};

TEST_F(TestTrain, OneIterationUnit)
{
    PROFILE_TEST
    Train train(*networkStub, *datasetStub, { { { nameOfFirstPartOfDataset, nameOfFirstNetworkInput }, { nameOfSecondPartOfDataset, nameOfSecondNetworkInput } }, nameOfLossTensor });
    train.useBatches(batchSize);
    dtype lossAfterOneIteration = train.oneIteration(optimizerStub);

    ASSERT_EQ(TrainableLayerStub::CallForwardInTrainModeCounter["tl1"], 1u);
    ASSERT_EQ(TrainableLayerStub::CallForwardInTestModeCounter["tl1"], 0u);
    ASSERT_EQ(TrainableLayerStub::CallBackwardCounter["tl1"], 1u);
    ASSERT_EQ(TrainableLayerStub::CallForwardInTrainModeCounter["tl2"], 1u);
    ASSERT_EQ(TrainableLayerStub::CallForwardInTestModeCounter["tl2"], 0u);
    ASSERT_EQ(TrainableLayerStub::CallBackwardCounter["tl2"], 1u);
    ASSERT_EQ(LossStub::CallForwardInTrainModeCounter, 1u);
    ASSERT_EQ(LossStub::CallForwardInTestModeCounter, 0u);
    ASSERT_EQ(LossStub::CallBackwardCounter, 1u);
    ASSERT_EQ(networkStub->getBatchSize(), 2u);

    ASSERT_FLOAT_TENSORS_EQ((*LoadingDataStub::GeneratedData[nameOfFirstPartOfDataset][0]), (*TrainableLayerStub::InputData["tl1"][0]), 1e-6_dt);
    ASSERT_FLOAT_TENSORS_EQ((*LoadingDataStub::GeneratedData[nameOfSecondPartOfDataset][0]), (*TrainableLayerStub::InputData["tl2"][0]), 1e-6_dt);

    ASSERT_FLOAT_TENSORS_EQ((*optimizerStub.mGradientsCopy["tl1::WeightsGradient"]), (*TrainableLayerStub::GradientCopies["tl1::WeightsGradient"]), 1e-6_dt);
    ASSERT_FLOAT_TENSORS_EQ((*optimizerStub.mGradientsCopy["tl1::BiasesGradient"]), (*TrainableLayerStub::GradientCopies["tl1::BiasesGradient"]), 1e-6_dt);
    ASSERT_FLOAT_TENSORS_EQ((*optimizerStub.mGradientsCopy["tl2::WeightsGradient"]), (*TrainableLayerStub::GradientCopies["tl2::WeightsGradient"]), 1e-6_dt);
    ASSERT_FLOAT_TENSORS_EQ((*optimizerStub.mGradientsCopy["tl2::BiasesGradient"]), (*TrainableLayerStub::GradientCopies["tl2::BiasesGradient"]), 1e-6_dt);

    ASSERT_NEAR(lossAfterOneIteration, LossStub::LossAfterBackwardCall.back(), 1e-6_dt);

    ASSERT_STREQ(optimizerStub.mCallLog.c_str(),
                 "processed param: tl1::Biases, processed gradient: tl1::BiasesGradient, "
                 "processed param: tl1::Weights, processed gradient: tl1::WeightsGradient, "
                 "processed param: tl2::Biases, processed gradient: tl2::BiasesGradient, "
                 "processed param: tl2::Weights, processed gradient: tl2::WeightsGradient, ");
}

TEST_F(TestTrain, OneEpochUnit)
{
    PROFILE_TEST
    Train train(*networkStub, *datasetStub, { { { nameOfFirstPartOfDataset, nameOfFirstNetworkInput }, { nameOfSecondPartOfDataset, nameOfSecondNetworkInput } }, nameOfLossTensor });
    train.useBatches(batchSize);
    train.oneEpoch(optimizerStub, [lossCheckPointStep = 2](size_t iterNum, dtype loss) {
        if (iterNum % lossCheckPointStep)
        {
            ASSERT_NEAR(loss, LossStub::LossAfterBackwardCall.back(), 1e-6_dt);
        }
    });
}

TEST_F(TestTrain, OneEpochWithLastIncompleteBatchAccountUnit)
{
    PROFILE_TEST
    datasetStub->configure(Dataset::USE_LAST_INCOMPLETE_BATCH);

    Train train(*networkStub, *datasetStub, { { { nameOfFirstPartOfDataset, nameOfFirstNetworkInput }, { nameOfSecondPartOfDataset, nameOfSecondNetworkInput } }, nameOfLossTensor });
    train.useBatches(batchSize);
    train.oneEpoch(optimizerStub, [](size_t, dtype loss) { ASSERT_NEAR(loss, LossStub::LossAfterBackwardCall.back(), 1e-6_dt); });
    ASSERT_EQ(networkStub->getBatchSize(), 1u);
}

TEST_F(TestTrain, ThrowAnExceptionIfBatchSizeForDatasetOrBatchSizeForNetworkHaveBeenChangedOutside)
{
    PROFILE_TEST
    Train train1(*networkStub, *datasetStub, { { { nameOfFirstPartOfDataset, nameOfFirstNetworkInput }, { nameOfSecondPartOfDataset, nameOfSecondNetworkInput } }, nameOfLossTensor });
    train1.useBatches(batchSize);

    datasetStub->generate(4);
    ASSERT_THROW(train1.oneIteration(optimizerStub), std::logic_error);

    datasetStub->generate(batchSize);
    networkStub->setBatchSize(4);
    ASSERT_THROW(train1.oneIteration(optimizerStub), std::logic_error);

    Train train2(*networkStub, *datasetStub, { { { nameOfFirstPartOfDataset, nameOfFirstNetworkInput }, { nameOfSecondPartOfDataset, nameOfSecondNetworkInput } }, nameOfLossTensor });
    train2.useMicroBatches(batchSize, microbatchSize);

    networkStub->setBatchSize(batchSize);
    ASSERT_THROW(train2.oneIteration(optimizerStub), std::logic_error);
}

TEST_F(TestTrain, ThrowAnExceptionIfSizesOfMappedNetworkInputAndPartOfDatasetIsNotEqualUnit)
{
    PROFILE_TEST
    Dataset dataset;
    dataset.describePart(nameOfFirstPartOfDataset, numberOfSamplesInDataset, 10, 10, 10);
    dataset.describePart(nameOfSecondPartOfDataset, numberOfSamplesInDataset, 10, 10, 10);
    dataset.setDataSourceFor(nameOfFirstPartOfDataset, std::make_unique<LoadingDataStub>(nameOfFirstPartOfDataset));
    dataset.setDataSourceFor(nameOfSecondPartOfDataset, std::make_unique<LoadingDataStub>(nameOfSecondPartOfDataset));
    dataset.generate(batchSize);

    Train train(*networkStub, dataset, { { { nameOfFirstPartOfDataset, nameOfFirstNetworkInput }, { nameOfSecondPartOfDataset, nameOfSecondNetworkInput } }, nameOfLossTensor });
    train.useBatches(batchSize);

    ASSERT_THROW(train.oneIteration(optimizerStub), std::logic_error);
}

TEST_F(TestTrain, OneIterationUsingMicroBatchesUnit)
{
    PROFILE_TEST
    Train train(*networkStub, *datasetStub, { { { nameOfFirstPartOfDataset, nameOfFirstNetworkInput }, { nameOfSecondPartOfDataset, nameOfSecondNetworkInput } }, nameOfLossTensor });
    train.useMicroBatches(batchSize, microbatchSize);
    dtype lossAfterOneIteration = train.oneIteration(optimizerStub);

    ASSERT_EQ(TrainableLayerStub::CallForwardInTrainModeCounter["tl1"], 2u);
    ASSERT_EQ(TrainableLayerStub::CallForwardInTestModeCounter["tl1"], 0u);
    ASSERT_EQ(TrainableLayerStub::CallBackwardCounter["tl1"], 2u);
    ASSERT_EQ(TrainableLayerStub::CallForwardInTrainModeCounter["tl2"], 2u);
    ASSERT_EQ(TrainableLayerStub::CallForwardInTestModeCounter["tl2"], 0u);
    ASSERT_EQ(TrainableLayerStub::CallBackwardCounter["tl2"], 2u);
    ASSERT_EQ(LossStub::CallForwardInTrainModeCounter, 2u);
    ASSERT_EQ(LossStub::CallForwardInTestModeCounter, 0u);
    ASSERT_EQ(LossStub::CallBackwardCounter, 2u);
    ASSERT_EQ(networkStub->getBatchSize(), 1u);

    ASSERT_INTERVALS_NEAR(LoadingDataStub::GeneratedData[nameOfFirstPartOfDataset][0]->begin(),
                          (LoadingDataStub::GeneratedData[nameOfFirstPartOfDataset][0]->begin() + (microbatchSize * (sizeOfFirstNetInput / batchSize))),
                          TrainableLayerStub::InputData["tl1"][0]->begin(),
                          TrainableLayerStub::InputData["tl1"][0]->end(),
                          1e-6_dt);

    ASSERT_INTERVALS_NEAR((LoadingDataStub::GeneratedData[nameOfFirstPartOfDataset][0]->begin() + (microbatchSize * (sizeOfFirstNetInput / batchSize))),
                          LoadingDataStub::GeneratedData[nameOfFirstPartOfDataset][0]->end(),
                          TrainableLayerStub::InputData["tl1"][1]->begin(),
                          TrainableLayerStub::InputData["tl1"][1]->end(),
                          1e-6_dt);

    ASSERT_INTERVALS_NEAR(LoadingDataStub::GeneratedData[nameOfSecondPartOfDataset][0]->begin(),
                          (LoadingDataStub::GeneratedData[nameOfSecondPartOfDataset][0]->begin() + (microbatchSize * (sizeOfSecondNetInput / batchSize))),
                          TrainableLayerStub::InputData["tl2"][0]->begin(),
                          TrainableLayerStub::InputData["tl2"][0]->end(),
                          1e-6_dt);

    ASSERT_INTERVALS_NEAR((LoadingDataStub::GeneratedData[nameOfSecondPartOfDataset][0]->begin() + (microbatchSize * (sizeOfSecondNetInput / batchSize))),
                          LoadingDataStub::GeneratedData[nameOfSecondPartOfDataset][0]->end(),
                          TrainableLayerStub::InputData["tl2"][1]->begin(),
                          TrainableLayerStub::InputData["tl2"][1]->end(),
                          1e-6_dt);

    ASSERT_FLOAT_TENSORS_EQ((*optimizerStub.mGradientsCopy["tl1::WeightsGradient"]), (*TrainableLayerStub::GradientCopies["tl1::WeightsGradient"]), 1e-6_dt);
    ASSERT_FLOAT_TENSORS_EQ((*optimizerStub.mGradientsCopy["tl1::BiasesGradient"]), (*TrainableLayerStub::GradientCopies["tl1::BiasesGradient"]), 1e-6_dt);
    ASSERT_FLOAT_TENSORS_EQ((*optimizerStub.mGradientsCopy["tl2::WeightsGradient"]), (*TrainableLayerStub::GradientCopies["tl2::WeightsGradient"]), 1e-6_dt);
    ASSERT_FLOAT_TENSORS_EQ((*optimizerStub.mGradientsCopy["tl2::BiasesGradient"]), (*TrainableLayerStub::GradientCopies["tl2::BiasesGradient"]), 1e-6_dt);

    ASSERT_NEAR(lossAfterOneIteration, std::accumulate(LossStub::LossAfterBackwardCall.begin(), LossStub::LossAfterBackwardCall.end(), 0_dt), 1e-6_dt);

    ASSERT_STD_STR_EQ(optimizerStub.mCallLog,
                      "processed param: tl1::Biases, processed gradient: tl1::BiasesGradient, "
                      "processed param: tl1::Weights, processed gradient: tl1::WeightsGradient, "
                      "processed param: tl2::Biases, processed gradient: tl2::BiasesGradient, "
                      "processed param: tl2::Weights, processed gradient: tl2::WeightsGradient, ");
}

TEST_F(TestTrain, ThrowAnExceptionIfAnAttemptToSetIncorrectMicroBatchSizeDetectedUnit)
{
    PROFILE_TEST
    Train train(*networkStub, *datasetStub, { { { nameOfFirstPartOfDataset, nameOfFirstNetworkInput }, { nameOfSecondPartOfDataset, nameOfSecondNetworkInput } }, nameOfLossTensor });

    ASSERT_THROW(train.useMicroBatches(2u, 3u), std::logic_error);

    ASSERT_THROW(train.useMicroBatches(3u, 2u), std::logic_error);
}

TEST_F(TestTrain, ThrowAnExceptionIfTrainModeIsNotDefinedUnit)
{
    PROFILE_TEST
    Train train(*networkStub, *datasetStub, { { { nameOfFirstPartOfDataset, nameOfFirstNetworkInput }, { nameOfSecondPartOfDataset, nameOfSecondNetworkInput } }, nameOfLossTensor });
    ASSERT_THROW(train.oneIteration(optimizerStub), std::logic_error);
}

TEST_F(TestTrain, ThrowAnExceptionIfSomeoneTriesToRedefineTrainingModeUnit)
{
    PROFILE_TEST
    Train train1(*networkStub, *datasetStub, { { { nameOfFirstPartOfDataset, nameOfFirstNetworkInput }, { nameOfSecondPartOfDataset, nameOfSecondNetworkInput } }, nameOfLossTensor });
    train1.useBatches(batchSize);
    ASSERT_THROW(train1.useBatches(batchSize), std::logic_error);
    ASSERT_THROW(train1.useMicroBatches(batchSize, microbatchSize), std::logic_error);

    Train train2(*networkStub, *datasetStub, { { { nameOfFirstPartOfDataset, nameOfFirstNetworkInput }, { nameOfSecondPartOfDataset, nameOfSecondNetworkInput } }, nameOfLossTensor });
    train2.useMicroBatches(batchSize, microbatchSize);
    ASSERT_THROW(train2.useBatches(batchSize), std::logic_error);
    ASSERT_THROW(train2.useMicroBatches(batchSize, microbatchSize), std::logic_error);
}

#include <training/common/Test.h>
#include <training/optimizers/SGD.h>

using dvec = std::vector<dtype>;

struct DifferentLossesAndReductionTypes : testing::TestWithParam<std::tuple<std::string, bool, std::string, dvec, dtype, dtype, dtype>>
{
    size_t BatchSize = 100;
    size_t MicroBatchSize = 10;
    uint32_t lossCheckStep = 100;
    uint32_t numberOfClasses = 10;

    const std::string& LossType = std::get<0>(GetParam());
    bool UseWeightsForLoss = std::get<1>(GetParam());
    const std::string& ReductionType = std::get<2>(GetParam());
    const dvec RealError = std::get<3>(GetParam());
    const dtype RealAccuracyBeforeTraining = 5.07_dt;
    const dtype RealAccuracyAfterTraining = std::get<4>(GetParam());
    const dtype ErrorEqualityThreshold = std::get<5>(GetParam());
    const dtype FinalAccuracyEqualityThreshold = std::get<6>(GetParam());

    std::unique_ptr<Workflow> Network1;
    std::unique_ptr<Workflow> Network2;

    void SetUp() final
    {
        Network1 = std::make_unique<Workflow>();

        Network1->add<raul::DataLayer>("in1", DataParams{ { "data" }, 1, 28, 28 });
        Network1->add<raul::DataLayer>("in2", DataParams{ { "labels" }, 1, 1, 10 });
        Network1->add<raul::ReshapeLayer>("reshape", ViewParams{ { "data" }, { "reshape_out" }, 1, -1, 28 * 28 });
        Network1->add<raul::LinearLayer>("fc", LinearParams{ { "reshape_out" }, { "fc_out" }, 10 });
        Network1->add<raul::SoftMaxActivation>("softmax", BasicParamsWithDim{ { "fc_out" }, { "softmax" } });

        raul::Names in = { "softmax", "labels" };

        if (UseWeightsForLoss)
        {
            Network1->add<raul::DataLayer>("loss_in", DataParams{ { "loss_weight" }, 1, 1, 10 });
            in.push_back("loss_weight");
        }

        if (LossType == SIGMOID_CROSS_ENTROPY_LOSS)
        {
            Network1->add<raul::SigmoidCrossEntropyLoss>("loss", LossParams{ in, { "loss" }, ReductionType });
        }
        else if (LossType == CROSS_ENTROPY_LOSS)
        {
            Network1->add<raul::CrossEntropyLoss>("loss", LossParams{ in, { "loss" }, ReductionType });
        }
        else if (LossType == KL_DIV_LOSS)
        {
            Network1->add<raul::KLDivLoss>("loss", LossParams{ in, { "loss" }, ReductionType });
        }
        else if (LossType == L1_LOSS)
        {
            Network1->add<raul::L1Loss>("loss", LossParams{ in, { "loss" }, ReductionType });
        }
        else if (LossType == MSE_LOSS)
        {
            Network1->add<raul::MSELoss>("loss", LossParams{ in, { "loss" }, ReductionType });
        }
        else if (LossType == NLL_LOSS)
        {
            Network1->add<raul::NLLLoss>("loss", LossParams{ in, { "loss" }, ReductionType });
        }

        Network1->preparePipelines();
        Network1->setBatchSize(BatchSize);
        Network1->prepareMemoryForTraining();

        Network2 = std::make_unique<Workflow>();
        Network2->add<raul::DataLayer>("in1", DataParams{ { "data" }, 1, 28, 28 });
        Network2->add<raul::DataLayer>("in2", DataParams{ { "labels" }, 1, 1, 10 });
        Network2->add<raul::ReshapeLayer>("reshape", ViewParams{ { "data" }, { "reshape_out" }, 1, -1, 28 * 28 });
        Network2->add<raul::LinearLayer>("fc", LinearParams{ { "reshape_out" }, { "fc_out" }, 10 });
        Network2->add<raul::SoftMaxActivation>("softmax", BasicParamsWithDim{ { "fc_out" }, { "softmax" } });

        raul::Names in2 = { "softmax", "labels" };

        if (UseWeightsForLoss)
        {
            Network2->add<raul::DataLayer>("loss_in", DataParams{ { "loss_weight" }, 1, 1, 10 });
            in2.push_back("loss_weight");
        }

        if (LossType == SIGMOID_CROSS_ENTROPY_LOSS)
        {
            Network2->add<raul::SigmoidCrossEntropyLoss>("loss", LossParams{ in2, { "loss" }, ReductionType });
        }
        else if (LossType == CROSS_ENTROPY_LOSS)
        {
            Network2->add<raul::CrossEntropyLoss>("loss", LossParams{ in2, { "loss" }, ReductionType });
        }
        else if (LossType == KL_DIV_LOSS)
        {
            Network2->add<raul::KLDivLoss>("loss", LossParams{ in2, { "loss" }, ReductionType });
        }
        else if (LossType == L1_LOSS)
        {
            Network2->add<raul::L1Loss>("loss", LossParams{ in2, { "loss" }, ReductionType });
        }
        else if (LossType == MSE_LOSS)
        {
            Network2->add<raul::MSELoss>("loss", LossParams{ in2, { "loss" }, ReductionType });
        }
        else if (LossType == NLL_LOSS)
        {
            Network2->add<raul::NLLLoss>("loss", LossParams{ in2, { "loss" }, ReductionType });
        }

        Network2->preparePipelines();
        Network2->setBatchSize(BatchSize);
        Network2->prepareMemoryForTraining();

        DataLoader dataLoader;
        MemoryManager& memoryManager1 = Network1->getMemoryManager();
        memoryManager1["fc::Weights"] = dataLoader.loadData(UT::tools::getTestAssetsDir() / "microbatching" / "0_fc.weight.data", numberOfClasses, 28 * 28);
        memoryManager1["fc::Biases"] = dataLoader.loadData(UT::tools::getTestAssetsDir() / "microbatching" / "0_fc.bias.data", 1, numberOfClasses);
        MemoryManager& memoryManager2 = Network2->getMemoryManager();
        memoryManager2["fc::Weights"] = dataLoader.loadData(UT::tools::getTestAssetsDir() / "microbatching" / "0_fc.weight.data", numberOfClasses, 28 * 28);
        memoryManager2["fc::Biases"] = dataLoader.loadData(UT::tools::getTestAssetsDir() / "microbatching" / "0_fc.bias.data", 1, numberOfClasses);
    }
};

// according MicrobatchingTestTopology.py
TEST_P(DifferentLossesAndReductionTypes, TrainingOfJustLinearLayerPlusSoftmaxPlusLossNetworkWithMicroBatchesAndWithoutThemShouldGiveTheSameAccuracyUnit)
{
    PROFILE_TEST
    std::cout << "Test training in microbatching mode with loss " << LossType << ", reduction " << ReductionType << ", use weight for loss: " << (UseWeightsForLoss ? "yes" : "no") << "\n";
    auto trainData1 = Dataset::MNIST_Train(UT::tools::getTestAssetsDir() / "MNIST");
    auto testData1 = Dataset::MNIST_Test(UT::tools::getTestAssetsDir() / "MNIST");

    auto trainData2 = Dataset::MNIST_Train(UT::tools::getTestAssetsDir() / "MNIST");
    auto testData2 = Dataset::MNIST_Test(UT::tools::getTestAssetsDir() / "MNIST");

    raul::Test test1(*Network1, testData1, { { { "images", "data" }, { "labels", "labels" } }, "softmax", "labels" });
    raul::Test test2(*Network2, testData2, { { { "images", "data" }, { "labels", "labels" } }, "softmax", "labels" });
    dtype testAcc1 = test1.run(BatchSize);
    dtype testAcc2 = test2.run(MicroBatchSize);
    ASSERT_NEAR(testAcc1, RealAccuracyBeforeTraining, 1e-2_dt);
    ASSERT_NEAR(testAcc2, RealAccuracyBeforeTraining, 1e-2_dt);
    std::cout << std::fixed << std::setprecision(2) << "accuracy before training: " << testAcc1 << '%' << std::endl;

    auto sgd = std::make_shared<raul::optimizers::SGD>(0.05_dt);
    std::unique_ptr<Train> train1;
    if (UseWeightsForLoss)
    {
        train1 = std::make_unique<Train>(*Network1, trainData1, Train::Parameters{ { { "images", "data" }, { "labels", "labels" }, { "labels", "loss_weight" } }, "loss" });
    }
    else
    {
        train1 = std::make_unique<Train>(*Network1, trainData1, Train::Parameters{ { { "images", "data" }, { "labels", "labels" } }, "loss" });
    }
    train1->useBatches(BatchSize);

    std::unique_ptr<Train> train2;
    if (UseWeightsForLoss)
    {
        train2 = std::make_unique<Train>(*Network2, trainData2, Train::Parameters{ { { "images", "data" }, { "labels", "labels" }, { "labels", "loss_weight" } }, "loss" });
    }
    else
    {
        train2 = std::make_unique<Train>(*Network2, trainData2, Train::Parameters{ { { "images", "data" }, { "labels", "labels" } }, "loss" });
    }
    train2->useMicroBatches(BatchSize, MicroBatchSize);
    for (size_t iteration = 0, idealLossIndex = 0; iteration < std::min(train1->numberOfIterations(), train2->numberOfIterations()); ++iteration)
    {
        raul::dtype testLoss1 = train1->oneIteration(*sgd);
        raul::dtype testLoss2 = train2->oneIteration(*sgd);
        if (iteration % lossCheckStep == 0)
        {
            CHECK_NEAR(testLoss1, RealError[idealLossIndex], ErrorEqualityThreshold);
            CHECK_NEAR(testLoss2, RealError[idealLossIndex], ErrorEqualityThreshold);
            ++idealLossIndex;
            std::cout << "iteration = " << iteration << '/' << std::min(train1->numberOfIterations(), train2->numberOfIterations()) << " loss = " << std::fixed << std::setprecision(6) << testLoss2
                      << std::endl;
        }
    }
    testAcc1 = test1.run(BatchSize);
    testAcc2 = test2.run(MicroBatchSize);
    ASSERT_NEAR(testAcc1, RealAccuracyAfterTraining, FinalAccuracyEqualityThreshold);
    ASSERT_NEAR(testAcc2, RealAccuracyAfterTraining, FinalAccuracyEqualityThreshold);
    std::cout << std::fixed << std::setprecision(2) << "accuracy after training: " << testAcc2 << '%' << std::endl;
}
INSTANTIATE_TEST_SUITE_P(
    TestTrain,
    DifferentLossesAndReductionTypes,
    testing::Values(
        //        std::make_tuple(SIGMOID_CROSS_ENTROPY_LOSS, false, "mean", dvec{ 0.735768, 0.735079, 0.735155, 0.734711, 0.733745, 0.733022 }, 23.22_dt, 1e-6_dt, 1e-2_dt), // not applicable when
        //        microbatches used std::make_tuple(SIGMOID_CROSS_ENTROPY_LOSS, false, "batch_mean", dvec{ 7.357683, 7.265026, 7.187192, 7.063695, 6.952471, 6.981649 }, 71.13_dt, 1e-6_dt, 1e-2_dt), //
        //        not applicable when microbatches used
        std::make_tuple(SIGMOID_CROSS_ENTROPY_LOSS, false, "sum", dvec{ 735.768372f, 677.346069f, 679.710571f, 672.958313f, 671.024414f, 674.033936f }, 82.67_dt, 1e-2_dt, 1e-2_dt),
        std::make_tuple(SIGMOID_CROSS_ENTROPY_LOSS, false, "custom_mean", dvec{ 0.735768f, 0.735079f, 0.735155f, 0.734711f, 0.733745f, 0.733022f }, 23.22_dt, 1e-6_dt, 1e-2_dt),
        std::make_tuple(SIGMOID_CROSS_ENTROPY_LOSS, false, "custom_batch_mean", dvec{ 7.357684f, 7.265026f, 7.187192f, 7.063694f, 6.952472f, 6.981648f }, 71.13_dt, 1e-4_dt, 1e-2_dt),
        //        std::make_tuple(SIGMOID_CROSS_ENTROPY_LOSS, true, "mean", dvec{ 0.073577, 0.073553, 0.073592, 0.073597, 0.073570, 0.073541 }, 5.87_dt, 1e-6_dt, 1e-2_dt), // not applicable when
        //        microbatches used std::make_tuple(SIGMOID_CROSS_ENTROPY_LOSS, true, "batch_mean", dvec{ 7.357683, 7.265026, 7.187193, 7.063695, 6.952471, 6.981648 }, 71.13_dt, 1e-6_dt, 1e-2_dt), //
        //        not applicable when microbatches used
        std::make_tuple(SIGMOID_CROSS_ENTROPY_LOSS, true, "sum", dvec{ 735.768372f, 677.346069f, 679.710510f, 672.958313f, 671.024414f, 674.033875f }, 82.67_dt, 1e-2_dt, 1e-2_dt),
        std::make_tuple(SIGMOID_CROSS_ENTROPY_LOSS, true, "custom_mean", dvec{ 0.735768f, 0.735079f, 0.735155f, 0.734711f, 0.733745f, 0.733022f }, 23.22_dt, 1e-6_dt, 1e-2_dt),
        std::make_tuple(SIGMOID_CROSS_ENTROPY_LOSS, true, "custom_batch_mean", dvec{ 7.357684f, 7.265026f, 7.187192f, 7.063695f, 6.952472f, 6.981648f }, 71.13_dt, 1e-4_dt, 1e-2_dt),
        //        std::make_tuple(SIGMOID_CROSS_ENTROPY_LOSS, true, "sum_over_weights",
        //        dvec{ 7.357683, 7.265026, 7.187192, 7.063694, 6.952471, 6.981648 }, 71.13_dt, 1e-6_dt, 1e-2_dt), // seems like not applicable when microbatches used
        //        std::make_tuple(SIGMOID_CROSS_ENTROPY_LOSS, true, "sum_over_nonzero_weights",
        //        dvec{ 7.357683, 7.265026, 7.187192, 7.063694, 6.952471, 6.981648 }, 71.13_dt, 1e-6_dt, 1e-2_dt), // seems like not applicable when microbatches used

        //        std::make_tuple(CROSS_ENTROPY_LOSS, false, "mean", dvec{ 0.248322, 0.192460, 0.177916, 0.153807, 0.120752, 0.116952 }, 80.11_dt, 1e-6_dt, 1e-2_dt), // not applicable when
        //        microbatches used std::make_tuple(CROSS_ENTROPY_LOSS, false, "batch_mean", dvec{ 2.483224, 0.760138, 0.783885, 0.672737, 0.441113, 0.486497 }, 88.72_dt, 1e-6_dt, 1e-2_dt), // not
        //        applicable when microbatches used std::make_tuple(CROSS_ENTROPY_LOSS, false, "sum", dvec{ 248.322372, -NAN, -NAN, -NAN, -NAN, -NAN }, 9.80_dt, 1e-6_dt, 1e-2_dt), // seems like not
        //        applicable to test topology or error in implementation
        std::make_tuple(CROSS_ENTROPY_LOSS, false, "custom_mean", dvec{ 0.248322f, 0.192460f, 0.177916f, 0.153807f, 0.120752f, 0.116952f }, 80.11_dt, 1e-6_dt, 1e-2_dt),
        std::make_tuple(CROSS_ENTROPY_LOSS, false, "custom_batch_mean", dvec{ 2.483224f, 0.760138f, 0.783885f, 0.672737f, 0.441113f, 0.486497f }, 88.72_dt, 1e-6_dt, 1e-2_dt),
        //        std::make_tuple(CROSS_ENTROPY_LOSS, true, "mean", dvec{ 0.024832, 0.023896, 0.024036, 0.023407, 0.022447, 0.021944 }, 21.26_dt, 1e-6_dt, 1e-2_dt), // not applicable when microbatches
        //        used std::make_tuple(CROSS_ENTROPY_LOSS, true, "batch_mean",
        //        dvec{ 2.483224, 0.760138, 0.783885, 0.672737, 0.441113, 0.486497 }, 88.72_dt, 1e-6_dt, 1e-2_dt), // not applicable when microbatches used
        //        std::make_tuple(CROSS_ENTROPY_LOSS, true, "sum", dvec{ 248.322372, -NAN, -NAN, -NAN, -NAN, -NAN }, 9.80_dt, 1e-6_dt, 1e-2_dt), // seems like not applicable to test topology or error
        //        in implementation
        std::make_tuple(CROSS_ENTROPY_LOSS, true, "custom_mean", dvec{ 0.248322f, 0.192460f, 0.177916f, 0.153807f, 0.120752f, 0.116952f }, 80.11_dt, 1e-6_dt, 1e-2_dt),
        std::make_tuple(CROSS_ENTROPY_LOSS, true, "custom_batch_mean", dvec{ 2.483224f, 0.760138f, 0.783885f, 0.672737f, 0.441113f, 0.486497f }, 88.72_dt, 1e-6_dt, 1e-2_dt),
        //        std::make_tuple(CROSS_ENTROPY_LOSS, true, "sum_over_weights",
        //        dvec{ 2.483224, 0.760138, 0.783885, 0.672737, 0.441113, 0.486497 }, 88.72_dt, 1e-6_dt, 1e-2_dt), // seems like not applicable when microbatches used
        //        std::make_tuple(CROSS_ENTROPY_LOSS, true, "sum_over_nonzero_weights",
        //        dvec{ 2.483224, 0.760138, 0.783885, 0.672737, 0.441113, 0.486497 }, 88.72_dt, 1e-6_dt, 1e-2_dt), // seems like not applicable when microbatches used

        //        std::make_tuple(KL_DIV_LOSS, false, "mean", dvec{ -0.008799, -0.009468, -0.009419, -0.009778, -0.010729, -0.011430 }, 22.55_dt, 1e-6_dt, 1e-2_dt), // not applicable when microbatches
        //        used std::make_tuple(KL_DIV_LOSS, false, "batch_mean",
        //        dvec{ -0.087986, -0.185431, -0.285597, -0.431693, -0.559907, -0.524317 }, 65.42_dt, 1e-6_dt, 1e-2_dt), // not applicable when microbatches used
        std::make_tuple(KL_DIV_LOSS, false, "sum", dvec{ -8.798616f, -83.790756f, -80.273216f, -84.256332f, -89.532570f, -86.703514f }, 91.17_dt, 1e-3_dt, 1e-2_dt),
        std::make_tuple(KL_DIV_LOSS, false, "custom_mean", dvec{ -0.008799f, -0.009468f, -0.009419f, -0.009778f, -0.010729f, -0.011430f }, 22.55_dt, 1e-6_dt, 1e-2_dt),
        std::make_tuple(KL_DIV_LOSS, false, "custom_batch_mean", dvec{ -0.087986f, -0.185431f, -0.285597f, -0.431693f, -0.559907f, -0.524317f }, 65.42_dt, 1e-6_dt, 1e-2_dt),
        //        std::make_tuple(KL_DIV_LOSS, true, "mean",
        //        dvec{ -0.000880, -0.000905, -0.000871, -0.000860, -0.000885, -0.000915 }, 5.86_dt, 1e-6_dt, 1e-2_dt), // not applicable when microbatches used
        //        std::make_tuple(KL_DIV_LOSS, true, "batch_mean",
        //        dvec{ -0.087986, -0.185431, -0.285597, -0.431693, -0.559907, -0.524317 }, 65.42_dt, 1e-6_dt, 1e-2_dt), // not applicable when microbatches used
        std::make_tuple(KL_DIV_LOSS, true, "sum", dvec{ -8.798616f, -83.790749f, -80.273216f, -84.256325f, -89.532578f, -86.703514f }, 91.17_dt, 1e-3_dt, 1e-2_dt),
        std::make_tuple(KL_DIV_LOSS, true, "custom_mean", dvec{ -0.008799f, -0.009468f, -0.009419f, -0.009778f, -0.010729f, -0.011430f }, 22.55_dt, 1e-6_dt, 1e-2_dt),
        std::make_tuple(KL_DIV_LOSS, true, "custom_batch_mean", dvec{ -0.087986f, -0.185431f, -0.285597f, -0.431693f, -0.559907f, -0.524317f }, 65.42_dt, 1e-6_dt, 1e-2_dt),
        //        std::make_tuple(KL_DIV_LOSS, true, "sum_over_weights",
        //        dvec{ -0.087986, -0.185431, -0.285597, -0.431693, -0.559907, -0.524317 }, 65.42_dt, 1e-6_dt, 1e-2_dt), // seems like not applicable when microbatches used
        //        std::make_tuple(KL_DIV_LOSS, true, "sum_over_nonzero_weights",
        //        dvec{ -0.087986, -0.185431, -0.285597, -0.431693, -0.559907, -0.524317 }, 65.42_dt, 1e-6_dt, 1e-2_dt), // seems like not applicable when microbatches used

        //        std::make_tuple(L1_LOSS, false, "mean", dvec{ 0.182403, 0.180050, 0.179337, 0.176722, 0.171147, 0.166789 }, 44.33_dt, 1e-6_dt, 1e-2_dt),  // not applicable when microbatches used
        //        std::make_tuple(L1_LOSS, false, "batch_mean", dvec{ 1.824028, 1.266708, 1.094287, 0.924498, 0.742730, 0.820494 }, 66.58_dt, 1e-6_dt, 1e-2_dt), // not applicable when microbatches
        //        used std::make_tuple(L1_LOSS, false, "sum",
        //        dvec{ 182.402771, 83.965050, 57.024399, 66.684738, 47.062927, 61.308048 }, 74.08_dt, 1e-6_dt, 1e-2_dt), // seems like  not applicable, too much computation error?
        std::make_tuple(L1_LOSS, false, "custom_mean", dvec{ 0.182403f, 0.180050f, 0.179337f, 0.176722f, 0.171147f, 0.166789f }, 44.33_dt, 1e-6_dt, 1e-2_dt),
        std::make_tuple(L1_LOSS, false, "custom_batch_mean", dvec{ 1.824028f, 1.266708f, 1.094287f, 0.924498f, 0.742730f, 0.820494f }, 66.58_dt, 1e-6_dt, 1e-2_dt),
        //        std::make_tuple(L1_LOSS, true, "mean", dvec{ 0.018240, 0.018181, 0.018244, 0.018257, 0.018196, 0.018128 }, 6.80_dt, 1e-6_dt, 1e-2_dt), // not applicable when microbatches used
        //        std::make_tuple(L1_LOSS, true, "batch_mean", dvec{ 1.824028, 1.266708, 1.094287, 0.924498, 0.742730, 0.820494 }, 66.58_dt, 1e-6_dt, 1e-2_dt), // not applicable when microbatches used
        //        std::make_tuple(L1_LOSS, true, "sum",
        //        dvec{ 182.402771, 83.965057, 57.024403, 66.684738, 47.062923, 61.308041 }, 74.08_dt, 1e-6_dt, 1e-2_dt), // seems like  not applicable, too much computation error?
        std::make_tuple(L1_LOSS, true, "custom_mean", dvec{ 0.182403f, 0.180050f, 0.179337f, 0.176722f, 0.171147f, 0.166789f }, 44.33_dt, 1e-6_dt, 1e-2_dt),
        std::make_tuple(L1_LOSS, true, "custom_batch_mean", dvec{ 1.824028f, 1.266708f, 1.094287f, 0.924498f, 0.742730f, 0.820494f }, 66.58_dt, 1e-6_dt, 1e-2_dt),
        //        std::make_tuple(L1_LOSS, true, "sum_over_weights",
        //        dvec{ 1.824028, 1.266708, 1.094287, 0.924498, 0.742730, 0.820494 }, 66.58_dt, 1e-6_dt, 1e-2_dt), // seems like not applicable when microbatches used
        //        std::make_tuple(L1_LOSS, true, "sum_over_nonzero_weights",
        //        dvec{ 1.824028, 1.266708, 1.094287, 0.924498, 0.742730, 0.820494 }, 66.58_dt, 1e-6_dt, 1e-2_dt))); // seems like not applicable when microbatches used
        //        std::make_tuple(MSE_LOSS, false, "mean", dvec{ 0.093770, 0.090881, 0.090382, 0.087515, 0.082823, 0.080160 }, 52.18_dt, 1e-6_dt, 1e-2_dt), // not applicable when microbatches used
        //        std::make_tuple(MSE_LOSS, false, "batch_mean",
        //        dvec{ 0.937696, 0.599041, 0.521841, 0.413530, 0.270350, 0.287436 }, 87.13_dt, 1e-6_dt, 1e-2_dt), // not applicable when microbatches used
        //        std::make_tuple(MSE_LOSS, false, "sum",
        //        dvec{ 93.769608, 63.876209, 40.704750, 37.774326, 35.517841, 29.699800 }, 82.05_dt, 1e-6_dt, 1e-2_dt), // seems like  not applicable, too much computation error?
        std::make_tuple(MSE_LOSS, false, "custom_mean", dvec{ 0.093770f, 0.090881f, 0.090382f, 0.087515f, 0.082823f, 0.080160f }, 52.18_dt, 1e-6_dt, 1e-2_dt),
        std::make_tuple(MSE_LOSS, false, "custom_batch_mean", dvec{ 0.937696f, 0.599041f, 0.521841f, 0.413530f, 0.270350f, 0.287436f }, 87.13_dt, 1e-6_dt, 1e-2_dt),
        //        std::make_tuple(MSE_LOSS, true, "mean", dvec{ 0.009377, 0.009319, 0.009412, 0.009377, 0.009298, 0.009233 }, 6.48_dt, 1e-6_dt, 1e-2_dt), // not applicable when microbatches used
        //        std::make_tuple(MSE_LOSS, true, "batch_mean", dvec{ 0.937696, 0.599041, 0.521841, 0.413530, 0.270350, 0.287436 }, 87.13_dt, 1e-6_dt, 1e-2_dt), // not applicable when microbatches
        //        used std::make_tuple(MSE_LOSS, true, "sum",
        //        dvec{ 93.769615, 63.876209, 40.704750, 37.774323, 35.517838, 29.699800 }, 82.05_dt, 1e-6_dt, 1e-2_dt), // seems like  not applicable, too much computation error?
        std::make_tuple(MSE_LOSS, true, "custom_mean", dvec{ 0.093770f, 0.090881f, 0.090382f, 0.087515f, 0.082823f, 0.080160f }, 52.18_dt, 1e-6_dt, 1e-2_dt),
        std::make_tuple(MSE_LOSS, true, "custom_batch_mean", dvec{ 0.937696f, 0.599041f, 0.521841f, 0.413530f, 0.270350f, 0.287436f }, 87.13_dt, 1e-6_dt, 1e-2_dt),
        //        std::make_tuple(MSE_LOSS, true, "sum_over_weights",
        //        dvec{ 0.937696, 0.599041, 0.521841, 0.413530, 0.270350, 0.287436 }, 87.13_dt, 1e-6_dt, 1e-2_dt),  // seems like not applicable when microbatches used
        //        std::make_tuple(MSE_LOSS, true, "sum_over_nonzero_weights",
        //        dvec{ 0.937696, 0.599041, 0.521841, 0.413530, 0.270350, 0.287436 }, 87.13_dt, 1e-6_dt, 1e-2_dt), // seems like not applicable when microbatches used

        //        std::make_tuple(NLL_LOSS, false, "mean",
        //        dvec{ -0.087986, -0.185431, -0.285597, -0.431693, -0.559907, -0.524317 }, 65.42_dt, 1e-6_dt, 1e-2_dt),  // not applicable when microbatches used
        //        std::make_tuple(NLL_LOSS, false, "batch_mean",
        //        dvec{ -0.087986, -0.185431, -0.285597, -0.431693, -0.559907, -0.524317 }, 65.42_dt, 1e-6_dt, 1e-2_dt),  // not applicable when microbatches used
        std::make_tuple(NLL_LOSS, false, "sum", dvec{ -8.798615f, -83.790756f, -80.273224f, -84.256317f, -89.532570f, -86.703506f }, 91.17_dt, 1e-3_dt, 1e-2_dt),
        std::make_tuple(NLL_LOSS, false, "custom_mean", dvec{ -0.008799f, -0.009468f, -0.009419f, -0.009778f, -0.010729f, -0.011430f }, 22.55_dt, 1e-6_dt, 1e-2_dt),
        std::make_tuple(NLL_LOSS, false, "custom_batch_mean", dvec{ -0.087986f, -0.185431f, -0.285597f, -0.431693f, -0.559907f, -0.524317f }, 65.42_dt, 1e-6_dt, 1e-2_dt),
        //        std::make_tuple(NLL_LOSS, true, "mean", dvec{ -0.008799, -0.009468, -0.009419, -0.009778, -0.010729, -0.011430 }, 22.55_dt, 1e-6_dt, 1e-2_dt), // not applicable when microbatches
        //        used std::make_tuple(NLL_LOSS, true, "batch_mean",
        //        dvec{ -0.087986, -0.185431, -0.285597, -0.431693, -0.559907, -0.524317 }, 65.42_dt, 1e-6_dt, 1e-2_dt), // not applicable when microbatches used
        std::make_tuple(NLL_LOSS, true, "sum", dvec{ -8.798615f, -83.790749f, -80.273216f, -84.256325f, -89.532578f, -86.703506f }, 91.17_dt, 1e-3_dt, 1e-2_dt),
        std::make_tuple(NLL_LOSS, true, "custom_mean", dvec{ -0.008799f, -0.009468f, -0.009419f, -0.009778f, -0.010729f, -0.011430f }, 22.55_dt, 1e-6_dt, 1e-2_dt),
        std::make_tuple(NLL_LOSS, true, "custom_batch_mean", dvec{ -0.087986f, -0.185431f, -0.285597f, -0.431693f, -0.559907f, -0.524317f }, 65.42_dt, 1e-6_dt, 1e-2_dt)));
//        std::make_tuple(NLL_LOSS, true, "sum_over_weights",
//        dvec{ -0.087986, -0.185431, -0.285597, -0.431693, -0.559907, -0.524317 }, 65.42_dt, 1e-6_dt, 1e-2_dt), // seems like not applicable when microbatches used
//        std::make_tuple(NLL_LOSS, true, "sum_over_nonzero_weights",
//        dvec{ -0.087986, -0.185431, -0.285597, -0.431693, -0.559907, -0.524317 }, 65.42_dt, 1e-6_dt, 1e-2_dt))); // seems like not applicable when microbatches used
