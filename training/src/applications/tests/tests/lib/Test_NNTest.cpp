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

#include <training/layers/basic/DataLayer.h>

#include <training/common/Test.h>

#include <tests/tools/TestTools.h>
using namespace raul;

class LayerStub : public BasicLayer
{
  public:
    static size_t CallForwardInTestModeCounter;
    static size_t CallForwardInTrainModeCounter;
    static size_t CallBackwardCounter;
    static size_t NumberOfBrokenSamples;

  public:
    LayerStub(const Name& name, const BasicParams& layerParameters, NetworkParameters& networkParameters)
        : BasicLayer(name, "LayerStub", layerParameters, networkParameters)
    {
        CallForwardInTestModeCounter = 0;
        CallForwardInTrainModeCounter = 0;
        CallBackwardCounter = 0;
        NumberOfBrokenSamples = 0;

        mNetworkParams.mWorkflow.copyDeclaration(name, mInputs[0], raul::Workflow::Usage::Forward, raul::Workflow::Mode::Read);
        mNetworkParams.mWorkflow.copyDeclaration(mName, mInputs[0], mOutputs[0], DEC_FORW_WRIT_NOMEMOPT);
    }

    void forwardComputeImpl(NetworkMode mode) final
    {
        if (mode == NetworkMode::Test)
        {
            copyInputToOutput();
            if (CallForwardInTestModeCounter % 2 != 0)
            {
                breakEachEvenSample();
            }
            ++CallForwardInTestModeCounter;
        }
        else
        {
            ++CallForwardInTrainModeCounter;
        }
    }
    void backwardComputeImpl() final { ++CallBackwardCounter; }

  private:
    void copyInputToOutput()
    {
        const auto& input = mNetworkParams.mMemoryManager[mInputs[0]];
        std::copy(input.begin(), input.end(), mNetworkParams.mMemoryManager[mOutputs[0]].begin());
    }

    void breakEachEvenSample()
    {
        auto& output = mNetworkParams.mMemoryManager[mOutputs[0]];
        for (size_t sampleIdx = 0; sampleIdx < output.getBatchSize(); ++sampleIdx)
        {
            auto start = output.data() + (sampleIdx * (output.size() / output.getBatchSize()));
            if (sampleIdx % 2 == 0)
            {
                std::next_permutation(start, start + (output.size() / output.getBatchSize()));
                ++NumberOfBrokenSamples;
            }
        }
    }
};
size_t LayerStub::CallForwardInTestModeCounter = 0;
size_t LayerStub::CallForwardInTrainModeCounter = 0;
size_t LayerStub::CallBackwardCounter = 0;
size_t LayerStub::NumberOfBrokenSamples = 0;

class DataLoadStub : public LoadData
{
  public:
    void prepareToReloadData() final {}

    std::unique_ptr<Tensor> operator()(size_t numberOfSamples, size_t sampleDepth, size_t sampleHeight, size_t sampleWidth) final
    {
        size_t idxOfElementWhichShouldBeEqualOne = 0;
        auto result = std::make_unique<Tensor>(numberOfSamples, sampleDepth, sampleHeight, sampleWidth);
        for (size_t i = 0; i < result->size(); ++i)
        {
            if (i == idxOfElementWhichShouldBeEqualOne)
            {
                (*result)[i] = 1_dt;
                idxOfElementWhichShouldBeEqualOne += sampleDepth * sampleHeight * sampleWidth + 1;
            }
            else
            {
                (*result)[i] = 0_dt;
            }
        }
        return result;
    }
};

struct TestNNTest : public testing::Test
{
    uint32_t numberOfSamplesInDataset = 7;
    uint32_t batchSize = 2;

    const char* nameOfDatasetPart = "dataset_part";

    const char* nameOfNetworkInput = "network_input";
    uint32_t networkInputDepth = 1;
    uint32_t networkInputHeight = 1;
    uint32_t networkInputWidth = 10;

    const char* nameOfNetworkOutput = "network_output";

    std::unique_ptr<Workflow> networkStub;
    std::unique_ptr<Dataset> datasetStub;

    void SetUp() final
    {
        networkStub = std::make_unique<Workflow>();

        networkStub->add<raul::DataLayer>("network_input", DataParams{ { nameOfNetworkInput }, networkInputDepth, networkInputHeight, networkInputWidth });
        networkStub->add<LayerStub>("network_output", BasicParams{ { nameOfNetworkInput }, { nameOfNetworkOutput } });

        networkStub->preparePipelines();
        networkStub->setBatchSize(batchSize);
        networkStub->prepareMemoryForTraining();

        datasetStub = std::make_unique<Dataset>();
        datasetStub->describePart(nameOfDatasetPart, numberOfSamplesInDataset, networkInputDepth, networkInputHeight, networkInputWidth);
        datasetStub->setDataSourceFor(nameOfDatasetPart, std::make_unique<DataLoadStub>());
    }

    void TearDown() final {}
};

TEST_F(TestNNTest, ShouldCorrectlyCalculateAccuracyUnit)
{
    PROFILE_TEST
    raul::Test test(*networkStub, *datasetStub, { { { "dataset_part", "network_input" } }, "network_output", "dataset_part" });
    dtype accuracy = test.run(batchSize);
    ASSERT_EQ(LayerStub::CallForwardInTestModeCounter, numberOfSamplesInDataset / batchSize);
    ASSERT_EQ(LayerStub::CallForwardInTrainModeCounter, 0u);
    ASSERT_EQ(LayerStub::CallBackwardCounter, 0u);
    size_t samplesCount = datasetStub->numberOfBatches() * datasetStub->getBatchSize();
    dtype accuracyBenchmark = (static_cast<dtype>(samplesCount - LayerStub::NumberOfBrokenSamples) / static_cast<dtype>(samplesCount)) * 100_dt;
    ASSERT_NEAR(accuracy, accuracyBenchmark, 1e-6_dt);
}

TEST_F(TestNNTest, ShouldCorrectlyWorkWithLastIncompliteBathCalculateAccuracyUnit)
{
    PROFILE_TEST
    datasetStub->configure(raul::Dataset::USE_LAST_INCOMPLETE_BATCH);

    raul::Test test(*networkStub, *datasetStub, { { { "dataset_part", "network_input" } }, "network_output", "dataset_part" });
    dtype accuracy = test.run(batchSize);
    size_t samplesCount = datasetStub->numberOfSamples();
    dtype accuracyBenchmark = (static_cast<dtype>(samplesCount - LayerStub::NumberOfBrokenSamples) / static_cast<dtype>(samplesCount)) * 100_dt;
    ASSERT_NEAR(accuracy, accuracyBenchmark, 1e-6_dt);
    ASSERT_EQ(networkStub->getBatchSize(), 1u);
}

TEST_F(TestNNTest, ThrowAnExceptionIfSizesOfMappedNetworkInputAndPartOfDatasetIsNotEqualUnit)
{
    PROFILE_TEST
    Dataset dataset;
    dataset.describePart(nameOfDatasetPart, numberOfSamplesInDataset, 10, 10, 10);
    dataset.setDataSourceFor(nameOfDatasetPart, std::make_unique<DataLoadStub>());

    raul::Test test(*networkStub, dataset, { { { "dataset_part", "network_input" } }, "network_output", "dataset_part" });
    ASSERT_THROW(test.run(batchSize), std::logic_error);
}
