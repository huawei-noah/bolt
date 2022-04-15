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

#include <training/tools/Dataset.h>

#include <tests/tools/TestTools.h>
using namespace raul;

namespace UT
{

class GeneratorOfTensorsFilledByCallCounter : public LoadData
{
  public:
    static std::map<std::string, uint8_t> CallCounters;

    explicit GeneratorOfTensorsFilledByCallCounter(const raul::Name& name)
        : mName(name)
    {
        CallCounters[name] = 0;
    }

    void prepareToReloadData() final { CallCounters[mName] = 0; }

    std::unique_ptr<Tensor> operator()(size_t numberOfSamples, size_t sampleDepth, size_t sampleHeight, size_t sampleWidth) final
    {
        ++CallCounters[mName];
        auto result = std::make_unique<Tensor>("", numberOfSamples, sampleDepth, sampleHeight, sampleWidth, static_cast<dtype>(CallCounters[mName]));
        return result;
    }

  private:
    std::string mName;
};
std::map<std::string, uint8_t> GeneratorOfTensorsFilledByCallCounter::CallCounters;

class ElementWiseMultiplicationByTwo : public Transform
{
  public:
    static std::map<std::string, uint8_t> CallCounters;

    explicit ElementWiseMultiplicationByTwo(const raul::Name& name)
        : mName(name)
    {
        CallCounters[name] = 0;
    }

    std::unique_ptr<Tensor> operator()(const Tensor& tsr) final
    {
        ++CallCounters[mName];
        auto result = std::make_unique<Tensor>(tsr.getBatchSize(), tsr.getDepth(), tsr.getHeight(), tsr.getWidth());
        std::transform(tsr.begin(), tsr.end(), result->begin(), [](dtype v) { return v * 2_dt; });
        return result;
    }

  private:
    std::string mName;
};
std::map<std::string, uint8_t> ElementWiseMultiplicationByTwo::CallCounters;

class ElementWiseMultiplicationByThree : public Transform
{
  public:
    static std::map<std::string, uint8_t> CallCounters;

    explicit ElementWiseMultiplicationByThree(const raul::Name& name)
        : mName(name)
    {
        CallCounters[name] = 0;
    }

    std::unique_ptr<Tensor> operator()(const Tensor& tsr) final
    {
        ++CallCounters[mName];
        auto result = std::make_unique<Tensor>(tsr.getBatchSize(), tsr.getDepth(), tsr.getHeight(), tsr.getWidth());
        std::transform(tsr.begin(), tsr.end(), result->begin(), [](dtype v) { return v * 3_dt; });
        return result;
    }

  private:
    std::string mName;
};
std::map<std::string, uint8_t> ElementWiseMultiplicationByThree::CallCounters;

struct TestDataset : public testing::Test
{
    struct DatasetPartDescription
    {
        struct SampleDimensions
        {
            uint32_t Depth;
            uint32_t Height;
            uint32_t Width;

            size_t Size() const { return Depth * Height * Width; }

            SampleDimensions(uint32_t d, uint32_t h, uint32_t w)
                : Depth(d)
                , Height(h)
                , Width(w)
            {
            }
        };

        std::string Name;
        SampleDimensions Sample;

        DatasetPartDescription(const char* name, const SampleDimensions& sampleDescription)
            : Name(name)
            , Sample(sampleDescription)
        {
        }
    };

    uint32_t NumberOfSamplesInDataset = 5;
    uint32_t BatchSize = 2;
    uint32_t IncompleteBatchSize = 1;
    DatasetPartDescription DatasetPart1{ "dataset_part1", { 1, 4, 6 } };
    DatasetPartDescription DatasetPart2{ "dataset_part2", { 1, 1, 3 } };

    std::unique_ptr<Dataset> dataset;

    void SetUp() final
    {
        dataset = std::make_unique<Dataset>();
        dataset->describePart(DatasetPart1.Name, NumberOfSamplesInDataset, DatasetPart1.Sample.Depth, DatasetPart1.Sample.Height, DatasetPart1.Sample.Width);
        dataset->setDataSourceFor(DatasetPart1.Name, std::make_unique<GeneratorOfTensorsFilledByCallCounter>("load_data1"));
        dataset->applyTo(DatasetPart1.Name, std::make_unique<ElementWiseMultiplicationByTwo>("transform1"));
        dataset->applyTo(DatasetPart1.Name, std::make_unique<ElementWiseMultiplicationByThree>("transform2"));
        dataset->describePart(DatasetPart2.Name, NumberOfSamplesInDataset, DatasetPart2.Sample.Depth, DatasetPart2.Sample.Height, DatasetPart2.Sample.Width);
        dataset->setDataSourceFor(DatasetPart2.Name, std::make_unique<GeneratorOfTensorsFilledByCallCounter>("load_data2"));
        dataset->applyTo(DatasetPart2.Name, std::make_unique<ElementWiseMultiplicationByTwo>("transform3"));
        dataset->applyTo(DatasetPart2.Name, std::make_unique<ElementWiseMultiplicationByThree>("transform4"));
        dataset->generate(BatchSize);
    }

    void ImitateDatasetWraparound()
    {
        for (auto& callCounter : GeneratorOfTensorsFilledByCallCounter::CallCounters)
        {
            callCounter.second = 0;
        }
    }
};

TEST_F(TestDataset, ShouldUseLoadersToLoadDataDuringGeneration)
{
    PROFILE_TEST
    const size_t loaderCallTimes = NumberOfSamplesInDataset / BatchSize;
    ASSERT_EQ(GeneratorOfTensorsFilledByCallCounter::CallCounters["load_data1"], loaderCallTimes);
    ASSERT_EQ(GeneratorOfTensorsFilledByCallCounter::CallCounters["load_data2"], loaderCallTimes);
    ASSERT_EQ(ElementWiseMultiplicationByTwo::CallCounters["transform1"], 0u);
    ASSERT_EQ(ElementWiseMultiplicationByTwo::CallCounters["transform3"], 0u);
    ASSERT_EQ(ElementWiseMultiplicationByThree::CallCounters["transform2"], 0u);
    ASSERT_EQ(ElementWiseMultiplicationByThree::CallCounters["transform4"], 0u);
}

TEST_F(TestDataset, ShouldApplyAllDefinedTransformationsToEachPartOfDatasetOnTheFlyUnit)
{
    PROFILE_TEST
    dataset->getData();
    dataset->getData();

    ASSERT_EQ(ElementWiseMultiplicationByTwo::CallCounters["transform1"], 2u);
    ASSERT_EQ(ElementWiseMultiplicationByTwo::CallCounters["transform3"], 2u);
    ASSERT_EQ(ElementWiseMultiplicationByThree::CallCounters["transform2"], 2u);
    ASSERT_EQ(ElementWiseMultiplicationByThree::CallCounters["transform4"], 2u);
}

TEST_F(TestDataset, ShouldGiveDataOnDemandUnit)
{
    PROFILE_TEST
    auto dataBatch = dataset->getData();
    const Tensor& result1 = dataBatch.get(DatasetPart1.Name);
    const Tensor& result2 = dataBatch.get(DatasetPart2.Name);

    ASSERT_EQ(result1.size(), size_t(BatchSize * DatasetPart1.Sample.Size()));
    std::for_each(result1.begin(), result1.end(), [](dtype v) { ASSERT_NEAR(v, 6_dt, 1e-6_dt); });
    ASSERT_EQ(result2.size(), size_t(BatchSize * DatasetPart2.Sample.Size()));
    std::for_each(result1.begin(), result1.end(), [](dtype v) { ASSERT_NEAR(v, 6_dt, 1e-6_dt); });
    std::for_each(result2.begin(), result2.end(), [](dtype v) { ASSERT_NEAR(v, 6_dt, 1e-6_dt); });
}

TEST_F(TestDataset, ShouldGiveDataInCircleUnit)
{
    PROFILE_TEST
    auto dataBatch1 = dataset->getData();
    std::for_each(dataBatch1.get(DatasetPart1.Name).begin(), dataBatch1.get(DatasetPart1.Name).end(), [](dtype v) { ASSERT_NEAR(v, 6_dt, 1e-6_dt); });
    std::for_each(dataBatch1.get(DatasetPart2.Name).begin(), dataBatch1.get(DatasetPart2.Name).end(), [](dtype v) { ASSERT_NEAR(v, 6_dt, 1e-6_dt); });

    auto dataBatch2 = dataset->getData();
    std::for_each(dataBatch2.get(DatasetPart1.Name).begin(), dataBatch2.get(DatasetPart1.Name).end(), [](dtype v) { ASSERT_NEAR(v, 12_dt, 1e-6_dt); });
    std::for_each(dataBatch2.get(DatasetPart2.Name).begin(), dataBatch2.get(DatasetPart2.Name).end(), [](dtype v) { ASSERT_NEAR(v, 12_dt, 1e-6_dt); });

    auto dataBatch3 = dataset->getData();
    std::for_each(dataBatch3.get(DatasetPart1.Name).begin(), dataBatch3.get(DatasetPart1.Name).end(), [](dtype v) { ASSERT_NEAR(v, 6_dt, 1e-6_dt); });
    std::for_each(dataBatch3.get(DatasetPart2.Name).begin(), dataBatch3.get(DatasetPart2.Name).end(), [](dtype v) { ASSERT_NEAR(v, 6_dt, 1e-6_dt); });
}

TEST_F(TestDataset, ShouldReloadDataIfGenerateCalled)
{
    PROFILE_TEST
    auto dataBatch1 = dataset->getData();
    std::for_each(dataBatch1.get(DatasetPart1.Name).begin(), dataBatch1.get(DatasetPart1.Name).end(), [](dtype v) { ASSERT_NEAR(v, 6_dt, 1e-6_dt); });
    std::for_each(dataBatch1.get(DatasetPart2.Name).begin(), dataBatch1.get(DatasetPart2.Name).end(), [](dtype v) { ASSERT_NEAR(v, 6_dt, 1e-6_dt); });

    dataset->generate(BatchSize);

    auto dataBatch2 = dataset->getData();
    std::for_each(dataBatch2.get(DatasetPart1.Name).begin(), dataBatch2.get(DatasetPart1.Name).end(), [](dtype v) { ASSERT_NEAR(v, 6_dt, 1e-6_dt); });
    std::for_each(dataBatch2.get(DatasetPart2.Name).begin(), dataBatch2.get(DatasetPart2.Name).end(), [](dtype v) { ASSERT_NEAR(v, 6_dt, 1e-6_dt); });
}

TEST_F(TestDataset, ShouldCorrectlyCalculateNumberOfBatchesAndNumberOfSamplesUnit)
{
    PROFILE_TEST
    size_t numberOfBatchesWithoutLastIncompleteBatch = dataset->numberOfBatches();
    size_t numberOfSamplesWithoutLastIncompleteBatch = dataset->numberOfSamples();

    dataset->configure(Dataset::Option::USE_LAST_INCOMPLETE_BATCH);
    dataset->generate(BatchSize);
    size_t numberOfBatchesWithLastIncompleteBatch = dataset->numberOfBatches();
    size_t numberOfSamplesWithLastIncompleteBatch = dataset->numberOfSamples();

    dataset->configure(Dataset::Option::SKIP_LAST_INCOMPLETE_BATCH);
    dataset->generate(BatchSize);
    size_t numberOfBatchesWhenLastIncompleteBatchSkippedAgain = dataset->numberOfBatches();
    size_t numberOfSamplesWhenLastIncompleteBatchSkippedAgain = dataset->numberOfSamples();

    ASSERT_EQ(numberOfBatchesWithoutLastIncompleteBatch, NumberOfSamplesInDataset / BatchSize);
    ASSERT_EQ(numberOfSamplesWithoutLastIncompleteBatch, NumberOfSamplesInDataset);
    ASSERT_EQ(numberOfBatchesWithLastIncompleteBatch, NumberOfSamplesInDataset / BatchSize + 1);
    ASSERT_EQ(numberOfSamplesWithLastIncompleteBatch, NumberOfSamplesInDataset);
    ASSERT_EQ(numberOfBatchesWhenLastIncompleteBatchSkippedAgain, NumberOfSamplesInDataset / BatchSize);
    ASSERT_EQ(numberOfSamplesWhenLastIncompleteBatchSkippedAgain, NumberOfSamplesInDataset);
}

TEST_F(TestDataset, ShouldUseLastIncompleteBatchIfNeedUnit)
{
    PROFILE_TEST
    dataset->configure(Dataset::Option::USE_LAST_INCOMPLETE_BATCH);
    dataset->generate(BatchSize);
    dataset->getData();
    dataset->getData();

    auto dataBatch1 = dataset->getData();
    const Tensor& incompleteBatch1 = dataBatch1.get(DatasetPart1.Name);
    const Tensor& incompleteBatch2 = dataBatch1.get(DatasetPart2.Name);
    ASSERT_EQ(incompleteBatch1.size(), size_t(IncompleteBatchSize * DatasetPart1.Sample.Size()));
    std::for_each(incompleteBatch1.begin(), incompleteBatch1.end(), [](dtype v) { ASSERT_NEAR(v, 18_dt, 1e-6_dt); });
    ASSERT_EQ(incompleteBatch2.size(), size_t(IncompleteBatchSize * DatasetPart2.Sample.Size()));
    std::for_each(incompleteBatch2.begin(), incompleteBatch2.end(), [](dtype v) { ASSERT_NEAR(v, 18_dt, 1e-6_dt); });

    ImitateDatasetWraparound();

    auto dataBatch2 = dataset->getData();
    const Tensor& resultAfterWraparound1 = dataBatch2.get(DatasetPart1.Name);
    const Tensor& resultAfterWraparound2 = dataBatch2.get(DatasetPart2.Name);

    ASSERT_EQ(resultAfterWraparound1.size(), size_t(BatchSize * DatasetPart1.Sample.Size()));
    std::for_each(resultAfterWraparound1.begin(), resultAfterWraparound1.end(), [](dtype v) { ASSERT_NEAR(v, 6_dt, 1e-6_dt); });
    ASSERT_EQ(resultAfterWraparound2.size(), size_t(BatchSize * DatasetPart2.Sample.Size()));
    std::for_each(resultAfterWraparound2.begin(), resultAfterWraparound2.end(), [](dtype v) { ASSERT_NEAR(v, 6_dt, 1e-6_dt); });
}

TEST_F(TestDataset, CanSkipLastIncompleteBatchAgainIfNeedUnit)
{
    PROFILE_TEST
    dataset->configure(Dataset::Option::USE_LAST_INCOMPLETE_BATCH);
    dataset->generate(BatchSize);
    dataset->configure(Dataset::Option::SKIP_LAST_INCOMPLETE_BATCH);
    dataset->generate(BatchSize);
    dataset->getData();
    dataset->getData();

    auto dataBatch3 = dataset->getData();
    const Tensor& result1 = dataBatch3.get(DatasetPart1.Name);
    const Tensor& result2 = dataBatch3.get(DatasetPart2.Name);
    ASSERT_EQ(result1.size(), BatchSize * DatasetPart1.Sample.Size());
    std::for_each(result1.begin(), result1.end(), [](dtype v) { ASSERT_NEAR(v, 6_dt, 1e-6_dt); });
    ASSERT_EQ(result2.size(), BatchSize * DatasetPart2.Sample.Size());
    std::for_each(result2.begin(), result2.end(), [](dtype v) { ASSERT_NEAR(v, 6_dt, 1e-6_dt); });
}

TEST_F(TestDataset, ShouldGiveDataInRandomizedOrderIfNeedUnit)
{
    PROFILE_TEST
    Dataset datasetWithRandomization;
    datasetWithRandomization.configure(raul::Dataset::USE_LAST_INCOMPLETE_BATCH);
    datasetWithRandomization.configure(raul::Dataset::RANDOMIZE_BATCH_SEQUENCE);
    datasetWithRandomization.describePart("part1", 100, 1, 1, 1);
    datasetWithRandomization.setDataSourceFor("part1", std::make_unique<GeneratorOfTensorsFilledByCallCounter>("load1"));
    datasetWithRandomization.describePart("part2", 100, 1, 1, 1);
    datasetWithRandomization.setDataSourceFor("part2", std::make_unique<GeneratorOfTensorsFilledByCallCounter>("load2"));
    datasetWithRandomization.generate(1);

    std::vector<dtype> firstResultsElements1;
    std::vector<dtype> firstResultsElements2;
    for (size_t i = 0; i < 100; ++i)
    {
        auto dataBatch = datasetWithRandomization.getData();
        firstResultsElements1.push_back(dataBatch.get("part1")[0]);
        firstResultsElements2.push_back(dataBatch.get("part2")[0]);
    }

    ASSERT_NEAR(std::accumulate(firstResultsElements1.begin(), firstResultsElements1.end(), dtype()), 5050_dt, 1e-6_dt);
    ASSERT_NEAR(std::accumulate(firstResultsElements2.begin(), firstResultsElements2.end(), dtype()), 5050_dt, 1e-6_dt);
    ASSERT_FALSE(std::is_sorted(firstResultsElements1.begin(), firstResultsElements1.end()));
    ASSERT_FALSE(std::is_sorted(firstResultsElements2.begin(), firstResultsElements2.end()));
    ASSERT_TRUE(std::equal(firstResultsElements1.begin(), firstResultsElements1.end(), firstResultsElements2.begin()));
}

TEST_F(TestDataset, ShouldTurnOffRandomizationOfBatchSequenceOnDemandUnit)
{
    PROFILE_TEST
    dataset->configure(raul::Dataset::RANDOMIZE_BATCH_SEQUENCE);
    dataset->generate(1);

    size_t attemptsToGetUnsortedBatchSequence = 100;
    std::vector<dtype> resultWithRandomization;
    for (size_t attemptIdx = 0; attemptIdx < attemptsToGetUnsortedBatchSequence; ++attemptIdx)
    {
        resultWithRandomization.clear();
        for (size_t i = 0; i < NumberOfSamplesInDataset; ++i)
        {
            auto dataBatch = dataset->getData();
            resultWithRandomization.push_back(dataBatch.get(DatasetPart1.Name)[0]);
        }
        if (!std::is_sorted(resultWithRandomization.begin(), resultWithRandomization.end()))
        {
            break;
        }
    }
    ASSERT_FALSE(std::is_sorted(resultWithRandomization.begin(), resultWithRandomization.end()));

    dataset->configure(raul::Dataset::TURN_OFF_BATCH_SEQUENCE_RANDOMIZATION);
    dataset->generate(1);

    attemptsToGetUnsortedBatchSequence = 100;
    std::vector<dtype> resultWithoutRandomization;
    for (size_t attemptIdx = 0; attemptIdx < attemptsToGetUnsortedBatchSequence; ++attemptIdx)
    {
        resultWithoutRandomization.clear();
        for (size_t i = 0; i < NumberOfSamplesInDataset; ++i)
        {
            auto dataBatch = dataset->getData();
            resultWithoutRandomization.push_back(dataBatch.get(DatasetPart1.Name)[0]);
        }
        ASSERT_TRUE(std::is_sorted(resultWithoutRandomization.begin(), resultWithoutRandomization.end()));
    }
}

TEST_F(TestDataset, ThrowAnExceptionIfAnAttemptToGetDataFromDatasetPartWithUnknownNameDetectedUnit)
{
    PROFILE_TEST
    auto dataBatch = dataset->getData();
    ASSERT_THROW(dataBatch.get("unknown part of dataset"), std::logic_error);
}

TEST_F(TestDataset, ThrowAnExceptionIfAnAttemptToConfigureAlreadyConfiguredPartOfDatasetDetectedUnit)
{
    PROFILE_TEST
    ASSERT_THROW(dataset->describePart("images", 6, 1, 1, 1), std::logic_error);
}

TEST_F(TestDataset, ThrowAnExceptionIfAnAttemptToConfigureUnknownPartOfDatasetDetectedUnit)
{
    PROFILE_TEST
    ASSERT_THROW(dataset->setDataSourceFor("unknown part of dataset", std::make_unique<GeneratorOfTensorsFilledByCallCounter>("test")), std::logic_error);
    ASSERT_THROW(dataset->applyTo("unknown part of dataset", std::make_unique<ElementWiseMultiplicationByTwo>("test")), std::logic_error);
}

TEST_F(TestDataset, ThrowAnExceptionIfAnAttemptToGetDataFromDatasetWithoutGenerationDetectedUnit)
{
    PROFILE_TEST
    Dataset datasetWithoutGenerationCall;
    datasetWithoutGenerationCall.describePart("images", 4, 1, 4, 6);
    datasetWithoutGenerationCall.setDataSourceFor("images", std::make_unique<GeneratorOfTensorsFilledByCallCounter>("load_data1"));
    datasetWithoutGenerationCall.applyTo("images", std::make_unique<ElementWiseMultiplicationByTwo>("transform1"));

    ASSERT_THROW(datasetWithoutGenerationCall.getData(), std::logic_error);
    ASSERT_THROW(datasetWithoutGenerationCall.numberOfBatches(), std::logic_error);
}

TEST_F(TestDataset, ThrowAnExceptionIfAnAttemptToRegisterPartsWithDifferentNumberOfSamplesDetectedUnit)
{
    PROFILE_TEST
    Dataset badlyConfiguredDataset;
    badlyConfiguredDataset.describePart("images", 4, 1, 4, 6);
    ASSERT_THROW(badlyConfiguredDataset.describePart("labels", 5, 1, 4, 6), std::logic_error);
}

TEST_F(TestDataset, ThrowAnExceptionIfAnAttemptToGetDataFromDatasetPartWitoutRegisteredDataSourceDetectedUnit)
{
    PROFILE_TEST
    Dataset badlyConfiguredDataset;
    badlyConfiguredDataset.describePart("images", 4, 1, 4, 6);
    badlyConfiguredDataset.applyTo("images", std::make_unique<ElementWiseMultiplicationByTwo>("transform1"));
    ASSERT_THROW(badlyConfiguredDataset.generate(2), std::logic_error);
}

TEST_F(TestDataset, ShouldReturnDataFromDataSourceWithoutAnyChangesIfNoOneTransformationRegisteredUnit)
{
    PROFILE_TEST
    Dataset datasetWithDefaultTransformation;
    datasetWithDefaultTransformation.describePart("images", 4, 1, 4, 6);
    datasetWithDefaultTransformation.setDataSourceFor("images", std::make_unique<GeneratorOfTensorsFilledByCallCounter>("stub"));
    datasetWithDefaultTransformation.generate(4);

    auto dataBatch = datasetWithDefaultTransformation.getData();
    const Tensor& result1 = dataBatch.get("images");
    ASSERT_EQ(result1.size(), size_t(4 * 1 * 4 * 6));
    for (auto it = result1.begin(); it != result1.end(); ++it)
    {
        ASSERT_NEAR(*it, 1_dt, 1e-6_dt);
    }
}

} // !namespace UT