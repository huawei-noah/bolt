// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <training/common/Test.h>

using namespace raul;

dtype Test::run(size_t numberOfSamplesInBatch)
{
    configureDataset(numberOfSamplesInBatch);

    size_t numberOfProcessedSamples = 0;
    size_t numberOfCorrectlyClassifiedSamples = 0;
    for (uint32_t batchIdx = 0; batchIdx < mDataset.numberOfBatches(); ++batchIdx)
    {
        auto dataBatch = mDataset.getData();
        resetBatchSizeForNetworkIfNeed(dataBatch.numberOfSamples());
        setUpNetworkInputsBy(dataBatch);
        numberOfCorrectlyClassifiedSamples += calculateNumberOfCorrectlyClassifiedSamplesFor(dataBatch);
        numberOfProcessedSamples += dataBatch.numberOfSamples();
    }

    return calculateAccuracyInPercents(numberOfCorrectlyClassifiedSamples, numberOfProcessedSamples);
}

void Test::configureDataset(size_t numberOfSamplesInBatch)
{
    if (numberOfSamplesInBatch > std::numeric_limits<uint32_t>::max())
    {
        throw std::logic_error("Too big number of samples in batch for Dataset (exceeds uint32_t max value)");
    }
    mDataset.generate(static_cast<uint32_t>(numberOfSamplesInBatch));
}

void Test::resetBatchSizeForNetworkIfNeed(size_t dataBatchSize)
{
    if (mNetwork.getBatchSize() != dataBatchSize)
    {
        mNetwork.setBatchSize(dataBatchSize);
    }
}

void Test::setUpNetworkInputsBy(const Dataset::DataBatch& dataBatch)
{
    if (mNetwork.getExecutionTarget() == ExecutionTarget::CPU)
    {
        for (const auto& dataFlowDescription : mParameters.mDataFlowConfiguration)
        {
            const auto& inputData = dataBatch.get(dataFlowDescription.mDatasetPartNameAsDataSource);
            auto& networkInput = mNetwork.getMemoryManager()[dataFlowDescription.mNetworkInputNameAsDataDestination];
            if (inputData.getDepth() != networkInput.getDepth())
            {
                throw std::logic_error("Depth of network input differs from Depth of input data");
            }
            if (inputData.getHeight() != networkInput.getHeight())
            {
                throw std::logic_error("Height of network input differs from Height of input data");
            }
            if (inputData.getWidth() != networkInput.getWidth())
            {
                throw std::logic_error("Width of network input differs from Width of input data");
            }
            std::copy(inputData.begin(), inputData.end(), networkInput.begin());
        }
    }
    else if (mNetwork.getExecutionTarget() == ExecutionTarget::CPUFP16)
    {
        for (const auto& dataFlowDescription : mParameters.mDataFlowConfiguration)
        {
            const auto& inputData = dataBatch.get(dataFlowDescription.mDatasetPartNameAsDataSource);
            auto& networkInput = mNetwork.getMemoryManager<MemoryManagerFP16>()[dataFlowDescription.mNetworkInputNameAsDataDestination];
            if (inputData.getDepth() != networkInput.getDepth())
            {
                throw std::logic_error("Depth of network input differs from Depth of input data");
            }
            if (inputData.getHeight() != networkInput.getHeight())
            {
                throw std::logic_error("Height of network input differs from Height of input data");
            }
            if (inputData.getWidth() != networkInput.getWidth())
            {
                throw std::logic_error("Width of network input differs from Width of input data");
            }
            std::transform(inputData.begin(), inputData.end(), networkInput.begin(), [](float val) -> half { return toFloat16(val); });
        }
    }
    else
    {
        THROW_NONAME("Test", "unsupported execution target");
    }
}

size_t Test::calculateNumberOfCorrectlyClassifiedSamplesFor(const Dataset::DataBatch& dataBatch)
{
    mNetwork.forwardPassTesting();

    size_t numberOfCorrectlyClassifiedSamples = 0;

    if (mNetwork.getExecutionTarget() == ExecutionTarget::CPU)
    {
        const auto& networkOutput = mNetwork.getMemoryManager()[mParameters.mNetworkOutputName];
        const auto& outputBenchmark = dataBatch.get(mParameters.mDatasetPartAsGroundTruth);
        const size_t batchSize = outputBenchmark.getBatchSize();
        const size_t sampleSize = outputBenchmark.size() / batchSize;
        for (size_t sampleIdx = 0; sampleIdx < batchSize; ++sampleIdx)
        {
            size_t intervalStart = sampleIdx * sampleSize;
            size_t intervalEnd = intervalStart + sampleSize;
            if (networkOutput.getMaxIndex(intervalStart, intervalEnd) == outputBenchmark.getMaxIndex(intervalStart, intervalEnd))
            {
                ++numberOfCorrectlyClassifiedSamples;
            }
        }
    }
    else if (mNetwork.getExecutionTarget() == ExecutionTarget::CPUFP16)
    {
        const auto& networkOutput = mNetwork.getMemoryManager<MemoryManagerFP16>()[mParameters.mNetworkOutputName];
        const auto& outputBenchmark = dataBatch.get(mParameters.mDatasetPartAsGroundTruth);
        const size_t batchSize = outputBenchmark.getBatchSize();
        const size_t sampleSize = outputBenchmark.size() / batchSize;
        for (size_t sampleIdx = 0; sampleIdx < batchSize; ++sampleIdx)
        {
            size_t intervalStart = sampleIdx * sampleSize;
            size_t intervalEnd = intervalStart + sampleSize;
            if (networkOutput.getMaxIndex(intervalStart, intervalEnd) == outputBenchmark.getMaxIndex(intervalStart, intervalEnd))
            {
                ++numberOfCorrectlyClassifiedSamples;
            }
        }
    }
    else
    {
        THROW_NONAME("Test", "unsupported execution target");
    }

    return numberOfCorrectlyClassifiedSamples;
}

dtype Test::calculateAccuracyInPercents(size_t numberOfCorrectlyClassifiedSamples, size_t samplesInAmount)
{
    return (static_cast<dtype>(numberOfCorrectlyClassifiedSamples) / static_cast<dtype>(samplesInAmount)) * 100_dt;
}