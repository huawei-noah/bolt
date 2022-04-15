// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <training/tools/Dataset.h>

using namespace raul;

class DoubleTransform : public Transform
{
  public:
    DoubleTransform(std::unique_ptr<Transform> firstTransform, std::unique_ptr<Transform> secondTransform)
        : mFirstTransform(std::move(firstTransform))
        , mSecondTransform(std::move(secondTransform))
    {
    }

    std::unique_ptr<Tensor> operator()(const Tensor& tsr) final { return (*mSecondTransform)(*((*mFirstTransform)(tsr))); }

  private:
    std::unique_ptr<Transform> mFirstTransform;
    std::unique_ptr<Transform> mSecondTransform;
};

class DefaultTransform : public Transform
{
  public:
    std::unique_ptr<Tensor> operator()(const Tensor& tsr) final
    {
        auto result = std::make_unique<Tensor>(tsr.getBatchSize(), tsr.getDepth(), tsr.getHeight(), tsr.getWidth());
        std::copy(tsr.begin(), tsr.end(), result->begin());
        return result;
    }
};

Dataset::Part::Part(uint32_t numberOfSamples, uint32_t sampleDepth, uint32_t sampleHeight, uint32_t sampleWidth)
    : mNumberOfSamples(numberOfSamples)
    , mSampleDepth(sampleDepth)
    , mSampleHeight(sampleHeight)
    , mSampleWidth(sampleWidth)
{
}

void Dataset::Part::setDataSource(std::unique_ptr<LoadData> dataSource)
{
    mLoadData = std::move(dataSource);
}

void Dataset::Part::apply(std::unique_ptr<Transform> transformation)
{
    if (!mTransformation)
    {
        mTransformation = std::move(transformation);
    }
    else
    {
        mTransformation = std::make_unique<DoubleTransform>(std::move(mTransformation), std::move(transformation));
    }
}

void Dataset::Part::generate(uint32_t batchSize, bool useIncompleteBatch)
{
    if (!mLoadData)
    {
        throw std::logic_error("There is no configured data source for registered data part");
    }
    if (!mTransformation)
    {
        mTransformation = std::make_unique<DefaultTransform>();
    }

    prepareToReloadData();
    size_t batchesInTotal = numberOfBatches(batchSize, useIncompleteBatch);
    size_t samplesCanBeLoaded = mNumberOfSamples;
    for (size_t i = 0; i < batchesInTotal; ++i)
    {
        size_t samplesToLoad = batchSize < samplesCanBeLoaded ? batchSize : samplesCanBeLoaded;
        auto data = (*mLoadData)(samplesToLoad, mSampleDepth, mSampleHeight, mSampleWidth);
        mRawDataBatches.emplace_back(std::move(data));
        samplesCanBeLoaded -= samplesToLoad;
    }
}

void Dataset::Part::prepareToReloadData()
{
    mRawDataBatches.clear();
    mLoadData->prepareToReloadData();
}

uint32_t Dataset::Part::numberOfBatches(uint32_t batchSize, bool useIncompleteBatch) const
{
    return (mNumberOfSamples / batchSize) + (useIncompleteBatch && (mNumberOfSamples % batchSize) ? 1 : 0);
}

std::unique_ptr<Tensor> Dataset::Part::getData(uint32_t batchIndex)
{
    if (batchIndex >= mRawDataBatches.size())
    {
        throw std::logic_error("Too big batch index");
    }

    return (*mTransformation)(*mRawDataBatches[batchIndex]);
}

const Tensor& Dataset::DataBatch::get(const raul::Name& name) const
{
    auto data = mDataParts.find(name);
    if (data == mDataParts.end())
    {
        throw std::logic_error("unknown name of dataset part");
    }

    return *data->second;
}

void Dataset::configure(Option option)
{
    switch (option)
    {
        case USE_LAST_INCOMPLETE_BATCH:
        {
            useIncompleteBatch = true;
            break;
        }
        case SKIP_LAST_INCOMPLETE_BATCH:
        {
            useIncompleteBatch = false;
            break;
        }
        case RANDOMIZE_BATCH_SEQUENCE:
        {
            needRandomizeBatchSequence = true;
            break;
        }
        case TURN_OFF_BATCH_SEQUENCE_RANDOMIZATION:
        {
            needRandomizeBatchSequence = false;
            break;
        }
        default:
            throw std::logic_error("Unknown option for dataset");
    }
}

uint32_t Dataset::numberOfSamples() const
{
    return mNumberOfSamples;
}

uint32_t Dataset::numberOfBatches() const
{
    if (datasetIsNotInitializedCorrectly())
    {
        throw std::logic_error("It is not possible to define how many batches in dataset until batch size is not defined, so call Dataset::generate method first");
    }

    return (mNumberOfSamples / mBatchSize) + (useIncompleteBatch && (mNumberOfSamples % mBatchSize) ? 1 : 0);
}

uint32_t Dataset::getBatchSize() const
{
    if (datasetIsNotInitializedCorrectly())
    {
        throw std::logic_error("Batch size is not defined, so call Dataset::generate method first");
    }

    return mBatchSize;
}

void Dataset::describePart(const Name& name, uint32_t numberOfSamples, uint32_t sampleDepth, uint32_t sampleHeight, uint32_t sampleWidth)
{
    if (partOfDatasetIsFound(name))
    {
        throw std::logic_error("Info about dataset part " + name + " already registered");
    }

    mParts.insert({ name, Part{ numberOfSamples, sampleDepth, sampleHeight, sampleWidth } });

    if (!mNumberOfSamples)
    {
        mNumberOfSamples = numberOfSamples;
    }
    if (mNumberOfSamples != numberOfSamples)
    {
        throw std::logic_error("An attempt to register parts of dataset with different number of samples detected");
    }
}

bool Dataset::partOfDatasetIsFound(const raul::Name& name) const
{
    return mParts.count(name) != 0;
}

void Dataset::setDataSourceFor(const Name& name, std::unique_ptr<LoadData> dataSource)
{
    if (partOfDatasetIsNotFound(name))
    {
        throw std::logic_error("part of dataset with name " + name + " is not registered");
    }

    auto part = mParts.find(name);
    part->second.setDataSource(std::move(dataSource));
}

bool Dataset::partOfDatasetIsNotFound(const raul::Name& name) const
{
    return mParts.count(name) == 0;
}

void Dataset::applyTo(const Name& name, std::unique_ptr<Transform> transformation)
{
    if (partOfDatasetIsNotFound(name))
    {
        throw std::logic_error("part of dataset with name " + name + " is not registered");
    }

    auto part = mParts.find(name);
    part->second.apply(std::move(transformation));
}

void Dataset::generate(uint32_t batchSize)
{
    mBatchSize = batchSize;
    if (needRandomizeBatchSequence)
    {
        mBatchIndexGenerator = std::make_unique<RandomSequence>(0u, numberOfBatches() - 1);
    }
    else
    {
        mBatchIndexGenerator = std::make_unique<MonotonicSequence>(0, numberOfBatches() - 1);
    }

    for (auto& part : mParts)
    {
        part.second.generate(batchSize, useIncompleteBatch);
    }
}

Dataset::DataBatch Dataset::getData()
{
    if (datasetIsNotInitializedCorrectly())
    {
        throw std::logic_error("Seems like method Dataset::generate is not called before getting data");
    }

    uint32_t batchIdx = mBatchIndexGenerator->getElement();
    for (auto& part : mParts)
    {
        mDataParts[part.first] = part.second.getData(batchIdx);
    }

    return DataBatch(mDataParts);
}

bool Dataset::datasetIsNotInitializedCorrectly() const
{
    return !mBatchSize;
}

Dataset Dataset::MNIST_Train(const std::filesystem::path& path)
{
    const dtype maxValueOfPixel = 255_dt;

    Dataset mnist;
    mnist.describePart("images", 60000, 1, 28, 28);
    mnist.setDataSourceFor("images", std::make_unique<LoadDataInIdxFormat>(path / "train-images-idx3-ubyte"));
    mnist.applyTo("images", std::make_unique<Normalize>(maxValueOfPixel));
    mnist.describePart("labels", 60000, 1, 1, 1);
    mnist.setDataSourceFor("labels", std::make_unique<LoadDataInIdxFormat>(path / "train-labels-idx1-ubyte"));
    mnist.applyTo("labels", std::make_unique<BuildOneHotVector>(10));

    return mnist;
}

Dataset Dataset::MNIST_Test(const std::filesystem::path& path)
{
    const dtype maxValueOfPixel = 255_dt;

    Dataset mnist;
    mnist.describePart("images", 10000, 1, 28, 28);
    mnist.setDataSourceFor("images", std::make_unique<LoadDataInIdxFormat>(path / "t10k-images-idx3-ubyte"));
    mnist.applyTo("images", std::make_unique<Normalize>(maxValueOfPixel));
    mnist.describePart("labels", 10000, 1, 1, 1);
    mnist.setDataSourceFor("labels", std::make_unique<LoadDataInIdxFormat>(path / "t10k-labels-idx1-ubyte"));
    mnist.applyTo("labels", std::make_unique<BuildOneHotVector>(10));

    return mnist;
}

Dataset Dataset::CIFAR_Train(const std::filesystem::path& path)
{
    const dtype maxValueOfPixel = 255_dt;

    Dataset cifar;
    cifar.describePart("images", 50000, 3, 32, 32);
    cifar.setDataSourceFor("images", std::make_unique<LoadDenselyPackedDataFromFileSequence>(path / "data_batch_", ".bin", 1, 0));
    cifar.applyTo("images", std::make_unique<Normalize>(maxValueOfPixel));
    cifar.describePart("labels", 50000, 1, 1, 1);
    cifar.setDataSourceFor("labels", std::make_unique<LoadDenselyPackedDataFromFileSequence>(path / "data_batch_", ".bin", 0, 3 * 32 * 32));
    cifar.applyTo("labels", std::make_unique<BuildOneHotVector>(10));

    return cifar;
}

Dataset Dataset::CIFAR_Test(const std::filesystem::path& path)
{
    const dtype maxValueOfPixel = 255_dt;

    Dataset cifar;
    cifar.describePart("images", 10000, 3, 32, 32);
    cifar.setDataSourceFor("images", std::make_unique<LoadDenselyPackedData>(path / "test_batch.bin", 1, 0));
    cifar.applyTo("images", std::make_unique<Normalize>(maxValueOfPixel));
    cifar.describePart("labels", 10000, 1, 1, 1);
    cifar.setDataSourceFor("labels", std::make_unique<LoadDenselyPackedData>(path / "test_batch.bin", 0, 3 * 32 * 32));
    cifar.applyTo("labels", std::make_unique<BuildOneHotVector>(10));

    return cifar;
}
