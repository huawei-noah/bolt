// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef DATASET_H
#define DATASET_H

#include <training/tools/DataLoad.h>
#include <training/tools/DataTransformations.h>
#include <training/tools/ElementSequence.h>

#include <unordered_map>

namespace raul
{

class Dataset
{
    class Part
    {
      public:
        Part(uint32_t numberOfSamples, uint32_t sampleDepth, uint32_t sampleHeight, uint32_t sampleWidth);

        void setDataSource(std::unique_ptr<LoadData> dataSource);
        void apply(std::unique_ptr<Transform> transformation);
        void generate(uint32_t batchSize, bool useIncompleteBatch);
        std::unique_ptr<Tensor> getData(uint32_t batchIndex);

      private:
        uint32_t numberOfBatches(uint32_t batchSize, bool useIncompleteBatch) const;
        void prepareToReloadData();

      private:
        std::vector<std::unique_ptr<Tensor>> mRawDataBatches;
        std::unique_ptr<LoadData> mLoadData = nullptr;
        std::unique_ptr<Transform> mTransformation = nullptr;
        uint32_t mNumberOfSamples;
        uint32_t mSampleDepth;
        uint32_t mSampleHeight;
        uint32_t mSampleWidth;
    };

  public:
    enum Option
    {
        USE_LAST_INCOMPLETE_BATCH,
        SKIP_LAST_INCOMPLETE_BATCH,
        RANDOMIZE_BATCH_SEQUENCE,
        TURN_OFF_BATCH_SEQUENCE_RANDOMIZATION
    };

  public:
    class DataBatch
    {
      public:
        DataBatch(const DataBatch&) = delete;
        DataBatch& operator=(const DataBatch&) = delete;
        DataBatch(DataBatch&&) = delete;
        DataBatch& operator=(DataBatch&&) = delete;

        const Tensor& get(const raul::Name& name) const;
        size_t numberOfSamples() const { return mDataParts.begin()->second->getBatchSize(); }

      private:
        explicit DataBatch(const std::unordered_map<std::string, std::unique_ptr<Tensor>>& dataParts)
            : mDataParts(dataParts)
        {
        }

      private:
        friend class Dataset;

      private:
        const std::unordered_map<std::string, std::unique_ptr<Tensor>>& mDataParts;
    };

  public:
    static Dataset MNIST_Train(const std::filesystem::path& path);
    static Dataset MNIST_Test(const std::filesystem::path& path);
    static Dataset CIFAR_Train(const std::filesystem::path& path);
    static Dataset CIFAR_Test(const std::filesystem::path& path);

    void configure(Option option);
    void describePart(const Name& name, uint32_t numberOfSamples, uint32_t sampleDepth, uint32_t sampleHeight, uint32_t sampleWidth);
    void setDataSourceFor(const Name& name, std::unique_ptr<LoadData> dataSource);
    void applyTo(const Name& name, std::unique_ptr<Transform> transformation);
    void generate(uint32_t batchSize);
    DataBatch getData();
    uint32_t numberOfSamples() const;
    uint32_t numberOfBatches() const;
    uint32_t getBatchSize() const;

  private:
    bool partOfDatasetIsNotFound(const raul::Name& name) const;
    bool partOfDatasetIsFound(const raul::Name& name) const;

    bool datasetIsNotInitializedCorrectly() const;

  private:
    std::unordered_map<std::string, Part> mParts;
    std::unordered_map<std::string, std::unique_ptr<Tensor>> mDataParts;
    std::unique_ptr<ElementSequence> mBatchIndexGenerator;
    uint32_t mNumberOfSamples = 0;
    uint32_t mBatchSize = 0;
    bool useIncompleteBatch = false;
    bool needRandomizeBatchSequence = false;
};

} // !namespace raul

#endif // DATASET_H
