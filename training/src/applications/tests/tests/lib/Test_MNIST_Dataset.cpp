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

#include <training/tools/Dataset.h>

using namespace raul;

struct TestMnist : public testing::Test
{
    size_t numberOfElementsInDataset = 60000;
    std::ifstream imagesDatasetStream;
    std::ifstream labelsDatasetStream;

    void SetUp() final
    {
        imagesDatasetStream.open(UT::tools::getTestAssetsDir() / "MNIST" / "train-images-idx3-ubyte", std::ios::binary);
        skipHeaderOfImagesDataset();
        labelsDatasetStream.open(UT::tools::getTestAssetsDir() / "MNIST" / "train-labels-idx1-ubyte", std::ios::binary);
        skipHeaderOfLabelsDataset();
    }

    void TearDown() final
    {
        imagesDatasetStream.close();
        labelsDatasetStream.close();
    }

    dtype getElemFromImagesDataset()
    {
        uint8_t c = 0;
        imagesDatasetStream.read((char*)&c, sizeof(c));
        return c / 255_dt;
    }

    std::vector<dtype> getElemFromLabelsDataset()
    {
        char c = 0;
        labelsDatasetStream.read(&c, sizeof(c));
        std::vector<dtype> result(10, 0_dt);
        result[c] = 1_dt;
        return result;
    }

  private:
    void skipHeaderOfImagesDataset()
    {
        char buf[sizeof(uint32_t)] = {};
        imagesDatasetStream.read(buf, sizeof(uint32_t)); // skip magic
        imagesDatasetStream.read(buf, sizeof(uint32_t)); // skip number of images
        imagesDatasetStream.read(buf, sizeof(uint32_t)); // skip number of rows
        imagesDatasetStream.read(buf, sizeof(uint32_t)); // skip number of columns
    }

    void skipHeaderOfLabelsDataset()
    {
        char buf[sizeof(uint32_t)] = {};
        labelsDatasetStream.read(buf, sizeof(uint32_t)); // skip magic
        labelsDatasetStream.read(buf, sizeof(uint32_t)); // skip number of labels
    }
};

TEST_F(TestMnist, ShouldCorrectlyReturnNumberOfBatchesAccordingBatchSizeUnit)
{
    PROFILE_TEST
    uint32_t batchSize = 64;
    Dataset dataset = Dataset::MNIST_Train(UT::tools::getTestAssetsDir() / "MNIST");
    dataset.generate(batchSize);
    ASSERT_EQ(dataset.numberOfBatches(), numberOfElementsInDataset / batchSize);
}

TEST_F(TestMnist, ShouldReturnPartOfDataOnDemandUnit)
{
    PROFILE_TEST
    uint32_t batchSize = 64;
    Dataset dataset = Dataset::MNIST_Train(UT::tools::getTestAssetsDir() / "MNIST");
    dataset.generate(batchSize);
    for (size_t batchIdx = 0; batchIdx < dataset.numberOfBatches(); ++batchIdx)
    {
        auto dataBatch = dataset.getData();
        const Tensor& images = dataBatch.get("images");
        for (auto elem = images.begin(); elem != images.end(); ++elem)
        {
            ASSERT_NEAR(*elem, getElemFromImagesDataset(), 1e-6_dt);
        }

        const Tensor& labels = dataBatch.get("labels");
        for (auto elem = labels.begin(); elem != labels.end();)
        {
            auto ref = getElemFromLabelsDataset();
            for (auto refElem : ref)
            {
                ASSERT_NEAR(*elem, refElem, 1e-6_dt);
                ++elem;
            }
        }
    }
}
