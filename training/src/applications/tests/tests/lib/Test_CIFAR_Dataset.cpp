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

struct TestCifar : public testing::Test
{
    size_t numberOfElementsInDataset = 50000;
    std::ifstream imagesDatasetStream;
    size_t imagesDatasetFileIdx = 1;
    size_t sizeOfImageSample = 3 * 32 * 32;
    size_t currentImageElemIdx = 0;
    std::ifstream labelsDatasetStream;
    size_t labelsDatasetFileIdx = 1;
    size_t sizeOfLabelSample = 1;
    size_t currentLabelElemIdx = 0;

    dtype getElemFromImagesDataset()
    {
        if (currentImageElemIdx == 0)
        {
            imagesDatasetStream.seekg(sizeOfLabelSample, std::ios::cur);
        }
        if (!imagesDatasetStream.is_open() || endOfFileIsReached(imagesDatasetStream))
        {
            imagesDatasetStream.close();
            imagesDatasetStream.open(UT::tools::getTestAssetsDir() / "CIFAR" / ("data_batch_" + std::to_string(imagesDatasetFileIdx++) + ".bin"), std::ios::binary);
            if (currentImageElemIdx == 0)
            {
                imagesDatasetStream.seekg(sizeOfLabelSample, std::ios::cur);
            }
        }

        uint8_t c = 0;
        imagesDatasetStream.read((char*)&c, sizeof(c));
        ++currentImageElemIdx;
        if (currentImageElemIdx >= sizeOfImageSample)
        {
            currentImageElemIdx = 0;
        }

        return c / 255_dt;
    }

    std::vector<dtype> getElemFromLabelsDataset()
    {
        if (!labelsDatasetStream.is_open() || endOfFileIsReached(labelsDatasetStream))
        {
            labelsDatasetStream.close();
            labelsDatasetStream.open(UT::tools::getTestAssetsDir() / "CIFAR" / ("data_batch_" + std::to_string(labelsDatasetFileIdx++) + ".bin"), std::ios::binary);
        }

        char c = 0;
        labelsDatasetStream.read(&c, sizeof(c));
        std::vector<dtype> result(10, 0_dt);
        result[c] = 1_dt;
        ++currentLabelElemIdx;
        if (currentLabelElemIdx >= sizeOfLabelSample)
        {
            currentLabelElemIdx = 0;
        }
        labelsDatasetStream.seekg(sizeOfImageSample, std::ios::cur);

        return result;
    }

    bool endOfFileIsReached(std::ifstream& in)
    {
        in.peek();
        return in.eof();
    }

  private:
};

TEST_F(TestCifar, ShouldCorrectlyReturnNumberOfBatchesAccordingBatchSizeUnit)
{
    PROFILE_TEST
    uint32_t batchSize = 64;
    Dataset dataset = Dataset::CIFAR_Train(UT::tools::getTestAssetsDir() / "CIFAR");
    dataset.generate(batchSize);
    ASSERT_EQ(dataset.numberOfBatches(), numberOfElementsInDataset / batchSize);
}

TEST_F(TestCifar, ShouldReturnPartOfDataOnDemandUnit)
{
    PROFILE_TEST
    uint32_t batchSize = 64;
    Dataset dataset = Dataset::CIFAR_Train(UT::tools::getTestAssetsDir() / "CIFAR");
    dataset.generate(batchSize);
    for (size_t batchIdx = 0; batchIdx < dataset.numberOfBatches(); ++batchIdx)
    {
        size_t elemCount = 0;
        auto dataBatch = dataset.getData();
        const Tensor& images = dataBatch.get("images");
        for (auto elem = images.begin(); elem != images.end(); ++elem)
        {
            auto ref = getElemFromImagesDataset();
            if (*elem != ref)
            {
                std::cout << "b " << batchIdx << " e " << elemCount << "\n";
            }
            ASSERT_NEAR(*elem, ref, 1e-6_dt);
            ++elemCount;
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
