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

#include <filesystem>
#include <fstream>

#include <training/tools/DataLoad.h>

#include <tests/tools/TestTools.h>
using namespace raul;

struct TestLoadDataFromFileInIdxFormat : public testing::Test
{
    size_t NUMBER_OF_TEST_DATA_SAMPLES = 6;
    size_t BATCH_SIZE = 2;
    std::filesystem::path fileWithTestData = "./test_data";
    std::vector<std::unique_ptr<TensorU8>> testData;

    void SetUp() final
    {
        std::ofstream out(fileWithTestData, std::ios::binary);
        uint32_t data = 0x02080000;
        out.write((const char*)&data, sizeof(data));
        uint32_t dim1 = 0x06000000;
        out.write((const char*)&dim1, sizeof(dim1));
        uint32_t dim2 = 0x02000000;
        out.write((const char*)&dim2, sizeof(dim2));

        uint8_t fillVal = 0;
        for (size_t i = 0; i < NUMBER_OF_TEST_DATA_SAMPLES / BATCH_SIZE; ++i)
        {
            auto tensor = std::make_unique<TensorU8>(BATCH_SIZE, 1, 2, 3);
            for (auto it = tensor->begin(); it != tensor->end(); ++it)
            {
                *it = fillVal++;
            }
            out.write((const char*)tensor->data(), tensor->size());
            testData.emplace_back(std::move(tensor));
        }
    }

    void TearDown() final
    {
        std::filesystem::remove(fileWithTestData);
        testData.clear();
    }
};

TEST_F(TestLoadDataFromFileInIdxFormat, CanLoadDataBatchByBatchUnit)
{
    PROFILE_TEST
    LoadDataInIdxFormat loadData(fileWithTestData);
    for (size_t batchIdx = 0; batchIdx < NUMBER_OF_TEST_DATA_SAMPLES / BATCH_SIZE; ++batchIdx)
    {
        auto ref = Tensor(BATCH_SIZE, 1, 2, 3);
        std::transform(testData[batchIdx]->begin(), testData[batchIdx]->end(), ref.begin(), [](uint8_t v) { return static_cast<dtype>(v); });
        auto result = loadData(BATCH_SIZE, 1, 2, 3);
        ASSERT_FLOAT_TENSORS_EQ((*result), ref, 1e-6_dt);
    }
}

TEST_F(TestLoadDataFromFileInIdxFormat, ShouldBePreparedToReloadDataOnDemandUnit)
{
    PROFILE_TEST
    LoadDataInIdxFormat loadData(fileWithTestData);
    auto ref = Tensor(BATCH_SIZE, 1, 2, 3);
    std::transform(testData[0]->begin(), testData[0]->end(), ref.begin(), [](uint8_t v) { return static_cast<dtype>(v); });

    auto result = loadData(BATCH_SIZE, 1, 2, 3);
    ASSERT_FLOAT_TENSORS_EQ((*result), ref, 1e-6_dt);

    loadData.prepareToReloadData();
    result = loadData(BATCH_SIZE, 1, 2, 3);
    ASSERT_FLOAT_TENSORS_EQ((*result), ref, 1e-6_dt);
}

TEST_F(TestLoadDataFromFileInIdxFormat, ThrowAnExceptionIfFileWithDataCannotBeOpenedUnit)
{
    PROFILE_TEST
    ASSERT_THROW(LoadDataInIdxFormat("unknown_file"), std::system_error);
}

struct TestLoadDataDenselyPackedFromFileSequence : public testing::Test
{
    static constexpr size_t NUMBER_OF_TEST_DATA_SAMPLES = 6;
    static constexpr size_t BATCH_SIZE = 2;
    std::filesystem::path filesWithTestData[2] = { "./test_data_0", "./test_data_1" };
    std::vector<std::unique_ptr<TensorU8>> testData1;
    std::vector<std::unique_ptr<TensorU8>> testData2;

    void SetUp() final
    {
        uint8_t fillVal = 0;
        for (size_t i = 0; i < NUMBER_OF_TEST_DATA_SAMPLES / BATCH_SIZE; ++i)
        {
            auto tensor = std::make_unique<TensorU8>(BATCH_SIZE, 3, 2, 3);
            for (auto it = tensor->begin(); it != tensor->end(); ++it)
            {
                *it = fillVal++;
            }
            testData1.emplace_back(std::move(tensor));
            tensor = std::make_unique<TensorU8>(BATCH_SIZE, 1, 1, 1);
            for (auto it = tensor->begin(); it != tensor->end(); ++it)
            {
                *it = fillVal++;
            }
            testData2.emplace_back(std::move(tensor));
        }

        std::ofstream out(filesWithTestData[0], std::ios::binary);
        out.write((const char*)testData1[0]->data(), 3 * 2 * 3);
        out.write((const char*)testData2[0]->data(), 1 * 1 * 1);
        out.write((const char*)testData1[0]->data() + 3 * 2 * 3, 3 * 2 * 3);
        out.write((const char*)testData2[0]->data() + 1 * 1 * 1, 1 * 1 * 1);
        out.write((const char*)testData1[1]->data(), 3 * 2 * 3);
        out.write((const char*)testData2[1]->data(), 1 * 1 * 1);
        out.close();

        out.open(filesWithTestData[1], std::ios::binary);
        out.write((const char*)testData1[1]->data() + 3 * 2 * 3, 3 * 2 * 3);
        out.write((const char*)testData2[1]->data() + 1 * 1 * 1, 1 * 1 * 1);
        out.write((const char*)testData1[2]->data(), 3 * 2 * 3);
        out.write((const char*)testData2[2]->data(), 1 * 1 * 1);
        out.write((const char*)testData1[2]->data() + 3 * 2 * 3, 3 * 2 * 3);
        out.write((const char*)testData2[2]->data() + 1 * 1 * 1, 1 * 1 * 1);
        out.close();
    }

    void TearDown() final
    {
        std::filesystem::remove(filesWithTestData[0]);
        std::filesystem::remove(filesWithTestData[1]);
        testData1.clear();
        testData2.clear();
    }
};

TEST_F(TestLoadDataDenselyPackedFromFileSequence, CanLoadDataBatchByBatchUnit)
{
    PROFILE_TEST
    LoadDenselyPackedDataFromFileSequence loadData1("./test_data_", "", 0, 1 * 1 * 1);
    for (size_t batchIdx = 0; batchIdx < NUMBER_OF_TEST_DATA_SAMPLES / BATCH_SIZE; ++batchIdx)
    {
        auto ref = Tensor(BATCH_SIZE, 3, 2, 3);
        std::transform(testData1[batchIdx]->begin(), testData1[batchIdx]->end(), ref.begin(), [](uint8_t v) { return static_cast<dtype>(v); });
        auto result = loadData1(BATCH_SIZE, 3, 2, 3);
        ASSERT_FLOAT_TENSORS_EQ((*result), ref, 1e-6_dt);
    }

    LoadDenselyPackedDataFromFileSequence loadData2("./test_data_", "", 3 * 2 * 3, 0);
    for (size_t batchIdx = 0; batchIdx < NUMBER_OF_TEST_DATA_SAMPLES / BATCH_SIZE; ++batchIdx)
    {
        auto ref = Tensor(BATCH_SIZE, 1, 1, 1);
        std::transform(testData2[batchIdx]->begin(), testData2[batchIdx]->end(), ref.begin(), [](uint8_t v) { return static_cast<dtype>(v); });
        auto result = loadData2(BATCH_SIZE, 1, 1, 1);
        ASSERT_FLOAT_TENSORS_EQ((*result), ref, 1e-6_dt);
    }
}

TEST_F(TestLoadDataDenselyPackedFromFileSequence, ShouldBePreparedToReloadDataOnDemandUnit)
{
    PROFILE_TEST
    LoadDenselyPackedDataFromFileSequence loadData1("./test_data_", "", 0, 1 * 1 * 1);
    LoadDenselyPackedDataFromFileSequence loadData2("./test_data_", "", 3 * 2 * 3, 0);
    auto ref1 = Tensor(BATCH_SIZE, 3, 2, 3);
    std::transform(testData1[0]->begin(), testData1[0]->end(), ref1.begin(), [](uint8_t v) { return static_cast<dtype>(v); });
    auto ref2 = Tensor(BATCH_SIZE, 1, 1, 1);
    std::transform(testData2[0]->begin(), testData2[0]->end(), ref2.begin(), [](uint8_t v) { return static_cast<dtype>(v); });

    auto result1 = loadData1(BATCH_SIZE, 3, 2, 3);
    ASSERT_FLOAT_TENSORS_EQ((*result1), ref1, 1e-6_dt);

    auto result2 = loadData2(BATCH_SIZE, 1, 1, 1);
    ASSERT_FLOAT_TENSORS_EQ((*result2), ref2, 1e-6_dt);

    loadData1.prepareToReloadData();
    loadData2.prepareToReloadData();

    result1 = loadData1(BATCH_SIZE, 3, 2, 3);
    ASSERT_FLOAT_TENSORS_EQ((*result1), ref1, 1e-6_dt);

    result2 = loadData2(BATCH_SIZE, 1, 1, 1);
    ASSERT_FLOAT_TENSORS_EQ((*result2), ref2, 1e-6_dt);
}

TEST_F(TestLoadDataDenselyPackedFromFileSequence, ThrowAnExceptionIfFileWithDataCannotBeOpenedUnit)
{
    PROFILE_TEST
    ASSERT_THROW(LoadDenselyPackedDataFromFileSequence("unknown_file", ".bin", 0, 0), std::system_error);
}

TEST_F(TestLoadDataDenselyPackedFromFileSequence, ShouldCheckThatFileSequenceStartsNotFromZeroUnit)
{
    PROFILE_TEST
    std::filesystem::remove(filesWithTestData[0]);
    ASSERT_NO_THROW(LoadDenselyPackedDataFromFileSequence("test_data_", "", 0, 0));
}

struct TestLoadDataDenselyPacked : public testing::Test
{
    static constexpr size_t NUMBER_OF_TEST_DATA_SAMPLES = 6;
    static constexpr size_t BATCH_SIZE = 2;
    std::filesystem::path fileWithTestData = "./test_data";
    std::vector<std::unique_ptr<TensorU8>> testData1;
    std::vector<std::unique_ptr<TensorU8>> testData2;

    void SetUp() final
    {
        uint8_t fillVal = 0;
        for (size_t i = 0; i < NUMBER_OF_TEST_DATA_SAMPLES / BATCH_SIZE; ++i)
        {
            auto tensor = std::make_unique<TensorU8>(BATCH_SIZE, 3, 2, 3);
            for (auto it = tensor->begin(); it != tensor->end(); ++it)
            {
                *it = fillVal++;
            }
            testData1.emplace_back(std::move(tensor));
            tensor = std::make_unique<TensorU8>(BATCH_SIZE, 1, 1, 1);
            for (auto it = tensor->begin(); it != tensor->end(); ++it)
            {
                *it = fillVal++;
            }
            testData2.emplace_back(std::move(tensor));
        }

        std::ofstream out(fileWithTestData, std::ios::binary);
        out.write((const char*)testData1[0]->data(), 3 * 2 * 3);
        out.write((const char*)testData2[0]->data(), 1 * 1 * 1);
        out.write((const char*)testData1[0]->data() + 3 * 2 * 3, 3 * 2 * 3);
        out.write((const char*)testData2[0]->data() + 1 * 1 * 1, 1 * 1 * 1);
        out.write((const char*)testData1[1]->data(), 3 * 2 * 3);
        out.write((const char*)testData2[1]->data(), 1 * 1 * 1);
        out.write((const char*)testData1[1]->data() + 3 * 2 * 3, 3 * 2 * 3);
        out.write((const char*)testData2[1]->data() + 1 * 1 * 1, 1 * 1 * 1);
        out.write((const char*)testData1[2]->data(), 3 * 2 * 3);
        out.write((const char*)testData2[2]->data(), 1 * 1 * 1);
        out.write((const char*)testData1[2]->data() + 3 * 2 * 3, 3 * 2 * 3);
        out.write((const char*)testData2[2]->data() + 1 * 1 * 1, 1 * 1 * 1);
        out.close();
    }

    void TearDown() final
    {
        std::filesystem::remove(fileWithTestData);
        testData1.clear();
    }
};

TEST_F(TestLoadDataDenselyPacked, CanLoadDataBatchByBatchUnit)
{
    PROFILE_TEST
    LoadDenselyPackedData loadData1("./test_data", 0, 1 * 1 * 1);
    for (size_t batchIdx = 0; batchIdx < NUMBER_OF_TEST_DATA_SAMPLES / BATCH_SIZE; ++batchIdx)
    {
        auto ref = Tensor(BATCH_SIZE, 3, 2, 3);
        std::transform(testData1[batchIdx]->begin(), testData1[batchIdx]->end(), ref.begin(), [](uint8_t v) { return static_cast<dtype>(v); });
        auto result = loadData1(BATCH_SIZE, 3, 2, 3);
        ASSERT_FLOAT_TENSORS_EQ((*result), ref, 1e-6_dt);
    }

    LoadDenselyPackedData loadData2("./test_data", 3 * 2 * 3, 0);
    for (size_t batchIdx = 0; batchIdx < NUMBER_OF_TEST_DATA_SAMPLES / BATCH_SIZE; ++batchIdx)
    {
        auto ref = Tensor(BATCH_SIZE, 1, 1, 1);
        std::transform(testData2[batchIdx]->begin(), testData2[batchIdx]->end(), ref.begin(), [](uint8_t v) { return static_cast<dtype>(v); });
        auto result = loadData2(BATCH_SIZE, 1, 1, 1);
        ASSERT_FLOAT_TENSORS_EQ((*result), ref, 1e-6_dt);
    }
}

TEST_F(TestLoadDataDenselyPacked, ShouldBePreparedToReloadDataUnit)
{
    PROFILE_TEST
    LoadDenselyPackedData loadData1("./test_data", 0, 1 * 1 * 1);
    LoadDenselyPackedData loadData2("./test_data", 3 * 2 * 3, 0);
    auto ref1 = Tensor(BATCH_SIZE, 3, 2, 3);
    std::transform(testData1[0]->begin(), testData1[0]->end(), ref1.begin(), [](uint8_t v) { return static_cast<dtype>(v); });
    auto ref2 = Tensor(BATCH_SIZE, 1, 1, 1);
    std::transform(testData2[0]->begin(), testData2[0]->end(), ref2.begin(), [](uint8_t v) { return static_cast<dtype>(v); });

    auto result1 = loadData1(BATCH_SIZE, 3, 2, 3);
    ASSERT_FLOAT_TENSORS_EQ((*result1), ref1, 1e-6_dt);

    auto result2 = loadData2(BATCH_SIZE, 1, 1, 1);
    ASSERT_FLOAT_TENSORS_EQ((*result2), ref2, 1e-6_dt);

    loadData1.prepareToReloadData();
    loadData2.prepareToReloadData();

    result1 = loadData1(BATCH_SIZE, 3, 2, 3);
    ASSERT_FLOAT_TENSORS_EQ((*result1), ref1, 1e-6_dt);

    result2 = loadData2(BATCH_SIZE, 1, 1, 1);
    ASSERT_FLOAT_TENSORS_EQ((*result2), ref2, 1e-6_dt);
}

TEST_F(TestLoadDataDenselyPacked, ThrowAnExceptionIfFileWithDataCannotBeOpenedUnit)
{
    PROFILE_TEST
    ASSERT_THROW(LoadDenselyPackedData("unknown_file.bin", 0, 0), std::system_error);
}

struct TestLoadDataInCustomNumpyFormat : public testing::Test
{
    static std::string testData1;
    static std::string testData2;
    static std::string testData3;
    static std::string testData4;

    size_t NumberOfSamples = 3;
    size_t SampleDepth = 2;
    size_t SampleHeight = 3;
    size_t SampleWidth = 2;

    Tensor refData1 = Tensor(3, 2, 3, 2, { 1_dt,  2_dt,  3_dt,  4_dt,  5_dt,  6_dt,  7_dt,  8_dt,  9_dt,  10_dt, 11_dt, 12_dt, 13_dt, 14_dt, 15_dt, 16_dt, 17_dt, 18_dt,
                                           19_dt, 20_dt, 21_dt, 22_dt, 23_dt, 24_dt, 25_dt, 26_dt, 27_dt, 28_dt, 29_dt, 30_dt, 31_dt, 32_dt, 33_dt, 34_dt, 35_dt, 36_dt });

    Tensor refData2 = Tensor(3, 1, 3, 2, { 1_dt, 2_dt, 3_dt, 4_dt, 5_dt, 6_dt, 7_dt, 8_dt, 9_dt, 10_dt, 11_dt, 12_dt, 13_dt, 14_dt, 15_dt, 16_dt, 17_dt, 18_dt });
    Tensor refData3 = Tensor(3, 1, 1, 2, { 1_dt, 2_dt, 3_dt, 4_dt, 5_dt, 6_dt });
    Tensor refData4 = Tensor(3, 1, 1, 1, { 1_dt, 2_dt, 3_dt });

    void SetUp() final
    {
        std::ofstream out1("./test_file1.data");
        out1 << testData1;

        std::ofstream out2("./test_file2.data");
        out2 << testData2;

        std::ofstream out3("./test_file3.data");
        out3 << testData3;

        std::ofstream out4("./test_file4.data");
        out4 << testData4;
    }

    void TearDown() final
    {
        std::filesystem::remove("./test_file1.data");
        std::filesystem::remove("./test_file2.data");
        std::filesystem::remove("./test_file3.data");
        std::filesystem::remove("./test_file4.data");
    }
};

TEST_F(TestLoadDataInCustomNumpyFormat, CanLoadDataUnit)
{
    {
        LoadDataInCustomNumpyFormat loadData("./test_file1.data");
        auto result = loadData(NumberOfSamples, SampleDepth, SampleHeight, SampleWidth);
        ASSERT_FLOAT_TENSORS_EQ((*result), refData1, 1e-6_dt);
    }
    {
        LoadDataInCustomNumpyFormat loadData("./test_file2.data");
        auto result = loadData(NumberOfSamples, 1, SampleHeight, SampleWidth);
        ASSERT_FLOAT_TENSORS_EQ((*result), refData2, 1e-6_dt);
    }
    {
        LoadDataInCustomNumpyFormat loadData("./test_file3.data");
        auto result = loadData(NumberOfSamples, 1, 1, SampleWidth);
        ASSERT_FLOAT_TENSORS_EQ((*result), refData3, 1e-6_dt);
    }
    {
        LoadDataInCustomNumpyFormat loadData("./test_file4.data");
        auto result = loadData(NumberOfSamples, 1, 1, 1);
        ASSERT_FLOAT_TENSORS_EQ((*result), refData4, 1e-6_dt);
    }
}

TEST_F(TestLoadDataInCustomNumpyFormat, ShouldThrowAnExceptionIfAnAttemptToOpenFileFailedUnit)
{
    ASSERT_THROW(LoadDataInCustomNumpyFormat("unknown_file.data"), std::system_error);
}

TEST_F(TestLoadDataInCustomNumpyFormat, ShouldBePreparedToReloadDataUnit)
{
    LoadDataInCustomNumpyFormat loadData("./test_file1.data");

    auto result = loadData(NumberOfSamples, SampleDepth, SampleHeight, SampleWidth);
    ASSERT_FLOAT_TENSORS_EQ((*result), refData1, 1e-6_dt);

    loadData.prepareToReloadData();

    result = loadData(NumberOfSamples, SampleDepth, SampleHeight, SampleWidth);
    ASSERT_FLOAT_TENSORS_EQ((*result), refData1, 1e-6_dt);
}

TEST_F(TestLoadDataInCustomNumpyFormat, ShouldThrowAnExceptionIfReadLessDataThenRequestedUnit)
{
    LoadDataInCustomNumpyFormat loadData("./test_file2.data");
    ASSERT_THROW(loadData(NumberOfSamples, SampleDepth, SampleHeight, SampleWidth), raul::Exception);
}

TEST_F(TestLoadDataInCustomNumpyFormat, ShouldThrowAnExceptionIfSourceContainsMoreDataThenRequestedUnit)
{
    LoadDataInCustomNumpyFormat loadData("./test_file1.data");
    ASSERT_THROW(loadData(NumberOfSamples, 1, SampleHeight, SampleWidth), raul::Exception);
}

std::string TestLoadDataInCustomNumpyFormat::testData1("[3 2 3 2]\n"
                                                       "[[[[ 1  2]\n"
                                                       "   [ 3  4]\n"
                                                       "   [ 5  6]]\n"
                                                       "\n"
                                                       "  [[ 7  8]\n"
                                                       "   [ 9 10]\n"
                                                       "   [11 12]]]\n"
                                                       "\n"
                                                       "\n"
                                                       " [[[13 14]\n"
                                                       "   [15 16]\n"
                                                       "   [17 18]]\n"
                                                       "\n"
                                                       "  [[19 20]\n"
                                                       "   [21 22]\n"
                                                       "   [23 24]]]\n"
                                                       "\n"
                                                       "\n"
                                                       " [[[25 26]\n"
                                                       "   [27 28]\n"
                                                       "   [29 30]]\n"
                                                       "\n"
                                                       "  [[31 32]\n"
                                                       "   [33 34]\n"
                                                       "   [35 36]]]]\n");

std::string TestLoadDataInCustomNumpyFormat::testData2("[3 3 2]\n"
                                                       "[[[ 1  2]\n"
                                                       "  [ 3  4]\n"
                                                       "  [ 5  6]]\n"
                                                       "\n"
                                                       " [[ 7  8]\n"
                                                       "  [ 9 10]\n"
                                                       "  [11 12]]\n"
                                                       "\n"
                                                       " [[13 14]\n"
                                                       "  [15 16]\n"
                                                       "  [17 18]]]\n");

std::string TestLoadDataInCustomNumpyFormat::testData3("[3 2]\n"
                                                       "[[1 2]\n"
                                                       " [3 4]\n"
                                                       " [5 6]]\n");

std::string TestLoadDataInCustomNumpyFormat::testData4("[3]\n"
                                                       "[1 2 3]\n");
