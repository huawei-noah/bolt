// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <training/tools/DataLoad.h>

using namespace raul;

std::unique_ptr<Tensor> LoadBinaryData::operator()(size_t numberOfSamples, size_t sampleDepth, size_t sampleHeight, size_t sampleWidth)
{
    auto result = std::make_unique<Tensor>(numberOfSamples, sampleDepth, sampleHeight, sampleWidth);
    auto tmp = loadData(numberOfSamples, sampleDepth, sampleHeight, sampleWidth);
    std::transform(tmp->begin(), tmp->end(), result->begin(), [](uint8_t v) { return static_cast<dtype>(v); });
    return result;
}

LoadDataInIdxFormat::LoadDataInIdxFormat(const path& path)
    : mFile(path)
{
    Init();
}

void LoadDataInIdxFormat::Init()
{
    if (mStream.is_open())
    {
        mStream.close();
    }

    mStream.open(mFile, std::ios::binary);
    if (!mStream.is_open())
    {
        throw std::system_error(errno, std::system_category(), "Cannot open file " + mFile.string());
    }
    skipHeaderOfFileInIdxFormat();
}

static uint32_t moveHighByteToPositionOfLow(uint32_t val)
{
    return val >> 24;
}
void LoadDataInIdxFormat::skipHeaderOfFileInIdxFormat()
{
    // see http://yann.lecun.com/exdb/mnist/ or another place with IDX format description
    static constexpr uint32_t NUMBER_OF_ENCODED_DIMENSIONS_MASK = 0xFF000000;
    uint32_t magic = 0;
    mStream.read((char*)&magic, sizeof(magic));
    uint32_t numberOfEncodedDimensions = moveHighByteToPositionOfLow(magic & NUMBER_OF_ENCODED_DIMENSIONS_MASK);
    for (uint32_t dimIdx = 0; dimIdx < numberOfEncodedDimensions; ++dimIdx)
    {
        uint32_t val = 0;
        mStream.read((char*)&val, sizeof(val));
    }
}

void LoadDataInIdxFormat::prepareToReloadData()
{
    Init();
}

std::unique_ptr<TensorU8> LoadDataInIdxFormat::loadData(size_t numberOfSamples, size_t sampleDepth, size_t sampleHeight, size_t sampleWidth)
{
    auto result = std::make_unique<TensorU8>(numberOfSamples, sampleDepth, sampleHeight, sampleWidth);
    mStream.read((char*)result->data(), result->size());
    return result;
}

LoadDenselyPackedData::LoadDenselyPackedData(const path& path, size_t skipBeforeBytes, size_t skipAfterBytes)
    : mFile(path)
    , mSkipBeforeBytes(skipBeforeBytes)
    , mSkipAfterBytes(skipAfterBytes)
{
    Init();
}

void LoadDenselyPackedData::Init()
{
    if (mStream.is_open())
    {
        mStream.close();
    }

    mStream.open(mFile, std::ios::binary);
    if (!mStream.is_open())
    {
        throw std::system_error(errno, std::system_category(), "Cannot open file " + mFile.string());
    }
}

void LoadDenselyPackedData::prepareToReloadData()
{
    Init();
}

std::unique_ptr<TensorU8> LoadDenselyPackedData::loadData(size_t numberOfSamples, size_t sampleDepth, size_t sampleHeight, size_t sampleWidth)
{
    auto result = std::make_unique<TensorU8>(numberOfSamples, sampleDepth, sampleHeight, sampleWidth);
    size_t sampleSize = sampleDepth * sampleHeight * sampleWidth;
    for (size_t sampleIdx = 0; sampleIdx < numberOfSamples; ++sampleIdx)
    {
        mStream.seekg(mSkipBeforeBytes, std::ios::cur);
        mStream.read((char*)result->data() + sampleIdx * sampleSize, sampleSize);
        mStream.seekg(mSkipAfterBytes, std::ios::cur);
    }
    return result;
}

LoadDenselyPackedDataFromFileSequence::LoadDenselyPackedDataFromFileSequence(const path& path, const std::string& fNameSuffix, size_t skipBeforeBytes, size_t skipAfterBytes)
    : mFilePrefix(path)
    , mFileSuffix(fNameSuffix)
    , mSkipBeforeBytes(skipBeforeBytes)
    , mSkipAfterBytes(skipAfterBytes)
{
    openFirstFileInSequence();
}

void LoadDenselyPackedDataFromFileSequence::openFirstFileInSequence()
{
    if (mStream.is_open())
    {
        mStream.close();
    }

    std::filesystem::path pathToFileWithSuffix = mFilePrefix.string() + std::to_string(mFileIdxInSequence) + mFileSuffix;
    mStream.open(pathToFileWithSuffix, std::ios::binary);
    if (!mStream.is_open())
    {
        pathToFileWithSuffix = mFilePrefix.string() + std::to_string(++mFileIdxInSequence) + mFileSuffix;
        mStream.open(pathToFileWithSuffix, std::ios::binary);
        if (!mStream.is_open())
        {
            throw std::system_error(errno, std::system_category(), "Cannot open file " + pathToFileWithSuffix.string());
        }
    }
}

void LoadDenselyPackedDataFromFileSequence::prepareToReloadData()
{
    openFirstFileInSequence();
}

std::unique_ptr<TensorU8> LoadDenselyPackedDataFromFileSequence::loadData(size_t numberOfSamples, size_t sampleDepth, size_t sampleHeight, size_t sampleWidth)
{
    auto result = std::make_unique<TensorU8>(numberOfSamples, sampleDepth, sampleHeight, sampleWidth);
    size_t sampleSize = sampleDepth * sampleHeight * sampleWidth;
    for (size_t sampleIdx = 0; sampleIdx < numberOfSamples; ++sampleIdx)
    {
        skipBytesBeforeSample();
        mStream.read((char*)result->data() + sampleIdx * sampleSize, sampleSize);
        skipBytesAfterSample();
    }
    return result;
}

void LoadDenselyPackedDataFromFileSequence::skipBytesBeforeSample()
{
    mStream.seekg(mSkipBeforeBytes, std::ios::cur);
    if (endOfFileIsReached())
    {
        openNextFileInSequence();
        mStream.seekg(mSkipBeforeBytes, std::ios::cur);
    }
}

bool LoadDenselyPackedDataFromFileSequence::endOfFileIsReached()
{
    mStream.peek();
    return mStream.eof();
}

void LoadDenselyPackedDataFromFileSequence::openNextFileInSequence()
{
    mStream.close();
    std::filesystem::path pathToFileWithSuffix = mFilePrefix.string() + std::to_string(++mFileIdxInSequence) + mFileSuffix;
    mStream.open(pathToFileWithSuffix, std::ios::binary);
}

void LoadDenselyPackedDataFromFileSequence::skipBytesAfterSample()
{
    mStream.seekg(mSkipAfterBytes, std::ios::cur);
    if (endOfFileIsReached())
    {
        openNextFileInSequence();
    }
}

LoadDataInCustomNumpyFormat::LoadDataInCustomNumpyFormat(const path& path)
    : mFile(path)
{
    Init();
}

void LoadDataInCustomNumpyFormat::Init()
{
    if (mStream.is_open())
    {
        mStream.close();
    }

    mStream.open(mFile);
    if (!mStream.is_open())
    {
        throw std::system_error(errno, std::system_category(), "Cannot open file " + mFile.string());
    }
    skipHeader();
}

void LoadDataInCustomNumpyFormat::skipHeader()
{
    std::string lineFromFile;
    std::getline(mStream, lineFromFile);
}

void LoadDataInCustomNumpyFormat::prepareToReloadData()
{
    Init();
}

std::unique_ptr<Tensor> LoadDataInCustomNumpyFormat::operator()(size_t numberOfSamples, size_t sampleDepth, size_t sampleHeight, size_t sampleWidth)
{
    auto result = std::make_unique<Tensor>(numberOfSamples, sampleDepth, sampleHeight, sampleWidth);
    for (size_t elemIdx = 0, elemIdxThreshold = result->size(); elemIdx < elemIdxThreshold; ++elemIdx)
    {
        bool sourceContainsLessDataThenRequested = mStream.eof();
        if (sourceContainsLessDataThenRequested)
        {
            THROW_NONAME("DataLoad", "File with data contains only " + std::to_string(elemIdx) + ", but requested " + std::to_string(elemIdxThreshold));
        }

        mStream >> (*result)[elemIdx];
        if (mStream.fail())
        {
            repeatAttemptToReadElementWith(elemIdx);
        }
    }

    if (fileContainsMoreDataThenRequested())
    {
        THROW_NONAME("DataLoad", "File " + mFile.string() + " contains more data then requested");
    }

    return result;
}

void LoadDataInCustomNumpyFormat::repeatAttemptToReadElementWith(size_t& elemIdx)
{
    mStream.clear();
    mStream.ignore();
    --elemIdx;
}

bool LoadDataInCustomNumpyFormat::fileContainsMoreDataThenRequested()
{
    while (!mStream.eof())
    {
        dtype tmp = 0_dt;
        mStream >> tmp;
        if (mStream.fail())
        {
            mStream.clear();
            mStream.ignore();
        }

        if (tmp != 0_dt)
        {
            return true;
        }
    }

    return false;
}
