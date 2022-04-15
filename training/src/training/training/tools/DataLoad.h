// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef DATALOAD_H
#define DATALOAD_H

#include <filesystem>
#include <fstream>

#include <training/common/Tensor.h>

namespace raul
{

struct LoadData
{
    virtual void prepareToReloadData() = 0;
    virtual std::unique_ptr<Tensor> operator()(size_t numberOfSamples, size_t sampleDepth, size_t sampleHeight, size_t sampleWidth) = 0;

    virtual ~LoadData() = default;
};

struct LoadBinaryData : public LoadData
{
    std::unique_ptr<Tensor> operator()(size_t numberOfSamples, size_t sampleDepth, size_t sampleHeight, size_t sampleWidth) final;

  private:
    virtual std::unique_ptr<TensorU8> loadData(size_t numberOfSamples, size_t sampleDepth, size_t sampleHeight, size_t sampleWidth) = 0;
};

class LoadDataInIdxFormat : public LoadBinaryData
{
    using path = std::filesystem::path;

  public:
    LoadDataInIdxFormat(const path& path);

    void prepareToReloadData() final;

  private:
    void Init();
    std::unique_ptr<TensorU8> loadData(size_t numberOfSamples, size_t sampleDepth, size_t sampleHeight, size_t sampleWidth) final;
    void skipHeaderOfFileInIdxFormat();

  private:
    path mFile;
    std::ifstream mStream;
};

class LoadDenselyPackedData : public LoadBinaryData
{
    using path = std::filesystem::path;

  public:
    LoadDenselyPackedData(const path& path, size_t skipBeforeBytes, size_t skipAfterBytes);

    void prepareToReloadData() final;

  private:
    void Init();
    std::unique_ptr<TensorU8> loadData(size_t numberOfSamples, size_t sampleDepth, size_t sampleHeight, size_t sampleWidth) final;

  private:
    path mFile;
    std::ifstream mStream;
    size_t mSkipBeforeBytes;
    size_t mSkipAfterBytes;
};

class LoadDenselyPackedDataFromFileSequence : public LoadBinaryData
{
    using path = std::filesystem::path;

  public:
    LoadDenselyPackedDataFromFileSequence(const path& path, const std::string& fNameSuffix, size_t skipBeforeBytes, size_t skipAfterBytes);

    void prepareToReloadData() final;

  private:
    std::unique_ptr<TensorU8> loadData(size_t numberOfSamples, size_t sampleDepth, size_t sampleHeight, size_t sampleWidth) final;
    void openFirstFileInSequence();
    void openNextFileInSequence();

    void skipBytesBeforeSample();
    void skipBytesAfterSample();

    bool endOfFileIsReached();

  private:
    path mFilePrefix;
    std::string mFileSuffix;
    std::ifstream mStream;
    size_t mSkipBeforeBytes;
    size_t mSkipAfterBytes;
    size_t mFileIdxInSequence = 0;
};

class LoadDataInCustomNumpyFormat : public LoadData
{
    using path = std::filesystem::path;

  public:
    LoadDataInCustomNumpyFormat(const path& path);

    void prepareToReloadData() final;

    std::unique_ptr<Tensor> operator()(size_t numberOfSamples, size_t sampleDepth, size_t sampleHeight, size_t sampleWidth) final;

  private:
    void Init();
    void skipHeader();
    void repeatAttemptToReadElementWith(size_t& elemIdx);
    bool fileContainsMoreDataThenRequested();

  private:
    std::ifstream mStream;
    path mFile;
};

} // !namespace raul

#endif // DATALOAD_H
