// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef DATASETS_H
#define DATASETS_H

#include <filesystem>
#include <string>

#include <training/common/DataLoader.h>
#include <training/common/Tensor.h>
#include <training/network/Workflow.h>
#include <training/optimizers/IOptimizer.h>
#include <training/optimizers/Optimizer.h>

namespace raul
{

class Datasets
{
  protected:
    template<typename T>
    void checkInputSize(const T& inputs, size_t batchSize) const;
    void checkInputSize(const raul::Tensor& inputs, size_t batchSize, const std::pair<size_t, size_t>& rescaleSize) const;

    long long mTimeTaken;
    long long mTimeTakenTest;

    typedef std::unique_ptr<TensorU8> TensorMem;

    TensorMem mTrainImages;
    TensorMem mTrainLabels;
    TensorMem mTestImages;
    TensorMem mTestLabels;

    size_t mTrainAmount;
    size_t mTestAmount;
    size_t mImageSize;
    size_t mImageDepth;
    size_t mNumClasses;
    const Tensor* mEncodedTrainLabels;
    const Tensor* mEncodedTestLabels;

    raul::DataLoader mDataLoader;

  public:
    Datasets()
        : mTimeTaken(0)
        , mTimeTakenTest(0)
        , mTrainImages(std::make_unique<TensorU8>(0))
        , mTrainLabels(std::make_unique<TensorU8>(0))
        , mTestImages(std::make_unique<TensorU8>(0))
        , mTestLabels(std::make_unique<TensorU8>(0))
        , mTrainAmount(0)
        , mTestAmount(0)
        , mImageSize(0)
        , mImageDepth(0)
        , mNumClasses(0)
    {
    }
    virtual ~Datasets() = default;

    virtual bool loadingData(const std::filesystem::path& path) = 0;

    virtual raul::dtype oneTrainIteration(raul::Workflow& network, raul::optimizers::IOptimizer*, size_t iterNumber, const std::string& loss_tensor_name = "loss");
    virtual raul::dtype oneTrainIterationMixedPrecision(raul::Workflow& network, raul::optimizers::IOptimizer*, size_t iterNumber, const std::string& loss_tensor_name = "loss");
    virtual raul::dtype
    oneTrainIteration(raul::Workflow& network, raul::optimizers::IOptimizer*, size_t iterNumber, const std::pair<size_t, size_t>& rescaleSize, const std::string& loss_tensor_name = "loss");

    virtual raul::dtype testNetwork(raul::Workflow& network);
    virtual raul::dtype testNetwork(raul::Workflow& network, const std::pair<size_t, size_t>& rescaleSize);

    size_t getImageSize() const { return mImageSize; }
    size_t getImageDepth() const { return mImageDepth; }

    virtual const TensorU8& getTrainImages() { return *mTrainImages; }
    virtual const TensorU8& getTestImages() { return *mTestImages; }
    virtual const TensorU8& getTrainLabels() { return *mTrainLabels; }
    virtual const TensorU8& getTestLabels() { return *mTestLabels; }
    virtual size_t getTrainImageAmount() { return mTrainAmount; }
    virtual size_t getTestImageAmount() { return mTestAmount; }
    float getTrainingTime() const { return static_cast<float>(mTimeTaken) / 1000.0f; }
    float getTestingTime() const { return static_cast<float>(mTimeTakenTest) / 1000.0f; }
};

class CIFAR10 : public Datasets
{
  public:
    CIFAR10();
    CIFAR10(std::istream& stream);
    bool loadingData(const std::filesystem::path& path) override;
    bool load(const std::filesystem::path& path);
};

class MNIST : public Datasets
{
  public:
    MNIST();
    MNIST(const std::filesystem::path& labelsFilePath, const std::filesystem::path& imagesFilepath);
    bool loadingData(const std::filesystem::path& path) override;
};

void reshapeTensor(const TensorU8& in, TensorU8& out);
} // namespace raul

#endif
