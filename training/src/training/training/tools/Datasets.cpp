// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "Datasets.h"

#include <algorithm>
#include <chrono>
#include <sstream>

#include "CIFARParser.h"
#include "MNISTParser.h"

#include <training/layers/BasicLayer.h>

namespace
{

template<typename T>
void St2Tensor(const uint8_t* beg, const uint8_t* end, typename T::iterator out)
{
    std::transform(beg, end, out, [](uint8_t _n) -> typename T::type { return static_cast<typename T::type>(_n) / static_cast<typename T::type>(255); });
}

} // anonymous namespace

namespace raul
{

template<typename T>
void Datasets::checkInputSize(const T& inputs, size_t batchSize) const
{
    if (inputs.getWidth() != mImageSize || inputs.getHeight() != mImageSize || inputs.getDepth() != mImageDepth || inputs.getBatchSize() != batchSize)
    {
        std::stringstream s;
        s << "Provided tensor size [" << inputs.getBatchSize() << " x " << inputs.getHeight() << " x " << inputs.getWidth() << " x " << inputs.getDepth() << "]"
          << " differs from required tensor size "
          << "[" << batchSize << " x " << mImageSize << " x " << mImageSize << " x " << mImageDepth << "]";

        THROW_NONAME("Datasets", s.str());
    }
}

void Datasets::checkInputSize(const raul::Tensor& inputs, size_t batchSize, const std::pair<size_t, size_t>& rescaleSize) const
{
    if (inputs.getWidth() != rescaleSize.first || inputs.getHeight() != rescaleSize.second || inputs.getDepth() != mImageDepth || inputs.getBatchSize() != batchSize)
    {
        std::stringstream s;
        s << "Provided tensor size [" << inputs.getBatchSize() << " x " << inputs.getHeight() << " x " << inputs.getWidth() << " x " << inputs.getDepth() << "]"
          << " differs from required tensor size "
          << "[" << batchSize << " x " << rescaleSize.first << " x " << rescaleSize.second << " x " << mImageDepth << "]";

        THROW_NONAME("Datasets", s.str());
    }
}

raul::dtype Datasets::oneTrainIteration(Workflow& network, raul::optimizers::IOptimizer* optimizer, size_t q, const std::string& loss_tensor_name)
{
    raul::dtype totalLoss = 0_dt;

    if (network.getExecutionTarget() == ExecutionTarget::CPU)
    {
        raul::Tensor& inputs = network.getMemoryManager<MemoryManager>().getTensor("data");
        const raul::Tensor& loss = network.getMemoryManager<MemoryManager>().getTensor(loss_tensor_name);
        raul::Tensor& labels = network.getMemoryManager<MemoryManager>().getTensor("labels");

        const size_t batchSize = network.getBatchSize();

        checkInputSize(inputs, batchSize);

        St2Tensor<raul::Tensor>(
            &(*mTrainImages)[0] + q * mImageSize * mImageSize * mImageDepth * batchSize, &(*mTrainImages)[0] + (1 + q) * mImageSize * mImageSize * mImageDepth * batchSize, inputs.begin());

        std::copy(mEncodedTrainLabels->begin() + q * batchSize * mNumClasses, mEncodedTrainLabels->begin() + (q + 1) * batchSize * mNumClasses, labels.begin());
        std::chrono::steady_clock::time_point timeStart = std::chrono::steady_clock::now();

        network.forwardPassTraining();
        totalLoss = loss[0];
        network.backwardPassTraining();

        auto params = network.getTrainableParameters();
        for (auto& p : params)
        {
            optimizer->operator()(network.getMemoryManager<MemoryManager>(), p.Param, p.Gradient);
        }

        mTimeTaken += std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - timeStart).count();
    }
    else if (network.getExecutionTarget() == ExecutionTarget::GPU)
    {
        MemoryManagerGPU& memoryManager = network.getMemoryManager<MemoryManagerGPU>();

        auto inputs = memoryManager["data"];
        const auto loss = memoryManager[loss_tensor_name];
        auto labels = memoryManager["labels"];

        const size_t batchSize = network.getBatchSize();

        checkInputSize(inputs.getTensor(), batchSize);

        Tensor inputsCPU(inputs.getTensor().getShape());
        Tensor labelsCPU(labels.getTensor().getShape());
        Tensor lossCPU(loss.getTensor().getShape());

        St2Tensor<raul::Tensor>(
            &(*mTrainImages)[0] + q * mImageSize * mImageSize * mImageDepth * batchSize, &(*mTrainImages)[0] + (1 + q) * mImageSize * mImageSize * mImageDepth * batchSize, inputsCPU.begin());
        inputs = inputsCPU;

        std::copy(mEncodedTrainLabels->begin() + q * batchSize * mNumClasses, mEncodedTrainLabels->begin() + (q + 1) * batchSize * mNumClasses, labelsCPU.begin());
        labels = labelsCPU;

        std::chrono::steady_clock::time_point timeStart = std::chrono::steady_clock::now();

        network.forwardPassTraining();
        lossCPU = loss;
        totalLoss = lossCPU[0];
        network.backwardPassTraining();

        auto params = network.getTrainableParameters<MemoryManagerGPU>();
        for (auto& p : params)
        {
            optimizer->operator()(network.getMemoryManager<MemoryManagerGPU>(), p.Param, p.Gradient);
        }

        mTimeTaken += std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - timeStart).count();
    }
    else if (network.getExecutionTarget() == ExecutionTarget::CPUFP16)
    {
        auto& inputs = network.getMemoryManager<MemoryManagerFP16>().getTensor("data");
        const auto& loss = network.getMemoryManager<MemoryManagerFP16>().getTensor(loss_tensor_name);
        auto& labels = network.getMemoryManager<MemoryManagerFP16>().getTensor("labels");

        const size_t batchSize = network.getBatchSize();

        checkInputSize(inputs, batchSize);

        St2Tensor<TensorFP16>(
            &(*mTrainImages)[0] + q * mImageSize * mImageSize * mImageDepth * batchSize, &(*mTrainImages)[0] + (1 + q) * mImageSize * mImageSize * mImageDepth * batchSize, inputs.begin());

        std::transform(mEncodedTrainLabels->begin() + q * batchSize * mNumClasses, mEncodedTrainLabels->begin() + (q + 1) * batchSize * mNumClasses, labels.begin(), [](dtype val) {
            return raul::toFloat16(val);
        });
        std::chrono::steady_clock::time_point timeStart = std::chrono::steady_clock::now();

        network.forwardPassTraining();
        totalLoss = loss[0];
        network.backwardPassTraining();

        auto params = network.getTrainableParameters<MemoryManagerFP16>();
        for (auto& p : params)
        {
            optimizer->operator()(network.getMemoryManager<MemoryManagerFP16>(), p.Param, p.Gradient);
        }

        mTimeTaken += std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - timeStart).count();
    }
    else
    {
        THROW_NONAME("Datasets", "unsupported execution target");
    }

    return totalLoss;
}

raul::dtype Datasets::oneTrainIterationMixedPrecision(Workflow& network, raul::optimizers::IOptimizer* optimizer, size_t q, const std::string& loss_tensor_name)
{
    raul::dtype totalLoss = 0_dt;

    if (network.getExecutionTarget() == ExecutionTarget::CPU)
    {
        raul::Tensor& inputs = network.getMemoryManager<MemoryManager>().getTensor("data");
        const raul::Tensor& loss = network.getMemoryManager<MemoryManager>().getTensor(loss_tensor_name);
        raul::Tensor& labels = network.getMemoryManager<MemoryManager>().getTensor("labels");

        const size_t batchSize = network.getBatchSize();

        checkInputSize(inputs, batchSize);

        St2Tensor<raul::Tensor>(
            &(*mTrainImages)[0] + q * mImageSize * mImageSize * mImageDepth * batchSize, &(*mTrainImages)[0] + (1 + q) * mImageSize * mImageSize * mImageDepth * batchSize, inputs.begin());

        std::copy(mEncodedTrainLabels->begin() + q * batchSize * mNumClasses, mEncodedTrainLabels->begin() + (q + 1) * batchSize * mNumClasses, labels.begin());
        std::chrono::steady_clock::time_point timeStart = std::chrono::steady_clock::now();

        network.forwardPassTraining();
        totalLoss = loss[0];
        network.backwardPassTraining();

        auto& mm = network.getMemoryManager<MemoryManager>();
        auto& mmFP16 = network.getMemoryManager<MemoryManagerFP16>();

        auto names = network.getTrainableParameterNames();
        for (auto& name : names)
        {
            if (mm.tensorExists(name))
                optimizer->operator()(mm, mm[name], mm[name.grad()]);
            else if (mmFP16.tensorExists(name))
                optimizer->operator()(mmFP16, mmFP16[name], mmFP16[name.grad()]);
        }

        mTimeTaken += std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - timeStart).count();
    }
    else if (network.getExecutionTarget() == ExecutionTarget::CPUFP16)
    {
        auto& inputs = network.getMemoryManager<MemoryManagerFP16>().getTensor("data");
        const auto& loss = network.getMemoryManager<MemoryManagerFP16>().getTensor(loss_tensor_name);
        auto& labels = network.getMemoryManager<MemoryManagerFP16>().getTensor("labels");

        const size_t batchSize = network.getBatchSize();

        checkInputSize(inputs, batchSize);

        St2Tensor<raul::TensorFP16>(
            &(*mTrainImages)[0] + q * mImageSize * mImageSize * mImageDepth * batchSize, &(*mTrainImages)[0] + (1 + q) * mImageSize * mImageSize * mImageDepth * batchSize, inputs.begin());

        std::transform(mEncodedTrainLabels->begin() + q * batchSize * mNumClasses, mEncodedTrainLabels->begin() + (q + 1) * batchSize * mNumClasses, labels.begin(), [](dtype val) {
            return raul::toFloat16(val);
        });
        std::chrono::steady_clock::time_point timeStart = std::chrono::steady_clock::now();

        network.forwardPassTraining();
        totalLoss = loss[0];
        network.backwardPassTraining();

        auto& mm = network.getMemoryManager<MemoryManager>();
        auto& mmFP16 = network.getMemoryManager<MemoryManagerFP16>();

        auto names = network.getTrainableParameterNames();
        for (auto& name : names)
        {
            if (mm.tensorExists(name))
                optimizer->operator()(mm, mm[name], mm[name.grad()]);
            else if (mmFP16.tensorExists(name))
                optimizer->operator()(mmFP16, mmFP16[name], mmFP16[name.grad()]);
        }

        mTimeTaken += std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - timeStart).count();
    }
    else
    {
        THROW_NONAME("Datasets", "unsupported execution target");
    }

    return totalLoss;
}

raul::dtype Datasets::oneTrainIteration(Workflow& network, raul::optimizers::IOptimizer* optimizer, size_t q, const std::pair<size_t, size_t>& rescaleSize, const std::string& loss_tensor_name)
{
    raul::Tensor& inputs = network.getMemoryManager<MemoryManager>().getTensor("data");
    const raul::Tensor& loss = network.getMemoryManager<MemoryManager>().getTensor(loss_tensor_name);
    raul::Tensor& labels = network.getMemoryManager<MemoryManager>().getTensor("labels");

    const size_t batchSize = network.getBatchSize();

    checkInputSize(inputs, batchSize, rescaleSize);

    {
        TensorU8 originalBatch(batchSize, mImageDepth, mImageSize, mImageSize);
        std::copy(
            &(*mTrainImages)[0] + q * mImageSize * mImageSize * mImageDepth * batchSize, &(*mTrainImages)[0] + (1 + q) * mImageSize * mImageSize * mImageDepth * batchSize, originalBatch.begin());

        TensorU8 rescaledBatch(batchSize, mImageDepth, rescaleSize.first, rescaleSize.second);

        reshapeTensor(originalBatch, rescaledBatch);

        St2Tensor<raul::Tensor>(&rescaledBatch[0], &rescaledBatch[0] + rescaledBatch.size(), inputs.begin());
    }

    std::copy(mEncodedTrainLabels->begin() + q * batchSize * mNumClasses, mEncodedTrainLabels->begin() + (q + 1) * batchSize * mNumClasses, labels.begin());
    std::chrono::steady_clock::time_point timeStart = std::chrono::steady_clock::now();

    network.forwardPassTraining();
    raul::dtype totalLoss = loss[0];
    network.backwardPassTraining();

    auto params = network.getTrainableParameters();
    for (auto& p : params)
        optimizer->operator()(network.getMemoryManager<MemoryManager>(), p.Param, p.Gradient);

    mTimeTaken += std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - timeStart).count();

    return totalLoss;
}

raul::dtype Datasets::testNetwork(raul::Workflow& network)
{
    std::chrono::steady_clock::time_point timeStart = std::chrono::steady_clock::now();

    const size_t batchSize = network.getBatchSize();

    const size_t stepsAmountTest = mTestAmount / batchSize;
    size_t correctLabelsCounter = 0;

    if (network.getExecutionTarget() == ExecutionTarget::CPU)
    {
        raul::Tensor& inputs = network.getMemoryManager<MemoryManager>().getTensor("data");
        /// @todo(ck): eliminate hardcoded tensor
        const raul::Tensor& softmax = network.getMemoryManager<MemoryManager>().getTensor("softmax");
        raul::Tensor& labels = network.getMemoryManager<MemoryManager>().getTensor("labels");

        checkInputSize(inputs, batchSize);

        for (size_t q = 0; q < stepsAmountTest; ++q)
        {
            St2Tensor<raul::Tensor>(
                &(*mTestImages)[0] + q * mImageSize * mImageSize * mImageDepth * batchSize, &(*mTestImages)[0] + (1 + q) * mImageSize * mImageSize * mImageDepth * batchSize, inputs.begin());

            std::copy(mEncodedTestLabels->begin() + q * batchSize * mNumClasses, mEncodedTestLabels->begin() + (q + 1) * batchSize * mNumClasses, labels.begin());

            network.forwardPassTesting();

            for (size_t w = 0; w < batchSize; ++w)
            {
                if (softmax.getMaxIndex(w * mNumClasses, (w + 1) * mNumClasses) == (*mTestLabels)[q * batchSize + w]) ++correctLabelsCounter;
            }
        }
    }
    else if (network.getExecutionTarget() == ExecutionTarget::GPU)
    {
        MemoryManagerGPU& memoryManager = network.getMemoryManager<MemoryManagerGPU>();

        auto inputs = memoryManager["data"];
        /// @todo(ck): eliminate hardcoded tensor
        const auto softmax = memoryManager["softmax"];
        auto labels = memoryManager["labels"];

        checkInputSize(inputs.getTensor(), batchSize);

        Tensor inputsCPU(inputs.getTensor().getShape());
        Tensor labelsCPU(labels.getTensor().getShape());
        Tensor softmaxCPU(softmax.getTensor().getShape());

        for (size_t q = 0; q < stepsAmountTest; ++q)
        {
            St2Tensor<raul::Tensor>(
                &(*mTestImages)[0] + q * mImageSize * mImageSize * mImageDepth * batchSize, &(*mTestImages)[0] + (1 + q) * mImageSize * mImageSize * mImageDepth * batchSize, inputsCPU.begin());
            inputs = inputsCPU;

            std::copy(mEncodedTestLabels->begin() + q * batchSize * mNumClasses, mEncodedTestLabels->begin() + (q + 1) * batchSize * mNumClasses, labelsCPU.begin());
            labels = labelsCPU;

            network.forwardPassTesting();

            softmaxCPU = softmax;

            for (size_t w = 0; w < batchSize; ++w)
            {
                if (softmaxCPU.getMaxIndex(w * mNumClasses, (w + 1) * mNumClasses) == (*mTestLabels)[q * batchSize + w]) ++correctLabelsCounter;
            }
        }
    }
    else if (network.getExecutionTarget() == ExecutionTarget::CPUFP16)
    {
        auto& inputs = network.getMemoryManager<MemoryManagerFP16>().getTensor("data");
        /// @todo(ck): eliminate hardcoded tensor
        const auto& softmax = network.getMemoryManager<MemoryManagerFP16>().getTensor("softmax");
        auto& labels = network.getMemoryManager<MemoryManagerFP16>().getTensor("labels");

        checkInputSize(inputs, batchSize);

        for (size_t q = 0; q < stepsAmountTest; ++q)
        {
            St2Tensor<raul::TensorFP16>(
                &(*mTestImages)[0] + q * mImageSize * mImageSize * mImageDepth * batchSize, &(*mTestImages)[0] + (1 + q) * mImageSize * mImageSize * mImageDepth * batchSize, inputs.begin());

            std::transform(mEncodedTestLabels->begin() + q * batchSize * mNumClasses, mEncodedTestLabels->begin() + (q + 1) * batchSize * mNumClasses, labels.begin(), [](dtype val) {
                return raul::toFloat16(val);
            });

            network.forwardPassTesting();

            for (size_t w = 0; w < batchSize; ++w)
            {
                if (softmax.getMaxIndex(w * mNumClasses, (w + 1) * mNumClasses) == (*mTestLabels)[q * batchSize + w]) ++correctLabelsCounter;
            }
        }
    }
    else
    {
        THROW_NONAME("Datasets", "unsupported execution target");
    }

    raul::dtype testAccuracy = static_cast<dtype>(correctLabelsCounter) / static_cast<dtype>(stepsAmountTest * batchSize) * 100.0_dt;

    mTimeTakenTest = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - timeStart).count();

    return testAccuracy;
}

raul::dtype Datasets::testNetwork(raul::Workflow& network, const std::pair<size_t, size_t>& rescaleSize)
{
    std::chrono::steady_clock::time_point timeStart = std::chrono::steady_clock::now();

    raul::Tensor& inputs = network.getMemoryManager<MemoryManager>().getTensor("data");
    const raul::Tensor& softmax = network.getMemoryManager<MemoryManager>().getTensor("softmax");
    raul::Tensor& labels = network.getMemoryManager<MemoryManager>().getTensor("labels");

    const size_t batchSize = network.getBatchSize();

    checkInputSize(inputs, batchSize, rescaleSize);

    const size_t stepsAmountTest = mTestAmount / batchSize;
    size_t correctLabelsCounter = 0;
    for (size_t q = 0; q < stepsAmountTest; ++q)
    {
        {
            TensorU8 originalBatch(batchSize, mImageDepth, mImageSize, mImageSize);
            std::copy(
                &(*mTestImages)[0] + q * mImageSize * mImageSize * mImageDepth * batchSize, &(*mTestImages)[0] + (1 + q) * mImageSize * mImageSize * mImageDepth * batchSize, originalBatch.begin());

            TensorU8 rescaledBatch(batchSize, mImageDepth, rescaleSize.first, rescaleSize.second);

            reshapeTensor(originalBatch, rescaledBatch);

            St2Tensor<raul::Tensor>(&rescaledBatch[0], &rescaledBatch[0] + rescaledBatch.size(), inputs.begin());
        }

        std::copy(mEncodedTestLabels->begin() + q * batchSize * mNumClasses, mEncodedTestLabels->begin() + (q + 1) * batchSize * mNumClasses, labels.begin());

        network.forwardPassTesting();

        for (size_t w = 0; w < batchSize; ++w)
        {
            if (softmax.getMaxIndex(w * mNumClasses, (w + 1) * mNumClasses) == (*mTestLabels)[q * batchSize + w]) ++correctLabelsCounter;
        }
    }
    raul::dtype testAccuracy = static_cast<dtype>(correctLabelsCounter) / static_cast<dtype>(stepsAmountTest * batchSize) * 100.0_dt;

    mTimeTakenTest = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - timeStart).count();

    return testAccuracy;
}

CIFAR10::CIFAR10()
{
    mImageSize = 32;
    mImageDepth = 3;
    mNumClasses = 10;
}

CIFAR10::CIFAR10(std::istream& stream)
    : CIFAR10()
{
    raul::CIFARPArser::LoadData(mTrainImages, mTrainLabels, stream);

    mTestImages = std::make_unique<TensorU8>(static_cast<raul::TensorU8::dt_range>(*mTrainImages));
    mTestLabels = std::make_unique<TensorU8>(static_cast<raul::TensorU8::dt_range>(*mTrainLabels));

    mEncodedTrainLabels = &mDataLoader.buildOneHotVector(*mTrainLabels, mNumClasses);
    mEncodedTestLabels = &mDataLoader.buildOneHotVector(*mTestLabels, mNumClasses);

    mTrainAmount = mTrainLabels->size();
    mTestAmount = mTestLabels->size();
}

bool CIFAR10::loadingData(const std::filesystem::path& path)
{
    mTrainAmount = 50000;
    mTestAmount = 10000;

    raul::CIFARPArser::LoadData(mTrainImages, mTrainLabels, path / "data_batch_1.bin");
    raul::CIFARPArser::LoadData(mTrainImages, mTrainLabels, path / "data_batch_2.bin");
    raul::CIFARPArser::LoadData(mTrainImages, mTrainLabels, path / "data_batch_3.bin");
    raul::CIFARPArser::LoadData(mTrainImages, mTrainLabels, path / "data_batch_4.bin");
    raul::CIFARPArser::LoadData(mTrainImages, mTrainLabels, path / "data_batch_5.bin");
    raul::CIFARPArser::LoadData(mTestImages, mTestLabels, path / "test_batch.bin");

    mEncodedTrainLabels = &mDataLoader.buildOneHotVector(*mTrainLabels, mNumClasses);
    mEncodedTestLabels = &mDataLoader.buildOneHotVector(*mTestLabels, mNumClasses);

    bool flag = raul::if_equals("CIFAR10[loadingData]: train_images size are not equals", mTrainImages->size(), mTrainAmount * mImageSize * mImageSize * mImageDepth);
    flag &= raul::if_equals("CIFAR10[loadingData]: train_labels size are not equals", mTrainLabels->size(), mTrainAmount);
    flag &= raul::if_equals("CIFAR10[loadingData]: test_images size are not equals", mTestImages->size(), mTestAmount * mImageSize * mImageSize * mImageDepth);
    flag &= raul::if_equals("CIFAR10[loadingData]: test labels size are not equals", mTestLabels->size(), mTestAmount);

    flag &= raul::if_equals("CIFAR10[loadingData]: encoded train_labels is failed", mEncodedTrainLabels->size(), mTrainLabels->size() * mNumClasses);
    flag &= raul::if_equals("CIFAR10[loadingData]: encoded mTestLabels is failed", mEncodedTestLabels->size(), mTestLabels->size() * mNumClasses);

    return flag;
}

bool CIFAR10::load(const std::filesystem::path& path)
{
    raul::CIFARPArser::LoadData(mTrainImages, mTrainLabels, path);

    mTestImages = std::make_unique<TensorU8>(static_cast<raul::TensorU8::dt_range>(*mTrainImages));
    mTestLabels = std::make_unique<TensorU8>(static_cast<raul::TensorU8::dt_range>(*mTrainLabels));

    mEncodedTrainLabels = &mDataLoader.buildOneHotVector(*mTrainLabels, mNumClasses);
    mEncodedTestLabels = &mDataLoader.buildOneHotVector(*mTestLabels, mNumClasses);

    mTrainAmount = mTrainLabels->size();
    mTestAmount = mTrainLabels->size();

    return true;
}

MNIST::MNIST()
{
    mImageSize = 28;
    mImageDepth = 1;
    mNumClasses = 10;
}

MNIST::MNIST(const std::filesystem::path& labelsFile, const std::filesystem::path& imagesFile)
    : MNIST()
{
    raul::MNISTParser::LoadData(mTrainLabels, mTrainImages, labelsFile, imagesFile);

    mTestImages = std::make_unique<TensorU8>(static_cast<raul::TensorU8::dt_range>(*mTrainImages));
    mTestLabels = std::make_unique<TensorU8>(static_cast<raul::TensorU8::dt_range>(*mTrainLabels));

    mEncodedTrainLabels = &mDataLoader.buildOneHotVector(*mTrainLabels, mNumClasses);
    mEncodedTestLabels = &mDataLoader.buildOneHotVector(*mTestLabels, mNumClasses);

    mTrainAmount = mTrainLabels->size();
    mTestAmount = mTestLabels->size();
}

bool MNIST::loadingData(const std::filesystem::path& path)
{
    mTrainAmount = 60000;
    mTestAmount = 10000;

    raul::MNISTParser::LoadData(mTrainLabels, mTrainImages, path / "train-labels-idx1-ubyte", path / "train-images-idx3-ubyte");
    raul::MNISTParser::LoadData(mTestLabels, mTestImages, path / "t10k-labels-idx1-ubyte", path / "t10k-images-idx3-ubyte");

    mEncodedTrainLabels = &mDataLoader.buildOneHotVector(*mTrainLabels, mNumClasses);
    mEncodedTestLabels = &mDataLoader.buildOneHotVector(*mTestLabels, mNumClasses);

    bool flag = raul::if_equals("MNIST[loadingData]: train_images size are not equals", mTrainImages->size(), mTrainAmount * mImageSize * mImageSize * mImageDepth);
    flag &= raul::if_equals("MNIST[loadingData]: train_labels size are not equals", mTrainLabels->size(), mTrainAmount);
    flag &= raul::if_equals("MNIST[loadingData]: test_images size are not equals", mTestImages->size(), mTestAmount * mImageSize * mImageSize * mImageDepth);
    flag &= raul::if_equals("MNIST[loadingData]: test labels size are not equals", mTestLabels->size(), mTestAmount);

    flag &= raul::if_equals("MNIST[loadingData]: encoded train_labels is failed", mEncodedTrainLabels->size(), mTrainLabels->size() * mNumClasses);
    flag &= raul::if_equals("MNIST[loadingData]: encoded mTestLabels is failed", mEncodedTestLabels->size(), mTestLabels->size() * mNumClasses);

    return flag;
}

void reshapeTensor(const TensorU8& in, TensorU8& out)
{
    if (in.empty()) THROW_NONAME("Datasets", "in Tensor empty");
    if (out.empty()) THROW_NONAME("Datasets", "out Tensor empty");
    if (in.getBatchSize() != out.getBatchSize()) THROW_NONAME("Datasets", "batch sizes doesn`t match");
    if (in.getDepth() != out.getDepth()) THROW_NONAME("Datasets", "depth doesn`t match");

    float ratioX = static_cast<float>(in.getWidth()) / static_cast<float>(out.getWidth());
    float ratioY = static_cast<float>(in.getHeight()) / static_cast<float>(out.getHeight());

    auto buffer4D = out.get4DView();
    auto image4D = in.get4DView();
    for (size_t i = 0; i < in.getBatchSize(); ++i)
    {
        for (size_t j = 0; j < in.getDepth(); ++j)
        {
            for (size_t k = 0; k < out.getHeight(); ++k)
            {
                for (size_t l = 0; l < out.getWidth(); ++l)
                {
                    const auto oldX = static_cast<size_t>(std::floor(static_cast<dtype>(l) * ratioX));
                    const auto oldY = static_cast<size_t>(std::floor(static_cast<dtype>(k) * ratioY));
                    buffer4D[i][j][k][l] = image4D[i][j][oldY][oldX];
                }
            }
        }
    }
}
} // namespace raul
