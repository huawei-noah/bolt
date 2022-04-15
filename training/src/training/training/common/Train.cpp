// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <training/common/Train.h>

namespace raul
{

struct TrainStrategy
{
    virtual raul::dtype oneTrainIteration(raul::optimizers::IOptimizer& optimizer, Train::ProcessTuning option) = 0;

    virtual ~TrainStrategy() = default;
};

class BatchMode : public TrainStrategy
{
  public:
    BatchMode(Workflow& network, Dataset& dataset, const Train::Parameters& parameters, size_t numberOfSamplesInBatch)
        : mParameters(parameters)
        , mNetwork(network)
        , mDataset(dataset)
        , mNumberOfSamplesInBatch(numberOfSamplesInBatch)
    {
        if (mNumberOfSamplesInBatch > std::numeric_limits<uint32_t>::max())
        {
            throw std::logic_error("Too big number of samples in batch for Dataset (exceeds uint32_t max value)");
        }
        mDataset.generate(static_cast<uint32_t>(mNumberOfSamplesInBatch));
        mNetwork.setBatchSize(mNumberOfSamplesInBatch);
        mNetwork.getNetworkParameters().mLossReductionCoefficient = mNumberOfSamplesInBatch;
    }

    dtype oneTrainIteration(optimizers::IOptimizer& optimizer, Train::ProcessTuning option) final
    {
        auto dataBatch = mDataset.getData();
        checkThatThereWasNoChangesOfBatchSizeForNetworkAndDatasetOutside();
        resetBatchSizeForNetworkIfNeed(dataBatch.numberOfSamples());
        setUpNetworkInputsBy(dataBatch);

        return doTrainIteration(optimizer, option);
    }

  private:
    void checkThatThereWasNoChangesOfBatchSizeForNetworkAndDatasetOutside()
    {
        if (mNetwork.getBatchSize() != mNumberOfSamplesInBatch || (!mBatchSizeChangedInside && mDataset.getBatchSize() != mNumberOfSamplesInBatch))
        {
            throw std::logic_error("It seems that batch size for network or dataset changed outside, please check your pipeline");
        }
    }

    void resetBatchSizeForNetworkIfNeed(size_t numberOfSamplesInBatch)
    {
        if (mNetwork.getBatchSize() != numberOfSamplesInBatch)
        {
            mNetwork.setBatchSize(numberOfSamplesInBatch);
            mNumberOfSamplesInBatch = numberOfSamplesInBatch;
            mNetwork.getNetworkParameters().mLossReductionCoefficient = mNumberOfSamplesInBatch;
            if (mBatchSizeChangedInside)
            {
                mBatchSizeChangedInside = false;
            }
            else
            {
                mBatchSizeChangedInside = true;
            }
        }
    }

    void setUpNetworkInputsBy(const Dataset::DataBatch& dataBatch)
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
            THROW_NONAME("Train", "unsupported execution target");
        }
    }

    dtype doTrainIteration(optimizers::IOptimizer& optimizer, Train::ProcessTuning option)
    {
        dtype error = 0.0_dt;

        if (mNetwork.getExecutionTarget() == ExecutionTarget::CPU)
        {
            mNetwork.forwardPassTraining();
            error = mNetwork.getMemoryManager()[mParameters.mNetworkOutputLossName][0];
            if (option == Train::ProcessTuning::SKIP_BACKWARD_PASS)
            {
                return error;
            }

            mNetwork.backwardPassTraining();

            auto params = mNetwork.getTrainableParameters();
            for (auto& manipulation : mParameters.mGradientPostprocessing)
            {
                manipulation->processGradients(params, mNetwork.getNetworkParameters());
            }

            for (auto& p : params)
            {
                optimizer(mNetwork.getMemoryManager(), p.Param, p.Gradient);
            }
        }
        else if (mNetwork.getExecutionTarget() == ExecutionTarget::CPUFP16)
        {
            mNetwork.forwardPassTraining();
            error = TODTYPE(mNetwork.getMemoryManager<MemoryManagerFP16>()[mParameters.mNetworkOutputLossName][0]);
            if (option == Train::ProcessTuning::SKIP_BACKWARD_PASS)
            {
                return error;
            }

            mNetwork.backwardPassTraining();

            auto params = mNetwork.getTrainableParameters<MemoryManagerFP16>();
            for (auto& manipulation : mParameters.mGradientPostprocessing)
            {
                manipulation->processGradients(params, mNetwork.getNetworkParameters());
            }

            for (auto& p : params)
            {
                optimizer(mNetwork.getMemoryManager<MemoryManagerFP16>(), p.Param, p.Gradient);
            }
        }
        else
        {
            THROW_NONAME("Train", "unsupported execution target");
        }

        return error;
    }

  private:
    const Train::Parameters& mParameters;
    Workflow& mNetwork;
    Dataset& mDataset;
    size_t mNumberOfSamplesInBatch;
    bool mBatchSizeChangedInside = false;
};

class MicroBatchMode : public TrainStrategy
{
  public:
    MicroBatchMode(Workflow& network, Dataset& dataset, const Train::Parameters& parameters, size_t numberOfSamplesInBatch, size_t numberOfSamplesInMicroBatch)
        : mParameters(parameters)
        , mNetwork(network)
        , mDataset(dataset)
        , mNumberOfSamplesInBatch(numberOfSamplesInBatch)
        , mNumberOfSamplesInMicroBatch(numberOfSamplesInMicroBatch)
    {
        if (mNumberOfSamplesInMicroBatch > mNumberOfSamplesInBatch)
        {
            throw std::logic_error("Size of microbatch should be less or equal to size of batch");
        }

        if (mNumberOfSamplesInBatch % mNumberOfSamplesInMicroBatch != 0)
        {
            throw std::logic_error("Size of batch should be divisible to size of microbatch without a rest. Incompleted microbatches are not allowed.");
        }

        if (mNumberOfSamplesInBatch > std::numeric_limits<uint32_t>::max())
        {
            throw std::logic_error("Too big number of samples in batch for Dataset (exceeds uint32_t max value)");
        }

        mNetwork.setBatchSize(mNumberOfSamplesInMicroBatch);
        mNetwork.getNetworkParameters().mLossReductionCoefficient = mNumberOfSamplesInBatch;
        mDataset.generate(static_cast<uint32_t>(mNumberOfSamplesInBatch));
    }

    dtype oneTrainIteration(optimizers::IOptimizer& optimizer, Train::ProcessTuning option) final
    {
        if (mNetwork.getBatchSize() != mNumberOfSamplesInMicroBatch || mDataset.getBatchSize() != mNumberOfSamplesInBatch)
        {
            throw std::logic_error("It seems that batch size for network or dataset changed outside, please check your pipeline");
        }

        auto dataBatch = mDataset.getData();
        if (dataBatch.numberOfSamples() != mNumberOfSamplesInBatch)
        {
            throw std::logic_error("During training in microbatching mode incompleted batches is not allowed");
        }

        const size_t numberOfMicroBatches = dataBatch.numberOfSamples() / mNumberOfSamplesInMicroBatch;
        dtype error = 0_dt;
        for (size_t microBatchIdx = 0; microBatchIdx < numberOfMicroBatches; ++microBatchIdx)
        {
            setUpNetworkInputsBy(dataBatch, microBatchIdx);
            error += doTrainIteration(option, microBatchIdx);
        }
        if (option == Train::ProcessTuning::SKIP_BACKWARD_PASS)
        {
            return error;
        }

        optimizeNetworkParameters(optimizer);

        return error;
    }

  private:
    void setUpNetworkInputsBy(const Dataset::DataBatch& dataBatch, const size_t microBatchIdx)
    {
        for (const auto& dataFlowDescription : mParameters.mDataFlowConfiguration)
        {
            const Tensor& data = dataBatch.get(dataFlowDescription.mDatasetPartNameAsDataSource);
            const size_t elementsInSample = data.size() / data.getBatchSize();
            const size_t elementsInMicroBatch = elementsInSample * mNumberOfSamplesInMicroBatch;
            auto microBatchBegin = data.begin() + microBatchIdx * elementsInMicroBatch;
            auto microBatchEnd = data.begin() + (microBatchIdx + 1) * elementsInMicroBatch;
            Tensor& networkInput = mNetwork.getMemoryManager()[dataFlowDescription.mNetworkInputNameAsDataDestination];
            std::copy(microBatchBegin, microBatchEnd, networkInput.begin());
        }
    }

    dtype doTrainIteration(Train::ProcessTuning option, size_t microbatchIndex)
    {
        if (microbatchIndex == 0)
        {
            mNetwork.forwardPassTraining(true); // zero gradients
        }
        else
        {
            mNetwork.forwardPassTraining(false); // keep gradients to accumulate result
        }

        dtype error = mNetwork.getMemoryManager()[mParameters.mNetworkOutputLossName][0];

        if (option == Train::ProcessTuning::SKIP_BACKWARD_PASS)
        {
            return error;
        }

        mNetwork.backwardPassTraining();

        return error;
    }

    void optimizeNetworkParameters(optimizers::IOptimizer& optimizer)
    {
        auto params = mNetwork.getTrainableParameters();
        for (auto& manipulation : mParameters.mGradientPostprocessing)
        {
            manipulation->processGradients(params, mNetwork.getNetworkParameters());
        }
        for (const auto& _ : params)
        {
            optimizer(mNetwork.getMemoryManager(), _.Param, _.Gradient);
        }
    }

  private:
    const Train::Parameters& mParameters;
    Workflow& mNetwork;
    Dataset& mDataset;
    size_t mNumberOfSamplesInBatch;
    size_t mNumberOfSamplesInMicroBatch;
};

Train::Train(Workflow& network, Dataset& dataset, const Parameters& parameters)
    : mParameters(parameters)
    , mNetwork(network)
    , mDataset(dataset)
{
}

Train::~Train() = default;

void Train::useBatches(size_t batchSize)
{
    if (mTrainStrategy)
    {
        throw std::logic_error("Training mode has been defined and cannot be changed without Train object reconstruction");
    }
    mTrainStrategy = std::make_unique<BatchMode>(mNetwork, mDataset, mParameters, batchSize);
}

void Train::useMicroBatches(size_t batchSize, size_t microBatchSize)
{
    if (mTrainStrategy)
    {
        throw std::logic_error("Training mode has been defined and cannot be changed without Train object reconstruction");
    }
    mTrainStrategy = std::make_unique<MicroBatchMode>(mNetwork, mDataset, mParameters, batchSize, microBatchSize);
}

void Train::oneEpoch(optimizers::IOptimizer& optimizer, std::function<void(size_t, dtype)> onEndOfIterationCallback)
{
    for (size_t i = 0; i < mDataset.numberOfBatches(); ++i)
    {
        dtype loss = oneIteration(optimizer);
        onEndOfIterationCallback(i, loss);
    }
}

dtype Train::oneIteration(optimizers::IOptimizer& optimizer, ProcessTuning option)
{
    if (!mTrainStrategy)
    {
        throw std::logic_error("Before start of training process needs to define training strategy (call Train::useBatches or Train::useMicroBatches method");
    }
    return mTrainStrategy->oneTrainIteration(optimizer, option);
}

size_t Train::numberOfIterations() const
{
    return mDataset.numberOfBatches();
}

} // !namespace raul
