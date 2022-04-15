// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef TRAIN_H
#define TRAIN_H

#include <map>

#include <training/network/Workflow.h>
#include <training/optimizers/IOptimizer.h>
#include <training/postprocessing/GradientPostprocessor.h>
#include <training/tools/Dataset.h>

namespace raul
{

/**
 * @brief General training procedure
 *
 */
class Train
{
    using Optimizer = optimizers::IOptimizer;
    using GradientManipulation = std::shared_ptr<postprocessing::GradientPostprocessor>;

  public:
    enum class ProcessTuning
    {
        NO_TUNING,
        SKIP_BACKWARD_PASS
    };

  public:
    struct Parameters
    {
        struct DataFlowDescription
        {
            std::string mDatasetPartNameAsDataSource;
            std::string mNetworkInputNameAsDataDestination;

            DataFlowDescription(const std::string& datasetPartNameAsDataSource, const std::string& networkInputNameAsDataDestination)
                : mDatasetPartNameAsDataSource(datasetPartNameAsDataSource)
                , mNetworkInputNameAsDataDestination(networkInputNameAsDataDestination)
            {
            }
        };

        std::vector<DataFlowDescription> mDataFlowConfiguration;
        std::vector<GradientManipulation> mGradientPostprocessing;
        std::string mNetworkOutputLossName;

        Parameters(const std::vector<DataFlowDescription>& dataFlow, const std::string& networkOutputLossName)
            : mDataFlowConfiguration(dataFlow)
            , mNetworkOutputLossName(networkOutputLossName)
        {
        }

        Parameters(const std::vector<DataFlowDescription>& dataFlow, const std::string& networkOutputLossName, const std::vector<GradientManipulation>& gradientPostprocessing)
            : mDataFlowConfiguration(dataFlow)
            , mGradientPostprocessing(gradientPostprocessing)
            , mNetworkOutputLossName(networkOutputLossName)
        {
        }
    };

  public:
    Train(Workflow& network, Dataset& dataset, const Parameters& parameters);

    /// @todo(ck): batch mode can be used as a default mode which sets ready-for-work Train state after construction
    void useBatches(size_t batchSize);
    /// @todo(ck): microbatching mode can be organised as an optional flag (state specification)
    void useMicroBatches(size_t batchSize, size_t microBatchSize);

    dtype oneIteration(Optimizer& optimizer, ProcessTuning option = ProcessTuning::NO_TUNING);
    [[nodiscard]] size_t numberOfIterations() const;

    void oneEpoch(Optimizer& optimizer, std::function<void(size_t, dtype)> onEndOfIterationCallback);

    ~Train();

  private:
    Parameters mParameters;
    Workflow& mNetwork;
    Dataset& mDataset;
    std::unique_ptr<struct TrainStrategy> mTrainStrategy;
};

} // !namespace raul

#endif // TRAIN_H
