// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef TEST_H
#define TEST_H

#include <map>

#include <training/network/Workflow.h>
#include <training/tools/Dataset.h>

namespace raul
{

class Test
{
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
        std::string mNetworkOutputName;
        std::string mDatasetPartAsGroundTruth;

        Parameters(const std::vector<DataFlowDescription>& dataFlow, const std::string& networkOutputName, const std::string& groundTruth)
            : mDataFlowConfiguration(dataFlow)
            , mNetworkOutputName(networkOutputName)
            , mDatasetPartAsGroundTruth(groundTruth)
        {
        }
    };

  public:
    Test(Workflow& network, Dataset& dataset, const Parameters& parameters)
        : mParameters(parameters)
        , mDataset(dataset)
        , mNetwork(network)
    {
    }

    dtype run(size_t numberOfSamplesInBatch);

  private:
    void configureDataset(size_t numberOfSamplesInBatch);
    void resetBatchSizeForNetworkIfNeed(size_t dataBatchSize);
    void setUpNetworkInputsBy(const Dataset::DataBatch& dataBatch);
    size_t calculateNumberOfCorrectlyClassifiedSamplesFor(const Dataset::DataBatch& dataBatch);
    dtype calculateAccuracyInPercents(size_t numberOfCorrectlyClassifiedSamples, size_t samplesInAmount);

  private:
    Parameters mParameters;
    Dataset& mDataset;
    Workflow& mNetwork;
};

} // !namespace raul

#endif // TEST_H
