// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef LOSS_WRAPPER_H
#define LOSS_WRAPPER_H

#include <training/base/common/Common.h>

#include <training/base/layers/basic/ElementWiseDivLayer.h>
#include <training/base/layers/basic/ElementWiseMulLayer.h>
#include <training/base/layers/basic/LossWrapperHelperLayer.h>
#include <training/base/layers/basic/ReduceBatchMeanLayer.h>
#include <training/base/layers/basic/ReduceMeanLayer.h>
#include <training/base/layers/basic/ReduceNonZeroLayer.h>
#include <training/base/layers/basic/ReduceSumLayer.h>

#include <training/base/loss/DivisorLossHelperLayer.h>

namespace raul
{

template<typename T>
void LossWrapperFunction(const Name& name, const LossParams& params, Workflow& work)
{
    auto mInputs = params.getInputs();

    if (params.reduction != LossParams::Reduction::None || mInputs.size() == 3)
    {
        auto prefix = "LossWrapper[" + name + "::ctor]: ";
        if (params.getInputs().size() != 2 && params.getInputs().size() != 3)
        {
            THROW("LossWrapper", name, "wrong number of input names");
        }
        if (params.getOutputs().size() != 1)
        {
            THROW("LossWrapper", name, "wrong number of output names");
        }

        if (params.getInputs().size() == 2 && (params.reduction == LossParams::Reduction::Sum_Over_Nonzero_Weights || params.reduction == LossParams::Reduction::Sum_Over_Weights))
        {
            THROW("LossWrapper", name, "reduction over weights used for unweighted loss");
        }

        auto mInputName = params.getInputs()[0];
        auto mLabelName = params.getInputs()[1];
        auto mWeightsName = params.getInputs().size() == 2 ? name + "::Weights" : params.getInputs()[2];
        auto mOutputName = params.getOutputs()[0];

        bool weighted = params.getInputs().size() == 3;
        bool isNone = params.reduction == LossParams::Reduction::None;                                      // element-wise loss
        bool isSum = params.reduction == LossParams::Reduction::Sum;                                        // sum of losses
        bool isMean = params.reduction == LossParams::Reduction::Mean;                                      // sum of losses divided by the total number of elements
        bool isBatchMean = params.reduction == LossParams::Reduction::Batch_Mean;                           // sum of losses divided by batch size
        bool isSumOverWeights = params.reduction == LossParams::Reduction::Sum_Over_Weights;                // sum of losses divided by sum of weights
        bool isSumOverNonZeroWeights = params.reduction == LossParams::Reduction::Sum_Over_Nonzero_Weights; // sum of losses divided by the number of nonzero weights
        bool isCustomMean = params.reduction == LossParams::Reduction::Custom_Mean; // use divisor specified in network params (i.e. custom batch size) multiplied by input total size

        Name preFinalName = name / "output_loss";
        Name nextName = name / "elementWiseLoss";
        // Loss function with Reduction::None
        work.add<T>(name / "loss", raul::LossParams{ { mInputName, mLabelName }, { isNone && !weighted ? preFinalName : nextName }, LossParams::Reduction::None });

        if (weighted)
        {
            // Get weighted loss
            work.add<ElementWiseMulLayer>(name / "mul", raul::ElementWiseLayerParams{ { nextName, mWeightsName }, { isNone ? preFinalName : name / "weightedLoss" } });
            nextName = name / "weightedLoss";
        }

        if (!isNone)
        {
            if (isMean)
            {
                work.add<ReduceMeanLayer>(name / "rmeanNonWeightedLoss", raul::BasicParamsWithDim{ { nextName }, { preFinalName } });
            }
            else if (isBatchMean)
            {
                work.add<ReduceBatchMeanLayer>(name / "rbmeanNonWeightedLoss", raul::BasicParamsWithDim{ { nextName }, { preFinalName } });
            }
            else
            {
                // Get sum of losses
                work.add<ReduceSumLayer>(name / "rsumNonWeightedLoss", raul::BasicParamsWithDim{ { nextName }, { isSum ? preFinalName : name / "weightedLossSum" } });

                Name divisorName = "";
                if (isSumOverWeights)
                {
                    work.add<ReduceSumLayer>(name / "rsumWeights", raul::BasicParamsWithDim{ { mWeightsName }, { name / "sumOfWeights" } });
                    divisorName = name / "sumOfWeights";
                }
                else if (isSumOverNonZeroWeights)
                {
                    work.add<ReduceNonZeroLayer>(name / "rsumWeights", raul::BasicParamsWithDim{ { mWeightsName }, { name / "sumOfWeights" } });
                    divisorName = name / "sumOfWeights";
                }
                else
                {
                    // Use custom divisor specified in network params (custom batch size or custom batch size * total size)
                    if (!isSum)
                    {
                        divisorName = name / "customDivisor";

                        work.add<DivisorLossHelperLayer>(divisorName, raul::BasicParams{ {}, { divisorName } }, isCustomMean, mInputName);
                    }
                }

                if (!divisorName.empty())
                {
                    work.add<ElementWiseDivLayer>(name / "div", raul::ElementWiseLayerParams{ { name / "weightedLossSum", divisorName }, { preFinalName } });
                }
            }
        }

        work.add<LossWrapperHelperLayer>(name / "helper", raul::BasicParams{ { preFinalName }, { mOutputName } }, params.mIsFinal);
    }
}

/**
 * @brief LossWrapper
 * Calculates specified type of loss with weights.
 */
template<typename T>
class LossWrapper
{
  public:
    LossWrapper(const Name& name, const LossParams& params, NetworkParameters& networkParameters);
    ~LossWrapper(){}

  private:
    Name mInputName;
    Name mLabelName;
    Name mOutputName;
    Name mWeightsName;
    LossParams::Reduction mReduction;
};

template<typename T>
LossWrapper<T>::LossWrapper(const Name& name, const LossParams& params, NetworkParameters& networkParameters)
    : mReduction(params.reduction)
{
    auto prefix = "LossWrapper[" + name + "::ctor]: ";
    if (params.getInputs().size() != 2 && params.getInputs().size() != 3)
    {
        THROW("LossWrapper", name, "wrong number of input names");
    }
    if (params.getOutputs().size() != 1)
    {
        THROW("LossWrapper", name, "wrong number of output names");
    }

    if (params.getInputs().size() == 2 && (mReduction == LossParams::Reduction::Sum_Over_Nonzero_Weights || mReduction == LossParams::Reduction::Sum_Over_Weights))
    {
        THROW("LossWrapper", name, "reduction over weights used for unweighted loss");
    }

    mInputName = params.getInputs()[0];
    mLabelName = params.getInputs()[1];
    mWeightsName = params.getInputs().size() == 2 ? name + "::Weights" : params.getInputs()[2];
    mOutputName = params.getOutputs()[0];

    bool weighted = params.getInputs().size() == 3;
    bool isNone = mReduction == LossParams::Reduction::None;                                      // element-wise loss
    bool isSum = mReduction == LossParams::Reduction::Sum;                                        // sum of losses
    bool isMean = mReduction == LossParams::Reduction::Mean;                                      // sum of losses divided by the total number of elements
    bool isBatchMean = mReduction == LossParams::Reduction::Batch_Mean;                           // sum of losses divided by batch size
    bool isSumOverWeights = mReduction == LossParams::Reduction::Sum_Over_Weights;                // sum of losses divided by sum of weights
    bool isSumOverNonZeroWeights = mReduction == LossParams::Reduction::Sum_Over_Nonzero_Weights; // sum of losses divided by the number of nonzero weights
    bool isCustomMean = mReduction == LossParams::Reduction::Custom_Mean;                         // use divisor specified in network params (i.e. custom batch size) multiplied by input total size

    Name preFinalName = name / "output_loss";
    Name nextName = name / "elementWiseLoss";
    // Loss function with Reduction::None
    networkParameters.mWorkflow.add<T>(name / "loss", raul::LossParams{ { mInputName, mLabelName }, { isNone && !weighted ? preFinalName : nextName }, LossParams::Reduction::None });

    if (weighted)
    {
        // Get weighted loss
        networkParameters.mWorkflow.add<ElementWiseMulLayer>(name / "mul", raul::ElementWiseLayerParams{ { nextName, mWeightsName }, { isNone ? preFinalName : name / "weightedLoss" } });
        nextName = name / "weightedLoss";
    }

    if (!isNone)
    {
        if (isMean)
        {
            networkParameters.mWorkflow.add<ReduceMeanLayer>(name / "rmeanNonWeightedLoss", raul::BasicParamsWithDim{ { nextName }, { preFinalName } });
        }
        else if (isBatchMean)
        {
            networkParameters.mWorkflow.add<ReduceBatchMeanLayer>(name / "rbmeanNonWeightedLoss", raul::BasicParamsWithDim{ { nextName }, { preFinalName } });
        }
        else
        {
            // Get sum of losses
            networkParameters.mWorkflow.add<ReduceSumLayer>(name / "rsumNonWeightedLoss", raul::BasicParamsWithDim{ { nextName }, { isSum ? preFinalName : name / "weightedLossSum" } });

            Name divisorName = "";
            if (isSumOverWeights)
            {
                networkParameters.mWorkflow.add<ReduceSumLayer>(name / "rsumWeights", raul::BasicParamsWithDim{ { mWeightsName }, { name / "sumOfWeights" } });
                divisorName = name / "sumOfWeights";
            }
            else if (isSumOverNonZeroWeights)
            {
                networkParameters.mWorkflow.add<ReduceNonZeroLayer>(name / "rsumWeights", raul::BasicParamsWithDim{ { mWeightsName }, { name / "sumOfWeights" } });
                divisorName = name / "sumOfWeights";
            }
            else
            {
                // Use custom divisor specified in network params (custom batch size or custom batch size * total size)
                if (!isSum)
                {
                    divisorName = name / "customDivisor";

                    networkParameters.mWorkflow.add<DivisorLossHelperLayer>(divisorName, raul::BasicParams{ {}, { divisorName } }, isCustomMean, mInputName);
                }
            }

            if (!divisorName.empty())
            {
                networkParameters.mWorkflow.add<ElementWiseDivLayer>(name / "div", raul::ElementWiseLayerParams{ { name / "weightedLossSum", divisorName }, { preFinalName } });
            }
        }
    }

    networkParameters.mWorkflow.add<LossWrapperHelperLayer>(name / "helper", raul::BasicParams{ { preFinalName }, { mOutputName } }, params.mIsFinal);
}
} // raul namespace

#endif
