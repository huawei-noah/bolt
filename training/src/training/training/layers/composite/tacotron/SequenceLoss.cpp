// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "SequenceLoss.h"

#include <training/layers/basic/ElementWiseDivLayer.h>
#include <training/layers/basic/ElementWiseMulLayer.h>
#include <training/layers/basic/ReduceBatchMeanLayer.h>
#include <training/layers/basic/ReduceMeanLayer.h>
#include <training/layers/basic/ReduceNonZeroLayer.h>
#include <training/layers/basic/ReduceSumLayer.h>
#include <training/loss/SoftmaxCrossEntropyLoss.h>

namespace raul::tacotron
{

void AddSequenceLoss(Workflow* work, const Name& name, const BasicParams& params, bool averageAcrossTimesteps, bool averageAcrossBatch, bool sumOverTimesteps, bool sumOverBatch)
{
    const auto& inputs = params.getInputs();
    const auto& outputs = params.getOutputs();

    std::string prefix = "Tacotron::AddSequenceLoss[" + name + "]: ";
    if (inputs.size() != 3)
    {
        THROW("Tacotron", name, "wrong number of input names");
    }
    if (outputs.size() != 1)
    {
        THROW("Tacotron", name, "wrong number of output names");
    }

    if (averageAcrossTimesteps && sumOverTimesteps)
    {
        THROW("Tacotron", name, "wrong flags chosen: cannot average across and sum over timesteps at the same time");
    }
    if (averageAcrossBatch && sumOverBatch)
    {
        THROW("Tacotron", name, "wrong flags chosen: cannot average across and sum over batch at the same time");
    }
    if (averageAcrossTimesteps && sumOverBatch)
    {
        THROW("Tacotron", name, "wrong flags chosen: cannot average across timesteps and sum over batch at the same time because of ambiguous order");
    }
    if (averageAcrossBatch && sumOverTimesteps)
    {
        THROW("Tacotron", name, "wrong flags chosen: cannot average across batch and sum over timesteps at the same time because of ambiguous order");
    }

    auto inputName = params.getInputs()[0];
    auto targetName = params.getInputs()[1];
    auto weightsName = params.getInputs()[2];

    work->add<SoftmaxCrossEntropyLoss>(name / "softmax_ce_loss", LossParams{ { inputName, targetName }, { name / "elementWiseLoss" }, LossParams::Reduction::None });
    work->add<ReduceSumLayer>(name / "sum_loss", BasicParamsWithDim{ { name / "elementWiseLoss" }, { name / "nonWeightedLoss" }, Dimension::Width });
    Name outputName = (averageAcrossTimesteps || averageAcrossBatch || sumOverTimesteps || sumOverBatch) ? name / "weightedLoss" : outputs[0];
    work->add<ElementWiseMulLayer>(name / "calc_weighted_loss", ElementWiseLayerParams{ { name / "nonWeightedLoss", weightsName }, { outputName } });

    // Reductions
    if (averageAcrossTimesteps || sumOverTimesteps)
    {
        work->add<ReduceSumLayer>(name / "sum_over_timesteps", BasicParamsWithDim{ { outputName }, { name / "summedOverTimestepsLoss" }, Dimension::Height });
        if (sumOverTimesteps)
        {
            work->add<ReduceNonZeroLayer>(name / "count_non_zero_weights_across_time", BasicParamsWithDim{ { weightsName }, { name / "divisor_time" }, Dimension::Height });
        }
        else
        {
            work->add<ReduceSumLayer>(name / "sum_weights_across_timesteps", BasicParamsWithDim{ { weightsName }, { name / "divisor_time" }, Dimension::Height });
        }
        outputName = (sumOverBatch || averageAcrossBatch) ? name / "reducedOverTimestepsLoss" : outputs[0];
        work->add<ElementWiseDivLayer>(name / "normalize_across_timesteps", ElementWiseLayerParams{ { name / "summedOverTimestepsLoss", name / "divisor_time" }, { outputName } });
    }

    if (averageAcrossBatch || sumOverBatch)
    {
        Name effectiveWeightsName = (averageAcrossTimesteps || sumOverTimesteps) ? name / "divisor_time" : weightsName;
        work->add<ReduceSumLayer>(name / "sum_over_batch",
                                  BasicParamsWithDim{ { averageAcrossTimesteps ? name / "summedOverTimestepsLoss" : outputName }, { name / "summedOverBatchLoss" }, Dimension::Batch });
        if (sumOverBatch)
        {
            work->add<ReduceNonZeroLayer>(name / "count_non_zero_weights_across_batch", BasicParamsWithDim{ { effectiveWeightsName }, { name / "divisor_batch" }, Dimension::Batch });
        }
        else
        {
            work->add<ReduceSumLayer>(name / "sum_weights_across_batch", BasicParamsWithDim{ { effectiveWeightsName }, { name / "divisor_batch" }, Dimension::Batch });
        }
        work->add<ElementWiseDivLayer>(name / "normalize_across_batch", ElementWiseLayerParams{ { name / "summedOverBatchLoss", name / "divisor_batch" }, { outputs[0] } });
    }
}

} // namespace raul::tacotron