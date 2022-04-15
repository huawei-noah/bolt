// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "MaskedCrossEntropy.h"
#include "SequenceMaskLayer.h"

#include <training/layers/basic/ElementWiseCompareLayer.h>
#include <training/layers/basic/ElementWiseDivLayer.h>
#include <training/layers/basic/ReduceNonZeroLayer.h>
#include <training/layers/basic/ReduceSumLayer.h>
#include <training/layers/basic/SelectLayer.h>
#include <training/layers/basic/SplitterLayer.h>
#include <training/layers/basic/TensorLayer.h>
#include <training/loss/SigmoidCrossEntropyLoss.h>

namespace raul::tacotron
{
void AddMaskedCrossEntropy(Workflow* work, const Name& name, const BasicParams& params, size_t outputsPerStep, float epsilon, bool isFinal)
{
    const auto& inputs = params.getInputs();
    const auto& outputs = params.getOutputs();

    std::string prefix = "Tacotron::MaskedCrossEntropy[" + name + "]: ";
    if (inputs.size() != 3)
    {
        THROW("Tacotron", name, "wrong number of input names");
    }

    if (outputs.size() != 1)
    {
        THROW("Tacotron", name, "wrong number of output names");
    }

    auto inputName = params.getInputs()[0];
    auto targetName = params.getInputs()[1];
    auto targetLengthsName = params.getInputs()[2];

    auto maskName = name / "mask";

    work->add<SequenceMaskLayer>(name / "sequence_mask", BasicParams{ { inputName, targetLengthsName }, { maskName } }, outputsPerStep);
    work->add<SigmoidCrossEntropyLoss>(name / "loss", LossParams{ { inputName, targetName, maskName }, { name / "loss" }, LossParams::Reduction::None, false });
    work->add<ReduceSumLayer>(name / "reduce_sum", BasicParamsWithDim{ { name / "loss" }, { name / "reduce_sum" }, Dimension::Default });

    auto nameForNonZero = name / "loss";
    auto divisor = name / "count_non_zero_exact";
    work->add<ReduceNonZeroLayer>(name / "count_non_zero_exact", BasicParamsWithDim{ { nameForNonZero }, { divisor }, Dimension::Default });

    if (epsilon > 0)
    {
        work->add<TensorLayer>(name / "threshold", TensorParams({ name / "threshold" }, raul::WShape(raul::BS(), 1U, 1U, 1U), epsilon));

        work->add<ElementWiseCompareLayer>(name / "cmp", ElementWiseComparisonLayerParams{ { nameForNonZero, name / "threshold" }, { name / "non_zero_thresholded" }, true, "greater" });

        work->add<ReduceNonZeroLayer>(name / "count_non_zero_thresholded", BasicParamsWithDim{ { name / "non_zero_thresholded" }, { name / "count_non_zero_thresholded" }, Dimension::Default });

        work->add<SplitterLayer>(name / "splitter", BasicParams{ { name / "count_non_zero_thresholded" }, { name / "non_zero_condition" } });

        nameForNonZero = name / "non_zero";
        work->add<SelectLayer>(name / "select_non_zero", ElementWiseLayerParams{ { name / "non_zero_condition", name / "count_non_zero_thresholded", divisor }, { name / "count_non_zero" } });

        divisor = name / "count_non_zero";
    }

    work->add<ElementWiseDivLayer>(name / "div", ElementWiseLayerParams{ { name / "reduce_sum", divisor }, { name / "output" } });

    work->add<LossWrapperHelperLayer>(name / "helper", raul::BasicParams{ { name / "output" }, { outputs[0] } }, isFinal);
}
} // namespace raul::tacotron
