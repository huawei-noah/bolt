// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "MaskedLoss.h"
#include "SequenceMaskLayer.h"

#include <training/layers/basic/ElementWiseSumLayer.h>
#include <training/loss/L1Loss.h>
#include <training/loss/MSELoss.h>

namespace raul::tacotron
{
void AddMaskedLoss(Workflow* work, const Name& name, const BasicParams& params, size_t outputsPerStep, MaskedLossType type, bool isFinal)
{
    const auto& inputs = params.getInputs();
    const auto& outputs = params.getOutputs();

    std::string prefix = "Tacotron::AddMaskedLoss[" + name + "]: ";
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

    switch (type)
    {
        case MaskedLossType::L1:
            work->add<L1Loss>(name / "loss", LossParams{ { inputName, targetName, maskName }, { params.getOutputs()[0] }, LossParams::Reduction::Sum_Over_Nonzero_Weights, isFinal });
            break;
        case MaskedLossType::L2:
            work->add<MSELoss>(name / "loss", LossParams{ { inputName, targetName, maskName }, { params.getOutputs()[0] }, LossParams::Reduction::Sum_Over_Nonzero_Weights, isFinal });
            break;
        case MaskedLossType::L1_L2:
            work->add<L1Loss>(name / "loss" / "l1", LossParams{ { inputName, targetName, maskName }, { name / "loss" / "l1" }, LossParams::Reduction::Sum_Over_Nonzero_Weights, isFinal });
            work->add<MSELoss>(name / "loss" / "l2", LossParams{ { inputName, targetName, maskName }, { name / "loss" / "l2" }, LossParams::Reduction::Sum_Over_Nonzero_Weights, isFinal });
            work->add<ElementWiseSumLayer>(name / "sum", ElementWiseLayerParams{ { name / "loss" / "l1", name / "loss" / "l2" }, { params.getOutputs()[0] } });
            break;
    }
}
} // namespace raul::tacotron
