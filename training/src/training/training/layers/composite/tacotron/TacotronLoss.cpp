// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "TacotronLoss.h"
#include "MaskedCrossEntropy.h"
#include "MaskedLoss.h"

#include <training/layers/basic/ElementWiseSumLayer.h>
#include <training/layers/basic/ScaleLayer.h>
#include <training/layers/basic/SplitterLayer.h>
#include <training/loss/L1Loss.h>
#include <training/loss/MSELoss.h>

namespace raul::tacotron
{
using namespace std;

void AddTacotronLoss(Workflow* work, const Name& name, const BasicParams& params, const TacotronParams& tparams, bool isFinal)
{
    auto prefix = "TacotronLoss[" + name + "]: ";
    const auto& inputs = params.getInputs();
    const auto& losses = params.getOutputs();

    if ((tparams.lossMultipliers.size() < 3 && (!tparams.withoutStopTokenLoss || tparams.useDurationPrediction)) ||
        (tparams.lossMultipliers.size() < 2 && (tparams.withoutStopTokenLoss && !tparams.useDurationPrediction)))
    {
        THROW("TacotronLoss", name, "wrong number of loss multipliers");
    }

    if (inputs.size() != 6)
    {
        THROW("TacotronLoss", name, "wrong number of inputs");
    }

    if (losses.size() != 1 &&
        (((tparams.withoutStopTokenLoss && !tparams.useDurationPrediction) && losses.size() != 3) || ((!tparams.withoutStopTokenLoss || tparams.useDurationPrediction) && losses.size() != 4)))
    {
        THROW("TacotronLoss", name, "wrong number of partial losses");
    }

    const auto decoderOutput = inputs[0];
    const auto melOutputs = inputs[1];
    const auto stopTokenPredictions = tparams.withoutStopTokenLoss ? Name() : inputs[2];
    const auto durationOutputs = tparams.useDurationPrediction ? inputs[2] : Name();

    const auto melTargets = inputs[3];
    const auto targetLengths = inputs[4];
    const auto durationTargets = tparams.useDurationPrediction ? inputs[5] : Name();
    const auto stopTokenTargets = tparams.withoutStopTokenLoss ? Name() : inputs[5];

    Name bName = losses.size() > 1 ? losses[1] : name / "before";
    Name aName = losses.size() > 1 ? losses[2] : name / "after";
    Name sName = losses.size() > 3 ? losses[3] : (tparams.useDurationPrediction ? name / "duration_loss" : name / "stop_token_loss");
    Name outputLoss = name / "loss";

    MaskedLossType lossType = tparams.useDurationPrediction ? MaskedLossType::L1_L2 : (tparams.maskUseSquaredError ? MaskedLossType::L2 : MaskedLossType::L1);

    AddMaskedLoss(work, name / "before", { { decoderOutput, melTargets, targetLengths }, { bName } }, tparams.outputsPerStep, lossType, false);
    AddMaskedLoss(work, name / "after", { { melOutputs, melTargets, targetLengths }, { aName } }, tparams.outputsPerStep, lossType, false);
    if (tparams.useDurationPrediction)
    {
        work->add<MSELoss>(name / "duration_loss", LossParams{ { durationOutputs, durationTargets }, { sName }, LossParams::Reduction::Mean, false });
    }
    else if (!tparams.withoutStopTokenLoss)
    {
        AddMaskedCrossEntropy(work, name / "stop_token_loss", { { stopTokenPredictions, stopTokenTargets, targetLengths }, { sName } }, tparams.outputsPerStep, tparams.maskedSigmoidEpsilon, false);
    }

    if (tparams.lossMultipliers[0] != 1.f)
    {
        work->add<ScaleLayer>(name / "scale_before_loss", ScaleParams({ bName }, { name / "before_loss_scaled" }, tparams.lossMultipliers[0]));
        bName = name / "before_loss_scaled";
    }
    if (tparams.lossMultipliers[1] != 1.f)
    {
        work->add<ScaleLayer>(name / "scale_after_loss", ScaleParams({ aName }, { name / "after_loss_scaled" }, tparams.lossMultipliers[1]));
        aName = name / "after_loss_scaled";
    }

    if (tparams.withoutStopTokenLoss && !tparams.useDurationPrediction)
    {
        work->add<ElementWiseSumLayer>(name / "total_loss", ElementWiseLayerParams{ { bName, aName }, { outputLoss } });
    }
    else
    {
        if (tparams.lossMultipliers[2] != 1.f)
        {
            auto lName = tparams.useDurationPrediction ? name / "scale_duration_loss" : name / "scale_stop_token_loss";
            auto sNameScaled = tparams.useDurationPrediction ? name / "duration_loss_scaled" : name / "stop_token_loss_scaled";
            work->add<ScaleLayer>(lName, ScaleParams({ sName }, { sNameScaled }, tparams.lossMultipliers[2]));
            sName = sNameScaled;
        }
        work->add<ElementWiseSumLayer>(name / "total_loss", ElementWiseLayerParams{ { bName, aName, sName }, { outputLoss } });
    }

    work->add<LossWrapperHelperLayer>(name / "helper", raul::BasicParams{ { outputLoss }, { losses[0] } }, isFinal);
}

} // namespace raul::tacotron
