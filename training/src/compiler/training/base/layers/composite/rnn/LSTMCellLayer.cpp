// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "LSTMCellLayer.h"

#include <training/base/layers/activations/SigmoidActivation.h>
#include <training/base/layers/activations/TanhActivation.h>
#include <training/base/layers/basic/ConcatenationLayer.h>
#include <training/base/layers/basic/ElementWiseMulLayer.h>
#include <training/base/layers/basic/ElementWiseSumLayer.h>
#include <training/base/layers/basic/FixedBiasLayer.h>
#include <training/base/layers/basic/SlicerLayer.h>
#include <training/base/layers/basic/trainable/LinearLayer.h>
#include <training/base/layers/composite/rnn/LSTMFusedGatesCalcLayer.h>
#include <training/base/layers/composite/rnn/ZoneoutLayer.h>

namespace raul
{

LSTMCellLayer::LSTMCellLayer(const Name& name, const LSTMCellParams& params, NetworkParameters& networkParameters)
{
    try
    {

        MEASURE_BLOCK("LSTMCellLayer[" + name + "::ctor]")
        if (params.getInputs().size() != 3U)
        {
            THROW("LSTMCellLayer", name, " wrong number of input names");
        }
        if (params.getOutputs().size() != 2U)
        {
            THROW("LSTMCellLayer", name, " wrong number of output names");
        }

        this->buildLayer(name, params, networkParameters);
    }
    catch (...)
    {
        THROW("LSTMCellLayer", name, "Cannot create LSTMCellLayer layer");
    }
}

void LSTMCellLayer::buildLayer(const Name& name, const LSTMCellParams& params, NetworkParameters& networkParameters)
{
    const auto parts = 4U;
    const auto nameInput = params.getInputs()[0];
    const auto nameHidden = params.getInputs()[1];
    const auto nameCell = params.getInputs()[2];
    const auto nameNewHidden = params.getOutputs()[0];
    const auto nameNewCell = params.getOutputs()[1];

    const auto sizeHidden = params.mHiddenFeatures ? *params.mHiddenFeatures : networkParameters.mWorkflow.getWidth(nameHidden);

    const bool useZoneout = params.mZoneout != 0.0_dt;

    const auto sharedLayer = params.getSharedLayer();
    bool useFusion = params.mUseFusion;
    if (!useFusion)
    {
        if (params.mUseSingleParamTensor)
        {
            if (!params.getSharedWeights().empty() && params.getSharedWeights().size() < 2U)
            {
                THROW("LSTMCellLayer", name, " wrong number of weight names");
            }

            networkParameters.mWorkflow.add<ConcatenationLayer>(name / "concat", BasicParamsWithDim({ nameInput, nameHidden }, { name / "concat" }, Dimension::Width));

            BasicParams linearParams = !params.getSharedWeights().empty() ? BasicParams{ { name / "concat" }, { name / "gates" }, { params.getSharedWeights()[0], params.getSharedWeights()[1] } }
                                                                          : !sharedLayer.empty() ? BasicParams{ { name / "concat" }, { name / "gates" }, sharedLayer / "linear" }
                                                                                                 : BasicParams{ { name / "concat" }, { name / "gates" } };

            networkParameters.mWorkflow.add<LinearLayer>(name / "linear", LinearParams(linearParams, sizeHidden * parts, params.mBias, params.frozen));
        }
        else
        {
            if (!params.getSharedWeights().empty() && params.getSharedWeights().size() < 4U)
            {
                THROW("LSTMCellLayer", name, " wrong number of weight names");
            }

            BasicParams ihParams{ { nameInput }, { name / "linear_ih" } }, hhParams{ { nameHidden }, { name / "linear_hh" } };
            if (!params.getSharedWeights().empty())
            {
                ihParams = { { nameInput }, { name / "linear_ih" }, { params.getSharedWeights()[0], params.getSharedWeights()[1] } };
                hhParams = { { nameHidden }, { name / "linear_hh" }, { params.getSharedWeights()[2], params.getSharedWeights()[3] } };
            }
            else if (!sharedLayer.empty())
            {
                ihParams = { { nameInput }, { name / "linear_ih" }, sharedLayer / "linear_ih" };
                hhParams = { { nameHidden }, { name / "linear_hh" }, sharedLayer / "linear_hh" };
            }

            networkParameters.mWorkflow.add<LinearLayer>(name / "linear_ih", LinearParams(ihParams, sizeHidden * parts, params.mBias, params.frozen));
            networkParameters.mWorkflow.add<LinearLayer>(name / "linear_hh", LinearParams(hhParams, sizeHidden * parts, params.mBias, params.frozen));
            networkParameters.mWorkflow.add<ElementWiseSumLayer>(name / "gates", ElementWiseLayerParams({ name / "linear_ih", name / "linear_hh" }, { name / "gates" }));
        }

        networkParameters.mWorkflow.add<SlicerLayer>(name / "slice", SlicingParams(name / "gates", { name / "gates[0]", name / "gates[1]", name / "gates[2]", name / "gates[3]" }, "width"));
        networkParameters.mWorkflow.add<SigmoidActivation>(name / "sigmoid_input", HSigmoidActivationParams({ name / "gates[0]" }, { name / "sigmoid_input" }));
        if (params.mForgetBias != 0.0_dt)
        {
            networkParameters.mWorkflow.add<FixedBiasLayer>(name / "forget_bias", FixedBiasParams({ name / "gates[1]" }, { name / "gates[1]_biased" }, params.mForgetBias));
            networkParameters.mWorkflow.add<SigmoidActivation>(name / "sigmoid_forget", HSigmoidActivationParams({ name / "gates[1]_biased" }, { name / "sigmoid_forget" }));
        }
        else
        {
            networkParameters.mWorkflow.add<SigmoidActivation>(name / "sigmoid_forget", HSigmoidActivationParams({ name / "gates[1]" }, { name / "sigmoid_forget" }));
        }

        networkParameters.mWorkflow.add<TanhActivation>(name / "tanh_gates", HSigmoidActivationParams({ name / "gates[2]" }, { name / "tanh_gates" }));
        networkParameters.mWorkflow.add<ElementWiseMulLayer>(name / "mul_input", ElementWiseLayerParams({ name / "sigmoid_input", name / "tanh_gates" }, { name / "mul_input" }));
        networkParameters.mWorkflow.add<ElementWiseMulLayer>(name / "mul_forget", ElementWiseLayerParams({ name / "sigmoid_forget", nameCell }, { name / "mul_forget" }));
        networkParameters.mWorkflow.add<ElementWiseSumLayer>(name / "sum_new_cell_state",
                                                             ElementWiseLayerParams({ name / "mul_input", name / "mul_forget" }, { useZoneout ? name / "new_cell_zoneout" : nameNewCell }));

        networkParameters.mWorkflow.add<SigmoidActivation>(name / "sigmoid_output", HSigmoidActivationParams({ name / "gates[3]" }, { name / "sigmoid_output" }));
        networkParameters.mWorkflow.add<TanhActivation>(name / "tanh_new_cell_state",
                                                        HSigmoidActivationParams({ useZoneout ? name / "new_cell_zoneout" : nameNewCell }, { name / "tanh_internal_new_cell_state" }));
        networkParameters.mWorkflow.add<ElementWiseMulLayer>(
            name / "mul_new_hidden_state", ElementWiseLayerParams({ name / "sigmoid_output", name / "tanh_internal_new_cell_state" }, { useZoneout ? name / "new_hidden_zoneout" : nameNewHidden }));
        if (useZoneout)
        {
            networkParameters.mWorkflow.add<ZoneoutLayer>(name / "zoneout_hidden", ZoneoutParams({ name / "new_hidden_zoneout", nameHidden }, { nameNewHidden }, params.mZoneout));
            networkParameters.mWorkflow.add<ZoneoutLayer>(name / "zoneout_cell", ZoneoutParams({ name / "new_cell_zoneout", nameCell }, { nameNewCell }, params.mZoneout));
        }
    }
    else
    {
        BasicParams fusedParams{ { nameInput, nameHidden, nameCell }, { nameNewHidden, nameNewCell } };
        if (!params.getSharedWeights().empty())
        {
            fusedParams = { { nameInput, nameHidden, nameCell }, { nameNewHidden, nameNewCell }, params.getSharedWeights() };
        }
        else if (!sharedLayer.empty())
        {
            fusedParams = { { nameInput, nameHidden, nameCell }, { nameNewHidden, nameNewCell }, sharedLayer / "linear" };
        }
        networkParameters.mWorkflow.add<LSTMFusedGatesCalcLayer>(
            name / "linear", LSTMFusedGatesCalcParams(fusedParams, sizeHidden * parts, params.mBias, params.mZoneout, params.mUseSingleParamTensor, params.mForgetBias, params.frozen));
    }
}

} // namespace raul