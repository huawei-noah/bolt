// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "GRUCellLayer.h"

#include "GRUFusedGatesCalcLayer.h"
#include <training/base/layers/activations/SigmoidActivation.h>
#include <training/base/layers/activations/TanhActivation.h>
#include <training/base/layers/basic/ElementWiseMulLayer.h>
#include <training/base/layers/basic/ElementWiseSubLayer.h>
#include <training/base/layers/basic/ElementWiseSumLayer.h>
#include <training/base/layers/basic/SlicerLayer.h>
#include <training/base/layers/basic/trainable/LinearLayer.h>

namespace raul
{

GRUCellLayer::GRUCellLayer(const Name& name, const GRUCellParams& params, NetworkParameters& networkParameters)
{
    if (params.getInputs().size() != 2U)
    {
        THROW("GRUCellLayer", name, "wrong number of input names");
    }
    if (params.getOutputs().size() != 1U)
    {
        THROW("GRUCellLayer", name, "wrong number of output names");
    }

    this->buildLayer(name, params, networkParameters);
}

void GRUCellLayer::buildLayer(const Name& name, const GRUCellParams& params, NetworkParameters& networkParameters)
{
    const auto parts = 3U;
    const auto nameInput = params.getInputs()[0];
    const auto nameHidden = params.getInputs()[1];
    const auto nameNewHidden = params.getOutputs()[0];

    const auto sizeHidden = networkParameters.mWorkflow.getWidth(nameHidden);

    const auto sharedLayer = params.getSharedLayer();

    if (!params.getSharedWeights().empty() && (params.getSharedWeights().size() < 2U || params.getSharedWeights().size() > 4U))
    {
        THROW("GRUCellLayer", name, "wrong number of weight names");
    }

    if (!params.mUseFusion)
    {
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

        // Calculate gates
        networkParameters.mWorkflow.add<LinearLayer>(name / "linear_ih", LinearParams(ihParams, sizeHidden * parts, params.mUseBiasForInput, params.frozen));
        networkParameters.mWorkflow.add<LinearLayer>(name / "linear_hh", LinearParams(hhParams, sizeHidden * parts, params.mUseBiasForHidden, params.frozen));

        networkParameters.mWorkflow.add<SlicerLayer>(name / "slice_ih",
                                                     SlicingParams(name / "linear_ih", { name / "half_ih_gates[0]", name / "half_ih_gates[1]", name / "half_ih_gates[2]" }, "width"));
        networkParameters.mWorkflow.add<SlicerLayer>(name / "slice_hh",
                                                     SlicingParams(name / "linear_hh", { name / "half_hh_gates[0]", name / "half_hh_gates[1]", name / "half_hh_gates[2]" }, "width"));

        networkParameters.mWorkflow.add<ElementWiseSumLayer>(name / "gates[0]", ElementWiseLayerParams({ name / "half_ih_gates[0]", name / "half_hh_gates[0]" }, { name / "gates[0]" }));
        networkParameters.mWorkflow.add<SigmoidActivation>(name / "sigmoid_gates[0]", HSigmoidActivationParams({ name / "gates[0]" }, { name / "sigmoid_gates[0]" }));
        networkParameters.mWorkflow.add<ElementWiseSumLayer>(name / "gates[1]", ElementWiseLayerParams({ name / "half_ih_gates[1]", name / "half_hh_gates[1]" }, { name / "gates[1]" }));
        networkParameters.mWorkflow.add<SigmoidActivation>(name / "sigmoid_gates[1]", HSigmoidActivationParams({ name / "gates[1]" }, { name / "sigmoid_gates[1]" }));

        networkParameters.mWorkflow.add<ElementWiseMulLayer>(name / "half_gates[2]", ElementWiseLayerParams({ name / "sigmoid_gates[0]", name / "half_hh_gates[2]" }, { name / "half_gates[2]" }));
        networkParameters.mWorkflow.add<ElementWiseSumLayer>(name / "gates[2]", ElementWiseLayerParams({ name / "half_ih_gates[2]", name / "half_gates[2]" }, { name / "gates[2]" }));
        networkParameters.mWorkflow.add<TanhActivation>(name / "tanh_gates[2]", HSigmoidActivationParams({ name / "gates[2]" }, { name / "tanh_gates[2]" }));

        // Calculate new hidden state
        networkParameters.mWorkflow.add<ElementWiseMulLayer>(name / "calculate_old_part", ElementWiseLayerParams({ name / "sigmoid_gates[1]", nameHidden }, { name / "old_part" }));
        networkParameters.mWorkflow.add<ElementWiseMulLayer>(name / "calculate_update_part", ElementWiseLayerParams({ name / "sigmoid_gates[1]", name / "tanh_gates[2]" }, { name / "update_part" }));
        networkParameters.mWorkflow.add<ElementWiseSubLayer>(name / "calculate_new_part", ElementWiseLayerParams({ name / "tanh_gates[2]", name / "update_part" }, { name / "new_part" }));
        networkParameters.mWorkflow.add<ElementWiseSumLayer>(name / "sum_new_hidden_state", ElementWiseLayerParams({ name / "old_part", name / "new_part" }, { nameNewHidden }));
    }
    else
    {
        BasicParams fusedParams{ { nameInput, nameHidden }, { nameNewHidden } };
        if (!params.getSharedWeights().empty())
        {
            fusedParams = { { nameInput, nameHidden }, { nameNewHidden }, { params.getSharedWeights()[0], params.getSharedWeights()[1], params.getSharedWeights()[2], params.getSharedWeights()[3] } };
        }
        else if (!sharedLayer.empty())
        {
            fusedParams = { { nameInput, nameHidden }, { nameNewHidden }, sharedLayer / "linear" };
        }

        networkParameters.mWorkflow.add<GRUFusedGatesCalcLayer>(name / "linear",
                                                                GRUFusedGatesCalcParams(fusedParams, sizeHidden * parts, params.mUseBiasForInput, params.mUseBiasForHidden, params.frozen));
    }
}

} // namespace raul