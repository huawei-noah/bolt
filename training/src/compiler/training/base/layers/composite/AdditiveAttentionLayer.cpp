// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "AdditiveAttentionLayer.h"

#include <training/base/layers/activations/SoftMaxActivation.h>
#include <training/base/layers/activations/TanhActivation.h>
#include <training/base/layers/basic/DropoutLayer.h>
#include <training/base/layers/basic/ElementWiseSumLayer.h>
#include <training/base/layers/basic/MaskedFillLayer.h>
#include <training/base/layers/basic/MatMulLayer.h>
#include <training/base/layers/basic/ReduceSumLayer.h>
#include <training/base/layers/basic/ReshapeLayer.h>
#include <training/base/layers/basic/TransposeLayer.h>

namespace
{
const raul::dtype MASK_FILL_VALUE = -1e9_dt;
} // anonymous namespace

namespace raul
{

AdditiveAttentionLayer::AdditiveAttentionLayer(const Name& name, const DropoutParams& params, NetworkParameters& networkParameters)
{
    auto prefix = "AdditiveAttention[" + name + "::ctor]: ";
    // Query, Key, Value, [Mask]
    if (params.getInputs().empty() || params.getInputs().size() > 4)
    {
        THROW("AdditiveAttention", name, "wrong number of input names");
    }

    if (params.getOutputs().size() != 1 && params.getOutputs().size() != 2)
    {
        THROW("AdditiveAttention", name, "wrong number of output names");
    }

    bool hasMask = params.getInputs().size() == 4 || params.getInputs().size() == 2;
    bool hasDropout = params.probability > 0.0_dt;
    bool needsPAttn = params.getOutputs().size() > 1;

    std::string maskName = hasMask ? params.getInputs().back() : "";

    auto [queryName, valueName, keyName] = std::make_tuple(params.getInputs()[0], params.getInputs()[0], params.getInputs()[0]);
    if (params.getInputs().size() > 2)
    {
        std::tie(queryName, valueName, keyName) = std::make_tuple(params.getInputs()[0], params.getInputs()[1], params.getInputs()[2]);
    }

    const int Tq = static_cast<int>(networkParameters.mWorkflow.getHeight(queryName));
    const int Tv = static_cast<int>(networkParameters.mWorkflow.getHeight(valueName));
    const int dim = static_cast<int>(networkParameters.mWorkflow.getWidth(valueName));

    // Layers

    // Reshape query to [batch, Tq, 1, dim]
    networkParameters.mWorkflow.add<raul::ReshapeLayer>(name / "reshapeQ", raul::ViewParams{ { queryName }, { name / "queryReshaped" }, Tq, 1, dim });

    // Sum query and value
    networkParameters.mWorkflow.add<raul::ElementWiseSumLayer>(name / "sumQK", raul::ElementWiseLayerParams{ { name / "queryReshaped", keyName }, { name / "sum" } });

    // Tanh activation
    networkParameters.mWorkflow.add<raul::TanhActivation>(name / "tanhQK", raul::BasicParams{ { name / "sum" }, { name / "tanh" } });

    // Reduction sum
    networkParameters.mWorkflow.add<raul::ReduceSumLayer>(name / "rSumQK", raul::BasicParamsWithDim{ { name / "tanh" }, { name / "scores" }, raul::Dimension::Width });

    if (hasMask)
    {
        networkParameters.mWorkflow.add<raul::MaskedFillLayer>(name / "mask", MaskedFillParams{ { name / "scores", maskName }, name / "masked_scores", MASK_FILL_VALUE, true });
    }

    // Transpose for faster softmax calculation
    networkParameters.mWorkflow.add<raul::TransposeLayer>(name / "t",
                                                          TransposingParams{ { hasMask ? name / "masked_scores" : name / "scores" }, { name / "scores_t" }, Dimension::Width, Dimension::Height });

    // Get probabilities
    networkParameters.mWorkflow.add<raul::SoftMaxActivation>(name / "softmax", raul::BasicParamsWithDim{ { name / "scores_t" }, { name / "sm" }, raul::Dimension::Width });

    // If pAttn needed, then return it
    auto pAttnName = (!hasDropout && needsPAttn) ? params.getOutputs()[1] : name / "smReshaped";

    // Reshape result to [batch, 1, Tq, Tv]
    networkParameters.mWorkflow.add<raul::ReshapeLayer>(name / "reshapeSm", raul::ViewParams{ { name / "sm" }, { pAttnName }, 1, Tq, Tv });

    if (hasDropout)
    {
        pAttnName = needsPAttn ? params.getOutputs()[1] : name / "do";
        networkParameters.mWorkflow.add<raul::DropoutLayer>(name / "dropout", DropoutParams{ { name / "smReshaped" }, { pAttnName }, params.probability });
    }

    networkParameters.mWorkflow.add<raul::MatMulLayer>(name / "mulV", raul::MatMulParams{ { pAttnName, valueName }, params.getOutputs()[0] });
}

}