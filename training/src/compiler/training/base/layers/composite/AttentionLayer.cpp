// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "AttentionLayer.h"

#include <training/base/common/NetworkParameters.h>
#include <training/base/layers/activations/SoftMaxActivation.h>
#include <training/base/layers/basic/DropoutLayer.h>
#include <training/base/layers/basic/MaskedFillLayer.h>
#include <training/base/layers/basic/MatMulLayer.h>
#include <training/base/layers/basic/TransposeLayer.h>

namespace
{
const raul::dtype MASK_FILL_VALUE = -1e9_dt;
} // anonymous namespace

namespace raul
{

AttentionLayer::AttentionLayer(const Name& name, const DropoutParams& params, NetworkParameters& networkParameters)
{
    auto prefix = "Attention[" + name + "::ctor]: ";
    // Query, Key, Value, [Mask]
    if (params.getInputs().empty() || params.getInputs().size() > 4)
    {
        THROW("Attention", name, "wrong number of input names");
    }

    if (params.getOutputs().size() != 1 && params.getOutputs().size() != 2)
    {
        THROW("Attention", name, "wrong number of output names");
    }

    bool hasMask = params.getInputs().size() == 2 || params.getInputs().size() == 4;
    bool hasDropout = params.probability > 0.0_dt;
    bool needsPAttn = params.getOutputs().size() > 1;

    std::string maskName = hasMask ? params.getInputs().back() : "";

    auto [queryName, valueName, keyName] = std::make_tuple(params.getInputs()[0], params.getInputs()[0], params.getInputs()[0]);
    if (params.getInputs().size() > 2)
    {
        std::tie(queryName, valueName, keyName) = std::make_tuple(params.getInputs()[0], params.getInputs()[1], params.getInputs()[2]);
    }

    networkParameters.mWorkflow.add<TransposeLayer>(name / "t", TransposingParams{ keyName, name / "key_t", Dimension::Width, Dimension::Height });

    networkParameters.mWorkflow.add<MatMulLayer>(
        name / "mulQK", MatMulParams{ { queryName, name / "key_t" }, name / "scores", static_cast<float>(1.0 / sqrt(TODTYPE(networkParameters.mWorkflow.getWidth(params.getInputs()[0])))) });

    if (hasMask)
    {
        networkParameters.mWorkflow.add<MaskedFillLayer>(name / "mask", MaskedFillParams{ { name / "scores", maskName }, name / "masked_scores", MASK_FILL_VALUE, true });
    }

    auto pAttnName = (needsPAttn && !hasDropout) ? params.getOutputs()[1] : name / "sm";

    networkParameters.mWorkflow.add<SoftMaxActivation>(name / "softmax", BasicParamsWithDim{ { hasMask ? name / "masked_scores" : name / "scores" }, { pAttnName }, Dimension::Width });

    if (hasDropout)
    {
        pAttnName = needsPAttn ? params.getOutputs()[1] : name / "do";
        networkParameters.mWorkflow.add<DropoutLayer>(name / "dropout", DropoutParams{ { name / "sm" }, { pAttnName }, params.probability });
    }

    networkParameters.mWorkflow.add<MatMulLayer>(name / "mulV", MatMulParams{ { pAttnName, valueName }, params.getOutputs()[0] });
}

} // namespace raul