// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "MultiHeadAttention.h"

#include <algorithm>

#include <training/base/common/MemoryManager.h>
#include <training/base/layers/basic/ReshapeLayer.h>
#include <training/base/layers/basic/TransposeLayer.h>
#include <training/base/layers/basic/trainable/LinearLayer.h>
#include <training/base/layers/composite/AttentionLayer.h>

namespace raul
{

MultiHeadAttentionLayer::MultiHeadAttentionLayer(const Name& name, const MultiHeadAttentionParams& params, NetworkParameters& networkParameters)
    : mHeads(params.heads)
    , mDropout(params.probability)
{
    MEASURE_BLOCK("MultiHeadAttention[" + name + "::ctor]")
    auto prefix = "MultiHeadAttention[" + name + "::ctor]: ";
    // Query, Key, Value, [Mask] or Query, [Mask]
    if (params.getInputs().empty() || params.getInputs().size() > 4)
    {
        THROW("MultiHeadAttention", name, "wrong number of input names");
    }
    if (params.getOutputs().size() != 1 && params.getOutputs().size() != 2)
    {
        THROW("MultiHeadAttention", name, "wrong number of output names");
    }

    bool hasMask = params.getInputs().size() == 4 || params.getInputs().size() == 2;

    std::string maskName = params.getInputs().size() == 4 ? params.getInputs()[3] : (params.getInputs().size() == 2 ? params.getInputs()[1] : "");
    std::string attnName = name / "attn";

    auto [queryName, valueName, keyName] = std::make_tuple(params.getInputs()[0], params.getInputs()[0], params.getInputs()[0]);
    if (params.getInputs().size() > 2)
    {
        std::tie(queryName, valueName, keyName) = std::make_tuple(params.getInputs()[0], params.getInputs()[1], params.getInputs()[2]);
    }

    size_t d_model = networkParameters.mWorkflow.getWidth(params.getInputs()[0]);
    int d_k = static_cast<int>(d_model / mHeads);

    std::string var[] = { "q", "k", "v" };
    std::string names[] = { queryName, valueName, keyName };

    for (size_t i = 0; i < 3; ++i)
    {
        std::string suffix = "[" + std::to_string(i) + "]";
        networkParameters.mWorkflow.add<LinearLayer>(name / "linears" + suffix, LinearParams{ { names[i] }, { name / var[i] + "_l" }, d_model, true, params.frozen });
        networkParameters.mWorkflow.add<ReshapeLayer>(name / "reshape_" + var[i], ViewParams{ name / var[i] + "_l", name / var[i] + "_v", -1, mHeads, d_k });
        networkParameters.mWorkflow.add<TransposeLayer>(name / "transp_" + var[i], TransposingParams{ name / var[i] + "_v", name / var[i] + "_t", Dimension::Depth, Dimension::Height });
    }

    Names attnInputs = { name / "q_t", name / "v_t", name / "k_t" };

    if (hasMask)
    {
        attnInputs.push_back(maskName);
    }

    AttentionLayer(name / "attn", DropoutParams{ attnInputs, { name / "attn" }, static_cast<float>(mDropout) }, networkParameters);
    networkParameters.mWorkflow.add<TransposeLayer>(name / "transp_attn", TransposingParams{ name / "attn", name / "attn_t", Dimension::Depth, Dimension::Height });

    if (!params.finalTransform)
    {
        networkParameters.mWorkflow.add<ReshapeLayer>(name / "reshape_attn", ViewParams{ name / "attn_t", params.getOutputs()[0], 1, -1, mHeads * d_k });
    }
    else
    {
        networkParameters.mWorkflow.add<ReshapeLayer>(name / "reshape_attn", ViewParams{ name / "attn_t", name / "attn_v", 1, -1, mHeads * d_k });
        networkParameters.mWorkflow.add<LinearLayer>(name / "linears[3]", LinearParams{ { name / "attn_v" }, { params.getOutputs()[0] }, d_model, true, params.frozen });
    }
}

} // namespace raul