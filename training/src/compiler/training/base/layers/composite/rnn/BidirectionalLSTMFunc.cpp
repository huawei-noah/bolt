// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "BidirectionalLSTMFunc.h"

namespace raul
{

void BidirectionalLSTMFunc(const Name& name, const LSTMParams& params, NetworkParameters& networkParameters, BidirectionalMergeType mergeType)
{
    try
    {
        auto directParams = params;
        auto reversedParams = params;
        reversedParams.mReversed = true;

        for (auto& tensor : directParams.getOutputs())
        {
            tensor /= "direct";
        }

        for (auto& tensor : reversedParams.getOutputs())
        {
            tensor /= "reversed";
        }

        LSTMLayer(name / "direct", directParams, networkParameters);
        LSTMLayer(name / "reversed", reversedParams, networkParameters);

        const Names inputs{ directParams.getOutputs()[0], reversedParams.getOutputs()[0] };
        const Names& output = params.getOutputs();

        switch (mergeType)
        {
            case BidirectionalMergeType::Sum:
                THROW("BidirectionalLSTMFunc", name, " sum merge not implemented");
            case BidirectionalMergeType::Mul:
                THROW("BidirectionalLSTMFunc", name, " mul merge not implemented");
            case BidirectionalMergeType::ConcatHeight:
                networkParameters.mWorkflow.add<ConcatenationLayer>(name / "concat", BasicParamsWithDim(inputs, output, "height"));
                break;
            case BidirectionalMergeType::ConcatDepth:
                networkParameters.mWorkflow.add<ConcatenationLayer>(name / "concat", BasicParamsWithDim(inputs, output, "depth"));
                break;
            case BidirectionalMergeType::ConcatWidth:
            default:
                networkParameters.mWorkflow.add<ConcatenationLayer>(name / "concat", BasicParamsWithDim(inputs, output, "width"));
        }
    }
    catch (...)
    {
        THROW("BidirectionalLSTMFunc", name, "Cannot create bidirectional LSTM layer");
    }
}

} // raul namespace