// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "SequentialMetaLayer.h"

// d.polubotko(TODO): implement
#if 0
#include <training/network/OperatorDef.h>

namespace raul
{

SequentialMetaLayer::SequentialMetaLayer(const Name& name, const SequentialParams& params, raul::NetworkParameters& networkParameters)
    : MetaLayer(name, "SequentialMetaLayer", params, networkParameters)
{
    MEASURE_BLOCK("SequentialMetaLayer[" + mName + "::ctor]")
    
    //d.polubotko(TODO): implement
        /*
    NameUnorderedSet internalTensors;

    for (size_t i = 0; i < params.mOperators.size(); ++i)
    {
        auto op = params.mOperators[i];
        if (i < params.mOperators.size() - 1)
        {
            internalTensors.insert(op.mParams->getOutputs().begin(), op.mParams->getOutputs().end());
        }
        OperatorDef* oper = mNetDef.addOp(name / op.mName, op.mType, op.mParams->clone());

        if (!params.getWeights().empty())
        {
            Names& operWeights = oper->mParams->getWeights();
            operWeights.clear();

            for (auto& weight : params.getWeights())
            {
                Name parentLayer = weight.getPrefix();
                Name subLayerWeightName = weight.getLastName();
                operWeights.push_back(parentLayer / op.mName / subLayerWeightName);
            }
        }
    }

    for (const auto& tensor : internalTensors)
    {
        mNetDef.renameTensor(tensor, name / tensor);
    }*/
}
} // namespace raul
#endif
