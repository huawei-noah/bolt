// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "TacotronDataInitializationLayer.h"

#include "impl/TacotronDataInitializationLayerCPU.h"
#include "impl/TacotronDataInitializationLayerGPU.h"

namespace raul
{
namespace tacotron
{

TacotronDataInitializationLayer::TacotronDataInitializationLayer(const Name& name, const BasicParams& params, const TacotronParams& tparams, raul::NetworkParameters& networkParameters)
    : BasicLayer(name, "TacotronDataInitialization", params, networkParameters)
{
    using namespace std;
    auto prefix = mTypeName + "[" + mName + "::ctor]: ";
    if (mInputs.size() != 1)
    {
        THROW(mTypeName, mName, "wrong number of inputs");
    }
    size_t expectedParamsCount = 0;
    if (tparams.useDurationPrediction)
    {
        expectedParamsCount = 2 + 2 * tparams.decoderLstmUnits.size();
    }
    else
    {
        expectedParamsCount = 1 + 2 * tparams.decoderLstmUnits.size() + (2 + (tparams.useAttentionRnn ? 2 : 0));
    }

    if (mOutputs.size() != expectedParamsCount)
    {
        THROW(mTypeName, mName, "wrong number of outputs. Expected " + to_string(expectedParamsCount) + ", got " + to_string(mOutputs.size()));
    }

    DECLARE_IMPL(TacotronDataInitializationLayer, TacotronDataInitializationLayerCPU<MemoryManager>, TacotronDataInitializationLayerGPU, TacotronDataInitializationLayerCPU<MemoryManagerFP16>)

    vector<WShape> shapes{ { BS(), 1u, 1u, tparams.numMels } };
    if (!tparams.useDurationPrediction)
    {
        mInitialAlignmentsName = mOutputs[2];
        size_t attentionSize = mNetworkParams.mWorkflow.getWidth(mInputs[0]);
        size_t allignmentsSize = mNetworkParams.mWorkflow.getHeight(mInputs[0]);

        shapes.emplace_back(BS(), 1u, 1u, attentionSize);
        shapes.emplace_back(BS(), 1u, 1u, allignmentsSize);

        if (tparams.useAttentionRnn)
        {
            shapes.emplace_back(BS(), 1u, 1u, tparams.attentionRnnUnits);
            shapes.emplace_back(BS(), 1u, 1u, tparams.attentionRnnUnits);
        }
    }
    else
    {
        size_t durationsSize = mNetworkParams.mWorkflow.getWidth(mInputs[0]);
        shapes.emplace_back(BS(), 1u, 1u, durationsSize);
    }

    shapes.emplace_back(BS(), 1u, 1u, tparams.decoderLstmUnits[0]);

    for (size_t i = 0; i < mOutputs.size(); ++i)
    {
        WShape shape;
        if (i < size(shapes))
        {
            shape = shapes[i];
        }
        else
        {
            shape = shapes.back();
        }

        mNetworkParams.mWorkflow.tensorNeeded(mName, mOutputs[i], shape, DEC_FORW_WRIT);
    }
}

}
} // namespace raul