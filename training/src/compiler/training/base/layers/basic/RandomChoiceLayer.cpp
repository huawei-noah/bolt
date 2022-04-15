// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "RandomChoiceLayer.h"

#include <training/base/common/Random.h>

namespace raul
{

RandomChoiceLayer::RandomChoiceLayer(const Name& name, const RandomChoiceParams& params, NetworkParameters& networkParameters)
    : BasicLayer(name, "RandomChoice", params, networkParameters)
    , mGenerator(static_cast<unsigned>(params.mSeed))
    , mSelectedInput(0)
{
    auto prefix = mTypeName + "[" + mName + "::ctor]: ";

    if (mOutputs.size() != 1)
    {
        THROW(mTypeName, mName, "wrong number of output names");
    }

    auto probs = params.mRatios;
    if (probs.size() != mInputs.size() && probs.size() != mInputs.size() - 1)
    {
        THROW(mTypeName, mName, "wrong number of ratios in params");
    }

    auto sum = std::accumulate(probs.begin(), probs.end(), 0.f);
    if (probs.size() == mInputs.size() - 1)
    {
        if (sum > 1.f)
        {
            THROW(mTypeName, mName, "sum of probabilities > 1");
        }
        probs.push_back(1.f - sum);
        sum = 1.f;
    }

    mSections.push_back(probs[0] / sum);
    for (size_t i = 1; i < probs.size(); ++i)
    {
        mSections.push_back(mSections.back() + probs[i] / sum);
    }

    shape outputShape{ 1u, mNetworkParams.mWorkflow.getDepth(mInputs[0]), mNetworkParams.mWorkflow.getHeight(mInputs[0]), mNetworkParams.mWorkflow.getWidth(mInputs[0]) };
    for (const auto& input : mInputs)
    {
        shape inputShape{ 1u, mNetworkParams.mWorkflow.getDepth(input), mNetworkParams.mWorkflow.getHeight(input), mNetworkParams.mWorkflow.getWidth(input) };
        if (inputShape != outputShape)
        {
            THROW(mTypeName, mName, "input tensor shapes must be the same");
        }

        mNetworkParams.mWorkflow.copyDeclaration(name, input, raul::Workflow::Usage::Forward, raul::Workflow::Mode::Read);
        mNetworkParams.mWorkflow.copyDeclaration(name, input, input.grad(), DEC_BACK_WRIT_ZERO);
    }

    mNetworkParams.mWorkflow.copyDeclaration(name, mInputs[0], mOutputs[0], DEC_FORW_WRIT);
    mNetworkParams.mWorkflow.copyDeclaration(name, mOutputs[0], mOutputs[0].grad(), DEC_BACK_READ);
}

void RandomChoiceLayer::forwardComputeImpl(NetworkMode)
{

    auto& output = mNetworkParams.mMemoryManager[mOutputs[0]];

    auto p = random::uniform::rand<raul::dtype>(0., 1.);
    mSelectedInput = mSections.size();
    for (size_t i = 0; i < mSections.size(); ++i)
    {
        if (p < mSections[i])
        {
            mSelectedInput = i;
            break;
        }
    }

    const auto& input = mNetworkParams.mMemoryManager[mInputs[mSelectedInput]];
    output = TORANGE(input);
}

void RandomChoiceLayer::backwardComputeImpl()
{

    const Tensor& deltas = mNetworkParams.mMemoryManager[mOutputs[0].grad()];

    for (size_t i = 0; i < mInputs.size(); ++i)
    {
        auto input = mInputs[i];
        // if (mNetworkParams.isGradNeeded(input))
        {
            Tensor& prevLayerDelta = mNetworkParams.mMemoryManager[input.grad()];
            if (i == mSelectedInput)
            {
                prevLayerDelta += deltas;
            }
        }
    }
}

} // namespace raul