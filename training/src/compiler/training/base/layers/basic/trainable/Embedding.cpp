// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "Embedding.h"

#include <string>

#include "impl/EmbeddingCPU.h"

namespace
{
const size_t NoPadding = std::numeric_limits<std::size_t>::max();
} // anonymous namespace

namespace raul
{

Embedding::Embedding(const Name& name, const EmbeddingParams& params, NetworkParameters& networkParameters)
    : TrainableLayer(name, "Embedding", params, networkParameters)
    , mDictionarySize(params.dictionarySize)
    , mEmbeddingSize(params.embeddingSize)
    , mPaddingIdx(params.paddingClass >= 0 ? params.paddingClass : NoPadding)
    , mScaleGradByFreq(params.scaleGradByFrequency)
{
    auto prefix = mTypeName + "[" + mName + "::ctor]: ";

    if (mInputs.size() != 1)
    {
        THROW(mTypeName, mName, "wrong number of input names");
    }
    if (mOutputs.size() != 1)
    {
        THROW(mTypeName, mName, "wrong number of output names");
    }

    DECLARE_IMPL(EmbeddingLayer, EmbeddingCPU<MemoryManager>, EmbeddingCPU<MemoryManagerFP16>)

    mInputName = mInputs[0];
    mOutputName = mOutputs[0];

    if (params.scaleOutput)
    {
        mOutputScale = TODTYPE(sqrt(static_cast<dtype>(mEmbeddingSize)));
    }

    if (mNetworkParams.mWorkflow.getWidth(mInputName) != 1 && mNetworkParams.mWorkflow.getHeight(mInputName) != 1)
    {
        THROW(mTypeName, mName, "either input tensor width or height must be 1");
    }

    mNetworkParams.mWorkflow.copyDeclaration(mName, mInputName, raul::Workflow::Usage::ForwardAndBackward, raul::Workflow::Mode::Read);

    // inputs.getWidth or inputs.getHeight - sentence length
    // inputs.getBatchSize - number of sentences in batch
    mNetworkParams.mWorkflow.tensorNeeded(
        mName,
        mOutputName,
        raul::WShape{ raul::BS(), mNetworkParams.mWorkflow.getDepth(mInputName), mNetworkParams.mWorkflow.getHeight(mInputName) * mNetworkParams.mWorkflow.getWidth(mInputName), mEmbeddingSize },
        DEC_FORW_WRIT);

    mNetworkParams.mWorkflow.copyDeclaration(mName, mOutputName, mOutputName.grad(), DEC_BACK_READ);

    mNetworkParams.mWorkflow.tensorNeeded(mName, mWeightsName, WShape{ 1u, 1u, mDictionarySize, mEmbeddingSize }, DEC_TRAINABLE);

    if (!mFrozen)
    {
        mNetworkParams.mWorkflow.copyDeclaration(mName, mWeightsName, mWeightsName.grad(), DEC_TRAINABLE_GRAD);
    }
}

}