// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "DynamicConvolutionAttentionParams.h"

namespace raul
{

DynamicConvolutionAttentionParams::hparams::hparams(size_t attnFilters, size_t attnKernel, size_t priorFilterSize, size_t attentionWinSize, raul::dtype alpha, raul::dtype beta)
    : mAttentionFilters(attnFilters)
    , mAttentionKernel(attnKernel)
    , mPriorFilterSize(priorFilterSize)
    , mAttentionWindowSize(attentionWinSize)
    , mPriorAlpha(alpha)
    , mPriorBeta(beta)
{
}

DynamicConvolutionAttentionParams::DynamicConvolutionAttentionParams(const Names& inputs, const Names& outputs, size_t numUnits, const hparams& params, bool cumulateWeights, bool paramFrozen)
    : TrainableParams(inputs, outputs, paramFrozen)
    , mNumUnits(numUnits)
    , mHparams(params)
    , mCumulateWeights(cumulateWeights)
{
}

DynamicConvolutionAttentionParams::DynamicConvolutionAttentionParams(const Names& inputs,
                                                                     const Names& outputs,
                                                                     const Name& sharedLayer,
                                                                     size_t numUnits,
                                                                     const hparams& params,
                                                                     bool cumulateWeights,
                                                                     bool paramFrozen)
    : TrainableParams(inputs, outputs, sharedLayer, paramFrozen)
    , mNumUnits(numUnits)
    , mHparams(params)
    , mCumulateWeights(cumulateWeights)
{
}

void DynamicConvolutionAttentionParams::print(std::ostream& stream) const
{
    BasicParams::print(stream);
    stream << "num units = " << mNumUnits << ", cumulate output = " << mCumulateWeights << "\n";
    stream << "Hyperparameters:\n";
    stream << "attention filter number = " << mHparams.mAttentionFilters << ", attention kernel size = " << mHparams.mAttentionKernel;
    stream << ", prior filter size = " << mHparams.mPriorFilterSize << ", attention window size = " << mHparams.mAttentionWindowSize;
    stream << ", prior alpha = " << mHparams.mPriorAlpha << ", prior beta = " << mHparams.mPriorBeta;
}

}
