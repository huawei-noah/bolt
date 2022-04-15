// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "LocationSensitiveAttentionParams.h"

namespace raul
{

LocationSensitiveAttentionParams::hparams::hparams(size_t attnFilters, size_t attnKernel, bool useTransAgent, bool useStepwiseMonotonicConstraintType)
    : mAttentionFilters(attnFilters)
    , mAttentionKernel(attnKernel)
    , mUseTransAgent(useTransAgent)
    , mUseStepwiseMonotonicConstraintType(useStepwiseMonotonicConstraintType)
{
}

LocationSensitiveAttentionParams::LocationSensitiveAttentionParams(const Names& inputs,
                                                                   const Names& outputs,
                                                                   size_t numUnits,
                                                                   const hparams& params,
                                                                   bool cumulateWeights,
                                                                   bool smoothing,
                                                                   dtype sigmoidNoise,
                                                                   bool useForward,
                                                                   bool paramFrozen)
    : TrainableParams(inputs, outputs, paramFrozen)
    , mNumUnits(numUnits)
    , mHparams(params)
    , mCumulateWeights(cumulateWeights)
    , mSmoothing(smoothing)
    , mSigmoidNoise(sigmoidNoise)
    , mUseForward(useForward)
{
}

LocationSensitiveAttentionParams::LocationSensitiveAttentionParams(const Names& inputs,
                                                                   const Names& outputs,
                                                                   const Name& sharedLayer,
                                                                   size_t numUnits,
                                                                   const hparams& params,
                                                                   bool cumulateWeights,
                                                                   bool smoothing,
                                                                   dtype sigmoidNoise,
                                                                   bool useForward,
                                                                   bool paramFrozen)
    : TrainableParams(inputs, outputs, sharedLayer, paramFrozen)
    , mNumUnits(numUnits)
    , mHparams(params)
    , mCumulateWeights(cumulateWeights)
    , mSmoothing(smoothing)
    , mSigmoidNoise(sigmoidNoise)
    , mUseForward(useForward)
{
}

void LocationSensitiveAttentionParams::print(std::ostream& stream) const
{
    BasicParams::print(stream);
    stream << "internal dimensionality = " << mNumUnits << ", cumulate output: " << (mCumulateWeights ? "true" : "false") << ",\n";
    stream << "smoothing: " << (mSmoothing ? "true" : "false") << ", sigmoid noise = " << mSigmoidNoise << ", use forward: " << (mUseForward ? "true" : "false") << "\n";
    stream << "Hyperparameters:\n";
    stream << "attention filter number = " << mHparams.mAttentionFilters << ", attention kernel size = " << mHparams.mAttentionKernel;
    stream << ", calculate transition probability: " << (mHparams.mUseTransAgent ? "true" : "false")
           << ", use stepwise monotonic constraint: " << (mHparams.mUseStepwiseMonotonicConstraintType ? "true" : "false");
}

}
