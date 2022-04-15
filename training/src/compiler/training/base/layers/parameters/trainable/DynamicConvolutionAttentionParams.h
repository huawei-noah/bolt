// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef DYNAMIC_CONVOLUTION_ATTENTION_PARAMS_H
#define DYNAMIC_CONVOLUTION_ATTENTION_PARAMS_H

#include "TrainableParams.h"

namespace raul
{

/** Parameters for Dynamic Convolution Attention Layer
 * @param inputs vector of names of input tensors
 * @param outputs vector of names of output tensors
 * @param numUnits depth of the query mechanism
 * @param hparams contains hyperparameters for dense and convolutional layers
 * @param cumulateWeights whether to add previous alignments to output
 */

struct DynamicConvolutionAttentionParams : public TrainableParams
{
    /** Hyper parameters for DCA
     * @param attnFilters number of filters in first conv1d layer
     * @param attnKernel kernel size in first conv1d layer
     * @param priorFilterSize kernel size of prior filter in second conv1d layer
     * @param alpha whether to mask encoder paddings
     * @param beta specifies the type of attention - normalized or not
     */
    struct hparams
    {
        hparams() = delete;

        hparams(size_t attnFilters, size_t attnKernel, size_t priorFilterSize, size_t attentionWinSize, raul::dtype alpha, raul::dtype beta);

        size_t mAttentionFilters;
        size_t mAttentionKernel;
        size_t mPriorFilterSize;
        size_t mAttentionWindowSize;
        raul::dtype mPriorAlpha;
        raul::dtype mPriorBeta;
    };

    DynamicConvolutionAttentionParams() = delete;

    DynamicConvolutionAttentionParams(const Names& inputs, const Names& outputs, size_t numUnits, const hparams& params, bool cumulateWeights = false, bool paramFrozen = false);

    DynamicConvolutionAttentionParams(const Names& inputs,
                                      const Names& outputs,
                                      const Name& sharedLayer,
                                      size_t numUnits,
                                      const hparams& params,
                                      bool cumulateWeights = false,
                                      bool paramFrozen = false);

    // Standard parameters
    size_t mNumUnits;
    hparams mHparams;
    bool mCumulateWeights;
    void print(std::ostream& stream) const override;
};

}

#endif