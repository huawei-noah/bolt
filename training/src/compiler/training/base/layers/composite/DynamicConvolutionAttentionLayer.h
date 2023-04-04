// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef DYNAMIC_CONVOLUTION_ATTENTION_LAYER_H
#define DYNAMIC_CONVOLUTION_ATTENTION_LAYER_H

#include <training/base/common/Common.h>
#include <training/base/layers/parameters/trainable/DynamicConvolutionAttentionParams.h>
#include <training/base/common/NetworkParameters.h>

namespace raul
{

/**
 * @brief Dynamic Convolution Attention Layer
 *
 * Impelements Bahdanau-style (cumulative) scoring function. The attention is location-based
 *
 *
 * Inputs:
 *     1. Query  [batch, 1, 1, decoder_output_size]
 *     2. State  [batch, 1, 1, alignments_size(i.e. max_time)]
 *     3. Memory [batch, 1, max_time, encoder_output_size]
 *     4. { MemorySeqLength } [batch, 1, 1, 1] - values to create mask.
 * Outputs: Probabilities [batch, 1, 1, alignments_size(i.e. max_time)],
 *          { values } [batch, 1, max_time, encoder_output_size] (if non-shared layer and mask provided),
 *          { next_state } [batch, 1, 1, alignments_size(i.e. max_time)] (if in cumulative mode),
 *          { max_attn } [batch, 1, 1, 1] (indices of maximum numbers in probabilities).
 *
 * Steps:
 * 1. If mask provided, zero some part of initial memory input.
 * 2. Using linear layer, process memory from [batch, 1, max_time, encoder_output_size] to [batch, 1, max_time, numUnits].
 * 3. Transposed initial state input goes through conv 1D layer create raw location feature
 * 4. Using linear layer, calculate final location features.
 * 5. Multiply output from previous step by attention_v tensor (mormalize this tensor if needed).
 * 6. In order to create raw dynamic filters, process query input throught two linear layers with tanh activation between.
 * 7. To produce final dynamic features, apply some transformations: reshape and transpose raw filters.
 * 8. Apply depthwise dynamic convolution 2D to produce dynamic features: use transposed state tensor as input and calculated dynamic filters.
 * 9. Reshape obtained result
 * 10. Using linear layer, get dynamic projection of these features.
 * 11. Apply prior filters in conv 1D layer.
 * 12. Exclude too small values.
 * 13. Log the result.
 * 14. Again exclude too small values.
 * 15. Calculate Bahdanau Score (i.e. energy) using calculated features and DCA trainable params.
 * 16. Final result - activated energy (SoftMax is used).
 * 17. If cumulative mode: calculate next_state as sum of previous state + current final result. Indices ofr maximum numbers in final result also can be obtained.
 *
 * @see
 * Eric Battenberg, RJ Skerry-Ryan, Soroosh Mariooryad, Daisy Stanton, David Kao, Matt Shannon, Tom Bagby,
 * “Location-Relative Attention Mechanisms For Robust Long-Form Speech Synthesis”, ICASSP 2020
 */

class DynamicConvolutionAttentionLayer
{

  public:
    DynamicConvolutionAttentionLayer(const Name& name, const DynamicConvolutionAttentionParams& params, raul::NetworkParameters& networkParameters);

    DynamicConvolutionAttentionLayer(DynamicConvolutionAttentionLayer&&) = default;
    DynamicConvolutionAttentionLayer(const DynamicConvolutionAttentionLayer&) = delete;
    DynamicConvolutionAttentionLayer& operator=(const DynamicConvolutionAttentionLayer&) = delete;

  private:
    // General params
    size_t mNumUnits;
    bool mCumulativeMode;

    // Parameters for first conv1d layer
    size_t mLocationConvolutionFilters;
    size_t mLocationConvolutionKernelSize;

    // Size of filter in second conv1d layer
    size_t mPriorFilterSize;

    // Values to fill prior filter
    raul::dtype mPriorAlpha;
    raul::dtype mPriorBeta;

    bool mHasMask;
    size_t mMaskLen;

    // Local trainable parameters
    raul::Name mAttentionVPName;
    raul::Name mAttentionBiasName;
};

} // raul namespace

#endif