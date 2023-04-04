// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef LOCATION_SENSITIVE_ATTENTION_LAYER_H
#define LOCATION_SENSITIVE_ATTENTION_LAYER_H

#include <training/base/common/Common.h>
#include <training/base/layers/parameters/trainable/LocationSensitiveAttentionParams.h>
#include <training/base/common/NetworkParameters.h>

namespace raul
{

/**
 * @brief Location Sensitive Attention Layer
 *
 * Impelements Bahdanau-style (cumulative) scoring function.
 * Usually referred to as "hybrid" attention (content-based + location-based)
 * Extends the additive attention described in:
 * "D. Bahdanau, K. Cho, and Y. Bengio, “Neural machine transla-
 * tion by jointly learning to align and translate,” in Proceedings of ICLR, 2015."
 * to use previous alignments as additional location features.
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
 * 1. Calculate transition probability using transition layer agent. Default value is 0.5 (optional).
 * 2. Using linear layer, process query tensor from [batch, 1, 1, decoder_output_size] to [batch, 1, 1, numUnits]
 * 3. Using linear layer, process memory from [batch, 1, max_time, encoder_output_size] to [batch, 1, max_time, numUnits] (only in parent layer).
 * 4. Extract location-based features from state tensor using 1D convolution, passing the result through linear layer.
 * 5. Calculate energy (i. e. location_sensitive_score).
 * 6. Calculate new alignments using softmax activation
 *  a. If smoothing enabled, use sigmoid and normalize instead.
 *  b. If stepwise monotonic constraint applied, calculate sigmoid activation of noisy energy, mix it with state and shifted state tensors.
 * 7. If use forward option enabled, mix calculated alignments with previous state. Proportion depends on transition probability tensor.
 * 8. In cumulative mode, calculate next_state output tensor as sum of new alignments and previous state.
 * 9. If it is needed, indices of max numbers in alingments can be calcuated too.
 *
 * @see
 * J. K. Chorowski, D. Bahdanau, D. Serdyuk, K. Cho, Y. Ben-gio,
 * “Attention-based models for speech recognition”, Advances in Neural Information Processing Systems, 2015, pp. 577–585.
 */

class LocationSensitiveAttentionLayer
{

  public:
    LocationSensitiveAttentionLayer(const Name& name, const LocationSensitiveAttentionParams& params, raul::NetworkParameters& networkParameters);

    LocationSensitiveAttentionLayer(LocationSensitiveAttentionLayer&&) = default;
    LocationSensitiveAttentionLayer(const LocationSensitiveAttentionLayer&) = delete;
    LocationSensitiveAttentionLayer& operator=(const LocationSensitiveAttentionLayer&) = delete;

  private:
    // General params
    size_t mNumUnits;
    bool mCumulativeMode;

    // Parameters for first conv1d layer
    size_t mLocationConvolutionFilters;
    size_t mLocationConvolutionKernelSize;

    bool mHasMask;
    size_t mMaskLen;

    dtype mTransitionProba;

    // Local trainable parameters
    raul::Name mAttentionVPName;
    raul::Name mAttentionBiasName;
};

} // raul namespace

#endif