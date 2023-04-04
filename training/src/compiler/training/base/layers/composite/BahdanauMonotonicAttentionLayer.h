// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef BAHDANAU_MONOTONIC_ATTENTION_LAYER_H
#define BAHDANAU_MONOTONIC_ATTENTION_LAYER_H

#include <training/base/common/Common.h>
#include <training/base/layers/parameters/trainable/BahdanauAttentionParams.h>
#include <training/base/common/NetworkParameters.h>

namespace raul
{

/**
 * @brief Bahdanau Monotonic Attention Layer
 *
 * This layer produce Bahdanau-style attention.
 *
 *
 * Inputs:
 *     1. Query  [batch, 1, 1, decoder_output_size]
 *     2. State  [batch, 1, 1, alignments_size(i.e. max_time)]
 *     3. Memory [batch, 1, max_time, encoder_output_size]
 *     4. { MemorySeqLength } [batch, 1, 1, 1] - values to create mask.
 *     5. { MaskValues } [batch, 1, alignments_size, 1] or broadcastable to this size -
 *     values, which replace weights from softmax according
 *     mask, generated using memory sequence length parameter.
 * Outputs: Probabilities [batch, 1, 1, alignments_size(i.e. max_time)],
 *          { values } [batch, 1, max_time, encoder_output_size] (if non-shared layer and mask provided),
 *          { max_attn } [batch, 1, 1, 1] (indices of maximum numbers in probabilities).
 *
 * Steps:
 * 1. Using linear layer or matrix multiplication, process Query from [batch, 1, 1, decoder_output_size] to [batch, 1, 1, numUnits].
 * 2. Using linear layer or matrix multiplication, Process Memory from [batch, 1, max_time, encoder_output_size] to [batch, 1, max_time, numUnits]. Mask Memory if needed.
 * 3. Sum obtained tensors.
 * 4. Take tanh activation from the result.
 * 5. Multiply output from previous step by attention_v tensor (mormalize this tensor if needed).
 * 6. Take reduction sum.
 * 7. Add bias to the result.
 * 8. Apply mask if needed: replace elements of a tensor by some specified value.
 * 9. Add noise to sigmoid input.
 * 10. Take sigmoid activation.
 * 11. Reflect calculated weights relativelty to unit.
 * If simple monotonic attention:
 *  12. Shift the result by 1 along Dimension::Width, fill new empty space by 1.0.
 *  13. Clamp obtained result in [some small value, 1.0].
 *  14. Take element-wise natural logarithm.
 *  15. Calculate cumulative sum along Dimension::Width.
 *  16. Take element-wise exponent.
 *  17. Clamp the result again.
 *  18. Divide State tensor by calculated one.
 *  19. Take reduction sum.
 *  20. Result is element-wise multiplication of normalized State tensor, sigmoid weights and exponential output.
 * If stepwise monotonic attention:
 *  12. Multiply reflected weights by state tensor
 *  13. Shift obtained result by 1 along Dimension::Width, fill new empty space by 0.0.
 *  14. Multiply state tensor by sigmoid output.
 *  15. Result is element-wise sum of outputs from steps 13 and 14.
 * @see
 * Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio, “Neural Machine Translation by Jointly Learning to Align and Translate”, ICLR 2015
 */
class BahdanauMonotonicAttentionLayer
{

  public:
    BahdanauMonotonicAttentionLayer(const Name& name, const BahdanauAttentionParams& params, NetworkParameters& networkParameters);

    BahdanauMonotonicAttentionLayer(BahdanauMonotonicAttentionLayer&&) = default;
    BahdanauMonotonicAttentionLayer(const BahdanauMonotonicAttentionLayer&) = delete;
    BahdanauMonotonicAttentionLayer& operator=(const BahdanauMonotonicAttentionLayer&) = delete;

  private:
    size_t mNumUnits;
    bool mNormalize;
    dtype mSigmoidNoise;
    dtype mScoreBiasInit;
    std::string mMode;

    // Internal params which need gradient
    Name mAttentionVName;
    Name mScoreBiasName;
    Name mAttentionBName;
    Name mAttentionGName;

    // Mask
    bool mHasMask;

    // Stepwise or not
    bool mStepwise;
};

} // raul namespace

#endif