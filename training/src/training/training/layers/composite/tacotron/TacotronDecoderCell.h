// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef TACOTRON_DECODER_CELL_H
#define TACOTRON_DECODER_CELL_H

#include <training/common/Common.h>
#include <training/layers/parameters/trainable/TacotronParams.h>

namespace raul
{
namespace tacotron
{
/**
 * params.inputs:
 *   - encoder_output
 *   - input_lengths: Tensor with shape [N, 1, 1, 1] where N is batch size and values are the lengths
 *    of each sequence in inputs.
 *   - previous_step_output: encoder output or output from previous decoding step
 *   - state: optional previous state tensors:
 *     - attention (initial values - zero)
 *     - alignments (initial values - Dirac distribution)
 *     - attention_rnn_state [optional - if tparams.useAttentionRnn == true]
 *       - h
 *       - c
 *     - rnn cell state:
 *       - rnn_cell_h (initial values - zero)
 *       - rnn_cell_c (initial values - zero)
 * params.outputs:
 *   - cell_outputs
 *   - stop_tokens
 *   - next state:
 *     - attention
 *     - alignments
 *     - attention_rnn_state [optional - if tparams.useAttentionRnn == true]
 *        - h
 *        - c
 *     - rnn cell state:
 *       - rnn_cell_h
 *       - rnn_cell_c
 */
void AddDecoderCell(Workflow* work, const Name& name, const BasicParams& params, const TacotronParams& tparams);
}
} // namespace raul::tacotron
#endif
