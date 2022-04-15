// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef DURATION_PREDICTOR_H
#define DURATION_PREDICTOR_H

#include <training/common/Common.h>
#include <training/layers/parameters/trainable/TacotronParams.h>

namespace raul
{
namespace tacotron
{
/**
 * @brief Tacotron Decoder RNN
 * @param work
 * @param name
 * @param params:
 *   inputs: encoder_output, input_lengths, duration_targets[, input_mask]
 *   outputs: durations_output, durations, upsampled_output, positional_embeddings
 * @param tparams
 * @return
 *
 */
void AddDurationPredictor(Workflow* work, const Name& name, const BasicParams& params, const TacotronParams& tparams);
}
} // namespace raul::tacotron
#endif
