// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef TACOTRON_LOSS_H
#define TACOTRON_LOSS_H

#include <training/common/Common.h>
#include <training/layers/BasicLayer.h>
#include <training/layers/parameters/trainable/TacotronParams.h>

namespace raul::tacotron
{

/**
 * @brief Tacotron Custom Loss
 *
 * @param work - pointer to Workflow
 * @param name
 * @param params:
 *   inputs:
 *           decoder_output, mel_outputs, stop_token_prediction, mel_targets, targets_lengths, stop_token_targets if !tparams.useDurationLoss
 *           decoder_output, mel_outputs, duration_outputs, mel_targets, targets_lengths, duration_targets if tparams.useDurationLoss
 *   outputs:
 *           loss [, before_loss, after_loss, stop_token_loss] if !tparams.withoutStopToken
 *           loss [, before_loss, after_loss, duration_loss] if tparams.useDurationLoss
 *           loss [, before_loss, after_loss] if tparams.withoutStopToken && !tparams.useDurationLoss
 * @param tparams - Tacotron params
 *
 * @return
 */
void AddTacotronLoss(Workflow* work, const Name& name, const BasicParams& params, const TacotronParams& tparams, bool isFinal = true);
} // namespace raul::tacotron

#endif
