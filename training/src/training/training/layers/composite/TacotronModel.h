// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef TACOTRON_MODEL_H
#define TACOTRON_MODEL_H

#include <training/common/Common.h>
#include <training/layers/parameters/trainable/TacotronParams.h>

namespace raul
{

/**
 * @brief Encoder embedding + Decoder + Postnet part of custom Tacotron 2 model for FSL single speaker task
 *
 * Reference: Nizhniy Novgorod model (see python subdir)
 *
 * The decoder is an autoregressive recurrent neural network which predicts
 * a mel spectrogram from the encoded input sequence one frame at a time.
 *
 * [1]J. Shen et al., “Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions”
 * arXiv:1712.05884 [cs], Feb. 2018. Available: http://arxiv.org/abs/1712.05884.
 *
 * @param tparams:
 *   TacotronParams:
 *     Inputs:
 *       - encoder_output
 *       - [speaker_embedding] if trainableSpeakerEmbedding == false
 *       - [encoded_residual] if useResidualEncoder == true
 *       - [embedded_languages] if useLanguageEmbedding == true
 *       - [duration_targets] if useDurationPrediction == true
 *       - [inputs] encoder inputs (if useDurationPrediction == true)
 *       - input_lengths
 *       - mel_targets: float32 Tensor with shape [N, 1, T_out, M] where N is batch size,
 *                   T_out is number of steps in the output time series, M is num_mels,
 *                   and values are entries in the mel spectrogram.
 *                   Only needed for training.
 *     Outputs:
 *       - decoder_output
 *       - mel_output
 *       - [durations] if useDurationPrediction == true
 *       - [stop_token_prediction] if useDurationPrediction == false
 */
void AddSingleSpeakerFslTacotronModel(Workflow* work, const Name& name, const TacotronParams& tparams);

} // raul namespace
#endif
