// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef TACOTRON_DATA_INITIALIZATION_LAYER_H
#define TACOTRON_DATA_INITIALIZATION_LAYER_H

#include <training/common/Common.h>
#include <training/layers/BasicLayer.h>
#include <training/layers/parameters/trainable/TacotronParams.h>

namespace raul
{
namespace tacotron
{
/*
 * inputs:
 *   - encoder_output
 * outputs:
 *   - zero_input - zeros of shape [N, 1, 1, numMels]
 *   if useDurationPrediction:
 *     - initial_duration_prediction - zeros of shape [N, 1, 1, positionalEmbeddingDim + width(encoder_output)]
 *   else:
 *     - initial_attention - zeros of shape [N, 1, 1, width(encoder_output)]
 *     - initial_alignment - Dirac distribution of shape [N, 1, 1, attentionDim]
 *     - [initial_attention_rnn_state] - 2 zero tensors of shape [N, 1, 1, attentionRnnUnits], if useAttentionRnn==true
 *   - initial_rnn_cell_state - 2*decoderLayers zero tensors of shape [N, 1, 1, decoderLstmUnits]
 */
class TacotronDataInitializationLayer : public BasicLayer
{
  public:
    TacotronDataInitializationLayer(const Name& name, const BasicParams& params, const TacotronParams& tparams, raul::NetworkParameters& networkParameters);

    TacotronDataInitializationLayer(TacotronDataInitializationLayer&&) = default;
    TacotronDataInitializationLayer(const TacotronDataInitializationLayer&) = delete;
    TacotronDataInitializationLayer& operator=(const TacotronDataInitializationLayer&) = delete;

  private:
    Name mInitialAlignmentsName;

    template<typename MM>
    friend class TacotronDataInitializationLayerCPU;
    friend class TacotronDataInitializationLayerGPU;
};
}
} // namespace raul::tacotron
#endif
