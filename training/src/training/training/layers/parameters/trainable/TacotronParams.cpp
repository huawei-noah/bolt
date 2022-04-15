// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "TacotronParams.h"

using namespace std;

namespace
{
template<typename T>
string vec2str(const vector<T>& v)
{
    string s = "(";
    for (size_t i = 0; i < v.size(); ++i)
    {
        s += to_string(v[i]);
        if (i != v.size() - 1)
        {
            s += ", ";
        }
    }
    return s + ")";
}
}

namespace raul
{

void TacotronParams::print(std::ostream& stream) const
{
    BasicParams::print(stream);
    map<string, string> m;
    m["outputs_per_step"] = to_string(outputsPerStep);
    m["speaker_embedding_size"] = to_string(speakerEmbeddingSize);
    m["num_mels"] = to_string(numMels);
    m["tacotron_teacher_forcing_ratio"] = to_string(teacherForcingRatio);
    m["batch_norm_position"] = batchnormBeforeActivation ? "before" : "after";
    m["attention"] = attentionType;
    m["stepwise_mode"] = attentionMode;
    m["attention_dim"] = to_string(attentionDim);
    m["prior_alpha"] = to_string(attentionPriorAlpha);
    m["prior_beta"] = to_string(attentionPriorBeta);
    m["prior_filter_size"] = to_string(attentionPriorFilterSize);
    m["attention_win_size"] = to_string(attentionWindowSize);
    m["attention_rnn_units"] = to_string(attentionRnnUnits);
    m["sigmoid_noise"] = to_string(attentionSigmoidNoise);
    m["score_bias_init"] = to_string(attentionScoreBiasInit);
    m["normalize_attention_energy"] = attentionNormalizeEnergy ? "True" : "False";

    m["postnet_num_layers"] = to_string(postnetKernelSize.size());
    m["postnet_channels"] = to_string(postnetChannels);
    m["postnet_kernel_size"] = vec2str(postnetKernelSize);

    m["prenet_layers"] = vec2str(prenetLayers);

    m["decoder_lstm_units"] = vec2str(decoderLstmUnits);
    m["decoder_layers"] = to_string(decoderLstmUnits.size());

    m["loss_multipliers"] = vec2str(lossMultipliers);

    m["to_square_error"] = maskUseSquaredError ? "True" : "False";
    m["without_stop_token"] = withoutStopTokenLoss ? "True" : "False";
    m["use_attention_rnn"] = useAttentionRnn ? "True" : "False";
    m["use_residual_rnn"] = useResidualRnn ? "True" : "False";
    m["use_residual_encoder"] = useResidualEncoder ? "True" : "False";
    m["use_language_embedding"] = useLanguageEmbedding ? "True" : "False";
    m["trainable_spk_embedding"] = trainableSpeakerEmbedding ? "True" : "False";

    m["language_embedding_len"] = to_string(languageEmbeddingLen);

    m["tacotron_postnet_dropout_rate"] = to_string(postnetDropoutRate);
    m["tacotron_zoneout_rate"] = to_string(zoneoutRate);
    m["tacotron_dropout_rate"] = to_string(dropoutRate);

    m["use_score_bias"] = "True";

    for (const auto& i : m)
    {
        stream << ", " << i.first << ": " << i.second;
    }
}

void TacotronParams::ensureConsistency(bool /*checkInputsAndOutputs*/) const
{
    if (useDurationPrediction)
    {
        if (!attentionType.empty() && attentionType != "None")
        {
            THROW_NONAME("TacotronParams", "duration prediction and attention can't be used together");
        }
        if (!withoutStopTokenLoss)
        {
            THROW_NONAME("TacotronParams", "duration prediction and stop token loss can't be used together");
        }
        if (useAttentionRnn)
        {
            THROW_NONAME("TacotronParams", "duration prediction and attention rnn can't be used together");
        }
    }
    else
    {
        if (attentionType.empty())
        {
            THROW_NONAME("TacotronParams", "attention mechanism not specified");
        }
        if (attentionType != "StepwiseMonotonic" && attentionType != "DynamicConvolutional")
        {
            THROW_NONAME("TacotronParams", "unsupported attention mechanism '" + attentionType + "'");
        }
    }
}

} // namespace raul
