// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef TACOTRON_PARAMS_H
#define TACOTRON_PARAMS_H

#include <string>
#include <vector>

#include "TrainableParams.h"

namespace raul
{

/**
 * @param input names of input tensors (input_ids, [token_type_ids], [attention_mask])
 * input_ids - Tensor of shape [batch_size, 1, 1, sequence_length] with the word token indices in the vocabulary
 * token_type_ids - an optional tensor of shape [batch_size, 1, 1, sequence_length] with the token types indices selected in [0, 1].
 *   Type 0 corresponds to a `sentence A` and type 1 corresponds to a `sentence B` token (see BERT paper for more details)
 * attention_mask - an optional tensor of shape [batch_size, 1, 1, sequence_length] with indices selected in [0, 1].
 *   It's a mask to be used if the input sequence length is smaller than the max input sequence length in the current batch.
 *   It's the mask that we typically use for attention when a batch has varying length sentences.
 * @param output names of output tensors
 *   2: final_hidden_state, pooled_output. final_hidden_state - full sequence of hidden-states corresponding to the last attention block
 *   1 + numHiddenLayers: all_hidden_states, pooled_output. all_hidden_states - a list of the full sequences of encoded-hidden-states at the end of each attention block)
 * @param paramVocabSize Vocabulary size of `inputs_ids` in `BertModel`
 * @param paramTypeVocabSize The vocabulary size of the `token_type_ids` passed into `BertModel`
 * @param paramNumHiddenLayers Number of hidden layers in the Transformer encoder
 * @param paramHiddenSize Size of the encoder layers and the pooler layer
 * @param paramIntermediateSize The size of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder
 * @param paramHeads Number of attention heads for each attention layer in the Transformer encoder
 * @param paramMaxPositionEmbeddings The maximum sequence length that this model might ever be used with. Typically set this to something large just in case (e.g., 512 or 1024 or 2048)
 * @param paramHiddenActivation The non-linear activation function name in the encoder and pooler. Could be "gelu", "relu" and "swish" or ID of any activation layer (e.g. LOG_SOFTMAX_ACTIVATION)
 * @param paramHiddenDropout The dropout probability for all fully connected layers in the embeddings, encoder, and pooler
 * @param paramAttentionDropout The dropout ratio for the attention probabilities
 */
struct TacotronParams : public TrainableParams
{
    TacotronParams() = delete;
    TacotronParams(const Names& inputs, const Names& outputs, const Names& weights, LayerExecutionTarget layerExecTarget = LayerExecutionTarget::Default, bool frozen = false)
        : TrainableParams(inputs, outputs, weights, layerExecTarget, frozen)
    {
    }

    // general
    uint32_t outputsPerStep = 3; // number of frames to generate at each decoding step (increase to speed up computation and allows for higher batch size, decreases G&L audio quality)
    // bool stopAtAny = true;                 // determines whether the decoder should stop when predicting <stop> to any frame or to all of them (True works pretty well)
    bool batchnormBeforeActivation = false; // determines whether we use batch norm before or after the activation function (relu). Matter for debate.
    // bool clipOutputs = false;               // whether to clip spectrograms to T2_output_range (even in loss computation). ie: Don't penalize model for exceeding output range and bring back to
    // borders.
    // float lowerBoundDecay = 0.1f; // small regularizer for noise synthesis by adding small range of penalty for silence regions. Set to 0 to clip in Tacotron range. Used only when clipOutputs=true

    uint32_t embeddingDim = 512; // dimension of embedding space
    // uint32_t encoderConvLayers = 3; // number of encoder convolutional layers
    // uint32_t encoderConvKernelSize = 5; // size of encoder convolution filters for each layer
    // uint32_t encoderConvChannels = 512; // number of encoder convolutions filters for each layer
    // uint32_t encoderLstmUnits = 256; // number of lstm units for each direction (forward and backward)

    bool useDurationPrediction = false;        // use duration prediction instead of attention
    uint32_t durationPredictorLstmUnits = 512; // duration_predictor_lstm_units
    uint32_t positionalEmbeddingDim = 32;
    uint32_t maxMelLength = 150;
    // attention
    std::string attentionType =
        "StepwiseMonotonic";                // ['LocationSensitive', 'StepwiseMonotonic', 'Forward', 'DynamicConvolutional'], set to empty string on None when not used (e.g. for duration prediction)
    std::string attentionMode = "parallel"; // ['hard', 'parallel']
    uint32_t attentionDim = 128;
    float attentionSigmoidNoise = 0.3f;
    float attentionScoreBiasInit = 0.5f;
    bool attentionNormalizeEnergy = true;

    float attentionPriorAlpha = 0.1f;
    float attentionPriorBeta = 0.9f;
    uint32_t attentionPriorFilterSize = 11;
    uint32_t attentionWindowSize = 7;
    uint32_t attentionRnnUnits = 512;

    bool useResidualEncoder = false;               // if true, encoded_residual must be provided as an additional non-trainable input
    bool useLanguageEmbedding = false;             // if true, embedded_languages must be provided as an additional non-trainable input
    bool concatConditionsWithEncoderOutput = true; // if true, conditions will be concatenated to encoder output; else - to prenet output
    uint32_t languageEmbeddingLen = 16;
    uint32_t encodedResidualLen = 16;

    bool useAttentionRnn = false;
    bool useResidualRnn = false;
    // these params are used for LocationSensitiveAttention only
    // bool attentionSmoothing = false; // whether to smooth the attention normalization function
    uint32_t attentionFilters = 32;    // number of attention convolution filters
    uint32_t attentionKernelSize = 31; // kernel size of attention convolution
    // bool useCummulativeWeights = false; // Whether to accumulate (sum) all previous attention weights or simply feed previous weights (Recommended: True)
    // bool useTransitionAgent = false;

    // decoder
    std::vector<uint32_t> prenetLayers = { 128, 128 };                      // size - number of PreNet layers, values - number of units in corresponding layer
    std::vector<uint32_t> decoderLstmUnits = std::vector<uint32_t>(3, 512); // number of decoder lstm units on each layer
    // uint32_t decoderMaxIters = 2500; // max decoder steps during inference (Just for safety from infinite loop cases)
    std::vector<uint32_t> postnetKernelSize = std::vector<uint32_t>(5, 5); // size of postnet convolution filters for each layer
    uint32_t postnetChannels = 64;                                         // number of postnet convolution filters for each layer
    float postnetDropoutRate = 0.1f;                                       // dropout rate for PostNet
    // cbhg
    // uint32_t cbhgConvKernels = 8; // all kernel sizes from 1 to cbhg_kernels will be used in the convolution bank of CBHG to act as "K-grams"
    // uint32_t cbhgConvChannels = 128; // channels of the convolution bank
    // uint32_t cbhgPoolingSize = 2; // pooling size of the CBHG
    // uint32_t cbhgProjectionChannels = 128; // projection channels of the CBHG (1st projection, 2nd is automatically set to num_mels)
    // uint32_t cbhgProjectionKernelSize = 3; // kernel_size of the CBHG projections
    // uint32_t cbhgHighwayLayers = 4; // number of HighwayNet layers
    // uint32_t cbhgHighwayUnits = 128; // number of units used in HighwayNet fully connected layers
    // uint32_t cbhgRnnUnits = 128; // Number of GRU units used in bidirectional RNN of CBHG block. CBHG output is 2x rnn_units in shape
    // loss
    bool maskUseSquaredError = false; // whether to use MSE or MAE
    // bool maskEncoder; // whether to mask encoder padding while computing attention. Set to true for better prosody but slower convergence.
    // bool maskDecoder = true; // whether to use loss mask for padded sequences (if False, <stop_token> loss function will not be weighted, else recommended pos_weight = 20)
    // float maskCrossEntropyPositionWeight = 1.f; // use class weights to reduce the stop token classes imbalance (by adding more penalty on False Negatives (FN)) (1 = disabled)
    // bool predictLinear; // Whether to add a post-processing network to the Tacotron to predict linear spectrograms (True mode Not tested!!)

    // regularization
    // float regularizationWeight = 1e-6f; // regularization weight (for L2 regularization)
    // bool regularizationScaleToOutputRange; // Whether to rescale regularization weight to adapt for outputs range (used when regularizationWeight is high and biasing the model)
    float zoneoutRate = 0.1f; // zoneout rate for all LSTM cells in the network
    float dropoutRate = 0.5f; // dropout rate for all convolutional layers and PreNet

    // bool useSymmetricMels = true; // whether to scale the data to be symmetric around 0. (Also multiplies the output range by 2, faster and cleaner convergence)
    // float maxAbsValue = 4.f;     // max absolute value of data. If symmetric, data will be [-max, max] else [0, max] (Must not be too big to avoid gradient explosion, not too small for fast
    // convergence)

    uint32_t speakerEmbeddingSize = 256; // size of speaker embedding for multi-speaker TTS

    // audio
    uint32_t numMels = 20; // number of mel-spectrogram channels and local conditioning dimensionality
    uint32_t maxMelFrames = 1500;

    // losses
    bool withoutStopTokenLoss = false;  // if true zero stop token loss is added to total loss
    float maskedSigmoidEpsilon = 5e-7f; // epsilon for correct comparison with 0 in MaskedSigmoidCrossEntropy for stop token loss calculation, if 0 standard masking is used
    std::vector<float> lossMultipliers = std::vector<float>(3, 1.f);
    // training
    float teacherForcingRatio = 1.f; // value in [0., 1.], 0.=0%, 1.=100%, determines the % of times we force next decoder inputs (1 - always use input from ground-truth)

    bool trainableSpeakerEmbedding = true;

    void print(std::ostream&) const override;

    void ensureConsistency(bool checkInputsAndOutputs) const;
};

} // raul namespace
#endif
