// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef BERT_PARAMS_H
#define BERT_PARAMS_H

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
 * @param paramHiddenDropout The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler
 * @param paramAttentionDropout The dropout ratio for the attention probabilities
 */
struct BERTParams : public TrainableParams
{
    BERTParams() = delete;
    BERTParams(const Names& inputs,
               const Names& outputs,
               uint32_t paramVocabSize,
               uint32_t paramTypeVocabSize = 2,
               uint32_t paramNumHiddenLayers = 12,
               uint32_t paramHiddenSize = 768,
               uint32_t paramIntermediateSize = 3072,
               uint32_t paramHeads = 12,
               uint32_t paramMaxPositionEmbeddings = 512,
               std::string paramHiddenActivation = "gelu",
               float paramHiddenDropout = 0.1f,
               float paramAttentionDropout = 0.1f,
               bool frozen = false);

    uint32_t vocabSize;
    uint32_t typeVocabSize;
    uint32_t numHiddenLayers;
    uint32_t hiddenSize;
    uint32_t intermediateSize;
    uint32_t heads;
    uint32_t maxPositionEmbeddings;
    std::string activation;
    float hiddenDropout;
    float attentionDropout;

    void print(std::ostream&) const override;
};

} // raul namespace
#endif
