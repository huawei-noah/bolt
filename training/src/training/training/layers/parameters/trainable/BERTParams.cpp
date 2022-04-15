// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "BERTParams.h"

namespace raul
{

BERTParams::BERTParams(const Names& inputs,
                       const Names& outputs,
                       uint32_t paramVocabSize,
                       uint32_t paramTypeVocabSize,
                       uint32_t paramNumHiddenLayers,
                       uint32_t paramHiddenSize,
                       uint32_t paramIntermediateSize,
                       uint32_t paramHeads,
                       uint32_t paramMaxPositionEmbeddings,
                       std::string paramHiddenActivation,
                       float paramHiddenDropout,
                       float paramAttentionDropout,
                       bool frozen)
    : TrainableParams(inputs, outputs, frozen)
    , vocabSize(paramVocabSize)
    , typeVocabSize(paramTypeVocabSize)
    , numHiddenLayers(paramNumHiddenLayers)
    , hiddenSize(paramHiddenSize)
    , intermediateSize(paramIntermediateSize)
    , heads(paramHeads)
    , maxPositionEmbeddings(paramMaxPositionEmbeddings)
    , activation(paramHiddenActivation)
    , hiddenDropout(paramHiddenDropout)
    , attentionDropout(paramAttentionDropout)
{
}

void BERTParams::print(std::ostream& stream) const
{
    BasicParams::print(stream);
    stream << "vocab_size: " << vocabSize << ", type_vocab_size: " << typeVocabSize << ", ";
    stream << "hidden_size: " << hiddenSize << ", num_hidden_layers: " << numHiddenLayers << ", ";
    stream << "hidden_act: " << activation << ", num_attention_heads: " << heads << ", ";
    stream << "intermediate_size: " << intermediateSize << ", hidden_dropout_prob: " << hiddenDropout << ", ";
    stream << "attention_probs_dropout_prob: " << attentionDropout << ", max_position_embeddings: " << maxPositionEmbeddings << ", ";
}

} // namespace raul
