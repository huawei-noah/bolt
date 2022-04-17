// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "Transformer.h"

#include <algorithm>

#include <training/base/common/MemoryManager.h>
#include <training/base/layers/activations/LogSoftMaxActivation.h>
#include <training/base/layers/activations/ReLUActivation.h>
#include <training/base/layers/basic/DropoutLayer.h>
#include <training/base/layers/basic/ElementWiseSumLayer.h>
#include <training/base/layers/basic/PositionalEncoding.h>
#include <training/base/layers/basic/SplitterLayer.h>
#include <training/base/layers/basic/TransposeLayer.h>
#include <training/base/layers/basic/trainable/Embedding.h>
#include <training/base/layers/basic/trainable/LayerNorm.h>
#include <training/base/layers/basic/trainable/LinearLayer.h>
#include <training/base/layers/composite/MultiHeadAttention.h>

namespace
{

using namespace raul;

const float TransformerLayerNormEps = 1e-6f;

// sublayer must use name.x as first input and name.y as output
void SublayerConnection(const Name& name, const Name& input, const Name& output, float dropout, bool frozen, std::function<void()> sublayer_creator, NetworkParameters& networkParameters)
{
    networkParameters.mWorkflow.add<SplitterLayer>(name / "splitter", BasicParams{ { input }, { name / "x1", name / "x2" } });
    networkParameters.mWorkflow.add<LayerNormLayer>(name / "norm", LayerNormParams{ name / "x2", name / "x", TransformerLayerNormEps, false, true, frozen });
    sublayer_creator();
    std::string out = name / "y";
    if (dropout > 0.0_dt)
    {
        networkParameters.mWorkflow.add<DropoutLayer>(name / "dropout", DropoutParams{ { out }, { name / "do" }, dropout });
        out = name / "do";
    }
    networkParameters.mWorkflow.add<ElementWiseSumLayer>(name / "sum", ElementWiseLayerParams{ { name / "x1", out }, { output } });
}

void PositionwiseFeedForward(const Name& name, const Name& input, const Name& output, const TransformerParams& params, NetworkParameters& networkParameters)
{
    std::string out = name / "relu";
    networkParameters.mWorkflow.add<LinearLayer>(name / "w_1", LinearParams{ { input }, { name / "w_1" }, params.feedForwardSize, true, params.frozen });
    networkParameters.mWorkflow.add<ReLUActivation>(name / "relu", BasicParams{ { name / "w_1" }, { out } });
    if (params.dropout > 0.0_dt)
    {
        networkParameters.mWorkflow.add<DropoutLayer>(name / "dropout", DropoutParams{ { out }, { name / "do" }, params.dropout });
        out = name / "do";
    }
    networkParameters.mWorkflow.add<LinearLayer>(name / "w_2", LinearParams{ { out }, { output }, params.modelSize, true, params.frozen });
}

void EncoderLayer(const Name& name, const Names& inputs, const Name& output, const TransformerParams& params, NetworkParameters& networkParameters)
{
    auto xName = inputs[0];
    auto maskName = inputs[1];

    auto selfAttention = [&name, &maskName, &params, &networkParameters]() {
        MultiHeadAttentionLayer(
            name / "self_attn", MultiHeadAttentionParams{ { name / "sublayer[0]" / "x", maskName }, name / "sublayer[0]" / "y", params.heads, params.dropout, true, params.frozen }, networkParameters);
    };

    auto ff = [&name, &params, &networkParameters]() { PositionwiseFeedForward(name / "feed_forward", name / "sublayer[1]" / "x", name / "sublayer[1]" / "y", params, networkParameters); };

    SublayerConnection(name / "sublayer[0]", xName, name / "sublayer[0]" / "z", params.dropout, params.frozen, selfAttention, networkParameters);
    SublayerConnection(name / "sublayer[1]", name / "sublayer[0]" / "z", output, params.dropout, params.frozen, ff, networkParameters);
}

void Encoder(const Name& name, const Names& inputs, const Name& output, const TransformerParams& params, NetworkParameters& networkParameters)
{
    auto xName = inputs[0];
    auto maskName = inputs[1];

    for (size_t i = 0; i < params.encoderDecoderLength; ++i)
    {
        std::string suffix = "[" + std::to_string(i) + "]";
        std::string out = name / "out" + suffix;
        EncoderLayer(name / "layers" + suffix, { xName, maskName }, out, params, networkParameters);
        xName = out;
    }

    networkParameters.mWorkflow.add<LayerNormLayer>(name / "norm", LayerNormParams{ xName, output, TransformerLayerNormEps, false, true, params.frozen });
}

void DecoderLayer(const Name& name, const Names& inputs, const Name& output, const TransformerParams& params, NetworkParameters& networkParameters)
{
    auto xName = inputs[0];
    auto memName = inputs[1];
    auto srcMaskName = inputs[2];
    auto tgtMaskName = inputs[3];

    auto selfAttention = [&name, &tgtMaskName, &params, &networkParameters]() {
        MultiHeadAttentionLayer(name / "self_attn",
                                MultiHeadAttentionParams{ { name / "sublayer[0]" / "x", tgtMaskName }, name / "sublayer[0]" / "y", params.heads, params.dropout, true, params.frozen },
                                networkParameters);
    };
    auto srcAttention = [&name, &memName, &srcMaskName, &params, &networkParameters]() {
        MultiHeadAttentionLayer(
            name / "src_attn",
            MultiHeadAttentionParams{ { name / "sublayer[1]" / "x", memName, memName, srcMaskName }, name / "sublayer[1]" / "y", params.heads, params.dropout, true, params.frozen },
            networkParameters);
    };

    auto ff = [&name, &params, &networkParameters]() { PositionwiseFeedForward(name / "feed_forward", name / "sublayer[2]" / "x", name / "sublayer[2]" / "y", params, networkParameters); };

    SublayerConnection(name / "sublayer[0]", xName, name / "sublayer[0]" / "z", params.dropout, params.frozen, selfAttention, networkParameters);
    SublayerConnection(name / "sublayer[1]", name / "sublayer[0]" / "z", name / "sublayer[1]" / "z", params.dropout, params.frozen, srcAttention, networkParameters);
    SublayerConnection(name / "sublayer[2]", name / "sublayer[1]" / "z", output, params.dropout, params.frozen, ff, networkParameters);
}

void Decoder(const Name& name, const Names& inputs, const Name& output, const TransformerParams& params, NetworkParameters& networkParameters)
{
    auto xName = inputs[0];
    auto memName = inputs[1];
    auto srcMaskName = inputs[2];
    auto tgtMaskName = inputs[3];

    Names memNames(params.encoderDecoderLength);
    for (size_t i = 0; i < params.encoderDecoderLength; ++i)
    {
        memNames[i] = name / "m" / std::to_string(i);
    }

    networkParameters.mWorkflow.add<SplitterLayer>(name / "mem_splitter", BasicParams{ { memName }, { memNames } });

    for (size_t i = 0; i < params.encoderDecoderLength; ++i)
    {
        std::string suffix = "[" + std::to_string(i) + "]";
        std::string out = name / "out" + suffix;
        DecoderLayer(name / "layers" + suffix, { xName, memNames[i], srcMaskName, tgtMaskName }, out, params, networkParameters);
        xName = out;
    }

    networkParameters.mWorkflow.add<LayerNormLayer>(name / "norm", LayerNormParams{ xName, output, TransformerLayerNormEps, false, true, params.frozen });
}
}

namespace raul
{

void CreateGenerator(const Name& name, const BasicParams& params, size_t vocabularySize, NetworkParameters& networkParameters)
{
    auto input = params.getInputs()[0];
    auto output = params.getOutputs()[0];

    networkParameters.mWorkflow.add<LinearLayer>(name / "proj", LinearParams{ { input }, { name / "proj" }, vocabularySize });
    networkParameters.mWorkflow.add<LogSoftMaxActivation>(name / "log_softmax", BasicParamsWithDim{ { name / "proj" }, { output }, Dimension::Width });
}

TransformerModel::TransformerModel(const Name& name, const TransformerParams& params, NetworkParameters& networkParameters)
{
    auto prefix = "TransformerModel[" + name + "::ctor]: ";
    // src, tgt, src_mask, tgt_mask
    if (params.getInputs().size() != 4)
    {
        THROW("TransformerModel", name, "wrong number of input names (must be 4)");
    }
    if (params.getOutputs().size() != 1)
    {
        THROW("TransformerModel", name, "wrong number of output names (must be 1)");
    }
    if (params.heads == 0)
    {
        THROW("TransformerModel", name, "heads count must be > 0");
    }

    auto srcName = params.getInputs()[0];
    auto tgtName = params.getInputs()[1];
    auto srcMaskName = params.getInputs()[2];
    auto tgtMaskName = params.getInputs()[3];

    networkParameters.mWorkflow.add<Embedding>(name / "src_embed" / "0" / "lut", EmbeddingParams{ srcName, name / "src_emb", params.frozen, params.srcVocabSize, params.modelSize });
    networkParameters.mWorkflow.add<Embedding>(name / "tgt_embed" / "0" / "lut", EmbeddingParams{ tgtName, name / "tgt_emb", params.frozen, params.tgtVocabSize, params.modelSize });

    if (params.dropout > 0._dt)
    {
        networkParameters.mWorkflow.add<PositionalEncoding>(name / "src_embed" / "1" / "pos_enc", PositionalEncodingParams{ name / "src_emb", name / "src_embed0", params.modelSize });
        networkParameters.mWorkflow.add<PositionalEncoding>(name / "tgt_embed" / "1" / "pos_enc", PositionalEncodingParams{ name / "tgt_emb", name / "tgt_embed0", params.modelSize });
        networkParameters.mWorkflow.add<DropoutLayer>(name / "src_embed" / "2" / "dropout", DropoutParams{ { name / "src_embed0" }, { name / "src_embed" }, params.dropout });
        networkParameters.mWorkflow.add<DropoutLayer>(name / "tgt_embed" / "2" / "dropout", DropoutParams{ { name / "tgt_embed0" }, { name / "tgt_embed" }, params.dropout });
    }
    else
    {
        networkParameters.mWorkflow.add<PositionalEncoding>(name / "src_posenc", PositionalEncodingParams{ name / "src_emb", name / "src_embed", params.modelSize });
        networkParameters.mWorkflow.add<PositionalEncoding>(name / "tgt_posenc", PositionalEncodingParams{ name / "tgt_emb", name / "tgt_embed", params.modelSize });
    }

    Encoder(name / "encoder", { name / "src_embed", srcMaskName }, name / "enc", params, networkParameters);
    Decoder(name / "decoder", { name / "tgt_embed", name / "enc", srcMaskName, tgtMaskName }, params.getOutputs()[0], params, networkParameters);
}

} // namespace raul