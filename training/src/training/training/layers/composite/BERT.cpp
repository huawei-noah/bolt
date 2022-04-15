// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "BERT.h"

#include <algorithm>

#include <training/api/API.h>
#include <training/layers/activations/SoftMaxActivation.h>
#include <training/layers/activations/TanhActivation.h>
#include <training/layers/basic/DropoutLayer.h>
#include <training/layers/basic/ElementWiseSumLayer.h>
#include <training/layers/basic/MatMulLayer.h>
#include <training/layers/basic/ReshapeLayer.h>
#include <training/layers/basic/SlicerLayer.h>
#include <training/layers/basic/SplitterLayer.h>
#include <training/layers/basic/TensorLayer.h>
#include <training/layers/basic/TransposeLayer.h>
#include <training/layers/basic/trainable/Embedding.h>
#include <training/layers/basic/trainable/LayerNorm.h>
#include <training/layers/basic/trainable/LinearLayer.h>

#include <training/network/Layers.h>

using namespace raul;

namespace
{

const raul::dtype MASK_FILL_VALUE = -1e4_dt;

} // anonymous namespace

namespace
{
using namespace raul;
constexpr float BertLayerNormEps = 1e-12f;

// TODO: test with AttentionLayer (they differ in mask usage but should give same results)
// Then Multi-Headed Attention with finalTransform=false could be used
void BertSelfAttention(const Name& name, const Names& inputs, const Name& output, const BERTParams& params, NetworkParameters& networkParameters)
{
    using namespace std;

    auto xName = inputs[0];
    auto maskName = inputs[1];

    networkParameters.mWorkflow.add<SplitterLayer>(name + ".splitter", BasicParams{ { xName }, { name + ".q", name + ".k", name + ".v" } });
    networkParameters.mWorkflow.add<LinearLayer>(name + ".query", LinearParams{ { name + ".q" }, { name + ".mixed_query_layer" }, params.hiddenSize, true, params.frozen });
    networkParameters.mWorkflow.add<LinearLayer>(name + ".key", LinearParams{ { name + ".k" }, { name + ".mixed_key_layer" }, params.hiddenSize, true, params.frozen });
    networkParameters.mWorkflow.add<LinearLayer>(name + ".value", LinearParams{ { name + ".v" }, { name + ".mixed_value_layer" }, params.hiddenSize, true, params.frozen });

    networkParameters.mWorkflow.add<ReshapeLayer>(
        name + ".reshape_query", ViewParams{ name + ".mixed_query_layer", name + ".mixed_query_layer_v", -1, static_cast<int>(params.heads), static_cast<int>(params.hiddenSize / params.heads) });
    networkParameters.mWorkflow.add<ReshapeLayer>(
        name + ".reshape_key", ViewParams{ name + ".mixed_key_layer", name + ".mixed_key_layer_v", -1, static_cast<int>(params.heads), static_cast<int>(params.hiddenSize / params.heads) });
    networkParameters.mWorkflow.add<ReshapeLayer>(
        name + ".reshape_value", ViewParams{ name + ".mixed_value_layer", name + ".mixed_value_layer_v", -1, static_cast<int>(params.heads), static_cast<int>(params.hiddenSize / params.heads) });

    networkParameters.mWorkflow.add<TransposeLayer>(name + ".transp_query", TransposingParams{ name + ".mixed_query_layer_v", name + ".query_layer", Dimension::Depth, Dimension::Height });
    networkParameters.mWorkflow.add<TransposeLayer>(name + ".transp_key", TransposingParams{ name + ".mixed_key_layer_v", name + ".key_layer", Dimension::Depth, Dimension::Height });
    networkParameters.mWorkflow.add<TransposeLayer>(name + ".transp_value", TransposingParams{ name + ".mixed_value_layer_v", name + ".value_layer", Dimension::Depth, Dimension::Height });

    networkParameters.mWorkflow.add<TransposeLayer>(name + ".key_t", TransposingParams{ name + ".key_layer", name + ".key_t", Dimension::Width, Dimension::Height });

    networkParameters.mWorkflow.add<MatMulLayer>(name + ".mulQK",
                                                 MatMulParams{ { name + ".query_layer", name + ".key_t" }, name + ".scores", static_cast<float>(1.0 / sqrt(params.hiddenSize / params.heads)) });

    networkParameters.mWorkflow.add<ElementWiseSumLayer>(name + ".sum", ElementWiseLayerParams{ { name + ".scores", maskName }, { name + ".attention_scores" } });

    auto pAttnName = name + ".attention_probs";
    networkParameters.mWorkflow.add<SoftMaxActivation>(name + ".softmax", BasicParamsWithDim{ { name + ".attention_scores" }, { pAttnName }, Dimension::Width });

    if (params.attentionDropout > 0)
    {
        networkParameters.mWorkflow.add<DropoutLayer>(name + ".dropout", DropoutParams{ { pAttnName }, { name + ".attention_probs_do" }, params.attentionDropout });
        pAttnName = name + ".attention_probs_do";
    }

    networkParameters.mWorkflow.add<MatMulLayer>(name + ".mulV", MatMulParams{ { pAttnName, name + ".value_layer" }, name + ".context_layer" });
    networkParameters.mWorkflow.add<TransposeLayer>(name + ".transp_context", TransposingParams{ name + ".context_layer", name + ".context_layer_t", Dimension::Depth, Dimension::Height });
    networkParameters.mWorkflow.add<ReshapeLayer>(name + ".reshape_context", ViewParams{ name + ".context_layer_t", output, 1, -1, static_cast<int>(params.hiddenSize) });
}

void BertAttention(const Name& name, const Names& inputs, const Name& output, const BERTParams& params, NetworkParameters& networkParameters)
{
    using namespace raul;
    using namespace std;

    auto xName = inputs[0];
    auto maskName = inputs[1];

    networkParameters.mWorkflow.add<SplitterLayer>(name + ".splitter", BasicParams{ { xName }, { name + ".input_tensor_1", name + ".input_tensor_2" } });
    BertSelfAttention(name + ".self", { name + ".input_tensor_1", maskName }, name + ".self_output", params, networkParameters);
    // BertSelfOutput
    networkParameters.mWorkflow.add<LinearLayer>(name + ".output.dense", LinearParams{ { name + ".self_output" }, { name + ".output.dense.hidden_states_" }, params.hiddenSize, true, params.frozen });
    if (params.hiddenDropout > 0.f)
    {
        networkParameters.mWorkflow.add<DropoutLayer>(name + ".output.dropout",
                                                      DropoutParams{ { name + ".output.dense.hidden_states_" }, { name + ".output.dense.hidden_states_do" }, params.hiddenDropout });
        networkParameters.mWorkflow.add<ElementWiseSumLayer>(name + ".sum",
                                                             ElementWiseLayerParams{ { name + ".output.dense.hidden_states_do", name + ".input_tensor_2" }, { name + ".output.dense.hidden_states" } });
    }
    else
    {
        networkParameters.mWorkflow.add<ElementWiseSumLayer>(name + ".output.sum",
                                                             ElementWiseLayerParams{ { name + ".output.dense.hidden_states_", name + ".input_tensor_2" }, { name + ".output.dense.hidden_states" } });
    }

    networkParameters.mWorkflow.add<LayerNormLayer>(name + ".output.LayerNorm", LayerNormParams{ name + ".output.dense.hidden_states", output, BertLayerNormEps, false, true, params.frozen });
}

void BertLayer(const Name& name, const Names& inputs, const Name& output, const BERTParams& params, NetworkParameters& networkParameters)
{
    using namespace raul;

    BertAttention(name + ".attention", inputs, name + ".attention_output", params, networkParameters);
    networkParameters.mWorkflow.add<SplitterLayer>(name + ".splitter", BasicParams{ { name + ".attention_output" }, { name + ".attention_output_1", name + ".attention_output_2" } });

    networkParameters.mWorkflow.add<LinearLayer>(name + ".intermediate.dense",
                                                 LinearParams{ { name + ".attention_output_1" }, { name + ".intermediate.hidden_states" }, params.intermediateSize, true, params.frozen });

    if (params.activation == "gelu")
    {
        networkParameters.mWorkflow.add<raul::GeLUErf>(name + ".intermediate.intermediate_act_fn", raul::BasicParams{ { name + ".intermediate.hidden_states" }, { name + ".intermediate_output" } });
    }
    else if (params.activation == "relu")
    {
        networkParameters.mWorkflow.add<raul::ReLUActivation>(name + ".intermediate.intermediate_act_fn",
                                                              raul::BasicParams{ { name + ".intermediate.hidden_states" }, { name + ".intermediate_output" } });
    }
    else if (params.activation == "swish")
    {
        networkParameters.mWorkflow.add<raul::HSwishActivation>(name + ".intermediate.intermediate_act_fn",
                                                                raul::HSwishActivationParams{ { name + ".intermediate.hidden_states" }, { name + ".intermediate_output" } });
    }

    networkParameters.mWorkflow.add<LinearLayer>(name + ".output.dense", LinearParams{ { name + ".intermediate_output" }, { name + ".output.hidden_states_" }, params.hiddenSize });
    if (params.hiddenDropout > 0.f)
    {
        networkParameters.mWorkflow.add<DropoutLayer>(name + ".output.dropout", DropoutParams{ { name + ".output.hidden_states_" }, { name + ".output.hidden_states_do" }, params.hiddenDropout });
        networkParameters.mWorkflow.add<ElementWiseSumLayer>(name + ".sum",
                                                             ElementWiseLayerParams{ { name + ".output.hidden_states_do", name + ".attention_output_2" }, { name + ".output.hidden_states" } });
    }
    else
    {
        networkParameters.mWorkflow.add<ElementWiseSumLayer>(name + ".output.sum",
                                                             ElementWiseLayerParams{ { name + ".output.hidden_states_", name + ".attention_output_2" }, { name + ".output.hidden_states" } });
    }

    networkParameters.mWorkflow.add<LayerNormLayer>(name + ".output.LayerNorm", LayerNormParams{ name + ".output.hidden_states", output, BertLayerNormEps, false, true, params.frozen });
}

void BertEncoder(const Name& name, const Names& inputs, const Names& outputs, const BERTParams& params, NetworkParameters& networkParameters)
{
    using namespace raul;
    auto xName = inputs[0];
    auto maskName = inputs[1];

    if (outputs.size() != 1 && outputs.size() != params.numHiddenLayers)
    {
        throw std::runtime_error("BertEncoder: wrong outputs number");
    }

    for (size_t i = 0; i < params.numHiddenLayers; ++i)
    {
        std::string suffix = "." + std::to_string(i);
        if (i == params.numHiddenLayers - 1)
        {
            BertLayer(name + ".layer" + suffix, { xName, maskName }, outputs.back(), params, networkParameters);
        }
        else
        {
            std::string out = name + ".out" + suffix;
            BertLayer(name + ".layer" + suffix, { xName, maskName }, out, params, networkParameters);

            if (outputs.size() == 1)
            {
                xName = out;
            }
            else
            {
                networkParameters.mWorkflow.add<SplitterLayer>(name + ".splitter" + suffix, BasicParams{ { out }, { out + "_1", outputs[i] } });
                xName = out + "_1";
            }
        }
    }
}
}

namespace raul
{

BERTModel::BERTModel(const Name& name, const BERTParams& params, NetworkParameters& networkParameters)
{
    using namespace std;

    auto prefix = "BERTModel[" + name + "::ctor]: ";

    // input_ids, [token_type_ids], [attention_mask]
    if (params.getInputs().size() > 3 || params.getInputs().empty())
    {
        throw runtime_error(prefix + "wrong number of input names (must be 1, 2 or 3)");
    }

    auto input_ids_name = params.getInputs()[0];
    auto token_type_ids_name = params.getInputs().size() > 1 ? params.getInputs()[1] : Name{};
    mAttentionMask = params.getInputs().size() > 2 ? params.getInputs()[2] : Name{};

    if (params.getOutputs().size() != 2 && params.getOutputs().size() != (1 + params.numHiddenLayers))
    {
        throw runtime_error(prefix + "wrong number of output names (must be 1, 2 or 1+)");
    }

    if (params.heads == 0)
    {
        throw runtime_error(prefix + "heads count must be > 0");
    }

    shape inputShape = shape{
        1u, networkParameters.mWorkflow.getDepth(params.getInputs()[0]), networkParameters.mWorkflow.getHeight(params.getInputs()[0]), networkParameters.mWorkflow.getWidth(params.getInputs()[0])
    };

    const Name mask = name + ".extended_attention_mask";

    networkParameters.mWorkflow.add<TensorLayer>(mask, raul::TensorParams{ { mask }, raul::WShape{ BS(), inputShape[1], inputShape[2], inputShape[3] }, 0.0_dt, DEC_FORW_WRIT });

    if (token_type_ids_name.empty())
    {
        token_type_ids_name = name + ".token_type_ids";

        networkParameters.mWorkflow.add<TensorLayer>(token_type_ids_name,
                                                     raul::TensorParams{ { token_type_ids_name }, raul::WShape{ BS(), inputShape[1], inputShape[2], inputShape[3] }, 0.0_dt, DEC_FORW_WRIT });
    }
    // Embeddings
    mPositionIdsName = name + ".embedding.position_ids";

    networkParameters.mWorkflow.add<TensorLayer>(mPositionIdsName,
                                                 raul::TensorParams{ { mPositionIdsName }, raul::WShape{ BS(), inputShape[1], inputShape[2], inputShape[3] }, 0.0_dt, DEC_FORW_WRIT });

    class Filler : public raul::BasicLayer
    {
      public:
        Filler(const Name& name, const BasicParams& params, NetworkParameters& networkParameters)
            : BasicLayer(name, "Filler", params, networkParameters, { false, false })
        {
            auto prefix = "Filler[" + mName + "::ctor]: ";

            mNetworkParams.mWorkflow.copyDeclaration(mName, mInputs[0], Workflow::Usage::Forward, Workflow::Mode::Write);

            mNetworkParams.mWorkflow.copyDeclaration(mName, mInputs[1], Workflow::Usage::Forward, Workflow::Mode::Write);

            mNetworkParams.mWorkflow.copyDeclaration(mName, mInputs[2], Workflow::Usage::Forward, Workflow::Mode::Read);
        }

        void forwardComputeImpl(raul::NetworkMode) override
        {
            auto& position_ids = mNetworkParams.mMemoryManager[mInputs[0]];
            auto position_ids_2d = position_ids.reshape(yato::dims(position_ids.getBatchSize(), position_ids.getWidth()));
            for (size_t n = 0; n < position_ids.getBatchSize(); ++n)
            {
                std::iota(position_ids_2d[n].begin(), position_ids_2d[n].end(), 0.0_dt);
            }

            auto& extended_attention_mask = mNetworkParams.mMemoryManager[mInputs[1]];
            if (mInputs[2].empty() || mNetworkParams.mMemoryManager[mInputs[2]].empty())
            {
                extended_attention_mask = 0.0_dt;
            }
            else
            {
                const auto& attention_mask = mNetworkParams.mMemoryManager[mInputs[2]];
#if defined(_OPENMP)
#pragma omp parallel for
#endif
                for (size_t q = 0; q < attention_mask.size(); ++q)
                    extended_attention_mask[q] = (1.0_dt - attention_mask[q]) * MASK_FILL_VALUE;
            }
        }

        void backwardComputeImpl() override {}
    };

    networkParameters.mWorkflow.add<Filler>(name / "filler", raul::BasicParams{ { mPositionIdsName, mask, mAttentionMask }, {} });

    // begin of BertEmbeddings
    networkParameters.mWorkflow.add<Embedding>(name + ".embeddings.word_embeddings",
                                               EmbeddingParams{ input_ids_name, name + ".word_embeddings", params.frozen, params.vocabSize, params.hiddenSize, 0 });
    networkParameters.mWorkflow.add<Embedding>(name + ".embeddings.position_embeddings",
                                               EmbeddingParams{ mPositionIdsName, name + ".position_embeddings", params.frozen, params.maxPositionEmbeddings, params.hiddenSize });
    networkParameters.mWorkflow.add<Embedding>(name + ".embeddings.token_type_embeddings",
                                               EmbeddingParams{ token_type_ids_name, name + ".token_type_embeddings", params.frozen, params.typeVocabSize, params.hiddenSize });

    networkParameters.mWorkflow.add<ElementWiseSumLayer>(
        name + ".embeddings.sum", ElementWiseLayerParams{ { name + ".word_embeddings", name + ".position_embeddings", name + ".token_type_embeddings" }, { name + ".embeddings_sum" } });

    if (params.hiddenDropout > 0)
    {
        networkParameters.mWorkflow.add<LayerNormLayer>(name + ".embeddings.LayerNorm",
                                                        LayerNormParams{ name + ".embeddings_sum", name + ".embeddings.norm", BertLayerNormEps, false, true, params.frozen });
        networkParameters.mWorkflow.add<DropoutLayer>(name + ".embeddings.dropout", DropoutParams{ { name + ".embeddings.norm" }, { name + ".embedding_output" }, params.hiddenDropout });
    }
    else
    {
        networkParameters.mWorkflow.add<LayerNormLayer>(name + ".embeddings.LayerNorm",
                                                        LayerNormParams{ name + ".embeddings_sum", name + ".embedding_output", BertLayerNormEps, false, true, params.frozen });
    }

    // end of BertEmbeddings

    // begin of Encoder
    Names encoded_layers(params.getOutputs().begin(), params.getOutputs().end() - 1);
    BertEncoder(name + ".encoder", { name + ".embedding_output", mask }, encoded_layers, params, networkParameters);
    // end of Encoder
    std::string sequence_output_name = encoded_layers.back();
    // begin of Pooler
    networkParameters.mWorkflow.add<SlicerLayer>(name + ".pooler.first_token_extractor", SlicingParams{ sequence_output_name, { name + ".pooler.first_token_tensor" }, "height", { 1 } });
    networkParameters.mWorkflow.add<LinearLayer>(name + ".pooler.dense",
                                                 LinearParams{ { name + ".pooler.first_token_tensor" }, { name + ".pooler.pooled_output" }, params.hiddenSize, true, params.frozen });
    networkParameters.mWorkflow.add<TanhActivation>(name + ".pooler.activation", BasicParams{ { name + ".pooler.pooled_output" }, { params.getOutputs().back() } });
    // end of Pooler
}

} // namespace raul
