#!/usr/local/bin/python
# -*- coding: utf-8 -*-

import math
from tensorflow2caffe import Tensorflow2Caffe

class Tensorflow2CaffeBert(Tensorflow2Caffe):
    def __init__(self,
            tensorflow_model_path, caffe_model_path_prefix, caffe_model_name,
            max_seq_length, embedding_dim, encoder_layers, num_heads,
            check=False, calc=False):
        Tensorflow2Caffe.__init__(self, tensorflow_model_path,
            caffe_model_path_prefix, caffe_model_name, check, calc)
        self.scopes[0] = "bert"
        self.max_seq_length = max_seq_length
        self.embedding_dim = embedding_dim
        self.encoder_layers = encoder_layers
        self.num_heads = num_heads

    def extract_embeddings(self, word_input_name, position_input_name, token_input_name):
        # embedding block
        self.scopes[1] = "embeddings"
        # word embedding
        word_embedding_name = "we"
        self.extract_embedding(word_input_name, 2, "word_embeddings", word_embedding_name)

        # position embedding
        position_embedding_name = "pe"
        self.extract_embedding(position_input_name, 2, "position_embeddings", position_embedding_name)

        # token type embedding
        token_type_embedding_name = "tte"
        self.extract_embedding(token_input_name, 2, "token_type_embeddings", token_type_embedding_name)

        # eltwise
        sum_name = "embedding_sum"
        self.add_sum([word_embedding_name,position_embedding_name,token_type_embedding_name], sum_name)

        # layer norm
        layer_norm_name = "embedding_ln"
        self.extract_layer_norm(sum_name, layer_norm_name, 2)

        return layer_norm_name

    def extract_encoder_attention(self, input_name, attention_mask_input_name, output_name_prefix, scope_id):
        # attention
        self.scopes[scope_id] = "attention"
        # attention-self
        self.scopes[scope_id+1] = "self"
        # attention-self-query
        self.scopes[scope_id+2] = "query"
        query_name = output_name_prefix + "att_self_query"
        self.extract_dense(input_name, query_name, scope_id+2, "query")
        # attention-self-key
        self.scopes[scope_id+2] = "key"
        key_name = output_name_prefix + "att_self_key"
        self.extract_dense(input_name, key_name, scope_id+2, "key")
        # attention-self-value
        self.scopes[scope_id+2] = "value"
        value_name = output_name_prefix + "att_self_value"
        self.extract_dense(input_name, value_name, scope_id+2, "value")

        # reshape
        query_reshape_name = query_name + "_r"
        key_reshape_name   = key_name + "_r"
        value_reshape_name = value_name + "_r"
        size_per_head = self.embedding_dim // self.num_heads
        self.add_reshape(query_name, query_reshape_name, [self.batch, -1, self.num_heads, size_per_head])
        self.add_reshape(key_name,   key_reshape_name,   [self.batch, -1, self.num_heads, size_per_head])
        self.add_reshape(value_name, value_reshape_name, [self.batch, -1, self.num_heads, size_per_head])

        # transpose
        query_transpose_name = query_name + "_t"
        key_transpose_name   = key_name + "_t"
        value_transpose_name = value_name + "_t"
        self.add_transpose(query_reshape_name, query_transpose_name, [0, 2, 1, 3])
        self.add_transpose(key_reshape_name,   key_transpose_name,   [0, 2, 3, 1])
        self.add_transpose(value_reshape_name, value_transpose_name, [0, 2, 1, 3])

        # query * key
        query_key_name = output_name_prefix + "att_self_qk"
        self.add_matmul(query_transpose_name, key_transpose_name, query_key_name)
        query_key_scale_name = output_name_prefix + "att_self_qks"
        self.add_multiply(query_key_name, query_key_scale_name, 1.0/math.sqrt(size_per_head))

        # query * key + mask
        if (attention_mask_input_name is None):
            sum_name = query_key_scale_name
        else:
            sum_name = output_name_prefix + "att_self_score"
            self.add_sum([query_key_scale_name, attention_mask_input_name], sum_name)

        # softmax
        prob_name = output_name_prefix + "att_self_prob"
        self.add_softmax(sum_name, prob_name)

        # prob * value
        context_name = output_name_prefix + "att_self_cont"
        self.add_matmul(prob_name, value_transpose_name, context_name)

        # transpose value
        context_transpose_name = output_name_prefix + "att_self_cont_t"
        self.add_transpose(context_name, context_transpose_name, [0, 2, 1, 3])

        # reshape 
        context_reshape_name = output_name_prefix + "att_self_cont_r"
        self.add_reshape(context_transpose_name, context_reshape_name, [self.batch, -1, self.num_heads*size_per_head])

        # attention-output
        output_name_prefix_new = output_name_prefix + "att_out_"
        attention_output_name = self.extract_output(context_reshape_name, input_name, output_name_prefix_new, scope_id+1)

        return attention_output_name

    def extract_output(self, input_name, eltwise_input_name, output_name_prefix, scope_id):
        self.scopes[scope_id] = "output"

        # output-dense
        dense_name = output_name_prefix + "den"
        self.extract_dense(input_name, dense_name, scope_id+1)

        # output-sum
        sum_name = output_name_prefix + "sum"
        self.add_sum([dense_name, eltwise_input_name], sum_name)

        # output-layer_norm
        layer_norm_name = output_name_prefix + "ln"
        self.extract_layer_norm(sum_name, layer_norm_name, scope_id+1)

        return layer_norm_name

    def extract_encoder_intermediate(self, input_name, output_name_prefix, scope_id):
        self.scopes[scope_id] = "intermediate"

        dense_name = output_name_prefix + "inter_den"
        self.extract_dense(input_name, dense_name, 4)
        
        gelu_name =  output_name_prefix + "inter_gelu"
        self.add_gelu(dense_name, gelu_name)

        return gelu_name

    def extract_encoder(self, input_name, attention_mask_input_name):
        self.scopes[1] = "encoder"
        encoder_names = []
        for i in range(self.encoder_layers):
            self.scopes[2] = "layer_" + str(i)
            scope_next_id = 3
            output_name_prefix = "layer" + str(i) + "_out_"

            # attention
            attention_name = self.extract_encoder_attention(input_name, attention_mask_input_name, output_name_prefix, scope_next_id)

            # intermediate
            intermediate_name = self.extract_encoder_intermediate(attention_name, output_name_prefix, scope_next_id)

            # output
            output_name = self.extract_output(intermediate_name, attention_name, output_name_prefix, scope_next_id)

            input_name = output_name

            encoder_names.append(input_name)

        return encoder_names

    def extract_pooler(self, input_name):
        self.scopes[1] = "pooler"
        
        slice_name = "pooler_slice"
        self.add_slice(input_name, [slice_name, "other"], 1, [1])

        dense_name = "pooler_den"
        self.extract_dense(slice_name, dense_name, 2)

        tanh_name = "pooler_tanh"
        self.add_tanh(dense_name, tanh_name)

        return tanh_name

    def generate(self, input=None):
        word_input_name = "bert_words"
        position_input_name = "bert_positions"
        token_input_name = "bert_token_type"
        mask_input_name = "bert_input_mask"
        word_input_shape = [self.batch, self.max_seq_length]
        position_input_shape = [self.batch, self.max_seq_length]
        token_input_shape = [self.batch, self.max_seq_length]
        mask_input_shape = [self.batch, self.max_seq_length]

        self.add_input(word_input_name, word_input_shape)
        self.add_input(position_input_name, position_input_shape)
        self.add_input(token_input_name, token_input_shape)
        #self.add_input(mask_input_name, mask_input_shape)
        self.set_input(input)

        attention_mask_name = None #"attention"
        #self.add_attention(mask_input_name, self.num_heads, self.max_seq_length, self.max_seq_length, attention_mask_name);

        output_name = self.extract_embeddings(word_input_name, position_input_name, token_input_name)
        output_names = self.extract_encoder(output_name, attention_mask_name)
        output_name = self.extract_pooler(output_names[-1])

        self.save()
