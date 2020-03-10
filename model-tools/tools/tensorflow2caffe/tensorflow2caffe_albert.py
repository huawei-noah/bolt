#!/usr/local/bin/python
# -*- coding: utf-8 -*-

import numpy as np

from Caffe import caffe_net
from tensorflow2caffe_bert import Tensorflow2CaffeBert
from operators import Operators

class Tensorflow2CaffeALBert(Tensorflow2CaffeBert):
    def __init__(self,
            tensorflow_model_path, caffe_model_path_prefix, caffe_model_name,
            max_seq_length, embedding_dim, encoder_layers, num_heads,
            check=False, calc=False):
        Tensorflow2CaffeBert.__init__(self, tensorflow_model_path, caffe_model_path_prefix, caffe_model_name,
            max_seq_length, embedding_dim, encoder_layers, num_heads, check, calc)

    def extract_embeddings(self, word_input_name, position_input_name, token_input_name):
        # embedding block
        self.scopes[1] = "embeddings"
        # word embedding
        word_embedding_name_1 = "we_1"
        self.extract_embedding(word_input_name, 2, "word_embeddings", word_embedding_name_1)
        self.scopes[2] = "word_embeddings_2"
        weight_name = self.generate_name(self.scopes, 3)
        weight = self.get_tensor(weight_name)
        word_embedding_name_2 = "we_2"
        layer = caffe_net.LayerParameter(name=word_embedding_name_2, type='InnerProduct',
                                      bottom=[word_embedding_name_1], top=[word_embedding_name_2])
        num_output = len(weight[0])
        layer.inner_product_param(num_output, bias_term=False)
        layer.add_data(weight)
        self.caffe_model.add_layer(layer)
        self.data_dict[word_embedding_name_2] = Operators.fully_connect(self.data_dict[word_embedding_name_1],
                                                             weight, None,
                                                             word_embedding_name_2)

        # position embedding
        position_embedding_name = "pe"
        self.extract_embedding(position_input_name, 2, "position_embeddings", position_embedding_name)

        # token type embedding
        token_type_embedding_name = "tte"
        self.extract_embedding(token_input_name, 2, "token_type_embeddings", token_type_embedding_name)

        # eltwise
        sum_name = "embedding_sum"
        self.add_sum([word_embedding_name_2,position_embedding_name,token_type_embedding_name], sum_name)

        # layer norm
        layer_norm_name = "embedding_ln"
        self.extract_layer_norm(sum_name, layer_norm_name, 2)

        return layer_norm_name

    def extract_encoder(self, input_name, attention_mask_input_name):
        self.scopes[1] = "encoder"
        encoder_names = []
        for i in range(self.encoder_layers):
            self.scopes[2] = "layer_shared"
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
