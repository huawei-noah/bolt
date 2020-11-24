#!/usr/local/bin/python
# -*- coding: utf-8 -*-

import numpy as np

import sys
sys.path.append("../")
sys.path.append("../../")
from Caffe import caffe_net
from operators import Operators
from tensorflow2caffe_bert import Tensorflow2CaffeBert

class Tensorflow2CaffeALBert(Tensorflow2CaffeBert):
    def __init__(self,
            tensorflow_model_path, caffe_model_path_prefix, caffe_model_name,
            max_seq_length, embedding_dim, encoder_layers, num_heads,
            check=False, calc=False):
        Tensorflow2CaffeBert.__init__(self, tensorflow_model_path, caffe_model_path_prefix, caffe_model_name,
            max_seq_length, embedding_dim, encoder_layers, num_heads, check, calc)

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

        output_name = self.extract_embeddings(word_input_name, position_input_name, token_input_name, ["", "word_embeddings_2", "bias"])
        output_names = self.extract_encoder(output_name, attention_mask_name)
        output_name = self.extract_pooler(output_names[-1])

        self.save_caffe_model()
