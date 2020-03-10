#!/usr/local/bin/python
# -*- coding: utf-8 -*-
from Caffe import caffe_net
from tensorflow2caffe_bert import Tensorflow2CaffeBert
from operators import Operators

class Tensorflow2CaffeTinyBert(Tensorflow2CaffeBert):
    def __init__(self,
            tensorflow_model_path, caffe_model_path_prefix, caffe_model_name,
            max_seq_length, embedding_dim, encoder_layers, num_heads,
            check=False, calc=False):
        Tensorflow2CaffeBert.__init__(self, tensorflow_model_path, caffe_model_path_prefix, caffe_model_name,
            max_seq_length, embedding_dim, encoder_layers, num_heads, check, calc)

    def extract_dense_prefix(self, input_name, dense_name, weight_name_prefix):
        kernel_name = weight_name_prefix + "weight"
        bias_name = weight_name_prefix + "bias"
        kernel = self.get_tensor(kernel_name)
        bias = self.get_tensor(bias_name)
        layer = caffe_net.LayerParameter(name=dense_name, type='InnerProduct',
                                      bottom=[input_name], top=[dense_name])
        num_output = len(kernel)
        layer.inner_product_param(num_output, bias_term=bias is not None)
        if bias is not None:
            if len(bias) != num_output:
                print("[ERROR] extract extract_dense_prefix")
            layer.add_data(kernel, bias)
        else:
            layer.add_data(kernel)
        self.caffe_model.add_layer(layer)
        self.data_dict[dense_name] = Operators.fully_connect(self.data_dict[input_name],
                                                             kernel.transpose((1, 0)), bias,
                                                             dense_name)

    def extract_intent_classifier(self, input_name):
        dense_name = "intent_classifier"
        weight_name_prefix = "intent_classifier_"
        self.extract_dense_prefix(input_name, dense_name, weight_name_prefix)

        softmax_name = "intent_softmax"
        self.add_softmax(dense_name, softmax_name)

        return softmax_name

    def extract_slot_classifier(self, input_name):
        dense_name = "slot_classifier"
        weight_name_prefix = "slot_classifier_"
        self.extract_dense_prefix(input_name, dense_name, weight_name_prefix)

        softmax_name = "slot_softmax"
        self.add_softmax(dense_name, softmax_name)

        return softmax_name

    def extract_mrpc_classifier(self, input_name):
        dense_name = "mrpc_classifier"
        weight_name_prefix = "classifier_"
        self.extract_dense_prefix(input_name, dense_name, weight_name_prefix)

        softmax_name = "mrpc_softmax"
        self.add_softmax(dense_name, softmax_name)

        return softmax_name

    def generate_intent_slot_task(self, input=None):
        word_input_name = "tinybert_words"
        position_input_name = "tinybert_positions"
        token_input_name = "tinybert_token_type"
        mask_input_name = "tinybert_mask"
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
        intent = self.extract_intent_classifier(output_name)
        slots = self.extract_slot_classifier(output_names[-1])

        self.save()

    def generate_mrpc_task(self, input=None):
        word_input_name = "tinybert_words"
        position_input_name = "tinybert_positions"
        token_input_name = "tinybert_token_type"
        mask_input_name = "tinybert_mask"
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
        mrpc = self.extract_mrpc_classifier(output_name)

        self.save()
