#!/usr/local/bin/python
# -*- coding: utf-8 -*-
import math
import numpy as np
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.training import checkpoint_utils

from Caffe import caffe_net
from operators import Operators


class Tensorflow2CaffeBert:
    def __init__(self, tensorflow_model_path, max_seq_length, encoder_layers, num_attentions, caffe_model_path_prefix, calc=False):
        self.scopes = ["" for i in range(10)]
        self.scopes[0] = "bert"
        self.tensorflow_model = pywrap_tensorflow.NewCheckpointReader(tensorflow_model_path)
        self.caffe_model = caffe_net.CaffeModel('')
        self.caffe_model.net.name = "bert"
        self.caffe_model_path_prefix = caffe_model_path_prefix
        self.max_seq_length = max_seq_length
        self.encoder_layers = encoder_layers
        self.num_attentions = num_attentions
        self.tensor_map = checkpoint_utils.list_variables(tensorflow_model_path)
        self.batch = 1
        self.check = True;
        self.name_dict = {}
        self.calculate = True;
        self.data_dict = {}
        Operators.set_calculate(calc)


    def get_tensor(self, name):
        if (self.check):
            if (name in self.name_dict):
                print("[ERROR] layer %s duplicate" % (name))
                exit(-1)
            else:
                self.name_dict[name] = 1
        return self.tensorflow_model.get_tensor(name)

    def weight_map(self):
        print(self.tensor_map)


    def generate_name(self, scopes, num):
        result = ""
        for i in range(num):
            result = result + scopes[i]
            if(i != num-1):
                result = result + "/"
    
        return result


    def add_input(self, output_name, input_shape):
        layer = caffe_net.LayerParameter(name=output_name, type='Input',
                                      top=[output_name])
        layer.input_param(input_shape)
        self.caffe_model.add_layer(layer)
        self.data_dict[output_name] = Operators.get_input(output_name, input_shape)


    def extract_layer_norm(self, input_name, layer_norm_name, scope_id):
        self.scopes[scope_id] = "LayerNorm"
        self.scopes[scope_id + 1] = "gamma"
        gamma_name = self.generate_name(self.scopes, scope_id + 2)
        gamma = self.get_tensor(gamma_name)
        self.scopes[scope_id + 1] = "beta"
        beta_name = self.generate_name(self.scopes, scope_id + 2)
        beta = self.get_tensor(beta_name)
        layer = caffe_net.LayerParameter(name=layer_norm_name, type='LayerNorm',
                                      bottom=[input_name], top=[layer_norm_name])
        layer.add_data(gamma, beta)
        self.caffe_model.add_layer(layer)
        self.data_dict[layer_norm_name] = Operators.layer_norm(self.data_dict[input_name],
                                                               gamma, beta,
                                                               layer_norm_name)


    def extract_dense(self, input_name, dense_name, scope_id, scope_name = "dense"):
        self.scopes[scope_id] = scope_name
        self.scopes[scope_id + 1] = "kernel"
        kernel_name = self.generate_name(self.scopes, scope_id + 2)
        kernel = self.get_tensor(kernel_name)
        self.scopes[scope_id + 1] = "bias"
        bias_name = self.generate_name(self.scopes, scope_id + 2)
        bias = self.get_tensor(bias_name)
        layer = caffe_net.LayerParameter(name=dense_name, type='InnerProduct',
                                      bottom=[input_name], top=[dense_name])
        num_output = len(kernel[0])
        kernel = kernel.transpose((1, 0))
        layer.inner_product_param(num_output, bias_term=bias is not None)
        if bias is not None:
            if len(bias) != num_output:
                print("[ERROR] extract extract_dense")
            layer.add_data(kernel, bias)
        else:
            layer.add_data(kernel)
        self.caffe_model.add_layer(layer)
        self.data_dict[dense_name] = Operators.fully_connect(self.data_dict[input_name],
                                                             kernel.transpose((1, 0)), bias,
                                                             dense_name)


    def reshape(self, input_name, output_name, shape):
        layer = caffe_net.LayerParameter(name=output_name, type='Reshape',
                                      bottom=[input_name], top=[output_name])
        layer.reshape_param(shape)
        self.caffe_model.add_layer(layer)
        self.data_dict[output_name] = Operators.reshape(self.data_dict[input_name], shape, output_name)


    def transpose(self, input_name, output_name, dim):
        layer = caffe_net.LayerParameter(name=output_name, type='Transpose',
                                      bottom=[input_name], top=[output_name])
        layer.transpose_param(dim)
        self.caffe_model.add_layer(layer)
        self.data_dict[output_name] = Operators.transpose(self.data_dict[input_name], dim, output_name)


    def add_matmul(self, input_a_name, input_b_name, output_name):
        layer = caffe_net.LayerParameter(name=output_name, type='MatMul',
                                      bottom=[input_a_name, input_b_name], top=[output_name])
        self.caffe_model.add_layer(layer)
        self.data_dict[output_name] = Operators.matmul(self.data_dict[input_a_name], self.data_dict[input_b_name], output_name)


    def add_multiply(self, input_name, output_name, scale):
        layer = caffe_net.LayerParameter(name=output_name, type='Multiply',
                                      bottom=[input_name], top=[output_name])
        layer.multiply_param(scale)
        self.caffe_model.add_layer(layer)
        self.data_dict[output_name] = Operators.multiply(self.data_dict[input_name], scale, output_name)


    def add_slice(self, input_name, output_names, axis, slice_point):
        layer = caffe_net.LayerParameter(name=output_names[0], type='Slice',
                                      bottom=[input_name], top=output_names)
        layer.slice_param(axis, slice_point)
        self.caffe_model.add_layer(layer)
        result = Operators.slice(self.data_dict[input_name], axis, slice_point, output_names)
        for i in range(len(output_names)):
            if (result is not None):
                self.data_dict[output_names[i]] = result[i]
            else:
                self.data_dict[output_names[i]] = None


    def add_attention(self, input_name, output_name):
        layer = caffe_net.LayerParameter(name=output_name, type='Attention',
                                      bottom=[input_name], top=[output_name])
        layer.attention_param(self.num_attentions)
        self.caffe_model.add_layer(layer)
        self.data_dict[output_name] = Operators.attention(self.data_dict[input_name], self.num_attentions, output_name)


    def add_sum(self, input_names, output_name):
        layer = caffe_net.LayerParameter(name=output_name, type='Eltwise',
                                      bottom=input_names,
                                      top=[output_name])
        layer.layerParameter.eltwise_param.operation = 1 #SUM
        self.caffe_model.add_layer(layer)
        data = []
        for name in input_names:
            data.append(self.data_dict[name])
        self.data_dict[output_name] = Operators.sum(data, output_name)
        

    def add_softmax(self, input_name, output_name):
        layer = caffe_net.LayerParameter(name=output_name, type='Softmax',
                                      bottom=[input_name],
                                      top=[output_name])
        self.caffe_model.add_layer(layer)
        self.data_dict[output_name] = Operators.softmax(self.data_dict[input_name], output_name)


    def add_gelu(self, input_name, output_name):
        layer = caffe_net.LayerParameter(name=output_name, type='Gelu',
                                      bottom=[input_name],
                                      top=[output_name])
        self.caffe_model.add_layer(layer)
        self.data_dict[output_name] = Operators.gelu(self.data_dict[input_name], output_name)


    def add_tanh(self, input_name, output_name):
        layer = caffe_net.LayerParameter(name=output_name, type='TanH',
                                      bottom=[input_name],
                                      top=[output_name])
        self.caffe_model.add_layer(layer)
        self.data_dict[output_name] = Operators.tanh(self.data_dict[input_name], output_name)


    def extract_embedding_kernel(self, input_name, scope_id, tensorflow_weight_name, output_name):
        self.scopes[scope_id] = tensorflow_weight_name
        weight_name = self.generate_name(self.scopes, scope_id+1)
        weight = self.get_tensor(weight_name)
        layer = caffe_net.LayerParameter(name=output_name, type='Embed',
                                      bottom=[input_name], top=[output_name])
        layer.add_data(weight)
        self.embedding_dim = len(weight[0])
        layer.embed_param(len(weight), self.embedding_dim)
        self.caffe_model.add_layer(layer)
        self.data_dict[output_name] = Operators.embedding(self.data_dict[input_name], weight, output_name)


    def extract_embedding(self, word_input_name, position_input_name, token_input_name):
        # embedding block
        self.scopes[1] = "embeddings"
        # word embedding
        word_embedding_name = "we"
        self.extract_embedding_kernel(word_input_name, 2, "word_embeddings", word_embedding_name)

        # position embedding
        position_embedding_name = "pe"
        self.extract_embedding_kernel(position_input_name, 2, "position_embeddings", position_embedding_name)

        # token type embedding
        token_type_embedding_name = "tte"
        self.extract_embedding_kernel(token_input_name, 2, "token_type_embeddings", token_type_embedding_name)

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
        size_per_head = self.embedding_dim / self.num_attentions
        self.reshape(query_name, query_reshape_name, [self.batch, self.max_seq_length, self.num_attentions, size_per_head])
        self.reshape(key_name,   key_reshape_name,   [self.batch, self.max_seq_length, self.num_attentions, size_per_head])
        self.reshape(value_name, value_reshape_name, [self.batch, self.max_seq_length, self.num_attentions, size_per_head])

        # transpose
        query_transpose_name = query_name + "_t"
        key_transpose_name   = key_name + "_t"
        value_transpose_name = value_name + "_t"
        self.transpose(query_reshape_name, query_transpose_name, [0, 2, 1, 3])
        self.transpose(key_reshape_name,   key_transpose_name,   [0, 2, 3, 1])
        self.transpose(value_reshape_name, value_transpose_name, [0, 2, 1, 3])

        # query * key
        query_key_name = output_name_prefix + "att_self_qk"
        self.add_matmul(query_transpose_name, key_transpose_name, query_key_name)
        query_key_scale_name = output_name_prefix + "att_self_qks"
        self.add_multiply(query_key_name, query_key_scale_name, 1.0/math.sqrt(size_per_head))

        # query * key + mask
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
        self.transpose(context_name, context_transpose_name, [0, 2, 1, 3])

        # reshape 
        context_reshape_name = output_name_prefix + "att_self_cont_r"
        self.reshape(context_transpose_name, context_reshape_name, [self.batch, self.max_seq_length, self.num_attentions*size_per_head])

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
    

    def save(self):
        self.caffe_model.save_prototxt(self.caffe_model_path_prefix + ".prototxt")
        self.caffe_model.save(self.caffe_model_path_prefix + ".caffemodel")

    
    def generate(self):
        word_input_name = "word"
        position_input_name = "position"
        token_input_name = "token_type"
        mask_input_name = "input_mask"
        word_input_shape = [self.batch, self.max_seq_length]
        position_input_shape = [self.batch, self.max_seq_length]
        token_input_shape = [self.batch, self.max_seq_length]
        mask_input_shape = [self.batch, self.max_seq_length]

        self.add_input(word_input_name, word_input_shape)
        self.add_input(position_input_name, position_input_shape)
        self.add_input(token_input_name, token_input_shape)
        self.add_input(mask_input_name, mask_input_shape)

        attention_mask_name = "attention"
        self.add_attention(mask_input_name, attention_mask_name);

        output_name = self.extract_embedding(word_input_name, position_input_name, token_input_name)
        output_names = self.extract_encoder(output_name, attention_mask_name)
        output_name = self.extract_pooler(output_names[-1])

        self.save()
