#!/usr/local/bin/python
# -*- coding: utf-8 -*-
import numpy as np
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.training import checkpoint_utils

from Caffe import caffe_net
from operators import Operators


class Tensorflow2Caffe:
    def __init__(self, tensorflow_model_path, caffe_model_path_prefix, caffe_model_name, check=False, calc=False):
        self.scopes = ["" for i in range(100)]
        self.tensorflow_model = pywrap_tensorflow.NewCheckpointReader(tensorflow_model_path)
        self.caffe_model = caffe_net.CaffeModel('')
        self.caffe_model.net.name = caffe_model_name
        self.caffe_model_path_prefix = caffe_model_path_prefix
        self.tensor_map = checkpoint_utils.list_variables(tensorflow_model_path)
        self.batch = 1
        self.check = check;
        self.name_dict = {}
        self.calculate = True;
        self.data_dict = {}
        Operators.set_calculate(calc)

    def set_add_layer(self, add_layer_set):
        self.caffe_model.set_add_layer(add_layer_set)

    def get_tensor(self, name):
        if (self.check):
            if (name in self.name_dict):
                print("[ERROR] layer %s weight duplicate" % (name))
                exit(-1)
            else:
                self.name_dict[name] = 1
        if (self.tensorflow_model.has_tensor(name)):
            return self.tensorflow_model.get_tensor(name)
        else:
            print("[ERROR] not found weight %s" % (name))
            return None

    def weight_map(self):
        print(self.tensor_map)
        return self.tensor_map

    def generate_name(self, scopes, num):
        result = ""
        for i in range(num):
            result = result + scopes[i]
            if(i != num-1):
                result = result + "/"
    
        return result

    def save(self):
        self.caffe_model.save_prototxt(self.caffe_model_path_prefix + ".prototxt")
        self.caffe_model.save(self.caffe_model_path_prefix + ".caffemodel")

    def get_data(self, name):
        return self.data_dict[name]

    def print_data(self, name):
        Operators.print_data(self.data_dict[name], name)

    def set_input(self, data):
        for key, value in data.items():
            self.data_dict[key] = value
            self.print_data(key)

    def add_input(self, output_name, input_shape):
        layer = caffe_net.LayerParameter(name=output_name, type='Input',
                                      top=[output_name])
        layer.input_param(input_shape)
        self.caffe_model.add_layer(layer)
        return output_name

    def extract_layer_norm(self, input_name, layer_norm_name, scope_id, layer_names=None):
        if (layer_names is None):
            self.scopes[scope_id] = "LayerNorm"
            self.scopes[scope_id + 1] = "gamma"
            gamma_name = self.generate_name(self.scopes, scope_id + 2)
            self.scopes[scope_id + 1] = "beta"
            beta_name = self.generate_name(self.scopes, scope_id + 2)
        else:
            self.scopes[scope_id] = layer_names[0]
            self.scopes[scope_id + 1] = layer_names[1]
            gamma_name = self.generate_name(self.scopes, scope_id + 2)
            self.scopes[scope_id + 1] = layer_names[2]
            beta_name = self.generate_name(self.scopes, scope_id + 2)
        gamma = self.get_tensor(gamma_name)
        beta = self.get_tensor(beta_name)
        layer = caffe_net.LayerParameter(name=layer_norm_name, type='LayerNorm',
                                      bottom=[input_name], top=[layer_norm_name])
        layer.add_data(gamma, beta)
        self.caffe_model.add_layer(layer)
        self.data_dict[layer_norm_name] = Operators.layer_norm(self.data_dict[input_name],
                                                               gamma, beta,
                                                               layer_norm_name)
        return layer_norm_name

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
                print("[ERROR] extract_dense failed")
                exit(0)
            layer.add_data(kernel, bias)
        else:
            layer.add_data(kernel)
        self.caffe_model.add_layer(layer)
        self.data_dict[dense_name] = Operators.fully_connect(self.data_dict[input_name],
                                                             kernel.transpose((1, 0)), bias,
                                                             dense_name)
        return dense_name

    def extract_denses(self, input_name, dense_names, output_nums, scope_id, scope_name = "dense"):
        self.scopes[scope_id] = scope_name
        self.scopes[scope_id + 1] = "kernel"
        kernel_name = self.generate_name(self.scopes, scope_id + 2)
        kernels = self.get_tensor(kernel_name)
        self.scopes[scope_id + 1] = "bias"
        bias_name = self.generate_name(self.scopes, scope_id + 2)
        biases = self.get_tensor(bias_name)

        last_sum = 0
        for index in range(len(output_nums)):
            kernel = kernels[:, last_sum:last_sum+output_nums[index]]
            bias = None
            if biases is not None:
                bias = biases[last_sum:last_sum+output_nums[index]]
            layer = caffe_net.LayerParameter(name=dense_names[index], type='InnerProduct',
                                          bottom=[input_name], top=[dense_names[index]])
            num_output = len(kernel[0])
            kernel = kernel.transpose((1, 0))
            layer.inner_product_param(num_output, bias_term=bias is not None)
            if bias is not None:
                if len(bias) != num_output:
                    print("[ERROR] extract_denses failed")
                    exit(0)
                layer.add_data(kernel, bias)
            else:
                layer.add_data(kernel)
            self.caffe_model.add_layer(layer)
            self.data_dict[dense_names[index]] = Operators.fully_connect(self.data_dict[input_name],
                                                                 kernel.transpose((1, 0)), bias,
                                                                 dense_names[index])
            last_sum = last_sum + output_nums[index]
        if (last_sum != len(kernels[0])):
            print("[ERROR] extract_denses failed")
            exit(0)
        return dense_names

    def add_reshape(self, input_name, output_name, shape):
        layer = caffe_net.LayerParameter(name=output_name, type='Reshape',
                                      bottom=[input_name], top=[output_name])
        layer.reshape_param(shape)
        self.caffe_model.add_layer(layer)
        self.data_dict[output_name] = Operators.reshape(self.data_dict[input_name], shape, output_name)
        return output_name
    
    def add_squeeze(self, input_name, output_name, axis):
        layer = caffe_net.LayerParameter(name=output_name, type='Squeeze',
                                      bottom=[input_name], top=[output_name])
        layer.squeeze_param(axis)
        self.caffe_model.add_layer(layer)
        self.data_dict[output_name] = Operators.squeeze(self.data_dict[input_name], axis, output_name)
        return output_name

    def add_transpose(self, input_name, output_name, dim):
        layer = caffe_net.LayerParameter(name=output_name, type='Transpose',
                                      bottom=[input_name], top=[output_name])
        layer.transpose_param(dim)
        self.caffe_model.add_layer(layer)
        self.data_dict[output_name] = Operators.transpose(self.data_dict[input_name], dim, output_name)
        return output_name

    def add_matmul(self, input_a_name, input_b_name, output_name, transpose_a=False, transpose_b=False):
        layer = caffe_net.LayerParameter(name=output_name, type='MatMul',
                                      bottom=[input_a_name, input_b_name], top=[output_name])
        layer.matmul_param(transpose_a, transpose_b)
        self.caffe_model.add_layer(layer)
        self.data_dict[output_name] = Operators.matmul(self.data_dict[input_a_name], transpose_a, self.data_dict[input_b_name], transpose_b, output_name)
        return output_name

    def add_multiply(self, input_name, output_name, scale=1, bias=0):
        layer = caffe_net.LayerParameter(name=output_name, type='Multiply',
                                      bottom=[input_name], top=[output_name])
        layer.multiply_param(scale, bias)
        self.caffe_model.add_layer(layer)
        self.data_dict[output_name] = Operators.multiply(self.data_dict[input_name], scale, bias, output_name)
        return output_name

    def add_prod(self, input_names, output_name):
        layer = caffe_net.LayerParameter(name=output_name, type='Eltwise',
                                      bottom=input_names,
                                      top=[output_name])
        layer.eltwise_param(0) #Prod
        self.caffe_model.add_layer(layer)
        self.data_dict[output_name] = self.data_dict[input_names[0]]
        for i in range(1, len(input_names)):
            self.data_dict[output_name] = Operators.matmultiply(self.data_dict[output_name], self.data_dict[input_names[i]], output_name)
        return output_name

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
        return output_names

    def add_attention(self, input_name, attention_num, from_seq_length, to_seq_length, output_name):
        layer = caffe_net.LayerParameter(name=output_name, type='Attention',
                                      bottom=[input_name], top=[output_name])
        layer.attention_param(attention_num, from_seq_length, to_seq_length)
        self.caffe_model.add_layer(layer)
        self.data_dict[output_name] = Operators.attention(self.data_dict[input_name],
                                          attention_num, from_seq_length, to_seq_length,
                                          output_name)
        return output_name

    def add_sum(self, input_names, output_name):
        layer = caffe_net.LayerParameter(name=output_name, type='Eltwise',
                                      bottom=input_names,
                                      top=[output_name])
        layer.eltwise_param(1) #SUM
        self.caffe_model.add_layer(layer)
        data = []
        for name in input_names:
            data.append(self.data_dict[name])
        self.data_dict[output_name] = Operators.sum(data, output_name)
        return output_name

    def add_softmax(self, input_name, output_name):
        layer = caffe_net.LayerParameter(name=output_name, type='Softmax',
                                      bottom=[input_name],
                                      top=[output_name])
        self.caffe_model.add_layer(layer)
        self.data_dict[output_name] = Operators.softmax(self.data_dict[input_name], output_name)
        return output_name

    def add_gelu(self, input_name, output_name):
        layer = caffe_net.LayerParameter(name=output_name, type='Gelu',
                                      bottom=[input_name],
                                      top=[output_name])
        self.caffe_model.add_layer(layer)
        self.data_dict[output_name] = Operators.gelu(self.data_dict[input_name], output_name)
        return output_name

    def add_tanh(self, input_name, output_name):
        layer = caffe_net.LayerParameter(name=output_name, type='TanH',
                                      bottom=[input_name],
                                      top=[output_name])
        self.caffe_model.add_layer(layer)
        self.data_dict[output_name] = Operators.tanh(self.data_dict[input_name], output_name)
        return output_name

    def add_relu(self, input_name, output_name):
        layer = caffe_net.LayerParameter(name=output_name, type='ReLU',
                                      bottom=[input_name],
                                      top=[output_name])
        self.caffe_model.add_layer(layer)
        self.data_dict[output_name] = Operators.relu(self.data_dict[input_name], output_name)
        return output_name

    def add_sigmoid(self, input_name, output_name):
        layer = caffe_net.LayerParameter(name=output_name, type='Sigmoid',
                                      bottom=[input_name],
                                      top=[output_name])
        self.caffe_model.add_layer(layer)
        self.data_dict[output_name] = Operators.sigmoid(self.data_dict[input_name], output_name)
        return output_name

    def add_swish(self, input_name, output_name, beta=1.0):
        layer = caffe_net.LayerParameter(name=output_name, type='Swish',
                                      bottom=[input_name],
                                      top=[output_name])
        self.caffe_model.add_layer(layer)
        self.data_dict[output_name] = Operators.swish(self.data_dict[input_name], beta, output_name)
        return output_name

    def add_weight(self, output_name, scope_id=None, weight_name=None, weight=None, transpose=None, data_type="FLOAT32"):
        if scope_id is not None:
            weight_name = self.generate_name(self.scopes, scope_id)
        if weight_name is not None:
            weight = self.get_tensor(weight_name)
        if weight is None:
            print("[ERROR] can not add null weight layer")
            exit(0)
        layer = caffe_net.LayerParameter(name=output_name+"_weight", type='SharedWeight',
                                         top=[output_name])
        if (transpose is not None):
            weight = weight.transpose(transpose)
        layer.weight_param(weight.shape, data_type)
        layer.add_data(weight)
        self.caffe_model.add_layer(layer)
        self.data_dict[output_name] = Operators.weight(weight, output_name)
        return output_name

    def add_embedding(self, input_name, weight_name, output_name, transpose=False):
        layer = caffe_net.LayerParameter(name=output_name, type='Embed',
                                      bottom=[input_name,weight_name], top=[output_name])
        weight = self.data_dict[weight_name]
        if transpose:
            input_dim = len(weight[0])
            embedding_dim = len(weight)
        else:
            input_dim = len(weight)
            embedding_dim = len(weight[0])
        layer.embed_param(input_dim, embedding_dim, transpose)
        self.caffe_model.add_layer(layer)
        self.data_dict[output_name] = Operators.embedding(self.data_dict[input_name], weight, transpose, output_name)
        return output_name

    def extract_embedding(self, input_name, scope_id, tensorflow_weight_name, output_name):
        self.scopes[scope_id] = tensorflow_weight_name
        weight_name = self.generate_name(self.scopes, scope_id+1)
        weight = self.get_tensor(weight_name)
        layer = caffe_net.LayerParameter(name=output_name, type='Embed',
                                      bottom=[input_name], top=[output_name])
        layer.add_data(weight)
        embedding_dim = len(weight[0])
        layer.embed_param(len(weight), embedding_dim, False)
        self.caffe_model.add_layer(layer)
        self.data_dict[output_name] = Operators.embedding(self.data_dict[input_name], weight, False, output_name)
        return output_name

    def add_axis_mean(self, input_name, axis, output_name):
        layer = caffe_net.LayerParameter(name=output_name, type='AxisMean',
                                      bottom=[input_name], top=[output_name])
        layer.axis_mean_param(axis)
        self.caffe_model.add_layer(layer)
        self.data_dict[output_name] = Operators.axis_mean(self.data_dict[input_name], axis, output_name)
        return output_name

    def add_expand_dims(self, input_name, axis, output_name):
        layer = caffe_net.LayerParameter(name=output_name, type='Unsqueeze',
                                      bottom=[input_name], top=[output_name])
        layer.unsqueeze_param(axis)
        self.caffe_model.add_layer(layer)
        self.data_dict[output_name] = Operators.expand_dims(self.data_dict[input_name], axis, output_name)
        return output_name

    def add_argmax(self, input_name, axis, output_name):
        layer = caffe_net.LayerParameter(name=output_name, type='ArgMax',
                                      bottom=[input_name], top=[output_name])
        layer.argmax_param(axis)
        self.caffe_model.add_layer(layer)
        self.data_dict[output_name] = Operators.argmax(self.data_dict[input_name], axis, output_name)
        return output_name

    def extract_lstm(self, input_name, state_name, output_name, scope_id, steps=-1, scope_name = "basic_lstm_cell"):
        self.scopes[scope_id] = scope_name
        self.scopes[scope_id + 1] = "kernel"
        kernel_name = self.generate_name(self.scopes, scope_id + 2)
        kernel = self.get_tensor(kernel_name)
        self.scopes[scope_id + 1] = "bias"
        bias_name = self.generate_name(self.scopes, scope_id + 2)
        bias = self.get_tensor(bias_name)
        
        layer = caffe_net.LayerParameter(name=output_name, type='LSTM',
                                      bottom=[input_name, state_name], top=[output_name])
        
        num_output_4 = len(kernel[0])
        if (bias is not None):
            if (len(bias) != num_output_4):
                print("[ERROR] extract_lstm failed")
                exit(0)
        num_output = num_output_4 // 4
        layer.lstm_param(num_output, steps)
        layer.add_data(kernel.transpose([1, 0]), bias)
        self.caffe_model.add_layer(layer)
        self.data_dict[output_name], self.data_dict[state_name] = Operators.lstm(self.data_dict[input_name],
                                                                             self.data_dict[state_name],
                                                                             kernel,
                                                                             bias,
                                                                             output_name,
                                                                             state_name)
        return output_name

    def add_check(self, left_name, right_name, condition, status_name):
        layer = caffe_net.LayerParameter(name=status_name, type='Check',
                                      bottom=[left_name, right_name], top=[status_name])
        layer.check_param(condition)
        self.caffe_model.add_layer(layer)
        self.data_dict[status_name] = Operators.check(self.data_dict[left_name],
                                          self.data_dict[right_name],
                                          condition,
                                          status_name)
        return status_name
    
    def add_copy(self, src_name, src_batch_stride, src_stride, src_offset,
            dst_name, dst_batch_stride, dst_stride, dst_offset,
            length,
            op_name,
            src_index_name=None, dst_index_name=None):
        src_index = None
        dst_index = None
        if (src_index_name is None):
            layer = caffe_net.LayerParameter(name=op_name, type='Copy',
                                      bottom=[src_name, dst_name], top=[op_name])
        else:
            layer = caffe_net.LayerParameter(name=op_name, type='Copy',
                                      bottom=[src_name, dst_name, src_index_name, dst_index_name], top=[op_name])
            src_index = self.data_dict[src_index_name]
            dst_index = self.data_dict[dst_index_name]
        layer.copy_param(src_batch_stride, src_stride, src_offset, dst_batch_stride, dst_stride, dst_offset, length)
        self.caffe_model.add_layer(layer)

        self.data_dict[dst_name] = Operators.copy(self.data_dict[src_name],
                                       src_batch_stride, src_stride, src_offset,
                                       self.data_dict[dst_name],
                                       dst_batch_stride, dst_stride, dst_offset,
                                       length,
                                       dst_name,
                                       src_index, dst_index)
        return dst_name

    def add_repeat(self, loops, repeat_start_name, op_name=None, status_name=None):
        layer = caffe_net.LayerParameter(name=op_name, type='Repeat',
                                      bottom=[repeat_start_name, status_name], top=[op_name])
        layer.repeat_param(loops)
        self.caffe_model.add_layer(layer)
        return repeat_start_name

    def add_memory(self, memory_name, memory_shape, data_type):
        layer = caffe_net.LayerParameter(name=memory_name+"_mem", type='PreAllocatedMemory',
                                         top=[memory_name])
        layer.memory_param(memory_shape, data_type)
        self.caffe_model.add_layer(layer)
        self.data_dict[memory_name] = Operators.zeros(memory_shape, memory_name)
        return memory_name
