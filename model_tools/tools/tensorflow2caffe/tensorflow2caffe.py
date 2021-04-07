#!/usr/local/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import math

from Caffe import caffe_net
from operators import Operators


class Tensorflow2Caffe:
    def __init__(self, tensorflow_model_path, caffe_model_path_prefix, caffe_model_name,
            check=False,
            calculate=False,
            quantization=False):
        self.scopes = ["" for i in range(100)]
        self.caffe_model = caffe_net.CaffeModel('')
        self.caffe_model.net.name = caffe_model_name
        self.caffe_model_path_prefix = caffe_model_path_prefix
        self.weight_map = self.load_tensorflow_model(tensorflow_model_path)
        self.batch = 1
        self.check = check;
        self.name_dict = {}
        self.calculate = calculate;
        self.data_dict = {}
        self.inputs = []
        self.weight_size_map = {}
        Operators.set_calculate(calculate)
        self.quantization = quantization
        self.quantization_max = {}

    def load_tensorflow_model_from_ckpt(self, tensorflow_model_path):
        from tensorflow.python.training import checkpoint_utils
        from tensorflow.python import pywrap_tensorflow
    
        self.tensor_map = checkpoint_utils.list_variables(tensorflow_model_path)
        reader = pywrap_tensorflow.NewCheckpointReader(tensorflow_model_path)
        variable_shape_map = reader.get_variable_to_shape_map()
        weight_map = {}
        for key in variable_shape_map:
            weight_map[key] = reader.get_tensor(key)
        return weight_map

    def load_tensorflow_model_from_pb(self, tensorflow_model_path):
        import tensorflow as tf
        from tensorflow.python.platform import gfile
        from tensorflow.python.framework import tensor_util

        with tf.Session() as sess:
            with gfile.FastGFile(tensorflow_model_path,'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                sess.graph.as_default()
                tf.import_graph_def(graph_def, name='')
                graph_nodes=[n for n in graph_def.node]
        weight_map = {}
        for node in graph_nodes:
            #print("[INFO] tensorflow pb node %s" % (node.name))
            if node.op == 'Const':
                weight_map[node.name] = tensor_util.MakeNdarray(node.attr['value'].tensor)
        return weight_map

    def load_keras_model_from_h5(self, keras_model_path):
        def isGroup(item):
            ret = False
            if (isinstance(item, h5py.Group)):
                ret = True
            return ret
        def isDataset(item):
            ret = False
            if (isinstance(item, h5py.Dataset)):
                ret = True
            return ret
        def getDataSetFromGroup(datasets, prefix, item):
            if (isGroup(item)):
                for key in item:
                    value = item[key]
                    if (prefix is None):
                        name = key
                    else:
                        name = prefix + "/" + key
                    getDataSetFromGroup(datasets, name, value)
            else:
                name_suffix = prefix.split(":")[-1]
                if (name_suffix == "0"):
                    name = prefix[:-2]
                else:
                    name = prefix
                datasets[name] = np.array(item)

        import h5py
        model = h5py.File(keras_model_path)
        weight_map = {}
        for layerName in model:
            if (layerName == "model_weights"):
                layer = model[layerName]
                getDataSetFromGroup(weight_map, None, layer)
        model.close()
        return weight_map

    def load_tensorflow_model(self, tensorflow_model_path):
        model_path = tensorflow_model_path.strip()
        model_path_suffix = model_path.split(".")
        if (model_path_suffix[-1].startswith("pb")):
            return self.load_tensorflow_model_from_pb(tensorflow_model_path)
        if (model_path_suffix[-1].startswith("ckpt")):
            return self.load_tensorflow_model_from_ckpt(tensorflow_model_path)
        if (model_path_suffix[-1].startswith("h5") or model_path_suffix[-1].startswith("hdf5")):
            return self.load_keras_model_from_h5(tensorflow_model_path)
        print("[ERROR] unrecognized file type %s, currently only support *.ckpt, *.pb, *.h5, *.hdf5" % (tensorflow_model_path))
        exit(1)

    def set_add_layer(self, add_layer_set):
        self.caffe_model.set_add_layer(add_layer_set)

    def rename_weight(self, input_name, output_name):
        self.weight_map[output_name] = self.weight_map[input_name]
        del self.weight_map[input_name]

    def concat_weight(self, input_names, output_name, axis):
        weights = []
        for name in input_names:
            weights.append(self.weight_map[name])
            del self.weight_map[name]
        self.weight_map[output_name] = np.concatenate(weights, axis=axis)

    def get_weight(self, name):
        if (self.check):
            if (name in self.name_dict):
                print("[ERROR] layer %s weight duplicate" % (name))
                exit(-1)
            else:
                self.name_dict[name] = 1
        if (name in self.weight_map):
            self.weight_size_map[name] = self.weight_map[name].size
            return self.weight_map[name]
        else:
            print("[WARNING] not found weight %s" % (name))
            return None

    def get_weights(self, scope_id, names):
        if (names is None or len(names) == 1):
            print("[ERROR] get_weights names array's length must > 1")
            exit(1)
        self.scopes[scope_id] = names[0]
        self.scopes[scope_id + 1] = names[1]
        weight0_name = self.generate_name(self.scopes, scope_id + 2)
        weight0 = self.get_weight(weight0_name)
        if (len(names) == 2):
            return weight0
        self.scopes[scope_id + 1] = names[2]
        weight1_name = self.generate_name(self.scopes, scope_id + 2)
        weight1 = self.get_weight(weight1_name)
        if (len(names) == 3):
            return weight0, weight1
        print("[ERROR] unsupported get_weights names array's length %d" (len(names)))
        exit(1)

    def get_tensor(self, name):
        if (name in self.data_dict.keys()):
            return self.data_dict[name]
        else:
            print("[ERROR] not found tensor %s" % (name))
            exit(1)

    def get_tensor_shape(self, name):
        if (not self.calculate):
            print("[ERROR] must in calculate mode to use get_tensor_shape %s" % (name))
            exit(1)
        return self.get_tensor(name).shape

    def print_weight_map(self):
        keys = sorted(list(self.weight_map.keys()))
        for key in keys:
            print("[INFO] weight %s shape %s" % (key, self.weight_map[key].shape))
            print(self.weight_map[key].reshape([-1])[0])

    def print_weight_statistics(self):
        #for key in self.weight_map.keys():
        #    if (key not in self.weight_size_map.keys()):
        #        print("[WARNING] not load weight %s" % (key))
        size = 0
        for _, value in self.weight_size_map.items():
            size = size + value
        print("[INFO] model parameter size %fM float32" % (size/1000.0/1000.0))
        
    def generate_name(self, scopes, length):
        result = ""
        for i in range(length):
            if (scopes[i] == ""):
                continue
            result = result + scopes[i]
            if(i != length - 1):
                result = result + "/"
        return result

    def save_caffe_model(self):
        self.print_weight_statistics()
        print("[INFO] save caffe model to %s.*" % (self.caffe_model_path_prefix))
        self.caffe_model.save_prototxt(self.caffe_model_path_prefix + ".prototxt")
        self.caffe_model.save(self.caffe_model_path_prefix + ".caffemodel")
        if (self.quantization):
            print("[INFO] save int8 quantization max value to %s.*" % (self.caffe_model_path_prefix+"_quant.txt"))
            quantizationMaxFile = open(self.caffe_model_path_prefix + "_quant.txt", "w")
            for key, value in self.quantization_max.items():
                if (value is not None):
                    quantizationMaxFile.write("%s %f\n" % (key, value))
            quantizationMaxFile.close()

    def print_tensor(self, name):
        Operators.print_data(self.get_tensor(name), name)

    def set_input(self, data):
        if (data is None):
            if (self.calculate):
                print("[ERROR] if you want to use model converter verify feature, please add not None input data of generate(),\n"
                      "        else please turn off calc operation in initialize function")
                exit(1)
            return;
        for key, value in data.items():
            self.data_dict[key] = value
            self.inputs.append(key)
            if (self.calculate):
                self.print_tensor(key)

    def save_input(self):
        file_path = "./input_shape.txt"
        print("[INFO] save input shape to %s" % (file_path))
        shape_file = open(file_path, "w")
        for key in self.inputs:
            value = self.get_tensor(key)
            content = key + "\n"
            shape_file.write(content)
            content = str(len(value.shape)) + "\n"
            shape_file.write(content)
            content = ""
            for i in value.shape:
                content = content + str(i) + " "
            content = content + "\n"
            shape_file.write(content)

            x = value.reshape([value.size])
            fmt = "%.18f"
            if (x.size > 0):
                if (isinstance(x[0], np.int64)):
                    fmt = "%d"
            print("[INFO] save input data %s to %s.txt" % (key, key))
            np.savetxt(key+".txt", x, fmt=fmt,delimiter="\n")
        shape_file.close()

    def preprocess_nchwc8_nchw_input(self, input_name, axis):
        shape = self.data_dict[input_name].shape
        inv_transpose_dims = [i for i in range(len(shape))]
        if (len(shape) == 3):
            if (axis == 1):
                input_data = self.data_dict[input_name].reshape([shape[0], shape[1], shape[2], 1])
            elif (axis == -1):
                transpose_dims = [0, 2, 1]
                inv_transpose_dims = [0, 2, 1]
                input_data = self.data_dict[input_name]
                input_data = input_data.transpose(transpose_dims)
                shape = input_data.shape
                input_data = input_data.reshape([shape[0], shape[1], shape[2], 1])
            else:
                print("[ERROR] unsupported preprocess axis %s" % (axis))
                exit(1)
        elif (len(shape) == 4):
            if (axis == 1):
                input_data = self.data_dict[input_name]
            elif (axis == -1):
                transpose_dims = [0, 3, 1, 2]
                inv_transpose_dims = [0, 2, 3, 1]
                input_data = self.data_dict[input_name]
                input_data = input_data.transpose(transpose_dims)
            else:
                print("[ERROR] unsupported preprocess axis %s" % (axis))
                exit(1)
        elif (len(shape) == 5 and shape[4] == 8):
            if (axis != 1):
                print("[ERROR] unsupported preprocess axis %s" % (axis))
                exit(1)
            input_data = self.data_dict[input_name].copy().transpose([0, 2, 3, 1, 4])
            input_shape = input_data.shape
            new_input_shape = [input_shape[0], input_shape[1], input_shape[2], input_shape[3]*input_shape[4]]
            input_data = input_data.reshape(new_input_shape).transpose([0, 3, 1, 2])
        else:
            print("[ERROR] unsupported preprocess input dims" % (len(shape)))
            exit(1)
        return input_data, shape, inv_transpose_dims

    def postprocess_nchwc8_nchw_output(self, output_data, input_shape, inv_transpose_dims):
        if (len(input_shape) == 3 or len(input_shape) == 4):
            output = output_data.transpose([0, 1, 4, 2, 3]).reshape(input_shape)
        else:
            output = output_data
        output = output.transpose(inv_transpose_dims)
        return output

    def add_input(self, input_name, input_shape):
        layer = caffe_net.LayerParameter(name=input_name, type='Input',
                                      top=[input_name])
        layer.input_param(input_shape)
        self.caffe_model.add_layer(layer)
        if (input_name not in self.data_dict.keys()):
            self.data_dict[input_name] = None
        return input_name

    def add_output(self, output_names):
        self.caffe_model.add_output(output_names)
        for item in output_names:
            print("[INFO] add model output %s" % (item))

    def add_quantization(self, scope_id, tensorflow_weight_name, output_name):
        self.scopes[scope_id] = tensorflow_weight_name
        self.scopes[scope_id+1] = "a_quant_max"
        weight_name = self.generate_name(self.scopes, scope_id+2)
        weight = self.get_weight(weight_name)
        self.quantization_max[output_name] = weight

    def add_concat(self, input_names, output_name, axis):
        layer = caffe_net.LayerParameter(name=output_name, type='Concat',
                    bottom=input_names, top=[output_name])
        layer.concat_param(axis)
        self.caffe_model.add_layer(layer)
        inputs = []
        for input_name in input_names:
            inputs.append(self.data_dict[input_name])
        self.data_dict[output_name] = Operators.concat(inputs, axis, output_name)
        return output_name

    def extract_layer_norm(self, input_name, output_name, scope_id,
            layer_names=["LayerNorm", "gamma", "beta"]):
        gamma, beta = self.get_weights(scope_id, layer_names)
        layer = caffe_net.LayerParameter(name=output_name, type='LayerNorm',
                    bottom=[input_name], top=[output_name])
        layer.add_data(gamma, beta)
        self.caffe_model.add_layer(layer)
        self.data_dict[output_name] = Operators.layer_norm(self.data_dict[input_name],
                                          gamma, beta,
                                          output_name)
        return output_name

    def extract_group_norm(self, input_name, groups, output_name, scope_id,
            data_format="NCHW",
            axis=1,
            layer_names=["GroupNorm", "gamma", "beta"]):
        assert (data_format == "NCHW")
        gamma, beta = self.get_weights(scope_id, layer_names)
        layer = caffe_net.LayerParameter(name=output_name, type='GroupNorm',
                    bottom=[input_name], top=[output_name])
        layer.add_data(gamma, beta)
        layer.group_norm_param(groups)
        self.caffe_model.add_layer(layer)
        if (self.data_dict[input_name] is not None):
            input_data, input_shape, inv_transpose_dims = self.preprocess_nchwc8_nchw_input(input_name, axis)
            output_data = Operators.group_norm(input_data,
                              groups,
                              gamma, beta,
                              output_name)
            self.data_dict[output_name] = self.postprocess_nchwc8_nchw_output(output_data, input_shape, inv_transpose_dims)
        else:
            self.data_dict[output_name] = None
        return output_name

    def extract_scale(self, input_name, output_name, scope_id,
            data_format="NCHW",
            axis=1,
            layer_names=["scale", "gamma", "beta"]):
        assert (data_format == "NCHW")
        gamma, beta = self.get_weights(scope_id, layer_names)
        layer = caffe_net.LayerParameter(name=output_name, type='Scale',
                    bottom=[input_name], top=[output_name])
        if beta is not None:
            layer.scale_param(axis=axis, bias_term=True)
            layer.add_data(gamma, beta)
        else:
            layer.scale_param(axis=axis, bias_term=False)
            layer.add_data(gamma)
        self.caffe_model.add_layer(layer)
        if (self.data_dict[input_name] is not None):
            input_data, input_shape, inv_transpose_dims = self.preprocess_nchwc8_nchw_input(input_name, axis)
            output_data = Operators.scale(input_data,
                              gamma, beta,
                              output_name)
            self.data_dict[output_name] = self.postprocess_nchwc8_nchw_output(output_data, input_shape, inv_transpose_dims)
        else:
            self.data_dict[output_name] = None
        return output_name

    def extract_batch_norm(self, input_name, output_name, scope_id,
            data_format="NCHW",
            axis=1, eps=1e-3,
            layer_names=["bn", "moving_mean", "moving_variance"]):
        assert (data_format == "NCHW")
        mean, var = self.get_weights(scope_id, layer_names)
        layer = caffe_net.LayerParameter(name=output_name, type='BatchNorm',
                    bottom=[input_name], top=[output_name])
        layer.batch_norm_param(axis=axis, eps=eps)
        layer.add_data(mean, var)
        self.caffe_model.add_layer(layer)
        if (self.data_dict[input_name] is not None):
            input_data, input_shape, inv_transpose_dims = self.preprocess_nchwc8_nchw_input(input_name, axis)
            output_data = Operators.batch_norm(input_data,
                              mean, var, eps,
                              output_name)
            self.data_dict[output_name] = self.postprocess_nchwc8_nchw_output(output_data, input_shape, inv_transpose_dims)
        else:
            self.data_dict[output_name] = None
        gamma = self.get_weights(scope_id, [layer_names[0], "gamma"])
        if (gamma is not None):
            scale_name = self.extract_scale(output_name, output_name+"_s",
                             scope_id, data_format, axis, [layer_names[0], "gamma", "beta"])
            self.data_dict[output_name] = self.data_dict[scale_name]
            output_name = scale_name
        return output_name

    def transpose_nchc8_nhc(self, x):
        x = self.add_transpose(x, x+"_nchc8_nhc", [0, 2, 3, 1, 4])
        shape = self.get_tensor_shape(x)
        assert(shape[2] == 1)
        x = self.add_reshape(x, x+"_r", [self.batch, -1, shape[3]*shape[4]])
        return x

    def transpose_nhwc_nchw(self, x):
        x = self.add_transpose(x, x+"_nhwc_nchw", [0, 3, 1, 2])
        return x

    def transpose_nhc_nchw(self, x):
        x = self.add_transpose(x, x+"_nhc_nch", [0, 2, 1])
        x = self.add_expand_dims(x, 3, x+"_nch_nchw")
        return x

    def calculate_convolution_padding(self, input_shape, kernel_size, strides, mode):
        i_h = input_shape[2]
        i_w = input_shape[3]
        f_h = kernel_size[0]
        f_w = kernel_size[1]
        s_h = strides[0]
        s_w = strides[1]
        if (mode == 'valid'):
            o_h = math.ceil((i_h - f_h + 1) / s_h)
            o_w = math.ceil((i_w - f_w + 1) / s_w)
        elif (mode == 'same'):
            o_h = math.ceil(i_h / s_h)
            o_w = math.ceil(i_w / s_w)
        else:
            print("[ERROR] unsupported padding mode %s" % (mode))
            exit(1)
        pad_h = max((o_h - 1) * s_h + f_h - i_h, 0)
        pad_w = max((o_w - 1) * s_w + f_w - i_w, 0)
        pad_top = math.floor(pad_h / 2)
        pad_bottom = pad_h - pad_top
        pad_left = math.floor(pad_w / 2)
        pad_right = pad_w - pad_left
        return [pad_top, pad_bottom, pad_left, pad_right]

    def extract_convolution(self, input_name, output_name, scope_id,
                            num_output, kernel_size, stride, padding,
                            data_format="NCHW", weight_format="HWCN",
                            axis=1, dilation=1, groups=1,
                            layer_names=["convolution", "kernel", "bias"]):
        kernel, bias = self.get_weights(scope_id, layer_names)
        if (weight_format == "HWCN"):
            if (len(kernel.shape) == 3):
                kernel = kernel.transpose([2, 1, 0])
                kernel = np.expand_dims(kernel, -1)
            elif (len(kernel.shape) == 4):
                kernel = kernel.transpose([3, 2, 0, 1])
            else:
                print("[ERROR] unsupported convolution kernel size")
                exit(1)
        layer = caffe_net.LayerParameter(name=output_name, type='Convolution',
                    bottom=[input_name], top=[output_name])
        if (bias is None):
            layer.add_data(kernel)
            bias_term = False
        else:
            layer.add_data(kernel, bias)
            bias_term = True
        layer.convolution_param(num_output, kernel_size, stride, padding,
                   bias_term , dilation, groups)
        self.caffe_model.add_layer(layer)
        assert (data_format == "NCHW")
        if (self.data_dict[input_name] is not None):
            input_data, input_shape, inv_transpose_dims = self.preprocess_nchwc8_nchw_input(input_name, axis)
            output_data = Operators.convolution(input_data,
                              kernel, bias,
                              num_output, kernel_size, stride, padding,
                              dilation, groups,
                              output_name)
            self.data_dict[output_name] = output_data #self.postprocess_nchwc8_nchw_output(output_data, input_shape, inv_transpose_dims)
        else:
            self.data_dict[output_name] = None
        return output_name

    def extract_dense(self, input_name, output_name, scope_id, scope_name="dense", share_index=0, share_num=1):
        if (isinstance(scope_name, str)):
            layer_names = [scope_name, "kernel", "bias"]
        elif (isinstance(scope_name, list)):
            layer_names = scope_name
        else:
            print("[ERROR] unsupported dense scope_name")
            exit(1)
        kernel, bias = self.get_weights(scope_id, layer_names)
        if (share_num == 1):
            layer = caffe_net.LayerParameter(name=output_name, type='InnerProduct',
                        bottom=[input_name], top=[output_name])
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
            self.data_dict[output_name] = Operators.fully_connect(self.data_dict[input_name],
                                             kernel.transpose((1, 0)), bias,
                                             output_name)
        else:
            self.scopes[scope_id] = layer_names[0]
            kernel_name = self.generate_name(self.scopes, scope_id+1) + "/kernel"
            bias_name = self.generate_name(self.scopes, scope_id+1) + "/bias"
            if (share_index == 0):
                self.add_weight(kernel_name, weight=kernel)
                if (bias is not None):
                    self.add_weight(bias_name, weight=bias)
            tmp_name = self.add_matmul(input_name, kernel_name, output_name+"/matmul"+str(share_index))
            if (bias is not None):
                self.add_sum([tmp_name, bias_name], output_name)
        return output_name

    def extract_denses(self, input_name, output_names, output_nums, scope_id, scope_name="dense", share_index=0, share_num=1):
        if (isinstance(scope_name, str)):
            layer_names = [scope_name, "kernel", "bias"]
        elif (isinstance(scope_name, list)):
            layer_names = scope_name
        else:
            print("[ERROR] unsupported dense scope_name")
            exit(1)
        kernels, biases = self.get_weights(scope_id, layer_names)
        if (share_num == 1):
            last_sum = 0
            for index in range(len(output_nums)):
                kernel = kernels[:, last_sum:last_sum+output_nums[index]]
                bias = None
                if biases is not None:
                    bias = biases[last_sum:last_sum+output_nums[index]]
                layer = caffe_net.LayerParameter(name=output_names[index], type='InnerProduct',
                            bottom=[input_name], top=[output_names[index]])
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
                self.data_dict[output_names[index]] = Operators.fully_connect(self.data_dict[input_name],
                                                         kernel.transpose((1, 0)), bias,
                                                         output_names[index])
                last_sum = last_sum + output_nums[index]
            if (last_sum != len(kernels[0])):
                print("[ERROR] extract_denses failed")
                exit(0)
        else:
            self.scopes[scope_id] = layer_names[0]
            kernel_name = self.generate_name(self.scopes, scope_id+1) + "/kernel"
            bias_name = self.generate_name(self.scopes, scope_id+1) + "/bias"
            if (share_index == 0):
                self.add_weight(kernel_name, weight=kernels)
                if (biases is not None):
                    self.add_weight(bias_name, weight=biases)
            tmp_name = self.add_matmul(input_name, kernel_name, self.generate_name(self.scopes, scope_id+1)+"/matmul"+str(share_index))
            if (biases is not None):
                tmp_name = self.add_sum([tmp_name, bias_name], self.generate_name(self.scopes, scope_id+1)+"/sum"+str(share_index))
            slice_point = []
            last_sum = 0
            for i in range(len(output_nums)-1):
                last_sum = last_sum + output_nums[i]
                slice_point.append(last_sum)
            shape_len = len(self.get_tensor_shape(self.generate_name(self.scopes, scope_id+1)+"/matmul"+str(share_index)))
            self.add_slice(tmp_name, output_names, shape_len-1, slice_point)
        return output_names

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
        layer = caffe_net.LayerParameter(name=output_name, type='Permute',
                    bottom=[input_name], top=[output_name])
        layer.permute_param(dim)
        self.caffe_model.add_layer(layer)
        self.data_dict[output_name] = Operators.transpose(self.data_dict[input_name], dim, output_name)
        return output_name

    def add_matmul(self, input_a_name, input_b_name, output_name, transpose_a=False, transpose_b=False):
        layer = caffe_net.LayerParameter(name=output_name, type='MatMul',
                    bottom=[input_a_name, input_b_name], top=[output_name])
        layer.matmul_param(transpose_a, transpose_b)
        self.caffe_model.add_layer(layer)
        self.data_dict[output_name] = Operators.matmul(self.data_dict[input_a_name], transpose_a,
                                          self.data_dict[input_b_name], transpose_b, output_name)
        return output_name

    def add_power(self, input_name, output_name, scale=1, shift=0, power=1):
        layer = caffe_net.LayerParameter(name=output_name, type='Power',
                    bottom=[input_name], top=[output_name])
        layer.power_param(scale, shift, power)
        self.caffe_model.add_layer(layer)
        self.data_dict[output_name] = Operators.power(self.data_dict[input_name], scale, shift, power, output_name)
        return output_name

    def add_div(self, input_names, output_name):
        layer = caffe_net.LayerParameter(name=output_name, type='Eltwise',
                    bottom=input_names,
                    top=[output_name])
        layer.eltwise_param(3) #Div
        self.caffe_model.add_layer(layer)
        data = []
        for name in input_names:
            data.append(self.data_dict[name])
        self.data_dict[output_name] = Operators.divide(self.data_dict[input_names[0]], self.data_dict[input_names[1]], output_name)
        return output_name

    def add_prod(self, input_names, output_name):
        layer = caffe_net.LayerParameter(name=output_name, type='Eltwise',
                    bottom=input_names,
                                      top=[output_name])
        layer.eltwise_param(0) #Prod
        self.caffe_model.add_layer(layer)
        self.data_dict[output_name] = self.data_dict[input_names[0]]
        for i in range(1, len(input_names)):
            self.data_dict[output_name] = Operators.matmultiply(self.data_dict[output_name],
                                              self.data_dict[input_names[i]], output_name)
        return output_name

    def add_l2norm(self, input_name, output_name):
        layer = caffe_net.LayerParameter(name=output_name, type='L2Norm',
                    bottom=[input_name],
                    top=[output_name])
        self.caffe_model.add_layer(layer)
        self.data_dict[output_name] = Operators.l2_norm(self.data_dict[input_name], output_name)
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

    def add_attention_mask(self, input_name, output_name, attn_trunc_len, same_length, mask):
        layer = caffe_net.LayerParameter(name=output_name, type='AttentionMask',
                    bottom=[input_name], top=[output_name])
        layer.attention_mask_param(attn_trunc_len, same_length, mask)
        self.caffe_model.add_layer(layer)
        self.data_dict[output_name] = Operators.attention_mask(self.data_dict[input_name],
                                          attn_trunc_len, same_length, mask,
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

    def add_softmax(self, input_name, output_name, axis):
        layer = caffe_net.LayerParameter(name=output_name, type='Softmax',
                    bottom=[input_name],
                    top=[output_name])
        layer.softmax_param(axis)
        self.caffe_model.add_layer(layer)
        self.data_dict[output_name] = Operators.softmax(self.data_dict[input_name], axis, output_name)
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

    def add_relu6(self, input_name, output_name):
        layer = caffe_net.LayerParameter(name=output_name, type='ReLU6',
                    bottom=[input_name],
                    top=[output_name])
        self.caffe_model.add_layer(layer)
        self.data_dict[output_name] = Operators.relu(self.data_dict[input_name], output_name, max_value=6)
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
            weight = self.get_weight(weight_name)
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
            input_dim = weight.shape[-1]
            embedding_dim = weight.shape[-2]
        else:
            input_dim = weight.shape[-2]
            embedding_dim = weight.shape[-1]
        layer.embed_param(input_dim, embedding_dim, transpose)
        self.caffe_model.add_layer(layer)
        self.data_dict[output_name] = Operators.embedding(self.data_dict[input_name], weight, transpose, output_name)
        return output_name

    def extract_embedding(self, input_name, scope_id, tensorflow_weight_name, output_name):
        self.scopes[scope_id] = tensorflow_weight_name
        weight_name = self.generate_name(self.scopes, scope_id+1)
        weight = self.get_weight(weight_name)
        layer = caffe_net.LayerParameter(name=output_name, type='Embed',
                    bottom=[input_name], top=[output_name])
        layer.add_data(weight)
        embedding_dim = len(weight[0])
        layer.embed_param(len(weight), embedding_dim, False)
        self.caffe_model.add_layer(layer)
        self.data_dict[output_name] = Operators.embedding(self.data_dict[input_name], weight, False, output_name)
        return output_name

    def add_relative_position_embedding(self, input_name, weight_name, axis, output_name, transpose=False):
        layer = caffe_net.LayerParameter(name=output_name, type='RelativePositionEmbed',
                    bottom=[input_name,weight_name], top=[output_name])
        weight = self.data_dict[weight_name]
        if transpose:
            input_dim = len(weight[0])
            embedding_dim = len(weight)
        else:
            input_dim = len(weight)
            embedding_dim = len(weight[0])
        layer.relative_position_embed_param(input_dim, embedding_dim, transpose, axis)
        self.caffe_model.add_layer(layer)
        self.data_dict[output_name] = Operators.relative_position_embedding(self.data_dict[input_name],
                                          weight, axis, output_name)
        return output_name

    def add_reduce_mean(self, input_name, axis, keep_dim, output_name):
        operation = 4 # MEAN
        layer = caffe_net.LayerParameter(name=output_name, type='Reduction',
                    bottom=[input_name], top=[output_name])
        layer.reduction_param(operation, axis, keep_dim)
        self.caffe_model.add_layer(layer)
        self.data_dict[output_name] = Operators.reduction(self.data_dict[input_name], None, operation, axis, output_name)
        return output_name

    def add_reduce_sum(self, input_name, axis, keep_dim, output_name, mask_input_name=None):
        operation = 1 # SUM
        bottom = [input_name]
        if (mask_input_name is not None):
            bottom.append(mask_input_name)
        layer = caffe_net.LayerParameter(name=output_name, type='Reduction',
                    bottom=bottom, top=[output_name])
        layer.reduction_param(operation, axis, keep_dim)
        self.caffe_model.add_layer(layer)
        if (mask_input_name is None):
            mask = None
        else:
            mask = self.data_dict[mask_input_name]
        self.data_dict[output_name] = Operators.reduction(self.data_dict[input_name], mask,
            operation, axis, output_name)
        return output_name

    def add_expand_dims(self, input_name, axis, output_name):
        layer = caffe_net.LayerParameter(name=output_name, type='Unsqueeze',
                    bottom=[input_name], top=[output_name])
        layer.unsqueeze_param(axis)
        self.caffe_model.add_layer(layer)
        self.data_dict[output_name] = Operators.expand_dims(self.data_dict[input_name], axis, output_name)
        return output_name

    def add_tile(self, input_name, loops, axis, output_name):
        layer = caffe_net.LayerParameter(name=output_name, type='Tile',
                                         bottom=[input_name], top=[output_name])
        layer.tile_param(axis, loops)
        self.caffe_model.add_layer(layer)
        self.data_dict[output_name] = Operators.tile(self.data_dict[input_name], loops, axis, output_name)
        return output_name

    def add_argmax(self, input_name, axis, output_name):
        layer = caffe_net.LayerParameter(name=output_name, type='ArgMax',
                    bottom=[input_name], top=[output_name])
        layer.argmax_param(axis)
        self.caffe_model.add_layer(layer)
        self.data_dict[output_name] = Operators.argmax(self.data_dict[input_name], axis, output_name)
        return output_name

    def extract_rnn(self, mode, input_name, state_name, output_name, scope_id,
            steps=-1, scope_name="basic_lstm_cell",
            use_proj=False, zoneout_cell=0, zoneout_output=0, linear_before_reset=False):
        if (isinstance(scope_name, str)):
            scope_name = [scope_name]
        bottom = [input_name]
        if (state_name is not None):
            bottom.append(state_name)
        layer = caffe_net.LayerParameter(name=output_name, type=mode,
                    bottom=bottom, top=[output_name])
        if (mode == "LSTM"):
            factor = 4
        elif (mode == "GRU"):
            factor = 3
        elif (mode == "GRU_LBR"):
            factor = 3
        else:
            print("[ERROR] RNN can not support %s" % (mode))
            exit(1)
        kernels = []
        biases = []
        projections = []
        projection_biases = []
        for i in range(len(scope_name)):
            kernel, bias = self.get_weights(scope_id, [scope_name[i], "kernel", "bias"])
            projection_size = 0;
            projection = None
            if (use_proj):
                self.scopes[scope_id] = scope_name[i]
                projection, projection_bias = self.get_weights(scope_id+1, ["projection", "kernel", "bias"])
                projection_size = projection.shape[0]
            num_output_4 = len(kernel[0])
            if (bias is not None):
                if (len(bias) != num_output_4):
                    print("[ERROR] extract_rnn failed")
                    exit(0)
            if (use_proj):
                num_output = projection.shape[1]
            else:
                num_output = num_output_4 // factor
            if (self.calculate and len(kernel) != self.get_tensor_shape(input_name)[-1] + num_output):
                kernel_2, bias_2 = self.get_weights(scope_id, [scope_name[i], "recurrent_kernel", "bias"])
                if (kernel_2 is not None):
                    kernel = np.concatenate([kernel, kernel_2], axis = 0)
            kernels.append(kernel.transpose([1, 0]))
            if (bias is None):
                bias = np.zeros([num_output_4 // 2])
            biases.append(bias)
            if (use_proj):
                projections.append(projection.transpose([1, 0]))
                if (projection_bias is not None):
                    projection_bias = np.zeros(num_output)
                projection_biases.append(projection_bias)
            else:
                projections.append(None)
                projection_biases.append(None)
        if (use_proj):
            if (projection_biases[0] is not None):
                layer.add_data(np.concatenate(kernels, axis=0), np.concatenate(biases, axis=0),
                    np.concatenate(projections, axis=0), np.concatenate(projection_biases, axis=0))
            else:
                layer.add_data(np.concatenate(kernels, axis=0), np.concatenate(biases, axis=0),
                    np.concatenate(projections, axis=0))
        else:
            layer.add_data(np.concatenate(kernels, axis=0),
                np.concatenate(biases, axis=0))
        if (mode == "LSTM"):
            layer.lstm_param(num_output, steps, projection_size, zoneout_cell, zoneout_output)
        elif (mode == "GRU"):
            layer.lstm_param(num_output, steps, projection_size, zoneout_cell, zoneout_output)
        elif (mode == "GRU_LBR"):
            layer.lstm_param(num_output, steps, projection_size, zoneout_cell, zoneout_output)
        else:
            print("[ERROR] RNN can not support %s" % (mode))
            exit(1)
        self.caffe_model.add_layer(layer)
        if (steps >= 0):
            self.data_dict[output_name] = Operators.fw_rnn(mode, self.data_dict[input_name],
                kernels[0],
                biases[0],
                projections[0],
                projection_biases[0],
                zoneout_cell, zoneout_output,
                output_name)
        elif (steps == -1):
            self.data_dict[output_name], self.data_dict[state_name] = Operators.rnn(mode, self.data_dict[input_name],
                self.data_dict[state_name],
                kernels[0],
                biases[0],
                projections[0],
                projection_biases[0],
                zoneout_cell, zoneout_output,
                output_name,
                state_name)
        elif (steps == -2):
            self.data_dict[output_name] = Operators.bi_rnn(mode, self.data_dict[input_name],
                kernels,
                biases,
                projections,
                projection_biases,
                zoneout_cell, zoneout_output,
                output_name)
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

    def add_jump(self, jump_start_name, output_name="jump", status_name=None):
        bottom = []
        if (status_name is not None):
            bottom = [jump_start_name, status_name]
        else:
            bottom = [jump_start_name]
        layer = caffe_net.LayerParameter(name=output_name, type='Jump',
                    bottom=bottom, top=[output_name])
        self.caffe_model.add_layer(layer)
        return jump_start_name

    def add_copy(self, src_name, src_batch_stride, src_stride, src_offset,
            dst_name, dst_batch_stride, dst_stride, dst_offset,
            length,
            output_name,
            src_index_name=None, dst_index_name=None):
        src_index = None
        dst_index = None
        if (src_index_name is None):
            layer = caffe_net.LayerParameter(name=output_name, type='Copy',
                        bottom=[src_name, dst_name], top=[output_name])
        else:
            layer = caffe_net.LayerParameter(name=output_name, type='Copy',
                        bottom=[src_name, dst_name, src_index_name, dst_index_name], top=[output_name])
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

    def add_repeat(self, loops, repeat_start_name,
            output_name="repeat", status_name=None, axis_name=None, axis=-1):
        bottom = [repeat_start_name]
        if (status_name is not None):
            bottom.append(status_name)
        if (axis_name is not None):
            bottom.append(axis_name)
        layer = caffe_net.LayerParameter(name=output_name, type='Repeat',
                    bottom=bottom, top=[output_name])
        layer.repeat_param(loops, axis)
        self.caffe_model.add_layer(layer)
        return repeat_start_name

    def add_repeat_set_times(self, repeat_name, input_name, input_axis, output_name="set_repeat_times"):
        layer = caffe_net.LayerParameter(name=output_name, type='RepeatSetTimesByAxisLength',
                    bottom=[repeat_name, input_name], top=[output_name])
        layer.repeat_set_times_by_axis_length_param(input_axis)
        self.caffe_model.add_layer(layer)
        return repeat_name

    def add_memory(self, memory_name, memory_shapes, data_type):
        layer = caffe_net.LayerParameter(name=memory_name+"_mem", type='PreAllocatedMemory',
            top=[memory_name])
        layer.memory_param(memory_shapes, data_type)
        self.caffe_model.add_layer(layer)
        self.data_dict[memory_name] = Operators.zeros(memory_shapes, memory_name)
        return memory_name

    def add_pad(self, input_name, output_name, padding_shapes, padding_values=None):
        layer = caffe_net.LayerParameter(name=output_name, type='Pad',
                    bottom=[input_name], top=[output_name])
        layer.padding_param(padding_shapes, padding_values)
        self.caffe_model.add_layer(layer)
        self.data_dict[output_name] = Operators.pad(self.data_dict[input_name], padding_shapes, padding_values, output_name)
        return output_name

    def add_relative_shift(self, input_name, output_name, axis, shift_length):
        layer = caffe_net.LayerParameter(name=output_name, type='RelativeShift',
                    bottom=[input_name], top=[output_name])
        layer.relative_shift_param(axis, shift_length)
        self.caffe_model.add_layer(layer)
        self.data_dict[output_name] = Operators.relative_shift(self.data_dict[input_name], axis, shift_length, output_name)
        return output_name

    def add_clip(self, input_name, output_name, min_value, max_value):
        layer = caffe_net.LayerParameter(name=output_name, type='Clip',
                    bottom=[input_name],
                    top=[output_name])
        layer.clip_param(min_value, max_value)
        self.caffe_model.add_layer(layer)
        self.data_dict[output_name] = Operators.clip(self.data_dict[input_name], min_value, max_value, output_name)
        return output_name

    def add_exp(self, input_name, output_name, base=-1, scale=1, shift=0):
        layer = caffe_net.LayerParameter(name=output_name, type='Exp',
                    bottom=[input_name],
                    top=[output_name])
        layer.exp_param(base, scale, shift)
        self.caffe_model.add_layer(layer)
        self.data_dict[output_name] = Operators.exp(self.data_dict[input_name], base, scale, shift, output_name)
        return output_name

    def add_softplus(self, input_name, output_name):
        layer = caffe_net.LayerParameter(name=output_name, type='SoftPlus',
                    bottom=[input_name],
                    top=[output_name])
        self.caffe_model.add_layer(layer)
        self.data_dict[output_name] = Operators.softplus(self.data_dict[input_name], output_name)
        return output_name

    def add_mish(self, input_name, output_name):
        layer = caffe_net.LayerParameter(name=output_name, type='Mish',
                    bottom=[input_name],
                    top=[output_name])
        self.caffe_model.add_layer(layer)
        self.data_dict[output_name] = Operators.mish(self.data_dict[input_name], output_name)
        return output_name
