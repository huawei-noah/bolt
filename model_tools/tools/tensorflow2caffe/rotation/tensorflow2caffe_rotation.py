#!/usr/local/bin/python
# -*- coding: utf-8 -*-

import sys
sys.path.append("../")
from tensorflow2caffe import Tensorflow2Caffe

class Tensorflow2CaffeRotation(Tensorflow2Caffe):
    def __init__(self,
            tensorflow_model_path, caffe_model_path_prefix, caffe_model_name,
            check=False, calc=False):
        Tensorflow2Caffe.__init__(self, tensorflow_model_path,
            caffe_model_path_prefix, caffe_model_name, check, calc)

    def generate(self, input=None):
        steps = 30
        input_name = "input"
        input_shape = [self.batch, 3, steps]
        self.add_input(input_name, input_shape)
        self.set_input(input)

        kernel_size = [3, 1]
        strides = [1, 1]
        padding = self.calculate_convolution_padding(self.get_tensor_shape(input_name), kernel_size, strides, 'same')
        x = self.extract_convolution(input_name, "conv1d_1", 0,
                            64, kernel_size, strides, padding,
                            data_format="NCHW",
                            dilation=1, groups=1, layer_names=['conv1d', "kernel", "bias"])
        x = self.add_tanh(x, "tanh_1")
        x = self.extract_convolution(x, "conv1d_2", 0,
                            64, kernel_size, strides, padding,
                            data_format="NCHW",
                            dilation=1, groups=1, layer_names=['conv1d_1', "kernel", "bias"])
        x = self.add_tanh(x, "tanh_2")
        x = self.transpose_nchc8_nhc(x)
        x = self.extract_rnn("LSTM", x, None, "lstm_backbone", steps=steps,
            scope_id=0, scope_name="lstm_backbone")
        x1 = self.extract_rnn("LSTM", x, None, "lstm_pose_2", steps=steps,
            scope_id=0, scope_name="lstm_pose_2")
        x2 = self.extract_rnn("LSTM", x, None, "lstm_scene_2", steps=steps,
            scope_id=0, scope_name="lstm_scene_2")
        x1 = self.extract_dense(x1, "pose_output", 0, "pose_output")
        x2 = self.extract_dense(x2, "scene_output", 0, "scene_output")

        self.save_caffe_model()
