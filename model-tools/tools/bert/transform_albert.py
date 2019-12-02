#!/usr/local/bin/python
# -*- coding: utf-8 -*-

from tensorflow2caffe_albert import Tensorflow2CaffeALBert


if __name__ == '__main__':
    #tensorflow_model_path = "../albert/albert_tiny/albert_model.ckpt"
    #encoder_layers = 4
    #caffe_model_path_prefix = "albert_tiny"
    tensorflow_model_path = "../albert/albert_base_zh_additional_36k_steps/albert_model.ckpt"
    encoder_layers = 12
    caffe_model_path_prefix = "albert_base"
    #tensorflow_model_path = "../albert/albert_large_zh/albert_model.ckpt"
    #encoder_layers = 24
    #caffe_model_path_prefix = "albert_large"

    seq_length = 128
    attention_nums = 12

    albert_caffe = Tensorflow2CaffeALBert(tensorflow_model_path, seq_length, encoder_layers, attention_nums, caffe_model_path_prefix)
    albert_caffe.weight_map()
    albert_caffe.generate()
