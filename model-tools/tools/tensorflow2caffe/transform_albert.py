#!/usr/local/bin/python
# -*- coding: utf-8 -*-

from tensorflow2caffe_albert import Tensorflow2CaffeALBert
import numpy as np


if __name__ == '__main__':
    #tensorflow_model_path = "../albert/albert_tiny/albert_model.ckpt"
    #encoder_layers = 4
    #caffe_model_path_prefix = "albert_tiny"
    tensorflow_model_path = "../albert/albert_base_zh_additional_36k_steps/albert_model.ckpt"
    encoder_layers = 12
    caffe_model_path_prefix = "albert_base"
    caffe_model_name = "albert_base"
    #tensorflow_model_path = "../albert/albert_large_zh/albert_model.ckpt"
    #encoder_layers = 24
    #caffe_model_path_prefix = "albert_large"

    max_seq_length = 128
    embedding_dim = 768
    num_heads = 12

    albert_caffe = Tensorflow2CaffeALBert(tensorflow_model_path, caffe_model_path_prefix, caffe_model_name,
                       max_seq_length, embedding_dim, encoder_layers, num_heads,
                       False, True)
    data = {}
    data["bert_words"]      = np.array([[101,1045,2342,1037,14764,2005,2296,5353,3531,102]])
    bert_length = len(data["bert_words"][0])
    data["bert_positions"]  = np.array([[i for i in range(bert_length)]])
    data["bert_token_type"] = np.array([[0] * bert_length])
    #data["bert_mask"]       = np.array([[1] * bert_length])
    albert_caffe.generate(data)
