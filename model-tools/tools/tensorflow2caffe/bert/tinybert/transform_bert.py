#!/usr/local/bin/python
# -*- coding: utf-8 -*-

from tensorflow2caffe_bert import Tensorflow2CaffeBert
import numpy as np


if __name__ == '__main__':
    tensorflow_model_path = "../../bert/cased_L-12_H-768_A-12/bert_model.ckpt"
    caffe_model_path_prefix = "bert_base"
    caffe_model_name = "bert_base"

    max_seq_length = 128
    embedding_dim = 768
    encoder_layers = 12
    num_heads = 12

    bert_caffe = Tensorflow2CaffeBert(tensorflow_model_path, caffe_model_path_prefix, caffe_model_name,
                     max_seq_length, embedding_dim, encoder_layers, num_heads,
                     False, True)
    data = {}
    data["bert_words"]      = np.array([[101,1045,2342,1037,14764,2005,2296,5353,3531,102]])
    bert_length = len(data["bert_words"][0])
    data["bert_positions"]  = np.array([[i for i in range(bert_length)]])
    data["bert_token_type"] = np.array([[0] * bert_length])
    #data["bert_mask"]       = np.array([[1] * bert_length])
    bert_caffe.generate(data)
