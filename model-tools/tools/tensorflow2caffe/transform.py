#!/usr/local/bin/python
# -*- coding: utf-8 -*-

from tensorflow2caffe_bert import Tensorflow2CaffeBert
from tensorflow2caffe_tinybert import Tensorflow2CaffeTinyBert


if __name__ == '__main__':
    tensorflow_model_path = "../tf_ckpt/model.ckpt"
    seq_length = 32
    encoder_layers = 4
    attention_nums = 12
    caffe_model_path_prefix = "tinybert"

    #bert_caffe = Tensorflow2CaffeBert(tensorflow_model_path, seq_length, encoder_layers, attention_nums, caffe_model_path_prefix)
    bert_caffe = Tensorflow2CaffeTinyBert(tensorflow_model_path, seq_length, encoder_layers, attention_nums, caffe_model_path_prefix)
    bert_caffe.weight_map()
    bert_caffe.generate()
