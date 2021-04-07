#!/usr/local/bin/python
# -*- coding: utf-8 -*-

from tensorflow2caffe_tinybert import Tensorflow2CaffeTinyBert
import numpy as np
import json

if __name__ == '__main__':
    tensorflow_model_path = "/data/models/bert/tinybert/tts-tinybert/model.ckpt-173506"
    configure_file_path = "/data/models/bert/tinybert/tts-tinybert/config.json"
    configure_file = open(configure_file_path)
    params = json.load(configure_file)
    configure_file.close()

    max_seq_length = 64
    embedding_dim = params["emb_size"]
    encoder_layers = params["num_hidden_layers"]
    num_heads = params["num_attention_heads"]
    caffe_model_path_prefix = "tts_tinybert"
    caffe_model_name = "tts_tinybert"

    bert_caffe = Tensorflow2CaffeTinyBert(tensorflow_model_path,
                     caffe_model_path_prefix, caffe_model_name,
                     max_seq_length, embedding_dim, encoder_layers, num_heads,
                     True, True)
    data = {}
    data["tinybert_words"]      = np.array([[101,1045,2342,1037,14764,2005,2296,5353,3531,102]])
    tinybert_length = len(data["tinybert_words"][0])
    data["tinybert_positions"]  = np.array([[i for i in range(tinybert_length)]])
    data["tinybert_token_type"] = np.array([[0] * tinybert_length])
    bert_caffe.generate_tts_preprocess_task(data)
