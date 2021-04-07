#!/usr/local/bin/python
# -*- coding: utf-8 -*-

from tensorflow2caffe_tinybert import Tensorflow2CaffeTinyBert
import numpy as np
import json

if __name__ == '__main__':
    tensorflow_model_path = "/data/models/bert/tinybert/disambiguate/model.ckpt"
    configure_file_path = "/data/models/bert/tinybert/disambiguate/bert_config.json"
    configure_file = open(configure_file_path)
    params = json.load(configure_file)
    configure_file.close()

    max_seq_length = 32
    embedding_dim = params["emb_size"]
    encoder_layers = params["num_hidden_layers"]
    num_heads = params["num_attention_heads"]
    caffe_model_path_prefix = "tinybert_disambiguate"
    caffe_model_name = "tinybert_disambiguate"

    bert_caffe = Tensorflow2CaffeTinyBert(tensorflow_model_path,
                     caffe_model_path_prefix, caffe_model_name,
                     max_seq_length, embedding_dim, encoder_layers, num_heads,
                     True, True)
    data = {}
    data["tinybert_words"] = np.array([[101, 3017, 5164,  678, 5341, 5686, 5688, 4680, 5564, 6577, 1920, 1104,
         2773, 5018,  671, 2108, 2001, 3813, 3924, 2193, 4028, 3330, 3247,  712,
         2898, 4638,  102]])
    tinybert_length = len(data["tinybert_words"][0])
    data["tinybert_positions"]  = np.array([[i for i in range(tinybert_length)]])
    data["tinybert_token_type"] = np.array([[0] * tinybert_length])
    data["tinybert_words_mask"] = np.array([[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         1, 1, 0, 0, 0, 0, 0]]])
    data["tinybert_dict_type"] = np.array([[5]])
    bert_caffe.print_weight_map()
    bert_caffe.generate_disambiguate_task(data)
