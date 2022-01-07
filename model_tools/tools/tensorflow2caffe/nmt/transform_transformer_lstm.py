#!/usr/local/bin/python
# -*- coding: utf-8 -*-

from tensorflow2caffe_transformer_lstm import Tensorflow2CaffeTransformerLstm
import numpy as np
import json

if __name__ == '__main__':
    tensorflow_model_path = "/data/bolt/model_zoo/tensorflow_models/tfm-rnn-288/model.ckpt-217809"
    configure_file_path = "/data/bolt/model_zoo/tensorflow_models/tfm-rnn-288/train_options.json"
    caffe_model_path_prefix = "transformer_lstm_nmt"
    caffe_model_name = "transformer_lstm_nmt"
    configure_file = open(configure_file_path)
    params = json.load(configure_file)
    configure_file.close()
    encoder_params = Tensorflow2CaffeTransformerLstm.default_encoder_params()
    decoder_params = Tensorflow2CaffeTransformerLstm.default_decoder_params()
    for key,value in params["model_params"]["encoder.params"].items():
        if (key in encoder_params):
            encoder_params[key] = value
    for key,value in params["model_params"]["decoder.params"].items():
        if (key in decoder_params):
            decoder_params[key] = value

    max_seq_length = 128
    max_decode_length = 128
    use_small_word_list = False
    max_candidates_size = max_seq_length * 50 + 2000
    nmt_caffe = Tensorflow2CaffeTransformerLstm(tensorflow_model_path,
                 caffe_model_path_prefix, caffe_model_name,
                 max_seq_length, max_decode_length,
                 encoder_params, decoder_params,
                 use_small_word_list, max_candidates_size,
                 check=False, calc=True)

    data = {}
    data["nmt_words"]     = np.array([[2056,1176,6492,897,285,50,121,809,53,2988,263,1252,14,76,407,383,2]])
    nmt_length = len(data["nmt_words"][0])
    data["nmt_positions"] = np.array([[i for i in range(nmt_length)]])
    data["nmt_candidates"] = np.array([[i for i in range(nmt_length)]])
    nmt_caffe.generate(data)
