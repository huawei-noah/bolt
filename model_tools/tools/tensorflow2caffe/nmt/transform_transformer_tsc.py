#!/usr/local/bin/python
# -*- coding: utf-8 -*-

from tensorflow2caffe_transformer_tsc import Tensorflow2CaffeTransformerTSC
import numpy as np

if __name__ == '__main__':
    tensorflow_model_path = "/data/bolt/model_zoo/tensorflow_models/nmt_tsc/model.ckpt-353000"
    encoder_params = Tensorflow2CaffeTransformerTSC.default_encoder_params()
    decoder_params = Tensorflow2CaffeTransformerTSC.default_decoder_params()

    max_seq_length = 128
    max_decode_length = 128
    nmt_caffe = Tensorflow2CaffeTransformerTSC(tensorflow_model_path,
                 "nmt_tsc_encoder", "nmt_tsc_encoder",
                 max_seq_length, max_decode_length,
                 encoder_params, decoder_params,
                 check=False, calc=True)
    encoder_data = {}
    encoder_data["encoder_words"] = np.array([[13024, 1657, 35399, 0]]) # result:[6160, 3057, 113, 157, 0]
    encoder_length = len(encoder_data["encoder_words"][0])
    encoder_data["encoder_positions"] = np.array([[i for i in range(encoder_length)]])
    nmt_caffe.generate_encoder(encoder_data)

    word = 0
    results = []
    for i in range(max_decode_length):
        nmt_caffe = Tensorflow2CaffeTransformerTSC(tensorflow_model_path,
                     "nmt_tsc_decoder", "nmt_tsc_decoder",
                     max_seq_length, max_decode_length,
                     encoder_params, decoder_params,
                     check=False, calc=True)
        decoder_data = {}
        decoder_data["decoder_words"] = np.array([[word]])
        decoder_data["decoder_positions"] = np.array([[i]])
        result = nmt_caffe.generate_decoder(decoder_data)
        word = nmt_caffe.get_tensor(result).tolist()[0][0][0]
        results.append(word)
        if (word == 0):
            break;
    print(results)
