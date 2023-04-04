#!/usr/local/bin/python
# -*- coding: utf-8 -*-

from tensorflow2caffe_convolution_transformer import Tensorflow2CaffeConvolutionTransformer
from convolution_transformer_params_slu import base_params
import numpy as np

def transform_encoder(model_path_prefix, data_path_prefix, quantization, FFN_decomposition):
    tensorflow_model_path = model_path_prefix + "/slu_encoder_stream.pb"
    caffe_model_path_prefix = "asr_convolution_transformer_encoder"
    caffe_model_name = "asr_convolution_transformer_encoder"
    first_frame = False
    asr_caffe = Tensorflow2CaffeConvolutionTransformer(base_params,
                       tensorflow_model_path, caffe_model_path_prefix, caffe_model_name,
                       nchwc8=True, first_frame=first_frame,
                       check=False, calc=True, quantization=quantization, FFN_decomposition=FFN_decomposition)

    data = {}
    if (first_frame):
        data["sounds"] = np.load(data_path_prefix + "/frame0.npy")
    else:
        data["sounds"] = np.load(data_path_prefix + "/chunk_235.npy")
    asr_caffe.generate_encoder(data)

if __name__ == '__main__':
    model_path_prefix = "."
    data_path_prefix = "./slu"

    quantization = False
    FFN_decomposition = False
    transform_encoder(model_path_prefix, data_path_prefix, quantization, FFN_decomposition)
