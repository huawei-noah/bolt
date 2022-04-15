#!/usr/local/bin/python
# -*- coding: utf-8 -*-

from tensorflow2caffe_convolution_transformer import Tensorflow2CaffeConvolutionTransformer
from convolution_transformer_reference_params import base_params
import numpy as np

def transform_reference(model_path_prefix, data_path_prefix, quantization, FFN_decomposition):
    tensorflow_model_path = model_path_prefix + "/encoder.pb"
    caffe_model_path_prefix = "asr_convolution_transformer_reference"
    caffe_model_name = "asr_convolution_transformer_reference"
    first_frame = True
    asr_caffe = Tensorflow2CaffeConvolutionTransformer(base_params,
                       tensorflow_model_path, caffe_model_path_prefix, caffe_model_name,
                       nchwc8=True, first_frame=first_frame,
                       check=False, calc=True, quantization=quantization, FFN_decomposition=FFN_decomposition)
    data = {}
    data["reference"] = np.array([[42, 39, 42, 39, 41, 38, 14, 28, 6, 36, 25, 23, 32, 15, 8, 36, 33, 39, 37, 19, 33, 8, 27, 27, 23, 5]])
    out = asr_caffe.generate_reference(data)
    np.save("reference.npy", asr_caffe.get_tensor(out))

def transform_encoder(model_path_prefix, data_path_prefix, quantization, FFN_decomposition):
    tensorflow_model_path = model_path_prefix + "/encoder.pb"
    caffe_model_path_prefix = "asr_convolution_transformer_encoder"
    caffe_model_name = "asr_convolution_transformer_encoder"
    first_frame = True
    asr_caffe = Tensorflow2CaffeConvolutionTransformer(base_params,
                       tensorflow_model_path, caffe_model_path_prefix, caffe_model_name,
                       nchwc8=True, first_frame=first_frame,
                       check=False, calc=True, quantization=quantization, FFN_decomposition=FFN_decomposition)
    data = {}
    data["reference"] = np.load("reference.npy")
    if (first_frame):
        data["sounds"] = np.load(data_path_prefix + "/frame0.npy")
    else:
        data["sounds"] = np.load(data_path_prefix + "/frame1.npy")
    asr_caffe.generate_encoder(data)

if __name__ == '__main__':
    model_path_prefix = "/data/bolt/model_zoo/tensorflow_models/jiaoyu_asr"
    data_path_prefix = "/data/bolt/model_zoo/tensorflow_models/jiaoyu_asr"

    quantization = False
    FFN_decomposition = False
    transform_reference(model_path_prefix, data_path_prefix, quantization, FFN_decomposition)
    transform_encoder(model_path_prefix, data_path_prefix, quantization, FFN_decomposition)
