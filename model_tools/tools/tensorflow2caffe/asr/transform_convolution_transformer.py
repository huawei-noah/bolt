#!/usr/local/bin/python
# -*- coding: utf-8 -*-

from tensorflow2caffe_convolution_transformer import Tensorflow2CaffeConvolutionTransformer
from convolution_transformer_params import base_params
import numpy as np

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
    if (first_frame):
        data["sounds"] = np.load(data_path_prefix + "/frame0.npy")
    else:
        data["sounds"] = np.load(data_path_prefix + "/frame1.npy")
    asr_caffe.generate_encoder(data)

def transform_prediction_net(model_path_prefix, quantization):
    tensorflow_model_path = model_path_prefix + "/pred_net.pb"
    caffe_model_path_prefix = "asr_convolution_transformer_prediction_net"
    caffe_model_name = "asr_convolution_transformer_prediction_net"
    asr_caffe = Tensorflow2CaffeConvolutionTransformer(base_params,
                       tensorflow_model_path, caffe_model_path_prefix, caffe_model_name,
                       check=False, calc=True, quantization=quantization)
    data = {}
    data["label"] = np.array([[1]])
    asr_caffe.generate_prediction_net(data)

def transform_joint_net(model_path_prefix, data_path_prefix, quantization):
    tensorflow_model_path = model_path_prefix + "/joint_net.pb"
    caffe_model_path_prefix = "asr_convolution_transformer_joint_net"
    caffe_model_name = "asr_convolution_transformer_joint_net"
    asr_caffe = Tensorflow2CaffeConvolutionTransformer(base_params,
                       tensorflow_model_path, caffe_model_path_prefix, caffe_model_name,
                       check=False, calc=True, quantization=quantization)
    shapes = {}
    shapes["encoder"] = [1, 512]
    shapes["prediction_net"] = [1, 512]
    data = {}
    data["encoder"] = np.load(data_path_prefix + "/encoder.npy")
    data["prediction_net"] = np.load(data_path_prefix + "/pred_net.npy")
    asr_caffe.print_weight_map()
    asr_caffe.generate_joint_net(shapes, data)

if __name__ == '__main__':
    model_path_prefix = "/data/bolt/model_zoo/tensorflow_models/OpenSeq2Seq/pipeline/model/transducer"
    data_path_prefix = "/data/bolt/model_zoo/tensorflow_models/OpenSeq2Seq"

    quantization = False
    FFN_decomposition = False
    transform_encoder(model_path_prefix, data_path_prefix, quantization, FFN_decomposition)
    transform_prediction_net(model_path_prefix, quantization)
    transform_joint_net(model_path_prefix, data_path_prefix, quantization)
