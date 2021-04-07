#!/usr/local/bin/python
# -*- coding: utf-8 -*-

from tensorflow2caffe_convolution_transformer_keras import Tensorflow2CaffeConvolutionTransformerKeras
import numpy as np

def transform_encoder(model_path_prefix):
    tensorflow_model_path = model_path_prefix + "/transformer_t_db_s8_r33_v1_i160_20200330_encoder.pb"
    caffe_model_path_prefix = "asr_convolution_transformer_encoder"
    caffe_model_name = "asr_convolution_transformer_encoder"
    params = Tensorflow2CaffeConvolutionTransformerKeras.default_params()
    asr_caffe = Tensorflow2CaffeConvolutionTransformerKeras(tensorflow_model_path, caffe_model_path_prefix, caffe_model_name,
                       params,
                       check=False, calc=True)
    shapes = {}
    shapes["encoder"] = [1, 32, 41, 1]
    data = {}
    data["encoder"] = np.ones(shapes["encoder"])
    asr_caffe.generate_encoder(shapes, data)

def transform_prediction_joint_net(model_path_prefix):
    tensorflow_model_path = model_path_prefix + "/transformer_t_db_s8_r33_v1_i160_20200330_decoder_joint.pb"
    caffe_model_path_prefix = "asr_convolution_transformer_joint_net"
    caffe_model_name = "asr_convolution_transformer_joint_net"
    params = Tensorflow2CaffeConvolutionTransformerKeras.default_params()
    asr_caffe = Tensorflow2CaffeConvolutionTransformerKeras(tensorflow_model_path, caffe_model_path_prefix, caffe_model_name,
                       params,
                       check=False, calc=True)
    shapes = {}
    shapes["prediction_net"] = [1, 1, 512]
    shapes["encoder"] = [1, 1, 512]
    data = {}
    data["prediction_net"] = np.ones(shapes["prediction_net"])
    data["encoder"] = np.ones(shapes["encoder"])
    asr_caffe.print_weight_map()
    asr_caffe.generate_prediction_joint_net(shapes, data)

if __name__ == '__main__':
    model_path_prefix = "/data/models/asr/transformer_t_db_s8_r33_v1_i160_20200330153153"

    transform_encoder(model_path_prefix)
    transform_prediction_joint_net(model_path_prefix)
