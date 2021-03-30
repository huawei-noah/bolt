#!/usr/local/bin/python
# -*- coding: utf-8 -*-

from tensorflow2caffe_convolution_transformer import Tensorflow2CaffeConvolutionTransformer
import numpy as np

def transform_encoder(model_path_prefix, data_path_prefix, quantization, block_id_start, block_id_end):
    tensorflow_model_path = model_path_prefix + "/encoder.pb"
    caffe_model_path_prefix = "asr_convolution_transformer_encoder"
    caffe_model_name = "asr_convolution_transformer_encoder"
    first_frame = True
    asr_caffe = Tensorflow2CaffeConvolutionTransformer(tensorflow_model_path, caffe_model_path_prefix, caffe_model_name,
                       nchwc8=True, first_frame=first_frame,
                       check=False, calc=True, quantization=quantization)
    data = {}
    if (first_frame):
        data["sounds"] = np.load(data_path_prefix + "/sound0_frame0.npy")
    else:
        data["sounds"] = np.load(data_path_prefix + "/sound0_frame1.npy")
    asr_caffe.generate_encoder(data, block_id_start, block_id_end)


def transform_prediction_net(model_path_prefix, quantization):
    tensorflow_model_path = model_path_prefix + "/pred_net.pb"
    caffe_model_path_prefix = "asr_convolution_transformer_prediction_net"
    caffe_model_name = "asr_convolution_transformer_prediction_net"
    asr_caffe = Tensorflow2CaffeConvolutionTransformer(tensorflow_model_path, caffe_model_path_prefix, caffe_model_name,
                       check=False, calc=True, quantization=quantization)
    data = {}
    data["label"] = np.array([[1]])
    asr_caffe.generate_prediction_net(data)

def transform_joint_net(model_path_prefix, data_path_prefix, quantization):
    tensorflow_model_path = model_path_prefix + "/joint_net.pb"
    caffe_model_path_prefix = "asr_convolution_transformer_joint_net"
    caffe_model_name = "asr_convolution_transformer_joint_net"
    asr_caffe = Tensorflow2CaffeConvolutionTransformer(tensorflow_model_path, caffe_model_path_prefix, caffe_model_name,
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
    transform_encoder(model_path_prefix, data_path_prefix, quantization, 0, -1)
    transform_prediction_net(model_path_prefix, quantization)
    transform_joint_net(model_path_prefix, data_path_prefix, quantization)
