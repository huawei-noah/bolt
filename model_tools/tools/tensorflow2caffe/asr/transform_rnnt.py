#!/usr/local/bin/python
# -*- coding: utf-8 -*-

from tensorflow2caffe_rnnt import Tensorflow2CaffeRNNT
import numpy as np

def transform_rnnt(mode):
    tensorflow_model_path = "/data/bolt/model_zoo/tensorflow_models/rnnt_spm2048_ms/model.ckpt-424000"
    params = Tensorflow2CaffeRNNT.default_params()
    if mode == "all":
        caffe_model_path_prefix = "asr_rnnt"
        caffe_model_name = "asr_rnnt"
        asr_caffe = Tensorflow2CaffeRNNT(tensorflow_model_path, caffe_model_path_prefix, caffe_model_name,
                           params,
                           check=False, calc=True)
        data = {}
        input = np.loadtxt('/data/bolt/model_zoo/tensorflow_models/rnnt_spm2048_ms/sound_data_0.txt')
        data["sounds"] = np.reshape(input, [1, -1, params['sequence.num_units']])
        asr_caffe.generate(data)
    else:
        asr_caffe = Tensorflow2CaffeRNNT(tensorflow_model_path, "asr_rnnt_encoder", "asr_rnnt_encoder",
                           params,
                           check=False, calc=False)
        asr_caffe.generate_encoder({})
        asr_caffe = Tensorflow2CaffeRNNT(tensorflow_model_path, "asr_rnnt_prediction_net", "asr_rnnt_prediction_net",
                           params,
                           check=False, calc=False)
        asr_caffe.generate_prediction_net({})
        asr_caffe = Tensorflow2CaffeRNNT(tensorflow_model_path, "asr_rnnt_joint_net", "asr_rnnt_joint_net",
                           params,
                           check=False, calc=False)
        asr_caffe.generate_joint_net({})

if __name__ == '__main__':
    transform_rnnt("all")
    transform_rnnt("sub")
