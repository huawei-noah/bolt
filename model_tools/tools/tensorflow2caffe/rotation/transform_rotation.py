#!/usr/local/bin/python
# -*- coding: utf-8 -*-

from tensorflow2caffe_rotation import Tensorflow2CaffeRotation
import numpy as np


if __name__ == '__main__':
    tensorflow_model_path = "/data/models/rotation/20200617LSTM_2convfront_209-99.970848083496.pb"
    caffe_model_path_prefix = "rotation"
    caffe_model_name = "rotation"

    rotation = Tensorflow2CaffeRotation(tensorflow_model_path, caffe_model_path_prefix, caffe_model_name,
                       False, True)
    data = {}
    data["input"] = np.ones([1,3,30,1])
    rotation.print_weight_map()
    rotation.generate(data)
