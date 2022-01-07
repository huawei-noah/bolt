#!/usr/local/bin/python
# -*- coding: utf-8 -*-

from tensorflow2caffe_punctuation import Tensorflow2CaffePunctuation
import numpy as np


if __name__ == '__main__':
    tensorflow_model_path = "bilstm.pb"
    caffe_model_path_prefix = "punctuation"
    caffe_model_name = "punctuation"

    bilstm = Tensorflow2CaffePunctuation(tensorflow_model_path, caffe_model_path_prefix, caffe_model_name,
                       False, True)
    data = {}
    data["input_text"] = np.array([[6, 13, 5, 14, 31, 234, 325, 161, 5, 182, 180, 266, 31, 234, 460, 62]])
    bilstm.print_weight_map()
    bilstm.generate(data)