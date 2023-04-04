#!/usr/local/bin/python
# -*- coding: utf-8 -*-

from tensorflow2caffe_shakkala import Tensorflow2CaffeShakkala
import numpy as np


if __name__ == '__main__':
    tensorflow_model_path = "second_model6.h5"
    caffe_model_path_prefix = "shakkala"
    caffe_model_name = "shakkala"

    rotation = Tensorflow2CaffeShakkala(tensorflow_model_path, caffe_model_path_prefix, caffe_model_name,
                        315,
                       False, True)
    data = {}
    data["input"] = np.ones([1,315])
    rotation.generate(data)
