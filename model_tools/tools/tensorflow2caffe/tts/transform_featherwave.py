#!/usr/local/bin/python
# -*- coding: utf-8 -*-

from tensorflow2caffe_featherwave import Tensorflow2CaffeFeatherWave
import numpy as np

def featherwave(tensorflow_model_path):
    tts_caffe = Tensorflow2CaffeFeatherWave(tensorflow_model_path, "tts_encoder", "tts_encoder",
                       check=False, calc=True)
    data = {}
    data["input_1"] = np.array([[4, 11, 75, 6, 90, 86, 6, 69]])
    data["input_2"] = np.array([[4, 11, 75, 6]])
    data["input_3"] = np.array([[1]*256])
    tts_caffe.generate(data)

if __name__ == '__main__':
    tensorflow_model_path = "/data/home/yuxianzhi/model_213.h5"
    featherwave(tensorflow_model_path)
    #print_all(tensorflow_model_path)
