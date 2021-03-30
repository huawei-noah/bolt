#!/usr/local/bin/python
# -*- coding: utf-8 -*-

from tensorflow2caffe_tactron2 import Tensorflow2CaffeTactron2
import os
import numpy as np

def text_to_speech(tensorflow_model_path):
    params = Tensorflow2CaffeTactron2.Parameters()
    tts_caffe = Tensorflow2CaffeTactron2(tensorflow_model_path, "tts_encoder_decoder", "tts_encoder_decoder",
                       params,
                       check=False, calc=True)

    data = {}
    data["tts_words"] = np.array([[4, 25, 14, 33, 11, 20, 1, 9, 14, 33, 27, 2, 20, 35, 15, 1, 10, 37, 11, 2, 30,
        34, 15, 7, 21, 1, 25, 14, 35, 21, 27, 3, 25, 14, 34, 27, 1, 25, 14, 35, 27, 1, 17, 36, 7, 20, 1, 37, 7, 0]])
    data["tts_alignments"] = np.zeros(data["tts_words"].shape)
    decoder_result, num = tts_caffe.generate_encoder_decoder(data)
    os.system('mv input_shape.txt encoder_decoder_input_shape.txt')

    tts_caffe = Tensorflow2CaffeTactron2(tensorflow_model_path, "tts_postnet", "tts_postnet",
                       params,
                       check=False, calc=True)
    data = {}
    data["tts_decoder"] = decoder_result[:, :int(num+1)*params.outputs_per_step, :]
    tts_caffe.generate_postnet(data)
    os.system('mv input_shape.txt postnet_input_shape.txt')

if __name__ == '__main__':
    tensorflow_model_path = "/data/models/tts/taco_pretrained-290000/tacotron_model.ckpt-290000"
    text_to_speech(tensorflow_model_path)
