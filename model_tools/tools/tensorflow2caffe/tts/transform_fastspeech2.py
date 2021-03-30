#!/usr/local/bin/python
# -*- coding: utf-8 -*-

from tensorflow2caffe_fastspeech2 import Tensorflow2CaffeFastSpeech2
import yaml
import numpy as np

def text_to_speech(tensorflow_model_path, config_path):
    f = open(config_path, 'r', encoding='utf-8')
    params = yaml.load(f.read())["fastspeech2_params"]
    f.close()
    tts_caffe = Tensorflow2CaffeFastSpeech2(tensorflow_model_path, "tts_encoder", "tts_encoder",
                       params,
                       check=False, calc=True)
    data = {}
    input_length = 12
    data["input_ids"] = np.array([[4, 11, 75, 6, 90, 86, 6, 69, 5, 61, 47, 1]])
    data["position_ids"] = np.array([[i for i in range(1, input_length+1)]])
    data["speaker_ids"] = np.array([[0]])
    data["speed_ratios"] = np.array([[1]])
    data["f0_ratios"] = np.array([[1]])
    data["energy_ratios"] = np.array([[1]])
    encoder_output_name, duration_output_name = tts_caffe.generate_encoder(data)
    encoder_output = tts_caffe.get_tensor(encoder_output_name)
    duration_output = tts_caffe.get_tensor(duration_output_name)

    tts_caffe = Tensorflow2CaffeFastSpeech2(tensorflow_model_path, "tts_decoder", "tts_decoder",
                       params,
                       check=False, calc=True)
    data = {}
    decoder_input = np.repeat(np.array([[i for i in range(0, input_length)]]), np.ceil(duration_output[0]).astype(int), axis=1)
    data["input_ids"] = decoder_input
    data["position_ids"] = np.array([[i for i in range(1, decoder_input.shape[1]+1)]])
    data["features"] = encoder_output
    tts_caffe.generate_decoder(data)

if __name__ == '__main__':
    tensorflow_model_path = "/data/bolt/model_zoo/tensorflow_models/fastspeech2/fs2-latest.pb"
    config_path = "/data/bolt/model_zoo/tensorflow_models/fastspeech2/fastspeech2.v1.yaml"
    text_to_speech(tensorflow_model_path, config_path)
