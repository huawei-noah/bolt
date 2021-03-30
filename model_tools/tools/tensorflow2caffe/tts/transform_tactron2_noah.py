#!/usr/local/bin/python
# -*- coding: utf-8 -*-

from tensorflow2caffe_tactron2_noah import Tensorflow2CaffeTactron2
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
    data["tts_emotions"] = np.array([[4]*data["tts_words"].size])
    decoder_result, num = tts_caffe.generate_encoder_decoder(data)
    os.system('mv input_shape.txt encoder_decoder_input_shape.txt')

    tts_caffe = Tensorflow2CaffeTactron2(tensorflow_model_path, "tts_postnet", "tts_postnet",
                       params,
                       check=False, calc=True)
    data = {}
    data["tts_decoder"] = decoder_result[:, :int(num+1)*params.outputs_per_step, :]
    tts_caffe.generate_postnet(data)
    os.system('mv input_shape.txt postnet_input_shape.txt')


def genrate_streaming_lstm_states(layers, num_units, output_name_prefix):
    states = {}
    for i in range(layers):
        state_shape = [1, num_units]
        state_name = output_name_prefix + "_layer" + str(i) + "_state"
        states[state_name] = np.zeros(state_shape)
    return states

def text_to_speech_streaming(tensorflow_model_path):
    params = Tensorflow2CaffeTactron2.Parameters()
    params.streaming = True
    params.max_iters = 12
    tts_caffe = Tensorflow2CaffeTactron2(tensorflow_model_path, "tts_encoder", "tts_encoder",
                       params,
                       check=False, calc=True)

    data = {}
    data["tts_words"] = np.array([[4, 25, 14, 33, 11, 20, 1, 9, 14, 33, 27, 2, 20, 35, 15, 1, 10, 37, 11, 2, 30,
        34, 15, 7, 21, 1, 25, 14, 35, 21, 27, 3, 25, 14, 34, 27, 1, 25, 14, 35, 27, 1, 17, 36, 7, 20, 1, 37, 7, 0]])
    data["tts_emotions"] = np.array([[4]*data["tts_words"].size])
    rnn_result = tts_caffe.generate_encoder(data)
    rnn_result_data = tts_caffe.get_tensor(rnn_result)
    os.system('mv input_shape.txt encoder_input_shape.txt')

    tts_caffe = Tensorflow2CaffeTactron2(tensorflow_model_path, "tts_decoder", "tts_decoder",
                       params,
                       check=False, calc=True)
    tts_caffe.print_weight_map()
    data = {}
    data[rnn_result] = rnn_result_data
    data["tts_alignments"] = np.zeros([1, rnn_result_data.shape[1]])
    data["decoder_input"] = np.zeros([1, params.num_mels])
    data["decoder_attention"] = np.zeros([1, rnn_result_data.shape[2]])
    data.update(genrate_streaming_lstm_states(1, params.decoder_lstm_units*2, "decoder_query"))
    data.update(genrate_streaming_lstm_states(params.decoder_layers, params.decoder_lstm_units*2, "decoder_lstm"))
    tts_caffe.generate_decoder(rnn_result, data)
    os.system('mv input_shape.txt decoder_input_shape.txt')

if __name__ == '__main__':
    tensorflow_model_path = "/data/bolt/model_zoo/tensorflow_models/taco_pretrained-290000/tacotron_model.ckpt-290000"
    #text_to_speech(tensorflow_model_path)
    text_to_speech_streaming(tensorflow_model_path)

