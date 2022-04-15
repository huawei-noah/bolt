#!/usr/local/bin/python
# -*- coding: utf-8 -*-

from tensorflow2caffe_tactron2_duration import Tensorflow2CaffeTactron2Duration
import numpy as np

def post_encoder(encoder, speaker, duration):
    seq_len = duration.shape[-1]
    encoder_dim = encoder.shape[-1]
    speaker_dim = speaker.shape[-1]
    y = np.zeros([1, int(duration.sum()) + 2, encoder_dim + 1 + speaker_dim])
    y[0, -1, encoder_dim + 1:] = speaker[0][0]
    j = 0
    s = 0
    for i in range(seq_len):
        s = s + duration[0][i]
        if (s >= int(s)):
            upper = int(s) + 1
        else:
            upper = int(s)
        k = 0.1
        for a in range(j, upper):
            y[0, j, :encoder_dim] = encoder[0][i]
            y[0][j][encoder_dim] = k
            k = k + 0.1
            y[0, j, encoder_dim + 1:] = speaker[0][0]
            j = j + 1
    return y

def text_to_speech(tensorflow_model_path):
    params = Tensorflow2CaffeTactron2Duration.Parameters()
    tts_caffe = Tensorflow2CaffeTactron2Duration(tensorflow_model_path, "tts_encoder_duration", "tts_encoder_duration",
                       params,
                       check=False, calc=True)
    tts_caffe.print_weight_map()
    data = {}
    data["words"] = np.array([[0, 212, 110, 149, 3, 96, 154, 150, 8, 93, 170, 3, 163, 168, 7, 212, 111, 150,
        3, 96, 154, 150, 4, 194, 7, 93, 168, 3, 166, 86, 7, 165, 81, 149, 3, 95,
        79, 150, 7, 148, 86, 3, 148, 98, 150, 4, 1]]);
    data["duration_masks"] = np.array([[1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1]])
    data["speaker"] = np.array([[0]]);
    encoder, speaker, duration = tts_caffe.generate_encoder(data)
    a = post_encoder(encoder, speaker, duration)
    np.save("encoder.npy", a)

    tts_caffe = Tensorflow2CaffeTactron2Duration(tensorflow_model_path, "tts_decoder", "tts_decoder",
                       params,
                       check=False, calc=True)
    a = np.load("encoder.npy")
    decoder_length = a.shape[1]
    data = {}
    data["encoder"] = a
    data["decoder_length"] = np.array([[decoder_length - 1]])
    a, _ = tts_caffe.generate_decoder(data)
    np.save("decoder.npy", a)

    tts_caffe = Tensorflow2CaffeTactron2Duration(tensorflow_model_path, "tts_postnet", "tts_postnet",
                       params,
                       check=False, calc=True)
    a = np.load("decoder.npy")[:, :decoder_length * params.outputs_per_step, :]
    data = {}
    data["decoder"] = a
    tts_caffe.generate_postnet(data)

if __name__ == '__main__':
    tensorflow_model_path = "/data/bolt/model_zoo/tensorflow_models/tactron_duration/tacotron_model.ckpt-457000"
    text_to_speech(tensorflow_model_path)
