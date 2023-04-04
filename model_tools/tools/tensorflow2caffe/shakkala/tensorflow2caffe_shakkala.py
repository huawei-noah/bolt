#!/usr/local/bin/python
# -*- coding: utf-8 -*-

import sys
sys.path.append("../")
from tensorflow2caffe import Tensorflow2Caffe

class Tensorflow2CaffeShakkala(Tensorflow2Caffe):
    def __init__(self,
            tensorflow_model_path, caffe_model_path_prefix, caffe_model_name, max_seq_length,
            check=False, calc=False):
        Tensorflow2Caffe.__init__(self, tensorflow_model_path,
            caffe_model_path_prefix, caffe_model_name, check, calc)
        self.max_seq_length = max_seq_length

    def generate(self, input=None):
        input_name = "input"
        input_shape = [self.batch, self.max_seq_length]
        self.add_input(input_name, input_shape)
        self.set_input(input)

        x = self.extract_embedding(input_name, 0, "embedding_7/embedding_7_1/embeddings", "embed")
        self.scopes[0] = "bidirectional_19/bidirectional_19_1"
        x = self.extract_rnn("LSTM", x, None, "bilstm0", 1,
            steps=-2, scope_name=["forward_lstm_19", "backward_lstm_19"])

        x = self.extract_batch_norm(x, "bn", 0,
            data_format="NCHW",
            axis=-1, eps=1e-3,
            layer_names=["batch_normalization_13/batch_normalization_13_1", "moving_mean", "moving_variance"])

        self.scopes[0] = "bidirectional_20/bidirectional_20_1"
        x = self.extract_rnn("LSTM", x, None, "bilstm1", 1,
            steps=-2, scope_name=["forward_lstm_20", "backward_lstm_20"])
        self.scopes[0] = "bidirectional_21/bidirectional_21_1"
        x = self.extract_rnn("LSTM", x, None, "bilstm2", 1,
            steps=-2, scope_name=["forward_lstm_21", "backward_lstm_21"])
        x = self.extract_dense(x, "dense", 0, scope_name="dense_7/dense_7_1")
        x = self.add_softmax(x, "output", axis=-1)

        self.save_caffe_model()
