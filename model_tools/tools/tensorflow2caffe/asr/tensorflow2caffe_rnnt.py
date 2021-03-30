#!/usr/local/bin/python
# -*- coding: utf-8 -*-

import math
import numpy as np
import sys
sys.path.append("../")
from tensorflow2caffe import Tensorflow2Caffe

class Tensorflow2CaffeRNNT(Tensorflow2Caffe):
    def __init__(self,
            tensorflow_model_path, caffe_model_path_prefix, caffe_model_name,
            params,
            check=False, calc=False):
        Tensorflow2Caffe.__init__(self, tensorflow_model_path,
            caffe_model_path_prefix, caffe_model_name, check, calc)
        self.params = params
        self.check_params()

    @staticmethod
    def default_params():
        return {
            "BLANK": 0,

            "sequence.max_length": 128,
            "sequence.num_units": 240,

            "encoder.lstm_cells": 6,
            "encoder.lstm_cell_type": "LSTMP",
            "encoder.lstm_cell_state_shape": 1664,
            "encoder.use_layer_normed_fc": True,
            "encoder.activation": "relu",
            "encoder.num_output": 640,

            "prediction_net.lstm_cells": 2,
            "prediction_net.lstm_cell_type": "LSTMP",
            "prediction_net.lstm_cell_state_shape": 1664,
            "prediction_net.use_layer_normed_fc": True,
            "prediction_net.activation": "relu",
            "prediction_net.num_output": 640,

            "output.activation": None,
        }

    def check_params(self):
        return
        if (self.params["sequence.num_units"] != self.params["encoder.lstm_cell_input_shape"]):
            print("[ERROR] sequence.num_units(%d) must equal encoder.lstm_cell_input_shape(%d)" 
                % (self.params["sequence.num_units"], self.params["encoder.lstm_cell_input_shape"]))
            exit(1)


    def layer_normed_fc(self, input_name, activation_fn, output_name_prefix, scope_id):
        fc_name = output_name_prefix + "_fc"
        self.extract_dense(input_name, fc_name, scope_id, scope_name="fully_connected")
        ln_name = output_name_prefix + "_ln"
        self.extract_layer_norm(fc_name, ln_name, scope_id, ["LayerNorm", "gamma", "beta"])
        result_name = ""
        if (activation_fn == "relu"):
            relu_name = output_name_prefix + "_relu"
            result_name = self.add_relu(ln_name, relu_name)
        else:
            print("[ERROR] unsupported activation function" % (activation_fn))
            exit(1)
        return result_name;

    def extract_encoder(self, input_name, states, scope_id):
        self.scopes[scope_id] = "encoder"

        self.scopes[scope_id+1] = "rnn"
        self.scopes[scope_id+2] = "multi_rnn_cell"
        last_input_name = input_name
        for layer_idx in range(self.params["encoder.lstm_cells"]):
            if (self.params["encoder.lstm_cell_type"] == "LSTMP"):
                self.scopes[scope_id+3] = "cell_" + str(layer_idx)
                self.scopes[scope_id+4] = "lstm_cell"
                lstm_output_name = "encoder_lstm" + str(layer_idx) + "_cell"
                self.extract_rnn("LSTM", last_input_name, states[layer_idx], lstm_output_name, scope_id+4, scope_name = "lstm_cell", use_proj=True)
                last_input_name = lstm_output_name
            else:
                print("[ERROR] unsupported lstm type %s" % (self.params["encoder.lstm_cell_type"]))
                exit(1)
        if (self.params["encoder.use_layer_normed_fc"]):
            result_name = self.layer_normed_fc(last_input_name, self.params["encoder.activation"], "encoder", scope_id+1)
        else:
            result_name = last_input_name
        return result_name

    def extract_prediction_net(self, input_name, states, scope_id):
        self.scopes[scope_id] = "prediction_net"

        eb_name = "prediction_net_embedding"
        self.scopes[scope_id+1] = "embedding"
        self.extract_embedding(input_name, scope_id+2, "embedding", eb_name)

        squeeze_name = eb_name + "_squeeze"
        self.add_squeeze(eb_name, axis=1, output_name=squeeze_name)

        self.scopes[scope_id+1] = "rnn"
        self.scopes[scope_id+2] = "multi_rnn_cell"
        last_input_name = squeeze_name
        for layer_idx in range(self.params["prediction_net.lstm_cells"]):
            if (self.params["prediction_net.lstm_cell_type"] == "LSTMP"):
                self.scopes[scope_id+3] = "cell_" + str(layer_idx)
                self.scopes[scope_id+4] = "lstm_cell"
                lstm_output_name = "prediction_net_lstm" + str(layer_idx) + "_cell"
                self.extract_rnn("LSTM", last_input_name, states[layer_idx], lstm_output_name, scope_id+4, scope_name = "lstm_cell", use_proj=True)
                last_input_name = lstm_output_name
            else:
                print("[ERROR] unsupported lstm type %s" % (self.params["prediction_net.lstm_cell_type"]))
                exit(1)

        if (self.params["prediction_net.use_layer_normed_fc"]):
            result_name = self.layer_normed_fc(last_input_name, self.params["prediction_net.activation"], "prediction_net", scope_id+1)
        else:
            result_name = last_input_name
        return result_name

    def extract_joint_net(self, input_names, scope_id):
        self.scopes[scope_id] = "joint_net"

        fc0_name = "joint_net_fc"
        self.extract_dense(input_names[0], fc0_name, scope_id+1, scope_name = "dense")
        #ep0_name = "joint_net_expand0"
        #self.add_expand_dims(fc0_name, axis=2, output_name=ep0_name)
        fc1_name = "joint_net_fc_1"
        self.extract_dense(input_names[1], fc1_name, scope_id+1, scope_name = "dense_1")
        #ep1_name = "joint_net_expand1"
        #self.add_expand_dims(fc1_name, axis=1, output_name=ep1_name)
        sum_name = "joint_net_sum"
        #self.add_sum([ep0_name, ep1_name], sum_name)
        self.add_sum([fc0_name, fc1_name], sum_name)
        tanh_name = "joint_net_tanh"
        self.add_tanh(sum_name, tanh_name)
        return tanh_name

    def extract_output(self, input_name, scope_id):
        self.scopes[scope_id] = "rnnt_output"

        fc_name = "output_fc"
        self.extract_dense(input_name, fc_name, scope_id+1, scope_name = "rnnt_output_fc")

        activation_name = ""
        if (self.params["output.activation"] is not None):
            print("[ERROR] unsupported activation function" % (self.params["output.activation"]))
            exit(1)
        else:
            activation_name = fc_name
        argmax_name = self.add_argmax(activation_name, axis=-1, output_name="output_argmax")
        return argmax_name

    def generate_encoder(self, input=None):
        encoder_input_name = "sounds"
        encoder_input_shape = [self.batch, self.params["sequence.num_units"]]
        self.add_input(encoder_input_name, encoder_input_shape)
        self.set_input(input)
        self.scopes[0] = "rnnt"
        encoder_states = []
        for layer_idx in range(self.params["encoder.lstm_cells"]):
            state_shape = [self.batch, self.params["encoder.lstm_cell_state_shape"]]
            state_name = "encoder_lstm" + str(layer_idx) + "_state"
            self.add_input(state_name, state_shape)
            encoder_states.append(state_name)
        encoder_output = self.extract_encoder(encoder_input_name, encoder_states, 1)
        self.save_caffe_model()

    def generate_prediction_net(self, input=None):
        prediction_net_input_name = "prediction_net_input"
        prediction_net_input_shape = [self.batch, 1]
        self.add_input(prediction_net_input_name, prediction_net_input_shape)
        self.set_input(input)
        self.scopes[0] = "rnnt"
        prediction_net_states = []
        for layer_idx in range(self.params["prediction_net.lstm_cells"]):
            state_shape = [self.batch, self.params["prediction_net.lstm_cell_state_shape"]]
            state_name = "prediction_net_lstm" + str(layer_idx) + "_state"
            self.add_input(state_name, state_shape)
            prediction_net_states.append(state_name)
        prediction_net_output = self.extract_prediction_net(prediction_net_input_name, prediction_net_states, 1)
        self.save_caffe_model()

    def generate_joint_net(self, input=None):
        encoder_output = "encoder"
        encoder_output_shape = [self.batch, self.params["encoder.num_output"]]
        prediction_net_output = "prediction_net"
        prediction_net_output_shape = [self.batch, self.params["prediction_net.num_output"]]
        self.add_input(encoder_output, encoder_output_shape)
        self.add_input(prediction_net_output, prediction_net_output_shape)
        self.set_input(input)
        self.scopes[0] = "rnnt"
        joint_net_output = self.extract_joint_net([encoder_output, prediction_net_output], 1)
        label_output = self.extract_output(joint_net_output, 0)
        self.save_caffe_model()

    def generate(self, input=None):
        sounds_input_name = "sounds"
        sounds_input_shape = [self.batch, self.params["sequence.max_length"], self.params["sequence.num_units"]]
        self.add_input(sounds_input_name, sounds_input_shape)
        self.set_input(input)

        labels_output_name = "labels"
        labels_output_shape = [self.batch, self.params["sequence.max_length"]]
        self.add_memory(labels_output_name, labels_output_shape, data_type="INT32")

        position_input_name = "position"
        position_input_shape = [self.batch, 1]
        self.add_memory(position_input_name, position_input_shape, data_type="INT32")

        encoder_input_name = "encoder_input"
        encoder_input_shape = [self.batch, self.params["sequence.num_units"]]
        self.add_memory(encoder_input_name, encoder_input_shape, data_type="FLOAT32")

        prediction_net_input_name = "prediction_net_input"
        prediction_net_input_shape = [self.batch, 1]
        self.add_memory(prediction_net_input_name, prediction_net_input_shape, data_type="INT32")

        prediction_net_status_name = "prediction_net_status"
        prediction_net_status_shape = [self.batch, 1]
        self.add_memory(prediction_net_status_name, prediction_net_status_shape, data_type="INT32")

        blank = "BLANK"
        weight = np.array([[self.params["BLANK"]] * self.batch])
        self.add_weight(blank, weight=weight, data_type="INT32")
        negative_one = "negative_one"
        weight = np.array([[-1] * self.batch])
        self.add_weight(negative_one, weight=weight, data_type="INT32")
        zero = "zero"
        weight = np.array([[0]*self.batch], dtype=int)
        self.add_weight(zero, weight=weight, data_type="INT32")

        # init position
        self.add_copy(negative_one, 1, 1, 0,
                      position_input_name, 1, 1, 0,
                      1, output_name="init_position")

        # init prediction_net input
        self.add_copy(blank, 1, 1, 0,
                      prediction_net_input_name, 1, 1, 0,
                      1, output_name="init_prediction_net_input")

        # init prediction_net status
        self.add_copy(zero, 1, 1, 0,
                      prediction_net_status_name, 1, 1, 0,
                      1, output_name="init_prediction_net_status")

        encoder_states = []
        for layer_idx in range(self.params["encoder.lstm_cells"]):
            state_shape = [self.batch, self.params["encoder.lstm_cell_state_shape"]]
            state_name = "encoder_lstm" + str(layer_idx) + "_state"
            self.add_memory(state_name, state_shape, data_type="FLOAT32")
            encoder_states.append(state_name)
        prediction_net_states = []
        for layer_idx in range(self.params["prediction_net.lstm_cells"]):
            state_shape = [self.batch, self.params["prediction_net.lstm_cell_state_shape"]]
            state_name = "prediction_net_lstm" + str(layer_idx) + "_state"
            self.add_memory(state_name, state_shape, data_type="FLOAT32")
            prediction_net_states.append(state_name)

        sequence_length = 1
        if (input is not None):
            sequence_length = self.get_tensor(sounds_input_name).shape[-2]

        copy_sound_name = "copy_to_encoder_input"
        repeat_name = "loops"
        prediction_net_output = ""
        self.add_jump(repeat_name, "jump_to_repeat", prediction_net_status_name)
        for step in range(sequence_length):
            self.scopes[0] = "rnnt"
            self.set_add_layer(step==0)

            position_input_name_new = position_input_name+"_add_one"
            self.add_power(position_input_name, position_input_name_new, scale=1, shift=1, power=1)
            self.add_copy(position_input_name_new, 1, 1, 0,
                          position_input_name, 1, 1, 0,
                          1, output_name="update_position")

            jump_name = "skip_blank"
            self.add_jump(copy_sound_name, jump_name, prediction_net_status_name)
            if (isinstance(self.get_tensor(prediction_net_status_name)[0][0], bool) and not self.get_tensor(prediction_net_status_name)[0][0]) \
            or (isinstance(self.get_tensor(prediction_net_status_name)[0][0], int) and self.get_tensor(prediction_net_status_name)[0][0] == 0) \
            or (isinstance(self.get_tensor(prediction_net_status_name)[0][0], float) and self.get_tensor(prediction_net_status_name)[0][0] == 0):
                prediction_net_output = self.extract_prediction_net(prediction_net_input_name, prediction_net_states, 1)

            self.add_copy(sounds_input_name,
                          self.params["sequence.max_length"]*self.params["sequence.num_units"], self.params["sequence.num_units"], 0,
                          encoder_input_name,
                          self.params["sequence.num_units"], self.params["sequence.num_units"], 0,
                          self.params["sequence.num_units"],
                          output_name=copy_sound_name,
                          src_index_name=position_input_name,
                          dst_index_name=zero)
            encoder_output = self.extract_encoder(encoder_input_name, encoder_states, 1)

            joint_net_output = self.extract_joint_net([encoder_output, prediction_net_output], 1)

            label_output = self.extract_output(joint_net_output, 0)
            self.add_copy(label_output, 1, 1, 0,
                          prediction_net_input_name, 1, 1, 0,
                          1, output_name="copy_to_prediction_net_input")
            self.add_copy(label_output,
                          1, 1, 0,
                          labels_output_name,
                          1, 1, 0,
                          1,
                          output_name="copy_to_global_labels",
                          src_index_name=zero,
                          dst_index_name=position_input_name)
            status_name = "check_label_is_blank"
            self.add_check(blank, label_output, "equal", status_name)
            # set prediction_net status
            self.add_copy(status_name, 1, 1, 0,
                          prediction_net_status_name, 1, 1, 0,
                          1, output_name="set_prediction_net_status")
            self.add_repeat(self.params["sequence.max_length"]-1, position_input_name_new,
                            output_name=repeat_name,
                            status_name=zero,
                            axis_name=sounds_input_name, axis=1)

        self.save_caffe_model()
