#!/usr/local/bin/python
# -*- coding: utf-8 -*-

import math
import sys
sys.path.append("../")
from tensorflow2caffe import Tensorflow2Caffe

class Tensorflow2CaffeFeatherWave(Tensorflow2Caffe):
    def __init__(self,
            tensorflow_model_path, caffe_model_path_prefix, caffe_model_name,
            check=False, calc=False):
        Tensorflow2Caffe.__init__(self, tensorflow_model_path,
            caffe_model_path_prefix, caffe_model_name, check, calc)
        self.state_length = self.weight_map["gru/gru/R_e"].shape[0]
        # split dual gru to two grus
        length = self.weight_map["gru/gru/R_e"].shape[-1] // 2
        array1 = ["u", "r", "e"]
        for id1 in array1:
            #self.weight_map["gru/gru/R_{}_coarse".format(id1)] = self.weight_map["gru/gru/R_{}".format(id1)][:, :length]
            #self.weight_map["gru/gru/R_{}_fine".format(id1)] = self.weight_map["gru/gru/R_{}".format(id1)][:, length:]
            #del self.weight_map["gru/gru/R_{}".format(id1)]
            self.weight_map["gru/gru/bias_{}_coarse".format(id1)] = self.weight_map["gru/gru/bias_{}".format(id1)][:length]
            self.weight_map["gru/gru/bias_{}_fine".format(id1)] = self.weight_map["gru/gru/bias_{}".format(id1)][length:]
            del self.weight_map["gru/gru/bias_{}".format(id1)]
        #self.concat_weight(["gru/gru/R_u_coarse", "gru/gru/R_r_coarse", "gru/gru/R_e_coarse"], "gru/gru/R_coarse", axis=1)
        #self.concat_weight(["gru/gru/I_u_coarse", "gru/gru/I_r_coarse", "gru/gru/I_e_coarse"], "gru/gru/I_coarse", axis=1)
        #self.concat_weight(["gru/gru/I_coarse", "gru/gru/R_coarse"], "gru/gru/coarse/kernel", axis=0)
        #self.concat_weight(["gru/gru/bias_u_coarse", "gru/gru/bias_r_coarse", "gru/gru/bias_e_coarse"], "gru/gru/coarse/bias", axis=0)
        #self.concat_weight(["gru/gru/R_u_fine", "gru/gru/R_r_fine", "gru/gru/R_e_fine"], "gru/gru/R_fine", axis=1)
        #self.concat_weight(["gru/gru/I_u_fine", "gru/gru/I_r_fine", "gru/gru/I_e_fine"], "gru/gru/I_fine", axis=1)
        #self.concat_weight(["gru/gru/I_fine", "gru/gru/R_fine"], "gru/gru/fine/kernel", axis=0)
        #self.concat_weight(["gru/gru/bias_u_fine", "gru/gru/bias_r_fine", "gru/gru/bias_e_fine"], "gru/gru/fine/bias", axis=0)

    def gru(self, network, inputs, state, r_u, r_r, r_e):
        #i_u_fine = K.dot(inputs, self.I_u_fine)
        #i_r_fine = K.dot(inputs, self.I_r_fine)
        #i_e_fine = K.dot(inputs, self.I_e_fine)    
        self.rename_weight("gru/gru/I_u_{}".format(network), "gru/gru/{}/u/kernel".format(network))
        self.rename_weight("gru/gru/I_r_{}".format(network), "gru/gru/{}/r/kernel".format(network))
        self.rename_weight("gru/gru/I_e_{}".format(network), "gru/gru/{}/e/kernel".format(network))
        self.rename_weight("gru/gru/bias_u_{}".format(network), "gru/gru/{}/u/bias".format(network))
        self.rename_weight("gru/gru/bias_r_{}".format(network), "gru/gru/{}/r/bias".format(network))
        self.rename_weight("gru/gru/bias_e_{}".format(network), "gru/gru/{}/e/bias".format(network))
        i_u = self.extract_dense(inputs, "I_u_{}".format(network), 0, "gru/gru/{}/u".format(network))
        i_r = self.extract_dense(inputs, "I_r_{}".format(network), 0, "gru/gru/{}/r".format(network))
        i_e = self.extract_dense(inputs, "I_e_{}".format(network), 0, "gru/gru/{}/e".format(network))
        #u = K.sigmoid(r_u + i_u + self.bias_u)
        #r = K.sigmoid(r_r + i_r + self.bias_r)
        sum_u = self.add_sum([r_u, i_u], "sum_u_{}".format(network))
        sum_r = self.add_sum([r_r, i_r], "sum_r_{}".format(network))
        u = self.add_sigmoid(sum_u, "u_{}".format(network))
        r = self.add_sigmoid(sum_r, "r_{}".format(network))

        #e = K.tanh(r*r_e + i_e + self.bias_e)
        reset = self.add_prod([r, r_e], "r*r_e_{}".format(network))
        sum_e = self.add_sum([reset, i_e], "sum_e_{}".format(network))
        e = self.add_tanh(sum_e, "e_{}".format(network))

        #output = u*state + (1.0 - u)*e
        left = self.add_prod([u, state], "u*prev_{}".format(network))
        right = self.add_power(u, "(1.0 - u)_{}".format(network), scale=-1, shift=1)
        right = self.add_prod([right, e], "(1.0 - u)*e_{}".format(network))
        output = self.add_sum([left, right], "gru_{}".format(network))
        return output

    def backbone(self, network, concat_output, state, r_u, r_r, r_e):
        #gru_output = self.extract_rnn("GRU_LBR", concat_output, state_name, network+"_gru", 0,
        #    steps=-1, scope_name="gru/gru/{}".format(network))
        gru_output = self.gru(network, concat_output, state, r_u, r_r, r_e)
        self.scopes[0] = 'affine_{}_layer'.format(network)
        affine_layer = self.extract_dense(gru_output, "affine_{}_layer".format(network), 1, self.scopes[0])
        softmax_layers = []
        for i in range(1, 5):
            self.scopes[0] = 'softmax_{}_layer_{}'.format(network, i)
            softmax_layer = self.extract_dense(affine_layer, 'softmax_{}_layer_{}'.format(network, i), 1, self.scopes[0])
            softmax_layers.append(softmax_layer)
        output = self.add_concat(softmax_layers, "{}_concatenate_2".format(network), axis=-1)
        return output, gru_output

    def generate(self, inputs=None):
        input_1 = "input_1"
        input_1_shape = [self.batch, 8]
        self.add_input(input_1, input_1_shape)
        input_3 = "input_3"
        input_3_shape = [self.batch, 256]
        self.add_input(input_3, input_3_shape)
        input_2 = "input_2"
        input_2_shape = [self.batch, 4]
        self.add_input(input_2, input_2_shape)
        self.set_input(inputs)
        state_shape = [self.batch, self.state_length]
        state = "state"
        state = self.add_memory(state, state_shape, data_type="FLOAT32")
        self.rename_weight("gru/gru/R_u", "gru/gru/R_u/kernel")
        self.rename_weight("gru/gru/R_r", "gru/gru/R_r/kernel")
        self.rename_weight("gru/gru/R_e", "gru/gru/R_e/kernel")
        R_u = self.extract_dense(state, "R_u", 0, "gru/gru/R_u")
        R_r = self.extract_dense(state, "R_r", 0, "gru/gru/R_r")
        R_e = self.extract_dense(state, "R_e", 0, "gru/gru/R_e")
        self.add_slice(state, ["state_coarse", "state_fine"], 1, [192])
        self.add_slice(R_u, ["R_u_coarse", "R_u_fine"], 1, [192])
        self.add_slice(R_r, ["R_r_coarse", "R_r_fine"], 1, [192])
        self.add_slice(R_e, ["R_e_coarse", "R_e_fine"], 1, [192])

        input_1_embed = self.extract_embedding(input_1, 0, "embedding/embedding/embeddings", "input_1_embed")
        input_1_reshape = self.add_reshape(input_1_embed, "input_1_reshape", [self.batch, -1])

        coarse_concat_input = [input_3, input_1_reshape]
        coarse_concat_output = self.add_concat(coarse_concat_input, "coarse_concat_input", axis=1)
        coarse_output, coarse_state = self.backbone("coarse", coarse_concat_output, "state_coarse", "R_u_coarse", "R_r_coarse", "R_e_coarse")

        fine_concat_input = [input_3, input_1_reshape, input_2]
        fine_concat_output = self.add_concat(fine_concat_input, "fine_concat_input", axis=1)
        fine_output, fine_state = self.backbone("fine", fine_concat_output, "state_fine", "R_u_fine", "R_r_fine", "R_e_fine")
        new_state = self.add_concat([coarse_state, fine_state], "state_new", axis=-1)
