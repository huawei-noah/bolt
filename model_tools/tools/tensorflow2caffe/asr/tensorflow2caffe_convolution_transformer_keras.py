#!/usr/local/bin/python
# -*- coding: utf-8 -*-

import math
import numpy as np
import sys
sys.path.append("../")
from tensorflow2caffe import Tensorflow2Caffe

class Tensorflow2CaffeConvolutionTransformerKeras(Tensorflow2Caffe):
    def __init__(self,
            tensorflow_model_path, caffe_model_path_prefix, caffe_model_name,
            base_params=None,
            nchwc8=True, first_frame=True,
            check=False, calc=False):
        Tensorflow2Caffe.__init__(self, tensorflow_model_path,
            caffe_model_path_prefix, caffe_model_name, check, calc)
        self.params = base_params
        self.nchwc8 = nchwc8
        self.first_frame = first_frame
        self.mode = "infer"
        self.state_data_path = "./"
        self.save_state = True

    @staticmethod
    def default_params():
        return {
                "max_sequence_length": 40,
                "encoder": [
                    {"num": 1,
                     "shape": [1, 1, 1, 41]},
                    {"num": 1,
                     "shape": [1, 32, 1, 21]},
                    {"num": 3,
                     "shape": [1, 8, 512]},
                    {"num": 1,
                     "shape": [1, 512, 1, 1]},
                    {"num": 3,
                     "shape": [1, 12, 512]},
                    {"num": 1,
                     "shape": [1, 512, 1, 1]},
                    {"num": 3,
                     "shape": [1, 16, 512]},
                    {"num": 1,
                     "shape": [1, 512, 1, 1]}
                ],
                "prediction_net": [
                    {"num": 3,
                     "shape": [1, 5, 512]}
                ]
            }

    def rel_shift(self, x):
        x = self.add_relative_shift(x, x+"_rel_shift", axis=3, shift_length=1)
        return x

    def rel_attn_core(self, q_head, k_head_h, v_head_h, k_head_r, recep_field, same_length, scale, scope_id, scope_name_prefix):
        """Core relative positional attention operations."""

        # content based attention score
        r_w_bias = self.add_weight(scope_name_prefix+"_r_w_bias", weight_name=scope_name_prefix+"/r_w_bias",
            transpose=None, data_type="FLOAT32")
        q_head_b = self.add_sum([q_head, r_w_bias], q_head+"_w_sum")
        ac = self.add_matmul(q_head_b, k_head_h, scope_name_prefix+"_ac")

        # position based attention score
        r_r_bias = self.add_weight(scope_name_prefix+"_r_r_bias", weight_name=scope_name_prefix+"/r_r_bias",
            transpose=None, data_type="FLOAT32")
        q_head_b = self.add_sum([q_head, r_r_bias], q_head+"_r_sum")
        bd = self.add_matmul(q_head_b, k_head_r, scope_name_prefix+"_bd")
        bd = self.rel_shift(bd)

        # merge attention scores and perform masking
        acbd = self.add_sum([ac, bd], scope_name_prefix+"_acbd")
        attn_score = self.add_power(acbd, scope_name_prefix+"_attn_score", scale=scale)
        attn_score = self.add_attention_mask(attn_score, attn_score+"_mask", recep_field, same_length, 1e30)

        # attention probability
        attn_prob = self.add_softmax(attn_score, scope_name_prefix+"_attn_prob", 3)
        # attention output
        attn_vec = self.add_matmul(attn_prob, v_head_h, scope_name_prefix+"_attn_vec")
        attn_vec = self.add_transpose(attn_vec, attn_vec+"_t", [0, 2, 1, 3])
        return attn_vec

    def relative_positional_encoding(self, d_model, recep_field, scope_name_prefix):
        """create relative positional encoding."""
        time_len = self.params['max_sequence_length']
        freq_seq = np.arange(0, d_model, 2.0, dtype=np.float32)
        inv_freq = 1 / (10000 ** (freq_seq / d_model))

        fwd_pos_seq = np.arange(time_len, -1, -1.0, dtype=np.float32)
        fwd_pos_seq = np.clip(fwd_pos_seq, -recep_field, recep_field)
        sinusoid_inp = np.matmul(np.expand_dims(fwd_pos_seq, 1), np.expand_dims(inv_freq, 0))
        pos_emb = np.concatenate([np.sin(sinusoid_inp), np.cos(sinusoid_inp)], axis=-1)
        pos_emb = np.tile(pos_emb, (1, 1))
        pos_emb = self.add_weight(scope_name_prefix+"_pos_dict", weight=pos_emb, data_type="FLOAT32")
        return pos_emb

    def RelHistoricalSelfAttention2D(self, inputs, n_head, d_head, recep_field, same_length, use_4d, scope_id, scope_name_prefix):
        d_model = n_head * d_head
        h_input = inputs
        scale = 1 / (d_head ** 0.5)

        pos_emb = self.relative_positional_encoding(d_model, recep_field, scope_name_prefix)
        pos_emb = self.add_relative_position_embedding(inputs, pos_emb, 1, scope_name_prefix+"_pos_emb", transpose=False)

        q_head_h = self.extract_dense(h_input, h_input+"_q", scope_id, [scope_name_prefix+"_hsa", "query_weights", "bias"])
        k_head_h = self.extract_dense(h_input, h_input+"_k", scope_id, [scope_name_prefix+"_hsa", "key_weights", "bias"])
        v_head_h = self.extract_dense(h_input, h_input+"_v", scope_id, [scope_name_prefix+"_hsa", "value_weights", "bias"])
        q_head_h = self.add_reshape(q_head_h, q_head_h+"_r", [self.batch, -1, n_head, d_head])
        k_head_h = self.add_reshape(k_head_h, k_head_h+"_r", [self.batch, -1, n_head, d_head])
        v_head_h = self.add_reshape(v_head_h, v_head_h+"_r", [self.batch, -1, n_head, d_head])
        q_head_h = self.add_transpose(q_head_h, q_head_h+"_t", [0, 2, 1, 3])
        k_head_h = self.add_transpose(k_head_h, k_head_h+"_t", [0, 2, 3, 1])
        v_head_h = self.add_transpose(v_head_h, v_head_h+"_t", [0, 2, 1, 3])

        # positional heads
        k_head_r = self.extract_dense(pos_emb, pos_emb+"_k", scope_id, [scope_name_prefix+"_hsa", "rel_weights", "bias"])
        k_head_r = self.add_reshape(k_head_r, k_head_r+"_r", [self.batch, -1, n_head, d_head])
        k_head_r = self.add_transpose(k_head_r, k_head_r+"_t", [0, 2, 3, 1])

        # core attention ops
        attn_vec = self.rel_attn_core(q_head_h, k_head_h, v_head_h, k_head_r, recep_field, same_length, scale, scope_id, scope_name_prefix+"_hsa")

        # post processing
        attn_vec = self.add_reshape(attn_vec, attn_vec+"_r", [self.batch, -1, d_model])
        output = self.extract_dense(attn_vec, attn_vec+"_output", scope_id, [scope_name_prefix+"_hsa", "output_weights", "bias"])
        return output

    def TransformerTransition(self, inputs, activation_fn, scope_id, tt_name_prefix):
        x = self.extract_dense(inputs, inputs+"_x", scope_id, [tt_name_prefix, "weights1", "biases1"])
        if (activation_fn == "relu"):
            x = self.add_relu(x, x+"_relu")
        else:
            print("[ERROR] unsupported activation function" % (activation_fn))
            exit(1)
        output = self.extract_dense(x, inputs+"_xx", scope_id, [tt_name_prefix, "weights2", "biases2"])
        return output

    def HistoricalSelfAttentionBlock(self, x, n_head, d_head, recep_field, merge_size,
        scope_id, scope_name_prefix, tt_name_prefix, use_4d=False,
        use_bn=False, use_final_norm=False,
        state=None, activation_fn="relu", same_length=False):

        # self-attention block
        residual = x
        x = self.add_concat([state, x], scope_name_prefix+"_concat", axis=1)
        slice_result = scope_name_prefix + "_slice"
        self.add_slice(x, [scope_name_prefix+"_other1", slice_result], 1, [-recep_field])
        if use_bn:
            x = self.extract_batch_norm(x, x+"_norm0", scope_id,
                data_format="NCHW", axis=-1, layer_names=[scope_name_prefix+"_norm0", "moving_mean", "moving_variance"])
        x = self.RelHistoricalSelfAttention2D(x, n_head, d_head, recep_field, same_length, use_4d,
            scope_id, scope_name_prefix)
        content = scope_name_prefix + "_padding_drop"
        self.add_slice(x, [scope_name_prefix+"_other2", content], 1, [recep_field])
        x = content
        x = self.add_sum([residual, x], scope_name_prefix+"_sum")
        if not use_bn:
            x = self.extract_layer_norm(x, x+"_norm0", scope_id, [scope_name_prefix+"_norm0", "gamma", "beta"])

        # feed forword block
        residual = x
        if use_bn:
            x = self.extract_batch_norm(x, x+"_norm1", scope_id,
                data_format="NCHW", axis=-1, layer_names=[scope_name_prefix+"_norm1", "moving_mean", "moving_variance"])
        x = self.TransformerTransition(x, activation_fn, scope_id, tt_name_prefix)
        x = self.add_sum([residual, x], scope_name_prefix+"_sum2")
        if not use_bn:
            x = self.extract_layer_norm(x, x+"_norm1", scope_id, [scope_name_prefix+"_norm1", "gamma", "beta"])

        if use_final_norm:
            x = self.extract_layer_norm(x, x+"_final_norm", scope_id, [scope_name_prefix+"_final_norm", "gamma", "beta"])
        return x, slice_result

    def generate_prediction_joint_net(self, input_shape, input=None):
        prediction_net_input_name = "prediction_net"
        prediction_net_input_shape = input_shape[prediction_net_input_name]
        self.add_input(prediction_net_input_name, prediction_net_input_shape)
        encoder_output_name = "encoder"
        encoder_output_shape = input_shape[encoder_output_name]
        self.add_input(encoder_output_name, encoder_output_shape)
        self.set_input(input)

        states = self.prepare_states(self.params["prediction_net"], "prediction_net_mem")
        self.save_input()

        attention_id = 1
        x = prediction_net_input_name
        new_states = []
        for i in range(3):
            tt_name_prefix = "transformer_transition_" + str(attention_id+8)
            x, state = self.HistoricalSelfAttentionBlock(x,
                n_head=8, d_head=64,
                recep_field=5, merge_size=1,
                scope_id=0, scope_name_prefix="decoder_hsa_" + str(attention_id), tt_name_prefix=tt_name_prefix, use_4d=False,
                state=states[i])
            new_states.append(state)
            attention_id += 1

        x = self.add_concat([x, encoder_output_name], "joint_net_input", 2)
        x = self.extract_dense(x, x+"_fc1", 0, ["joint_forward", "kernel", "bias"])
        x = self.add_tanh(x, "joint_net_tanh")
        x = self.extract_dense(x, "joint_output_fc", 0, ["joint_classification", "kernel", "bias"])
        self.save_caffe_model()

    def prepare_states(self, states_shape, output_name_prefix, init_with_none=False):
        states = []
        state_id = 0
        for item in states_shape:
            num = item["num"]
            state_shape = item["shape"]
            if (init_with_none):
                state_shape = [0] * len(state_shape)
            for i in range(num):
                mem_name = output_name_prefix + str(state_id)
                if (self.first_frame):
                    data = {mem_name: np.zeros(state_shape)}
                else:
                    file_data = np.load(self.state_data_path + "/" + mem_name + ".npy")
                    data = {mem_name: file_data}
                    state_shape = file_data.shape
                self.add_input(mem_name, state_shape)
                self.set_input(data)
                states.append(mem_name)
                state_id += 1
        return states

    def Conv2DBlock(self, x, filters, scope_name_prefix, kernel_size=[3, 3], strides=[1, 1],
                 use_relu=True, axis=2, state=None, scope_id=0):
        if (len(self.get_tensor_shape(x)) == 5 and len(self.get_tensor_shape(state)) == 4):
            shape = self.get_tensor_shape(state)
            self.data_dict[state] = self.data_dict[state].reshape(
                        [self.batch, shape[1]//8, -1, shape[3], 8])
        x = self.add_concat([state, x], scope_name_prefix+"_concat", axis=axis)
        slice_result = scope_name_prefix + "_slice"
        self.add_slice(x, [scope_name_prefix+"_other1", slice_result], axis, [-1])
        padding = self.calculate_convolution_padding(self.get_tensor_shape(x), kernel_size, strides, 'same')
        y = self.extract_convolution(x, scope_name_prefix+"_conv", scope_id,
                            filters, kernel_size, strides, padding,
                            data_format="NCHW",
                            dilation=1, groups=1, layer_names=[scope_name_prefix+"_conv", "kernel", "bias"])
        y = self.extract_batch_norm(y, scope_name_prefix+"_bn", scope_id, layer_names=[scope_name_prefix+"_bn", "moving_mean", "moving_variance"])
        if use_relu:
            y = self.add_relu6(y, scope_name_prefix+"_relu6")
        content = scope_name_prefix + "_padding_drop"
        self.add_slice(y, [scope_name_prefix+"_other2", content], axis, [1])
        return content, slice_result

    def AlignmentBlock(self, x, filters, scope_name_prefix, extend=4, scope_id=0):
        stride = [1, 1]
        shape = self.get_tensor_shape(x)
        x = self.add_reshape(x, scope_name_prefix+"_r", [self.batch, -1, shape[2], 1])
        if extend > 0:
            padding0 = self.calculate_convolution_padding(self.get_tensor_shape(x), [1, 1], stride, 'same')
            x = self.extract_convolution(x, scope_name_prefix+"_conv0", scope_id,
                                filters*extend, [1, 1], stride, padding0,
                                data_format="NCHW",
                                dilation=1, groups=1, layer_names=[scope_name_prefix+"_conv0", "kernel", "bias"])
            x = self.extract_batch_norm(x, scope_name_prefix+"_bn0", scope_id, layer_names=[scope_name_prefix+"_bn0", "moving_mean", "moving_variance"])
            x = self.add_relu6(x, scope_name_prefix+"_act0")

            padding1 = self.calculate_convolution_padding(self.get_tensor_shape(x), [1, 1], stride, 'same')
            x = self.extract_convolution(x, scope_name_prefix+"_conv1", scope_id,
                                filters, [1, 1], stride, padding1,
                                data_format="NCHW",
                                dilation=1, groups=1, layer_names=[scope_name_prefix+"_conv1", "kernel", "bias"])
            x = self.extract_batch_norm(x, scope_name_prefix+"_bn1", scope_id, layer_names=[scope_name_prefix+"_bn1", "moving_mean", "moving_variance"])
        else:
            padding0 = self.calculate_convolution_padding(self.get_tensor_shape(x), [1, 1], stride, 'same')
            x = self.extract_convolution(x, scope_name_prefix+"_conv0", scope_id,
                                filters, [1, 1], stride, padding0,
                                data_format="NCHW",
                                dilation=1, groups=1, layer_names=[scope_name_prefix+"_conv0", "kernel", "bias"])
            x = self.extract_batch_norm(x, scope_name_prefix+"_bn0", scope_id, layer_names=[scope_name_prefix+"_bn0", "moving_mean", "moving_variance"])
        return x

    def ResConvBlock(self, x, filters, scope_name_prefix, stride=2, axis=2, state=None, scope_id=0):
        stride = [stride, stride]
        padding0 = self.calculate_convolution_padding(self.get_tensor_shape(x), [1, 1], stride, 'same')
        residual = self.extract_convolution(x, scope_name_prefix+"_resconv", scope_id,
                            filters[0], [1, 1], stride, padding0,
                            data_format="NCHW",
                            dilation=1, groups=1, layer_names=[scope_name_prefix+"_resconv", "kernel", "bias"])
        residual = self.extract_batch_norm(residual, scope_name_prefix+"_resbn", scope_id, layer_names=[scope_name_prefix+"_resbn", "moving_mean", "moving_variance"])

        x = self.add_relu6(x, scope_name_prefix+"_act0")
        x = self.add_concat([state, x], scope_name_prefix+"_concat", axis=axis)
        slice_result = scope_name_prefix + "_slice"
        self.add_slice(x, [scope_name_prefix+"_other1", slice_result], axis, [-1])
        padding1 = self.calculate_convolution_padding(self.get_tensor_shape(x), [3, 1], stride, 'same')
        x = self.extract_convolution(x, scope_name_prefix+"_conv1", scope_id,
                            filters[1], [3, 1], stride, padding1,
                            data_format="NCHW",
                            dilation=1, groups=1, layer_names=[scope_name_prefix+"_conv1", "kernel", "bias"])
        x = self.extract_batch_norm(x, scope_name_prefix+"_bn1", scope_id, layer_names=[scope_name_prefix+"_bn1", "moving_mean", "moving_variance"])
        x = self.add_relu6(x, scope_name_prefix+"_act1")
        content = scope_name_prefix + "_padding_drop"
        self.add_slice(x, [scope_name_prefix+"_other2", content], axis, [1])
        x = content

        padding2 = self.calculate_convolution_padding(self.get_tensor_shape(x), [1, 1], stride, 'same')
        x = self.extract_convolution(x, scope_name_prefix+"_conv2", scope_id,
                            filters[2], [1, 1], [1, 1], padding2,
                            data_format="NCHW",
                            dilation=1, groups=1, layer_names=[scope_name_prefix+"_conv2", "kernel", "bias"])
        x = self.extract_batch_norm(x, scope_name_prefix+"_bn2", scope_id, layer_names=[scope_name_prefix+"_bn2", "moving_mean", "moving_variance"])
        x = self.add_relu6(x, scope_name_prefix+"_act2")

        padding3 = self.calculate_convolution_padding(self.get_tensor_shape(x), [1, 1], stride, 'same')
        x = self.extract_convolution(x, scope_name_prefix+"_conv3", scope_id,
                            filters[3], [1, 1], [1, 1], padding3,
                            data_format="NCHW",
                            dilation=1, groups=1, layer_names=[scope_name_prefix+"_conv3", "kernel", "bias"])
        x = self.extract_batch_norm(x, scope_name_prefix+"_bn3", scope_id, layer_names=[scope_name_prefix+"_bn3", "moving_mean", "moving_variance"])
        x = self.add_sum([x, residual], scope_name_prefix+"_sum")
        return x, slice_result

    def TransformerEncoder(self, x, states):
        scope_id = 0
        new_states = []
        x = self.transpose_nhwc_nchw(x)
        # main blocks
        attention_id = 1
        state_id = 0

        # single-conv block
        x, state = self.Conv2DBlock(x, filters=32, scope_name_prefix="conv_1",
                        kernel_size=[3, 11], strides=[1, 2],
                        axis=2, state=states[state_id], scope_id=scope_id)
        state_id += 1
        new_states.append(state)
        x, state = self.Conv2DBlock(x, filters=32, scope_name_prefix="conv_2",
                        kernel_size=[3, 7], strides=[2, 1],
                        axis=2, state=states[state_id], scope_id=scope_id)
        state_id += 1
        new_states.append(state)
        x = self.AlignmentBlock(x, 512, scope_name_prefix="alignment", scope_id=scope_id)

        # attention block 1-3
        x = self.transpose_nchc8_nhc(x)
        for i in range(3):
            tt_name_prefix = "transformer_transition"
            if (attention_id != 1):
                tt_name_prefix += "_" + str(attention_id-1)
            x, state = self.HistoricalSelfAttentionBlock(x, n_head=8, d_head=64, recep_field=8, merge_size=2,
                                             scope_id=scope_id, scope_name_prefix="hsa_" + str(attention_id), tt_name_prefix=tt_name_prefix, use_4d=True,
                                             state=states[state_id])
            state_id += 1
            new_states.append(state)
            attention_id += 1
        x = self.transpose_nhc_nchw(x)

        # res-conv block 1
        x, state = self.ResConvBlock(x, filters=[512, 1024, 2048, 512], scope_name_prefix="conv_block_1",
                         stride=2, axis=2, state=states[state_id], scope_id=scope_id)
        state_id += 1
        new_states.append(state)

        # attention block 4-6
        x = self.transpose_nchc8_nhc(x)
        for i in range(3):
            tt_name_prefix = "transformer_transition"
            if (attention_id != 1):
                tt_name_prefix += "_" + str(attention_id-1)
            x, state = self.HistoricalSelfAttentionBlock(x, n_head=8, d_head=64, recep_field=12, merge_size=4,
                                             scope_id=scope_id, scope_name_prefix="hsa_" + str(attention_id), tt_name_prefix=tt_name_prefix, use_4d=True,
                                             state=states[state_id])
            state_id += 1
            new_states.append(state)
            attention_id += 1
        x = self.transpose_nhc_nchw(x)

        # res-conv block 2
        x, state = self.ResConvBlock(x, filters=[512, 1024, 2048, 512], scope_name_prefix="conv_block_2",
                         stride=2, axis=2, state=states[state_id], scope_id=scope_id)
        state_id += 1
        new_states.append(state)

        # attention block 7-9
        x = self.transpose_nchc8_nhc(x)
        for i in range(3):
            tt_name_prefix = "transformer_transition"
            if (attention_id != 1):
                tt_name_prefix += "_" + str(attention_id-1)
            x, state = self.HistoricalSelfAttentionBlock(x, n_head=8, d_head=64, recep_field=16, merge_size=8,
                                             scope_id=scope_id, scope_name_prefix="hsa_" + str(attention_id), tt_name_prefix=tt_name_prefix, use_4d=True,
                                             state=states[state_id])
            state_id += 1
            new_states.append(state)
            attention_id += 1
        x = self.transpose_nhc_nchw(x)

        # res-conv block 3
        x, state = self.ResConvBlock(x, filters=[1024, 1024, 4096, 1024], scope_name_prefix="conv_block_3",
                         stride=1, axis=2, state=states[state_id], scope_id=scope_id)
        state_id += 1
        new_states.append(state)

        # layers for output
        stride = [1, 1]
        padding = self.calculate_convolution_padding(self.get_tensor_shape(x), [1, 1], stride, 'valid')
        x = self.extract_convolution(x, "joint_encoder_trans", scope_id,
                            512, [1, 1], stride, padding,
                            data_format="NCHW",
                            dilation=1, groups=1, layer_names=["joint_encoder_trans", "kernel", "bias"])
        x = self.extract_batch_norm(x, "joint_encoder_trans_bn", scope_id, layer_names=["joint_encoder_trans_bn", "moving_mean", "moving_variance"])
        x = self.add_relu6(x, "encoder_output")
        return x, new_states

    def generate_encoder(self, input_shape, input=None):
        encoder_input_name = "encoder"
        encoder_input_shape = input_shape[encoder_input_name]
        self.add_input(encoder_input_name, encoder_input_shape)
        self.set_input(input)

        states = self.prepare_states(self.params["encoder"], "encoder_mem")
        self.save_input()
        self.TransformerEncoder(encoder_input_name, states)
        self.save_caffe_model()
