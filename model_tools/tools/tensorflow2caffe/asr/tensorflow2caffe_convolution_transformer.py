#!/usr/local/bin/python
# -*- coding: utf-8 -*-

import math
import numpy as np
import sys
sys.path.append("../")
from tensorflow2caffe import Tensorflow2Caffe
from convolution_transformer_params import base_params

class Tensorflow2CaffeConvolutionTransformer(Tensorflow2Caffe):
    def __init__(self,
            tensorflow_model_path, caffe_model_path_prefix, caffe_model_name,
            nchwc8=True, first_frame=True,
            check=False, calc=False, quantization=False):
        Tensorflow2Caffe.__init__(self, tensorflow_model_path,
            caffe_model_path_prefix, caffe_model_name, check, calc, quantization)
        self.base_params = base_params
        self.params = {}
        self.mode = "infer"
        self.nchwc8 = nchwc8
        self.first_frame = first_frame
        self.state_data_path = "./"
        self.save_state = True

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

    def positionwise_FF(self, input, d_model, d_inner, dropout, kernel_initializer,
                    scope='ff', is_training=True, norm_type='ln',
                    scope_id=-1, output_name_prefix=""):
        self.scopes[scope_id] = scope
        output = input
        if norm_type == 'bn':
            output = self.extract_batch_norm(output, output_name_prefix+"_FF_bn", scope_id+1,
                         data_format="NCHW", axis=-1,
                         layer_names=["batch_normalization", "moving_mean", "moving_variance"])
        elif norm_type == 'pre_ln':
            output = self.extract_layer_norm(output, output_name_prefix+"_FF_ln", scope_id+1, ["LayerNorm", "gamma", "beta"])
        self.add_quantization(scope_id+1, "quant_ffn_input", output)
        output = self.extract_dense(output, output_name_prefix+"_FF_fc1", scope_id+1, scope_name="layer_1")
        output = self.add_relu(output, output+"_FF_relu")
        self.add_quantization(scope_id+1, "quant_ffn_middle", output)
        output = self.extract_dense(output, output_name_prefix+"_FF_fc2", scope_id+1, scope_name="layer_2")
        output = self.add_sum([output, input], output_name=output_name_prefix+"_FF_sum")
        if norm_type == 'ln':
            output = self.extract_layer_norm(output, output_name_prefix+"_FF_ln", scope_id+1, ["LayerNorm", "gamma", "beta"])
        else:
            assert norm_type in ['bn', 'pre_ln']
            output = output
        return output

    def group_norm(x, scop_id, data_format, group_num=32):
        if data_format == 'channels_last':
            x = self.add_transpose(x, x+"_t1", [0, 3, 1, 2])
        x = self.extract_group_norm(x, group_num, x+"_gn", scope_id, data_format="NCHW", layer_names=None)
        output = self.add_transpose(x, x+"_t2", [0, 2, 3, 1])
        return output

    def neg_slice(self, input_name, axis, length, output_name_prefix):
        other_name = output_name_prefix + "_other"
        output_name = output_name_prefix + "_neg_slice"
        self.add_slice(input_name, [other_name, output_name], axis=axis, slice_point=[-length])
        return output_name

    def row_conv(self, name, input_layer, batch, channels, width, activation_fn,
                 data_format, norm_type='batch_norm', gn_group_num=0,
                 scope_id=-1, output_name_prefix=""):
        print("[ERROR] currently not support row_conv")
        exit(1)
        if width < 2:
            return input_layer
    
        if data_format == 'channels_last':
            x = self.add_reshape(input_layer, [batch, -1, 1, channels], output_name_prefix+"_row_conv_r")
            x = self.add_transpose(x, [0, 3, 1, 2], output_name_prefix+"_row_conv_t")  # B C T
        else:
            x = self.add_transpose(input_layer, [0, 2, 1], output_name_prefix+"_row_conv_t")  # B C T
            x = self.add_reshape(x, [batch, channels, -1, 1], output_name_prefix+"_row_conv_r")
        y = self.extract_convolution(x, output_name_prefix+"_row_conv", scope_id+1,
                            channels, [width, 1], [1, 1], [(width-1)//2, (width-1)//2, 0, 0],
                            data_format="NCHW",
                            dilation=1, groups=channels, layer_names=None)
        if norm_type != 'batch_norm':
            y = self.add_nchwc8_nchw(y, output_name_prefix+"_nchwc8_nchw")
        if norm_type == 'batch_norm':
            bn = self.extract_batch_norm(y, output_name_prefix+"_row_conv_bn", scope_id+1)
            bn = self.add_nchwc8_nchw(bn, output_name_prefix+"_nchwc8_nchw")
        elif norm_type == 'group_norm':
            assert data_format == 'channels_last'
            bn = self.group_norm(y, scope_id+1, data_format)
        elif norm_type == 'layer_norm':
            assert data_format == 'channels_last'
            bn = self.extract_layer_norm(y, output_name_prefix+"_row_conv_ln", scope_id+1)
        else:
            assert norm_type == 'skip'
            bn = y
        if (activation_fn == "relu"):
            relu_name = output_name_prefix + "_relu"
            output = self.add_relu(bn, relu_name)
        else:
            print("[ERROR] unsupported activation function" % (activation_fn))
            exit(1)
        output = self.add_transpose(output, [0, 2, 3, 1], output_name_prefix+"_t")
        output = self.add_reshape(output, [batch, -1, channels], output_name_prefix+"_r")
        return output

    def conv_proj_res(self, layer_type, name, inputs, filters, proj_filters, kernel_size, activation_fn, strides, padding, data_format,
                      dilation=1, norm_type='batch_norm',
                      scope_id=-1, output_name_prefix=""):
        print("[ERROR] currently not support conv_proj_res")
        exit(1)
        self.scopes[scope_id] = name
        assert norm_type == 'batch_norm'
        output = self.extract_batch_norm(inputs, output_name_prefix+"_bn", scope_id+1)
        assert layer_type == 'conv1d'
        assert activation_fn is not None
        output = self.extract_convolution(output, output_name_prefix+"_row_conv", scope_id+1,
                            kernel_size[0], kernel_size[2:4], strides, padding,
                            data_format="NCHW",
                            dilation=dilation, groups=1, layer_names=None)
        if (activation_fn == "relu"):
            relu_name = output_name_prefix + "_relu"
            output = self.add_relu(output, relu_name)
        else:
            print("[ERROR] unsupported activation function" % (activation_fn))
            exit(1)
        output = self.extract_convolution(output, proj_filters, [1], [1], padding,
            1, 1, output_name_prefix+"_conv")
        output = self.extract_convolution(output, output_name_prefix+"_row_conv", scope_id+1,
                            proj_filters, [1, 1], [1, 1], padding,
                            data_format="NCHW",
                            dilation=1, groups=1, layer_names=None)
        output = self.add_nchwc8_nchw(output)
        output = self.add_sum([output, inputs], output_name_prefix+"_conv_proj_res")
        return output

    def conv_bn_actv(self, layer_type, name, inputs, filters, kernel_size, activation_fn,
                     strides, padding, data_format,
                     dilation=1, norm_type='batch_norm', gn_group_num=0,
                     scope_id=-1, output_name_prefix=""):
        groups = 1
        if layer_type == 'sep_conv1d':
            groups = filters
        if data_format == "channels_last":
            inputs = self.add_transpose(inputs, output_name_prefix+"_pre_t", [0, 3, 1, 2])
        self.add_quantization(scope_id, "quant_"+name, inputs)
        conv = self.extract_convolution(inputs, output_name_prefix+"_conv", scope_id,
                            filters, kernel_size, strides, padding,
                            data_format="NCHW",
                            dilation=dilation, groups=groups, layer_names=[name, "kernel", "bias"])
        if norm_type != 'batch_norm':
            conv = self.add_nchwc8_nchw(conv, output_name_prefix+"_nchwc8_nchw")
        squeeze = False
        if "conv1d" in layer_type and norm_type == 'batch_norm':
            #axis = 1 if data_format == 'channels_last' else 2
            ## NWC --> NHWC
            #conv = self.add_expand_dims(conv, axis=axis, output_name=conv+"_expand")
            squeeze = True

        if norm_type == 'skip':
            bn = conv
        elif norm_type == 'group_norm':
            print("[ERROR] currently not support online group norm")
            exit(1)
        elif norm_type == 'layer_norm':
            assert data_format == 'channels_last'
            bn = self.extract_layer_norm(conv, output_name_prefix+"_ln", scope_id+1, ["LayerNorm", "gamma", "beta"])
        elif norm_type == 'online_batch_norm':
            print("[ERROR] currently not support online batch norm")
            exit(1)
        else:
            assert norm_type == 'batch_norm'
            bn = self.extract_batch_norm(conv, output_name_prefix+"_bn", scope_id+1)
            bn = self.add_nchwc8_nchw(bn, output_name_prefix+"_nchwc8_nchw")
        if data_format == "channels_last":
            bn = self.add_transpose(bn, output_name_prefix+"_post_t", [0, 2, 3, 1])
        if squeeze:
            bn = self.add_squeeze(bn, axis=2, output_name=bn+"_squeeze")
        if (activation_fn is None):
            output = bn
        elif (activation_fn == "relu"):
            relu_name = output_name_prefix + "_relu"
            output = self.add_relu(bn, relu_name)
        else:
            print("[ERROR] unsupported activation function %s" % (activation_fn))
            exit(1)
        return output

    def conv_layer_wrapper(self, inputs, conv_type, conv_layer_params,
                           gn_group_num, name, mask, layout, data_format,
                           decode_state=None, is_decoding=False,
                           scope_id=-1, output_name_prefix=""):
        self.scopes[scope_id] = name
        ch_out = conv_layer_params['num_channels']
        kernel_size = conv_layer_params['kernel_size']  # [T,F] format
        strides = conv_layer_params['stride']  # [T,F] format
        padding = conv_layer_params['padding']
        new_mems = None
        if is_decoding and kernel_size[0] > 1:
            assert decode_state is not None
            inputs = self.add_concat([decode_state, inputs], output_name_prefix+"_concat", axis=1)
            if strides[0] == 1:
                mem_len = 2
            else:
                assert strides[0] == 2
                mem_len = 1
            new_mems = self.neg_slice(inputs, axis=1, length=mem_len, output_name_prefix=output_name_prefix)

        if padding.lower() == 'same':
            if conv_type == 'conv2d':
                assert kernel_size[1] % 2 == 1
                pad_num = kernel_size[1] // 2
                left_pad_num = pad_num - 1 if self.get_tensor_shape(inputs)[-2] % 2 == 0 else pad_num
            else:
                pad_num = 0
                left_pad_num = 0
            padding = 'VALID'
            padding = [0, 0, left_pad_num, pad_num]
        else:
            padding = [0, 0, 0, 0]
            if kernel_size[0] == 1:
                assert decode_state is None
                assert len(kernel_size) == 1
            new_mems = None
        act = conv_layer_params['activation_fn']
        norm_type = conv_layer_params.get('norm_type', 'batch_norm')

        if mask is not None and not is_decoding:
            use_2d_conv = conv_type == 'conv2d'
            if use_2d_conv:
                assert data_format == 'channels_last'

        if layout == 'BFTC' or layout == 'BCFT':
            kernel_size = kernel_size[::-1]
            strides = strides[::-1]
        if conv_type == 'conv1d':
            inputs = self.add_expand_dims(inputs, 2, output_name_prefix+"_conv1d_expand")
            strides.append(1)
            kernel_size.append(1)
        shape = self.get_tensor_shape(inputs)
        padding[1] = math.ceil((shape[1] + padding[0] + padding[1] - kernel_size[0] + 1) / strides[0]) \
                     * strides[0] + kernel_size[0] - 1 - padding[0] - shape[1]
        padding[3] = math.ceil((shape[2] + padding[2] + padding[3] - kernel_size[1] + 1) / strides[1]) \
                     * strides[1] + kernel_size[1] - 1 - padding[2] - shape[2]
        if conv_type == 'conv_proj_res':
            proj_filters = conv_layer_params['proj_num_channels']
            inputs = self.conv_proj_res(
                layer_type='conv1d',
                name=name,
                inputs=inputs,
                filters=ch_out,
                proj_filters=proj_filters,
                kernel_size=kernel_size,
                activation_fn=act,
                strides=strides,
                padding=padding,
                norm_type=norm_type,
                data_format=data_format,
                scope_id=scope_id,
                output_name_prefix=output_name_prefix
            )
        else:
            inputs = self.conv_bn_actv(
                layer_type=conv_type,
                name=name,
                inputs=inputs,
                filters=ch_out,
                kernel_size=kernel_size,
                activation_fn=act,
                strides=strides,
                padding=padding,
                norm_type=norm_type,
                gn_group_num=gn_group_num,
                data_format=data_format,
                scope_id=scope_id,
                output_name_prefix=output_name_prefix
            )
        return inputs, new_mems

    def conv_bn_actv_nchwc8(self, layer_type, name, inputs, filters, kernel_size, activation_fn,
            strides, padding, data_format,
            dilation=1, norm_type='batch_norm', gn_group_num=0,
            layer_id=0, layer_num=1,
            scope_id=-1, output_name_prefix=""):
        groups = 1
        if layer_type == 'sep_conv1d':
            groups = filters
        if data_format == "channels_last" and layer_id == 0:
            inputs = self.add_transpose(inputs, output_name_prefix+"_pre_t", [0, 3, 1, 2])
        self.add_quantization(scope_id, "quant_"+name, inputs)
        conv = self.extract_convolution(inputs, output_name_prefix+"_conv", scope_id,
                   filters, kernel_size, strides, padding,
                   data_format="NCHW",
                   dilation=dilation, groups=groups, layer_names=[name, "kernel", "bias"])
        if norm_type != 'batch_norm' and norm_type != 'skip':
            print("[ERROR] currently not support %s in conv_bn_actv" % (norm_type))
            exit(1)
        squeeze = False
        if "conv1d" in layer_type and norm_type == 'batch_norm':
            squeeze = True
        if norm_type == 'skip':
            bn = conv
        elif norm_type == 'group_norm':
            print("[ERROR] currently not support online group norm")
            exit(1)
        elif norm_type == 'layer_norm':
            print("[ERROR] currently not support layer norm")
            exit(1)
        elif norm_type == 'online_batch_norm':
            print("[ERROR] currently not support online batch norm")
            exit(1)
        else:
            assert norm_type == 'batch_norm'
            bn = self.extract_batch_norm(conv, output_name_prefix+"_bn", scope_id+1)
        if (activation_fn is None):
            output = bn
        elif (activation_fn == "relu"):
            output = self.add_relu(bn, output_name_prefix+"_relu")
        else:
            print("[ERROR] unsupported activation function %s" % (activation_fn))
            exit(1)
        if (layer_id == layer_num - 1):
            output = self.add_transpose(output, output_name_prefix+"_nchwc8_nchw_t", [0, 2, 3, 1, 4])
            output_shape = self.get_tensor_shape(output)
            if squeeze:
                #output = self.add_reshape(output, output_name_prefix+"_nchwc8_nchw_r", [0, 0, 0, -1])
                #output = self.add_squeeze(output, axis=2, output_name=bn+"_squeeze")
                output = self.add_reshape(output, output_name_prefix+"_nchwc8_nchw_r", [self.batch, -1, output_shape[2]*output_shape[3]*output_shape[4]])
            else:
                output = self.add_reshape(output, output_name_prefix+"_nchwc8_nchw_r", [self.batch, -1, output_shape[2], output_shape[3]*output_shape[4]])
        return output

    def conv_layer_wrapper_nchwc8(self, inputs, conv_type, conv_layer_params,
            gn_group_num, name, mask, layout, data_format,
            decode_state=None, is_decoding=False,
            layer_id=0, layer_num=1,
            scope_id=-1, output_name_prefix=""):
        self.scopes[scope_id] = name
        ch_out = conv_layer_params['num_channels']
        kernel_size = conv_layer_params['kernel_size']  # [T,F] format
        strides = conv_layer_params['stride']  # [T,F] format
        padding = conv_layer_params['padding']
        new_mems = None
        if is_decoding and kernel_size[0] > 1:
            assert decode_state is not None
            if layer_id == 0:
                axis = 1
            else:
                axis = 2
                decode_state_shape = self.get_tensor_shape(decode_state)
                if (len(decode_state_shape) == 4):
                    self.data_dict[decode_state] = self.data_dict[decode_state].reshape(
                        [self.batch, decode_state_shape[1]//8, -1, decode_state_shape[3], 8])
                    assert(decode_state_shape[1]//8 != 0)
            inputs = self.add_concat([decode_state, inputs], output_name_prefix+"_concat", axis)
            if strides[0] == 1:
                mem_len = 2
            else:
                assert strides[0] == 2
                mem_len = 1
            new_mems = new_mems = self.neg_slice(inputs, axis=axis, length=mem_len, output_name_prefix=output_name_prefix)
        if (layer_id == 0):
            h_axis = 1
            w_axis = 2
        else:
            h_axis = 2
            w_axis = 3
        if padding.lower() == 'same':
            if conv_type == 'conv2d':
                assert kernel_size[1] % 2 == 1
                pad_num = kernel_size[1] // 2
                left_pad_num = pad_num - 1 if self.get_tensor_shape(inputs)[w_axis] % 2 == 0 else pad_num
            else:
                pad_num = 0
                left_pad_num = 0
            padding = 'VALID'
            padding = [0, 0, left_pad_num, pad_num]
        else:
            padding = [0, 0, 0, 0]
            if kernel_size[0] == 1:
                assert decode_state is None
                assert len(kernel_size) == 1
            new_mems = None
        act = conv_layer_params['activation_fn']
        norm_type = conv_layer_params.get('norm_type', 'batch_norm')

        if mask is not None and not is_decoding:
            use_2d_conv = conv_type == 'conv2d'
            if use_2d_conv:
                assert data_format == 'channels_last'

        if layout == 'BFTC' or layout == 'BCFT':
            kernel_size = kernel_size[::-1]
            strides = strides[::-1]
        if conv_type == 'conv1d':
            if (layer_id == 0):
                inputs = self.add_expand_dims(inputs, 2, output_name_prefix+"_conv1d_expand")
            strides.append(1)
            kernel_size.append(1)
        shape = self.get_tensor_shape(inputs)
        padding[1] = math.ceil((shape[h_axis] + padding[0] + padding[1] - kernel_size[0] + 1) / strides[0]) * strides[0] + kernel_size[0] - 1 - padding[0] - shape[h_axis]
        padding[3] = math.ceil((shape[w_axis] + padding[2] + padding[3] - kernel_size[1] + 1) / strides[1]) * strides[1] + kernel_size[1] - 1 - padding[2] - shape[w_axis]
        if conv_type == 'conv_proj_res':
            print("[ERROR] currently not support conv_proj in wonv_wrapper_nchwc8")
            exit(1)
        else:
            inputs = self.conv_bn_actv_nchwc8(
                layer_type=conv_type,
                name=name,
                inputs=inputs,
                filters=ch_out,
                kernel_size=kernel_size,
                activation_fn=act,
                strides=strides,
                padding=padding,
                norm_type=norm_type,
                gn_group_num=gn_group_num,
                data_format=data_format,
                layer_id=layer_id,
                layer_num=layer_num,
                scope_id=scope_id,
                output_name_prefix=output_name_prefix
            )
        return inputs, new_mems

    def rel_shift(self, x):
        x = self.add_relative_shift(x, x+"_rel_shift", axis=3, shift_length=1)
        #x_size = self.get_tensor_shape(x)
        #x = self.add_pad(x, x+"_pad", [[0, 0], [0, 0], [0, 0], [1, 0]])
        #x = self.add_reshape(x, x+"_r", [x_size[0], x_size[1], x_size[3] + 1, x_size[2]])
        #result_name = x + "_slice"
        #self.add_slice(x, [x+"_other", result_name], axis=2, slice_point=[1])
        #x = self.add_reshape(result_name, x+"_rel_shif", x_size)
        return x

    def rel_multihead_attn(self, w, r, r_w_bias, r_r_bias, attn_mask, mems, d_model,
                           n_head, d_head, dropout, dropatt, is_training,
                           kernel_initializer, scope='rel_attn', norm_type='ln',
                           use_mq_attn=False, use_xl_pos_enc=True,
                           attn_mask_parameters=None,
                           pos_emb_parameters=None,
                           scope_id=-1, output_name_prefix=""):
        if is_training:
          assert mems is None
    
        scale = 1 / (d_head ** 0.5)
        self.scopes[scope_id] = scope
        query_depth = n_head * d_head
        key_depth = n_head * d_head
        value_depth = n_head * d_head
        if norm_type == 'bn':
            #w_t = self.add_transpose(w, output_name_prefix+"_bn_pre_t", [0, 2, 1])
            #w_t = self.add_expand_dims(w_t, axis=3, output_name=output_name_prefix+"_bn_expand")
            #w_norm = self.extract_batch_norm(w_t, output_name_prefix+"_bn", scope_id+1,
            #             data_format="NCHW", layer_names=["batch_normalization", "moving_mean", "moving_variance"])
            #w_norm = self.add_squeeze(w_norm, axis=3, output_name=output_name_prefix+"_bn_squeeze")
            #w_norm = self.add_transpose(w_norm, output_name_prefix+"_bn_post_t", [0, 2, 1])
            w_norm = self.extract_batch_norm(w, output_name_prefix+"_bn", scope_id+1,
                         data_format="NCHW", axis=-1, layer_names=["batch_normalization", "moving_mean", "moving_variance"])
        elif norm_type == 'pre_ln':
            w_norm = self.extract_layer_norm(w, output_name_prefix+"_ln", scope_id+1, ["LayerNorm", "gamma", "beta"])
        else:
            assert norm_type == 'ln'
            w_norm = w
        self.add_quantization(scope_id+1, "quant_attn_input", w_norm)
        if use_mq_attn:
            w_head_q = output_name_prefix + "_multihead_q"
            w_head_k = output_name_prefix + "_multihead_k"
            w_head_v = output_name_prefix + "_multihead_v"
            self.extract_dense(w_norm, scope_id+1, "mhead_q")
            self.extract_denses(w_norm, [w_head_k, w_head_v], [key_depth, value_depth], scope_id+1, "shead_kv")
            self.add_quantization(scope_id+1, "quant_heads_q", w_head_q)
            self.add_quantization(scope_id+1, "quant_heads_kv", w_head_k)
            self.add_quantization(scope_id+1, "quant_heads_kv", w_head_v)
            w_head_q = self.add_reshape(w_head_q, w_head_q+"_r", [self.batch, -1, n_head, d_head])
            w_head_k = self.add_reshape(w_head_k, w_head_k+"_r", [self.batch, -1, d_head, 1])
            w_head_v = self.add_reshape(w_head_v, w_head_v+"_r", [self.batch, -1, d_head])
        else:
            w_head_q = output_name_prefix + "_multihead_q"
            w_head_k = output_name_prefix + "_multihead_k"
            w_head_v = output_name_prefix + "_multihead_v"
            self.extract_denses(w_norm, [w_head_q, w_head_k, w_head_v], [key_depth, key_depth, value_depth], scope_id+1, "qkv")
            self.add_quantization(scope_id+1, "quant_heads_qkv", w_head_q)
            self.add_quantization(scope_id+1, "quant_heads_qkv", w_head_k)
            self.add_quantization(scope_id+1, "quant_heads_qkv", w_head_v)
            w_head_q = self.add_reshape(w_head_q, w_head_q+"_r", [self.batch, -1, n_head, d_head])
            w_head_k = self.add_reshape(w_head_k, w_head_k+"_r", [self.batch, -1, n_head, d_head])
            w_head_v = self.add_reshape(w_head_v, w_head_v+"_r", [self.batch, -1, n_head, d_head])
        if mems is not None:
            k_mems, v_mems = mems
            w_head_k = self.add_concat([k_mems, w_head_k], w_head_k+"_concat", axis=1)
            w_head_v = self.add_concat([v_mems, w_head_v], w_head_v+"_concat", axis=1)
            new_mems = w_head_k, w_head_v
        else:
            new_mems = None

        if (r is None):
            r = self.add_relative_position_embedding(w_head_k, pos_emb_parameters, axis=1, output_name=output_name_prefix+"_rel_pos_emb")
        if use_xl_pos_enc:
            r_head_k = self.extract_dense(r, output_name_prefix+"_multihead_r", scope_id+1, scope_name="r")
        else:
            r_head_k = r
        r_head_k = self.add_reshape(r_head_k, r_head_k+"_r", [self.batch, -1, n_head, d_head])
        if use_xl_pos_enc:
            rw_head_q = self.add_sum([w_head_q, r_w_bias], w_head_q+"_rw")
            rr_head_q = self.add_sum([w_head_q, r_r_bias], w_head_q+"_rr")
        else:
            rw_head_q = w_head_q
            rr_head_q = w_head_q

        if use_mq_attn:
            rw_head_qt = self.add_transpose(rw_head_q, rw_head_q+"_t", [0, 2, 1, 3])
            w_head_kt = self.add_transpose(w_head_k, w_head_k+"_t", [0, 2, 3, 1])
            AC = self.add_matmul(rw_head_qt, w_head_kt, output_name_prefix+"_AC")
        else:
            rw_head_qt = self.add_transpose(rw_head_q, rw_head_q+"_t", [0, 2, 1, 3])
            w_head_kt = self.add_transpose(w_head_k, w_head_k+"_t", [0, 2, 3, 1])
            AC = self.add_matmul(rw_head_qt, w_head_kt, output_name_prefix+"_AC")
        if use_xl_pos_enc:
            rr_head_qt = self.add_transpose(rr_head_q, rr_head_q+"_t", [0, 2, 1, 3])
        else:
            rr_head_qt = rw_head_qt
        r_head_kt = self.add_transpose(r_head_k, r_head_k+"_t", [0, 2, 3, 1])
        BD = self.add_matmul(rr_head_qt, r_head_kt, output_name_prefix+"_BD")
        BD = self.rel_shift(BD)

        attn_score = self.add_sum([AC, BD], output_name_prefix+"_ACBD")
        attn_score = self.add_power(attn_score, output_name_prefix+"_ACBD_s", scale=scale)
        if (attn_mask is None):
            attn_trunc_len, same_length = attn_mask_parameters
            if attn_trunc_len is not None:
                attn_score = self.add_attention_mask(attn_score, attn_score+"_mask", attn_trunc_len, same_length, 1e30)
        attn_prob = self.add_softmax(attn_score, attn_score+"_softmax", 3)
        self.add_quantization(scope_id+1, "quant_attn_prob", attn_prob)

        if use_mq_attn:
            w_head_vt = self.add_transpose(w_head_v, w_head_v+"_t", [0, 2, 1, 3])
            attn_vec = self.add_matmul(attn_prob, w_head_vt, output_name_prefix+"_cont")
        else:
            w_head_vt = self.add_transpose(w_head_v, w_head_v+"_t", [0, 2, 1, 3])
            attn_vec = self.add_matmul(attn_prob, w_head_vt, output_name_prefix+"_cont")

        attn_vec = self.add_transpose(attn_vec,  output_name_prefix+"_cont_t", [0, 2, 1, 3])
        attn_vec = self.add_reshape(attn_vec, output_name_prefix+"_cont_r", [self.batch, -1, n_head*d_head])
        self.add_quantization(scope_id+1, "quant_attn_vec", attn_vec)
        attn_out = self.extract_dense(attn_vec, attn_vec+"_fc", scope_id+1, scope_name="o")
        output = self.add_sum([attn_out, w], output_name_prefix+"_rel_multihead_sum")
        if norm_type == 'ln':
            output = self.extract_layer_norm(output, output+"_ln2", scope_id+1, ["LayerNorm", "gamma", "beta"])
        else:
            assert norm_type in ['bn', 'pre_ln']
            output = output
        return output, new_mems

    def _cache_decode_mem(self, curr_kv, prev_mem, mem_len=0, output_name_prefix=""):
        assert prev_mem is not None
        assert mem_len >= 0
        k_mem, v_mem = curr_kv
        k_prev_mem, v_prev_mem = prev_mem
        new_k_mem = k_mem #self.add_concat([k_prev_mem, k_mem], output_name_prefix+"_concat_k", 1)
        new_v_mem = v_mem #self.add_concat([v_prev_mem, v_mem], output_name_prefix+"_concat_v", 1)
        if mem_len > 0:
            assert mem_len > 0
            new_k_mem = self.neg_slice(new_k_mem, axis=1, length=mem_len, output_name_prefix=output_name_prefix+"_k")
            new_v_mem = self.neg_slice(new_v_mem, axis=1, length=mem_len, output_name_prefix=output_name_prefix+"_v")
        return (new_k_mem, new_v_mem)

    def transformer_block(self, input, n_layer, d_model, n_head, d_head, d_inner, dropout, dropatt, dropinp, initializer, mode,
                          mems=None, att_trunc_len=0, pos_emb_cache=None, same_length=False, clamp_len=-1, untie_r=False,
                          scope='transformer', norm_type='ln', output_norm_type='', use_mq_attn=False, pre_compute=False,
                          use_xl_pos_enc=True, mult_query_decode=False,
                          scope_id=-1, output_name_prefix=""):
        print('[INFO] transformer block params: n_layer: {}, d_model: {}, n_head: {}, d_head: {}, d_inner: {}, dropout: {}, dropatt: {}, dropinp: {}'\
            .format(n_layer, d_model, n_head, d_head, d_inner, dropout, dropatt, dropinp))
        assert mode in ['train', 'eval', 'infer']
        is_training = mode == 'train'
        is_decoding = mems is not None
        if is_decoding:
            assert mode == 'infer'
        if isinstance(d_inner, int):
            d_inner = [d_inner] * n_layer
        self.scopes[scope_id] = scope
        if use_xl_pos_enc:
            print("[ERROR] currently not support xl pos encoding")
            exit(1)
            #if untie_r:
            #    r_w_bias = tf.get_variable('r_w_bias', [n_layer, n_head, d_head],
            #                             initializer=initializer)
            #    r_r_bias = tf.get_variable('r_r_bias', [n_layer, n_head, d_head],
            #                               initializer=initializer)
            #else:
            #    r_w_bias = tf.get_variable('r_w_bias', [n_head, d_head],
            #                               initializer=initializer)
            #    r_r_bias = tf.get_variable('r_r_bias', [n_head, d_head],
            #                               initializer=initializer)
        else:
            if untie_r:
                r_w_bias = [None] * n_layer
                r_r_bias = [None] * n_layer
            else:
                r_w_bias = None
                r_r_bias = None
        is_decoding_var_att_trunc = is_decoding and not isinstance(att_trunc_len, int)
        #qlen = self.get_tensor_shape(input)[0]
        #if is_decoding_var_att_trunc:
        #    mlen = [self.get_tensor_shape(mems[l_i][0])[0] for l_i in range(n_layer)]
        #    klen = [mlen[l_i] + qlen for l_i in range(n_layer)]
        #else:
        #    mlen = self.get_tensor_shape(mems[0][0])[0] if mems is not None else 0
        #    klen = mlen + qlen
        if not is_decoding:
            assert mems is None
            #assert mlen == 0
            assert not same_length

        attn_mask = [None] * n_layer
        if is_decoding_var_att_trunc:
            #if pre_compute and mode == 'infer':
            #    if mult_query_decode:
            #        print('precompute multi query attn mask for decoding var att')
            #        attn_mask_caches = [tf.constant(mask_cache(8, 32, trunc_len), dtype=tf.float32) for trunc_len in att_trunc_len]
            #        attn_mask = [neg_slice_m(attn_mask_cache, [qlen, qlen+mlen[l_i]]) for l_i, attn_mask_cache in enumerate(attn_mask_caches)]
            #    else:
            #        print('precompute attn mask for decoding var att')
            #        # todo remove unnecessary mask
            #        attn_mask = [tf.zeros([qlen, qlen + mlen[l_i]]) for l_i in range(n_layer)]
            #else:
            #    attn_mask = [_create_mask(qlen, mlen[l_i], att_trunc_len=att_trunc_len[l_i], same_length=same_length) for l_i in range(n_layer)]
            attn_mask_parameters = []
            for l_i in range(n_layer):
                trunc_len = att_trunc_len[l_i]
                if pre_compute and mode == 'infer':
                    if not mult_query_decode:
                        trunc_len = None #-1
                attn_mask_parameters.append((trunc_len, same_length))
        elif not isinstance(att_trunc_len, int):
            assert len(att_trunc_len) == n_layer
            assert not is_decoding
            #if pre_compute and mode == 'infer':
            #    print('precompute attn mask')
            #    attn_mask_caches = [tf.constant(mask_cache(1024, 0, trunc_len), dtype=tf.float32) for trunc_len in att_trunc_len]
            #    attn_mask = [attn_mask_cache[:qlen, :qlen] for attn_mask_cache in attn_mask_caches]
            #else:
            #    attn_mask = [_create_mask(qlen, mlen, att_trunc_len=trunc_len, same_length=same_length) for trunc_len in att_trunc_len]
            attn_mask_parameters = []
            for l_i in range(n_layer):
                trunc_len = att_trunc_len[l_i]
                attn_mask_parameters.append((trunc_len, same_length))
        else:
            #attn_mask = _create_mask(qlen, mlen, att_trunc_len=att_trunc_len, same_length=same_length)
            #attn_mask = [attn_mask] * n_layer
            att_trunc_len = [att_trunc_len] * n_layer
            attn_mask_parameters = [(att_trunc_len, same_length)] * n_layer
        if use_xl_pos_enc:
            #if is_decoding_var_att_trunc:
            #    pos_emb = [neg_slice(pos_emb_cache, klen[l_i]) for l_i in range(n_layer)]
            #else:
            #    if pos_emb_cache is None:
            #        pos_seq = tf.range(klen - 1, -1, -1.0)
            #        if clamp_len > 0:
            #            pos_seq = tf.minimum(pos_seq, clamp_len)
            #        inv_freq = 1 / (10000 ** (tf.range(0, d_model, 2.0) / d_model))
            #        pos_emb = positional_embedding(pos_seq, inv_freq)
            #    else:
            #        pos_emb = neg_slice(pos_emb_cache, klen)
            pos_emb_parameters = [pos_emb_cache] * n_layer
            pos_emb = [None] * n_layers
        else:
            #if not isinstance(att_trunc_len, int):
            #    max_relative_position = [att_trunc_len[i] + 1 for i in range(n_layer)]
            #else:
            #    max_relative_position = [(att_trunc_len + 1) if att_trunc_len > 0 else 16] * n_layer
            #pos_emb = [get_shaw_relative_embeddings_left(max_relative_position[i],
            #                                             klen[i] if is_decoding_var_att_trunc else klen,
            #                                             d_model,
            #                                             'rel_pos_emb_{}'.format(i))
            #           for i in range(n_layer)]
            pos_emb_parameters = []
            for i in range(n_layer):
                self.scopes[scope_id+1] = "rel_pos_emb_" + str(i)
                weight_name = output_name_prefix + "_rel_pos_emb_" + str(i)
                self.add_weight(weight_name, scope_id=scope_id+2, weight_name=None, weight=None, transpose=None, data_type="FLOAT32")
                pos_emb_parameters.append(weight_name)
            pos_emb = [None] * n_layer

        if mems is None:
            mems = [None] * n_layer
        output=input
        new_mems = []
        for i in range(n_layer):
            self.scopes[scope_id+1] = "layer_" + str(i)
            output, kv_mems = self.rel_multihead_attn(
                w=output,
                r=pos_emb[i] if (is_decoding_var_att_trunc or not use_xl_pos_enc) else pos_emb,
                r_w_bias=r_w_bias if not untie_r else r_w_bias[i],
                r_r_bias=r_r_bias if not untie_r else r_r_bias[i],
                attn_mask=attn_mask[i],
                mems=mems[i],
                d_model=d_model,
                n_head=n_head,
                d_head=d_head,
                dropout=dropout,
                dropatt=dropatt,
                is_training=is_training,
                kernel_initializer=initializer,
                norm_type=norm_type,
                use_mq_attn=use_mq_attn,
                use_xl_pos_enc=use_xl_pos_enc,
                attn_mask_parameters=attn_mask_parameters[i],
                pos_emb_parameters=pos_emb_parameters[i],
                scope_id=scope_id+2,
                output_name_prefix=output_name_prefix+"_layer"+str(i))

            # cache new mems
            if is_decoding:
                new_mems.append(self._cache_decode_mem(kv_mems, mems[i], att_trunc_len[i], output_name_prefix=output_name_prefix+"_layer"+str(i)))

            output = self.positionwise_FF(
                input=output,
                d_model=d_model,
                d_inner=d_inner[i],
                dropout=dropout,
                kernel_initializer=initializer,
                is_training=is_training,
                norm_type=norm_type,
                scope_id=scope_id+2,
                output_name_prefix=output_name_prefix+"_layer"+str(i))

        if output_norm_type == 'ln':
            output = self.extract_layer_norm(output, output_name_prefix+"_ln", scope_id+1)
        elif output_norm_type == 'bn':
            #output_t = self.add_transpose(output, output_name_prefix+"_bn_pre_t", [0, 2, 1])
            #output_t = self.add_expand_dims(output_t, axis=3, output_name=output_name_prefix+"_bn_expand")
            #output = self.extract_batch_norm(output_t, output_name_prefix+"_bn", scope_id+1, data_format="NCHW", layer_names=["batch_normalization", "moving_mean", "moving_variance"])
            #output = self.add_squeeze(output, axis=3, output_name=output_name_prefix+"_bn_squeeze")
            #output = self.add_transpose(output, output_name_prefix+"_bn_post_t", [0, 2, 1])
            output = self.extract_batch_norm(output, output_name_prefix+"_bn", scope_id+1,
                data_format="NCHW", axis=-1, layer_names=["batch_normalization", "moving_mean", "moving_variance"])
        else:
          assert output_norm_type == ''

        if not is_decoding:
            assert len(new_mems) == 0
        return output, new_mems

    def proj_transformer_block(self, input, n_layer, d_model, n_head, d_head, d_inner, dropout, dropatt, dropinp, initializer, mode,
                               input_project=False, decode_state=None, att_trunc_len=0, pos_emb_cache=None, norm_type='ln', output_norm_type='',
                               use_mq_attn=False, pre_compute=False, use_xl_pos_enc=True, mult_query_decode=False, scope_id=-1, output_name_prefix=""):
        self.scopes[scope_id] = 'transformer_block'
        orig_input_dim = self.get_tensor_shape(input)[-1]
        if input_project:
            assert orig_input_dim != d_model
            input = self.extract_dense(input, output_name_prefix+"_proj", scope_id+1, scope_name="input_proj")
        else:
            assert orig_input_dim == d_model
        # [B, T, C] --> [T, B, C]
        #input = self.add_transpose(input, output_name_prefix+"_pre_t", [1, 0, 2])
        output, new_decode_state = self.transformer_block(input,
                                                     n_layer=n_layer, d_model=d_model, n_head=n_head, d_head=d_head, d_inner=d_inner,
                                                     dropout=dropout, dropatt=dropatt, dropinp=dropinp, initializer=initializer, mode=mode,
                                                     mems=decode_state, att_trunc_len=att_trunc_len, pos_emb_cache=pos_emb_cache,
                                                     norm_type=norm_type, output_norm_type=output_norm_type, use_mq_attn=use_mq_attn,
                                                     pre_compute=pre_compute, use_xl_pos_enc=use_xl_pos_enc, mult_query_decode=mult_query_decode,
                                                     scope_id=scope_id+1, output_name_prefix=output_name_prefix)
        # [T, B, C] --> [B, T, C]
        #output = self.add_transpose(output, output_name_prefix+"_post_t", [1, 0, 2])
        return output, new_decode_state

    def transformer_block_wrapper(self, input, initializer, mode, transformer_block_params, decode_state=None, pos_emb_cache=None,
                              pre_compute=False, mult_query_decode=False,
                              scope_id=-1, output_name_prefix=""):
        n_layer = transformer_block_params['n_layer']
        d_model = transformer_block_params['d_model']
        n_head = transformer_block_params['n_head']
        d_head = transformer_block_params['d_head']
        d_inner = transformer_block_params['d_inner']
        dropout_keep_prob = transformer_block_params['dropout_keep_prob']
        input_keep_prob = transformer_block_params.get('input_keep_prob', dropout_keep_prob)
        att_keep_prob = transformer_block_params.get('att_keep_prob', dropout_keep_prob)
        input_project = transformer_block_params.get('input_project', False)
        att_trunc_len = transformer_block_params.get('att_trunc_len', 0)
        norm_type = transformer_block_params.get('norm_type', 'ln')
        output_norm_type = transformer_block_params.get('output_norm_type', '')
        use_mq_attn = transformer_block_params.get('use_mq_attn', False)
        use_xl_pos_enc = transformer_block_params.get('use_xl_pos_enc', True)
    
        valid_params = {'n_layer', 'd_model', 'n_head', 'd_head', 'd_inner', 'dropout_keep_prob', 'input_keep_prob',
                        'att_keep_prob', 'input_project', 'att_trunc_len', 'norm_type', 'output_norm_type', 'use_mq_attn',
                        'use_xl_pos_enc'}
        for k in transformer_block_params.keys():
            if k not in valid_params:
                raise ValueError('unknown transformer parameter: {}'.format(k))
    
        output, new_decode_state = self.proj_transformer_block(input, n_layer=n_layer, d_model=d_model, n_head=n_head, d_head=d_head,
                                        d_inner=d_inner, dropout=1.0-dropout_keep_prob, dropatt=1.0-att_keep_prob,
                                        dropinp=1.0 - input_keep_prob,
                                        initializer=initializer, mode=mode,
                                        input_project=input_project, decode_state=decode_state,
                                        att_trunc_len=att_trunc_len, pos_emb_cache=pos_emb_cache,
                                        norm_type=norm_type, output_norm_type=output_norm_type,
                                        use_mq_attn=use_mq_attn,
                                        pre_compute=pre_compute,
                                        use_xl_pos_enc=use_xl_pos_enc,
                                        mult_query_decode=mult_query_decode,
                                        scope_id=scope_id,
                                        output_name_prefix=output_name_prefix)
        return output, new_decode_state

    def _transformer_block(self, input, transformer_block_params, initializer, pos_emb_cache=None, decode_state=None, scope_id=-1, output_name_prefix=""):
        output, new_decode_state = self.transformer_block_wrapper(input,
                                              initializer=initializer,
                                              mode=self.mode,
                                              transformer_block_params=transformer_block_params,
                                              pos_emb_cache=pos_emb_cache,
                                              pre_compute=None,
                                              decode_state=decode_state,
                                              mult_query_decode=True,
                                              scope_id=scope_id,
                                              output_name_prefix=output_name_prefix)
        return output, new_decode_state

    def prepare_convolution_states(self, conv_layers, output_name_prefix, init_with_none=False):
        conv_states = []
        for i in range(len(conv_layers)):
            if (init_with_none):
                state_shape = [0] * 4
            else:
                state_shape = conv_layers[i].get('states', None)
                if (self.nchwc8 and i != 0 and state_shape is not None):
                    if (len(state_shape) == 3):
                        tmp = state_shape[1]
                        state_shape[1] = state_shape[2]
                        state_shape[2] = tmp
                        state_shape.append(1)
                    elif (len(state_shape) == 4):
                        tmp = state_shape[3]
                        state_shape[3] = state_shape[2]
                        state_shape[2] = state_shape[1]
                        state_shape[1] = tmp
                    else:
                        print("[ERROR] unsupported state shape %d" % (len(state_shape)))
                        exit(1)
            if (state_shape is not None):
                state_name = output_name_prefix + "_layer" + str(i) + "_mem"
                if (self.first_frame):
                    data = {state_name: np.zeros(state_shape)}
                else:
                    file_data = np.load(self.state_data_path + "/" + state_name + ".npy")
                    data = {state_name: file_data}
                    state_shape = file_data.shape
                #self.add_memory(state_name, state_shape, data_type="FLOAT32")
                self.add_input(state_name, state_shape)
                self.set_input(data)
                conv_states.append(state_name)
            else:
                conv_states.append(None)
        return conv_states

    def prepare_transformer_states(self, transformer_layers, output_name_prefix, init_with_none=False):
        transformer_states = []
        layers = transformer_layers.get('n_layer', 0)
        attn_trunc_lens = transformer_layers.get("att_trunc_len", [None]*layers)
        n_head = transformer_layers['n_head']
        d_head = transformer_layers['d_head']
        for i in range(layers):
            if (attn_trunc_lens[i] is not None):
                kmem_name = output_name_prefix + "_layer" + str(i) + "_kmem"
                vmem_name = output_name_prefix + "_layer" + str(i) + "_vmem"
                if (init_with_none):
                    state_shape = [0] * 4
                else:
                    state_shape = [1, attn_trunc_lens[i], n_head, d_head]
                if (self.first_frame):
                    data = {kmem_name: np.zeros(state_shape),
                            vmem_name: np.zeros(state_shape)}
                    state_shape_k = state_shape
                    state_shape_v = state_shape
                else:
                    file_data_k = np.load(self.state_data_path + "/" + kmem_name + ".npy")
                    file_data_v = np.load(self.state_data_path + "/" + vmem_name + ".npy")
                    data = {kmem_name: file_data_k,
                            vmem_name: file_data_v}
                    state_shape_k = file_data_k.shape
                    state_shape_v = file_data_v.shape
                #self.add_memory(kmem_name, state_shape, data_type="FLOAT32")
                #self.add_memory(vmem_name, state_shape, data_type="FLOAT32")
                self.add_input(kmem_name, state_shape_k)
                self.add_input(vmem_name, state_shape_v)
                self.set_input(data)
            else:
                kmem_name = None
                vmem_name = None
            transformer_states.append((kmem_name, vmem_name))
        return transformer_states

    def prepare_states(self, params, output_name_prefix,
        init_convolution_with_none=False,
        init_transformer_with_none=False,
        block_id_start=0,
        block_id_end=-1):
        block_states = []
        if (block_id_end == -1):
            block_id_end = len(params['net_blocks'])
        for block_i in range(block_id_start, block_id_end):
            block_params = params['net_blocks'][block_i]
            block_state = []
            trunk_i = -1
            if ('conv_layers' in block_params.keys()):
                trunk_i = trunk_i + 1
                conv_layers = block_params['conv_layers']
                block_state.append(self.prepare_convolution_states(conv_layers,
                    output_name_prefix+"_block"+str(block_i)+"_trunk"+str(trunk_i),
                    init_convolution_with_none))

            if ('transformer_block_params' in block_params.keys()):
                trunk_i = trunk_i + 1
                transformer_layers = block_params['transformer_block_params']
                block_state.append(self.prepare_transformer_states(transformer_layers,
                    output_name_prefix+"_block"+str(block_i)+"_trunk"+str(trunk_i),
                    init_transformer_with_none))
            block_states.append(block_state)
        return block_states

    def extract_encoder(self, input_dict, scope_id, block_id_start=0, block_id_end=-1):
        self.params = self.base_params["encoder_params"]
        self.scopes[scope_id] = "ds2_encoder"
        source_sequence = input_dict['source_tensors']
        decoding_states = input_dict.get('decoding_states', None)
        is_decoding = decoding_states is not None
        self._pre_compute = False
        if is_decoding:
            assert self.mode == 'infer'

        data_format = self.params.get('data_format', 'channels_last')
        use_output_fc = self.params.get('output_fc', False)
        output_norm_type = self.params.get('output_norm_type', '')
        residual_type = self.params.get('residual_type', None)
        use_group_dense = residual_type == 'group_dense'
        if self.params['rnn_type'] not in ['wd_cudnn_lstm']:
            assert residual_type is None

        #max_len = tf.reduce_max(src_length) + self._get_additional_pad_num(src_length)

        #if self.params['use_conv_mask']:
        #    mask = tf.sequence_mask(
        #      lengths=src_length, maxlen=max_len,
        #      dtype=source_sequence.dtype
        #    )
        #    mask = tf.expand_dims(mask, 2)
        #else:
        #    mask = None
        mask = None

        # BTF -> BCTF
        if (block_id_start == 0):
            input_layer = self.add_expand_dims(source_sequence, axis=-1, output_name="encoder_input_expand")
        else:
            input_layer = source_sequence

        assert data_format == 'channels_last'
        if data_format=='channels_last' or data_format=='BTFC':
            layout  = 'BTFC'
            dformat = 'channels_last'
        elif data_format=='channels_first' or data_format=='BCTF':
            layout  = 'BCTF'
            dformat = 'channels_first'
        elif data_format=='BFTC':
            layout  = 'BFTC'
            dformat = 'channels_last'
        elif data_format=='BCFT':
            layout  = 'BCFT'
            dformat = 'channels_first'
        else:
            print("[WARNING] unsupported data format: will use channels_last (BTFC) instead")
            layout  = 'BTFC'
            dformat = 'channels_last'

        #input_layer is BTFC
        if   layout == 'BCTF':
           top_layer = self.add_transpose(input_layer, "encoder_input_transpose", [0, 3, 1, 2])
        elif layout == 'BFTC':
           top_layer = self.add_transpose(input_layer, "encoder_input_transpose", [0, 2, 1, 3])
        elif layout == 'BCFT':
           top_layer = self.add_transpose(input_layer, "encoder_input_transpose", [0, 3, 2, 1])
        else:
           top_layer = input_layer
        new_decode_states = []
        if (block_id_end == -1):
            block_id_end = len(self.params['net_blocks']);
        for block_i in range(block_id_start, block_id_end):
            block_local_id = block_i - block_id_start
            block_params = self.params['net_blocks'][block_i]
            self.scopes[scope_id+1] = "block_" + str(block_i)
            output_name_prefix = "encoder_block" + str(block_i)
            new_block_decode_states = []
            conv_layers = block_params['conv_layers']
            block_conv_type = block_params.get('conv_type', 'conv1d')
            use_2d_conv = block_conv_type == 'conv2d'
            gn_group_num = block_params.get('gn_group_num', [0] * len(conv_layers))

            # ----- Convolutional layers ---------------------------------------------
            new_conv_decode_states = []
            for idx_conv in range(len(conv_layers)):
                conv_layer_params = conv_layers[idx_conv]
                conv_type = conv_layer_params.get('conv_type', block_conv_type)
                conv_layer_params.setdefault('activation_fn', self.params['activation_fn'])

                if (self.nchwc8):
                    top_layer, new_conv_decode_state_i = self.conv_layer_wrapper_nchwc8(
                        inputs=top_layer,
                        conv_type=conv_type,
                        conv_layer_params=conv_layer_params,
                        gn_group_num=gn_group_num[idx_conv],
                        name="conv{}".format(idx_conv + 1),
                        #inputs_len=src_length,
                        mask=mask,
                        #max_len=max_len,
                        layout=layout,
                        data_format=dformat,
                        decode_state=decoding_states[block_local_id][0][idx_conv] if is_decoding else None,
                        is_decoding=is_decoding,
                        layer_id=idx_conv,
                        layer_num=len(conv_layers),
                        scope_id=scope_id+2,
                        output_name_prefix=output_name_prefix+"_conv"+str(idx_conv)
                    )
                else:
                    top_layer, new_conv_decode_state_i = self.conv_layer_wrapper(
                        inputs=top_layer,
                        conv_type=conv_type,
                        conv_layer_params=conv_layer_params,
                        gn_group_num=gn_group_num[idx_conv],
                        name="conv{}".format(idx_conv + 1),
                        #inputs_len=src_length,
                        mask=mask,
                        #max_len=max_len,
                        layout=layout,
                        data_format=dformat,
                        decode_state=decoding_states[block_local_id][0][idx_conv] if is_decoding else None,
                        is_decoding=is_decoding,
                        scope_id=scope_id+2,
                        output_name_prefix=output_name_prefix+"_conv"+str(idx_conv)
                    )
                new_conv_decode_states.append(new_conv_decode_state_i)
                #if (block_i == 2 and idx_conv == 2):
                #    exit(1)
            new_block_decode_states.append(new_conv_decode_states)

            # convert layout --> BTFC
            if data_format == 'channels_first':
                top_layer = self.add_transpose(top_layer, output_name_prefix+"_t", [0, 2, 3, 1])

            if   layout == 'BCTF': # BCTF --> BTFC
                top_layer = self.add_transpose(top_layer, output_name_prefix+"_t", [0, 2, 3, 1])
            elif layout == 'BFTC': # BFTC --> BTFC
                top_layer = self.add_transpose(top_layer, output_name_prefix+"_t", [0, 2, 1, 3])
            elif layout == 'BCFT': # BCFT --> BTFC
                top_layer = self.add_transpose(top_layer, output_name_prefix+"_t", [0, 3, 2, 1])

            num_rnn_layers = block_params['num_rnn_layers']
            group_dense_group_spec = block_params.get('group_dense_group_spec', None)
            if use_group_dense:
                print("[ERROR] currently not support group dense")
                exit(1)

            ## reshape to [B, T, FxC]
            if use_2d_conv:
                f = self.get_tensor_shape(top_layer)[2]
                c = self.get_tensor_shape(top_layer)[3]
                fc = f * c
                top_layer = self.add_reshape(top_layer, output_name_prefix+"_r", [self.batch, -1, fc])

            if self.params.get('ln_fc_after_conv', False):
                assert not use_group_dense
                top_layer = self.layer_normed_fc(top_layer, "relu", output_name_prefix, scope_id+1)

            # ----- RNN ---------------------------------------------------------------
            if num_rnn_layers > 0:
                print("[ERROR] currently not support RNN")
                exit(1)

            transformer_block_params = block_params.get('transformer_block_params', None)
            if transformer_block_params:
                if self._pre_compute and self.mode == 'infer':
                    print("[ERROR] currently not support pre_compute in encoder")
                    exit(1)
                else:
                    pos_emb_cache = None

                if is_decoding:
                    transformer_decode_state = decoding_states[block_local_id][-1]
                    assert transformer_decode_state is not None
                else:
                    transformer_decode_state = None
                initializer = None
                top_layer, new_transformer_decode_state = self._transformer_block(top_layer, transformer_block_params,
                                                    initializer=initializer,
                                                    pos_emb_cache=pos_emb_cache, decode_state=transformer_decode_state,
                                                    scope_id=scope_id+2,
                                                    output_name_prefix=output_name_prefix+"_transformer")
                new_block_decode_states.append(new_transformer_decode_state)
                #if (block_i == 3):
                #    exit(1)
            new_decode_states.append(new_block_decode_states)

        if self.params['row_conv']:
            channels = self.get_tensor_shape(top_layer)[-1]
            top_layer = row_conv(
                name="row_conv",
                input_layer=top_layer,
                batch=self.batch,
                channels=channels,
                activation_fn=self.params['activation_fn'],
                width=self.params['row_conv_width'],
                data_format=data_format,
                norm_type=self.params.get('norm_type', 'batch_norm'),
                gn_group_num=gn_group_num[-1]
            )

        if use_output_fc:
            assert output_norm_type == ''
            output_norm_type = 'ln_fc'
        if output_norm_type == 'ln_fc':
            #c = self.get_tensor_shape(top_layer)[-1]
            #top_layer = self.add_reshape(top_layer, "encoder_output_r", [-1, c])
            outputs = self.layer_normed_fc(top_layer, self.params['activation_fn'], "encoder_output", scope_id+1)
        elif output_norm_type == 'layer_norm':
            outputs = self.extract_layer_norm(top_layer, "encoder_output_ln", scope_id+1)
        elif output_norm_type == 'batch_norm':
            outputs = self.extract_batch_norm(top_layer, "encoder_output_bn", scope_id+1, data_format="NCHW")
        else:
            outputs = top_layer
        return {
            'outputs': outputs,
            'decode_state': new_decode_states
        }

    def extract_prediction_net(self, input_dict, scope_id):
        pred_net_params = self.base_params["decoder_params"]['pred_net_params']
        assert self.mode == 'infer'
        source_sequence = input_dict['source_tensors']
        decoding_states = input_dict.get('decoding_states', None)
        assert decoding_states is not None
        embedded_inputs = "prediction_net_embedding"
        self.extract_embedding(source_sequence, scope_id, "PredNetEmbeddingMatrix", embedded_inputs)

        if pred_net_params['norm_inputs']:
            pred_net_outputs = tf.contrib.layers.layer_norm(embedded_inputs, begin_norm_axis=-1)
        else:
            pred_net_outputs = embedded_inputs
        transformer_block_params = pred_net_params['transformer_block_params']
        init_dict = None
        initializer = None
        pred_net_outputs, new_pred_net_cell_state = self._transformer_block(pred_net_outputs,
                                                   transformer_block_params, initializer=initializer,
                                                   decode_state=decoding_states, pos_emb_cache=None,
                                                   scope_id=scope_id,
                                                   output_name_prefix="prediction_net")
        #pred_net_outputs_last_dim = pred_net_outputs.get_shape().as_list()[-1]
        #pred_net_outputs = tf.reshape(pred_net_outputs, [self.batch, pred_net_outputs_last_dim])
        return {
            'outputs': pred_net_outputs,
            'decode_state': new_pred_net_cell_state
        }

    def extract_joint_net(self, input_dict, scope_id):
        encoder_output = input_dict["encoder"]
        prediction_net_output = input_dict["prediction_net"]
        fc0_name = "joint_encoder_fc"
        self.add_quantization(scope_id, "quant_enc_joint_input", encoder_output)
        self.extract_dense(encoder_output, fc0_name, scope_id, scope_name="joint_encoder_fc")
        #ep0_name = "joint_net_expand0"
        #self.add_expand_dims(fc0_name, axis=2, output_name=ep0_name)
        fc1_name = "joint_pred_net_fc"
        self.add_quantization(scope_id, "quant_pred_joint_input", prediction_net_output)
        self.extract_dense(prediction_net_output, fc1_name, scope_id, scope_name="joint_pred_net_fc")
        #ep1_name = "joint_net_expand1"
        #self.add_expand_dims(fc1_name, axis=1, output_name=ep1_name)
        sum_name = "joint_net_sum"
        #self.add_sum([ep0_name, ep1_name], sum_name)
        self.add_sum([fc0_name, fc1_name], sum_name)
        activation_fn = self.base_params["decoder_params"]["joint_net_params"]["activation_fn"]
        if (activation_fn == "relu"):
            relu_name = "joint_net_relu"
            result_name = self.add_relu(sum_name, relu_name)
        else:
            print("[ERROR] unsupported activation function" % (activation_fn))
            exit(1)
        self.add_quantization(scope_id, "quant_joint_middle", result_name)
        result_name = self.extract_dense(result_name, "joint_output_fc", scope_id, scope_name="joint_output_fc")
        argmax_name = self.add_argmax(result_name, axis=-1, output_name="output_argmax")
        return argmax_name

    def generate_encoder(self, input=None, block_id_start=0, block_id_end=-1):
        sounds_input_name = "sounds"
        if (block_id_start == 0):
            sounds_input_shape = [self.batch, self.base_params["sequence.max_length"], self.base_params["sequence.num_units"]]
        else:
            sounds_input_shape = input[sounds_input_name].shape
        self.add_input(sounds_input_name, sounds_input_shape)
        self.set_input(input)

        input_dict = {'source_tensors': sounds_input_name,
                      'decoding_states': self.prepare_states(self.base_params["encoder_params"],
                           "encoder", False, True, block_id_start, block_id_end)
                     }
        self.save_input()
        output = self.extract_encoder(input_dict, 0, block_id_start, block_id_end)
        self.save_caffe_model()
        if (self.save_state and self.calculate):
            for index in range(len(output['decode_state'])):
                block_i = block_id_start + index
                file_path_prefix_block = self.state_data_path + "/encoder_block" + str(block_i)
                for trunk_i in range(len(output['decode_state'][index])):
                    file_path_prefix_trunk = file_path_prefix_block + "_trunk" + str(trunk_i)
                    for layer_i in range(len(output['decode_state'][index][trunk_i])):
                        data = output['decode_state'][index][trunk_i][layer_i]
                        if (data is None):
                            continue
                        if (isinstance(data, str)):
                            file_path = file_path_prefix_trunk + "_layer" + str(layer_i) + "_mem.npy"
                            np.save(file_path, self.get_tensor(data))
                            print("[INFO] save encoder block %d trunk %d layer %d state to %s" % (block_i, trunk_i, layer_i, file_path))
                        elif (isinstance(data, tuple)):
                            file_k_path = file_path_prefix_trunk + "_layer" + str(layer_i) + "_kmem.npy"
                            file_v_path = file_path_prefix_trunk + "_layer" + str(layer_i) + "_vmem.npy"
                            state_k_data, state_v_data = data
                            np.save(file_k_path, self.get_tensor(state_k_data))
                            np.save(file_v_path, self.get_tensor(state_v_data))
                            print("[INFO] save encoder block %d trunk %d layer %d k state to %s" % (block_i, trunk_i, layer_i, file_k_path))
                            print("[INFO] save encoder block %d trunk %d layer %d v state to %s" % (block_i, trunk_i, layer_i, file_v_path))
                        else:
                            print("[ERROR] unrecognized state array type")
                            exit(1)

    def generate_prediction_net(self, input=None):
        label_input_name = "label"
        label_input_shape = [self.batch, 1]
        self.add_input(label_input_name, label_input_shape)
        self.set_input(input)

        input_dict = {'source_tensors': label_input_name,
                      'decoding_states': self.prepare_transformer_states(
                          self.base_params["decoder_params"]["pred_net_params"]["transformer_block_params"], "prediction_net",
                          True)
                     }
        self.save_input()
        output = self.extract_prediction_net(input_dict, 0)
        self.save_caffe_model()
        if (self.save_state and self.calculate):
            file_path_prefix = self.state_data_path + "/prediction_net"
            for layer_i in range(len(output['decode_state'])):
                data = output['decode_state'][layer_i]
                if (data is None):
                    continue
                if (isinstance(data, str)):
                    file_path = file_path_prefix + "_layer" + str(layer_i) + "_mem.npy"
                    np.save(file_path, self.get_tensor(data))
                    print("[INFO] save prediction net layer %d state to %s" % (layer_i, file_path))
                elif (isinstance(data, tuple)):
                    file_k_path = file_path_prefix + "_layer" + str(layer_i) + "_kmem.npy"
                    file_v_path = file_path_prefix + "_layer" + str(layer_i) + "_vmem.npy"
                    state_k_data, state_v_data = data
                    np.save(file_k_path, self.get_tensor(state_k_data))
                    np.save(file_v_path, self.get_tensor(state_v_data))
                    print("[INFO] save prediction net layer %d k state to %s" % (layer_i, file_k_path))
                    print("[INFO] save prediction net layer %d v state to %s" % (layer_i, file_v_path))
                else:
                    print("[ERROR] unrecognized state array type")
                    exit(1)

    def generate_joint_net(self, input_shape, input=None):
        encoder_output_name = "encoder"
        encoder_output_shape = input_shape["encoder"]
        self.add_input(encoder_output_name, encoder_output_shape)
        prediction_net_output_name = "prediction_net"
        prediction_net_output_shape = input_shape["prediction_net"]
        self.add_input(prediction_net_output_name, prediction_net_output_shape)
        self.set_input(input)

        input_dict = {'encoder': encoder_output_name,
                      'prediction_net': prediction_net_output_name}
        self.save_input()
        self.extract_joint_net(input_dict, 0)
        self.save_caffe_model()
