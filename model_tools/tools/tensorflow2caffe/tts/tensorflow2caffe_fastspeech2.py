#!/usr/local/bin/python
# -*- coding: utf-8 -*-

import math
import sys
sys.path.append("../")
from tensorflow2caffe import Tensorflow2Caffe

class Tensorflow2CaffeFastSpeech2(Tensorflow2Caffe):
    def __init__(self,
            tensorflow_model_path, caffe_model_path_prefix, caffe_model_name,
            config,
            check=False, calc=False):
        Tensorflow2Caffe.__init__(self, tensorflow_model_path,
            caffe_model_path_prefix, caffe_model_name, check, calc)
        self.config = config
        self.is_compatible_encoder = True

    def VariantPredictor(self, encoder_hidden_states, speaker_ids, scope_id, predictor_name, predictor_id):
        output_name_prefix = predictor_name
        self.scopes[scope_id] = predictor_name
        if self.config["n_speakers"] > 1:
            speaker_embeddings = self.extract_embedding(speaker_ids, scope_id, "word_embeddings", output_name_preix+"_embed")
            speaker_fcs = self.extract_dense(speaker_embeddings, output_name_prefix+"_fc", scope_id, proj_word_embedding_names)
            speaker_features = self.add_softplus(speaker_fcs)
            encoder_hidden_states = self.add_sum([encoder_hidden_states, speaker_features], output_name_prefix+"_sum")
        kernel_size = [self.config["variant_predictor_kernel_size"], 1]
        strides = [1, 1]
        self.scopes[scope_id + 1] = "sequential_" + str(predictor_id)
        x = encoder_hidden_states
        for i in range(self.config["variant_prediction_num_conv_layers"]):
            if (i != 0):
                x = self.transpose_nhc_nchw(x)
            padding = self.calculate_convolution_padding(self.get_tensor_shape(x), kernel_size, strides, 'same')
            x = self.extract_convolution(x, output_name_prefix+"_conv_"+str(i), scope_id+2,
                self.config["variant_predictor_filter"], kernel_size, strides, padding,
                data_format="NCHW",
                dilation=1, groups=1,
                layer_names=["conv_._{}".format(i),
                    "conv1d/ExpandDims_1/ReadVariableOp/resource", "BiasAdd/ReadVariableOp/resource"])
            x = self.add_relu(x, output_name_prefix+"_relu_"+str(i))
            x = self.transpose_nchc8_nhc(x)
            x = self.extract_layer_norm(x, output_name_prefix+"_LN_"+str(i), scope_id+2,
                layer_names=["LayerNorm_._{}".format(i), "batchnorm/mul/ReadVariableOp/resource", "batchnorm/ReadVariableOp/resource"])
        outputs = self.extract_dense(x, output_name_prefix+"_dese", scope_id+1,
            ["dense_"+str(predictor_id), "Tensordot/ReadVariableOp/resource", "BiasAdd/ReadVariableOp/resource"])
        if predictor_name == "duration_predictor":
            outputs = self.add_squeeze(outputs, output_name_prefix+"_squeeze", -1)
        return outputs

    def Embeddings(self, input_ids, position_ids, speaker_ids, scope_id, output_name_prefix):
        input_embeddings = self.extract_embedding(input_ids, scope_id, "embeddings/Gather/resource", output_name_prefix+"_word_embed")
        position_embeddings = self.extract_embedding(position_ids, scope_id,
            "embeddings/position_embeddings/GatherNd/resource", output_name_prefix+"_position_embed")
        embeddings = self.add_sum([input_embeddings, position_embeddings], output_name_prefix+"_sum")
        if self.config["n_speakers"] > 1:
            speaker_embeddings = self.extract_embedding(unput_ids, scope_id, "word_embeddings", output_name_prefix+"_speaker_embed")
            speaker_fcs = self.extract_dense(speaker_embeddings, output_name_prefix+"_fc", scope_id, proj_word_embedding_names)
            speaker_features = self.add_softplus(speaker_fcs)
            embeddings = self.add_sum([speaker_features, embeddings], output_name_prefix+"_sum2")
        return embeddings

    def Attention(self, input_name, attention_mask_input_name, output_name_prefix, scope_id,
        num_heads, intermediate_size, intermediate_kernel_size, hidden_act, hidden_size):
        # attention
        self.scopes[scope_id] = "attention"
        # attention-self
        self.scopes[scope_id+1] = "self"
        # attention-self-query
        self.scopes[scope_id+2] = "query"
        query_name = output_name_prefix + "_att_self_query"
        self.extract_dense(input_name, query_name, scope_id+2, ["query", "Tensordot/ReadVariableOp/resource", "BiasAdd/ReadVariableOp/resource"])
        # attention-self-key
        self.scopes[scope_id+2] = "key"
        key_name = output_name_prefix + "_att_self_key"
        self.extract_dense(input_name, key_name, scope_id+2, ["key", "Tensordot/ReadVariableOp/resource", "BiasAdd/ReadVariableOp/resource"])
        # attention-self-value
        self.scopes[scope_id+2] = "value"
        value_name = output_name_prefix + "_att_self_value"
        self.extract_dense(input_name, value_name, scope_id+2, ["value", "Tensordot/ReadVariableOp/resource", "BiasAdd/ReadVariableOp/resource"])

        # reshape
        query_reshape_name = query_name + "_r"
        key_reshape_name   = key_name + "_r"
        value_reshape_name = value_name + "_r"
        size_per_head = self.get_tensor_shape(query_name)[2] // num_heads
        self.add_reshape(query_name, query_reshape_name, [self.batch, -1, num_heads, size_per_head])
        self.add_reshape(key_name,   key_reshape_name,   [self.batch, -1, num_heads, size_per_head])
        self.add_reshape(value_name, value_reshape_name, [self.batch, -1, num_heads, size_per_head])

        # transpose
        query_transpose_name = query_name + "_t"
        key_transpose_name   = key_name + "_t"
        value_transpose_name = value_name + "_t"
        self.add_transpose(query_reshape_name, query_transpose_name, [0, 2, 1, 3])
        self.add_transpose(key_reshape_name,   key_transpose_name,   [0, 2, 3, 1])
        self.add_transpose(value_reshape_name, value_transpose_name, [0, 2, 1, 3])

        # query * key
        query_key_name = output_name_prefix + "_att_self_qk"
        self.add_matmul(query_transpose_name, key_transpose_name, query_key_name)
        query_key_scale_name = output_name_prefix + "_att_self_qks"
        self.add_power(query_key_name, query_key_scale_name, scale=1.0/math.sqrt(size_per_head))

        # query * key + mask
        if (attention_mask_input_name is None):
            sum_name = query_key_scale_name
        else:
            sum_name = output_name_prefix + "_att_self_score"
            self.add_sum([query_key_scale_name, attention_mask_input_name], sum_name)

        # softmax
        prob_name = output_name_prefix + "_att_self_prob"
        self.add_softmax(sum_name, prob_name, 3)

        # prob * value
        context_name = output_name_prefix + "_att_self_cont"
        self.add_matmul(prob_name, value_transpose_name, context_name)

        # transpose value
        context_transpose_name = output_name_prefix + "_att_self_cont_t"
        self.add_transpose(context_name, context_transpose_name, [0, 2, 1, 3])

        # reshape
        context_reshape_name = output_name_prefix + "_att_self_cont_r"
        self.add_reshape(context_transpose_name, context_reshape_name, [self.batch, -1, num_heads*size_per_head])

        # attention-output
        output_name_prefix_new = output_name_prefix + "_att_out_"
        attention_output_name = self.attention_output(context_reshape_name, input_name, output_name_prefix_new, scope_id+1)

        intermediate_output_name = self.intermediate_output(attention_output_name, output_name_prefix+"_iter",
            scope_id, intermediate_size, intermediate_kernel_size, hidden_act, hidden_size)
        sum_name1 = output_name_prefix + "_sum_intermediate"
        self.add_sum([attention_output_name, intermediate_output_name], sum_name1)
        self.scopes[scope_id] = "output"
        output_name = self.extract_layer_norm(sum_name1, output_name_prefix+"_ln",
            scope_id+1, ["LayerNorm/batchnorm", "mul/ReadVariableOp/resource", "ReadVariableOp/resource"])
        return output_name

    def attention_output(self, input_name, element_wise_input_name, output_name_prefix, scope_id):
        self.scopes[scope_id] = "output"

        # output-dense
        dense_name = output_name_prefix + "_den"
        self.extract_dense(input_name, dense_name,
            scope_id+1, ["dense", "Tensordot/ReadVariableOp/resource", "BiasAdd/ReadVariableOp/resource"])
        #output-sum
        sum_name = output_name_prefix + "_sum"
        self.add_sum([dense_name, element_wise_input_name], sum_name)

        # output-layer_norm
        layer_norm_name = output_name_prefix + "_ln"
        self.extract_layer_norm(sum_name, layer_norm_name,
            scope_id+1, ["LayerNorm/batchnorm", "mul/ReadVariableOp/resource", "ReadVariableOp/resource"])
        return layer_norm_name

    def intermediate_output(self, input_name, output_name_prefix, scope_id, intermediate_size, intermediate_kernel_size, hidden_act, hidden_size):
        self.scopes[scope_id] = "intermediate"
        kernel_size = [intermediate_kernel_size, 1]
        strides = [1, 1]
        x = self.transpose_nhc_nchw(input_name)
        padding = self.calculate_convolution_padding(self.get_tensor_shape(x), kernel_size, strides, 'same')
        x = self.extract_convolution(x, output_name_prefix+"_conv_1", scope_id+1,
            intermediate_size, kernel_size, strides, padding,
            data_format="NCHW",
            dilation=1, groups=1, layer_names=["conv1d_1", "conv1d/ExpandDims_1/ReadVariableOp/resource", "BiasAdd/ReadVariableOp/resource"])
        if (hidden_act == "relu"):
            x = self.add_relu(x, output_name_prefix+"_relu")
        elif (hidden_act == "mish"):
            x = self.add_mish(x, output_name_prefix+"_softplus")
        else:
            print("[ERROR] currently not support %s" % (hidden_act))
            exit(1)
        x = self.extract_convolution(x, output_name_prefix+"_conv_2", scope_id+1,
            hidden_size, kernel_size, strides, padding,
            data_format="NCHW",
            dilation=1, groups=1, layer_names=["conv1d_2", "conv1d/ExpandDims_1/ReadVariableOp/resource", "BiasAdd/ReadVariableOp/resource"])
        x = self.transpose_nchc8_nhc(x)
        return x

    def Encoder(self, input_name, scope_id):
        self.scopes[scope_id] = "encoder"
        x = input_name
        for i in range(self.config["encoder_num_hidden_layers"]):
            self.scopes[scope_id+1] = "layer_._" + str(i)
            x = self.Attention(x, None, "encoder_"+str(i), scope_id+2, self.config["encoder_num_attention_heads"],
                self.config["encoder_intermediate_size"], self.config["encoder_intermediate_kernel_size"],
                self.config["encoder_hidden_act"], self.config["encoder_hidden_size"])
        return x

    def Decoder(self, hidden_states, scope_id, position_ids):
        self.scopes[scope_id] = "decoder"
        position_embeddings = self.extract_embedding(position_ids, scope_id+1, "position_embeddings/GatherNd/resource", "decoder_position_embed")
        if self.is_compatible_encoder is False:
            hidden_states = self.project_compatible_decoder(hidden_states)
        hidden_states = self.add_sum([hidden_states, position_embeddings], "decoder_input")
        if self.config["n_speakers"] > 1:
            speaker_embeddings = self.extract_embedding(unput_ids, scope_id, "word_embeddings", "decoder_speaker_embed")
            speaker_fcs = self.extract_dense(speaker_embeddings, output_name_prefix+"_fc", scope_id, "decoder_speaker_dense")
            speaker_features = tf.math.softplus(speaker_fcs)
            hidden_states = self.add_sum([speaker_features, hidden_states], "decoder_input_sum2")
        x = hidden_states
        for i in range(self.config["decoder_num_hidden_layers"]):
            self.scopes[scope_id+1] = "layer_._" + str(i)
            x = self.Attention(x, None, "decoder_"+str(i), scope_id+2, self.config["decoder_num_attention_heads"],
                self.config["decoder_intermediate_size"], self.config["decoder_intermediate_kernel_size"],
                self.config["decoder_hidden_act"], self.config["decoder_hidden_size"])
        return x

    def Postnet(self, inputs, scope_id):
        self.scopes[scope_id] = "postnet"
        x = self.transpose_nhc_nchw(inputs)
        kernel_size = [self.config["postnet_conv_kernel_sizes"], 1]
        strides = [1, 1]
        padding = self.calculate_convolution_padding(self.get_tensor_shape(x), kernel_size, strides, 'same')
        for i in range(self.config["n_conv_postnet"]):
            if i < self.config["n_conv_postnet"] - 1:
                filters = self.config["postnet_conv_filters"]
                act = "tanh"
            else:
                filters = self.config["num_mels"]
                act = None
            x = self.extract_convolution(x, "postnet_conv_"+str(i), scope_id+1,
                filters, kernel_size, strides, padding,
                data_format="NCHW",
                dilation=1, groups=1,
                layer_names=["conv_._{}".format(i), "conv1d/ExpandDims_1/ReadVariableOp/resource", "BiasAdd/ReadVariableOp/resource"])
            x = self.extract_batch_norm(x, "postnet_bn_"+str(i), scope_id+1,
                layer_names=["batch_norm_._{}/batchnorm".format(i), "ReadVariableOp_1/resource", "ReadVariableOp/resource"])
            x = self.extract_scale(x, "postnet_scale_"+str(i), scope_id+1,
                layer_names=["batch_norm_._{}/batchnorm".format(i), "mul/ReadVariableOp/resource", "ReadVariableOp_2/resource"])
            if act == "tanh":
                x = self.add_tanh(x, "postnet_tanh_"+str(i))
            elif act is not None:
                print("[ERROR] currently not support %s" % (act))
                exit(1)
        x = self.transpose_nchc8_nhc(x)
        return x

    def f0_energy_embedding(self, input_name, scope_id, scope_name):
        x = self.transpose_nhc_nchw(input_name)
        kernel_size = [9, 1]
        strides = [1, 1]
        padding = self.calculate_convolution_padding(self.get_tensor_shape(x), kernel_size, strides, 'same')
        x = self.extract_convolution(x, scope_name, scope_id,
            self.config["encoder_hidden_size"], kernel_size, strides, padding,
            data_format="NCHW",
            dilation=1, groups=1,
            layer_names=[scope_name, "conv1d/ExpandDims_1/ReadVariableOp/resource", "BiasAdd/ReadVariableOp/resource"])
        x = self.transpose_nchc8_nhc(x)
        return x

    def generate_encoder(self, inputs=None):
        input_ids = "input_ids"
        input_ids_shape = [self.batch, self.config["max_input_length"]]
        self.add_input(input_ids, input_ids_shape)
        position_ids = "position_ids"
        position_ids_shape = [self.batch, self.config["max_input_length"]]
        self.add_input(position_ids, position_ids_shape)
        speaker_ids = "speaker_ids"
        speaker_ids_shape = [self.batch, 1]
        self.add_input(speaker_ids, speaker_ids_shape)
        speed_ratios = "speed_ratios"
        speed_ratios_shape = [self.batch, 1]
        self.add_input(speed_ratios, speed_ratios_shape)
        f0_ratios = "f0_ratios"
        f0_ratios_shape = [self.batch, 1]
        self.add_input(f0_ratios, f0_ratios_shape)
        energy_ratios = "energy_ratios"
        energy_ratios_shape = [self.batch, 1]
        self.add_input(energy_ratios, energy_ratios_shape)
        self.set_input(inputs)
        embedding_output = self.Embeddings(input_ids, position_ids, speaker_ids, 0, "encoder")
        last_encoder_hidden_states = self.Encoder(embedding_output, 0)
        x_transpose = self.transpose_nhc_nchw(last_encoder_hidden_states)

        # energy predictor, here use last_encoder_hidden_states, u can use more hidden_states layers
        # rather than just use last_hidden_states of encoder for energy_predictor.
        duration_outputs = self.VariantPredictor(x_transpose, speaker_ids, 0, "duration_predictor", 3)
        duration_outputs = self.add_exp(duration_outputs, "duration_exp")
        duration_outputs = self.add_power(duration_outputs, "duration_sub_one", scale=1, shift=-1, power=1)
        duration_outputs = self.add_relu(duration_outputs, "duration_relu")
        duration_outputs = self.add_prod([duration_outputs, speed_ratios], "duration_outputs")

        f0_outputs = self.VariantPredictor(x_transpose, speaker_ids, 0, "f0_predictor", 1)
        f0_outputs = self.add_prod([f0_outputs, f0_ratios], "f0_outputs")

        energy_outputs = self.VariantPredictor(x_transpose, speaker_ids, 0, "energy_predictor", 2)
        energy_outputs = self.add_prod([energy_outputs, energy_ratios], "energy_outputs")

        f0_embedding = self.f0_energy_embedding(f0_outputs, 0, "f0_embeddings")
        energy_embedding = self.f0_energy_embedding(energy_outputs, 0, "energy_embeddings")

        last_encoder_hidden_states = self.add_sum([last_encoder_hidden_states, f0_embedding,  energy_embedding], "features")
        self.save_caffe_model()
        return last_encoder_hidden_states, duration_outputs

    def generate_decoder(self, inputs=None):
        input_ids = "input_ids"
        input_ids_shape = [self.batch, self.config["max_position_embeddings"]]
        self.add_input(input_ids, input_ids_shape)
        position_ids = "position_ids"
        position_ids_shape = [self.batch, self.config["max_position_embeddings"]]
        self.add_input(position_ids, position_ids_shape)
        features = "features"
        features_shape = [self.batch, self.config["max_input_length"], self.config["encoder_hidden_size"]]
        self.add_input(features, features_shape)
        self.set_input(inputs)
        last_encoder_hidden_states = self.add_embedding(input_ids, features, "input_embedding")
        decoder_output = self.Decoder(last_encoder_hidden_states, 0, position_ids)

        # here u can use sum or concat more than 1 hidden states layers from decoder.
        mel_before = self.extract_dense(decoder_output, "mel_before", 0,
            ["mel_before", "Tensordot/ReadVariableOp/resource", "BiasAdd/ReadVariableOp/resource"])
        mel_after = self.Postnet(mel_before, 0)
        output = self.add_sum([mel_before, mel_after], "mel_after")
        self.save_caffe_model()
        return output
