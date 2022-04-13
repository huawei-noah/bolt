#!/usr/local/bin/python
# -*- coding: utf-8 -*-

import math
import numpy as np
from tensorflow2caffe_tactron2 import Tensorflow2CaffeTactron2

class Tensorflow2CaffeTactron2Duration(Tensorflow2CaffeTactron2):
    def __init__(self,
            tensorflow_model_path, caffe_model_path_prefix, caffe_model_name,
            params,
            check=False, calc=False):
        Tensorflow2CaffeTactron2.__init__(self, tensorflow_model_path,
            caffe_model_path_prefix, caffe_model_name, params, check, calc)
        self.params.batch_norm_position = 'after'
        self.params.clip_outputs = True
        self.params.lower_bound_decay = 0.1
        self.params.decoder_layers = 2
        self.params.decoder_lstm_units = 1024
        self.params.outputs_per_step = 3

    def generate_encoder(self, inputs=None):
        hp = self.params
        word_input_name = "words"
        word_input_shape = [self.batch, hp.max_sequence_length]
        self.add_input(word_input_name, word_input_shape)
        mask_input_name = "duration_masks"
        mask_input_shape = [self.batch, hp.max_sequence_length]
        self.add_input(mask_input_name, mask_input_shape)
        speaker_input_name = "speaker"
        speaker_input_shape = [self.batch, 1]
        self.add_input(speaker_input_name, speaker_input_shape)
        self.set_input(inputs)
        self.save_input()

        self.scopes[0] = "Tacotron_model"
        self.scopes[1] = "inference"
        speaker = self.extract_speaker(speaker_input_name, 2)
        duration = self.extract_duration(word_input_name, mask_input_name, 2)
        embedding_inputs = "tts_word_embedding"
        self.extract_embedding(word_input_name, 2, "inputs_embedding", embedding_inputs)
        convolution_result = self.EncoderConvolutions(embedding_inputs, hparams=hp, scope='encoder_convolutions', scope_id=2, output_name_prefix="encoder")
        rnn_result = self.EncoderRNN(convolution_result, size=hp.encoder_lstm_units,
                zoneout=hp.tacotron_zoneout_rate, scope='encoder_LSTM', scope_id=2, output_name_prefix="encoder")
        self.save_caffe_model()
        return self.get_tensor(rnn_result), self.get_tensor(speaker), self.get_tensor(duration)

    def ExpanderConvolutions(self, inputs, scope_id):
        self.scopes[scope_id] = "expander_convolutions"
        kernel_size = [7, 1]
        strides = [1, 1]
        channels = 512
        inputs = self.transpose_nhc_nchw(inputs)
        x = inputs
        for i in range(2):
            self.scopes[scope_id+1] = 'conv_layer_{}_expander_convolutions'.format(i + 1)
            padding = self.calculate_convolution_padding(self.get_tensor_shape(x), kernel_size, strides, 'same')
            x = self.extract_convolution(x, "expander_conv_"+str(i+1), scope_id+2,
                                channels, kernel_size, strides, padding,
                                data_format="NCHW",
                                dilation=1, groups=1, layer_names=['conv1d', "kernel", "bias"])
            x = self.add_relu(x, "expander_relu_"+str(i+1))
            x = self.extract_batch_norm(x, "expander_bn_"+str(i+1), scope_id+2,
                layer_names=["batch_normalization", "moving_mean", "moving_variance"])
        x = self.transpose_nchc8_nhc(x)
        return x

    def generate_decoder(self, inputs=None):
        hp = self.params
        expander_input_name = "encoder"
        expander_input_shape = [self.batch, hp.max_iters, hp.encoder_lstm_units * 2 + 1 + hp.tacotron_speaker_embedding]
        self.add_input(expander_input_name, expander_input_shape)
        length_input_name = "decoder_length"
        length_input_shape = [self.batch, 1]
        self.add_input(length_input_name, length_input_shape)

        decoder_input_name = "decoder_input"
        decoder_input_shape = [self.batch, hp.num_mels]
        decoder_attention_name = "decoder_attention"
        rnn_result_dim = hp.encoder_lstm_units * 2
        decoder_attention_shape = [self.batch, rnn_result_dim]
        self.add_memory(decoder_input_name, decoder_input_shape, data_type="FLOAT32")
        self.add_memory(decoder_attention_name, decoder_attention_shape, data_type="FLOAT32")
        decoder_query_lstm_states = self.prepare_decoder_states(hp.decoder_layers, hp.decoder_lstm_units*2, "decoder_query")
        self.set_input(inputs)
        self.save_input()

        self.scopes[0] = "Tacotron_model"
        self.scopes[1] = "inference"
        context_vectors = self.ExpanderConvolutions(expander_input_name, 2)

        negative_one = "negative_one"
        weight = np.array([[-1] * self.batch])
        self.add_weight(negative_one, weight=weight, data_type="INT32")
        zero = "zero"
        weight = np.array([[0]*self.batch])
        self.add_weight(zero, weight=weight, data_type="INT32")
        position_input_name = "decoder_position"
        position_input_shape = [self.batch, 1]
        self.add_memory(position_input_name, position_input_shape, data_type="INT32")
        self.add_copy(negative_one, 1, 1, 0,
                      position_input_name, 1, 1, 0,
                      1, output_name="init_decoder_position")

        decoder_result_name = "decoder_result"
        decoder_result_shape = [self.batch, hp.outputs_per_step*hp.max_iters, hp.num_mels]
        self.add_memory(decoder_result_name, decoder_result_shape, data_type="FLOAT32")
        x = decoder_input_name
        self.scopes[2] = "decoder"
        index = 0
        for i in range(hp.max_iters):
            self.set_add_layer(i == 0)
            position_input_name_new = position_input_name+"_add_one"
            self.add_power(position_input_name, position_input_name_new, scale=1, shift=1, power=1)
            self.add_copy(position_input_name_new, 1, 1, 0,
                          position_input_name, 1, 1, 0,
                          1, output_name="update_position")

            prenet = self.Prenet(x, layers_sizes=hp.prenet_layers, scope_id=3, scope='decoder_prenet', output_name_prefix="decoder")

            LSTM_input = self.add_concat([prenet, decoder_attention_name], "decoder_concat1", axis=-1)

            LSTM_output = self.DecoderRNN(LSTM_input, decoder_query_lstm_states, layers=hp.decoder_layers,
                        size=hp.decoder_lstm_units, zoneout=hp.tacotron_zoneout_rate,
                        scope_id=3, scope='decoder_LSTM', output_name_prefix="decoder_query_lstm")
            self.add_copy(context_vectors,
                          -1, rnn_result_dim, 0,
                          decoder_attention_name,
                          rnn_result_dim, rnn_result_dim, 0,
                          rnn_result_dim,
                          output_name="copy_context_vector",
                          src_index_name=position_input_name,
                          dst_index_name=zero)
            
            projections_input = self.add_concat([LSTM_output, decoder_attention_name], "decoder_concat2", axis=-1)

            frame_projection = self.FrameProjection(projections_input, hp.num_mels * hp.outputs_per_step,
                        scope_id=3,  scope='linear_transform_projection', output_name_prefix="decoder")

            self.add_copy(frame_projection,
                          hp.outputs_per_step*hp.num_mels, hp.outputs_per_step*hp.num_mels, 0,
                          decoder_result_name,
                          hp.outputs_per_step*hp.max_iters*hp.num_mels, hp.outputs_per_step*hp.num_mels, 0,
                          hp.outputs_per_step*hp.num_mels,
                          output_name="copy_to_global_decoder_buffer",
                          src_index_name=zero,
                          dst_index_name=position_input_name)
            next_input = "decoder_next_input"
            self.add_slice(frame_projection, ["other", next_input], 1, [(hp.outputs_per_step-1)*hp.num_mels])
            self.add_copy(next_input,
                          hp.num_mels, hp.num_mels, 0,
                          x,
                          hp.num_mels, hp.num_mels, 0,
                          hp.num_mels,
                          output_name="copy_to_next_decoder_input")
            status = "decoder_check"
            self.add_check(position_input_name, length_input_name, "equal", status)
            index = index + 1
            self.add_repeat(hp.max_iters-1, position_input_name_new, output_name="repeat", status_name=status)
            if (self.get_tensor(status)[0] or index > hp.max_iters-1):
                break;
        outputs = [decoder_result_name, position_input_name]
        self.add_output(outputs)
        self.save_caffe_model()
        return self.get_tensor(decoder_result_name), self.get_tensor(position_input_name)

    def extract_duration(self, word, duration_mask, scope_id):
        self.scopes[scope_id] = "duration_predictor"

        embedding_inputs = "tts_duration_embedding"
        self.extract_embedding(word, scope_id + 1, "duration_embedding", embedding_inputs)
        
        self.scopes[scope_id + 1] = "encoder_convolutions"
        x = self.transpose_nhc_nchw(embedding_inputs)
        for i in range(2):
            self.scopes[scope_id + 2] = 'conv_layer_{}_encoder_convolutions'.format(i + 1)
            kernel_size = [7, 1]
            strides = [1, 1]
            padding = self.calculate_convolution_padding(self.get_tensor_shape(x), kernel_size, strides, 'same')
            x = self.extract_convolution(x, "duration_conv_"+str(i+1), scope_id+3,
                                256, kernel_size, strides, padding,
                                data_format="NCHW",
                                dilation=1, groups=1, layer_names=['conv1d', "kernel", "bias"])
            x = self.add_relu(x, "duration_relu_"+str(i+1))
            x = self.extract_batch_norm(x, "duration_bn_"+str(i+1), scope_id + 3,
                layer_names=["batch_normalization", "moving_mean", "moving_variance"])
        x = self.transpose_nchc8_nhc(x);
        x = self.extract_dense(x, "duration_dense", scope_id + 1, "projection_duration")
        x = self.add_relu(x, "duration_dense_relu")
        x = self.add_squeeze(x, "duration_squeeze", axis=2)
        x = self.add_prod([x, duration_mask], "duration")
        return x

    def extract_speaker(self, speaker, scope_id):
        self.scopes[scope_id] = "speaker_scope"
        speaker_inputs = "tts_speaker_embedding"
        self.extract_embedding(speaker, scope_id + 1, "speaker_embedding", speaker_inputs)

        h = self.extract_dense(speaker_inputs, "speaker_h", scope_id + 1, "speaker_processor/H")
        h = self.add_relu(h, "speaker_relu")
        t = self.extract_dense(speaker_inputs, "speaker_t", scope_id + 1, "speaker_processor/T")
        t = self.add_sigmoid(t, "speaker_sigmoid")
        ht1 = self.add_prod([h, t], "speaker_ht")
        tt = self.add_power(t, "1_t", scale=-1, shift=1, power=1)
        ht2 = self.add_prod([speaker_inputs, tt], "speaker_xt")
        y = self.add_sum([ht1, ht2], "highwaynet")
        return y

    def generate_postnet(self, inputs=None):
        hp = self.params
        decoder_result_name = "decoder"
        decoder_result_shape = [self.batch, hp.outputs_per_step*hp.max_iters, hp.num_mels]
        self.add_input(decoder_result_name, decoder_result_shape)
        self.set_input(inputs)
        self.save_input()

        self.scopes[0] = "Tacotron_model"
        self.scopes[1] = "inference"

        T2_output_range = (-hp.max_abs_value, hp.max_abs_value) if hp.symmetric_mels else (0, hp.max_abs_value)
        if hp.clip_outputs:
            decoder_result_name = self.add_clip(decoder_result_name, "clip_decoder", T2_output_range[0] - hp.lower_bound_decay, T2_output_range[1])
        #Postnet
        postnet = self.Postnet(decoder_result_name, hparams=hp, scope_id=2, scope='postnet_convolutions', output_name_prefix="postnet")
        projected_residual = self.FrameProjection(postnet, hp.num_mels, scope_id=2, scope='postnet_projection', output_name_prefix="postnet_projection")
        mel_outputs = self.add_sum([decoder_result_name, projected_residual], "mel_sum")
        mel_outputs = self.transpose_nhc_nch(mel_outputs)
        if hp.clip_outputs:
            mel_outputs = self.add_clip(mel_outputs, "clip_residual", T2_output_range[0] - hp.lower_bound_decay, T2_output_range[1])
        self.save_caffe_model()
        return self.get_tensor(mel_outputs)
