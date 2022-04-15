#!/usr/local/bin/python
# -*- coding: utf-8 -*-

import math
import numpy as np
import sys
sys.path.append("../")
from tensorflow2caffe import Tensorflow2Caffe

class Tensorflow2CaffeTactron2(Tensorflow2Caffe):
    def __init__(self,
            tensorflow_model_path, caffe_model_path_prefix, caffe_model_name,
            params,
            check=False, calc=False):
        Tensorflow2Caffe.__init__(self, tensorflow_model_path,
            caffe_model_path_prefix, caffe_model_name, check, calc)
        self.params = params

    class Parameters:
        def __init__(self):
            self.streaming = False

            self.max_sequence_length = 128 # max input sequence
            self.num_mels = 80 #Number of mel-spectrogram channels and local conditioning dimensionality
            self.outputs_per_step = 1 #number of frames to generate at each decoding step (increase to speed up computation and allows for higher batch size, decreases G&L audio quality)
            self.tacotron_zoneout_rate = 0.1 #zoneout rate for all LSTM cells in the network

	        #Mel and Linear spectrograms normalization/scaling and clipping
            self.signal_normalization = True #Whether to normalize mel spectrograms to some predefined range (following below parameters)
            self.allow_clipping_in_normalization = True #Only relevant if mel_normalization = True
            self.symmetric_mels = True #Whether to scale the data to be symmetric around 0. (Also multiplies the output range by 2, faster and cleaner convergence)
            self.max_abs_value = 4. #max absolute value of data. If symmetric, data will be [-max, max] else [0, max] (Must not be too big to avoid gradient explosion, 

            #Limits
            self.min_level_db = -100
            self.ref_level_db = 20

            #Encoder parameters
            self.enc_conv_num_layers = 3 #number of encoder convolutional layers
            self.enc_conv_kernel_size = 5 #size of encoder convolution filters for each layer
            self.enc_conv_channels = 512 #number of encoder convolutions filters for each layer
            self.encoder_lstm_units = 256 #number of lstm units for each direction (forward and backward)

            #Attention mechanism
            self.mask_encoder = True #whether to mask encoder padding while computing attention. Set to True for better prosody but slower convergence.
            self.mask_decoder = False #Whether to use loss mask for padded sequences (if False, <stop_token> loss function will not be weighted, else recommended pos_weight = 20)
            self.smoothing = False #Whether to smooth the attention normalization function
            self.attention_dim = 128 #dimension of attention space
            self.attention_filters = 32 #number of attention convolution filters
            self.attention_kernel = 31 #kernel size of attention convolution
            self.cumulative_weights = True #Whether to cumulate (sum) all previous attention weights or simply feed previous weights (Recommended: True)

            self.decoder_lstm_units = 512 #number of decoder lstm units on each layer

            #Attention synthesis constraints
            #"Monotonic" constraint forces the model to only look at the forwards attention_win_size steps.
            #"Window" allows the model to look at attention_win_size neighbors, both forward and backward steps.
            self.synthesis_constraint = False  #Whether to use attention windows constraints in synthesis only (Useful for long utterances synthesis)
            self.synthesis_constraint_type = 'window' #can be in ('window', 'monotonic').
            self.attention_win_size = 7 #Side of the window. Current step does not count. If mode is window and attention_win_size is not pair, the 1 extra is provided to backward part of the window.

            #Decoder
            self.prenet_layers = [256, 256] #number of layers and number of units of prenet
            self.max_iters = 2000 #Max decoder steps during inference (Just for safety from infinite loop cases)

            #Residual postnet
            self.postnet_num_layers = 5 #number of postnet convolutional layers
            self.postnet_kernel_size = 5 #size of postnet convolution filters for each layer
            self.postnet_channels = 512 #number of postnet convolution filters for each layer

    def EncoderConvolutions(self, inputs, hparams, activation="relu", scope_id=0, scope="enc_conv_layers", output_name_prefix=""):
        self.scopes[scope_id] = scope
        kernel_size = [hparams.enc_conv_kernel_size, 1]
        strides = [1, 1]
        channels = hparams.enc_conv_channels
        activation = activation
        enc_conv_num_layers = hparams.enc_conv_num_layers

        inputs = self.transpose_nhc_nchw(inputs)
        x = inputs
        for i in range(enc_conv_num_layers):
            self.scopes[scope_id+1] = 'conv_layer_{}_'.format(i + 1) + scope
            padding = self.calculate_convolution_padding(self.get_tensor_shape(x), kernel_size, strides, 'same')
            x = self.extract_convolution(x, output_name_prefix+"_conv_"+str(i+1), scope_id+2,
                                channels, kernel_size, strides, padding,
                                data_format="NCHW",
                                dilation=1, groups=1, layer_names=['conv1d', "kernel", "bias"])
            x = self.extract_batch_norm(x, output_name_prefix+"_bn_"+str(i+1), scope_id+2,
                layer_names=["batch_normalization", "moving_mean", "moving_variance"])
            if activation == "relu":
                x = self.add_relu(x, output_name_prefix+"_"+scope+"_"+activation+"_"+str(i+1))
            else:
                print("[ERROR] unsupported activation layer %s in EncoderConvolutions" % (activation))
                exit(1)
        x = self.transpose_nchc8_nhc(x)
        return x

    def EncoderRNN(self, inputs, size=256, zoneout=0.1, scope_id=0, scope="encoder_LSTM", output_name_prefix=""):
        self.scopes[scope_id] = scope
        self.scopes[scope_id+1] = "bidirectional_rnn"
        lstm_output_name = output_name_prefix + "_Bi-LSTM"
        outputs = self.extract_rnn("LSTM", inputs, None, lstm_output_name,
            scope_id+2, steps=-2, scope_name=["fw/encoder_fw_LSTM", "bw/encoder_bw_LSTM"],
            zoneout_cell=zoneout, zoneout_output=zoneout)
        return outputs

    def Prenet(self, inputs, layers_sizes=[256, 256], activation="relu", scope_id=0, scope='prenet', output_name_prefix=""):
        self.scopes[scope_id] = scope
        x = inputs
        for i, size in enumerate(layers_sizes):
            x = self.extract_dense(x, output_name_prefix+"_prenet_dense_"+str(i+1), scope_id+1, 'dense_{}'.format(i + 1))
            if activation == "relu":
                x = self.add_relu(x, output_name_prefix+"_prenet_relu_"+str(i+1))
            else:
                print("[ERROR] unsupported activation layer %s in Prenet" % (activation))
                exit(1)
        return x

    def DecoderRNN(self, inputs, states, layers=2, size=1024, zoneout=0.1, scope_id=0, scope="decoder_rnn", output_name_prefix=""):
        self.scopes[scope_id] = scope
        self.scopes[scope_id+1] = "multi_rnn_cell"
        x = inputs
        for i in range(layers):
            lstm_output_name = output_name_prefix + "_lstm" + str(i)
            x = self.extract_rnn("LSTM", x, states[i], lstm_output_name,
                scope_id=scope_id+2, scope_name='cell_{}/decoder_LSTM_{}'.format(i, i+1),
                zoneout_cell=zoneout,
                zoneout_output=zoneout)
        return x

    def FrameProjection(self, inputs, shape=80, activation=None, scope_id=0, scope="Linear_projection", output_name_prefix=""):
        self.scopes[scope_id] = scope
        x = self.extract_dense(inputs, output_name_prefix+"_"+scope, scope_id+1, 'projection_{}'.format(scope))
        if activation == "relu":
            x = self.add_relu(x, output_name_prefix+"_"+scope+"_"+activation)
        elif activation == None:
            x = x
        else:
            print("[ERROR] unsupported activation layer %s in FrameProjection" % (activation))
            exit(1)
        return x

    def StopProjection(self, inputs, shape=1, activation="sigmoid", scope_id=0, scope="stop_token_projection", output_name_prefix=""):
        self.scopes[scope_id] = scope
        x = self.extract_dense(inputs, output_name_prefix+"_"+scope, scope_id+1, 'projection_{}'.format(scope))
        if activation == "sigmoid":
            x = self.add_sigmoid(x, output_name_prefix+"_"+scope+"_"+activation)
        else:
            print("[ERROR] unsupported activation layer %s in StopProjection" % (activation))
            exit(1)
        return x

    def Postnet(self, inputs, hparams, activation="tanh", scope_id=0, scope="postnet_convolutions", output_name_prefix=""):
        self.scopes[scope_id] = scope
        kernel_size = [hparams.postnet_kernel_size, 1]
        channels = hparams.postnet_channels
        strides = [1, 1]
        activation = activation
        postnet_num_layers = hparams.postnet_num_layers

        inputs = self.transpose_nhc_nchw(inputs)
        x = inputs
        for i in range(postnet_num_layers - 1):
            self.scopes[scope_id+1] = 'conv_layer_{}_'.format(i + 1) + scope
            padding = self.calculate_convolution_padding(self.get_tensor_shape(x), kernel_size, strides, 'same')
            x = self.extract_convolution(x, output_name_prefix+"_conv_"+str(i+1), scope_id+2,
                                channels, kernel_size, strides, padding,
                                data_format="NCHW",
                                dilation=1, groups=1, layer_names=['conv1d', "kernel", "bias"])
            x = self.extract_batch_norm(x, output_name_prefix+"_bn_"+str(i+1), scope_id+2,
                layer_names=["batch_normalization", "moving_mean", "moving_variance"])
            if activation == "tanh":
                x = self.add_tanh(x, output_name_prefix+"_"+scope+"_"+activation+"_"+str(i+1))
            else:
                print("[ERROR] unsupported activation layer %s in EncoderConvolutions" % (activation))
                exit(1)

        layer_id = 5
        self.scopes[scope_id+1] = 'conv_layer_{}_'.format(layer_id) + scope
        padding = self.calculate_convolution_padding(self.get_tensor_shape(x), kernel_size, strides, 'same')
        x = self.extract_convolution(x, output_name_prefix+"_conv_"+str(layer_id), scope_id+2,
                            channels, kernel_size, strides, padding,
                            data_format="NCHW",
                            dilation=1, groups=1, layer_names=['conv1d', "kernel", "bias"])
        x = self.extract_batch_norm(x, output_name_prefix+"_bn_"+str(layer_id), scope_id+2,
            layer_names=["batch_normalization", "moving_mean", "moving_variance"])

        x = self.transpose_nchc8_nhc(x)
        return x

    def _compute_attention(self, cell_output, attention_state,
                           attention_layer, prev_max_attentions, encoder_outputs, hp, scope_id, output_name_prefix):
        alignments, next_attention_state = self.LocationSensitiveAttention(
            cell_output, state=attention_state, prev_max_attentions=prev_max_attentions,
            num_units=hp.attention_dim, memory=encoder_outputs, hparams=hp, scope_id=scope_id,
            mask_encoder=hp.mask_encoder, smoothing=hp.smoothing,
            cumulate_weights=hp.cumulative_weights, output_name_prefix=output_name_prefix)
        expanded_alignments = self.add_expand_dims(alignments, 1, output_name_prefix+"_alignment_expand")
        context = self.add_matmul(expanded_alignments, encoder_outputs, output_name_prefix+"_context")
        context = self.add_squeeze(context, output_name_prefix+"_context_squeeze", 1)
        if attention_layer is not None:
            print("[ERROR] unsupported attention layer")
            exit(1)
        else:
            attention = context
        return attention, alignments, next_attention_state

    def LocationSensitiveAttention(self,
                 query, state, prev_max_attentions,
                 num_units,
                 memory,
                 hparams,
                 scope_id=0,
                 mask_encoder=True,
                 memory_sequence_length=None,
                 smoothing=False,
                 cumulate_weights=True,
                 name='LocationSensitiveAttention',
                 output_name_prefix=""):
        _cumulate = cumulate_weights
        synthesis_constraint = hparams.synthesis_constraint
        attention_win_size = hparams.attention_win_size
        constraint_type = hparams.synthesis_constraint_type
        previous_alignments = state

        keys = self.extract_dense(memory, output_name_prefix+"_keys", scope_id-1, "memory_layer")

        self.scopes[scope_id-1] = "decoder"
        self.scopes[scope_id] = "Location_Sensitive_Attention"

        processed_query = self.extract_dense(query, output_name_prefix+"_query", scope_id+1, "query_layer")
        processed_query = self.add_expand_dims(processed_query, 1, output_name_prefix+"_query_expand")

        expanded_alignments = self.add_expand_dims(previous_alignments, 2, output_name_prefix+"_align_expand")
        expanded_alignments = self.transpose_nhc_nchw(expanded_alignments)
        kernel_size = [hparams.attention_kernel, 1]
        strides = [1, 1]
        padding = self.calculate_convolution_padding(self.get_tensor_shape(expanded_alignments), kernel_size, strides, 'same')
        f = self.extract_convolution(expanded_alignments, output_name_prefix+"_conv", scope_id+1,
                           hparams.attention_filters, kernel_size, strides, padding,
                           data_format="NCHW",
                           dilation=1, groups=1, layer_names=['location_features_convolution', "kernel", "bias"])
        f = self.transpose_nchc8_nhc(f)
        processed_location_features = self.extract_dense(f, output_name_prefix+"_location", scope_id+1, ["", "location_features_layer/kernel", "attention_bias"])

        #energy = self._location_sensitive_score(processed_query, processed_location_features, self.keys)
        sum_result = self.add_sum([processed_query, processed_location_features, keys], output_name_prefix+"_sum1")
        tanh_result = self.add_tanh(sum_result, output_name_prefix+"_tanh")
        fc_result = self.extract_scale(tanh_result, output_name_prefix+"_scale", scope_id+1, axis=-1, layer_names=["", "attention_variable_projection", "bias"])
        energy = self.add_reduce_sum(fc_result, 2, False, output_name="decoder_reduce_sum")

        if synthesis_constraint:
            print("[ERROR] not support synthesis_constraint")
            exit(1)
        if (smoothing):
            print("[ERROR] unsupported smoothing softmax")
            exit(1)
        else:
            alignments = self.add_softmax(energy, output_name_prefix+"_softmax", -1)

        if _cumulate:
            next_state = self.add_sum([alignments, previous_alignments], output_name_prefix+"_sum2")
        else:
            next_state = alignments

        return alignments, next_state

    def prepare_decoder_states(self, layers, num_units, output_name_prefix):
        states = []
        for i in range(layers):
            state_shape = [self.batch, num_units]
            state_name = output_name_prefix + "_layer" + str(i) + "_state"
            if (self.params.streaming):
                state_name = self.add_input(state_name, state_shape)
            else:
                state_name = self.add_memory(state_name, state_shape, data_type="FLOAT32")
            states.append(state_name)
        return states

    def generate_encoder(self, inputs=None):
        hp = self.params
        word_input_name = "tts_words"
        word_input_shape = [self.batch, hp.max_sequence_length]
        self.add_input(word_input_name, word_input_shape)
        self.set_input(inputs)
        self.save_input()

        self.scopes[0] = "Tacotron_model"
        self.scopes[1] = "inference"
        embedding_inputs = "tts_word_embedding"
        self.extract_embedding(word_input_name, 2, "inputs_embedding", embedding_inputs)
        convolution_result = self.EncoderConvolutions(embedding_inputs, hparams=hp, scope='encoder_convolutions', scope_id=2, output_name_prefix="encoder")
        rnn_result = self.EncoderRNN(convolution_result, size=hp.encoder_lstm_units,
                zoneout=hp.tacotron_zoneout_rate, scope='encoder_LSTM', scope_id=2, output_name_prefix="encoder")
        if (self.params.streaming):
            self.save_caffe_model()
        return rnn_result

    def generate_decoder(self, rnn_result, inputs=None):
        hp = self.params
        cumulated_alignments = "tts_alignments"
        cumulated_alignments_shape = [self.batch, hp.max_sequence_length]
        self.add_input(cumulated_alignments, cumulated_alignments_shape)
        decoder_input_name = "decoder_input"
        decoder_input_shape = [self.batch, hp.num_mels]
        decoder_attention_name = "decoder_attention"
        rnn_result_dim = hp.encoder_lstm_units * 2
        decoder_attention_shape = [self.batch, rnn_result_dim]
        if (self.params.streaming):
            self.add_input(decoder_input_name, decoder_input_shape)
            self.add_input(decoder_attention_name, decoder_attention_shape)
            rnn_result_shape = [self.batch, hp.max_sequence_length, rnn_result_dim]
            self.add_input(rnn_result, rnn_result_shape)
            alignments_history = "tts_alignments_history"
            alignments_history_shape = [self.batch, hp.max_iters, hp.max_sequence_length]
            self.add_memory(alignments_history, alignments_history_shape, data_type="FLOAT32")
        else:
            self.add_memory(decoder_input_name, decoder_input_shape, data_type="FLOAT32")
            self.add_memory(decoder_attention_name, decoder_attention_shape, data_type="FLOAT32")
        decoder_query_lstm_states = self.prepare_decoder_states(1, hp.decoder_lstm_units*2, "decoder_query")
        self.set_input(inputs)
        self.save_input()
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
        self.scopes[0] = "Tacotron_model"
        self.scopes[1] = "inference"
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

            LSTM_output = self.DecoderRNN(LSTM_input, decoder_query_lstm_states, layers=1,
                        size=hp.decoder_lstm_units, zoneout=hp.tacotron_zoneout_rate,
                        scope_id=3, scope='decoder_query_LSTM', output_name_prefix="decoder_query_lstm")
            context_vector, alignments, new_cumulated_alignments = self._compute_attention(
                        LSTM_output,
                        cumulated_alignments,
                        attention_layer=None,
                        prev_max_attentions=None,
                        encoder_outputs=rnn_result,
                        hp=hp, scope_id=3, output_name_prefix="decoder_attention")
            self.add_copy(context_vector,
                          hp.decoder_lstm_units, hp.decoder_lstm_units, 0,
                          decoder_attention_name,
                          hp.decoder_lstm_units, hp.decoder_lstm_units, 0,
                          hp.decoder_lstm_units,
                          output_name="copy_decoder_attention")
            self.add_copy(new_cumulated_alignments,
                          -1, -1, 0,
                          cumulated_alignments,
                          -1, -1, 0,
                          -1,
                          output_name="copy_cumulated_alignments")
            if (self.params.streaming):
                self.add_copy(alignments,
                              -1, -1, 0,
                              alignments_history,
                              hp.max_sequence_length, hp.max_sequence_length, 0,
                              -1,
                              output_name="copy_to_alignments_history",
                              src_index_name=zero,
                              dst_index_name=position_input_name)
            projections_input = self.add_concat([LSTM_output, context_vector], "decoder_concat2", axis=-1)

            frame_projection = self.FrameProjection(multi_rnn_output, hp.num_mels * hp.outputs_per_step,
                        scope_id=3,  scope='linear_transform_projection', output_name_prefix="decoder")

            stop_projection = self.StopProjection(projections_input, shape=hp.outputs_per_step,
                        scope_id=3,  scope='stop_token_projection', output_name_prefix="decoder")
            stop_projection = self.add_power(stop_projection, "decoder_stop_sub", scale=1, shift=-0.5, power=1)
            stop_projection = self.add_relu(stop_projection, "decoder_stop_relu")
            stop_projection = self.add_reduce_sum(stop_projection, 1, False, "decoder_stop_sum")
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
            self.add_check(stop_projection, zero, "great", status)
            index = index + 1
            self.add_repeat(hp.max_iters-1, position_input_name_new, output_name="repeat", status_name=status)
            if (self.get_tensor(status)[0] or index > hp.max_iters-1):
                break;
        if (self.params.streaming):
            mels = decoder_result_name
            if (hp.signal_normalization):
                mels = self.convert_db_melgan_log(self._denormalize(mels, hp), hp)
            outputs = [mels, position_input_name, stop_projection, decoder_attention_name,
                cumulated_alignments, alignments_history]
            outputs.extend(decoder_query_lstm_states)
            outputs.extend(decoder_lstm_states)
        else:
            outputs = [decoder_result_name, position_input_name]
        self.add_output(outputs)
        self.save_caffe_model()
        return self.get_tensor(decoder_result_name), self.get_tensor(position_input_name)

    def generate_encoder_decoder(self, inputs=None):
        rnn_result = self.generate_encoder(inputs)
        return self.generate_decoder(rnn_result, inputs)

    def convert_db_melgan_log(self, mel, hparams):
        #return (mel + hparams.ref_level_db) / 20
        return self.add_power(mel, "mel", scale=1/20.0, shift=hparams.ref_level_db/20.0, power=1)

    def _denormalize(self, D, hparams):
        if hparams.allow_clipping_in_normalization:
            if hparams.symmetric_mels:
                #return (((np.clip(D, -hparams.max_abs_value,
                #                  hparams.max_abs_value) + hparams.max_abs_value) * -hparams.min_level_db / (
                #                 2 * hparams.max_abs_value))
                #        + hparams.min_level_db)
                clip_result = self.add_clip(D, "mel_clip", -hparams.max_abs_value, hparams.max_abs_value)
                b = -hparams.min_level_db / (2 * hparams.max_abs_value)
                return self.add_power(clip_result, "mel_denormalize", scale=b, shift=hparams.max_abs_value*b+hparams.min_level_db, power=1)
            else:
                #return ((np.clip(D, 0,
                #                 hparams.max_abs_value) * -hparams.min_level_db / hparams.max_abs_value) + hparams.min_level_db)
                clip_result = self.add_clip(D, "mel_clip", 0, hparams.max_abs_value)
                return self.add_power(clip_result, "mel_denormalize", scale=-hparams.min_level_db / hparams.max_abs_value,
                    shift=hparams.min_level_db, power=1)

    
        if hparams.symmetric_mels:
            #return (((D + hparams.max_abs_value) * -hparams.min_level_db / (
            #        2 * hparams.max_abs_value)) + hparams.min_level_db)
            b = -hparams.min_level_db / (2 * hparams.max_abs_value)
            return self.add_power(D, "mel_denormalize", scale=b, shift=hparams.max_abs_value*b+hparams.min_level_db, power=1)
        else:
            #return ((D * -hparams.min_level_db / hparams.max_abs_value) + hparams.min_level_db)
            return self.add_power(D, "mel_denormalize", scale=-hparams.min_level_db / hparams.max_abs_value, shift=hparams.min_level_db, power=1)

    def generate_postnet(self, inputs=None):
        hp = self.params
        decoder_result_name = "tts_decoder"
        decoder_result_shape = [self.batch, hp.outputs_per_step*hp.max_iters, hp.num_mels]
        self.add_input(decoder_result_name, decoder_result_shape)
        self.set_input(inputs)
        self.save_input()

        self.scopes[0] = "Tacotron_model"
        self.scopes[1] = "inference"

        #Postnet
        postnet = self.Postnet(decoder_result_name, hparams=hp, scope_id=2, scope='postnet_convolutions', output_name_prefix="postnet")
        projected_residual = self.FrameProjection(postnet, hp.num_mels, scope_id=2, scope='postnet_projection', output_name_prefix="postnet_projection")
        mel_outputs = self.add_sum([decoder_result_name, projected_residual], "mel_sum")
        mel_outputs = self.transpose_nhc_nchw(mel_outputs)
        if (hp.signal_normalization):
            mels = self.convert_db_melgan_log(self._denormalize(mel_outputs, hp), hp)

        self.save_caffe_model()
