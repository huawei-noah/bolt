#!/usr/local/bin/python
# -*- coding: utf-8 -*-

import math
import numpy as np
import sys
sys.path.append("../")
from tensorflow2caffe import Tensorflow2Caffe

class Tensorflow2CaffeTransformerLstm(Tensorflow2Caffe):
    def __init__(self,
            tensorflow_model_path, caffe_model_path_prefix, caffe_model_name,
            max_seq_length, max_decode_length,
            encoder_params, decoder_params,
            use_small_word_list=False, max_candidate_size=0,
            check=False, calc=False):
        Tensorflow2Caffe.__init__(self, tensorflow_model_path,
            caffe_model_path_prefix, caffe_model_name, check, calc)
        self.scopes[0] = "seq2seq_model"
        self.encoder_params = encoder_params
        self.decoder_params = decoder_params
        self.max_seq_length = max_seq_length
        self.max_decode_length = max_decode_length
        self.use_small_word_list = use_small_word_list
        self.max_candidate_size = max_candidate_size
        if (self.use_small_word_list and self.max_candidate_size == 0):
            self.max_candidate_size = self.max_seq_length * 50 + 2000
        self.encoder_outputs = {}

    @staticmethod
    def default_encoder_params():
        return {
            "num_units": 512,
            "num_layers": 6,
            "layer.preprocess": "n",
            "layer.postprocess": "da",
            "ffn.num_units": 2048,
            "ffn.activation": "relu", # relu or swish
            "attention.num_heads": 8,
            "attention.branch": False, # weighted transformer in https://arxiv.org/pdf/1711.02132.pdf
            "attention.relpos": 0, # relative position representation in https://arxiv.org/pdf/1803.02155.pdf
            "dropout_rate": 0.1,
            "position.enable": True,
            "position.combiner_fn": "tensorflow.add",
            "initializer":  "uniform_unit_scaling",
            "init_scale": 1.0,
            "share_level": 1 # every 2 layers share the same params
        }

    @staticmethod
    def default_decoder_params():
        return {
            "num_layers": 6,
            "num_units": 512,
            "layer.preprocess": "n",
            "layer.postprocess": "da",
            "attention.self_average": False,
            "attention.num_heads": 8,
            "attention.branch": False,
            "attention.relpos": 0,
            "ffn.num_units": 2048,
            "ffn.activation": "relu",
            "dropout_rate": 0.1,
            "position.enable": True,
            "position.combiner_fn": "tensorflow.add",
            "position.max_length": 1000,
            "decode_length_factor": 2.,
            "flex_decode_length": True,
            "initializer": "uniform_unit_scaling",
            "init_scale": 1.0,
            "attention.weighted_avg": False,
            "forget_bias": 1.0,
            "rnn.cell_type": "lstm",
            "sum_att": False
        }

    def ffn(self, x, output_name_prefix, scope_id, activation="relu"):
        self.scopes[scope_id] = "ffn"

        dense_name_1 = output_name_prefix + "_ffn_conv1"
        self.extract_dense(x, dense_name_1, scope_id+1, "conv1")
        
        activation_name = output_name_prefix + "_ffn_act"
        activation_support = False
        if (activation == "relu"):
            activation_support = True
            self.add_relu(dense_name_1, activation_name)
        if (activation == "swish"):
            activation_support = True
            self.add_swish(dense_name_1, activation_name)
        if (not activation_support):
            print("[ERROR] unsupported FFN activation %s" % (activation))
            exit(0)

        dense_name_2 = output_name_prefix + "_ffn_conv2"
        self.extract_dense(activation_name, dense_name_2, scope_id+1, "conv2")
        return dense_name_2

    def additive_attention(self, q,
                           k,
                           v,
                           mask,
                           attention_mask,
                           output_name_prefix,
                           name=None):
        print("[ERROR] unsupported additive attention")
        exit(0)

    def dot_product_attention(self, q,
                              k,
                              v,
                              mask,
                              attention_mask,
                              output_name_prefix,
                              edge_k=None,
                              edge_v=None,
                              name=None):
        if (edge_k is not None):
            sum_name = output_name_prefix + "_dot_ek"
            k = self.add_sum([k, edge_k], sum_name)
        # query * key
        query_key_name = output_name_prefix + "_dot_qk"
        self.add_matmul(q, k, query_key_name)

        if (mask is not None):
            scores = output_name_prefix + "_dot_scores"
            self.add_prod([query_key_name, mask], scores)
            query_key_name = output_name_prefix + "_dot_scores_mask"
            self.add_sum([scores, attention_mask], query_key_name)
        
        # softmax
        scores_normalized = output_name_prefix + "_dot_score_norm"
        self.add_softmax(query_key_name, scores_normalized, 3)
        
        if edge_v is not None:
            sum_name = output_name_prefix + "_dot_ev"
            v = self.add_sum([v, edge_v], sum_name)
        context = output_name_prefix + "_dot_cont"
        self.add_matmul(scores_normalized, v, context)

        return scores_normalized, context

    def multihead_attention(self, query,
                        memory,
                        mask,
                        attention_mask,
                        key_depth,
                        value_depth,
                        output_depth,
                        num_heads,
                        sequence_length,
                        output_name_prefix,
                        scope_id,
                        name=None,
                        cache=None,
                        branch=False,
                        filter_depth=None,
                        activation="relu",
                        relpos=0,
                        sum_att=False,
                        **kwargs):
        self.scopes[scope_id] = "multihead_attention"
        key_depth_per_head = key_depth // num_heads
        value_depth_per_head = value_depth // num_heads
        if memory is None:
            query_name = output_name_prefix + "_multihead_q"
            key_name = output_name_prefix + "_multihead_k"
            value_name = output_name_prefix + "_multihead_v"
            self.extract_denses(query, [query_name, key_name, value_name], [key_depth, key_depth, value_depth], scope_id+1, "qkv")
            key_reshape_name   = key_name + "_r"
            value_reshape_name = value_name + "_r"
            self.add_reshape(key_name,   key_reshape_name,   [self.batch, -1, num_heads, key_depth_per_head])
            self.add_reshape(value_name, value_reshape_name, [self.batch, -1, num_heads, value_depth_per_head])
            key_transpose_name   = key_name + "_t"
            value_transpose_name = value_name + "_t"
            self.add_transpose(key_reshape_name,   key_transpose_name,   [0, 2, 3, 1])
            self.add_transpose(value_reshape_name, value_transpose_name, [0, 2, 1, 3])
        else:
            query_name = output_name_prefix + "_multihead_q"
            self.extract_dense(query, query_name, scope_id+1, "q")
            #key_name = output_name_prefix + "_multihead_k"
            #value_name = output_name_prefix + "_multihead_v"
            #self.extract_denses(memory, [key_name, value_name], [key_depth, value_depth], scope_id+1, "kv")
            #key_name = memory["key"]
            #value_name = memory["value"]
            key_transpose_name = memory["key"]
            value_transpose_name = memory["value"]

        # reshape
        query_reshape_name = query_name + "_r"
        self.add_reshape(query_name, query_reshape_name, [self.batch, -1, num_heads, key_depth_per_head])

        # transpose
        query_transpose_name = query_name + "_t"
        self.add_transpose(query_reshape_name, query_transpose_name, [0, 2, 1, 3])

        edge_k = None
        edge_v = None
        if cache is not None:
            print("[ERROR] cache NOT_SUPPORTED")
            exit(0)

        query_scale_name = output_name_prefix + "_multihead_qs"
        self.add_power(query_transpose_name, query_scale_name, scale=1.0/math.sqrt(key_depth_per_head))

        if relpos > 0: 
            print("[ERROR] relpos>0 NOT_SUPPORTED")
            exit(0)
        if sum_att:
            scores, x = self.additive_attention(
                query_scale_name, key_transpose_name, value_transpose_name, mask, attention_mask)
        else:
            scores, x = self.dot_product_attention(
                query_scale_name, key_transpose_name, value_transpose_name, mask, attention_mask, output_name_prefix,
                edge_k=edge_k, edge_v=edge_v)
        if branch:
            print("[ERROR] branch=True NOT_SUPPORTED")
            exit(0)
        else:
            # transpose
            x_t = output_name_prefix + "_multihead_out_t"
            self.add_transpose(x, x_t, [0, 2, 1, 3])
            # reshape 
            x_r = output_name_prefix + "_multihead_out_r"
            #self.add_reshape(x_t, x_r, [self.batch, sequence_length, num_heads*value_depth_per_head])
            self.add_reshape(x_t, x_r, [self.batch, -1, num_heads*value_depth_per_head])
            # dense
            x = output_name_prefix + "_multihead_out_dense"
            self.extract_dense(x_r, x, scope_id+1, "output_transform")
        return scores, x

    def self_attention_sublayer(self, x, mask, attention_mask, num_units, num_heads, sequence_length, output_name_prefix, scope_id, memory=None, cache=None, branch=False, 
                            filter_depth=None, activation="relu", relpos=0, sum_att=False):
        att_scores, x = self.multihead_attention(
            query=x,
            memory=memory,
            mask=mask,
            attention_mask=attention_mask,
            key_depth=num_units,
            value_depth=num_units,
            output_depth=num_units,
            num_heads=num_heads,
            sequence_length=sequence_length,
            output_name_prefix=output_name_prefix,
            scope_id=scope_id,
            cache=cache,
            branch=branch,
            filter_depth=filter_depth,
            activation=activation,
            relpos=relpos,
            sum_att=sum_att
        )
        return att_scores, x

    def layer_process(self, x, output_name_prefix, scope_id, y=None, mode=None):
        if not mode or mode == "none":
          return x
        
        index = 0
        for m in mode:
            if m == 'a':
                output_name = output_name_prefix + "_a" + str(index) 
                x = self.add_sum([x, y], output_name)
            elif m == 'n':
                output_name = output_name_prefix + "_n" + str(index) 
                x = self.extract_layer_norm(x, output_name, scope_id, ["layer_norm", "gamma", "beta"])
            elif m == 'd':
                print("[INFO] dropout")
            else:
                print("[ERROR] unknown layer process %s" % (m))
            index += 1
        return x

    def position_encoding(self, length, depth, output_name_prefix=None,
                          min_timescale=1,
                          max_timescale=1e4):
        positions = np.arange(length)
        depths = np.arange(depth)
        # correspond to log(10000^(1/(d-1)))
        log_timescale_increment = (
            math.log(max_timescale / min_timescale) / (depth - 1))
        # correspond to 1 / 10000^(i/(d-1)), i=0....d-1
        inv_timescales = min_timescale * np.exp(depths * -1 * log_timescale_increment)
        # pos / 10000^(i/(d-1))
        scaled_time = np.expand_dims(positions, 1) * np.expand_dims(inv_timescales, 0)
        # intead of using SIN and COS interleaved
        # it's  the same to first use SIN then COS
        # as they are applied to the same position
        position_embedding_weight = np.concatenate((np.sin(scaled_time), np.cos(scaled_time)), axis=1)

        output_name = "_position_dict"
        if (output_name_prefix is not None):
            output_name = output_name_prefix + output_name
        self.add_weight(output_name=output_name, weight=position_embedding_weight)
        return output_name

    def encode(self, inputs, sequence_mask, attention_mask, scope_id, output_name_prefix):
        num_units = self.encoder_params["num_units"]

        self.scopes[scope_id] = "target_space_emb"
        adder = output_name_prefix + "_emb"
        weight_name = output_name_prefix + "_target_space_emb"
        self.add_weight(weight_name, scope_id=scope_id+1)
        self.add_sum([inputs, weight_name], adder)

        if (self.encoder_params["position.enable"]):
            position_input_name = "nmt_positions"
            position_input_shape = [self.batch, self.max_seq_length]
            self.add_input(position_input_name, position_input_shape)
            #weight = np.array([[i for i in range(sequence_length)] * self.batch])
            #self.add_weight(position_input_name, weight=weight, data_type="INT32")

            position_embedding_dict_name = self.position_encoding(length=self.max_seq_length,
                                               depth=self.encoder_params["num_units"] // 2,
                                               output_name_prefix=output_name_prefix)
            position_embedding_name = output_name_prefix + "position_embedding"
            self.add_embedding(position_input_name,
                               position_embedding_dict_name,
                               position_embedding_name)

            if (self.encoder_params["position.combiner_fn"] != "tensorflow.add"):
                print("[ERROR] position embedding unsupported")
                exit(0)
            output_name = "we+pe"
            self.add_sum([adder, position_embedding_name], output_name)
            adder = output_name

        x = adder
        for i in range(self.encoder_params["num_layers"]):
            layer_idx = i // self.encoder_params["share_level"]
            self.scopes[scope_id] = "layer_" + str(layer_idx)
            self.scopes[scope_id+1] = "self_attention"
            output_name_prefix_new = output_name_prefix + "_layer" + str(i)

            # preprocess
            x_preprocess = self.layer_process(x,
                               output_name_prefix_new + "_pre1",
                               scope_id+2,
                               mode=self.encoder_params["layer.preprocess"])

            # attention
            _, y = self.self_attention_sublayer(x=x_preprocess,
                                  mask=sequence_mask,
                                  attention_mask=attention_mask,
                                  num_units=self.encoder_params["num_units"],
                                  num_heads=self.encoder_params["attention.num_heads"],
                                  sequence_length=self.max_seq_length,
                                  output_name_prefix=output_name_prefix_new,
                                  scope_id=scope_id+2,
                                  branch=self.encoder_params["attention.branch"],
                                  filter_depth=self.encoder_params["ffn.num_units"],
                                  activation=self.encoder_params["ffn.activation"],
                                  relpos=self.encoder_params["attention.relpos"],
                              )

            # post process
            x = self.layer_process(x,
                    output_name_prefix_new + "_post1",
                    scope_id+2,
                    y=y, mode=self.encoder_params["layer.postprocess"])

            # ffn
            self.scopes[scope_id+1] = "ffn"
            x_preprocess = self.layer_process(x,
                               output_name_prefix_new + "_pre2",
                               scope_id+2,
                               mode=self.encoder_params["layer.preprocess"])
            y = self.ffn(x_preprocess, output_name_prefix_new, scope_id+2, activation=self.encoder_params["ffn.activation"])
            x = self.layer_process(x,
                    output_name_prefix_new + "_post2",
                    scope_id+2,
                    y=y, mode=self.encoder_params["layer.postprocess"])

        outputs = self.layer_process(x,
                      output_name_prefix + "_att_post",
                      scope_id,
                      mode=self.encoder_params["layer.preprocess"])
        self.encoder_outputs["encoder_output"] = outputs
        return outputs

    def encoder_post_process(self, output_name_prefix):
        self.scopes[0] = "seq2seq_model"
        self.scopes[1] = "rnnformer_decoder"
        self.scopes[3] = "encdec_attention"
        self.scopes[4] = "multihead_attention"
        key_depth = self.decoder_params["num_units"]
        value_depth = self.decoder_params["num_units"]
        num_heads = self.decoder_params["attention.num_heads"]
        key_depth_per_head = key_depth // num_heads
        value_depth_per_head = value_depth // num_heads
        encoder_outputs = []
        for layer_idx in range(self.decoder_params["num_layers"]):
            self.scopes[2] = "layer_%d" % (layer_idx)
            key_name = output_name_prefix + "_layer" + str(layer_idx) + "_multihead_k"
            value_name = output_name_prefix + "_layer" + str(layer_idx) + "_multihead_v"
            memory = self.encoder_outputs["encoder_output"]
            self.extract_denses(memory, [key_name, value_name], [key_depth, value_depth], 5, ["kv", "kernel", "bias"])
            key_reshape_name   = key_name + "_r"
            value_reshape_name = value_name + "_r"
            self.add_reshape(key_name,   key_reshape_name,   [self.batch, -1, num_heads, key_depth_per_head])
            self.add_reshape(value_name, value_reshape_name, [self.batch, -1, num_heads, value_depth_per_head])
            key_transpose_name   = key_name + "_t"
            value_transpose_name = value_name + "_t"
            self.add_transpose(key_reshape_name,   key_transpose_name,   [0, 2, 3, 1])
            self.add_transpose(value_reshape_name, value_transpose_name, [0, 2, 1, 3])
            encoder_outputs.append({"key": key_transpose_name, "value": value_transpose_name})
        self.encoder_outputs["encoder_output"] = encoder_outputs

    def attention_ffn_block(self, inputs, encoder_mask, attention_mask, scope_id, output_name_prefix, state=None, position=None):
        x = inputs
        attentions = []
        state_cache = []
        for layer_idx in range(self.decoder_params["num_layers"]):
            self.scopes[scope_id] = "layer_%d" % (layer_idx)
            output_name_prefix_new = output_name_prefix + "_layer_" + str(layer_idx)

            # RNN sublayer            
            self.scopes[scope_id+1] = "rnn_sublayer"
            cur_state = state[layer_idx]
            # Preprocess 
            x_process = output_name_prefix_new + "_pre1"
            x = self.layer_process(x, x_process, scope_id+2, mode=self.decoder_params["layer.preprocess"])

            x = self.add_squeeze(x, axis=1, output_name=x+"_squeeze")

            y = output_name_prefix_new + "_cell"
            self.extract_rnn("LSTM", x, cur_state, y, scope_id+2, scope_name = "basic_lstm_cell")
            state_cache.append(cur_state)
            #Postprocess
            x_process = output_name_prefix_new + "_post1"
            x = self.layer_process(x, x_process, scope_id+2, y=y, mode=self.decoder_params["layer.postprocess"])
            x = self.add_expand_dims(x, axis=1, output_name=x+"_expand")
            
            # Encdec sublayer
            self.scopes[scope_id+1] = "encdec_attention"
            # Preprocess
            x_preprocess = output_name_prefix_new + "_pre2"
            x_preprocess = self.layer_process(x, x_preprocess, scope_id+2, mode=self.decoder_params["layer.preprocess"])
            # Cross attention
            att_scores, y = self.self_attention_sublayer(x=x_preprocess,
                                  mask=encoder_mask,
                                  attention_mask=attention_mask,
                                  num_units=self.decoder_params["num_units"],
                                  num_heads=self.decoder_params["attention.num_heads"],
                                  sequence_length=1,
                                  output_name_prefix=output_name_prefix_new,
                                  scope_id=scope_id+2,
                                  memory=self.encoder_outputs["encoder_output"][layer_idx],
                                  branch=self.decoder_params["attention.branch"],
                                  filter_depth=self.decoder_params["ffn.num_units"],
                                  activation=self.decoder_params["ffn.activation"],
                                  sum_att=self.decoder_params["sum_att"]
                              )
            # Post process
            x_process = output_name_prefix_new + "_post2"
            x = self.layer_process(x, x_process, scope_id+2, y=y, mode=self.decoder_params["layer.postprocess"])
            
            att_context = x

            if not self.decoder_params["attention.weighted_avg"]:
                print("[WARNING] unused attention scores")
                attentions.append(att_scores)
            else:
                print("[ERROR] unsupported attention weighted average")
                
            #if attention is None:
            #    attention = att_scores
            #else:
            #    if not self.decoder_params["attention.weighted_avg"]:
            #        output_name = attention + "add_socres"
            #        #attention = self.add_sum([attention, att_scores], output_name)
            #    else:
            #        print("[ERROR] unsupported attention weighted average")
            #        #attention = np.concatenate([attention, att_scores], axis = 1)

            # FFN sublayer
            self.scopes[scope_id+1] = "ffn"
            # Preprocess
            x_preprocess = output_name_prefix_new + "_pre3"
            x_preprocess = self.layer_process(x, x_preprocess, scope_id+2, mode=self.decoder_params["layer.preprocess"])
            # FFN 
            y = self.ffn(x_preprocess, output_name_prefix_new, scope_id+2, activation=self.decoder_params["ffn.activation"])
            # Postprocess
            x_process = output_name_prefix_new + "_post3"
            x = self.layer_process(x, x_process, scope_id+2, y=y, mode=self.decoder_params["layer.postprocess"])
        state = state_cache

        #Preproecss
        x_process = output_name_prefix_new + "_pre4"
        outputs = self.layer_process(x, x_process, scope_id, mode=self.decoder_params["layer.preprocess"])

        attention = None
        if not self.decoder_params["attention.weighted_avg"]:
            attention = self.add_sum(attentions, "decoder_attention_sum")
            attention = self.add_reduce_mean(attention, axis=1, keep_dim=False, output_name="decoder_attention_mean")
            attention = self.add_power(attention, "decoder_attention_avg", scale=1.0/self.decoder_params["num_layers"])
        else:
            print("[ERROR] unsupported attention weighted average")
        
        return outputs, state, attention, att_context

    def add_projection(self, input_name, weight_name, output_name_prefix):
        matmul_name = output_name_prefix + "_matmul"
        self.add_matmul(input_name, weight_name, matmul_name, transpose_a=False, transpose_b=True)

        argmax_name = self.add_argmax(matmul_name, axis=-1, output_name=output_name_prefix+"_argmax")

        return argmax_name

    def extract_decoder(self, sequence_mask, attention_mask, max_decode_length, scope_id, output_name_prefix):
        # sos=1
        sos = "sos"
        weight = np.array([[1] * self.batch])
        self.add_weight(sos, weight=weight, data_type="INT32")

        negative_one = "negative_one"
        weight = np.array([[-1] * self.batch])
        self.add_weight(negative_one, weight=weight, data_type="INT32")

        decoder_start_name = output_name_prefix + "_words"
        decoder_start_shape = [self.batch, 1]
        self.add_memory(decoder_start_name, decoder_start_shape, data_type="INT32")
        self.add_copy(sos, 1, 1, 0,
                      decoder_start_name, 1, 1, 0,
                      1, output_name="init_decoder")

        position_input_name = output_name_prefix + "_position"
        position_input_shape = [self.batch, 1]
        self.add_memory(position_input_name, position_input_shape, data_type="INT32")
        self.add_copy(negative_one, 1, 1, 0,
                      position_input_name, 1, 1, 0,
                      1, output_name="init_decoder_position")

        zero = "zero"
        weight = np.array([[0]*self.batch])
        self.add_weight(zero, weight=weight, data_type="INT32")

        # eos=2
        eos = "eos"
        weight = np.array([[2]*self.batch])
        self.add_weight(eos, weight=weight, data_type="INT32")

        decoder_output_shape = [self.batch, max_decode_length]
        decoder_output = "decoder_output"
        self.add_memory(decoder_output, decoder_output_shape, data_type="INT32")

        decoder_attention_shape = [self.batch, max_decode_length, self.max_seq_length]
        decoder_attention = "decoder_attention"
        self.add_memory(decoder_attention, decoder_attention_shape, data_type="FLOAT32")

        position_embedding_dict_name = self.position_encoding(length=self.decoder_params["position.max_length"],
                                           depth=self.decoder_params["num_units"] // 2,
                                           output_name_prefix=output_name_prefix)
        state = []
        state_shape = [self.batch, self.encoder_params["num_units"]+self.decoder_params["num_units"]]
        for layer_idx in range(self.decoder_params["num_layers"]):
            state_name = output_name_prefix + "_layer" + str(layer_idx) + "_state"
            self.add_memory(state_name, state_shape, data_type="FLOAT32")
            state.append(state_name)
        sample_ids = decoder_start_name
        for step in range(max_decode_length):
            # whether to add caffe layer
            self.set_add_layer(step==0)

            position_input_name_new = position_input_name+"_add_one"
            self.add_power(position_input_name, position_input_name_new, scale=1, shift=1)
            self.add_copy(position_input_name_new, 1, 1, 0,
                          position_input_name, 1, 1, 0,
                          1, output_name="update_position")
            

            output_name_prefix_new = output_name_prefix + "_step" + str(step)
            cur_inputs = output_name_prefix_new + "_words_embedding"
            self.add_embedding(sample_ids, self.source_modality, cur_inputs)#, transpose=True)
            cur_inputs_scale = output_name_prefix_new + "_words_embedding" + "_s"
            self.add_power(cur_inputs, cur_inputs_scale, scale=(self.decoder_params['num_units'] ** 0.5))

            position_embedding_name = output_name_prefix_new + "_position_embedding"
            self.add_embedding(position_input_name, position_embedding_dict_name, position_embedding_name)

            output_name = output_name_prefix_new + "_embedding"
            cur_inputs_pos = self.add_sum([cur_inputs_scale, position_embedding_name], output_name)

            cell_outputs, state, attention, att_context = self.attention_ffn_block(inputs=cur_inputs_pos, 
                                encoder_mask=sequence_mask,
                                attention_mask=attention_mask,
                                scope_id=scope_id,
                                output_name_prefix = output_name_prefix_new,
                                state=state,
                                position=step)
            self.add_copy(attention,
                          -1, -1, 0,
                          decoder_attention,
                          max_decode_length*self.max_seq_length, self.max_seq_length, 0,
                          -1,
                          output_name="copy_attention_to_global_buffer",
                          src_index_name=zero,
                          dst_index_name=position_input_name)

            current_ids = self.add_projection(cell_outputs, self.source_modality, output_name)
            self.add_copy(current_ids, 1, 1, 0,
                          sample_ids, 1, 1, 0,
                          1, output_name="copy_to_next_input")
            self.add_copy(current_ids,
                          1, 1, 0,
                          decoder_output,
                          max_decode_length, 1, 0,
                          1,
                          output_name="copy_word_to_global_buffer",
                          src_index_name=zero,
                          dst_index_name=position_input_name)
            status = output_name + "_check"
            self.add_check(current_ids, eos, "equal", status)
            self.add_repeat(max_decode_length-1, position_input_name_new, output_name="repeat", status_name=status)
            if (self.get_tensor(status)[0]):
                break;

        return self.get_tensor(decoder_output), self.get_tensor(decoder_attention)
    
    def generate(self, input=None):
        encoder_word_input_name = "nmt_words"
        encoder_word_input_shape = [self.batch, self.max_seq_length]
        self.add_input(encoder_word_input_name, encoder_word_input_shape)
        if (self.use_small_word_list):
            decoder_candidate_input_name = "nmt_candidates"
            decoder_candidate_input_shape = [self.batch, self.max_candidate_size]
            self.add_input(decoder_candidate_input_name, decoder_candidate_input_shape)
        self.set_input(input)

        self.scopes[1] = "source_modality"
        self.scopes[2] = "embedding"
        self.source_modality = "source_modality"
        self.add_weight(output_name=self.source_modality, scope_id=3)#, transpose=[1,0])

        encoder_embedding = "encoder_embedding"
        self.add_embedding(encoder_word_input_name, self.source_modality, encoder_embedding)#, transpose=True)
        encoder_embedding = self.add_power(encoder_embedding, encoder_embedding+"_s", scale=math.sqrt(self.encoder_params["num_units"]))

        if (self.use_small_word_list):
            small_words_embedding = "small_words_embedding"
            self.add_embedding(decoder_candidate_input_name, self.source_modality, small_words_embedding)#, transpose=True)
            self.source_modality = small_words_embedding
        mask_input_name = None
        encoder_attention_mask = None
        decoder_attention_mask = None
        #mask_input_name = "nmt_mask"
        #mask_input_shape = [self.batch, self.max_seq_length]
        #self.add_input(mask_input_name, mask_input_shape)

        self.scopes[1] = "transformer_encoder"
        #encoder_attention_mask = "encoder_attention_mask"
        #self.add_attention(mask_input_name, self.encoder_params['attention.num_heads'], self.max_seq_length, self.max_seq_length, encoder_attention_mask)
        encoders = self.encode(encoder_embedding, mask_input_name, encoder_attention_mask, 2, output_name_prefix="transformer")
        self.encoder_post_process("transformer_decoder")

        self.scopes[1] = "rnnformer_decoder"
        #decoder_attention_mask = "decoder_attention_mask"
        #self.add_attention(mask_input_name, self.decoder_params['attention.num_heads'], 1, self.max_seq_length, decoder_attention_mask)
        decoders = self.extract_decoder(mask_input_name, decoder_attention_mask, self.max_decode_length, 2, output_name_prefix="rnnformer")

        self.save_caffe_model()
