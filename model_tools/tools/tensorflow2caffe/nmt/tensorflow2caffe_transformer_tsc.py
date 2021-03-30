#!/usr/local/bin/python
# -*- coding: utf-8 -*-

import math
import numpy as np
import sys
sys.path.append("../")
from tensorflow2caffe import Tensorflow2Caffe

class Tensorflow2CaffeTransformerTSC(Tensorflow2Caffe):
    def __init__(self,
            tensorflow_model_path, caffe_model_path_prefix, caffe_model_name,
            max_seq_length, max_decode_length,
            encoder_params, decoder_params,
            check=False, calc=False):
        Tensorflow2Caffe.__init__(self, tensorflow_model_path,
            caffe_model_path_prefix, caffe_model_name, check, calc)
        self.scopes[0] = "seq2seq_model"
        self.encoder_params = encoder_params
        self.decoder_params = decoder_params
        self.max_seq_length = max_seq_length
        self.max_decode_length = max_decode_length
        self.encoder_outputs = {}

    @staticmethod
    def default_encoder_params():
        return {
            "num_layers": 6,
            "num_units": 512,
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
            "share_level": 1
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
            "sum_att": False,
            "share_level": 1
        }

    def ffn(self, x, output_name_prefix, scope_id, activation="relu", share_index=0, share_num=1):
        self.scopes[scope_id] = "ffn_layer"

        dense_name_1 = output_name_prefix + "_ffn_conv1"
        self.extract_dense(x, dense_name_1, scope_id+1, ["input_layer/linear", "matrix", "bias"],
            share_index=share_index, share_num=share_num)
        
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
        self.extract_dense(activation_name, dense_name_2, scope_id+1, ["output_layer/linear", "matrix", "bias"],
            share_index=share_index, share_num=share_num)
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
                        share_index=0,
                        share_num=1,
                        **kwargs):
        self.scopes[scope_id] = "multihead_attention"
        if memory is None:
            query_name = output_name_prefix + "_multihead_q"
            key_name = output_name_prefix + "_multihead_k"
            value_name = output_name_prefix + "_multihead_v"
            self.extract_denses(query, [query_name, key_name, value_name],
                [key_depth, key_depth, value_depth], scope_id+1, ["qkv_transform", "matrix", "bias"],
                share_index=share_index, share_num=share_num)

            # 结合历史序列的key value，生成当前key value，在axis=1处进行concat
            if cache is not None:
                key_name = self.add_concat([cache["self_key"], key_name], key_name + "_cache", axis=1)
                value_name = self.add_concat([cache["self_value"], value_name], value_name + "_cache", axis=1)
                # 更新缓存
                cache["self_key"] = key_name
                cache["self_value"] = value_name

        else:
            query_name = output_name_prefix + "_multihead_q"
            self.extract_dense(query, query_name, scope_id+1, ["q_transform", "matrix", "bias"],
                share_index=share_index, share_num=share_num)
            #key_name = output_name_prefix + "_multihead_k"
            #value_name = output_name_prefix + "_multihead_v"
            #self.extract_denses(memory, [key_name, value_name], [key_depth, value_depth], scope_id+1, ["kv_transform", "matrix", "bias"])
            key_name = memory["key"]
            value_name = memory["value"]

        # reshape
        query_reshape_name = query_name + "_r"
        key_reshape_name   = key_name + "_r"
        value_reshape_name = value_name + "_r"
        key_depth_per_head = key_depth // num_heads
        value_depth_per_head = value_depth // num_heads
        #self.add_reshape(query_name, query_reshape_name, [self.batch, sequence_length, num_heads, key_depth_per_head])
        #self.add_reshape(key_name,   key_reshape_name,   [self.batch, self.max_seq_length, num_heads, key_depth_per_head])
        #self.add_reshape(value_name, value_reshape_name, [self.batch, self.max_seq_length, num_heads, value_depth_per_head])
        self.add_reshape(query_name, query_reshape_name, [self.batch, -1, num_heads, key_depth_per_head])
        self.add_reshape(key_name,   key_reshape_name,   [self.batch, -1, num_heads, key_depth_per_head])
        self.add_reshape(value_name, value_reshape_name, [self.batch, -1, num_heads, value_depth_per_head])

        # transpose
        query_transpose_name = query_name + "_t"
        key_transpose_name   = key_name + "_t"
        value_transpose_name = value_name + "_t"
        self.add_transpose(query_reshape_name, query_transpose_name, [0, 2, 1, 3])
        self.add_transpose(key_reshape_name,   key_transpose_name,   [0, 2, 3, 1])
        self.add_transpose(value_reshape_name, value_transpose_name, [0, 2, 1, 3])

        edge_k = None
        edge_v = None

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
            self.extract_dense(x_r, x, scope_id+1, ["output_transform", "matrix", "bias"],
                share_index=share_index, share_num=share_num)
        return scores, x

    def self_attention_sublayer(self, x, mask, attention_mask, num_units, num_heads, sequence_length,
                                output_name_prefix, scope_id, memory=None, cache=None, branch=False,
                                filter_depth=None, activation="relu", relpos=0, sum_att=False,
                                share_index=0, share_num=1):
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
            sum_att=sum_att,
            share_index=share_index,
            share_num=share_num
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
                x = self.extract_layer_norm(x, output_name, scope_id, ["layer_norm", "scale", "offset"])
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

        if (self.encoder_params["position.enable"]):
            position_input_name = "encoder_positions"
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
            self.add_sum([inputs, position_embedding_name], output_name)
            adder = output_name

        x = adder
        for i in range(self.encoder_params["num_layers"]//self.encoder_params["share_level"]):
            x_input = x
            for j in range(self.encoder_params["share_level"]):
                self.set_add_layer(j==0)

                layer_idx = i
                self.scopes[scope_id] = "layer_" + str(layer_idx)
                self.scopes[scope_id+1] = "self_attention"
                output_name_prefix_new = output_name_prefix + "_layer" + str(i)

                # preprocess
                x_preprocess = self.layer_process(x,
                                   output_name_prefix_new + "_pre1",
                                   scope_id+2,
                                   mode=self.encoder_params["layer.preprocess"])
                loop_start = x_preprocess

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
                self.scopes[scope_id+1] = "feed_forward"
                x_preprocess = self.layer_process(x,
                                   output_name_prefix_new + "_pre2",
                                   scope_id+2,
                                   mode=self.encoder_params["layer.preprocess"])
                y = self.ffn(x_preprocess, output_name_prefix_new, scope_id+2, activation=self.encoder_params["ffn.activation"])
                x = self.layer_process(x,
                        output_name_prefix_new + "_post2",
                        scope_id+2,
                        y=y, mode=self.encoder_params["layer.postprocess"])
                if (self.encoder_params["share_level"] > 1):
                    self.add_copy(x,
                                  -1, -1, 0,
                                  x_input,
                                  -1, -1, 0,
                                  -1,
                                  output_name="encoder_copy"+str(i))
                    self.add_repeat(self.encoder_params["share_level"]-1, loop_start, output_name="encoder_repeat"+str(i))

        self.set_add_layer(True)
        outputs = self.layer_process(x,
                      output_name_prefix + "_att_post",
                      scope_id,
                      mode=self.encoder_params["layer.preprocess"])
        self.encoder_outputs["encoder_output"] = outputs
        return outputs

    def attention_ffn_block(self, inputs, encoder_mask, attention_mask, scope_id, output_name_prefix, state=None, position=None):
        x = inputs
        attention = None
        state_cache = []
        for layer_idx in range(self.decoder_params["num_layers"]):
            self.scopes[scope_id] = "layer_%d" % (layer_idx // self.decoder_params["share_level"])
            output_name_prefix_new = output_name_prefix + "_layer_" + str(layer_idx)

            # RNN sublayer            
            self.scopes[scope_id+1] = "self_attention"
            cur_state = state[layer_idx]
            # Preprocess 
            x_process = output_name_prefix_new + "_pre1"
            x_process = self.layer_process(x, x_process, scope_id+2, mode=self.decoder_params["layer.preprocess"])

            #x = self.add_squeeze(x, axis=1, output_name=x+"_squeeze")

            y = output_name_prefix_new + "_self_attention"
            #self.extract_rnn("LSTM", x, cur_state, y, scope_id+2, scope_name = "basic_lstm_cell")
            # attention
            _, y = self.self_attention_sublayer(x_process,
                                                None, None,
                                                num_units=self.encoder_params["num_units"],
                                                num_heads=self.encoder_params["attention.num_heads"],
                                                sequence_length=self.max_seq_length,
                                                output_name_prefix=y,
                                                scope_id=scope_id + 2,
                                                branch=self.encoder_params["attention.branch"],
                                                filter_depth=self.encoder_params["ffn.num_units"],
                                                activation=self.encoder_params["ffn.activation"],
                                                relpos=self.encoder_params["attention.relpos"],
                                                cache=cur_state,
                                                share_index=layer_idx % self.decoder_params["share_level"],
                                                share_num=self.decoder_params["share_level"]
                                                )

            state_cache.append(cur_state)
            #Postprocess
            x_process = output_name_prefix_new + "_post1"
            x = self.layer_process(x, x_process, scope_id+2, y=y, mode=self.decoder_params["layer.postprocess"])
            #x = self.add_expand_dims(x, axis=1, output_name=x+"_expand")
            
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
                                  sum_att=self.decoder_params["sum_att"],
                                  share_index=layer_idx % self.decoder_params["share_level"],
                                  share_num=self.decoder_params["share_level"]
                              )
            # Post process
            x_process = output_name_prefix_new + "_post2"
            x = self.layer_process(x, x_process, scope_id+2, y=y, mode=self.decoder_params["layer.postprocess"])
            
            att_context = x

            if not self.decoder_params["attention.weighted_avg"]:
                print("[WARNING] unused attention scores")
                #att_scores = self.add_axis_mean(att_scores, axis=1, output_name=att_scores+"_mean")
            else:
                print("[ERROR] unsupported attention weighted average")
                
            if attention is None:
                attention = att_scores
            else:
                if not self.decoder_params["attention.weighted_avg"]:
                    output_name = attention + "add_socres"
                    #attention = self.add_sum([attention, att_scores], output_name)
                else:
                    print("[ERROR] unsupported attention weighted average")
                    #attention = np.concatenate([attention, att_scores], axis = 1)

            # FFN sublayer
            self.scopes[scope_id+1] = "feed_forward"
            # Preprocess
            x_preprocess = output_name_prefix_new + "_pre3"
            x_preprocess = self.layer_process(x, x_preprocess, scope_id+2, mode=self.decoder_params["layer.preprocess"])
            # FFN 
            y = self.ffn(x_preprocess, output_name_prefix_new, scope_id+2, activation=self.decoder_params["ffn.activation"],
                         share_index=layer_idx % self.decoder_params["share_level"],
                         share_num=self.decoder_params["share_level"])
            # Postprocess
            x_process = output_name_prefix_new + "_post3"
            x = self.layer_process(x, x_process, scope_id+2, y=y, mode=self.decoder_params["layer.postprocess"])
        state = state_cache

        #Preproecss
        x_process = output_name_prefix_new + "_pre4"
        outputs = self.layer_process(x, x_process, scope_id, mode=self.decoder_params["layer.preprocess"])

        if not self.decoder_params["attention.weighted_avg"]:
            output_name = attention + "_div"
            #attention = self.add_power(attention, output_name, scale=1.0/self.decoder_params["num_layers"])
        else:
            print("[ERROR] unsupported attention weighted average")
        
        return outputs, state, att_context

    def add_projection(self, input_name, weight_name, output_name_prefix):
        matmul_name = output_name_prefix + "_matmul"
        self.add_matmul(input_name, weight_name, matmul_name, transpose_a=False, transpose_b=True)

        argmax_name = self.add_argmax(matmul_name, axis=-1, output_name=output_name_prefix+"_argmax")

        return argmax_name

    def encoder_post_process(self, output_name_prefix):
        self.scopes[0] = "transformer"
        self.scopes[1] = "decoder"
        self.scopes[3] = "encdec_attention"
        self.scopes[4] = "multihead_attention"
        key_depth = self.decoder_params["num_units"]
        value_depth = self.decoder_params["num_units"]
        encoder_outputs = []
        for layer_idx in range(self.decoder_params["num_layers"]):
            self.scopes[2] = "layer_%d" % (layer_idx // self.decoder_params["share_level"])
            key_name = output_name_prefix + "_layer" + str(layer_idx) + "_multihead_k"
            value_name = output_name_prefix + "_layer" + str(layer_idx) + "_multihead_v"
            memory = self.encoder_outputs["encoder_output"]
            self.extract_denses(memory, [key_name, value_name], [key_depth, value_depth], 5, ["kv_transform", "matrix", "bias"],
                share_index=layer_idx % self.decoder_params["share_level"],
                share_num=self.decoder_params["share_level"])
            encoder_outputs.append({"key": key_name, "value": value_name})
        self.encoder_outputs["encoder_output"] = encoder_outputs

    def save_encoder_states(self):
        output_name_prefix = "decoder"
        for layer_idx in range(self.decoder_params["num_layers"]):
            key_name = output_name_prefix + "_layer" + str(layer_idx) + "_multihead_k"
            value_name = output_name_prefix + "_layer" + str(layer_idx) + "_multihead_v"
            key_data = self.get_tensor(self.encoder_outputs["encoder_output"][layer_idx]["key"])
            value_data = self.get_tensor(self.encoder_outputs["encoder_output"][layer_idx]["value"])
            np.savetxt(key_name+".txt", key_data.reshape([key_data.size]))
            np.savetxt(value_name+".txt", value_data.reshape([value_data.size]))

    def prepare_decoder_states(self, decoder_position_input):
        output_name_prefix = "decoder"
        self.encoder_outputs["encoder_output"] = []
        states = []
        position = self.get_tensor(decoder_position_input)[0][0]
        for layer_idx in range(self.decoder_params["num_layers"]):
            prefix = "decoder_layer" + str(layer_idx)
            key0_name = prefix + "_kmem"
            value0_name = prefix + "_vmem"
            if (position == 0):
                state0_shape = [self.batch, 0, 0]
                key0_data = np.zeros(state0_shape)
                value0_data = np.zeros(state0_shape)
                state0_shape = [self.batch, self.max_decode_length, self.decoder_params["num_units"]]
            else:
                key0_data = np.load(key0_name + ".npy")
                value0_data = np.load(value0_name + ".npy")
                state0_shape = key0_data.shape
            self.add_input(key0_name, state0_shape)
            self.add_input(value0_name, state0_shape)
            states.append({"self_key": key0_name, "self_value": value0_name})

            key1_name = output_name_prefix + "_layer" + str(layer_idx) + "_multihead_k"
            value1_name = output_name_prefix + "_layer" + str(layer_idx) + "_multihead_v"
            key1_data = np.loadtxt(key1_name + ".txt")
            value1_data = np.loadtxt(value1_name + ".txt")
            key1_data = key1_data.reshape([self.batch, -1, self.decoder_params["num_units"]])
            value1_data = value1_data.reshape([self.batch, -1, self.decoder_params["num_units"]])
            self.add_input(key1_name, key1_data.shape)
            self.add_input(value1_name, value1_data.shape)
            self.encoder_outputs["encoder_output"].append({"key": key1_name, "value": value1_name})
            data = {key0_name: key0_data,
                    value0_name: value0_data,
                    key1_name: key1_data,
                    value1_name: value1_data}
            self.set_input(data)
        return states

    def save_decoder_states(self, decoder_states, decoder_position_input):
        position = self.get_tensor(decoder_position_input)[0][0]
        states = []
        for layer_idx in range(self.decoder_params["num_layers"]):
            prefix = "decoder_layer" + str(layer_idx)
            key_name = prefix + "_kmem"
            value_name = prefix + "_vmem"
            np.save(key_name +".npy", self.get_tensor(decoder_states[layer_idx]["self_key"]))
            np.save(value_name +".npy", self.get_tensor(decoder_states[layer_idx]["self_value"]))
            states.append(decoder_states[layer_idx]["self_key"])
            states.append(decoder_states[layer_idx]["self_value"])
        return states

    def generate_decoder(self, input=None):
        position = input["decoder_positions"][0][0]
        decoder_word_input_name = "decoder_words"
        decoder_word_input_shape = [self.batch, 1]
        self.add_input(decoder_word_input_name, decoder_word_input_shape)
        decoder_position_input_name = "decoder_positions"
        decoder_position_input_shape = [self.batch, 1]
        self.add_input(decoder_position_input_name, decoder_position_input_shape)
        self.set_input(input)

        output_name_prefix = "transformer_decoder"
        self.scopes[0] = "transformer"
        self.scopes[1] = "target_embedding"
        self.target_modality = "target_embedding"
        self.add_weight(output_name=self.target_modality, scope_id=2)

        decoder_states = self.prepare_decoder_states(decoder_position_input_name)
        self.save_input()

        zero = "zero"
        zero_weight = np.zeros([self.batch, self.decoder_params["num_units"]])
        self.add_weight(output_name=zero, weight=zero_weight)

        sos = "sos"
        sos_weight = np.zeros([self.batch, 1])
        self.add_weight(output_name=sos, weight=sos_weight)

        position_embedding_name = output_name_prefix + "_position_embedding"
        position_embedding_dict_name = self.position_encoding(length=self.decoder_params["position.max_length"],
                                           depth=self.decoder_params["num_units"] // 2,
                                           output_name_prefix=output_name_prefix)

        # word embedding
        word_embedding_shape = [self.batch, self.decoder_params["num_units"]]
        word_embedding_result = output_name_prefix + "_words_embedding_buffer"
        self.add_memory(word_embedding_result, word_embedding_shape, data_type="FLOAT32")
        is_first_word = "is_first_word"
        self.add_check(sos, decoder_position_input_name, "equal", is_first_word)
        jump_name = "skip_first_word_embedding"
        self.add_jump(position_embedding_name, jump_name, is_first_word)
        is_first_word_data = self.get_tensor(is_first_word).tolist()[0]
        if (isinstance(is_first_word_data, bool) and not is_first_word_data) \
        or (isinstance(is_first_word_data, int) and is_first_word_data == 0) \
        or (isinstance(is_first_word_data, float) and is_first_word_data == 0):
            cur_inputs = output_name_prefix + "_words_embedding"
            self.add_embedding(decoder_word_input_name, self.target_modality, cur_inputs)
            cur_inputs_scale = output_name_prefix + "_words_embedding" + "_s"
            self.add_power(cur_inputs, cur_inputs_scale, scale=(self.decoder_params['num_units'] ** 0.5))
            self.add_copy(cur_inputs_scale, self.decoder_params["num_units"], self.decoder_params["num_units"], 0,
                          word_embedding_result, self.decoder_params["num_units"], self.decoder_params["num_units"], 0,
                          self.decoder_params["num_units"], output_name="copy_word_embedding")

        # position embedding
        self.add_embedding(decoder_position_input_name, position_embedding_dict_name, position_embedding_name)

        output_name = output_name_prefix + "_embedding"
        cur_inputs_pos = self.add_sum([word_embedding_result, position_embedding_name], output_name)
        self.scopes[1] = "decoder"
        cell_outputs, state, att_context = self.attention_ffn_block(inputs=cur_inputs_pos,
                            encoder_mask=None,
                            attention_mask=None,
                            scope_id=2,
                            output_name_prefix=output_name_prefix,
                            state=decoder_states,
                            position=decoder_position_input_name)

        self.scopes[1] = "softmax"
        self.softmax = "softmax"
        self.add_weight(output_name=self.softmax, scope_id=2)
        current_ids = self.add_projection(cell_outputs, self.softmax, output_name)
        # current_ids = self.add_projection(cell_outputs, self.target_modality, output_name)
        output = self.save_decoder_states(state, decoder_position_input_name)
        output.append(current_ids)
        self.add_output(output)
        self.save_caffe_model()
        return current_ids
    
    def generate_encoder(self, input=None):
        encoder_word_input_name = "encoder_words"
        encoder_word_input_shape = [self.batch, self.max_seq_length]
        self.add_input(encoder_word_input_name, encoder_word_input_shape)
        self.set_input(input)

        self.scopes[0] = "transformer"
        self.scopes[1] = "source_embedding"
        self.source_modality = "source_embedding"
        self.add_weight(output_name=self.source_modality, scope_id=2)

        encoder_embedding = "encoder_embedding"
        self.add_embedding(encoder_word_input_name, self.source_modality, encoder_embedding)
        encoder_embedding = self.add_power(encoder_embedding, encoder_embedding+"_s", scale=math.sqrt(self.encoder_params["num_units"]))

        # source embedding bias
        self.scopes[1] = "bias"
        embedding_bias = "bias"
        self.add_weight(output_name=embedding_bias, scope_id=2)
        encoder_embedding = self.add_sum([encoder_embedding, embedding_bias], encoder_embedding + "_b")

        self.scopes[1] = "encoder"
        encoder_attention_mask = "encoder_attention_mask"
        encoders = self.encode(encoder_embedding, None, encoder_attention_mask, 2, output_name_prefix="transformer_encoder")
        self.encoder_post_process("transformer_decoder")

        self.save_input()
        self.save_caffe_model()
        self.save_encoder_states()
