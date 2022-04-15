#!/usr/local/bin/python
# -*- coding: utf-8 -*-

import sys
sys.path.append("../")
from tensorflow2caffe import Tensorflow2Caffe
from Caffe import caffe_net
from operators import Operators



class Tensorflow2CaffePunctuation(Tensorflow2Caffe):
    def __init__(self,
            tensorflow_model_path, caffe_model_path_prefix, caffe_model_name,
            check=False, calc=False):
        Tensorflow2Caffe.__init__(self, tensorflow_model_path,
            caffe_model_path_prefix, caffe_model_name, check, calc)

    def generate(self, input = None):
        #batch_seq_length = 16
        input_text_name = "input_text" 
        input_text_shape = [self.batch, 16]
        self.add_input(input_text_name, input_text_shape)
        self.set_input(input)
        #embedding
        x = self.extract_embedding(input_text_name, 0, "emb_table", "embedding_lookup")
        #bilstm
        x = self.extract_rnn("LSTM", x, None, "BiLSTM", 0, steps = -2, scope_name = ["BiLSTM/fw/lstm_cell", "BiLSTM/bw/lstm_cell"])
        #FC
        weight = self.get_weight("W")
        bias = self.get_weight("b")
        layer = caffe_net.LayerParameter("wb_fc_output", type='InnerProduct',
                        bottom=[x], top=["wb_fc_output"])
        num_output = len(weight[0])
        weight = weight.transpose((1,0))
        layer.inner_product_param(num_output, bias_term=bias is not None)
        if len(bias) != num_output:
            print("[ERROR] extract_dense failed")
            exit(0)
        layer.add_data(weight, bias)
        self.caffe_model.add_layer(layer)
        self.data_dict["wb_fc_output"] = Operators.fully_connect(self.data_dict[x],
                                             weight.transpose((1, 0)), bias,
                                             "wb_fc_output")
        x = "wb_fc_output"
        #softmax
        x = self.add_softmax(x, "softmax_output", -1)
        #argmax
        x = self.add_argmax(x, -1, "output")
        self.save_caffe_model()

