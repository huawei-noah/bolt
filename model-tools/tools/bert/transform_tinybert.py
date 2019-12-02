#!/usr/local/bin/python
# -*- coding: utf-8 -*-

from tensorflow2caffe_tinybert import Tensorflow2CaffeTinyBert
import numpy as np

if __name__ == '__main__':
    tensorflow_model_path = "../tinybert_eps-3/tf_ckpt/model.ckpt"
    seq_length = 32
    encoder_layers = 4
    attention_nums = 12
    caffe_model_path_prefix = "tinybert"

    bert_caffe = Tensorflow2CaffeTinyBert(tensorflow_model_path, seq_length, encoder_layers, attention_nums, caffe_model_path_prefix, True)
    bert_caffe.weight_map()
    bert_caffe.generate()
    intent = bert_caffe.data_dict["intent_softmax"]
    slot =  bert_caffe.data_dict["slot_softmax"]
    intent_label = np.argmax(intent)
    print("intent %d: %f" % (intent_label, intent[intent_label]))
    slot_labels = "slot_label:"
    for i in range(len(slot[0])):
         slot_label = np.argmax(slot[0][i])
        slot_labels = slot_labels + " " + str(slot_label))
    print("%s" % (slot_labels))
