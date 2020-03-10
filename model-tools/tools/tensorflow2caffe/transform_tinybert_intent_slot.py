#!/usr/local/bin/python
# -*- coding: utf-8 -*-

from tensorflow2caffe_tinybert import Tensorflow2CaffeTinyBert
import numpy as np

if __name__ == '__main__':
    tensorflow_model_path = "../tinybert_eps-3/tf_ckpt/model.ckpt"
    max_seq_length = 32
    embedding_dim = 312
    encoder_layers = 4
    num_heads = 12
    caffe_model_path_prefix = "tinybert_intent_slot"
    caffe_model_name = "tinybert_intent_slot"

    bert_caffe = Tensorflow2CaffeTinyBert(tensorflow_model_path,
                     caffe_model_path_prefix, caffe_model_name,
                     max_seq_length, embedding_dim, encoder_layers, num_heads,
                     True, True)
    data = {}
    data["tinybert_words"]      = np.array([[101,1045,2342,1037,14764,2005,2296,5353,3531,102]])
    tinybert_length = len(data["tinybert_words"][0])
    data["tinybert_positions"]  = np.array([[i for i in range(tinybert_length)]])
    data["tinybert_token_type"] = np.array([[0] * tinybert_length])
    #data["tinybert_mask"]       = np.array([[1] * tinybert_length])
    bert_caffe.generate_intent_slot_task(data)

    intent = bert_caffe.data_dict["intent_softmax"]
    slot =  bert_caffe.data_dict["slot_softmax"]
    intent_label = np.argmax(intent)
    print("intent %d: %f" % (intent_label, intent[intent_label]))
    slot_labels = ""
    for i in range(len(slot[0])):
        slot_label = np.argmax(slot[0][i])
        slot_labels = slot_labels + str(slot_label) + " "
    print("slot_label: %s" % (slot_labels))
