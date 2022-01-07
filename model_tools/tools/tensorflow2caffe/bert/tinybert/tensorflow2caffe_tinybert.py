#!/usr/local/bin/python
# -*- coding: utf-8 -*-
import sys
sys.path.append("../")
sys.path.append("../../")
from Caffe import caffe_net
from operators import Operators
from tensorflow2caffe_bert import Tensorflow2CaffeBert

class Tensorflow2CaffeTinyBert(Tensorflow2CaffeBert):
    def __init__(self,
            tensorflow_model_path, caffe_model_path_prefix, caffe_model_name,
            max_seq_length, embedding_dim, encoder_layers, num_heads,
            check=False, calc=False):
        Tensorflow2CaffeBert.__init__(self, tensorflow_model_path, caffe_model_path_prefix, caffe_model_name,
            max_seq_length, embedding_dim, encoder_layers, num_heads, check, calc)
        self.max_ambiguate = (self.max_seq_length * self.max_seq_length - 1) // 2

    def extract_dense_prefix(self, input_name, dense_name, kernel_name, bias_name):
        kernel = self.get_weight(kernel_name)
        bias = self.get_weight(bias_name)
        layer = caffe_net.LayerParameter(name=dense_name, type='InnerProduct',
                                      bottom=[input_name], top=[dense_name])
        num_output = len(kernel)
        layer.inner_product_param(num_output, bias_term=bias is not None)
        if bias is not None:
            if len(bias) != num_output:
                print("[ERROR] extract extract_dense_prefix")
            layer.add_data(kernel, bias)
        else:
            layer.add_data(kernel)
        self.caffe_model.add_layer(layer)
        self.data_dict[dense_name] = Operators.fully_connect(self.data_dict[input_name],
                                                             kernel.transpose((1, 0)), bias,
                                                             dense_name)
        return dense_name

    def extract_intent_classifier(self, input_name):
        dense_name = "intent_classifier"
        weight_name_prefix = "intent_classifier_"
        self.extract_dense_prefix(input_name, dense_name, weight_name_prefix+"weight", weight_name_prefix+"bias")

        softmax_name = "intent_softmax"
        self.add_softmax(dense_name, softmax_name, -1)

        return softmax_name

    def extract_slot_classifier(self, input_name):
        dense_name = "slot_classifier"
        weight_name_prefix = "slot_classifier_"
        self.extract_dense_prefix(input_name, dense_name, weight_name_prefix+"weight", weight_name_prefix+"bias")

        softmax_name = "slot_softmax"
        self.add_softmax(dense_name, softmax_name, -1)

        return softmax_name

    def extract_mrpc_classifier(self, input_name):
        dense_name = "mrpc_classifier"
        weight_name_prefix = "classifier_"
        self.extract_dense_prefix(input_name, dense_name, weight_name_prefix+"weight", weight_name_prefix+"bias")

        softmax_name = "mrpc_softmax"
        self.add_softmax(dense_name, softmax_name, -1)

        return softmax_name

    def extract_tts_preprocess_task(self, x, scope_id, scope_name, weight_name, bias_name):
        self.scopes[scope_id] = scope_name
        for i in range(3):
            name = scope_name + '_dense%d' % (i+1)
            x = self.extract_dense(x, name, scope_id+1, scope_name='dense_%d' % (i+1))
            x = self.add_relu(x, name+"_relu")
        logits = self.extract_dense_prefix(x, scope_name+"_dense", weight_name, bias_name)
        #logits = tf.reshape(logits, [batch_size, seq_length, num_labels])
        pred_ids = self.add_argmax(logits, axis=-1, output_name=scope_name+"_argmax")
        return pred_ids

    def generate_intent_slot_task(self, input=None):
        word_input_name = "tinybert_words"
        position_input_name = "tinybert_positions"
        token_input_name = "tinybert_token_type"
        #attention_mask_input_name = "tinybert_attention_mask"
        word_input_shape = [self.batch, self.max_seq_length]
        position_input_shape = [self.batch, self.max_seq_length]
        token_input_shape = [self.batch, self.max_seq_length]
        #attention_mask_input_shape = [self.batch, self.max_seq_length]

        self.add_input(word_input_name, word_input_shape)
        self.add_input(position_input_name, position_input_shape)
        self.add_input(token_input_name, token_input_shape)
        #self.add_input(attention_mask_input_name, attention_mask_input_shape)
        self.set_input(input)
        self.save_input()

        attention_mask_name = None #"attention"
        #self.add_attention(attention_mask_input_name, self.num_heads, self.max_seq_length, self.max_seq_length, attention_mask_name);

        output_name = self.extract_embeddings(word_input_name, position_input_name, token_input_name)
        output_names = self.extract_encoder(output_name, attention_mask_name)
        output_name = self.extract_pooler(output_names[-1])
        intent = self.extract_intent_classifier(output_name)
        slots = self.extract_slot_classifier(output_names[-1])

        self.save_caffe_model()

    def generate_mrpc_task(self, input=None):
        word_input_name = "tinybert_words"
        position_input_name = "tinybert_positions"
        token_input_name = "tinybert_token_type"
        #attention_mask_input_name = "tinybert_attention_mask"
        word_input_shape = [self.batch, self.max_seq_length]
        position_input_shape = [self.batch, self.max_seq_length]
        token_input_shape = [self.batch, self.max_seq_length]
        attention_mask_input_shape = [self.batch, self.max_seq_length]

        self.add_input(word_input_name, word_input_shape)
        self.add_input(position_input_name, position_input_shape)
        self.add_input(token_input_name, token_input_shape)
        #self.add_input(attention_mask_input_name, attention_mask_input_shape)
        self.set_input(input)
        self.save_input()

        attention_mask_name = None #"attention"
        #self.add_attention(attention_mask_input_name, self.num_heads, self.max_seq_length, self.max_seq_length, attention_mask_name);

        output_name = self.extract_embeddings(word_input_name, position_input_name, token_input_name)
        output_names = self.extract_encoder(output_name, attention_mask_name)
        output_name = self.extract_pooler(output_names[-1])
        mrpc = self.extract_mrpc_classifier(output_name)

        self.save_caffe_model()

    def generate_disambiguate_task(self, input=None):
        word_input_name = "tinybert_words"
        position_input_name = "tinybert_positions"
        token_input_name = "tinybert_token_type"
        word_mask_input_name = "tinybert_words_mask"
        dict_input_name = "tinybert_dict_type"
        word_input_shape = [self.batch, self.max_seq_length]
        position_input_shape = [self.batch, self.max_seq_length]
        token_input_shape = [self.batch, self.max_seq_length]
        word_mask_input_shape = [self.batch, self.max_ambiguate, self.max_seq_length]
        dict_input_shape = [self.batch, self.max_ambiguate]

        self.add_input(word_input_name, word_input_shape)
        self.add_input(position_input_name, position_input_shape)
        self.add_input(token_input_name, token_input_shape)
        self.add_input(word_mask_input_name, word_mask_input_shape)
        self.add_input(dict_input_name, dict_input_shape)
        self.set_input(input)
        self.save_input()

        attention_mask_name = None

        output_name = self.extract_embeddings(word_input_name, position_input_name, token_input_name, ["dense", "kernel", "bias"])
        output_names = self.extract_encoder(output_name, attention_mask_name)
        output_name1 = self.add_reduce_sum(output_names[-1], 1, True, "mask_reduce_sum", word_mask_input_name)
        output_name2 = self.extract_embedding(dict_input_name, 0, "onehot_vec_weight", "dict_embedding_dense")
        onehot_bias = self.add_weight("onehot_vec_bias", weight_name="onehot_vec_bias", data_type="FLOAT32")
        output_name2 = self.add_sum([output_name2, onehot_bias], "dict_embedding")
        output_name = self.add_concat([output_name1, output_name2], "mask_result_concat", 2)
        dense_name = "slot_classifier1"
        weight_name_prefix = "slot_classifier1_"
        output_name = self.extract_dense_prefix(output_name, dense_name, weight_name_prefix+"weight", weight_name_prefix+"bias")
        slots = self.extract_slot_classifier(dense_name)

        self.save_caffe_model()

    def generate_tts_preprocess_task(self, input=None):
        word_input_name = "tinybert_words"
        position_input_name = "tinybert_positions"
        token_input_name = "tinybert_token_type"
        word_input_shape = [self.batch, self.max_seq_length]
        position_input_shape = [self.batch, self.max_seq_length]
        token_input_shape = [self.batch, self.max_seq_length]

        self.add_input(word_input_name, word_input_shape)
        self.add_input(position_input_name, position_input_shape)
        self.add_input(token_input_name, token_input_shape)
        self.set_input(input)
        self.save_input()

        attention_mask_name = None

        output_name = self.extract_embeddings(word_input_name, position_input_name, token_input_name)
        output_names = self.extract_encoder(output_name, attention_mask_name)
        self.scopes[0] = "loss"
        output1 = self.extract_tts_preprocess_task(output_names[-1], 1, 'dense_layer', "output_weights1", "output_bias1")
        output2 = self.extract_tts_preprocess_task(output_names[-1], 1, 'dense_layer2', "output_weights2", "output_bias2")

        self.save_caffe_model()
