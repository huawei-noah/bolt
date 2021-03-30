import os
import argparse
import time
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

from tokenization import BertTokenizer
import modeling


def get_labels(label_file):
    with open(label_file, 'r', encoding='utf-8') as f:
        labels = f.readlines()

    labels = [label.strip() for label in labels]
    return labels


def create_model(bert_config, input_ids, input_mask, segment_ids,
                        num_intent_labels, num_slot_labels):
    model = modeling.BertModel(config=bert_config,
                               is_training=False,
                               input_ids=input_ids,
                               input_mask=input_mask,
                               token_type_ids=segment_ids,
                               use_one_hot_embeddings=False)

    output_layer = model.get_pooled_output()
    print('output layer :{}'.format(output_layer))
    sequence_output = model.get_sequence_output()
    print('output layer :{}'.format(sequence_output))

    hidden_size = output_layer.shape[-1].value

    intent_classifier_weight = tf.get_variable(
        "intent_classifier_weight", [num_intent_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    intent_classifier_bias = tf.get_variable(
        "intent_classifier_bias", [num_intent_labels], initializer=tf.zeros_initializer())
    intent_logits = tf.matmul(output_layer, intent_classifier_weight, transpose_b=True)
    intent_confidence = tf.nn.softmax(tf.nn.bias_add(intent_logits, intent_classifier_bias), name='intent_confidence')
    intent = tf.argmax(intent_confidence, axis = -1, name='intent',output_type=tf.int32)
    slot_classifier_weight = tf.get_variable(
        "slot_classifier_weight", [num_slot_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    slot_classifier_bias = tf.get_variable(
        "slot_classifier_bias", [num_slot_labels], initializer=tf.zeros_initializer())
    slot_logits = tf.tensordot(sequence_output, tf.transpose(slot_classifier_weight), axes = [[2], [0]])
    slot = tf.argmax(tf.nn.softmax(tf.nn.bias_add(slot_logits, slot_classifier_bias)), 
                                    axis = -1, name='slot',output_type=tf.int32)
    return intent, intent_confidence, slot


def write_data(f, array):
    array = np.array(array)
    num = array.size
    array = np.reshape(array, [num])
    output = str(len(array)) + " "
    for x in array:
        output = output + str(x) + " ";
    f.write("%s\n" % (output))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default=None, required=True, type=str)
    parser.add_argument("--ckpt", default=None, type=str)
    parser.add_argument("--use_pb", default=None, action='store_true')
    parser.add_argument("--use_bolt", default=False, type=bool)
    args = parser.parse_args()

    tokenizer = BertTokenizer.from_pretrained(args.path)
    intent_list = get_labels(os.path.join(args.path,'intention_labels.txt'))
    slot_list = get_labels(os.path.join(args.path,'slot_labels.txt'))
    with tf.Session() as sess:
        if args.use_pb:
            print("[INFO] use tensorflow .pb model")
            with gfile.FastGFile(os.path.join(args.path,'model.pb'), 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                sess.graph.as_default()
                tf.import_graph_def(graph_def, name='')
        else:
            print("[INFO] use tensorflow .ckpt model")
            bert_config = modeling.BertConfig.from_json_file(os.path.join(args.ckpt,'config.json'))
            input_ids = tf.placeholder(tf.int32, [None, None], name="input_ids")
            input_mask = tf.placeholder(tf.int32, [None, None], name="input_mask")
            segment_ids = tf.placeholder(tf.int32, [None, None], name="segment_ids")
            tvars = tf.trainable_variables()
            initialized_variable_names = {}
            intent, intent_confidence, slot= create_model(
                bert_config, input_ids, input_mask, segment_ids, len(intent_list),len(slot_list))
            saver = tf.train.Saver()
            saver.restore(sess,os.path.join(args.ckpt, 'model.ckpt'))

        input_ids = sess.graph.get_tensor_by_name('input_ids:0')
        input_mask = sess.graph.get_tensor_by_name('input_mask:0')
        segment_ids = sess.graph.get_tensor_by_name('segment_ids:0')

        intent = sess.graph.get_tensor_by_name('intent:0')
        intent_confidence = sess.graph.get_tensor_by_name('intent_confidence:0')
        slot = sess.graph.get_tensor_by_name('slot:0')

        text = "i need a reminder for every weekend please"
        tokens = tokenizer.tokenize(text)
        tokens = ["[CLS]"] + tokens + ["[SEP]"]

        input_id = tokenizer.convert_tokens_to_ids(tokens)
        seq_length = len(tokens)
        position = []
        for i in range(seq_length):
            position.append(i)
        segment = [0] * seq_length

        if (args.use_bolt):
            print("[INFO] use bolt inference")
            f = open("sequence.seq", "w")
            write_data(f, input_id)
            write_data(f, position)
            write_data(f, segment)
            f.close()
            os.system("./adb_run.sh")
            f = open("result.txt", "r")
            lines = f.readlines();
            for line in lines:
                line = line.strip();
                if (line.startswith("intent:")):
                    array = line.split(" ")
                    intent_id = array[1]
                    intent_prob = array[2]
                if (line.startswith("slot:")):
                    array = line.split(" ")
                    slot_ids = []
                    for i in range(i):
                        slot_ids.append(int(array[i+1]))
                if (line.startswith("avg_time:")):
                    array = line.split(":")
                    time_use = (array[1].split("ms"))[0]
        else:
            print("[INFO] use tensorflow inference")
            time_start = time.time()
            ret1, ret2, ret3 = sess.run([intent,intent_confidence,slot],  feed_dict={input_ids: [input_id], input_mask: [mask],segment_ids:[segment]})
            time_end = time.time()
            time_use = (time_end - time_start) * 1000.0
            intent_id = ret1[0]
            intent_prob = ret2[0][ret1[0]]
            slot_ids = ret3[0]

        print('\t'.join(tokens))
        print('\t'.join([slot_list[slot_id] for slot_id in slot_ids]))
        print("intention: {}".format(intent_id))
        print("intent confidence: {}".format(intent_prob))
        print("time: {} ms".format(time_use))
