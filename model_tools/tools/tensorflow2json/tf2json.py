from google.protobuf import json_format
import sys
import json
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.python.framework import tensor_util
import math

GRAPH_PB_PATH = ""
SAVE_JSON_PATH = ""
if (len(sys.argv) != 2):
    print("Error input, please input 2 params(GRAPH_PB_PATH and SAVE_JSON_PATH) respectively.\n")
else:
    GRAPH_PB_PATH = sys.argv[0]
    SAVE_JSON_PATH = sys.argv[1]

global_weight_dict = {}
with tf.Session() as sess:
    with gfile.FastGFile(GRAPH_PB_PATH, "rb") as f:
        graph_def = tf.GraphDef()        
        graph_def.ParseFromString(f.read())
        json_string = json_format.MessageToJson(graph_def)
        
        json_string = json_string.replace(' ', '')
        json_string = json_string.replace('\n', '')

        d = json.loads(json_string)
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')
        graph_nodes=[n for n in graph_def.node]
        for item in graph_nodes:
            if item.op == "Const":
                weight_values = (tensor_util.MakeNdarray(item.attr['value'].tensor).astype("float64")).flatten().tolist()
                for wvIndex in range(len(weight_values)):
                    if weight_values[wvIndex]  == math.inf:
                        weight_values[wvIndex] = np.finfo(np.float64).max
                    elif weight_values[wvIndex]  == -math.inf:
                        weight_values[wvIndex] = np.finfo(np.float64).min
                tmp_numpy_arr = tensor_util.MakeNdarray(item.attr['value'].tensor)
                weight_values_new = tmp_numpy_arr.astype('float64').flatten().tolist()
                weight_op_name = item.name
                global_weight_dict[weight_op_name] = weight_values  

        totalConstIndex = 0
        constIndex = 0
        for node in d["node"]:
            if node["op"] == "Const":
                totalConstIndex = totalConstIndex + 1   
                node["attr"]["value"]["tensor"]["tensorContent"] = global_weight_dict[node["name"]]
                constIndex = constIndex + 1 

        final_dict = json.dumps(d)
        with open(SAVE_JSON_PATH, 'w', encoding="utf-8") as f:
            json.dump(final_dict, f, ensure_ascii=False)
