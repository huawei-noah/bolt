#!/usr/bin/python

import onnx
import onnx.helper as helper
import sys

if (len(sys.argv) != 4):
    print("usage: ./xx.py original_onnx_model_path new_onnx_model_path new_output_names(split_by_comma)")
    exit(1)
model_name = sys.argv[1]
model_name_new = sys.argv[2]
new_output_names = sys.argv[3].strip().split(",")

print("[INFO] add model output %s to %s, and saved in %s." % (new_output_names, model_name, model_name_new))

model = onnx.load(model_name)
nodes = []
for name in new_output_names:
    intermediate_layer_value_info = helper.ValueInfoProto()
    intermediate_layer_value_info.name = name
    nodes.append(intermediate_layer_value_info)
model.graph.output.extend(nodes)
onnx.save(model, model_name_new)
