#!/usr/bin/python
import sys
import os
import onnx
import onnx.checker
import onnx.utils
from onnx.tools import update_model_dims
from common import *

if (len(sys.argv) != 4):
    print("usage: ./xx.py original_onnx_model_path new_onnx_model_path shape_file_path")
    exit(1)
model_name = sys.argv[1]
model_name_new = sys.argv[2]
shape_path = sys.argv[3]

print("[INFO] use shape in %s to change model %s input dimension, and saved in %s." % (shape_path, model_name, model_name_new))

input_shapes = None
if (os.path.isfile(shape_path)):
    input_shapes = read_shape(shape_path)
else:
    print("[ERROR] can not read shape file %s." % (shape_path))
    exit(1)

model = onnx.load(model_name)
for i in range(model.graph.input.__len__()):
    dim_proto0 = model.graph.input[i].type.tensor_type.shape.dim
    input_name = model.graph.input[i].name
    print("input: %s, old dimension:" % (input_name))
    print(model.graph.input[i].type.tensor_type.shape.dim)

    # change dimension on some axis
    if (input_shapes is not None and input_name in input_shapes):
        shape = input_shapes[input_name]
        for j in range(len(shape)):
            dim_proto0[j].dim_value = shape[j]

    print("input: %s, new dimension:" % (input_name))
    print(model.graph.input[i].type.tensor_type.shape.dim)
onnx.save(model, model_name_new)
