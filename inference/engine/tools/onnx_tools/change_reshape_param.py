#!/usr/bin/python

import onnx
import numpy as np

model_name = "../old.onnx"
model_name_new = "./new.onnx"
layer_name = {"440", "660"}

onnx_model = onnx.load(model_name)
graph = onnx_model.graph
for i in range(graph.initializer.__len__()):
    name = graph.initializer[i].name
    if (name in layer_name):
        graph.initializer.remove(graph.initializer[i])
        new_param = onnx.helper.make_tensor(name, onnx.TensorProto.INT64, [2], [-1, 512])
        graph.initializer.insert(i, new_param)
        print(graph.initializer[i])
onnx.checker.check_model(onnx_model)
onnx.save(onnx_model, model_name_new)
