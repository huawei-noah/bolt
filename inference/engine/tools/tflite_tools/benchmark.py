#!/usr/bin/python
import tensorflow as tf
from tensorflow import keras
import numpy as np
import sys
import os
import time
sys.path.append('../onnx_tools')
from common import *

loops = 16
warm = 1

if (len(sys.argv) < 2):
    print("usage: ./xx.py tfliteModelPath inputDataPath newOutputPath")
    print("    inputDataPath is optional, can be replaced by None")
    print("    newOutputPath is optional, can be replaced by None")
    print("[ERROR] please provide a valid tflite model path.\n");
    exit(1)
model_path = sys.argv[1]
data_path = None
new_out_path = None
if (len(sys.argv) >= 3):
    data_path = sys.argv[2]
    if (data_path == 'None'):
        data_path = None
if (len(sys.argv) >= 4):
    new_out_path = sys.argv[3]
    if (new_out_path == 'None'):
        new_out_path = None
print("[INFO] use data(%s) to infer %s, with new outputs(%s).\n" % (data_path, model_path, new_out_path))

interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

print("Input Information:")
input_shapes = None
if (data_path is not None and os.path.isfile(data_path + "/shape.txt")):
    input_shapes = read_shape(data_path + "/shape.txt")
inputs_info = interpreter.get_input_details()
for i in range(len(inputs_info)):
    input_name = inputs_info[i]['name']
    input_shape = inputs_info[i]['shape']
    if (input_shapes is not None and input_name in input_shapes):
        input_shape = input_shapes[input_name]

    if (data_path is None):
        data = np.ones(input_shape)
    elif (os.path.isfile(data_path)):
        data = np.loadtxt(data_path).reshape(input_shape)
    elif (os.path.isdir(data_path)):
        path = data_path + "/" + input_name + ".txt"
        if (os.path.isfile(path)):
            data = np.loadtxt(path).reshape(input_shape)
        else:
            print("[ERROR] can not find %s file in directory %s.\n" % (path, data_path))
            exit(1)
    data = data.astype(inputs_info[i]['dtype'])
    print("Input Tensor %s %s" % (input_name, string(data, 8)))
    interpreter.set_tensor(inputs_info[i]['index'], data)


for i in range(warm):
    interpreter.invoke()

start = time.time()
for i in range(loops):
    interpreter.invoke()
end = time.time()
total = end - start

print("\nBenchmark result:")
outputs_info = interpreter.get_output_details()
for i in range(outputs_info.__len__()):
    output_name = outputs_info[i]['name']
    output_shape = outputs_info[i]['shape']
    data = interpreter.get_tensor(outputs_info[i]['index'])
    print("Output Tensor %s %s" % (output_name, string(data, 8)))

idxes = {}
tensors = interpreter.get_tensor_details()
for tensor in tensors:
    idxes[tensor['name']] = tensor['index']
names = []
if (new_out_path != None):
    f = open(new_out_path, "r")
    names = f.readlines()
    f.close()
for name in names:
    name = name.strip()
    idx = idxes[name]
    data = interpreter.get_tensor(idx)
    print("Output Tensor %s %s" % (name, string(data, 8)))
print("\nrun avg_time:%fms/data" % (total / loops * 1000))
