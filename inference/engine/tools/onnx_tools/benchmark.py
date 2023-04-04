#!/usr/bin/python
import numpy as np
import onnxruntime
import sys
import os
import time
from common import *

loops = 16
warm = 1

if (len(sys.argv) < 2):
    print("usage: ./xx.py onnxModelPath inputDataPath")
    print("[ERROR] please provide a valid onnx model path.\n");
    exit(1)
model_path = sys.argv[1]
data_path = None
if (len(sys.argv) >= 3):
    data_path = sys.argv[2]
    print("[INFO] use data(%s) to infer %s.\n" % (data_path, model_path))
else:
    print("[INFO] use data(1) to infer %s.\n" % (model_path))

options = onnxruntime.SessionOptions()
#options.enable_profiling = True
options.intra_op_num_threads = 1
options.inter_op_num_threads = 1
options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
inference_session = onnxruntime.InferenceSession(model_path, options, providers=['CPUExecutionProvider'])

print("Input Information:")
input_shapes = None
if (data_path is not None and os.path.isfile(data_path + "/shape.txt")):
    input_shapes = read_shape(data_path + "/shape.txt")
model_inputs = {}
type_mapping = {"tensor(int64)": np.int64,
    "tensor(uint64)": np.uint64,
    "tensor(int32)": np.int32,
    "tensor(uint32)": np.uint32,
    "tensor(int8)": np.int8,
    "tensor(uint8)": np.uint8,
    "tensor(bool)": np.bool,
    "tensor(float)": np.float32}
inputs_info = inference_session.get_inputs()
for i in range(inputs_info.__len__()):
    input_name = inputs_info[i].name
    input_type = inputs_info[i].type
    input_shape = inputs_info[i].shape
    input_shape_old = input_shape.copy()

    input_dynamic = False;
    for i in range(len(input_shape)):
        if (isinstance(input_shape[i], str)):
            input_dynamic = True;
            if (i == 0):
                input_shape[i] = 1;
            else:
                input_shape[i] = 11;
    if (input_shapes is not None and input_name in input_shapes):
        input_shape = input_shapes[input_name]
    elif (input_dynamic):
        print("[WARNING] %s is dynamic input shape %s, we will change it to %s."
                % (input_name, input_shape_old, input_shape))

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
    data = data.astype(type_mapping[input_type])
    model_inputs[input_name] = data 
    print("Input Tensor %s %s" % (input_name, string(data, 8)))

for i in range(warm):
    model_outputs = inference_session.run(None, model_inputs)

start = time.time()
for i in range(loops):
    model_outputs = inference_session.run(None, model_inputs)
end = time.time()
total = end - start

print("\nBenchmark result:")
outputs_info = inference_session.get_outputs()
for i in range(outputs_info.__len__()):
    output_name = outputs_info[i].name
    output_type = outputs_info[i].type
    output_shape = outputs_info[i].shape
    data = model_outputs[i]
    print("Output Tensor %s %s" % (output_name, string(data, 8)))
print("\nrun avg_time:%fms/data" % (total / loops * 1000))
