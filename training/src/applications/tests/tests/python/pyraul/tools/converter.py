# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#!/usr/bin/env python
import torch
import numpy as np
import os


def cvt_tensor_to_raul(
    tensor: torch.Tensor, name: str = "tensor", memory_manager="memory_manager"
):
    def get_index_check(idx):
        return len(tensor.shape) > idx and tensor.shape[idx] or 1

    assert len(tensor.shape) < 5
    print(
        f'{memory_manager}.createTensor("{name}", {get_index_check(0)}, {get_index_check(1)}, {get_index_check(2)}, {get_index_check(3)});'
    )
    data = ", ".join([f"{x}_dt" for x in tensor.numpy().flat])
    print(f'{memory_manager}["{name}"] = raul::dtVec {{{data}}};')


def cvt_model_to_raul(model):
    wire = "data"
    for i, layer in enumerate(model.children()):
        name = layer._get_name()
        if name == "Linear":
            out_wire = f"fc{i}"
            outputs_count = layer.out_features
            print(
                f'netdef.addOp(  "{out_wire}", FULLY_CONNECTED_LAYER, createParam(raul::FCParams{{ {{ "{wire}" }}, {{ "{out_wire}" }}, {outputs_count} }}));'
            )
            wire = out_wire
            continue
        if name == "Swish":
            out_wire = f"swish{i}"
            print(
                f'netdef.addOp(  "{out_wire}", SWISH_ACTIVATION, createParam(raul::SwishActivationParams({{ {{ "{wire}" }}, {{ "{out_wire}" }})));'
            )
            wire = out_wire
            continue
        if name == "LogSoftmax":
            out_wire = f"softmax{i}"
            print(
                f'netdef.addOp(  "{out_wire}", LOG_SOFTMAX_ACTIVATION, createParam(raul::BasicParams{{ {{ "{wire}" }}, {{ "{out_wire}" }} }}));'
            )
            wire = out_wire
            continue
        if name == "HSigmoid":
            out_wire = f"hsigmoid{i}"
            print(
                f'netdef.addOp(  "{out_wire}", HSIGMOID_ACTIVATION, createParam(raul::HSigmoidActivationParams{{ {{ "{wire}" }}, {{ "{out_wire}" }} }}));'
            )
            wire = out_wire
            continue
        print(f"Unknown layer '{name}'")
