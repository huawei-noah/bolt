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
from enum import Enum
import numpy as np
import os
from collections import namedtuple
from typing import Optional
import math

_default_trace_file = "loss.txt"
_default_wights_file = "weights.txt"
_default_checkpoint_file = "checkpoint.pth.tar"

DumpConfig = namedtuple("DumpConfig", ("filename", "step"))


def gen_cpp_dtVec(tensor, name="tensor"):
    return (
        f"const raul::Tensor {name}{{" + ", ".join([f"{x:e}_dt" for x in tensor]) + "};"
    )


class DumpMode(Enum):
    flatten_transpose = 0
    transpose_flatten = 1


def dump_tensor(tensor, filename: str, mode: Optional[DumpMode]):
    with open(filename, "w") as outfile:
        data = tensor.data.cpu()
        if mode == DumpMode.flatten_transpose:
            data = data.flatten()
            data = np.transpose(data)
        elif mode == DumpMode.transpose_flatten:
            data = np.transpose(data)
            data = data.flatten()
        np.savetxt(outfile, data)


def dump_weights(
    model,
    filename: str = _default_wights_file,
    mode: Optional[DumpMode] = None,
    filter="",
) -> None:
    param_list = [
        (name, param)
        for (name, param) in model.named_parameters()
        if filter in name and param.requires_grad
    ]
    state_list = [
        (name, param)
        for (name, param) in model.state_dict().items()
        if "running" in name and filter in name
    ]
    for name, param in param_list + state_list:
        dump_tensor(param, add_suffix_to_filepath(filename, name), mode)


def dump_loss(loss, filename: str = _default_trace_file) -> None:
    with open(filename, "a") as out_file:
        print(loss.item(), file=out_file)


def add_index_to_filepath(filename: str, index: int) -> str:
    add_suffix_to_filepath(filename, str(index))


def add_suffix_to_filepath(filename: str, suffix: str) -> str:
    data = list(os.path.splitext(filename))
    data[-2] = f"{data[-2]}_{suffix}"
    return "".join(data)


def save_checkpoint(state, filename=_default_checkpoint_file):
    torch.save(state, filename)


def get_number_of_precision_digits(bits):
    return math.ceil(math.log10(2 ** bits)) + 1


def print_torch_tensor(name, tensor, slice_obj=None, grad=False):
    typename = tensor.type().lower()
    show = lambda x: f"{x.item()}"
    if "float" in typename or "32" in typename:
        # IEEE float has 24 mantissa bits => 9 numbers
        show = lambda x: f"{x.item():.9}"
    elif "double" in typename or "64" in typename:
        # IEEE double has 53 mantissa => 17 numbers
        show = lambda x: f"{x.item():.17}"
    elif "half" in typename or "16" in typename:
        # IEEE half has 11 mantissa bits => 5 numbers
        show = lambda x: f"{x.item():.5}"
    data = tensor.data.flatten()
    data = data[slice_obj] if slice_obj else data
    data = [show(x) for x in data]
    print(f"{name} ({tensor.shape}):\n{data}")
    if grad:
        grad_data = tensor.grad.flatten()
        grad_data = grad_data[slice_obj] if slice_obj else grad_data
        grad_data = [show(x) for x in grad_data]
        print(f"grad of {name} ({tensor.grad.shape}):\n{grad_data}")


def size(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
