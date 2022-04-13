# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


from pyraul.tools.save_weights import save_weights
from pyraul.tools.dumping import gen_cpp_dtVec
import torchvision.models as models
import numpy as np
import torch


def simple_test():
    """Save weights for TestDataLoader.BinarySimpleDataLoad"""
    fd = open("weights/binary_dataloader_simple_test.bin", "wb")
    torch.tensor(
        np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), dtype=torch.float32
    ).numpy().tofile(fd)


def test_save_one_bias():
    """Save bias for TestDataLoader.BinaryBiasDataLoad"""
    alexnet = models.alexnet(True)
    bias = alexnet.features[3].state_dict()["bias"]

    print("Bias shape:", bias.shape)
    print(gen_cpp_dtVec(bias, "golden_bias_values"))

    fd = open("weights/binary_dataloader_bias_test.bin", "wb")
    bias.numpy().tofile(fd)


def test_save_one_weight():
    """Save convolution weights for TestDataLoader.BinaryWeightsLoad"""
    resnet = models.resnet18(True)
    convolution = resnet.layer3[1].conv1.state_dict()
    data = convolution["weight"].flatten()

    print("Weight shape:", convolution["weight"].shape)
    print(gen_cpp_dtVec(data[0 :: data.size()[0] // 100], "golden_biases_values"))

    fd = open("weights/binary_dataloader_weight_test.bin", "wb")
    convolution["weight"].numpy().tofile(fd)


simple_test()
test_save_one_bias()
print("------------------------------------------")
test_save_one_weight()
