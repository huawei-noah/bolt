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


import torch

torch.manual_seed(0)
torch.set_printoptions(precision=8)

# Seq1 unit
rnn = torch.nn.GRU(4, 3, 1, batch_first=True)
input = torch.randn(2, 1, 4, requires_grad=True)
h0 = torch.zeros([1, 2, 3], requires_grad=True)
rnn.weight_ih_l0 = torch.nn.Parameter(torch.ones([9, 4]))
rnn.weight_hh_l0 = torch.nn.Parameter(torch.ones([9, 3]))
rnn.bias_ih_l0 = torch.nn.Parameter(torch.ones([9]))
rnn.bias_hh_l0 = torch.nn.Parameter(torch.ones([9]))
print("Input: ", input)
print("Hidden: ", h0)

## Forward
output, hn = rnn(input, h0)
print("Output: ", output)
print("New hidden: ", hn)

## Backward
hn.sum().backward()
print("Input gradient: ", input.grad)
print("Hidden gradient: ", h0.grad)
print("IH weights gradient: ", rnn.weight_ih_l0.grad)
print("HH weights gradient: ", rnn.weight_hh_l0.grad)
print("IH biases gradient: ", rnn.bias_ih_l0.grad)
print("HH biases gradient: ", rnn.bias_hh_l0.grad)

# Seq3 unit
rnn = torch.nn.GRU(5, 4, 1, batch_first=True)
input = torch.randn(2, 3, 5, requires_grad=True)
h0 = torch.zeros([1, 2, 4], requires_grad=True)
rnn.weight_ih_l0 = torch.nn.Parameter(torch.ones([12, 5]))
rnn.weight_hh_l0 = torch.nn.Parameter(torch.ones([12, 4]))
rnn.bias_ih_l0 = torch.nn.Parameter(torch.ones([12]))
rnn.bias_hh_l0 = torch.nn.Parameter(torch.ones([12]))
print("Input: ", input)
print("Hidden: ", h0)

## Forward
output, hn = rnn(input, h0)
print("Output: ", output)
print("New hidden: ", hn)

## Backward
output.sum().backward()
print("Input gradient: ", input.grad)
print("Hidden gradient: ", h0.grad)
print("IH weights gradient: ", rnn.weight_ih_l0.grad)
print("HH weights gradient: ", rnn.weight_hh_l0.grad)
print("IH biases gradient: ", rnn.bias_ih_l0.grad)
print("HH biases gradient: ", rnn.bias_hh_l0.grad)

# Seq7 unit
rnn = torch.nn.GRU(4, 5, 1, batch_first=True)
input = torch.randn(2, 7, 4, requires_grad=True)
h0 = torch.zeros([1, 2, 5], requires_grad=True)
print("IH weights: ", rnn.weight_ih_l0)
print("HH weights: ", rnn.weight_hh_l0)
print("IH biases: ", rnn.bias_ih_l0)
print("HH biases: ", rnn.bias_hh_l0)
print("Input: ", input)
print("Hidden: ", h0)

## Forward
output, hn = rnn(input, h0)
print("Output: ", output)
print("New hidden: ", hn)

## Backward
output.sum().backward()
print("Input gradient: ", input.grad)
print("Hidden gradient: ", h0.grad)
print("IH weights gradient: ", rnn.weight_ih_l0.grad)
print("HH weights gradient: ", rnn.weight_hh_l0.grad)
print("IH biases gradient: ", rnn.bias_ih_l0.grad)
print("HH biases gradient: ", rnn.bias_hh_l0.grad)

# Seq5 external state unit
rnn = torch.nn.GRU(4, 3, 1, batch_first=True)
input = torch.randn(3, 5, 4, requires_grad=True)
h0 = torch.randn(1, 3, 3, requires_grad=True)
print("IH weights: ", rnn.weight_ih_l0)
print("HH weights: ", rnn.weight_hh_l0)
print("IH biases: ", rnn.bias_ih_l0)
print("HH biases: ", rnn.bias_hh_l0)
print("Input: ", input)
print("Hidden: ", h0)

## Forward
output, hn = rnn(input, h0)
print("Output: ", output)
print("New hidden: ", hn)

## Backward
output.sum().backward()
print("Input gradient: ", input.grad)
print("Hidden gradient: ", h0.grad)
print("IH weights gradient: ", rnn.weight_ih_l0.grad)
print("HH weights gradient: ", rnn.weight_hh_l0.grad)
print("IH biases gradient: ", rnn.bias_ih_l0.grad)
print("HH biases gradient: ", rnn.bias_hh_l0.grad)
