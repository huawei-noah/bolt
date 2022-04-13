# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


import torch


def test(x, conv):
    print("==== TEST START ====")
    torch.nn.init.ones_(conv.weight)
    torch.nn.init.ones_(conv.bias)
    y = x
    if y.grad is not None:
        torch.nn.init.zeros_(y.grad)
    z = conv(y)
    print(conv)
    print("conv", z)
    g = torch.ones_like(z)
    z.backward(g)
    print("grad", y.grad)
    print("w.grad", conv.weight.grad)
    print("b.grad", conv.bias.grad)
    print("==== TEST END  ====")


x = torch.tensor(
    [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]],
    requires_grad=True,
)

conv = torch.nn.Conv1d(
    in_channels=2, out_channels=2, kernel_size=3, padding=0, dilation=1, stride=1
)
test(x, conv)
print(conv.weight.shape)

conv = torch.nn.Conv1d(
    in_channels=2,
    out_channels=2,
    kernel_size=3,
    padding=0,
    dilation=1,
    stride=1,
    groups=2,
)
test(x, conv)
print(conv.weight.shape)

conv = torch.nn.Conv1d(
    in_channels=2, out_channels=2, kernel_size=3, padding=2, dilation=1, stride=1
)
test(x, conv)

conv = torch.nn.Conv1d(
    in_channels=2, out_channels=2, kernel_size=3, padding=2, dilation=1, stride=2
)
test(x, conv)

conv = torch.nn.Conv1d(
    in_channels=2, out_channels=1, kernel_size=3, padding=2, dilation=1, stride=3
)
test(x, conv)

conv = torch.nn.Conv1d(
    in_channels=2, out_channels=1, kernel_size=3, padding=2, dilation=1, stride=5
)
test(x, conv)

conv = torch.nn.Conv1d(
    in_channels=2,
    out_channels=4,
    kernel_size=3,
    padding=2,
    dilation=1,
    stride=5,
    groups=2,
)
test(x, conv)
