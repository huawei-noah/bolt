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

torch.random.manual_seed(0)

bs = 2
hidden = 2
input_size = 3
rnn = torch.nn.LSTMCell(input_size, hidden)  # (input_size, hidden_size)
# [-0.0053, 0.3793, -0.5820, -0.5204, -0.2723, 0.1896, -0.0140, 0.5607, -0.0628,0.1871, -0.2137, -0.1390, -0.6755, -0.4683, -0.2915,0.0262, 0.2795, 0.4243, -0.4794, -0.3079, 0.2568, 0.5872, -0.1455,  0.5291]
print(rnn.weight_ih)
# [-0.1140, 0.0748, 0.6403, -0.6560, -0.4452, -0.1790, -0.2756, 0.6109, -0.4583, -0.3255, -0.4940, -0.6622, -0.4128, 0.6078,0.3155, 0.3427]
print(rnn.weight_hh)
# [ 0.0372, -0.3625,  0.1196, -0.6602, -0.5109, -0.3645,  0.4461,  0.4146]
print(rnn.bias_ih)
# [-0.3136, -0.0255,  0.4522,  0.7030,  0.2806,  0.0955,  0.4741, -0.4163]
print(rnn.bias_hh)
input = torch.randn(bs, input_size)  # (time_steps, batch, input_size)
hx = torch.randn(bs, hidden)  # (batch, hidden_size)
cx = torch.randn(bs, hidden)

# [-0.0209, -0.7185,  0.5186, -1.3125,  0.1920,  0.5428 ]
print(input)
# [0.2734, -0.9181, -0.0404,  0.2881]
print(cx)
# [-2.2188,  0.2590, -1.0297, -0.5008]
print(hx)

hx, cx = rnn(input, (hx, cx))
# [0.4616, -0.5580, 0.2754,  0.4600]
print(cx)
# [0.3941, -0.2222, 0.2289,  0.1148]
print(hx)
hx, cx = rnn(input, (hx, cx))
# [ 0.1913, -0.4290, 0.2704,  0.0704]
print(cx)
# [ 0.1377, -0.2434, 0.2198,  0.0282]
print(hx)
