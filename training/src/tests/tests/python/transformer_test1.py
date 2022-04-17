# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import os
from transformer import (
    make_model,
    Batch,
    LabelSmoothing,
    NoamOpt,
    run_epoch,
    greedy_decode,
    SimpleLossCompute,
)


def save_weights(path, index, model):
    if not os.path.exists(path):
        os.mkdir(path)

    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.data.dim() == 4:
                for i in range(0, param.data.shape[0]):
                    with open(
                        path + str(index) + "_" + name + "_" + str(i) + ".txt", "w"
                    ) as outfile:
                        for j in range(0, param.data.shape[1]):
                            np.savetxt(outfile, param.data[i, j].cpu())
            else:
                with open(path + str(index) + "_" + name + ".txt", "w") as outfile:
                    np.savetxt(outfile, param.data.cpu())


def data_gen(V, batch, nbatches):
    "Generate random data for a src-tgt copy task."
    for i in range(nbatches):
        data = torch.from_numpy(np.random.randint(1, V, size=(batch, 10)))
        # print(data)
        data = data.type(torch.LongTensor)
        data[:, 0] = 1
        src = Variable(data, requires_grad=False)
        tgt = Variable(data, requires_grad=False)
        yield Batch(src, tgt, 0)


def data_gen2(input, batch, nbatches):
    "Generate random data for a src-tgt copy task."
    for i in range(nbatches):
        data = torch.from_numpy(input[i * batch : (i + 1) * batch])
        data = data.type(torch.LongTensor)
        data[:, 0] = 1
        src = Variable(data, requires_grad=False)
        tgt = Variable(data, requires_grad=False)
        yield Batch(src, tgt, 0)


class TestOpt:
    "Optim wrapper that implements rate."

    def __init__(self, optimizer):
        self.optimizer = optimizer

    def step(self):
        self.optimizer.step()


torch.set_printoptions(precision=10)
# Train the simple copy task.
V = 11
MODEL_SIZE = 64
Heads = 4
LENGTH = 4
criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)

np.random.seed(0)
torch.manual_seed(0)
model = make_model(V, V, N=LENGTH, h=Heads, d_model=MODEL_SIZE, dropout=0.0)

for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

P = [x[0] for x in model.named_parameters()]

model_opt = TestOpt(torch.optim.SGD(model.parameters(), lr=0.002))

nepoch = 20  # 10
nepoch2 = 10
batch_size = 30
nbatches = 20

input = []
np.random.seed(0)
input = np.random.randint(1, V, size=(batch_size * nbatches * (nepoch + nepoch2), 10))

path = "../../../../testAssets/transformer/{}_{}_{}__{}/".format(
    MODEL_SIZE, Heads, LENGTH, nepoch
)

for epoch in range(nepoch):
    model.train()
    tl = run_epoch(
        data_gen2(
            input[epoch * nbatches * batch_size : (epoch + 1) * nbatches * batch_size],
            batch_size,
            nbatches,
        ),
        model,
        SimpleLossCompute(model.generator, criterion, model_opt),
    )
    print("Epoch", epoch, ":", tl)

save_weights(path, nepoch, model)
with open(path + "input.txt", "w") as outfile:
    np.savetxt(outfile, input)
print("Fine-tuning")

for epoch in range(nepoch2):
    model.train()
    tl = run_epoch(
        data_gen2(
            input[
                (epoch + nepoch)
                * nbatches
                * batch_size : (epoch + nepoch + 1)
                * nbatches
                * batch_size
            ],
            batch_size,
            nbatches,
        ),
        model,
        SimpleLossCompute(model.generator, criterion, model_opt),
    )
    print("Epoch", epoch + nepoch, ":", tl)

model.eval()
src = Variable(torch.LongTensor([[i + 1 for i in range(V - 1)]]))
src_mask = Variable(torch.ones(1, 1, V - 1))
print("before decode")
print(greedy_decode(model, src, src_mask, max_len=V - 1, start_symbol=1))
