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
from transformer import (
    make_model,
    Batch,
    LabelSmoothing,
    NoamOpt,
    run_epoch,
    greedy_decode,
    SimpleLossCompute,
)


def data_gen(V, batch, nbatches):
    "Generate random data for a src-tgt copy task."
    for i in range(nbatches):
        data = torch.from_numpy(np.random.randint(1, V, size=(batch, 10)))
        print(data)
        data = data.type(torch.LongTensor)
        data[:, 0] = 1
        src = Variable(data, requires_grad=False)
        tgt = Variable(data, requires_grad=False)
        yield Batch(src, tgt, 0)


# Train the simple copy task.
V = 11
MODEL_SIZE = 10
Heads = 2
criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
model = make_model(V, V, N=2, h=Heads, d_model=MODEL_SIZE, dropout=0.0)

for p in model.parameters():
    nn.init.ones_(p)

model_opt = NoamOpt(
    model.src_embed[0].d_model,
    1,
    400,
    torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9),
)

nepoch = 5  # 10
batch_size = 2  # 30
nbatches = 2  # 20
nbatches_eval = 5

np.random.seed(0)

for epoch in range(nepoch):
    model.train()
    run_epoch(
        data_gen(V, batch_size, nbatches),
        model,
        SimpleLossCompute(model.generator, criterion, model_opt),
    )
    model.eval()
    print(
        run_epoch(
            data_gen(V, batch_size, nbatches_eval),
            model,
            SimpleLossCompute(model.generator, criterion, None),
        )
    )

model.eval()
src = Variable(torch.LongTensor([[i + 1 for i in range(V - 1)]]))
src_mask = Variable(torch.ones(1, 1, V - 1))
print("before decode")
print(greedy_decode(model, src, src_mask, max_len=V - 1, start_symbol=1))
