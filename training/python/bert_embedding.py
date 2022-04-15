import os
import pickle
import numpy as np
import torch
import math, time
from modeling import BertConfig, BertEmbeddings


def to_tensor(data, key, type=torch.long):
    shape = np.array(data[0][key]).shape

    if len(shape) == 0:
        res = torch.zeros(len(data), 1, dtype=torch.long)
        for i in range(len(data)):
            res[i, :] = data[i][key]
    else:
        res = torch.zeros(len(data), shape[-1], dtype=torch.long)
        for i in range(len(data)):
            res[i, :] = torch.LongTensor(data[i][key]).view(-1)

    return res


model = BertEmbeddings(BertConfig("../testAssets/bert/4l_384d_tf_ckpt/config.json"))

test_path = "../testAssets/bert/dataset/dev"

with open(test_path, "rb") as file:
    data = pickle.load(file)

dd = data[1:2]

input_ids = to_tensor(dd, "input_ids")
segment_ids = to_tensor(dd, "segment_ids")

model.forward(input_ids, segment_ids)
