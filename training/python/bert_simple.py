import os
import pickle
import numpy as np
import torch
import math, time
from modeling import (
    load_tf_weights_in_bert,
    BertModel,
    BertConfig,
    BertForSequenceClassification,
)
from optimization import BertAdam
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler


def to_dataset(data):
    input_ids = []
    segment_ids = []
    input_mask = []
    label_id = []
    for d in data:
        input_ids.append(torch.LongTensor([d["input_ids"]]))
        segment_ids.append(torch.LongTensor([d["segment_ids"]]))
        input_mask.append(torch.LongTensor([d["input_mask"]]))
        label_id.append(torch.LongTensor([d["label_id"]]))
    input_ids = torch.cat(input_ids, dim=0)
    segment_ids = torch.cat(segment_ids, dim=0)
    input_mask = torch.cat(input_mask, dim=0)
    label_id = torch.tensor(label_id)
    return TensorDataset(input_ids, segment_ids, input_mask, label_id)


model = BertForSequenceClassification.from_pretrained(
    "../testAssets/bert/4l_384d_tf_ckpt", num_labels=2
)

torch.nn.init.ones_(model.classifier.weight)
torch.nn.init.zeros_(model.classifier.bias)

test_path = "../testAssets/bert/dataset/dev"

with open(test_path, "rb") as file:
    test_data = pickle.load(file)

print("Test size: {}".format(len(test_data)))

batch_size = 16
test_data = test_data[0:batch_size]

test_dataset = to_dataset(test_data)

dataloader = DataLoader(
    test_dataset,  # The validation samples.
    sampler=SequentialSampler(test_dataset),  # Pull out batches sequentially.
    batch_size=batch_size,  # Evaluate with this batch size.
)

batch = [b for b in dataloader][0]

model.train()

loss = model(
    batch[0], token_type_ids=batch[1], attention_mask=batch[2], labels=batch[3]
)
print("loss=", loss.item())
model.zero_grad()
loss.backward()
