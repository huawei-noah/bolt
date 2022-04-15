import os
import pickle
import numpy as np
import torch
import torch.onnx
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


def export_dataset(dataset, path):
    with open(path, "w") as outfile:
        outfile.write("{} {}\n".format(len(dataset), len(dataset[0][0])))
        for d in dataset:
            np.savetxt(outfile, d[0].cpu())
            np.savetxt(outfile, d[1].cpu())
            np.savetxt(outfile, d[2].cpu())
            outfile.write("{}\n".format(d[3].cpu().item()))


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


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return (torch.sum(pred_flat == labels_flat) / float(len(labels_flat))).item()


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


model = BertForSequenceClassification.from_pretrained(
    "../testAssets/bert/4l_384d_tf_ckpt", num_labels=2
)

torch.random.manual_seed(0)
torch.nn.init.uniform_(model.classifier.weight, -1, 1)
torch.nn.init.uniform_(model.classifier.bias, -1, 1)

print("hidden_size=", model.config.hidden_size)

train_path = "../testAssets/bert/dataset/train"
test_path = "../testAssets/bert/dataset/dev"

path = "../testAssets/bert/pretrained/"
save_weights(path, 0, model)


with open(test_path, "rb") as file:
    test_data = pickle.load(file)
with open(train_path, "rb") as file:
    train_data = pickle.load(file)

print("Training size: {}".format(len(train_data)))
print("Test size: {}".format(len(test_data)))

# train_data = train_data[1:256]
# test_data = test_data[1:32]

train_dataset = to_dataset(train_data)
test_dataset = to_dataset(test_data)

# export_dataset(test_dataset, "../testAssets/bert/dataset/test.txt")
# export_dataset(train_dataset, "../testAssets/bert/dataset/train.txt")

nepoch = 1
batch_size = 32
lr = 2e-5

train_dataloader = DataLoader(
    train_dataset,  # The training samples.
    sampler=SequentialSampler(train_dataset),  # Select batches randomly
    batch_size=batch_size,  # Trains with this batch size.
)

validation_dataloader = DataLoader(
    test_dataset,  # The validation samples.
    sampler=SequentialSampler(test_dataset),  # Pull out batches sequentially.
    batch_size=8,  # Evaluate with this batch size.
)

num_training_steps = len(train_dataloader) * nepoch

# optimizer = BertAdam(model.parameters(), lr=lr, e=eps, schedule='warmup_linear', warmup=0,
#                     t_total=num_training_steps)

model.eval()
total_eval_accuracy = 0
total_eval_loss = 0
cnt = 0
for batch in validation_dataloader:
    with torch.no_grad():
        logits = model(
            batch[0], token_type_ids=batch[1], attention_mask=batch[2], labels=None
        )
    total_eval_accuracy += flat_accuracy(logits, batch[3])
    cnt += 1
    print(total_eval_accuracy / (cnt * 8))
avg_eval_accuracy = total_eval_accuracy / len(validation_dataloader)
print("  Accuracy: {0:.2f}".format(avg_eval_accuracy))

# torch.onnx.export(torch_model,               # model being run
#                   batch,                         # model input (or a tuple for multiple inputs)
#                   "bert.onnx",   # where to save the model (can be a file or file-like object)
#                   export_params=True,        # store the trained parameter weights inside the model file
#                   opset_version=10,          # the ONNX version to export the model to
#                   do_constant_folding=True,  # whether to execute constant folding for optimization
#                   input_names = ['input_ids', 'token_type_ids', 'attention_mask', 'labels'],   # the model's input names
#                   output_names = ['loss'], # the model's output names
#                   dynamic_axes={'input_ids' : {0 : 'batch_size'},    # variable lenght axes
#                                 'token_type_ids' : {0 : 'batch_size'},
#                                 'attention_mask' : {0 : 'batch_size'},
#                                 'labels' : {0 : 'batch_size'}})

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

for epoch in range(nepoch):
    print("Epoch", epoch)
    start = time.time()
    model.train()
    total_loss = 0
    last_loss = 0
    for step, batch in enumerate(train_dataloader):
        # Progress update every 40 batches.
        if step % 40 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            elapsed = time.time() - start
            # Report progress.
            print(
                "  Batch {:>5,}  of  {:>5,}. Loss {:}  Elapsed: {:}.".format(
                    step, len(train_dataloader), last_loss, elapsed
                )
            )
        model.zero_grad()
        loss = model(
            batch[0], token_type_ids=batch[1], attention_mask=batch[2], labels=batch[3]
        )
        last_loss = loss.item()
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    avg_train_loss = total_loss / len(train_dataloader)
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Epoch time: {0:.2f}".format(time.time() - start))
    model.eval()
    total_eval_accuracy = 0
    total_eval_loss = 0
    for batch in validation_dataloader:
        with torch.no_grad():
            logits = model(
                batch[0], token_type_ids=batch[1], attention_mask=batch[2], labels=None
            )
        total_eval_accuracy += flat_accuracy(logits, batch[3])
    avg_eval_accuracy = total_eval_accuracy / len(validation_dataloader)
    print("  Accuracy: {0:.2f}".format(avg_eval_accuracy))
    # Calculate the average loss over all of the batches.
    # for batch in validation_dataloader:
    #     with torch.no_grad():
    #         loss = model(batch[0], token_type_ids=batch[1], attention_mask=batch[2], labels=batch[3])
    #     total_eval_loss += loss.item()
    # avg_eval_loss = total_eval_loss / len(validation_dataloader)
    # print("  Validation Loss: {0:.2f}".format(avg_eval_loss))

    with open("accuracy_log_1.txt", "a") as al:
        al.write("{0} {1}\n".format(epoch, avg_eval_accuracy))
    save_weights(path, epoch + 1, model)
