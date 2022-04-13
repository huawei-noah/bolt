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
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import numpy as np

import torch.nn.functional as F

torch.manual_seed(0)


class Softmax(nn.Module):
    def forward(self, input):
        exp_x = torch.exp(input)
        y = exp_x / exp_x.sum(1).unsqueeze(1).expand_as(exp_x)
        return y


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)

        self.fc1 = nn.Linear(6 * 28 * 28, 84)
        self.fc2 = nn.Linear(84, 10)
        self.soft = Softmax()
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

        nn.init.zeros_(self.conv1.bias)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        x = self.conv1(x)
        x = x.reshape(-1, 6 * 28 * 28)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.soft(x)
        return x


criterion = nn.CrossEntropyLoss()
learning_rate = 1e-2
batch_size = 50
_epoch = 1
curdir = "./weights/"


transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

trainset = torchvision.datasets.CIFAR10(
    root="./data/", train=True, download=True, transform=transform
)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size)

testset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform
)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size)

classes = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

import matplotlib.pyplot as plt
import os


def saveWeights(index, model):
    if not os.path.exists(curdir):
        os.mkdir(curdir)

    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.data.dim() == 4:
                for i in range(0, param.data.shape[0]):
                    with open(
                        curdir + str(index) + "_" + name + "_" + str(i) + ".txt", "w"
                    ) as outfile:
                        for j in range(0, param.data.shape[1]):
                            np.savetxt(outfile, param.data[i, j])
            else:
                with open(curdir + str(index) + "_" + name + ".txt", "w") as outfile:
                    np.savetxt(outfile, np.transpose(param.data))


criterion = nn.CrossEntropyLoss()
model = Model()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
for epoch in range(_epoch):  # loop over the dataset multiple times

    running_loss = 0.0
    model.train(True)
    class_correct = 0
    class_total = 0
    for data in testloader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(batch_size):
            label = labels[i]
            class_correct += c[i].item()
            class_total += 1
    print("Accuracy: {:.4f}%".format(100 * class_correct / class_total))
    for i, data in enumerate(trainloader, 0):
        if i < 1:
            saveWeights(i, model)
        # get the inputs
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 100 == 99:  # print every 2000 mini-batches
            print("{:.6f}".format(running_loss / 100))
            running_loss = 0.0
    model.train(False)
    class_correct = 0
    class_total = 0
    for data in testloader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(batch_size):
            label = labels[i]
            class_correct += c[i].item()
            class_total += 1
    print("Accuracy: {:.4f}%".format(100 * class_correct / class_total))
print("Finished Training")
