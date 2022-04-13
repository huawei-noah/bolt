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
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import os
import time
import sys

num_classes = 10
num_channels = 3
batch_size = 50
learning_rate = 0.01

curdir = "./weights/"

torch.manual_seed(0)


class Softmax(nn.Module):
    def forward(self, input):
        exp_x = torch.exp(input)
        y = exp_x / exp_x.sum(1).unsqueeze(1).expand_as(exp_x)
        return y


class LeNet(nn.Module):
    def __init__(self, num_class=10, num_channel=1, include_top=True):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(num_channel, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.include_top = include_top
        if self.include_top:
            self.flatten = nn.Flatten()
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, num_class)

        self.softmax = Softmax()

        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.zeros_(self.conv1.bias)
        nn.init.zeros_(self.conv2.bias)

        if self.include_top:
            nn.init.xavier_uniform_(self.fc1.weight)
            nn.init.xavier_uniform_(self.fc2.weight)
            nn.init.xavier_uniform_(self.fc3.weight)
            nn.init.zeros_(self.fc1.bias)
            nn.init.zeros_(self.fc2.bias)
            nn.init.zeros_(self.fc3.bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        if self.include_top:
            x = self.flatten(x)
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.fc3(x)

        out = x.view(x.size(0), num_classes)
        out = self.softmax(out)
        return out


def predict(test_loader, model):
    correct = 0
    total = 0
    # ~ with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print(
        "Accuracy of the network on the 10000 test images: {:.2f} %".format(
            100 * correct / total
        )
    )


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


def CrossEntropy(y, target):
    ones = torch.sparse.torch.eye(num_classes)
    t = ones.index_select(0, target).type(y.data.type())
    t = Variable(t)
    loss = (-t * torch.log(y)).sum() / y.size(0)
    return loss, y


def main():

    train_dataset = torchvision.datasets.CIFAR10(
        root="./data/", train=True, transform=transforms.ToTensor(), download=True
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root="./data/", train=False, transform=transforms.ToTensor()
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=False
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False
    )

    model = LeNet(num_classes, num_channels)

    predict(test_loader, model)

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    total_step = len(train_loader)

    if os.path.exists(curdir + "loss.txt"):
        os.remove(curdir + "loss.txt")

    timeTaken = 0

    for i, (images, labels) in enumerate(train_loader):

        start = time.time()

        outputs = model(images)
        # print(outputs)
        #        if i < 1:
        #            saveWeights(i, model)

        loss, lossInput = CrossEntropy(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        timeTaken += time.time() - start

        # if i % 100 == 0:
        #  with open(curdir + 'loss.txt', 'a') as outfile:
        #    print(loss.item(), file = outfile)

        if i % 100 == 0:
            print("Step [{:4d}/{}], Loss: {:.8f}".format(i, total_step, loss.item()))

    predict(test_loader, model)

    print("Time taken = {:.4f}".format(timeTaken))


if __name__ == "__main__":
    main()
