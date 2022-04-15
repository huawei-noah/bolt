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

num_classes = 10
batch_size = 50
learning_rate = 0.01

curdir = "./weights/"


class Softmax(nn.Module):
    def forward(self, input):
        exp_x = torch.exp(input)
        y = exp_x / exp_x.sum(1).unsqueeze(1).expand_as(exp_x)
        return y


class NeuralNet(nn.Module):
    def __init__(self, num_classes):
        super(NeuralNet, self).__init__()

        torch.manual_seed(0)

        self.conv1 = nn.Conv1d(1, 16, kernel_size=5, stride=2, padding=2, bias=True)
        self.fc1 = nn.Linear(16 * 392, num_classes, bias=True)
        self.softmax = Softmax()

        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.uniform_(self.conv1.bias, 0, 1)
        nn.init.uniform_(self.fc1.bias, 0, 1)

    def forward(self, x):
        x = x.view(50, 1, -1)
        out = self.conv1(x)
        out = out.reshape(-1, self.fc1.in_features)
        out = self.fc1(out)
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


def saveWeights(index, model, dir):

    if not os.path.exists(dir):
        os.mkdir(dir)

    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.data.dim() == 4:
                for i in range(0, param.data.shape[0]):
                    with open(
                        dir + str(index) + "_" + name + "_" + str(i) + ".txt", "w"
                    ) as outfile:
                        for j in range(0, param.data.shape[1]):
                            np.savetxt(outfile, param.data[i, j])
            else:
                with open(dir + str(index) + "_" + name + ".txt", "w") as outfile:
                    np.savetxt(outfile, torch.squeeze(param.data))


def CrossEntropy(y, target):
    ones = torch.sparse.torch.eye(num_classes)
    t = ones.index_select(0, target).type(y.data.type())
    t = Variable(t)
    loss = (-t * torch.log(y)).sum() / y.size(0)
    return loss, y


def main():

    train_dataset = torchvision.datasets.MNIST(
        root="./data/mnist", train=True, transform=transforms.ToTensor(), download=True
    )

    test_dataset = torchvision.datasets.MNIST(
        root="./data/mnist", train=False, transform=transforms.ToTensor()
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=False
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False
    )

    model = NeuralNet(num_classes)

    predict(test_loader, model)

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    total_step = len(train_loader)

    if os.path.exists(curdir + "loss.txt"):
        os.remove(curdir + "loss.txt")

    timeTaken = 0

    for i, (images, labels) in enumerate(train_loader):

        start = time.time()

        outputs = model(images)
        if i < 1:
            saveWeights(i, model, "../../../../testAssets/test_cnn_layer/conv1d/")
        loss, lossInput = CrossEntropy(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        timeTaken += time.time() - start

        # if i % 100 == 0:
        #    with open(curdir + 'loss.txt', 'a') as outfile:
        #        print(loss.item(), file = outfile)

        if i % 100 == 0:
            print("Step [{:4d}/{}], Loss: {:.6f}".format(i, total_step, loss.item()))

    predict(test_loader, model)

    print("Time taken = {:.4f}".format(timeTaken))


if __name__ == "__main__":
    main()
