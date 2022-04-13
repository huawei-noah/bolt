# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#!/usr/bin/env python3

import torch
import torch.nn as nn
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import os
import sys
import getopt
import time


class Softmax(nn.Module):
    def forward(self, input):
        exp_x = torch.exp(input)
        y = exp_x / exp_x.sum(1).unsqueeze(1).expand_as(exp_x)
        return y


class NeuralNet(nn.Module):
    def __init__(self, num_classes, test_type):
        super(NeuralNet, self).__init__()
        self.test_type = test_type

        torch.manual_seed(0)

        self.conv1 = nn.Conv2d(1, 1, kernel_size=5, stride=1)
        if self.test_type == "const_sym_pad":
            self.pad1 = nn.ConstantPad2d(20, 0.0)
        elif self.test_type == "ref_sym_pad":
            self.pad1 = nn.ReflectionPad2d(20)
        elif self.test_type == "rep_sym_pad":
            self.pad1 = nn.ReplicationPad2d(20)
        elif self.test_type == "const_asym_pad":
            self.pad1 = nn.ConstantPad2d([10, 5, 20, 15], 0.0)
        elif self.test_type == "ref_asym_pad":
            self.pad1 = nn.ReflectionPad2d([10, 5, 20, 15])
        elif self.test_type == "rep_asym_pad":
            self.pad1 = nn.ReplicationPad2d([10, 5, 20, 15])
        else:
            print("ERROR: unknown test type")

        if self.test_type == "const_sym_pad":
            self.pad2 = nn.ConstantPad2d(20, 0.0)
        elif self.test_type == "ref_sym_pad":
            self.pad2 = nn.ReflectionPad2d(20)
        elif self.test_type == "rep_sym_pad":
            self.pad2 = nn.ReplicationPad2d(20)
        elif self.test_type == "const_asym_pad":
            self.pad2 = nn.ConstantPad2d([10, 5, 20, 15], 0.0)
        elif self.test_type == "ref_asym_pad":
            self.pad2 = nn.ReflectionPad2d([10, 5, 20, 15])
        elif self.test_type == "rep_asym_pad":
            self.pad2 = nn.ReplicationPad2d([10, 5, 20, 15])
        else:
            print("ERROR: unknown test type")
        if (
            self.test_type == "const_sym_pad"
            or self.test_type == "ref_sym_pad"
            or self.test_type == "rep_sym_pad"
        ):
            self.fc1 = nn.Linear(10816, 10)
        elif (
            self.test_type == "const_asym_pad"
            or self.test_type == "ref_asym_pad"
            or self.test_type == "rep_asym_pad"
        ):
            self.fc1 = nn.Linear(5076, 10)
        else:
            print("ERROR: unknown test type")
        self.softmax = Softmax()

        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.conv1.bias)
        nn.init.zeros_(self.fc1.bias)

    def forward(self, x):
        out = self.conv1(x)
        out = self.pad1(out)
        out = self.pad2(out)
        if (
            self.test_type == "const_sym_pad"
            or self.test_type == "ref_sym_pad"
            or self.test_type == "rep_sym_pad"
        ):
            out = out.reshape(-1, 10816)
        elif (
            self.test_type == "const_asym_pad"
            or self.test_type == "ref_asym_pad"
            or self.test_type == "rep_asym_pad"
        ):
            out = out.reshape(-1, 5076)
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


def saveWeights(backup_dir, index, model):
    if not os.path.exists(backup_dir):
        os.mkdir(backup_dir)

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name + " size: ", param.size())
            if param.data.dim() == 4:
                for i in range(0, param.data.shape[0]):
                    with open(
                        backup_dir + str(index) + "_" + name + "_" + str(i) + ".txt",
                        "w",
                    ) as outfile:
                        for j in range(0, param.data.shape[1]):
                            np.savetxt(outfile, param.data[i, j])
            else:
                with open(
                    backup_dir + str(index) + "_" + name + ".txt", "w"
                ) as outfile:
                    np.savetxt(outfile, param.data)


def CrossEntropy(y, target, num_classes):
    ones = torch.sparse.torch.eye(num_classes)
    t = ones.index_select(0, target).type(y.data.type())
    t = Variable(t)
    loss = (-t * torch.log(y)).sum() / y.size(0)
    return loss, y


def show_reference():
    print("./PaddingLayerTestTopology.py ", end="")
    print("<--test <test type>> ", end="")
    print("[--save-data] ", end="")
    print("[--backup <path where should be saved weights and losses>] ")
    print(", where test type can be 'const_sym_pad'")


def main(command_line_options):
    save_data = False
    test_type = None
    num_classes = 10
    learning_rate = 0.01
    batch_size = 50
    backup_dir = "./data"
    try:
        options, _ = getopt.getopt(
            command_line_options, "", ["help", "save-data", "test=", "backup="]
        )
        for option, argument in options:
            if option == "--help":
                show_reference()
                return 0
            if option == "--save-data":
                save_data = True
            elif option == "--test":
                test_type = argument
            elif option == "--backup":
                backup_dir = argument
            else:
                print("ERROR: unknown option")
                return 1

    except getopt.GetoptError:
        print("ERROR: wrong command line options")
        return 2

    if test_type is None:
        print("ERROR: test type should be defined with option --test")
        return 1

    train_dataset = torchvision.datasets.MNIST(
        root="./data/mnist",
        train=True,
        transform=transforms.ToTensor(),
        download=True,
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

    model = NeuralNet(num_classes, test_type)

    predict(test_loader, model)
    if save_data:
        saveWeights(backup_dir, 0, model)

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    total_step = len(train_loader)
    if os.path.exists(backup_dir + "loss.txt"):
        os.remove(backup_dir + "loss.txt")

    timeTaken = 0
    for i, (images, labels) in enumerate(train_loader):
        start = time.time()
        outputs = model(images)
        loss, lossInput = CrossEntropy(outputs, labels, num_classes)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        timeTaken += time.time() - start

        if i % 100 == 0 and save_data == True:
            with open(backup_dir + "loss.txt", "a") as outfile:
                print(loss.item(), file=outfile)

        if i % 100 == 0:
            print("Step [{:4d}/{}], Loss: {:.6f}".format(i, total_step, loss.item()))

    predict(test_loader, model)
    print("Time taken = {:.4f}".format(timeTaken))


if __name__ == "__main__":
    main(sys.argv[1:])
