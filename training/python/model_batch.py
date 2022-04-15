from torch.nn import Module
from torch import nn
import torch

torch.manual_seed(0)


class Softmax(nn.Module):
    def forward(self, input):
        exp_x = torch.exp(input)
        y = exp_x / exp_x.sum(1).unsqueeze(1).expand_as(exp_x)
        return y


class Model(Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2)
        self.batch1 = nn.BatchNorm2d(16, momentum=0)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2)
        self.batch2 = nn.BatchNorm2d(32, momentum=0)
        self.fc1 = nn.Linear(1152, 128)
        self.sigm = nn.Sigmoid()
        self.fc2 = nn.Linear(128, 10)
        self.softmax = Softmax()

        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

        nn.init.zeros_(self.conv1.bias)
        nn.init.zeros_(self.conv2.bias)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        out = self.conv1(x)
        # print(out.shape)
        # out = self.batch1(out)
        # out = self.relu(out)
        out = self.conv2(out)
        # out = self.batch2(out)
        # out = self.relu(out)
        # print(out.shape)
        out = out.reshape(-1, 1152)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.softmax(out)
        return out
