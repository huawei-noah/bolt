from torch.nn import Module
from torch import nn
import torch.nn.functional as F
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
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.fc2 = nn.Linear(512, 10)
        self.softmax = Softmax()

        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        # nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

        nn.init.zeros_(self.conv1.bias)
        nn.init.zeros_(self.conv2.bias)
        # nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        self.conv1Input = x
        out = self.conv1(x)
        # out = self.relu(out)
        # out = self.maxpool1(out)
        out = self.conv2(out)
        # out = self.relu2(out)
        out = out.reshape(-1, 512)
        # out = self.fc1(out)
        # out = self.sigm(out)
        # out = self.droput(out)
        out = self.fc2(out)
        out = self.softmax(out)
        return out
