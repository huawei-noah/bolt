import torch

torch.manual_seed(0)
torch.set_printoptions(precision=8)

# Ones hidden unit
rnn = torch.nn.GRU(9, 5, 1, batch_first=True)
input = torch.randn(3, 1, 9, requires_grad=True)
h0 = torch.zeros([1, 3, 5], requires_grad=True)
rnn.weight_ih_l0 = torch.nn.Parameter(torch.ones([15, 9]))
rnn.weight_hh_l0 = torch.nn.Parameter(torch.ones([15, 5]))
rnn.bias_ih_l0 = torch.nn.Parameter(torch.ones([15]))
rnn.bias_hh_l0 = torch.nn.Parameter(torch.ones([15]))
print("Input: ", input)
print("Hidden: ", h0)

## Forward
output, hn = rnn(input, h0)
print("New hidden: ", hn)

## Backward
hn.sum().backward()
print("Input gradient: ", input.grad)
print("Hidden gradient: ", h0.grad)

# Ones hidden unit
rnn = torch.nn.GRU(4, 3, 1, batch_first=True)
input = torch.randn(2, 1, 4, requires_grad=True)
h0 = torch.randn(1, 2, 3, requires_grad=True)
rnn.weight_ih_l0 = torch.nn.Parameter(torch.ones([9, 4]))
rnn.weight_hh_l0 = torch.nn.Parameter(torch.ones([9, 3]))
rnn.bias_ih_l0 = torch.nn.Parameter(torch.ones([9]))
rnn.bias_hh_l0 = torch.nn.Parameter(torch.ones([9]))
print("Input: ", input)
print("Hidden: ", h0)

## Forward
output, hn = rnn(input, h0)
print("New hidden: ", hn)

## Backward
hn.sum().backward()
print("Input gradient: ", input.grad)
print("Hidden gradient: ", h0.grad)

# Random weights unit
rnn = torch.nn.GRU(7, 4, 1, batch_first=True)
input = torch.randn(3, 1, 7, requires_grad=True)
h0 = torch.randn(1, 3, 4, requires_grad=True)
print("IH weights: ", rnn.weight_ih_l0)
print("HH weights: ", rnn.weight_hh_l0)
print("IH bias: ", rnn.bias_ih_l0)
print("HH bias: ", rnn.bias_hh_l0)
print("Input: ", input)
print("Hidden: ", h0)

## Forward
output, hn = rnn(input, h0)
print("New hidden: ", hn)

## Backward
hn.sum().backward()
print("Input gradient: ", input.grad)
print("Hidden gradient: ", h0.grad)
print("IH weights gradient: ", rnn.weight_ih_l0.grad)
print("HH weights gradient: ", rnn.weight_hh_l0.grad)
print("IH biases gradient: ", rnn.bias_ih_l0.grad)
print("HH biases gradient: ", rnn.bias_hh_l0.grad)
