import torch

torch.manual_seed(0)
torch.set_printoptions(precision=8)

# Deterministic unit
relu = torch.nn.LeakyReLU(0.05)
input = torch.randn(2, 3, 4, 5, requires_grad=True)
print("Input: ", input)

## Forward
output = relu(input)
print("Output: ", output)

## Backward
output.sum().backward()
print("Input gradient: ", input.grad)
