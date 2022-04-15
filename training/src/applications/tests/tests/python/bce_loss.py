import torch

# Simple unit
input = torch.rand(2, 3, requires_grad=True)
target = torch.rand(2, 3)
loss = torch.nn.BCELoss(reduction="none")
print("Input: ", input)
print("Target: ", target)
output = loss(input, target)
print("Loss: ", output)
output.sum().backward()
print("Gradient for input: ", input.grad)

# Corner cases
loss = torch.nn.BCELoss(reduction="none")
input = torch.tensor(
    [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.998, 0.002, 1.0], [0.999, 0.0001, 0.11]],
    requires_grad=True,
)
target = torch.tensor(
    [[0.0, 1.0, 0.99], [0.0, 1.0, 0.5], [0.0001, 0.997, 1.0], [0.999, 0.001, 0.00001]]
)
print("Input: ", input)
print("Target: ", target)
output = loss(input, target)
print("Loss: ", loss)
output.sum().backward()
print("Gradient for input: ", input.grad)
