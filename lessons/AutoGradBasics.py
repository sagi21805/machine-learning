import torch

# x = torch.randn(3, requires_grad = True)
# y = x + 2
# print(x)
# z = y*y+2
# print(z)
# # z = z.mean()                                                  #gives the average value of the tensor
# v = torch.tensor([0.1, 1.0, 0.001], dtype = torch.float32)
# print(z.backward(v))                                            # dz/dx
# print(x.grad)

# x.requires_grad_(False)                                         #makes the requires_grad False
# x.detach()                                                      #makes the requires_grad False

#training example:

weights = torch.ones(4, requires_grad = True)

for epoch in range(3):
    model_output = (weights * 3).sum()
    print(model_output)
    print(model_output.backward())
    print(weights.grad)
    weights.grad.zero_()