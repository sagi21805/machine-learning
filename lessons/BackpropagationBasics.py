import torch

x = torch.tensor(1.0)
y = torch.tensor(2.0)

weight = torch.tensor(1.0, requires_grad = True)

#forward pass and calculate the loss

y_hat = weight * x
loss = (y_hat - y)**2 #loss is always calculated like this

#backward pass

loss.backward() #calculae the gradient
print(weight.grad)

### update wights
### next forward and backward 