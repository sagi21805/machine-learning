import torch
import numpy as np

a: int = 5
b: int = 3
x = torch.rand(a, b)                         #creates a 2d tensor with size a and lenth b with random values in each place
print(x)
print(x[3, :])                               #prints a specific row or colum in the tesnor
print(x[1, 1].item())                        #prints the value of the tensor only if the tensor have one elements
print(x.view(a * b))                         #reshapes the tensor, the number of elements in the tensor need to be saved. (in this case makes the 2d tensor 1d)
# for val in x: 
#     for number in val: 
#         print(number.item())

c = torch.ones(5)
print(c) 
d = c.numpy()                               #the .numpy() func transforms the torch tensor into a numpy array
print(d)

c.add_(1)                                   #IMPORTENT!!!! if you transform a tesnor into a numpy array they will share the same memory location, that means that if you change and tensor the array will change and if you change the array the tensor will change
print(c)                                    #IMPORTENT the above statement is true only if the tesor and the np array are located in the cpu, in the gpu it won't happen
print(d)                                    #IMPORTENT the .numpy() func can only converts tesors that are located in the cpu. calculations of tesnors in the gpu will be much more faster than in the cpu

device = torch.device("cuda")               #sets the device the tesor created in to the gpu insted of the cpu
y = torch.rand(a, b, device = device)       #creates the tensor on the gpu
print(y)
z = torch.rand(a, b)
print(z)
z = z.to(device)                            #moves the tensor to the gpu
m = y + z
print(m)
print(m.device)
m = m.to("cpu")                             #moves the tesnor to the cpu
print(m.device)
m.numpy()
print(m)

h = torch.ones(a, b, device = device)
h[1, 1] = 10
print(h[1,1].item() , h[0, 0].item())