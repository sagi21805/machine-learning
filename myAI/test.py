import random
import torch


global I 
I = 0
def generator():
    global I
    I += 1
    yield I
    
for i in generator():
    print(i)

for i in range(10):
    for num in generator():
        pass
    print(num)