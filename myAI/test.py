import numpy as np
import struct
import matplotlib.pyplot as plt
import os
import torch

# PATH = "C:\\vscodeprojects\\python\\machine-learning\\MNIST_tarinData\\"
# TRAINING_PHOTOS = os.listdir(PATH)[0]
# TRAINING_LABELS = os.listdir(PATH)[1]

# with open(PATH + TRAINING_PHOTOS,'rb') as file:
#     magic, size = struct.unpack(">II", file.read(8))
#     nrows, ncols = struct.unpack(">II", file.read(8))
#     data1 = np.fromfile(file, dtype=np.dtype(np.uint8).newbyteorder('>'))
#     data1 = data1.reshape((size, nrows * ncols))
#     data1 = data1.reshape((size, nrows, ncols))


# with open(PATH + TRAINING_LABELS,'rb') as file:
#     magic, numOfItems = struct.unpack(">II", file.read(8))
#     print(magic, numOfItems)
#     data = np.fromfile(file, dtype=np.dtype(np.uint8).newbyteorder('>'))
#     print(data)

    
for i in range(60000):
    print(i)
    if i == 0:
        break