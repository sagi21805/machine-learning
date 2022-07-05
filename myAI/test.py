import numpy as np
import struct
import matplotlib.pyplot as plt
from matplotlib import animation
import os
import torch
import threading
import cv2
import time

PATH = ".\\DATA\\"
TRAINING_PHOTOS = os.listdir(PATH)[2]
TRAINING_LABELS = os.listdir(PATH)[3]
TESTING_PHOTOS = os.listdir(PATH)[0]
TESTING_LABELS = os.listdir(PATH)[1]

TrainPhotos = open(PATH + TRAINING_PHOTOS,'rb')
magic, size = struct.unpack(">II", TrainPhotos.read(8))
nrows, ncols = struct.unpack(">II", TrainPhotos.read(8))
TrainData = np.fromfile(TrainPhotos, dtype=np.dtype(np.uint8).newbyteorder('>'))
TrainData = TrainData.reshape((size, nrows * ncols))

TestPhotos = open(PATH + TESTING_PHOTOS,'rb')
magic, size = struct.unpack(">II", TestPhotos.read(8))
nrows, ncols = struct.unpack(">II", TestPhotos.read(8))
TestData = np.fromfile(TestPhotos, dtype=np.dtype(np.uint8).newbyteorder('>'))
TestData = TestData.reshape((size, nrows * ncols))

def getData(data: list[list]) -> list[list]:
    DataList: list[list] = []
    for n in range(1):
        pixelList = []
        for i in data[n]:
            i = i / 255
            pixelList.append(i)
        DataList.append(pixelList)
    return torch.tensor(pixelList)


plt.ion()

fig1, ax1 = plt.subplots()

array = TrainData[0].reshape(28, 28)
axim1 = ax1.imshow(array, cmap='gist_gray')


del array

for i in range(5000):
    print(".", end="")
    matrix = TrainData[i+1].reshape(28, 28)
    
    axim1.set_data(matrix)
    fig1.canvas.flush_events()
print()