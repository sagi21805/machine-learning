import numpy as np
import matplotlib.pyplot as plt
import os
import struct

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


plt.ion()
fig1, ax1 = plt.subplots()
array = np.array(TrainData[0]).reshape(28, 28)
axim1 = ax1.imshow(array, cmap='gist_gray')
for i in range(60000):
    matrix = np.array(TrainData[i]).reshape(28, 28)
    axim1.set_data(matrix)
    fig1.canvas.flush_events()
    if i > 20:
        plt.pause(4)