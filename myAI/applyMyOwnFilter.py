import matplotlib.pyplot as plt
from pandas import array
import torch
import struct
import numpy as np
import os
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.image as mpimg


PATH = ".\\DATA\\"
TRAINING_PHOTOS = os.listdir(PATH)[2]
TRAINING_LABELS = os.listdir(PATH)[3]
TESTING_PHOTOS = os.listdir(PATH)[0]
TESTING_LABELS = os.listdir(PATH)[1]
DEVICE = torch.device

TrainPhotos = open(PATH + TRAINING_PHOTOS,'rb')
magic, size = struct.unpack(">II", TrainPhotos.read(8))
nrows, ncols = struct.unpack(">II", TrainPhotos.read(8))
TrainData = np.fromfile(TrainPhotos, dtype=np.dtype(np.uint8).newbyteorder('>'))
TrainData = TrainData.reshape((size, nrows * ncols))

# img = TrainData[0].reshape(28, 28)
# plt.imshow(img, cmap='gist_gray')
# plt.show()

padding = 1 #how much pixels per one down
stride = 1 #how much pixels per slide
kernel_sizeX = 3
kernel_sizeY = 3
dataSize = 1
photoSizeX = 28
photoSizeY = 28
filter = [[0,-2, 0], [-1, 2, -1], [0, 2, 0]]
filterMatrix = np.array(filter)


def ApplyFillter(data):
    PhotoAfterFillter = []
    for x in range(dataSize):
        pixelTimes = 0
        rowTimes = 0
        photo = data[25].reshape(photoSizeX, photoSizeY)
        for n in range(25):     
            for i in range(25):
                kernel = []    
                for row in range(padding * rowTimes, (kernel_sizeY + (padding * rowTimes))):
                    pixelList = []
                    for pixel in range(stride * pixelTimes, (kernel_sizeX + (stride * pixelTimes))):
                        pixel = photo[row][pixel]
                        pixelList.append(pixel)
                    kernel.append(pixelList)
                kernel = np.array(kernel)
                newPixel = kernel * filterMatrix
                PhotoAfterFillter.append(newPixel.sum())
                pixelTimes += 1
            rowTimes += 1
            pixelTimes = 0
        PhotoAfterFillter = np.array(PhotoAfterFillter)
    return PhotoAfterFillter
                        
