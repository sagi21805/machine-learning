import torch
import os
import struct
import numpy as np
import torchvision.transforms as transforms
from PIL import Image

    
PATH = ".\\DATA\\"
TRAINING_PHOTOS = os.listdir(PATH)[2]
TRAINING_LABELS = os.listdir(PATH)[3]
TESTING_PHOTOS = os.listdir(PATH)[0]
TESTING_LABELS = os.listdir(PATH)[1]

def getTarget_PNG(list: list, i:int):
    targetList = [0, 0, 0, 0, 0, 0, 0]
    target = int(list[i].split(".")[0])
    targetList[target] = 1
    targetTensor = torch.tensor(targetList)
    return targetTensor

def dataIntoTensor_PNG(list: list, path: str, i: int):
    img = Image.open(path + list[i])
    convert_tensor = transforms.ToTensor()
    x: torch.Tensor = convert_tensor(img)
    return x

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

TrainLabels = open(PATH + TRAINING_LABELS,'rb')
magic, numOfItems = struct.unpack(">II", TrainLabels.read(8))
TrainList = np.fromfile(TrainLabels, dtype=np.dtype(np.uint8).newbyteorder('>'))

TestLabels = open(PATH + TESTING_LABELS,'rb')
magic, numOfItems = struct.unpack(">II", TestLabels.read(8))
TestList = np.fromfile(TestLabels, dtype=np.dtype(np.uint8).newbyteorder('>'))

def changeScale(data: list) -> list:
    DataList: list[list] = []
    for i in data:
        i = i / 255
        DataList.append(i)
    return DataList

def getTarget(TargetList: list, i: int) -> torch.Tensor:
    list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    list[TargetList[i]] = 1
    TargetTensor = torch.tensor(list)
    return TargetTensor