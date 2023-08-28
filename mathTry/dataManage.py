import torch
import struct
import numpy as np
import torchvision.transforms as transforms
from PIL import Image

    
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

def readBinaryData(path: str):
    with open(path ,'rb') as photos:
        magic, size = struct.unpack(">II", photos.read(8)) #first 8 bits are magic number (random number) and the size of the Data
        nrows, ncols = struct.unpack(">II", photos.read(8))
        data = np.fromfile(photos, dtype=np.dtype(np.uint8).newbyteorder('>'))
        return (data.reshape((size, nrows * ncols, 1))) / 255

def readBinaryLabels(path: str):
    with open(path ,'rb') as labels:
        magic, numOfItems = struct.unpack(">II", labels.read(8)) #first 8 bits are magic number (random number) and the size of the labels
        return np.fromfile(labels, dtype=np.dtype(np.uint8).newbyteorder('>'))


def getTarget(labelList: list, i: int) -> torch.Tensor:
    list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    list[labelList[i]] = 1
    TargetTensor = torch.tensor(list)
    return TargetTensor

def getPrepredData(dataPath:str, labelpath:str):
    # prepered data is a list of (img, desired output)
    imgData = readBinaryData(dataPath)
    labelData = readBinaryLabels(labelpath)
    if len(labelData) == len(imgData):
        preperedData = []
        for index in range(len(labelData)):
            preperedData.append((imgData[index], labelData[index]))
    return preperedData
