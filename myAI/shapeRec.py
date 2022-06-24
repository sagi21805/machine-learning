import torch
import torch.nn as nn
from torchvision import transforms
import torch.optim as optim
from PIL import Image
import os
import numpy as np
import pandas as pd
from torchdata.datapipes.iter import IterableWrapper, FileOpener
import matplotlib.pyplot as plt
import SimpleITK as sitk
import sys


PATH = "C:\\vscodeprojects\\python\\machine-learning\\myAI\\shape_data\\"
#C:\\Users\\Demacia\\Desktop\\sagi\\machine-learning\\myAI\\shape_data\\
DATALIST: list = os.listdir(PATH)

class ShapeRecognizer(nn.Module):
    def __init__(self):
        super(ShapeRecognizer, self).__init__()
        self.neuralNetwork = nn.Sequential(
            nn.Linear(784, 8),
            nn.ReLU(),
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )
    
    def forward(self, input):
        return self.neuralNetwork(input)
    
    i = -1
    def getTarget():
        ShapeRecognizer.i += 1
            # targetList = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            # target = int(DATALIST[i].split(".")[0])
            # targetList[target - 1] = 1
            # targetTensor = torch.tensor(targetList)
        yield torch.tensor([int(DATALIST[ShapeRecognizer.i].split(".")[0])])

    def dataIntoTensor():
        ShapeRecognizer.i += 1
        img = Image.open(PATH + DATALIST[ShapeRecognizer.i])
        convert_tensor = transforms.ToTensor()
        x = convert_tensor(img)
        yield x
    
    # def converImgToCsv():
    #     for i in ShapeRecognizer.dataIntoTensor():
    #         imgTensor_np = i.numpy()
    #         imgTensor_np = imgTensor_np.reshape(784)
    #         dataFrame = pd.DataFrame(imgTensor_np)
    #         dataFrame.to_csv("testfile", index=False)
    #         dataFrame = pd.read_csv("testfile")
    #         yield dataFrame


    def showImgOnPlt():
        for i in DATALIST:
            img1 = DATALIST[i]
            plt.imshow(img1)
            yield 

    def train(self):
        self.loss = nn.MSELoss()
        optimizer = optim.SGD(self.neuralNetwork.parameters(), lr=0.01, momentum=0.99)
        imgCount = 0
        success = 0
        for epoch in range(1):  # loop over the dataset multiple times
            running_loss = 0.0
            # random.shuffle(DATALIST)
            for i in range(len(DATALIST)):
                for input in ShapeRecognizer.dataIntoTensor():
                    input = input.view(784)
                    input = input.to(torch.float32)
                for target in ShapeRecognizer.getTarget():
                    target = target.to(torch.float32)
                ShapeRecognizer.showImgOnPlt()
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                output = self.forward(input)
                loss = self.loss(output, target)
                loss.backward()
                optimizer.step()
                # print statistics
                running_loss += loss.item()
                print("train: \n")
                print("guess: ", output)
                print("real: ", target)
                print("loss: ", loss.item())
                if target == output.item():
                    success += 1
                imgCount += 1
                print("img count: ", imgCount, "\n","success: ", success, "\n", "success rate: ", success / imgCount)
                print("epoch: ", epoch)
                print("\n")
                
                    


REC = ShapeRecognizer()
REC.train()