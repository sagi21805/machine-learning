import torch
import torch.nn as nn
from torchvision import transforms
import torch.optim as optim
from PIL import Image
import os
import numpy as np
import pandas as pd
from torchdata.datapipes.iter import IterableWrapper, FileOpener

PATH = "C:\\Users\\Demacia\\Desktop\\sagi\\machine-learning\\myAI\\shape_data\\"

class ShapeRecognizer(nn.Module):
    def __init__(self):
        super(ShapeRecognizer, self).__init__()
        self.neuralNetwork = nn.Sequential(
            nn.Linear(784, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
    
    def forward(self, input):
        return self.neuralNetwork(input)
    
    def getName():
        for i in range(len(os.listdir(PATH))):
            yield torch.tensor([int(os.listdir(PATH)[i].split(".")[0])])

    def dataIntoTensor():
        for i in range(len(os.listdir(PATH))):
                print(os.listdir(PATH)[i])
                img = Image.open(PATH + os.listdir(PATH)[i])
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
    
    def getData():
        for input in ShapeRecognizer.dataIntoTensor():
            pass
        for target in ShapeRecognizer.getName():
            pass
        yield [input, target]

    def train(self):
        self.loss = nn.MSELoss()
        optimizer = optim.SGD(self.neuralNetwork.parameters(), lr=0.001, momentum=0.9)
        for epoch in range(2):  # loop over the dataset multiple times

            running_loss = 0.0
            for list in ShapeRecognizer.getData():
                pass
            for i in range(len(os.listdir(PATH))):
                # get the inputs; data is a list of [inputs, labels
                input = list[0].view(784)
                target = list[1]
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                output = self.forward(input)
                loss = self.loss(output, target)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:    # print every 2000 mini-batches
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                    running_loss = 0.0


REC = ShapeRecognizer()
REC.train()