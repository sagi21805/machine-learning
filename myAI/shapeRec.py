import torch
import torch.nn as nn
from torchvision import transforms
import torch.optim as optim
from PIL import Image
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import struct
import numpy as np

PATH = "C:\\vscodeprojects\\python\\machine-learning\\DATA\\"
TRAINING_PHOTOS = os.listdir(PATH)[2]
TRAINING_LABELS = os.listdir(PATH)[3]
TESTING_PHOTOS = os.listdir(PATH)[0]
TESTING_LABELS = os.listdir(PATH)[1]

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
            nn.Linear(16, 10),
            nn.ReLU(),
            
        )
    
    def forward(self, input):
        return self.neuralNetwork(input)
    
    # i = -1
    # def getTarget(list: list):
    #     targetList = [0, 0, 0, 0, 0, 0, 0]
    #     target = int(list[ShapeRecognizer.i].split(".")[0])
    #     targetList[target] = 1
    #     targetTensor = torch.tensor(targetList)
    #     # yield torch.tensor([int(DATALIST[ShapeRecognizer.i].split(".")[0])])
    #     return targetTensor

    # def dataIntoTensor(list: list, path: str):
    #     img = Image.open(path + list[ShapeRecognizer.i])
    #     convert_tensor = transforms.ToTensor()
    #     x: torch.Tensor = convert_tensor(img)
    #     return x

    # def showImg(list: list, path: str):
    #     img = mpimg.imread(path + list[ShapeRecognizer.i])
    #     imgplot = plt.imshow(img)
    #     plt.show()
    #     plt.close()
    #     yield ShapeRecognizer.iii
        
    # def converImgToCsv():
    #     for i in ShapeRecognizer.dataIntoTensor():
    #         imgTensor_np = i.numpy()
    #         imgTensor_np = imgTensor_np.reshape(784)
    #         dataFrame = pd.DataFrame(imgTensor_np)
    #         dataFrame.to_csv("testfile", index=False)
    #         dataFrame = pd.read_csv("testfile")
    #         yield dataFrame
    i = -1
    
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
    
    def getData(data) -> torch.Tensor:
        dataList = []
        for i in data[ShapeRecognizer.i]:
            i = i / 255
            dataList.append(i)
        dataTensor = torch.tensor([dataList])
        return dataTensor
    
    
    TrainLabels = open(PATH + TRAINING_LABELS,'rb')
    magic, numOfItems = struct.unpack(">II", TrainLabels.read(8))
    TrainList = np.fromfile(TrainLabels, dtype=np.dtype(np.uint8).newbyteorder('>'))
    
    TestLabels = open(PATH + TESTING_LABELS,'rb')
    magic, numOfItems = struct.unpack(">II", TestLabels.read(8))
    TestList = np.fromfile(TestLabels, dtype=np.dtype(np.uint8).newbyteorder('>'))
    
    def getTarget(TargetList) -> torch.Tensor:
        list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        list[TargetList[ShapeRecognizer.i]] = 1
        TargetTensor = torch.tensor(list)
        return TargetTensor

    def evaluate(self, input: torch.Tensor, target: list):
        for i, number in enumerate(target):
            if number == 1:
                target = i
        output = self.forward(input).detach()
        highest = 0
        guess = 0
        for array in output:
            for i, item in enumerate(array):
                if item > highest:
                    highest = item
                    guess = i
        try:
            highest = highest.item()
        except:
            pass
        print("guess: ", guess, "\ncertenty: ", highest, "\ntraget: ", target, "\n")
        return [target, guess]
            
    def Train(self):
        criterion = nn.MSELoss()
        optimizer = optim.SGD(self.parameters(), lr=0.001, momentum = 0.9)

        for epoch in range(2):  # loop over the dataset multiple times
            # random.shuffle(DATALIST)
            ShapeRecognizer.i = -1
            print("epoch: ", epoch)
            for i in range(60000):
                ShapeRecognizer.i += 1
                input =  ShapeRecognizer.getData(ShapeRecognizer.TrainData)
                input = input.to(torch.float32)
                target = ShapeRecognizer.getTarget(ShapeRecognizer.TrainList)
                target = target.to(torch.float32)
                # for img in ShapeRecognizer.showImg():
                #     pass
                
                optimizer.zero_grad()

                # forward + backward + optimize
                output = self.forward(input)   
                loss: torch.Tensor = criterion(output, target)
                loss.backward()
                optimizer.step()
                # print statistics
                # print("loss: ",loss.item())
                print("photo: ", i)
            ShapeRecognizer.TrainLabels.close()
            ShapeRecognizer.TrainPhotos.close()
                
    def Test(self):
        ShapeRecognizer.i = -1
        score = 0
        for i in range(10000):
            ShapeRecognizer.i += 1
            input =  ShapeRecognizer.getData(ShapeRecognizer.TestData)
            input = input.to(torch.float32)
            target = ShapeRecognizer.getTarget(ShapeRecognizer.TestList)
            target = target.to(torch.float32)   
            self.evaluate(input, target)
            if self.evaluate(input, target)[0] == self.evaluate(input, target)[1]:
                score += 1 
            finalScore = score / 10000
        print(finalScore)
            


REC = ShapeRecognizer()
REC.eval()
REC.Train()
REC.Test()