import torch
import torch.nn as nn
import torch.optim as optim
import os
import struct
import numpy as np
import time
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms

PATH = ".\\DATA\\"
TRAINING_PHOTOS = os.listdir(PATH)[2]
TRAINING_LABELS = os.listdir(PATH)[3]
TESTING_PHOTOS = os.listdir(PATH)[0]
TESTING_LABELS = os.listdir(PATH)[1]
DEVICE = torch.device
class ShapeRecognizer(nn.Module):
    def __init__(self):
        super(ShapeRecognizer, self).__init__()
        self.convLayers = nn.Sequential(
            nn.Conv2d(1, 4, 5),
            nn.ReLU(),
            nn.Conv2d(4, 16, 5),
            nn.ReLU(),
            nn.Conv2d(16, 64, 5),
            nn.ReLU(),
            nn.Conv2d(64, 256, 5),
            nn.ReLU(),
            nn.MaxPool2d(5, 5)
            
            # 97% = (1, 6, 5), (6, 16, 5)
        )
        self.linearLayers = nn.Sequential(
            nn.Linear(1024, 16),
            # nn.Linear(784, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 10),
            nn.ReLU(),
        )
    
    def forward(self, input: torch.Tensor):
        input = self.convLayers(input)
        input = input.view(-1)
        input = self.linearLayers(input)
        return input
    
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
    
    def getData(data: list[list]) -> list[list]:
        DataList: list[list] = []
        for n in range(len(data)):
            pixelList = []
            for i in data[n]:
                i = i / 255
                pixelList.append(i)
            DataList.append(pixelList)
            print(n)
        return DataList
    
    
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
        output = output.view(10)
        highest = 0
        guess = 0
        sum = 0
        exactGuess = 0
        for i, item in enumerate(output):
            sum += item
            if item.item() > highest:
                highest = item
                guess = i
        try:
            highest = highest.item()
        except:
            pass
        try:
            sum = sum.item()
        except:
            pass
        try:
            print("guess: ", guess, "\ncertenty: ", highest / sum, "\ntraget: ", target, "\n")
            if (highest / sum) == 1:
                exactGuess += 1
            return [target, guess, exactGuess]
        except:
            print("didn't guess")
            return [1, 2, 0]
        

        
            
    def Train(self):
        criterion = nn.MSELoss()
        optimizer = optim.SGD(self.parameters(), lr=0.001, momentum = 0.9)
        print("Loading Data")
        print("\n")
        DataList = ShapeRecognizer.getData(ShapeRecognizer.TrainData)
        plt.ion()
        fig1, ax1 = plt.subplots()
        array = np.array(DataList[0]).reshape(28, 28)
        axim1 = ax1.imshow(array, cmap='gist_gray')
        del array
        for epoch in range(2):  # loop over the dataset multiple times
            # random.shuffle(DATALIST)
            ShapeRecognizer.i = -1
            for i in range(60000):
                ShapeRecognizer.i += 1
                input =  torch.tensor([DataList[ShapeRecognizer.i]])
                input = input.to(torch.float32)
                input = input.view(1, 1, 28, 28)
                # input = input.view(784)
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
                print("[Epoch %s / 60000]" % i , end = "\r")
                matrix = np.array(DataList[ShapeRecognizer.i]).reshape(28, 28)
                axim1.set_data(matrix)
                fig1.canvas.flush_events()
            print("\n")

        ShapeRecognizer.TrainLabels.close()
        ShapeRecognizer.TrainPhotos.close()

                
    def Test(self):
        ShapeRecognizer.i = -1
        score = 0
        print("loading TestData")
        DataList = ShapeRecognizer.getData(ShapeRecognizer.TestData)
        exactGuess = 0
        for i in range(10000):
            ShapeRecognizer.i += 1
            input =  torch.tensor([DataList[ShapeRecognizer.i]])
            input = input.to(torch.float32)
            input = input.view(1, 1, 28, 28)
            # input = input.view(784)
            target = ShapeRecognizer.getTarget(ShapeRecognizer.TestList)
            target = target.to(torch.float32) 
            target = target.view(10)  
            self.evaluate(input, target)
            if self.evaluate(input, target)[0] == self.evaluate(input, target)[1]:
                score += 1 
                exactGuess += self.evaluate(input, target)[2]
            self.finalScore = (score / 10000) * 100
        print(self.finalScore) 
        print(exactGuess)
            


REC = ShapeRecognizer()
REC.eval()
REC.Train()
REC.Test()
FILE = "numberRec%s.pth" % REC.finalScore
torch.save(REC, FILE)