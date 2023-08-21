import torch
import torch.nn as nn
import torch.optim as optim
import os
import applyMyOwnFilter as fillter
import mnistData
import numpy as np
PATH = ".\\DATA\\"
TRAINING_PHOTOS = os.listdir(PATH)[2]
TRAINING_LABELS = os.listdir(PATH)[3]
TESTING_PHOTOS = os.listdir(PATH)[0]
TESTING_LABELS = os.listdir(PATH)[1]
DEVICE = torch.device
class Recognizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.convLayers = nn.Sequential(
            nn.Conv2d(1, 4, 5),
            nn.ReLU(),
            nn.Conv2d(4, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(3, 3)
            
            # 97% = (1, 6, 5), (6, 16, 5)
        )
        self.linearLayers = nn.Sequential(
            nn.Linear(400, 16),
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
    
    i = -1

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
            highest = highest.item()
            sum = sum.item()
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
        # plt.ion()
        # fig1, ax1 = plt.subplots()
        # array = np.array(DataList[0]).reshape(28, 28)
        # axim1 = ax1.imshow(array, cmap='gist_gray')
        # del array
        for epoch in range(2):
            Recognizer.i = -1
            for i in range(60000):
                Recognizer.i += 1
                input = data.changeScale(data.TrainData[Recognizer.i])
                input = torch.tensor(fillter.ApplyFillter(Recognizer.i, input, (3, 3), [[0,-2, 0], [-1, 2, -1], [0, 2, 0]], 28, 28))
                input = input.to(torch.float32)
                input = input.view(1, 1, 25, 25)
                target = data.getTarget(data.TrainList, Recognizer.i)
                target = target.to(torch.float32)
                
                optimizer.zero_grad()
                output = self.forward(input)   
                loss: torch.Tensor = criterion(output, target)
                loss.backward()
                optimizer.step()
                x = i + 1
                print("[img %s / 60000]" % x , end = "\r")
                # matrix = np.array(DataList[Recognizer.i]).reshape(28, 28)
                # axim1.set_data(matrix)
                # fig1.canvas.flush_events()
            print("\n")

                
    def Test(self):
        Recognizer.i = -1
        score = 0
        print("loading TestData")
        exactGuess = 0
        for i in range(10000):
            Recognizer.i += 1
            input = data.changeScale(data.TrainData[Recognizer.i])
            input = torch.tensor(fillter.ApplyFillter(Recognizer.i, np.array(input), 3, 3, [[0,-2, 0], [-1, 2, -1], [0, 2, 0]], 28, 28))
            input = input.to(torch.float32)
            input = input.view(1, 1, 25, 25)
            
            target = data.getTarget(data.TestList, Recognizer.i)
            target = target.to(torch.float32) 
            target = target.view(10)  
            self.evaluate(input, target)
            if self.evaluate(input, target)[0] == self.evaluate(input, target)[1]:
                score += 1 
                exactGuess += self.evaluate(input, target)[2]
            self.finalScore = (score / 10000) * 100
        print(self.finalScore) 
        print(exactGuess)

            
if __name__ == "__main__":
    REC = Recognizer()
    REC.eval()
    REC.Train()
    REC.Test()
    FILE = "numberRec%s.pth" % REC.finalScore
    torch.save(REC, FILE)