import torch
import torch.nn as nn
from torchvision import transforms
import torch.optim as optim
from PIL import Image
import os

PATH = "C:\\vscodeprojects\\python\\machine-learning\\myAI\\shape_data\\"
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
    
   
    def getData():
        global I
        for I in range(len(os.listdir(PATH))):
                print(os.listdir(PATH)[I])
                img = Image.open(PATH + os.listdir(PATH)[I])
                convert_tensor = transforms.ToTensor()
                x = convert_tensor(img)
                yield x

    def train(self):
        self.loss = nn.MSELoss()
        optimizer = optim.SGD(self.neuralNetwork.parameters(), lr=0.001, momentum=0.9)
        for epoch in range(2):  # loop over the dataset multiple times

            running_loss = 0.0
            for i in range(len(os.listdir(PATH))):
                # get the inputs; data is a list of [inputs, labels

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.forward(DATALIST[0])
                loss = self.loss(outputs, torch.tensor([int(os.listdir(PATH)[I].split(".")[0])]))
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:    # print every 2000 mini-batches
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                    running_loss = 0.0


REC = ShapeRecognizer()
for image in ShapeRecognizer.getData():
    DATALIST = [image.view(784), os.listdir(PATH)[I].split(".")[0]]
REC.train()
print("1")