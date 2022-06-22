import torch
import torch.nn as nn

class ShapeRecgognizer(nn.Module):
    def __init__(self):
        super(ShapeRecgognizer, self).__init__()
        self.neuralNetwork = nn.Sequential(
            nn.Linear(784, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )
    
    def forward(self, input):
        return self.neuralNetwork(input)
        
x = torch.rand(28, 28)
x = x.reshape(784)
print(x)
REC = ShapeRecgognizer()
output = REC(x)
print(output)