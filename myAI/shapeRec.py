import torch
import torch.nn as nn

class ShapeRecgognizer(nn.Module):
    def __init__(self):
        super(ShapeRecgognizer, self).__init__()
        self.neuralNetwork = nn.Sequential(
            nn.Linear(3, 2),
            nn.ReLU(),
            # nn.Linear(16, 16),
            # nn.ReLU(),
            # nn.Linear(16, 16),
            # nn.ReLU(),
            nn.Linear(2, 2),
            nn.ReLU(),
            nn.Linear(2, 1),
        )
    
    def forward(self, input):
        return self.neuralNetwork(input)
        
x = torch.tensor([10, 10, 10])
x = x.long()
REC = ShapeRecgognizer()
print(REC.forward(x))
