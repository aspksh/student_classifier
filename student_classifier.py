import torch.nn as nn

class StudentClassifier(nn.Module):

    def __init__(self):

        super().__init__()

        self.layer1 = nn.Linear(2,4)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(4,1)

    def forward(self,x):

        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)

        return x

