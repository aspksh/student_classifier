class StudentClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(2, 16)
        self.relu1  = nn.ReLU()
        self.layer2 = nn.Linear(16, 8)
        self.relu2  = nn.ReLU()
        self.layer3 = nn.Linear(8, 1)
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.layer2(x)
        x = self.relu2(x)
        x = self.layer3(x)
        return x
