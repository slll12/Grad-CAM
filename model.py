import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels=1,out_channels=8,kernel_size=3,stride=2,padding=1)#28
        self.r1 = nn.ReLU()
        self.b1 = nn.BatchNorm2d(8)
        self.p1 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.f1 = nn.Linear(7*7*8,84)
        self.f2 = nn.Linear(84,10)
    def forward(self, x):
        x = self.conv_1(x)
        x = self.r1(x)
        x = self.b1(x)
        x = self.p1(x)
        x = nn.Flatten()(x)
        x = self.f1(x)
        x = nn.ReLU()(x)
        x = self.f2(x)

        return x