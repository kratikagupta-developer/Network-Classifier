import torch.nn as nn
import torch.nn.functional as F
import torch


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(73, 128)
        self.bn_input = nn.BatchNorm1d(128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 5)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = F.dropout(x)
        x = self.bn_input(x)
        x = self.fc2(x)
        return x
