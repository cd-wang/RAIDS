from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
import torchvision


class mynet(nn.Module):

    def __init__(self):
        super(mynet, self).__init__()
        self.fc1 = nn.Linear(1, 3)
        self.fc2 = nn.Linear(3, 2)


    def forward(self, *inp):

        out = self.fc1(torch.stack(inp))
        out = self.fc2(out)

        return F.softmax(out, dim=-1).view(-1, 2)