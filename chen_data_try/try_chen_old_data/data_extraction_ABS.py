from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image


# Ignore warnings
import warnings

import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
import torchvision

warnings.filterwarnings("ignore")

num=1

class intrusion_data(Dataset):

    def __init__(self, csv_file, root_dir, cs, transform=None):
        frames=[]
        self.root_dir = root_dir
        self.transform = transform
        folder_name = os.listdir(root_dir)
        all_sub_folders = []

        for folder in folder_name:
            if os.path.isdir(os.path.join(root_dir,folder)): #if curr directory is a folder
                all_sub_folders.append(os.path.join(root_dir, folder))

        for folder in all_sub_folders:
            data_frame = pd.read_csv(os.path.join(folder, csv_file))
            data_frame['folder'] = folder
            # print(folder)
            frames.append(data_frame)

        self.control_data=pd.concat(frames)
        self.cs=cs

    def __len__(self):

        return (len(self.control_data))

    def __getitem__(self,idx):

        global num

        num=num+1

        image_path = [os.path.join(self.control_data.iloc[idx1,20],self.control_data.iloc[idx1,5]) for idx1 in range(idx,idx+(30*self.cs),30)]

        temp=[]
        for i,img_path in enumerate(image_path):

            data = self.control_data.iloc[idx+(i*30),np.r_[6]].as_matrix()
            data = data.astype('float').reshape(-1)
            anam = self.control_data.iloc[idx+(i*30),19]


            temp.append((img_path,data,anam))

        return temp

class mynet(nn.Module):

    def __init__(self):
        super(mynet, self).__init__()
        self.fc1 = nn.Linear(1, 2)
        self.fc2 = nn.Linear(2, 2)


    def forward(self, *inp):

        out = self.fc1(torch.stack(inp))
        out = self.fc2(out)

        return F.softmax(out, dim=-1).view(-1, 2)






