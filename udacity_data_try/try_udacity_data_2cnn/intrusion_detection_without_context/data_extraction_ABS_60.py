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
warnings.filterwarnings("ignore")

num=1
# use_gpu = torch.cuda.is_available()
class intrusion_data(Dataset):

    def __init__(self, csv_file, root_dir, cs, transform=None):
        #         self.control_data=pd.read_csv(csv_file)
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


        # return (len(self.control_data)-(240*self.cs)-1)
         return (len(self.control_data))

    def __getitem__(self,idx):

        global num
        # print(self.control_data.iloc[idx,1],num,'\r')
        num=num+1
        #print(num)

        image_path = [os.path.join(self.control_data.iloc[idx1,20],self.control_data.iloc[idx1,5]) for idx1 in range(idx,idx+(30*self.cs),30)]
        # print(str(idx), ": image_path:", image_path)

        temp=[]
        for i,img_path in enumerate(image_path):
            # img = Image.open(img_path)
            # if self.transform:
            #     img = self.transform(img)
            # data = self.control_data.iloc[idx+(i*30),np.r_[6,12,17]].as_matrix()
            data = self.control_data.iloc[idx+(i*30),np.r_[6,12,17]].as_matrix()

#             print(data.shape)
            data = data.astype('float').reshape(-1)
            anam = self.control_data.iloc[idx+(i*30),19]
            temp.append((img_path,data,anam))
            # print(str(i), ": image_path_loop:", img_path)

        return temp
        





