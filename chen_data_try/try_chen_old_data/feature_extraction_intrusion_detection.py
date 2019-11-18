from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import cv2
import torch.optim as optim
from data_extraction_ABS import *
from torch.autograd import Variable

import numpy as np
from keras.models import load_model
import pandas as pd
import glob
from config import TestConfig
from train import *

import numpy as np
from collections import deque
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from skimage.exposure import rescale_intensity


class Model(object):
    def __init__(self,
                 model_path,
                 X_train_mean_path):

        self.model = model_path
        self.X_mean = np.load(X_train_mean_path)
        self.mean_angle = np.array([-0.004179079])
        self.img0 = None
        self.state = deque(maxlen=2)

    def predict(self, img_path):
        img1 = load_img(img_path, grayscale=True, target_size=(66, 200))
        img1 = img_to_array(img1)

        if self.img0 is None:
            self.img0 = img1
            return self.mean_angle

        elif len(self.state) < 1:
            img = img1 - self.img0
            img = rescale_intensity(img, in_range=(-255, 255), out_range=(0, 255))
            img = np.array(img, dtype=np.uint8)  # to replicate initial model
            self.state.append(img)
            self.img0 = img1

            return self.mean_angle

        else:
            img = img1 - self.img0
            img = rescale_intensity(img, in_range=(-255, 255), out_range=(0, 255))

            img = np.array(img, dtype=np.uint8)  # to replicate initial model

            self.state.append(img)
            self.img0 = img1

            X = np.concatenate(self.state, axis=-1)
            X = X[:, :, ::-1]

            X = np.expand_dims(X, axis=0)
            X = X.astype('float32')
            X -= self.X_mean
            X /= 255.0

            return self.model.predict(X)[0]

face_dataset = intrusion_data(csv_file='interpolated_random_a_10702to24371_part1_0p2to0p9_9569.csv',root_dir='/home/pi/new_try_raspberry_pi--/chen_data_try/chen_old_try/chen_old_data',
                              cs=1,transform=transforms.Compose([transforms.Resize(256),transforms.RandomResizedCrop(224),transforms.ToTensor()]))
test_dataset=intrusion_data(csv_file='interpolated_random_a_10702to24371_part1_0p2to0p9_4103.csv',root_dir='/home/pi/new_try_raspberry_pi--/chen_data_try/chen_old_try/chen_old_data',cs=1,
                            transform=transforms.Compose([transforms.Resize(256),transforms.RandomResizedCrop(224),transforms.ToTensor()]))

dataloader = DataLoader(face_dataset, batch_size=1, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

config = TestConfig()

ch = config.num_channels
row = config.img_height
col = config.img_width
model_org = create_comma_model_large_dropout(row, col, ch, load_weights=True)

model = Model(model_org,
              "data/X_train_gray_diff2_mean.npy")

net = mynet()

criterion = F.nll_loss
optimizer = optim.Adam(net.parameters(), lr=0.01)

def accuracy(act , out):
    wrong1=0
    for a,f in zip(act,out):
        if a!=f:
            wrong1+=1
    return wrong1

epochs = 40
cnt = 0
for iter_x in range(epochs):
    inter = .10
    loss1 = 0
    finale = 0
    # train_loss = 0
    finale_1 = 0
    finale_0 = 0
    get_1 = 0
    get_0 = 0
    target_1_indices_1 = 0
    target_1_indices_0 = 0
    target_0_indices_1 = 0
    target_0_indices_0 = 0
    for i_batch, sample in enumerate(dataloader):  # for each training i_batch

        if i_batch / len(dataloader) > inter:
            print('completed %epoch ', inter)
            inter += .10

        data = []
        result = []
        for sm in sample:
            imag, dat, res = sm
            data = dat
            result.append(res)

        for img in imag:
            preds = model.predict(img)
            preds = preds.astype('float').reshape(-1)
            preds = preds[0]

        target = torch.stack(result)
        target = target.view(-1)

        final_vars = []
        final_vars = torch.FloatTensor([[[abs(data-preds)]]])
        x = net(*final_vars)
        values, indices = x.max(1)
        loss = criterion(x, Variable(target))
        loss.backward()
        optimizer.step()
        net.zero_grad()

        if(i_batch>10):
            break
    # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&TESTING &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    for i_batch, sample in enumerate(test_loader):


        data = []
        result = []
        for sm in sample:
            imag, dat, res = sm
            data = dat
            result.append(res)

        for img in imag:
            preds = model.predict(img)
            preds = preds.astype('float').reshape(-1)
            preds = preds[0]

        final_vars = torch.FloatTensor([[[abs(data - preds)]]])
        target = torch.stack(result)
        target = target.view(-1)

        x = net(*final_vars)
        values, indices = x.max(1)
        print("target_indices: ", target, indices.data)
        loss1 += accuracy(target, indices.data)
        finale += 1

        for Target, Indices in zip(target, indices.data):
            if Target == 1:
                finale_1 += 1
            else:
                finale_0 += 1
            if Indices == 1:
                get_1 += 1
            else:
                get_0 += 1
            if Target == 1 and Indices == 1:
                target_1_indices_1 += 1
            elif Target == 1 and Indices == 0:
                target_1_indices_0 += 1
            elif Target == 0 and Indices == 0:
                target_0_indices_0 += 1
            elif Target == 0 and Indices == 1:
                target_0_indices_1 += 1
        #
        if(i_batch>10):
            break

    acc = 1.0 - (loss1 / finale)
    acc11 = target_1_indices_1 / (target_1_indices_1 + target_1_indices_0)
    print('accuracy=', acc)
    print('accuracy11=', acc11)
    print("finale_1_0_get_1_0: ", finale_1, finale_0, get_1, get_0)
    print("target_indices11,10,00,01: ", target_1_indices_1, target_1_indices_0, target_0_indices_0, target_0_indices_1)

    file2write = open("accuracy_file_f_e_predict_a_consecutive.txt", 'a')
    file2write.write("accuracy= " + str(acc) + "\n")
    file2write.write("accuracy1111= " + str(acc11) + "\n")

    file2write.write(
        "finale_1_0_get_1_0= " + "\n" + str(finale_1) + "\n" + str(finale_0) + "\n" + str(get_1) + "\n" + str(
            get_0) + "\n")
    file2write.write(
        "target_indices11,10,00,01= " + "\n" + str(target_1_indices_1) + "\n" + str(target_1_indices_0) + "\n" + str(
            target_0_indices_0) + "\n" + str(target_0_indices_1) + "\n")
    file2write.close()

    filenm = 'epoch_' + str(cnt) + '.pt'
    torch.save(net, os.path.join(('saved_consecutive_models'), filenm))

    cnt = cnt + 1
    if (cnt % 10 == 0):
        print('EPOCH completed by %', (cnt / 40) * 100)





