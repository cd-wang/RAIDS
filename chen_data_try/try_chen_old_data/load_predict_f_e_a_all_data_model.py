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
import cv2
import torch.optim as optim
from data_extraction_ABS import *

from torch.optim import lr_scheduler
from torch.autograd import Variable
from model_abs_linear import *

import numpy as np
from keras.models import load_model
import pandas as pd
import glob
from config import TestConfig
from train import *
from time import gmtime, strftime

import numpy as np
from collections import deque
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from skimage.exposure import rescale_intensity
from train import *

import datetime

time_a = datetime.datetime.now()

print("intrusion_detection_start: ", time_a)


class Model(object):
    def __init__(self,
                 model_path,
                 X_train_mean_path):

        self.model = model_path
        # self.model = load_model(model_path)
        self.X_mean = np.load(X_train_mean_path)
        self.mean_angle = np.array([-0.004179079])
        self.img0 = None
        self.state = deque(maxlen=2)

    def predict(self, img_path):
        img1 = load_img(img_path, grayscale=True, target_size=(66, 200))
        img1 = img_to_array(img1)

        if self.img0 is None:
            self.img0 = img1

            # print("00000000")
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


test_dataset=intrusion_data(csv_file='interpolated_random_a_10702to24371_part1_0p2to0p9_4103.csv',root_dir='/home/pi/new_try_raspberry_pi--/chen_data_try/chen_old_try/chen_old_data',cs=1,
                            transform=transforms.Compose([transforms.Resize(256),transforms.RandomResizedCrop(224),transforms.ToTensor()]))

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


net.eval()
net = torch.load('saved_consecutive_models/epoch_38.pt')
loss1 = 0
finale = 0
train_loss = 0
finale_1 = 0
finale_0 = 0
get_1 = 0
get_0 = 0
target_1_indices_1 = 0
target_1_indices_0 = 0
target_0_indices_1 = 0
target_0_indices_0 = 0
for i_batch, sample in enumerate(test_loader):
    ave_time = 0
    max_time = 0

    data = []
    result = []
    for sm in sample:
        imag, dat, res = sm
        data = dat
        result.append(res)


    for img in imag:
        # print("load_one_image_start: ", strftime("%Y-%m-%d %H:%M:%S", gmtime()))
        # time1 = strftime("%Y-%m-%d %H:%M:%S", gmtime())
        time1 = datetime.datetime.now()

        preds = model.predict(img)

        preds = preds.astype('float').reshape(-1)
        preds = preds[0]
        # print("preds: ", preds)
        # time2 = strftime("%Y-%m-%d %H:%M:%S", gmtime())
        # time2 = datetime.datetime.now()
        # k = time2 - time1
        #
        # print("del_time1-time2: ", k, time1, time2)
        # print("load_one_image_end: ", strftime("%Y-%m-%d %H:%M:%S", gmtime()))

    time_start = time1
    target = torch.stack(result)
    target = target.view(-1)
    # final_vars = []
    # for tens in data:
    #     final_vars.append(Variable(tens))

    final_vars = torch.FloatTensor([[[abs(data - preds)]]])

    # final_vars = torch.FloatTensor([[[data, preds]]]).cuda()
    # print("preds_data_ibatch: ", preds, data, i_batch)

    x = net(*final_vars)

    values, indices = x.max(1)
    # print("target_indices: ", target, indices.data)
    # time3 = strftime("%Y-%m-%d %H:%M:%S", gmtime())
    time3 = datetime.datetime.now()

    # print("detection_result: ", time3)
    k = time3-time_start
    print("input-output: ", k)
    # k =int(k)
    #
    # if max_time < k :
    #     max_time = k
    # ave_time = ave_time +k

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

    # if (i_batch > 1000):
    #     break
# print("max_time: ", max_time)
# print("_all_time: ", ave_time)


acc = 1.0 - (loss1 / finale)
acc11 = target_1_indices_1 / (target_1_indices_1 + target_1_indices_0)
print('accuracy=', acc)
print('accuracy11=', acc11)
print("finale_1_0_get_1_0: ", finale_1, finale_0, get_1, get_0)
print("target_indices11,10,00,01: ", target_1_indices_1, target_1_indices_0, target_0_indices_0, target_0_indices_1)

file2write = open("accuracy_file_f_e_predict_a_random.txt", 'a')
file2write.write("accuracy= " + str(acc) + "\n")
file2write.write("accuracy1111= " + str(acc11) + "\n")

file2write.write(
    "finale_1_0_get_1_0= " + "\n" + str(finale_1) + "\n" + str(finale_0) + "\n" + str(get_1) + "\n" + str(
        get_0) + "\n")
file2write.write(
    "target_indices11,10,00,01= " + "\n" + str(target_1_indices_1) + "\n" + str(target_1_indices_0) + "\n" + str(
        target_0_indices_0) + "\n" + str(target_0_indices_1) + "\n")
file2write.close()

time_b = datetime.datetime.now()
print("intrusion_detection_end: ", time_b)
print("all_time: ", time_b-time_a)



















