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


from model_abs_linear import *
from torch.optim import lr_scheduler
from torch.autograd import Variable


import numpy as np
from keras.models import load_model
import pandas as pd
import glob
from config import TestConfig
from train import *

from time import gmtime, strftime

config = TestConfig()

data_path = config.data_path
data_name = config.data_name
ch = config.num_channels
row = config.img_height
col = config.img_width
# val_part = config.val_part
# model_path = config.model_path

X_train1 = np.load(data_path + "/X_train_round2_" + config.data_name + "_80m160_01-03--13-46-00_72710to262220_f10.npy")
# X_train2 = np.load(data_path + "/X_train_round2_" + config.data_name + "_part2.npy")
# X_train3 = np.load(data_path + "/X_train_round2_" + config.data_name + "_part3.npy")
# X_train4 = np.load(data_path + "/X_train_round2_" + config.data_name + "_part4.npy")
# X_train5 = np.load(data_path + "/X_train_round2_" + config.data_name + "_part5.npy")

X_train = X_train1[:13266]

# X_train = np.concatenate((X_train1[:3080], X_train2[:11056], X_train3[:1381], X_train4[:2964], X_train5[:5181]), axis=0)
print( "Loading model...")
# model = load_model(config.model_path)
model = create_comma_model_large_dropout(row, col, ch, load_weights=True)

print( "Loading training data mean...")
X_train_mean = np.load(config.X_train_mean_path)
X_train = X_train
X_train = X_train.astype('float32')
X_train -= X_train_mean
X_train /= 255.0

print( "Predicting...")
preds_train = model.predict(X_train)
preds_train = preds_train[:, 0]
dummy_preds = np.repeat(config.angle_train_mean, config.num_channels)
preds_train = np.concatenate((dummy_preds, preds_train))

print("predict_angle_start: ", strftime("%Y-%m-%d %H:%M:%S", gmtime()))

X_test = X_train1[13266:]
# X_test = np.concatenate((X_train1[3080:], X_train2[11056:], X_train3[1381:], X_train4[2964:], X_train5[5181:]), axis=0)

print( "Loading model...")
model = create_comma_model_large_dropout(row, col, ch, load_weights=True)

print( "Loading training data mean...")
X_train_mean = np.load(config.X_train_mean_path)
X_test = X_test
# X_test = np.load(test_path)
X_test = X_test.astype('float32')
X_test -= X_train_mean
X_test /= 255.0

print( "Predicting...")
preds = model.predict(X_test)
preds = preds[:, 0]
dummy_preds = np.repeat(config.angle_train_mean, config.num_channels)
preds = np.concatenate((dummy_preds, preds))
preds_test = preds
print("preds_test: ", preds_test)
print("predict_angle_end: ", strftime("%Y-%m-%d %H:%M:%S", gmtime()))

# preds_train = preds[:3080]
# preds_test = preds[3080:]


face_dataset = intrusion_data(csv_file='01-03--13-46-00_72710to262220_f10_consecu_attack_0p2_0p9_13267.csv',root_dir='./data',cs=1,transform=transforms.Compose([transforms.Resize(256),transforms.RandomResizedCrop(224),transforms.ToTensor()]))
test_dataset=intrusion_data(csv_file='01-03--13-46-00_72710to262220_f10_consecu_attack_0p2_0p9_5687.csv',root_dir='./data',cs=1,transform=transforms.Compose([transforms.Resize(256),transforms.RandomResizedCrop(224),transforms.ToTensor()]))
# out=[]
dataloader = DataLoader(face_dataset, batch_size=1, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


# use_gpu = torch.cuda.is_available()
net = mynet()
# if use_gpu:
#     net.cuda()

criterion = F.nll_loss
# criterion.cuda()
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
        preds = preds_train
        # print("preds_______ibatch___: ", preds, i_batch)

        if i_batch / len(dataloader) > inter:
            print('completed %epoch ', inter)
            inter += .10
        # print("i_batch: ", i_batch)
        preds = preds[i_batch]
        # print("i_batch_preds: ", i_batch, preds)

        data = []
        result = []
        for sm in sample:
            imag, dat, res = sm
            data = dat
            result.append(res)

        target = torch.stack(result)
        target = target.view(-1)

        final_vars = []
        # for tens in data:
        #     final_vars.append(Variable(tens))
        final_vars = torch.FloatTensor([[[abs(data-preds)]]])
        # print("preds_data_ibatch: ", preds, data, i_batch)

        x = net(*final_vars)
        values, indices = x.max(1)
        loss = criterion(x, Variable(target))
        # train_loss += loss.data[0]
        loss.backward()
        optimizer.step()
        net.zero_grad()

        # if(i_batch>10):
        #     break
    # print('Train Epoch: {} \tLoss: {:.6f}'.format(
    #     iter_x, train_loss / len(dataloader)))
    # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&TESTING &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    print("test_start: ", strftime("%Y-%m-%d %H:%M:%S", gmtime()))

    for i_batch, sample in enumerate(test_loader):
        preds = preds_test
        preds = preds[i_batch]

        data = []
        result = []
        for sm in sample:
            imag, dat, res = sm
            data = dat
            result.append(res)

        target = torch.stack(result)
        target = target.view(-1)
        # final_vars = []
        # for tens in data:
        #     final_vars.append(Variable(tens))

        final_vars = torch.FloatTensor([[[abs(data-preds)]]])

        # final_vars = torch.FloatTensor([[[data, preds]]]).cuda()
        # print("preds_data_ibatch: ", preds, data, i_batch)

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

        # if(i_batch>10128):
        #     break

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

    # filenm = 'epoch_' + str(cnt) + '.pt'
    # # print(filenm)
    # torch.save(net, os.path.join(('saved_consecutive_models'), filenm))

    cnt = cnt + 1
    if (cnt % 10 == 0):
        print('EPOCH completed by %', (cnt / 40) * 100)
    print("test_end: ", strftime("%Y-%m-%d %H:%M:%S", gmtime()))




